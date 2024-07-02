import torch
import torch.nn as nn
import numpy as np
from typing import Union,Optional
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.falcon.modeling_falcon import FalconLinear
import os
from loguru import logger

try:
    import owq_cuda
except:
    print('OWQ CUDA kernel extension is not installed.')

class QuantLinear(nn.Module):

    def __init__(self, bits, infeatures, outfeatures, outlierfeatures, bias, dtype, name):
        super().__init__()
        assert bits in [3,4], "Only 3,4 bits are supported."
        
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.outlierfeatures = outlierfeatures
        
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer('scales', torch.zeros((outfeatures, 1), dtype=dtype))
        self.register_buffer('zeros', torch.zeros((outfeatures // 2, 1), dtype=torch.uint8))
        self.register_buffer('bias', torch.zeros(outfeatures, dtype=dtype))
        
        self.register_buffer(
            'oweight', torch.zeros((outlierfeatures, outfeatures), dtype=dtype)
        )
        self.register_buffer(
            'outlieridx', torch.zeros((outlierfeatures), dtype=torch.int)
        )
        
        self.faster = True
        self.dtype = dtype
        self.name = name
        
    def pack(self, linear, scales, zeros, outlieridx:torch.Tensor, sym:bool=False):
        dtype = linear.weight.dtype
        
        if sym:
            zeros += 2**(self.bits - 1)
            
        if linear.bias is not None:
            self.bias = linear.bias.to(dtype)
            
        self.outlieridx = outlieridx

        if self.outlierfeatures > 0:
            self.oweight = torch.index_select(linear.weight.data, 1, self.outlieridx).t().contiguous()
        
        intweight = torch.round((linear.weight.data + zeros * scales) / scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        if self.outlierfeatures > 0:
            for idx in outlieridx:
                intweight[idx,:] = zeros.numpy().astype(np.uint32).squeeze()
        qweight = np.zeros(
            (self.infeatures // 32 * self.bits, self.outfeatures), dtype=np.uint32
        )
        
        self.scales = scales.to(dtype)
        zeros = zeros.to(torch.uint8)
        zeros_int = torch.zeros((zeros.shape[0] // 2, zeros.shape[1]), dtype=torch.uint8)
        for i in range(zeros_int.shape[0]):
            zeros_int[i] = (zeros[2*i] | zeros[2*i + 1] << 4)
        self.zeros = zeros_int
        
        i = 0
        row = 0
        if self.bits == 3:
            while row < qweight.shape[0]:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
        elif self.bits == 4:
            while row < qweight.shape[0]:
                for j in range(i, i + 8):
                    qweight[row] |= intweight[j] << (4 * (j - i))
                i += 8
                row += 1
        else:
            raise NotImplementedError

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)
    
    def set_kernel(self, faster):
        if self.outlierfeatures % 2 > 0:
            print("Number of outlier is not even. manually set to faster=False.")
            faster = False
        self.faster = faster
        
        if faster == False:
            self.oweight = self.oweight.float()
            self.scales = self.scales.float()
        
        # for outliermatvec kernel
        if self.outlierfeatures > 0:
            BLOCKWIDTH = owq_cuda.GetBLOCKWIDTH()
            NUMBLOCK = (self.infeatures + BLOCKWIDTH - 1) // BLOCKWIDTH
            self.register_buffer('cnt', torch.zeros([NUMBLOCK], dtype=torch.int))
            self.register_buffer('outrow', torch.zeros([NUMBLOCK], dtype=torch.int))
            self.cnt = torch.bincount(
                (self.outlieridx).to(torch.long) // BLOCKWIDTH,
                minlength=NUMBLOCK).to(torch.int)
            self.outrow = torch.zeros_like(self.cnt)

            for i in range(1, NUMBLOCK):
                self.outrow[i] = self.outrow[i-1] + self.cnt[i-1]
        
        # set operation kernel
        if self.bits == 3:
            if faster:
                self.matvec = owq_cuda.vecquant3matmul_faster
                self.outmatvec = owq_cuda.vecquant3outliermatmul_faster
                self.dequant = owq_cuda.matquant3dequant_faster
            else:
                self.matvec = owq_cuda.vecquant3matmul
                self.outmatvec = owq_cuda.vecquant3outliermatmul
                self.dequant = owq_cuda.matquant3dequant
        elif self.bits == 4:
            if faster:
                self.matvec = owq_cuda.vecquant4matmul_faster
                self.outmatvec = owq_cuda.vecquant4outliermatmul_faster
                self.dequant = owq_cuda.matquant4dequant_faster
            else:
                self.matvec = owq_cuda.vecquant4matmul
                self.outmatvec = owq_cuda.vecquant4outliermatmul
                self.dequant = owq_cuda.matquant4dequant
        else: # support only 3, 4 bits
            raise NotImplementedError
        
        if self.outlierfeatures > 0:
            self.matmul = QuantMatMul.apply  
            if self.faster:
                self.forward = self.forward_faster_outlier
            else:
                self.forward = self.forward_normal_outlier
        else:
            if self.faster:
                self.forward = self.forward_faster
            else:
                self.forward = self.forward_normal

    def forward_faster_outlier(self, x):
        if x.shape[-1] == x.numel():
            y = self.bias.clone()
            self.outmatvec(
                x, self.qweight, y,
                self.scales, self.zeros,
                self.oweight, self.outlieridx,
                self.outrow, self.cnt
                )
        else:
            matshape = (self.infeatures, self.outfeatures)
            y = self.matmul(x, self.oweight, self.dequant, 
                self.qweight, self.scales, 
                self.zeros, matshape,
                self.outlierfeatures, self.outlieridx,
                self.bias)
        return y
    
    def forward_normal_outlier(self, x):
        if x.shape[-1] == x.numel():
            dtype = x.dtype
            y = self.bias.float()
            x = x.float()
            self.outmatvec(
                x, self.qweight, y,
                self.scales, self.zeros,
                self.oweight, self.outlieridx,
                self.outrow, self.cnt
                )
            y = y.to(dtype)
        else:
            matshape = (self.infeatures, self.outfeatures)
            y = self.matmul(x, self.oweight, self.dequant, 
                self.qweight, self.scales, 
                self.zeros, matshape,
                self.outlierfeatures, self.outlieridx,
                self.bias)
        return y
    
    def forward_faster(self, x):
        if x.shape[-1] == x.numel():
            y = self.bias.clone()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(x.dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=x.dtype, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t(), self.bias)
        return y
    
    def forward_normal(self, x):
        if x.shape[-1] == x.numel():
            dtype = x.dtype
            y = self.bias.float()
            x = x.float()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=torch.float, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t().to(x.dtype), self.bias)
        return y

class QuantMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids, bias):
        
        # 1. Dequantize
        out = torch.empty(shape, dtype=oweight.dtype, device=oweight.device)        
        fn_dequant(qweight, out, scales, zeros)
        out[outids, :] = oweight
        out = out.t()
        
        # 2. Matmul
        output = torch.nn.functional.linear(x.to(bias.dtype), out.to(bias.dtype), bias)
        
        ctx.dequant_params = [oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids]
        ctx.tensors = torch.index_select(x, -1, outids)
        ctx.n_out = n_out
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_outlier = ctx.tensors
        oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids = ctx.dequant_params
        
        # Dequantize
        out = torch.empty(shape, dtype=oweight.dtype, device=oweight.device)
        fn_dequant(qweight, out, scales, zeros)
        out[outids, :] = oweight
        out = out.t()
            
        grad_input = None
        grad_oweight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, out.to(grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_oweight = torch.matmul(grad_output.transpose(-2,-1), x_outlier.to(grad_output.dtype))
        
        return grad_input, grad_oweight, None, None, None, None, None, None, None, None
    
def make_quant(module, n_out_infos, wbits, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in n_out_infos:
            setattr(
                module, attr, 
                QuantLinear(wbits, 
                            tmp.in_features, 
                            tmp.out_features, 
                            n_out_infos[name1].n_out, 
                            tmp.bias is not None, 
                            tmp.weight.dtype,
                            name1).to(tmp.weight.device)
            )
    for name1, child in module.named_children():
        make_quant(child, n_out_infos, wbits, name + '.' + name1 if name != '' else name1)

def find_layers(module, layers=[nn.Linear, FalconLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def process_owq(model_name_or_path,
               model: nn.Module,
               faster: Optional[bool] = True,
               ):
    try:
        matching_files = []
        for root, dirs, files in os.walk(model_name_or_path):
            for file in files:
                if file.endswith("-owq.pth") or file.endswith("-owq.pt"):
                    matching_files.append(os.path.join(root, file))
        checkpoint_path = matching_files[0]
        print("find checkpoint file: ", checkpoint_path)
    except:
        #报错说明没有找到
        raise FileNotFoundError(f"Model {model_name_or_path} not found.")
    ckpt = torch.load(checkpoint_path)
    wbits = ckpt['bits']
    
    if ckpt['packing']:
        logger.info(f"Loading packed model {checkpoint_path} ....")
        
        n_out_dict = ckpt['n_out_dict']
        make_quant(model, n_out_dict, wbits)
        
        # support old format
        for n, v in ckpt['model_state_dict'].items():
            if n.endswith('oweight') and v.shape[0] > v.shape[1]:
                ckpt['model_state_dict'][n] = v.t().contiguous()
                
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        qlayers = find_layers(model, [QuantLinear])
        for name in qlayers:
            qlayers[name].set_kernel(faster)
    else:
        logger.info(f"Loading fake quantized model {checkpoint_path} ....")
        model.load_state_dict(ckpt['model_state_dict'], strict=False) 
              
    
    logger.info(f"change model with owq layer sussessfully.")
    del ckpt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    
    print("Done.")
    return model,True
    