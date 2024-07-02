import importlib
from typing import TYPE_CHECKING, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_auto_owq_available, is_optimum_available, is_torch_available, logging
from ..utils.quantization_config import owqConfig, QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class OwqHfQuantizer(HfQuantizer):
    """
    Quantizer of the owq method - for owq the quantizer support calibration of the model through
    `auto_owq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    requires_calibration = True
    required_packages = ["optimum", "owq"]
    optimum_quantizer = None

    # owq的量化参数初始设置
    def __init__(self, quantization_config, **kwargs):
    super().__init__(quantization_config, **kwargs)
    self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_owq_available():
            raise ImportError("Using `owq` quantization requires owq_cuda...")
        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError("Converting into 8-bit weights from tf/flax weights is currently not supported...")
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once("You have loaded an owq model on CPU and have a CUDA device available...")
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError("You are attempting to load an owq model with a device_map that contains a CPU or disk device...")


    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with owq.")
        return torch_dtype
    
    # 量化权重并进行预处理，然后将量化后的权重和权重尺度注册到模块中。需要进行改写
    def create_quantized_param(self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str, target_device: "torch.device", state_dict: Dict[str, Any], unexpected_keys: Optional[List[str]] = None):
        from eetq import quantize_and_preprocess_weights
        module, tensor_name = get_module_from_name(model, param_name)
        new_value, weight_scale = quantize_and_preprocess_weights(param_value)
        module._buffers[tensor_name] = new_value.to(target_device)
        module.register("weight_scales", weight_scale.to(target_device))

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            model = self.optimum_quantizer.convert_model(model)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    @property
    def is_serializable(self):
        return True