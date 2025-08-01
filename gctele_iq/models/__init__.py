from typing import Dict, Type

class BaseModel:
    """所有模型的基类，定义统一接口"""
    
    def __init__(self, model_path: str, gpu_ids: list[int], memory_size: list[int]):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.memory_size = memory_size
        # 其他通用初始化逻辑（如设备配置）

    def generate(self, inputs: list[str], max_new_tokens: int) -> list[str]:
        """生成文本的统一接口"""
        raise NotImplementedError("子类必须实现 generate 方法")

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value


# 导入所有具体模型
from .Qwen import QwenModel
from .GLM import GLMModel
from .Telechat import TelechatModel
from .Yi import YiModel
from .Gemma import GemmaModel
from .Minicpm import MinicpmModel
from .Llama import LlamaModel
from .Jiutian import JiutianModel
from .Baichuan import BaichuanModel
from .Qwen3 import Qwen3Model
from .pangu import panguModel

# 模型名称到类的映射表
model_mapping: Dict[str, Type[BaseModel]] = {
    "qwen2.5-7b-instruct": QwenModel,
    "qwen2.5-14b-instruct": QwenModel,
    "glm-4-9b-chat": GLMModel,
    "glm-4-32b-chat": GLMModel,
    "telechat2-35b-nov": TelechatModel,
    "telechat2.5-35b": TelechatModel,
    "yi-1.5-9b-chat": YiModel,
    "gemma-2-9b-it": GemmaModel,
    "gemma-3-12b-it": GemmaModel,
    "gemma-3-27b-it": GemmaModel,
    "minicpm3-4b": MinicpmModel,
    "llama3-8b-instruct": LlamaModel,
    "jiutian-139moe-chat": JiutianModel,
    "baichuan2-13b-chat": BaichuanModel,
    "qwen3-8b": Qwen3Model,
    "qwen3-4b": Qwen3Model,
    "qwen3-14b": Qwen3Model,
    "qwen3-32b": Qwen3Model,
    "pangu-72b": panguModel,
}
