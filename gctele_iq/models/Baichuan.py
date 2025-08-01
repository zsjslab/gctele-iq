from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import torch
import warnings
warnings.filterwarnings('ignore')

class BaichuanModel(BaseModel):
    def __init__(self, model_path: str, gpu_ids: list[int], memory_size: list[int]):
        super().__init__(model_path, gpu_ids, memory_size)       
        # 初始化分词器
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        # 将模型设置为评估模式
        self.model.eval()
        

    def _load_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(
            self.model_path, 
            legacy=True
            )
        # 添加填充token的显式设置
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
        return tokenizer       
    
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("baichuan模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)      

        return response
        
    

