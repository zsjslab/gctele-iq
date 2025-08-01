from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class YiModel(BaseModel):
    def __init__(self, model_path: str, gpu_ids: list[int], memory_size: list[int]):
        super().__init__(model_path, gpu_ids, memory_size)       
        # 初始化分词器
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        # 将模型设置为评估模式
        self.model.eval()
        

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
            print(f"Setting `pad_token_id` to `eos_token_id`: {tokenizer.eos_token_id}")
        return tokenizer       
    
    def _load_model(self):      
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,  
            device_map="auto", 
            torch_dtype='auto'
        )
        print("yi-1.5-9b-chat模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to('cuda'), eos_token_id=self.tokenizer.eos_token_id,max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)                 

        return response