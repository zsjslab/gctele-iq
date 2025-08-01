from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class JiutianModel(BaseModel):
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
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
            print(f"Setting `pad_token_id` to `eos_token_id`: {tokenizer.eos_token_id}")
        return tokenizer       
    
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True)
        print("jiutian-139moe-chat模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        text = "Human:\n" + prompt + "\n\nAssistant:\n"
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False,padding=True, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.03,do_sample=False,eos_token_id=0,pad_token_id=self.tokenizer.eos_token_id, use_cache=False)
        # response = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        # 解码生成的输出
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 "Assistant:" 后的内容
        response = decoded_output.split("Assistant:")[1].strip()       

        return response