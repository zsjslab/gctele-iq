from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

class LlamaModel(BaseModel):
    def __init__(self, model_path: str, gpu_ids: list[int], memory_size: list[int]):
        super().__init__(model_path, gpu_ids, memory_size)       
        # 初始化分词器
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        # 将模型设置为评估模式
        self.model.model.eval()
        

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
        model = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("llama3-8b-instruct模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": prompt},
        ]
        outputs = self.model(
            messages,
            max_new_tokens=max_new_tokens,
        )
        # 获取生成的文本内容（这里是最后一个角色的内容，即助手的回答）
        response = outputs[0]['generated_text'][-1]['content']
        return response