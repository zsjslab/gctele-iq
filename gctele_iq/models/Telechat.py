from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TelechatModel(BaseModel):
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
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.float16)
        print("telechat2-35b模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        messages = [{"role": "user", "content": prompt}]
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 准备模型输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # print("开始生成")
        # 生成回复
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        # print("结束生成")
        # 去除输入部分，保留新生成的输出部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # 解码生成的文本
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]                   

        return response