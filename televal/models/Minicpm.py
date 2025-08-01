from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

class MinicpmModel(BaseModel):
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
            torch_dtype=torch.bfloat16, 
            device_map="auto", trust_remote_code=True
            )
        print("minicpm3-4b模型加载成功！")
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        messages = [
            {"role": "user", "content": prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
        attention_mask = torch.ones_like(model_inputs, dtype=torch.long).to("cuda")
        model_outputs = self.model.generate(
            model_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_p=0.7,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_token_ids = [
            model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
        ]
        response = self.tokenizer.batch_decode(output_token_ids, 
                skip_special_tokens=True)[0]
        return response