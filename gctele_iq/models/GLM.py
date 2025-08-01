from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GLMModel(BaseModel):
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
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True,
                                tokenize=True,
                                return_tensors="pt",
                                return_dict=True,
                                ).to(self.model.device)
        gen_kwargs = {"max_length": max_new_tokens, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace("\n", "")        

        return response