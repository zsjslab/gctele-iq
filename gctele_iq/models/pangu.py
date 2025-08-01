from . import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

import torch

class panguModel(BaseModel):
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
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True
        )
        return tokenizer       
    
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        return model

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用Qwen3的对话模板生成
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.6
        )        
        
        messages = [
            {"role": "system", "content": "你是一位运营商领域的专家。"}, # define your system prompt here
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # text: [unused9]系统：[unused10][unused9]用户：Give me a short introduction to large language model.[unused10][unused9]助手：
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # model_inputs.input_ids: tensor([[1, 45887, 70914, 89246, 45892, 45887, 62205, 89246, 38805, 42624, 45509, 24759, 739, 41839, 21500, 6138, 20257, 49, 45892, 45887, 74458, 89246]], device='npu:0'),

        # conduct text completion
        outputs = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, eos_token_id=45892, return_dict_in_generate=True, generation_config=generation_config)

        input_length = model_inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        output_sent = self.tokenizer.decode(generated_tokens[0])

        # parsing thinking content
        thinking_content = output_sent.split("[unused17]")[0].split("[unused16]")[-1].strip()
        content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()
        
        return content