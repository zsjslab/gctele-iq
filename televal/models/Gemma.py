from . import BaseModel
import torch
from transformers import pipeline
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

class GemmaModel(BaseModel):
    def __init__(self, model_path: str, gpu_ids: list[int], memory_size: list[int]):
        super().__init__(model_path, gpu_ids, memory_size)       
        # # 初始化分词器
        # self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        # # 将模型设置为评估模式
        # self.model.model.eval()
    
    def _load_model(self):
        if 'gemma-2-9b-it' in self.model_path.lower():
            model = pipeline(
                "text-generation",
                model=self.model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            return model
        else:
            model = Gemma3ForConditionalGeneration.from_pretrained(    
                self.model_path, device_map="auto"
            ).eval()
            return model            

    def generate(self, prompt, max_new_tokens) -> list[str]:
        # 使用对话模板生成
        
        if 'gemma-2-9b-it' in self.model_path.lower():
            messages = [
                {"role": "user", "content": prompt},
            ]
            outputs = self.model(messages, max_new_tokens=max_new_tokens)
            response = outputs[0]["generated_text"][-1]["content"].strip()
            return response 
        else:
            processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
            # Define the chat messages
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ],
            ]
            # Tokenize the input messages
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            # print("Input messages tokenized successfully.")
            # Generate the response
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded