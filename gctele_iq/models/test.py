from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model_local_path = "/workspace/LLMFiles/LLMs/pangu"

generation_config = GenerationConfig(
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.6
)

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_local_path, 
    use_fast=False, 
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_local_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "你是一位运营商领域的专家。"}, # define your system prompt here
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# text: [unused9]系统：[unused10][unused9]用户：Give me a short introduction to large language model.[unused10][unused9]助手：
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# model_inputs.input_ids: tensor([[1, 45887, 70914, 89246, 45892, 45887, 62205, 89246, 38805, 42624, 45509, 24759, 739, 41839, 21500, 6138, 20257, 49, 45892, 45887, 74458, 89246]], device='npu:0'),

# conduct text completion
outputs = model.generate(**model_inputs, max_new_tokens=5120, eos_token_id=45892, return_dict_in_generate=True, generation_config=generation_config)

input_length = model_inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
output_sent = tokenizer.decode(generated_tokens[0])

# parsing thinking content
thinking_content = output_sent.split("[unused17]")[0].split("[unused16]")[-1].strip()
content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()

print("\nthinking content:", thinking_content)
print("\ncontent:", content)