import re
from openai import OpenAI

# -----------------------------
# 辅助函数(针对推理模型)
# -----------------------------
def clean_response(text):
    """清理响应中可能存在的 '<think>...</think>' 部分"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).lstrip()


# -----------------------------
# 调用模型函数（待评估模型）
# -----------------------------
def model_call(prompt, model_name):
    
    client = OpenAI(
        api_key="XXX", 
        base_url="XXX"
    )
    chat_completion = client.chat.completions.create(
        messages=[
            { "role": "user","content": prompt,}
        ],
        model = model_name,
    )
    result = chat_completion.choices[0].message.content
    
    return clean_response(result)

# -----------------------------
# 调用模型函数（裁判模型）
# -----------------------------
def model_eval(prompt):
    
    client = OpenAI(
        api_key="XXX", 
        base_url="XXX"
    )
    chat_completion = client.chat.completions.create(
        messages=[
            { "role": "user","content": prompt,}
        ],
        model = "DeepSeek-V3",
    )
    result = chat_completion.choices[0].message.content    
    return result 


