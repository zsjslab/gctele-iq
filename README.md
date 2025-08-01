
# 📡gctele-iq电信运营商人机测评工具

`gctele-iq` 是一个用于评估大语言模型（LLMs）和人类专家在电信运营商领域任务中表现的测评工具包，支持本地模型与 API 模型的推理调用，内置客观题与主观题的评估方法，以及题型维度的可视化结果输出。


## 📦源码安装

```bash
pip install gctele-iq
```
##  📁项目目录结构要求

```text
your_project/
├── datas/                  # 存放待评估数据集（如 A.json / B.json）
├── outputs/                # 自动生成，存储模型推理与评估结果
├── configs/
│   ├── model_configs.py    # 配置本地模型名称和路径
│   └── api_models.py       # 配置 API 模型和用于主观评估的裁判大模型
├── gctele_iq/              # 安装后的主程序包（无需手动改动）
```
## 📄数据准备
将测评数据集（如 A.json）上传至 your_project/datas/ 目录中。我们已提供示例数据集，可用于功能验证。
## 🧠模型配置

### 1. 本地模型（可选）

在 `configs/model_configs.py` 中加入模型名称及其路径：
```python
model_configs = {
    'qwen2.5-7b-instruct': '/path/to/Qwen2.5-7B-Instruct',
    'qwen2.5-14b-instruct': '/path/to/Qwen2.5-14B-Instruct',
}
```

### 2. API 模型（可选）

在 `configs/api_models.py` 中配置两个函数：

```python
from openai import OpenAI

# ✅ 必填：裁判模型（用于主观题评估）
def model_eval(prompt):
    client = OpenAI(api_key="YOUR_KEY", base_url="https://api.example.com")
    chat_completion = client.chat.completions.create(
        messages=[{ "role": "user", "content": prompt }],
        model="DeepSeek-V3",
    )
    return chat_completion.choices[0].message.content

# ✅ 可选：评估模型（用于推理答题），当以 API 模式调用时需填写
def model_call(prompt, model_name):
    client = OpenAI(api_key="YOUR_KEY", base_url="https://api.example.com")
    chat_completion = client.chat.completions.create(
        messages=[{ "role": "user", "content": prompt }],
        model=model_name,
    )
    result = chat_completion.choices[0].message.content
    return result
```

##  🚀一键评估使用示例
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from gctele_iq.generation import generator_local, generator_api
from gctele_iq.evaluation import compute_metrics, compute_metrics_LLM, compute_scores
from gctele_iq.visualization import compute_ranking_average
import pandas as pd
import IPython.display as disp

def run_all_steps(dataset_name, model, gpu_ids=None, is_api=False):
    print("\n Step 1: 生成回答")
    if is_api:
        generator_api.main(dataset_name=dataset_name, model=model)
    else:
        generator_local.main(dataset_name=dataset_name, model=model, gpu_ids=gpu_ids)

    print("\n Step 2: 客观题评估")
    compute_metrics.main(dataset_name=dataset_name, model=model)

    print("\n Step 3: 主观题评估")
    compute_metrics_LLM.main(dataset_name=dataset_name, model=model)

    print("\n Step 4: 综合得分")
    compute_scores.main(dataset_name=dataset_name, model=model)

    print("\n Step 5: 可视化输出")
    df_type, df_cat = compute_ranking_average.main(dataset_name=[dataset_name])
    df_type, df_cat = compute_ranking_single.main(dataset_name=dataset_name)
    num_cols_type = [c for c in df_type.columns if c not in ["Rank", "Model"]]
    num_cols_cat  = [c for c in df_cat.columns  if c not in ["Rank", "Model"]]
    
    styler1 = df_type.style.format({c: "{:.2f}" for c in num_cols_type}).hide(axis="index")
    styler2 = df_cat.style.format({c: "{:.2f}" for c in num_cols_cat}).hide(axis="index")
    
    disp.display(styler1)
    disp.display(styler2)

# 示例1：使用本地模型评估
run_all_steps(dataset_name="A", model="qwen2.5-14b-instruct", gpu_ids=[2, 3], is_api=False)

# 示例2：使用 API 模型评估
run_all_steps(dataset_name="B", model="DeepSeek-V3", is_api=True)
```
## 📊输出结果说明

运行后，系统将自动生成以下输出文件至 `outputs/{dataset}/{model}/`：

- `record.json`：模型推理记录
- `evaluation.json`：记录了主观题的逐题评分详情
- `result.json`：整合主观题和客观题评估准确率
- `score.json`：题型加权得分
- `题型排名.csv` 和 `维度排名.csv`：在 `outputs/` 下生成的可视化评估结果
## 📬联系我们
如有需求或建议，欢迎联系：wang.yingying@ustcinfo.com