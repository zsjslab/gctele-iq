import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import pandas as pd
import IPython.display as disp
from gctele_iq.generation import generator_local, generator_api
from gctele_iq.evaluation import compute_metrics, compute_metrics_LLM, compute_scores
from gctele_iq.visualization import compute_ranking_single, compute_ranking_average

def run_all_steps(dataset_name, model, is_api, list_models, gpu_ids=None):
    # print("\n Step 1: 生成回答 ")
    # if is_api:
    #     generator_api.main(dataset_name=dataset_name, model=model)
    # else:
    #     if gpu_ids is None:
    #         raise ValueError("请为本地模型指定 gpu_ids 参数")
    #     generator_local.main(dataset_name=dataset_name, model=model, gpu_ids=gpu_ids)
        
    # print("\n Step 2: 客观题评估")
    # compute_metrics.main(dataset_name=dataset_name, model=model)

    # print("\n Step 3: 主观题评估")
    # compute_metrics_LLM.main(dataset_name=dataset_name, model=model)

    # print("\n Step 4: 综合得分")
    # compute_scores.main(dataset_name=dataset_name, model=model)

    print("\n Step 5: 可视化输出")
    df_type, df_cat = compute_ranking_single.main(dataset_name=dataset_name)
    num_cols_type = [c for c in df_type.columns if c not in ["Rank", "Model"]]
    num_cols_cat  = [c for c in df_cat.columns  if c not in ["Rank", "Model"]]
    
    styler1 = df_type.style.format({c: "{:.2f}" for c in num_cols_type}).hide(axis="index")
    styler2 = df_cat.style.format({c: "{:.2f}" for c in num_cols_cat}).hide(axis="index")
    
    disp.display(styler1)
    disp.display(styler2)
    
run_all_steps(dataset_name="A", model="qwen2.5-14b-instruct", is_api=False,  gpu_ids=[2, 3])

# run_all_steps(dataset_name="B", model="DeepSeek-V3","qwen2.5-14b-instruct", is_api=True)
