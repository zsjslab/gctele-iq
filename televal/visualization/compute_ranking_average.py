import os
import json
import argparse
import pandas as pd
from tabulate import tabulate

pd.set_option('display.float_format', lambda x: f'{x:.2f}')

CATEGORY_LIST = [
    'Communication Principles & Network Fundamentals',
    'Market Landscape & Strategic Orientation',
    'Operational Processes & Support Systems',
    'Product Business & Service System',
    'Strategic Emerging Technology Trends'
]
TYPE_LIST = ['SCQ', 'MCQ', 'TFQ', 'QA', 'CSQ']
OPS_SUBCATS = [
    'Operational Processes & Support Systems - AC',
    'Operational Processes & Support Systems - CS',
    'Operational Processes & Support Systems - OS'
]

def list_models_in_dir(project_dir, dataset_name):
    base = os.path.join(project_dir, 'outputs', dataset_name)
    if not os.path.exists(base):
        raise FileNotFoundError(f"路径不存在: {base}")
    
    return [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith('.')
    ]

def load_scores(project_dir, dataset_name, model):
    path = os.path.join(project_dir, 'outputs', dataset_name, model, 'score.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"[未找到] 模型 {model} 的结果文件不存在：{path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_type_table(data):
    records = []
    for model, entries in data.items():
        df = pd.DataFrame(entries)
        row = {'Model': model}
        for t in TYPE_LIST:
            outside = df[(df['Type'] == t) & (~df['category'].isin(OPS_SUBCATS))]['score'].sum()
            subs = []
            for sub in OPS_SUBCATS:
                val = df[(df['Type'] == t) & (df['category'] == sub)]['score'].mean()
                if pd.notna(val):
                    subs.append(val)
            inside = sum(subs) / len(OPS_SUBCATS) if subs else 0.0
            row[t] = outside + inside
        row['Total'] = sum(row[t] for t in TYPE_LIST)
        records.append(row)

    df = pd.DataFrame(records)
    df[TYPE_LIST + ['Total']] = df[TYPE_LIST + ['Total']].round(2)
    df['Rank'] = df['Total'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values(['Rank', 'Model'])
    cols = ['Rank', 'Model'] + TYPE_LIST + ['Total']
    return df[cols]

def compute_category_table(data):
    records = []
    for model, entries in data.items():
        df = pd.DataFrame(entries)
        row = {'Model': model}
        normal_cats = [c for c in CATEGORY_LIST if c != 'Operational Processes & Support Systems']
        for cat in normal_cats:
            row[cat] = df.loc[df['category'] == cat, 'score'].sum()
        for sub in OPS_SUBCATS:
            type_means = []
            for t in TYPE_LIST:
                m = df.loc[(df['Type'] == t) & (df['category'] == sub), 'score'].mean()
                if pd.notna(m):
                    type_means.append(m)
            row[sub] = sum(type_means)
        ops_agg = 0.0
        for t in TYPE_LIST:
            subs = []
            for sub in OPS_SUBCATS:
                m = df.loc[(df['Type'] == t) & (df['category'] == sub), 'score'].mean()
                if pd.notna(m):
                    subs.append(m)
            ops_agg += (sum(subs) / len(OPS_SUBCATS)) if subs else 0.0
        row['Total'] = sum(row[cat] for cat in normal_cats) + ops_agg
        records.append(row)

    df_out = pd.DataFrame(records)
    display_cols = normal_cats + OPS_SUBCATS + ['Total']
    df_out[display_cols] = df_out[display_cols].round(2)
    df_out['Rank'] = df_out['Total'].rank(ascending=False, method='min').astype(int)
    df_out = df_out.sort_values(['Rank', 'Model'])
    cols = ['Rank', 'Model'] + display_cols
    return df_out[cols]

def main(project_dir='.', dataset_name=None, list_models=None):
    if not dataset_name:
        raise ValueError("必须指定至少一个数据集名称")

    if list_models:
        MODELS = [m.strip() for m in list_models.split(',')]
    else:
        MODELS = list_models_in_dir(project_dir, dataset_name[0])  # 默认取第一个数据集的目录结构

    combined_data = {model: [] for model in MODELS}

    for dataset_name in dataset_name:
        for model in MODELS:
            try:
                entries = load_scores(project_dir, dataset_name, model)
                for entry in entries:
                    entry['dataset'] = dataset_name
                combined_data[model].extend(entries)
            except FileNotFoundError as e:
                print(f"[警告] 跳过模型 {model} 在数据集 {dataset_name}，原因：{e}")

    averaged_data = {}
    for model, entries in combined_data.items():
        if not entries:
            continue
        df = pd.DataFrame(entries)
        grouped = df.groupby(['Type', 'category'])['score'].mean().reset_index()
        averaged_data[model] = grouped.to_dict(orient='records')

    if not averaged_data:
        raise RuntimeError("没有任何模型数据被成功加载，请检查路径或文件")

    df_type = compute_type_table(averaged_data)
    df_cat = compute_category_table(averaged_data)

    if len(dataset_name) == 1:
        out_dir = os.path.join(project_dir, 'outputs', dataset_name[0])
    else:
        out_dir = os.path.join(project_dir, 'outputs', '_'.join(dataset_name) + '_average')

    
    os.makedirs(out_dir, exist_ok=True)
    df_type.to_csv(os.path.join(out_dir, '题型排名.csv'), index=False)
    df_cat.to_csv(os.path.join(out_dir, '维度排名.csv'), index=False)
    print(f"\n保存结果至 {out_dir}")
    return df_type, df_cat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化多个数据集平均排名')
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', nargs='+', required=True, help='多个数据集名称（用空格分隔）')
    parser.add_argument('--list_models', type=str, help='逗号分隔的模型名称列表，如 "a,b,c"')
    args = parser.parse_args()

    main(
        project_dir=args.project_dir,
        dataset_name=args.dataset_name,
        list_models=args.list_models
    )




# python -m televal.visualization.compute_ranking_average --dataset_name A B C

