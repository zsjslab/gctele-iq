import os
import json
import argparse
import pandas as pd

# 设置浮点数显示格式：控制台打印保留两位小数
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
    df[TYPE_LIST + ['Total']] = df[TYPE_LIST + ['Total']].round(2)  # 保留两位小数
    df['Rank'] = df['Total'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values(['Rank', 'Model'])
    cols = ['Rank', 'Model'] + TYPE_LIST + ['Total']
    return df[cols]


def compute_category_table(data):
    records = []
    for model, entries in data.items():
        df = pd.DataFrame(entries)
        row = {'Model': model}
        # 1. 先计算四个普通大类
        normal_cats = [c for c in CATEGORY_LIST if c != 'Operational Processes & Support Systems']
        for cat in normal_cats:
            row[cat] = df.loc[df['category'] == cat, 'score'].sum()
        # 2. 分别计算 OPS 的三个子类，并存储它们
        for sub in OPS_SUBCATS:
            type_means = []
            for t in TYPE_LIST:
                m = df.loc[(df['Type'] == t) & (df['category'] == sub), 'score'].mean()
                if pd.notna(m):
                    type_means.append(m)
            row[sub] = sum(type_means)
        # 3. 计算原始的“Operational Processes & Support Systems”聚合值
        ops_agg = 0.0
        # 按题型先在三子类里取均值，再累加五种题型
        for t in TYPE_LIST:
            subs = []
            for sub in OPS_SUBCATS:
                m = df.loc[(df['Type'] == t) & (df['category'] == sub), 'score'].mean()
                if pd.notna(m):
                    subs.append(m)
            ops_agg += (sum(subs) / len(OPS_SUBCATS)) if subs else 0.0
        # 4. 计算 Total：四个普通大类 + 聚合的 OPS 值
        row['Total'] = sum(row[cat] for cat in normal_cats) + ops_agg
        records.append(row)
    # 5. 构造 DataFrame 并四舍五入保留两位小数
    df_out = pd.DataFrame(records)
    display_cols = normal_cats + OPS_SUBCATS + ['Total']
    df_out[display_cols] = df_out[display_cols].round(2)
    # 6. 排名并排序
    df_out['Rank'] = df_out['Total'].rank(ascending=False, method='min').astype(int)
    df_out = df_out.sort_values(['Rank', 'Model'])
    # 7. 最终返回，以 Model 为索引
    cols = ['Rank', 'Model'] + display_cols
    return df_out[cols]


def main(project_dir='.', dataset_name='A', list_models=None):
    if list_models:
        MODELS = list_models
    else:
        MODELS = list_models_in_dir(project_dir, dataset_name)

    # 加载所有模型分数（添加异常处理）
    data = {}
    for m in MODELS:
        try:
            data[m] = load_scores(project_dir, dataset_name, m)
        except FileNotFoundError as e:
            print(f"[警告] 跳过模型 {m}，原因：{e}")

    if not data:
        raise RuntimeError("未能加载任何模型的 score.json，请检查路径和文件是否存在")

    df_type = compute_type_table(data)
    df_cat = compute_category_table(data)

    # 保存 CSV 文件
    out_dir = os.path.join(project_dir, 'outputs', dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    df_type.to_csv(os.path.join(out_dir, '题型排名.csv'), index=False)
    df_cat.to_csv(os.path.join(out_dir, '维度排名.csv'), index=False)

    return df_type, df_cat




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化模型排名')
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--list_models', type=str, help='逗号分隔的模型名称列表，如 "a,b,c"')
    args = parser.parse_args()

    main(
        project_dir=args.project_dir,
        dataset_name=args.dataset_name,
        list_models=args.list_models
    )




# python -m televal.visualization.compute_ranking_single --dataset_name A
