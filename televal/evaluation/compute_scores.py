import os
import json
import argparse

def load_results(result_path):
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_scores(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_score(entry, type_weight):
    tp = entry.get('Type')
    weight = type_weight.get(tp, 0)
    n = entry.get('test_data_nums', 0)
    # 获取准确率或指标值
    if tp == 'QA':
        accuracy = entry.get('accuracy', 0)
    else:
        accuracy = entry.get('accuracy', 0)
    score = n * accuracy * weight
    return round(score, 2)


def main(dataset_name=None, model=None, weights='{"SCQ": 1, "MCQ": 2, "TFQ": 1, "QA": 4, "CSQ": 10}', project_dir='.', args=None):
    if args is not None:
        # 来自命令行 argparse
        dataset_name = args.dataset_name
        model = args.model
        weights = args.weights
        project_dir = args.project_dir
        
    base = os.path.join(project_dir, 'outputs', dataset_name, model)
    result_path = os.path.join(base, 'result.json')
    score_path = os.path.join(base, 'score.json')

    # 如果分数统计结果文件已存在，则跳过分数计算
    if os.path.exists(score_path):
        print(f"分数结果文件{score_path}已经存在！")
        return 
        
    print('\n================================ 开始按照权重计算分数 ====================================')
    
    type_weight = json.loads(weights)
    results = load_results(result_path)

    scored = []
    for entry in results:
        entry = entry.copy()
        entry['score'] = compute_score(entry, type_weight)
        scored.append(entry)

    save_scores(score_path, scored)
    print(f"每种题型的权重分数保存至{score_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算每种题型的权重分数')
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, default='{"SCQ": 1, "MCQ": 2, "TFQ": 1, "QA": 4, "CSQ": 10}', help='题型权重的JSON字符串')
    args = parser.parse_args()
    main(args=args)

# python -m televal.evaluation.compute_scores --dataset_name A --model qwen2.5-7b-instruct