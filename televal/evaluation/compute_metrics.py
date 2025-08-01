import os
import json
import argparse
import re
from collections import defaultdict


def extract_ticket_numbers(text: str) -> str:
    items = text.split('、')
    ticket_numbers = []
    for item in items:
        match = re.search(r'[A-Z]\d*', item)
        if match:
            ticket_numbers.append(match.group())
    ticket_numbers = sorted(set(ticket_numbers))
    return '、'.join(ticket_numbers)


def compute_mcq_accuracy(hypothesis: str, reference: str) -> float:
    hyp = extract_ticket_numbers(hypothesis)
    hyp_set = set(re.findall(r'[A-Z]\d*', hyp))
    ref_set = set(re.findall(r'[A-Z]\d*', reference))
    return 1.0 if hyp_set == ref_set else 0.0


def load_records(record_path: str):
    with open(record_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            return json.loads(content)  # 读取整个 JSON 列表
        except json.JSONDecodeError:
            return [json.loads(line) for line in content.splitlines() if line.strip()]



def group_records(records):
    grouped = defaultdict(list)
    for rec in records:
        key = (rec.get('category'), rec.get('Type'))
        grouped[key].append(rec)
    return grouped


def evaluate_group(category, qtype, recs, llm_name, dataset_name):
    n = len(recs)
    preds = [r['response'] for r in recs]
    golds = [r['truth_answer'] for r in recs]

    result = {
        'category': category,
        'Type': qtype,
        'test_data_nums': n,
        'llm': llm_name,
        'dataset_name': dataset_name
    }
    if qtype in ('SCQ', 'MCQ', 'TFQ'):
        accuracies = [compute_mcq_accuracy(p, g) for p, g in zip(preds, golds)]
        result['accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return result


def main(dataset_name=None, model=None, project_dir='.', args=None):
    if args is not None:
        # 来自命令行 argparse
        dataset_name = args.dataset_name
        model = args.model
        project_dir = args.project_dir    

    output_dir = os.path.join(project_dir, 'outputs', dataset_name, model)
    record_path = os.path.join(output_dir, 'record.json')
    result_path = os.path.join(output_dir, 'result.json')

    # 如果结果文件已存在，则跳过客观题评估
    if os.path.exists(result_path):
        print(f"客观题结果文件{result_path}已经存在！")
        return
    
    records = load_records(record_path)
    grouped = group_records(records)

    results = []
    allowed_qtypes = {'SCQ', 'MCQ', 'TFQ', 'QA'}

    print('\n================================ 开始评估客观题 ====================================')
    for (category, qtype), recs in grouped.items():
        if qtype not in allowed_qtypes:
            continue
        entry = evaluate_group(category, qtype, recs, model, dataset_name)
        results.append(entry)

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存评估结果到{result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估LLM的输出')
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--model', type=str, required=True, help='大模型名称')
    args = parser.parse_args()
    main(args=args)


# python -m televal.evaluation.compute_metrics --dataset_name A --model qwen2.5-7b-instruct

