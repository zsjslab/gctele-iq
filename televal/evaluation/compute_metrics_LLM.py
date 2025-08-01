import os
import json
import argparse
import re
from tqdm import tqdm
from configs.api_models import model_eval

# 从 eval_prompt 包导入评分模板
from televal.evaluation.eval_prompt.qa import EVAL_PROMPT as QA_PROMPT
from televal.evaluation.eval_prompt.ac import EVAL_PROMPT as AC_PROMPT
from televal.evaluation.eval_prompt.cs import EVAL_PROMPT as CS_PROMPT
from televal.evaluation.eval_prompt.os import EVAL_PROMPT as OS_PROMPT

# CSQ 各子类别对应的 prompt
EVAL_PROMPTS_CSQ = {
    'Operational Processes & Support Systems - AC': AC_PROMPT,
    'Operational Processes & Support Systems - CS': CS_PROMPT,
    'Operational Processes & Support Systems - OS': OS_PROMPT,
}

def extract_score(text, qtype):
    """
    优先从文本末尾的 Python 字典字符串中提取 '综合得分'。
    如果失败，则使用正则表达式从文本中查找 "综合得分: <数字>" 模式。
    - CSQ: 1–10 整数
    - QA : 1.0–4.0 浮点数
    """
    score_value_str = None
    # 1. 尝试从末尾的 Python dict 字符串中提取 '综合得分'
    dict_match = re.search(r"\{[^{}]*?'综合得分'\s*:\s*([0-9]+(?:\.[0-9]+)?)[^{}]*?\}", text)
    if dict_match:
        try:
            potential_score_str = dict_match.group(1)
            float(potential_score_str)
            score_value_str = potential_score_str
        except (ValueError, IndexError):
            score_value_str = None # 捕获失败或转换失败，则清空

    # 2. 如果第一步未成功获取分数，则使用 fallback 正则
    if score_value_str is None:
        # 这个正则匹配 "综合得分" (可选括号内容) : (空格) 数字
        fallback_match = re.search(r'综合得分(?:（[^）]+）)?[：:]\s*([0-9]+(?:\.[0-9]+)?)', text)
        if fallback_match:
            score_value_str = fallback_match.group(1)

    # 如果两种方法都未能提取到分数字符串，则返回 None
    if score_value_str is None:
        return None

    # 3. 转换、验证分数并返回
    try:
        val = float(score_value_str)
    except ValueError:
        return None 

    if qtype == 'CSQ':
        if val.is_integer() and 1 <= int(val) <= 10:
            return int(val)
    elif qtype == 'QA': 
        if 1.0 <= val <= 4.0:
            return round(val, 2)

    return None # 如果 qtype 不匹配或范围不匹配


# def load_records(path: str):
#     with open(path, 'r', encoding='utf-8') as f:
#         return [json.loads(line) for line in f if line.strip()]

def load_records(record_path: str):
    with open(record_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            return json.loads(content)  # 读取整个 JSON 列表
        except json.JSONDecodeError:
            return [json.loads(line) for line in content.splitlines() if line.strip()]


def main(dataset_name=None, model=None, project_dir='.', args=None):
    if args is not None:
        # 来自命令行 argparse
        dataset_name = args.dataset_name
        model = args.model
        project_dir = args.project_dir
        
    base        = os.path.join(project_dir, 'outputs', dataset_name, model)
    record_path = os.path.join(base, 'record.json')
    eval_path   = os.path.join(base, 'evaluation.json')
    result_path = os.path.join(base, 'result.json')

    # 如果主观题结果文件已存在，则跳过主观题评估
    if os.path.exists(eval_path):
        print(f"主观题结果文件{eval_path}已经存在！")
        return 

    print('\n================================ 开始评估主观题 ====================================')
    
    # 1. 读入所有推理记录
    records = load_records(record_path)

    # 2. 准备 detailed 与 existing_summary
    detailed = json.load(open(eval_path, 'r', encoding='utf-8')) if os.path.exists(eval_path) else []
    all_summary = json.load(open(result_path, 'r', encoding='utf-8')) if os.path.exists(result_path) else []
    others = [r for r in all_summary if r.get('Type') not in ('QA','CSQ')]
    summary_map = { (r['Type'], r['category']): r for r in all_summary }

    summary = []
    
    # 3. QA 题型：按 category 分组评估
    qa_groups = {}
    for rec in records:
        if rec.get('Type') == 'QA':
            qa_groups.setdefault(rec['category'], []).append(rec)

    for cat, recs in qa_groups.items():
        qa_scores = []
        for rec in tqdm(recs, desc=f'Scoring QA [{cat}]'):
            for _ in range(1):
                # 自动补 prompt（如未提供）
                prompt_completion = rec.get('prompt') or (
                    f"你是电信运营商领域的专家。请针对以下问题，提供简洁、准确、直接的回答。\n\n"
                    f"问题：{rec.get('question', '')}\n\n"
                    f"答案："
                )
                full = QA_PROMPT.format(
                    prompt=prompt_completion,
                    output=rec['response'],
                    truth_answer=rec['truth_answer']
                )
                reply = model_eval(full)
                sc = extract_score(reply, rec.get("Type"))
                if sc is not None:
                    qa_scores.append(sc)
                # detailed 记录追加
                detailed.append({
                    **rec,
                    'raw_reply': reply,
                    'score': sc,
                    'llm': model,
                    'dataset_name': dataset_name
                })
        avg = sum(qa_scores) / len(qa_scores) if qa_scores else 0.0
        summary.append({
            'category': cat,
            'Type': 'QA',
            'test_data_nums': len(recs),
            'llm': model,
            'dataset_name': dataset_name,
            'accuracy': round(avg / 4, 4)
        })

    # 4. CSQ 题型：按子类评估
    csq_groups = {}
    for rec in records:
        if rec.get('Type') == 'CSQ' and rec['category'] in EVAL_PROMPTS_CSQ:
            csq_groups.setdefault(rec['category'], []).append(rec)

    for cat, recs in csq_groups.items():
        tpl = EVAL_PROMPTS_CSQ[cat]
        csq_scores = []
        for rec in tqdm(recs, desc=f'Scoring CSQ [{cat}]'):
            for _ in range(1):
                # 自动补 prompt（如未提供）
                prompt_completion = rec.get('prompt') or (
                    f"你是电信运营商领域的专家。接下来我将给出一道案例分析题，请对问题进行准确的回答。\n\n"
                    f"问题：{rec.get('question', '')}\n\n"
                    f"答案："
                )
                full = tpl.format(
                    prompt=prompt_completion,
                    output=rec['response'],
                    truth_answer=rec['truth_answer']
                )
                reply = model_eval(full)
                sc = extract_score(reply, rec.get("Type"))
                if sc is not None:
                    csq_scores.append(sc)
                # detailed 记录追加
                detailed.append({
                    **rec,
                    'raw_reply': reply,
                    'score': sc,
                    'llm': model,
                    'dataset_name': dataset_name
                })
        avg = sum(csq_scores) / len(csq_scores) if csq_scores else 0.0
        summary.append({
            'category': cat,
            'Type': 'CSQ',
            'test_data_nums': len(recs),
            'llm': model,
            'dataset_name': dataset_name,
            'accuracy': round(avg / 10, 4)
        })

    # 5. 写入详细评估
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    # 6. 写入汇总：保留原有 + 新增 QA/CSQ
    # with open(result_path, 'w', encoding='utf-8') as f:
    #     json.dump(existing_summary + summary, f, ensure_ascii=False, indent=2)

    # print(f"裁判模型评分细则保存到{eval_path}, 评分结果保存到{result_path}")
    merged_qacsq = []
    for item in summary:
        key = (item['Type'], item['category'])
        if key in summary_map:
            merged = { **summary_map[key], **item }
        else:
            merged = item
        merged_qacsq.append(merged)
    final_summary = others + merged_qacsq
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(f"裁判模型评分细则保存到{eval_path}, 评分结果保存到{result_path}")    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model',        type=str, required=True)
    args = parser.parse_args()
    main(args= args)


# python -m televal.evaluation.compute_metrics_LLM --dataset_name A --model qwen2.5-7b-instruct

