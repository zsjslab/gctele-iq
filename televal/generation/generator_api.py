import os
import sys
import time
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from televal.utils.data_helper import DataHelper
from configs.api_models import model_call



def load_existing_records(record_path):
    """
    读取已保存的记录文件，返回记录列表及起始索引
    """
    records = []
    start_idx = 0
    if os.path.exists(record_path):
        try:
            with open(record_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            start_idx = len(records)
        except json.JSONDecodeError as e:
            print(f"警告：读取记录文件时发生错误（第{e.lineno}行），将从头开始！")
            records = []
            start_idx = 0
    return records, start_idx


def save_record(record_path, record):
    """
    将单条记录追加保存到文件中
    """
    with open(record_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def prepare_output_directory(project_dir, dataset_name, model_name):
    """
    创建输出文件夹，并返回记录文件和结果文件的路径
    """
    output_path = os.path.join(project_dir, 'outputs', dataset_name, model_name)
    os.makedirs(output_path, exist_ok=True)
    record_path = os.path.join(output_path, 'record.json')
    result_path = os.path.join(output_path, 'result.json')
    return output_path, record_path, result_path



def main(dataset_name=None, model=None, project_dir='.', args=None):
    if args is not None:
        # 来自命令行 argparse
        dataset_name = args.dataset_name
        model = args.model
        project_dir = args.project_dir
    
    # 准备输出目录和记录文件路径
    output_path, record_path, result_path = prepare_output_directory(
        project_dir, dataset_name, model
    )
    
    # 如果结果文件已存在，则跳过推理和评测
    if os.path.exists(result_path):
        print(f"结果文件{result_path}已经存在！")
        return    
    
    # 读取已存在的记录，确定起始索引
    records, start_idx = load_existing_records(record_path)
    
    # 载入测试数据
    data_helper = DataHelper(project_dir, dataset_name)
    test_inputs, test_answers = data_helper.get_inputs()
    raw_samples = data_helper.get_metadata()  # 每条包含 category, Type, question, answer, options
    
    print('\n================================ 评测数据和模型信息 ================================')
    print('dataset_name:', dataset_name)
    print('llm:', model)
    print('input_example:\n', test_inputs[0])
    
    print('\n================================ 开始推理 ==========================================')
    start_time = time.time()
    
    for i in tqdm(range(start_idx, len(test_inputs))):
        prompt = test_inputs[i]
        truth_answer = test_answers[i]
        # 调用模型接口
        response = model_call(prompt, model)

        # 从 raw_samples 取出对应的 metadata
        sample_meta = raw_samples[i]
        
        # 保存当前记录
        record = {
            'category': sample_meta.get('category'),
            'Type': sample_meta.get('Type'),
            'prompt': prompt,
            'response': response,
            'truth_answer': truth_answer
        }
        save_record(record_path, record)
    
    elapsed = (time.time() - start_time) / 60
    print(f'\n推理完成！耗时 {elapsed:.3f} 分钟')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, default='A', help="数据集名称")
    parser.add_argument('--model', type=str, default='DeepSeek-V3', help="模型名称")
    args = parser.parse_args()
    main(args=args)
        
# python -m televal.generation.generator_api --dataset_name A --model DeepSeek-V3

