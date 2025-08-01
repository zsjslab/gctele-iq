import os
import torch
torch.cuda.empty_cache()
import argparse
import time
import json
from tqdm import tqdm
import ast
import importlib.util
import warnings

# 导入依赖模块
from gctele_iq.models import model_mapping
from gctele_iq.utils.data_helper import DataHelper
from configs.model_configs import model_configs


warnings.filterwarnings('ignore')


def load_existing_records(record_path):
    """
    读取已存在的记录文件，返回记录列表及起始索引。
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
    将单条记录追加保存到记录文件中。
    """
    with open(record_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def prepare_output_directory(project_dir, dataset_name,  model):
    """
    根据传入参数创建输出文件夹，并返回记录文件和结果文件的路径。
    """
    output_path = os.path.join(project_dir, 'outputs', dataset_name,  model)
    os.makedirs(output_path, exist_ok=True)
    record_path = os.path.join(output_path, 'record.json')
    result_path = os.path.join(output_path, 'result.json')
    return output_path, record_path, result_path


def main(dataset_name=None, model=None, project_dir='.', gpu_ids=None, memory_size='[40,40,40,40]', max_new_tokens=1024, args=None):
    if args is not None:
        # 解析必要参数
        gpu_ids = args.gpu_ids
        memory_size = args.memory_size
        max_new_tokens = args.max_new_tokens
        model = args.model
        project_dir = args.project_dir
        dataset_name = args.dataset_name

    # 设置可见 GPU 环境变量
    if gpu_ids is not None:
        visible_devices = ','.join(str(i) for i in gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        print(f"[INFO] 已设置 CUDA_VISIBLE_DEVICES={visible_devices}")  
    
    model_path = model_configs[model]
    # 构造输出目录及记录文件路径
    output_path, record_path, result_path = prepare_output_directory(
        project_dir, dataset_name,  model
    )

    # 如果结果文件已存在，则跳过推理和评测
    if os.path.exists(result_path):
        print(f"结果文件{result_path}已经存在！")
        return

    print(f'\n=========================== { model} 模型加载 ======================================')
    # 加载本地模型
    ModelClass = model_mapping[ model]
    llm = ModelClass(model_path, gpu_ids, memory_size)
    # tokenizer = llm.tokenizer  # 如有需要可使用
    print(f'\n{ model} 加载成功！')


    # 加载测试数据和 prompt 模板
    data_helper = DataHelper(project_dir, dataset_name)
    test_inputs, test_answers = data_helper.get_inputs()
    raw_samples = data_helper.get_metadata()  # 每条包含 category, Type, question, answer, options

    print('\n=========================== 评测数据和模型信息 ===========================================')
    print('dataset_name:', dataset_name)
    print('llm:',  model)
    print('llm_path:', model_path)
    print('input_example:\n', test_inputs[0])

    # 断点续传：检查已存在的记录
    _, start_idx = load_existing_records(record_path)

    print('\n================================ 开始推理 ===============================================')
    start_time = time.time()

    # 遍历测试数据，调用本地模型生成响应并保存记录
    for i in tqdm(range(start_idx, len(test_inputs))):
        test_input = test_inputs[i]
        truth_answer = test_answers[i]
        response = llm.generate(test_input, max_new_tokens)
        # 从 raw_samples 取出对应的 metadata
        sample_meta = raw_samples[i]

        record = {
            'category': sample_meta.get('category'),
            'Type': sample_meta.get('Type'),
            'prompt': test_input,
            'response': response,
            'truth_answer': truth_answer
        }
        save_record(record_path, record)

    elapsed = (time.time() - start_time) / 60
    print(f'\n推理完成！耗时 {elapsed:.3f} 分钟')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 工程目录
    parser.add_argument('--project_dir', type=str, default='.', help="项目根目录路径")
    parser.add_argument('--dataset_name', type=str, default='A', help="选择测评数据集")
    # 模型名称，对应 model_configs.py 中的配置
    parser.add_argument('--model', type=str, default='qwen1.5-14b-chat', help="选择本地大模型")
    # 模型 generate 接口的 max_new_tokens 参数
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    # 多卡推理设置
    parser.add_argument('--gpu_ids', type=ast.literal_eval, default=[0,1,2,3])
    parser.add_argument('--memory_size', type=ast.literal_eval, default=[40,40,40,40])
    args = parser.parse_args()

    main(args=args)


#  CUDA_VISIBLE_DEVICES=1,2,3 python -m gctele_iq.generation.generator_local --dataset_name A --model qwen2.5-7b-instruct
