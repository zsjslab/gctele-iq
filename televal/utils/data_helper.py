import os
import json
import importlib

class DataHelper:
    """
    DataHelper 用于读取单一数据集的 JSON 样本，并根据题型动态加载对应的 prompt 模板。
    样本必须包含字段：category, Type, question, answer, options (可选)
    模板文件路径：prompt_configs/{type}.py，内部需定义 zero_shot_template 字符串。
    """
    def __init__(self, project_dir, dataset_name):
        # 数据文件路径
        data_path = os.path.join(project_dir, 'datas',  f'{dataset_name}.json')
        self.examples = self._read_json(data_path)

    @staticmethod
    def _read_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        return examples

    def get_inputs(self):
        prompts = []
        labels = []
        for item in self.examples:
            q_type = item.get('Type', '').upper()
            question = item.get('question', '').strip()
            options_list = item.get('options', [])

            # 构造 options 字符串
            options_str = '\n'.join(options_list) if options_list else ''

            # 动态加载模板模块（保持原始大小写）
            module_name = f"televal.generation.prompt_configs.{q_type}"
            tpl = importlib.import_module(module_name)
            template = tpl.zero_shot_template

            # 填充模板占位符
            prompt = template.replace('{question}', question)
            prompt = prompt.replace('{options}', options_str)

            prompts.append(prompt)
            labels.append(item.get('answer', '').strip())
        return prompts, labels

    def get_metadata(self):
        return self.examples.copy()