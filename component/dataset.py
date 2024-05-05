"""
loguru 是一个用于 Python 的日志记录库,通过pip install loguru 安装
"""
from loguru import logger
from torch.utils.data import Dataset
from component import json_csv_help
import json


"""
统一处理预训练数据
UnifiedSFTDataset(Dataset): 继承自
"""
class UnifiedTurnTrainDataSet(Dataset):
    """
    __init__ 构造函数:用于在创建一个对象时进行初始化操作。
    """
    def __init__(cli,train_file_path,train_file_name,tokenizer,max_seq_length,template):
        cli.tokenizer = tokenizer
        cli.template_name = template.template_name
        cli.system_format = template.system_format
        cli.user_format = template.user_format
        cli.assistant_format = template.assistant_format
        cli.system = template.system

        cli.max_seq_length=max_seq_length
        logger.info(f'使用模板 "{template.template_name}" 进行训练')
        logger.info("加载数据：{}",train_file_name)
        # 将csv文件转换问json
        file=json_csv_help.from_csv_to_jsonl(train_file_path,train_file_name)

        with open(file,'r',encoding='utf-8') as f:
            data_list = f.readlines()
        logger.info("从数据集共加载 {} 条数据".format(len(data_list)))
        cli.data_list=data_list
        
    def __len__(cli):
        return len(cli.data_list)
    
    def __getitem__(cli, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = cli.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # 设置系统信息
        if cli.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else cli.system
            # system信息不为空
            if system is not None:
                system_text = cli.system_format.format(content=system)
                input_ids = cli.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = cli.user_format.format(content=human, stop_token=cli.tokenizer.eos_token)
            assistant = cli.assistant_format.format(content=assistant, stop_token=cli.tokenizer.eos_token)

            input_tokens = cli.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = cli.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:cli.max_seq_length]
        target_mask = target_mask[:cli.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
    
    