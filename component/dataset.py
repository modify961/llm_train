"""
loguru 是一个用于 Python 的日志记录库,通过pip install loguru 安装
"""
from loguru import logger
from torch.utils.data import Dataset

"""
统一处理预训练数据
UnifiedSFTDataset(Dataset): 继承自
"""
class UnifiedTurnTrainDataSet(Dataset):
    """
    __init__ 构造函数:用于在创建一个对象时进行初始化操作。
    """
    def __init__(cli,file,tokenizer,max_seq_length,template):
        cli.tokenizer = tokenizer
        cli.template_name = template.template.name
        cli.system_format = template.system_format
        cli.user_format = template.user_format
        cli.assistant_format = template.assistant_format
        cli.system = template.system

        cli.max_seq_length=max_seq_length
        logger.info(f'使用模板 "{template.template.name}" 进行训练')
        logger.info("加载数据：{}",file)
        with open(file,'r',encoding='utf-8') as f:
            data_list = f.readlines()
        logger.info("从数据集共加载 {} 条数据".format(len(data_list)))
        cli.data_list=data_list
        
    def __len__(cli):
        return len(cli.data_list)