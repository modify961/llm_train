"""
argparse 是 Python 标准库中的一个模块，用于解析命令行参数。它允许开发者编写具有用户友好界面的命令行工具，使用户可以轻松地使用该工具，并通过命令行选项和参数来配置工具的行为。使用 argparse，开发者可以定义程序所需的参数，并为这些参数指定描述、类型、默认值等属性。然后，argparse 会解析命令行输入，并根据定义的参数规则来提取和验证输入，最后生成一个易于使用的命令行接口。
HfArgumentParser：是对 Python 标准库中的 argparse.ArgumentParser 进行了扩展，用于解析和管理 Hugging Face 的模型训练和评估过程中的参数。具体来说，它接受一个元组作为参数，元组中包含了自定义参数类（例如 CustomizedArguments）和训练参数类（例如 TrainingArguments）。然后，它会解析命令行输入，并根据这两个参数类中定义的参数规则来提取和验证输入，最终生成一个方便用户使用的命令行接口，用于配置模型训练和评估的参数。
"""
import argparse
import os
import json
from os.path import join
from loguru import logger
from component.argument import CustomizedArguments
from component.template import template_map
from component.dataset import UnifiedTurnTrainDataSet
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments
)
"""
读取训练的配置参数
"""
def init_config(config_name):
    if config_name is None:
        raise Exception("训练配置不得为空")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_file", type=str,default=config_name,help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_config_file=args.train_config_file
    # 读取训练的参数配置
    parser=HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_config_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("训练参数:{}".format(training_args))
    # 加载预训练配置文件
    with open(config_name,"r") as f:
        train_args=json.load(f)
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)
    # 检查配置
    assert args.task_type in ['pretrain', 'sft', 'dpo'], "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora'], "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"

    return args, training_args


def loadfile():
    """
    raise 关键字用于引发异常。当你使用 raise 语句时，你可以指定要引发的异常类型和可选的错误消息。
    这允许你在程序执行过程中遇到错误或不符合预期的情况时，主动触发异常，从而中断当前的程序执行流程，并将控制权传递给异常处理程序。
    """
    if template_map is None or len(template_map) == 0:
        raise Exception("模板列表为空=")
    templat_name="qwen"
    if templat_name not in template_map.keys():
        raise Exception(f"模板名:{templat_name},不在模板列表中，全部的模板名称为{template_map.keys()}")
    template=template_map[templat_name]
    print(template)
    # UnifiedTurnTrainDataSet()

# 主函数
def main():
    # 文件路径，注意：windows和linux下路径格式都为trainconfig/qwen1.5-7b-sft-lora.json
    config_name="trainconfig/qwen1.5-7b-sft-lora.json"
    # 1、加载配置和环境检测
    args, training_args = init_config(config_name)
    # 2、加载训练数据
    # loadfile()

# 主函数入口
# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    main()