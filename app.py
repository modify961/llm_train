from component.template import template_map
from component.dataset import UnifiedTurnTrainDataSet
"""
读取训练的配置参数
"""
def init_config(config_name):
    if config_name is None:
        raise Exception("训练配置不得为空")
    
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
    loadfile()

# 主函数入口
# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    main()