"""
argparse 是 Python 标准库中的一个模块，用于解析命令行参数。它允许开发者编写具有用户友好界面的命令行工具，使用户可以轻松地使用该工具，并通过命令行选项和参数来配置工具的行为。使用 argparse，开发者可以定义程序所需的参数，并为这些参数指定描述、类型、默认值等属性。然后，argparse 会解析命令行输入，并根据定义的参数规则来提取和验证输入，最后生成一个易于使用的命令行接口。
HfArgumentParser：是对 Python 标准库中的 argparse.ArgumentParser 进行了扩展，用于解析和管理 Hugging Face 的模型训练和评估过程中的参数。具体来说，它接受一个元组作为参数，元组中包含了自定义参数类（例如 CustomizedArguments）和训练参数类（例如 TrainingArguments）。然后，它会解析命令行输入，并根据这两个参数类中定义的参数规则来提取和验证输入，最终生成一个方便用户使用的命令行接口，用于配置模型训练和评估的参数。
"""
import argparse
import os
import json
import torch
from os.path import join
from loguru import logger
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from component.argument import CustomizedArguments
from component.template import template_map
from component.dataset import UnifiedTurnTrainDataSet
from trl import DPOTrainer, get_kbit_device_map
import torch.nn as nn
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer
)

"""
找出所有全连接层，为所有全连接添加adapter
"""
def find_all_linear_names(model, train_mode):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

def load_unsloth_model(args, training_args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        trust_remote_code=True,
        load_in_4bit=True if args.train_mode == 'qlora' else False,
    )
    if args.train_mode in ['lora', 'qlora']:
        logger.info('Initializing PEFT Model...')
        target_modules = find_all_linear_names(model, args.train_mode)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=training_args.seed,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f'target_modules: {target_modules}')
    return {
        'model': model,
        'ref_model': None,
        'peft_config': None
    }
"""
加载模型
"""
def load_model(args, training_args):
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'加载基础模型: {args.model_name_or_path}')
    logger.info(f'训练方式 {args.train_mode}')

    
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # 三元表达式 如果training_args.fp16为真（True），则bnb_4bit_compute_dtype被赋值为torch.float16，否则被赋值为torch.bfloat16。
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # moe模型，需要考虑负载均衡的loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    # QLoRA: 将所有非int8模块转换为全精度（fp32）以保证稳定性。
    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft']:
        # 将量化模型转换为可lora训练
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # LoRA: 启用输入嵌入的梯度。
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft']:
        # 向后兼容
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 初始化训练参数
    if args.train_mode == 'full':
        peft_config = None
    else:
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # init peft model
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # init ref_model
    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.train_mode == 'full' else None
    # pretrain和sft，不需要ref_model
    else:
        ref_model = None

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }

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
"""
加载tokenizer
"""
def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    return tokenizer


def load_dataset(args, tokenizer,training_args):
    """
    raise 关键字用于引发异常。当你使用 raise 语句时，你可以指定要引发的异常类型和可选的错误消息。
    这允许你在程序执行过程中遇到错误或不符合预期的情况时，主动触发异常，从而中断当前的程序执行流程，并将控制权传递给异常处理程序。
    """
    if template_map is None or len(template_map) == 0:
        raise Exception("模板列表为空=")
    templat_name=args.template_name
    if templat_name not in template_map.keys():
        raise Exception(f"模板名:{templat_name},不在模板列表中，全部的模板名称为{template_map.keys()}")
    template=template_map[templat_name]
    logger.info("对话模板:{}".format(template))
    train_dataset = UnifiedTurnTrainDataSet(args.train_file_path,args.train_file_name,tokenizer,args.max_seq_length, template)
    return train_dataset


# 主函数
def main():
    # 文件路径，注意：windows和linux下路径格式都为trainconfig/qwen1.5-7b-sft-lora.json
    config_name="trainconfig/qwen1.5-7b-sft-lora.json"
    # 1、加载配置和环境检测
    args, training_args = init_config(config_name)
    # 2、加载load_tokenizer
    tokenizer=load_tokenizer(args)
    # 3、加载训练数据，和collator，默认设置为None
    train_dataset = load_dataset(args, tokenizer,training_args)
    data_collator = None
    # 4、加载模型,是否使用unsloth框架
    if args.use_unsloth:
        components = load_unsloth_model(args, training_args)
    else:
        components = load_model(args, training_args)
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']
    # 5、 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # 6、开始训练
    train_result=trainer.train()
    # 保存checkout 及tokenizer
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)
    # 7、 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# 主函数入口
# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    main()