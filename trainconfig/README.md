# 微调参数说明

## "output_dir": "output/qwen1.5-7b-sft-lora",
    记录微调过程日志、checkout、以及微调后参数文件的路径
## "model_name_or_path": "modles/Qwen",
    需要微调的的模型
## "train_file_path":"data",
    微调语料文件的路径，
## "train_file_name": "aspcoder_train.csv",
    微调的文件名字，格式为  category,human,assistant。需要存放在train_file_path指定的路径下的csv的文件夹下。系统微调时会自动读取对应的csv文件，将其转换为jsonl文件。同时读取至内存
- category：类型，一般作为标记语料的类型。
- human：提问的语句
- assistant：机器回答的内容

## "template_name": "qwen",
    模型的名称，取值一般从模型文件的config.json文件里面的model_type的值。
## "train_mode": "lora",
    训练模式：目前支持 
- lora:一种微调方式
    帮助文档：https://zhuanlan.zhihu.com/p/623543497
- qlora: 概念文档:https://zhuanlan.zhihu.com/p/666234324
- full: 全训练
## "num_train_epochs": 1,
整个训练数据集上训练模型的完整遍历次数：在大模型训练中，num_train_epochs 是一个训练参数，指的是在整个训练数据集上训练模型的完整遍历次数。每次遍历被称为一个“epoch”。这个参数用于控制训练过程的长度，即模型看到训练数据的次数。

例如，如果 num_train_epochs 设置为 3，这意味着整个训练数据集将被用来训练模型三次。这个参数对模型的学习效果和训练时间都有重要影响。较少的训练周期可能导致模型欠拟合，而过多的训练周期则可能导致过拟合，特别是当训练数据不足以支持重复学习时。
## "per_device_train_batch_size": 1,
每个设备上的训练批次大小

## "gradient_accumulation_steps": 16,
梯度累积算法：https://zhuanlan.zhihu.com/p/650710443


"learning_rate": 2e-4,
"max_seq_length": 1024,
"logging_steps": 100,
"save_steps": 100,
"save_total_limit": 1,
"lr_scheduler_type": "constant_with_warmup",
"warmup_steps": 100,
"lora_rank": 64,
"lora_alpha": 16,
"lora_dropout": 0.05,
"gradient_checkpointing": true,
"disable_tqdm": false,
"optim": "paged_adamw_32bit",
"seed": 42,
"fp16": true,
"report_to": "tensorboard",
"dataloader_num_workers": 0,
"save_strategy": "steps",
"weight_decay": 0,
"max_grad_norm": 0.3,
"remove_unused_columns": false