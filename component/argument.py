from dataclasses import dataclass, field
from typing import Optional

"""
自定义参数
"""
@dataclass
class CustomizedArguments:
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file_path: str = field(metadata={"help": "训练集存储的文件路径。如果task_type=pretrain，请将扫描其下面的所有csv文件"})
    train_file_name: str = field(metadata={"help": "训练集存储的文件名，csv格式。如果task_type=pretrain，该参数无需指定"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    max_prompt_length: int = field(default=512, metadata={"help": "dpo时，prompt的最大长度"})
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "预训练时tokenize的线程数量"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[pretrain, sft]"})
    train_mode: str = field(default="qlora", metadata={"help": "训练方式：[full, qlora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    use_unsloth: Optional[bool] = field(default=False, metadata={"help": "use sloth or not"})
