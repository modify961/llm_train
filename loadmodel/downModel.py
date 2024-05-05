import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

# 从modelscope下载-Llama-3-8B-Instruct的模型
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='E:/models', revision='master')
