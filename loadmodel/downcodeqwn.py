import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

# 从modelscope下载-CodeQwen1.5-7B-Chat的模型
model_dir = snapshot_download('Qwen/CodeQwen1.5-7B-Chat', cache_dir='E:/models', revision='master')
