import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
import platform 

"""
从魔搭社区下载模型
下载全需要先执行 pip install modelscope 
网址：https://www.modelscope.cn/home
"""

os_name = platform.system()
print("当前系统是："+os_name)
if os_name=='Windows':
    model_dir = snapshot_download('Qwen/CodeQwen1.5-7B-Chat', cache_dir='E:/models', revision='master')
else:
    model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
