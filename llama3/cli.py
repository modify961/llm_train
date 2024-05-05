from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import platform 
import datetime

# 模型路径
modelPath = 'llama3-8B'
stop_stream = False
os_name = platform.system()
clear_command = 'cls' if os_name=='Windows' else 'clear'
# from_pretrained函数根据指定的模型名称或路径，从Hugging Face Model Hub（模型存储库）中加载相应的预训练tokenizer。
# 加载的tokenizer可以直接用于对文本进行预处理，以便输入到预训练模型中进行推理或微调。
# 共有一个固定参数，已经无固定参数
# 固定参数：pretrained_model_name_or_path  预训练模型的名称（huggface）或者本地存储的路径
# **kwargs：变参：常用的有
#           cache_dir：指定模型文件的缓存目录。
#           revision：指定要加载的模型的特定版本（通常用于从模型存储库中指定一个特定的Git提交版本）。
#           proxies：指定用于下载模型文件的代理服务器。
#           use_fast：指定是否使用快速模式加载tokenizer，即是否使用快速但不一定准确的实现。        
tokenizer=AutoTokenizer.from_pretrained(modelPath, trust_remote_code=False)
# 加载预训练的语言模型
# device_map="map"
# torch_dtype=torch.bfloat16 这是指定模型参数的数据类型。在这里，模型的参数将使用16位浮点数（bfloat16）来存储
model=AutoModelForCausalLM.from_pretrained(modelPath,device_map="auto",torch_dtype=torch.bfloat16)
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息


def build_input(prompt,history=[]):
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    history.append({'role':'user','content':prompt})
    prompt_str=""
    # 拼接历史对话
    for item in history:
        if item['role']=='user':
            prompt_str=prompt_str+user_format.format(content=item['content'])
        else:
            prompt_str=prompt_str+assistant_format.format(content=item['content'])
    return prompt_str

def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

def obtain_answer(input_str):
    # 将文本作为输入，并将其转换为模型可以理解的数字序列。
    input_ids= tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').cuda()
    # print("输出文本："+input_ids)
    generated_ids = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True,
    top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    output = generated_ids.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(output)
    response = response.strip().replace('<|eot_id|>',"").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()

    # 获取当前时间
    now =datetime.datetime.now()
    time= now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    # 执行GPU内存清理
    torch_gc()
    return response
def main():
    history = []
    # 引用全局变量stop_stream，方法内修改一个全局变量的值时必须使用global声明
    global stop_stream
    print("欢迎使用llama3")
    while True:
        # input() 函数用于从用户处接收输入。当调用 input() 函数时，程序会暂停执行，等待用户输入一些内容，并按下回车键。然后，input() 函数会将用户输入的内容作为一个字符串返回给程序
        query = input("\n 用户：")
        # strip() 函数去除了字符串 text 两端的空格
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用llama3")
            continue
        count = 0
        in_put=build_input(query,history)
        print("in_put:"+in_put)
        out_put=obtain_answer(in_put)
        history.append({'role':'assistant','content':out_put})
        #   print("history"+len(history))
        print("ollama:"+out_put+"\n\n")

# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    main()