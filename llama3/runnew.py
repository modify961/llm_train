from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model

mode_path = 'Instruct'
lora_path = 'llama3_lora'

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

def build_input(prompt,history=[]):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages
    
def obtain_answer(input_str):
    text = tokenizer.apply_chat_template(input_str, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    history = []
    # 引用全局变量stop_stream，方法内修改一个全局变量的值时必须使用global声明
    global stop_stream
    print("欢迎使用AspCoder")
    while True:
        # input() 函数用于从用户处接收输入。当调用 input() 函数时，程序会暂停执行，等待用户输入一些内容，并按下回车键。然后，input() 函数会将用户输入的内容作为一个字符串返回给程序
        query = input("\n 用户：")
        # strip() 函数去除了字符串 text 两端的空格
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用AspCoder")
            continue
        count = 0
        in_put=build_input(query,[])
        out_put=obtain_answer(in_put)
        print("ollama:"+out_put+"\n\n")

# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    main()