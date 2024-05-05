# api.py  通过api使用模型
    函数讲解：
    AutoModelForCausalLM.from_pretrained 和 AutoModel.from_pretrained 都是 Hugging Face Transformers 库中用于加载预训练模型的函数，但它们针对的模型类型有所不同，因此存在一些区别：
    AutoModelForCausalLM.from_pretrained：
    这个函数用于加载预训练的语言模型，特别是那些能够生成文本序列的模型，例如 GPT 系列模型。
    它会加载一个针对生成式任务（如文本生成）进行了微调的模型，通常包含了一个自回归（autoregressive）的头部结构，使模型能够在生成文本时逐步预测下一个词。
    适用于生成文本、对话等任务。
    AutoModel.from_pretrained：

    这个函数用于加载预训练的通用语言模型，它不包含特定的头部结构，可以应用于各种下游任务，如分类、命名实体识别等。
    它加载的模型通常是预训练的语言模型的基础部分，不包含任何特定任务的头部结构。
    适用于各种 NLP 任务，如分类、标注、问答等。
    总的来说，AutoModelForCausalLM.from_pretrained 适用于生成文本的任务，而 AutoModel.from_pretrained 则更通用，适用于各种 NLP 任务，它们加载的模型具有不同的头部结构和适用场景。

    global 关键字用于在函数内部声明一个变量是全局变量，而不是局部变量。这意味着在函数内部使用该变量时，Python会从最外层的作用域中寻找该变量的值。
    当你在函数内部需要修改一个全局变量的值时，你需要使用 global 关键字声明该变量。如果不使用 global 关键字，Python会将你的变量视为一个新的局部变量，而不是修改全局变量。

    import os 是一个导入语句，用来导入Python的标准库模块 os。os 模块提供了许多函数和变量，用于与操作系统交互，包括文件和目录的管理（创建、删除、修改）、获取环境变量、执行操作系统命令等。

    这里是一些使用 os 模块常见的功能：

    操作文件和目录：

    os.mkdir(path)：创建一个名为 path 的目录。
    os.makedirs(path)：创建多级目录。
    os.remove(path)：删除一个文件。
    os.rmdir(path)：删除一个空目录。
    os.listdir(path)：列出指定目录下的文件和子目录。
    路径操作：

    os.path.join(path1[, path2[, ...]])：将多个路径组件合并成一个路径。
    os.path.split(path)：将路径分割成目录和文件名。
    os.path.exists(path)：检查给定路径是否存在。
    执行操作系统命令：

    os.system(command)：运行操作系统命令。
    环境变量和进程参数：

    os.environ：一个包含环境变量的字典。
    os.getenv(key, default=None)：获取环境变量的值。

    import platform 是用于导入名为 platform 的标准库模块。这个模块提供了许多用于获取和处理平台信息的函数，包括操作系统类型、计算机的硬件架构、Python解释器的版本信息等。

    一旦导入了 platform 模块，你可以使用它提供的各种函数来获取有关当前系统和Python环境的信息。以下是一些 platform 模块常用的函数：

    获取操作系统信息：

    platform.system()：返回操作系统的名称（如 'Windows'、'Linux'、'Darwin' 等）。
    platform.release()：返回操作系统的发行版本号。
    获取计算机的硬件架构：

    platform.machine()：返回计算机的硬件架构（如 'x86'、'x86_64'、'arm64' 等）。
    获取Python解释器的版本信息：

    platform.python_version()：返回当前Python解释器的版本号。
    platform.python_compiler()：返回Python解释器的编译器信息。
    其他系统信息：

    platform.platform()：返回包含系统名称、发行版本、硬件类型等信息的字符串。
    platform.uname()：返回一个包含系统信息的元组。

    在 Python 中，string.format() 方法是用于格式化字符串的一种常见方式。它允许您将变量的值插入到字符串中的特定位置。这是一个基本的用法示例：
name = "Alice"
age = 30
formatted_string = "My name is {}, and I am {} years old.".format(name, age)
print(formatted_string)

这将输出：
My name is Alice, and I am 30 years old.

在这个例子中，{} 是占位符，format() 方法中传入的参数按顺序替换了这些占位符。
您还可以使用大括号中的数字来指定要替换的参数的顺序，如下所示：
name = "Bob"
age = 25
formatted_string = "My name is {1}, and I am {0} years old.".format(age, name)
print(formatted_string)

这将输出：
My name is Bob, and I am 25 years old.

另外，您还可以使用关键字参数来指定要替换的值的名称，如下所示：
name = "Carol"
age = 35
formatted_string = "My name is {name}, and I am {age} years old.".format(name=name, age=age)
print(formatted_string)

这将输出：
My name is Carol, and I am 35 years old.

string.format() 方法还支持更复杂的格式化，包括指定字段的宽度、对齐方式、精度等。例如：
pi = 3.141592653589793
formatted_pi = "The value of pi is approximately {:.2f}.".format(pi)
print(formatted_pi)

这将输出：
The value of pi is approximately 3.14.

在这个例子中，:.2f 指定了使用两位小数的浮点数格式。


tokenizer.encode 是自然语言处理中用于将文本转换为模型可接受的输入格式的方法之一，通常用于将文本转换为数字化表示（tokens）。这在使用深度学习模型进行文本处理时非常常见。

这个方法通常由各种文本处理工具包（如Hugging Face的transformers库中的Tokenizer类）提供。它将文本作为输入，并将其转换为模型可以理解的数字序列。

以下是一些常见的参数：

text (str)：要编码的文本。
add_special_tokens (bool, optional)：是否在序列的开头和结尾添加特殊标记，如CLS和SEP。默认为True。
padding (str or bool, optional)：指定是否对序列进行填充，并指定填充的方式。可以是'longest'、'max_length'、'do_not_pad'等。默认为'longest'。
truncation (bool, optional)：指定是否对序列进行截断。默认为False。
max_length (int, optional)：指定编码后的最大长度。超出此长度的部分将被截断或填充，取决于参数设置。
return_tensors (str, optional)：指定返回的张量类型，如'tensor'或'pt'（PyTorch）。
return_attention_mask (bool, optional)：是否返回注意力掩码，用于标识输入序列的哪些部分是真实文本，哪些是填充的部分。
return_token_type_ids (bool, optional)：是否返回token类型的ID，通常在处理有两个句子的任务时使用。
这些参数可能会根据不同的工具包或库而有所不同，但这些是最常见的参数之一。


在大型语言模型（LLM）中，"token"通常指的是文本数据被分解的最小单位，用于模型处理和训练。在不同的模型和处理系统中，这些单位可以是词、子词（subword）、或者更小的字符片段。例如，在一些使用BERT模型的场景中，文本通常被分解为子词单元。这种分解有助于模型更好地处理语言中的词形变化和未知词汇。
如何计算一句话的token数
计算一句话的token数通常涉及以下步骤：

1.分词：根据所使用的模型的分词方法（如BPE、WordPiece等），将句子分解成tokens。
2.计数：统计得到的tokens的数量。

使用Python实现token计数
以下是一个使用Python和transformers库（此库提供了许多预训练模型和分词器）来计算一句话的token数的示例。首先，你需要安装transformers库：
pip install transformers

然后，你可以使用以下Python代码来计算token数：
from transformers import BertTokenizer

# 初始化一个分词器，这里以BERT模型的分词器为例
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义一句话
sentence = "Hello, how are you today?"

# 使用分词器对句子进行分词
tokens = tokenizer.tokenize(sentence)

# 输出tokens和tokens的数量
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

这段代码首先加载了一个基于BERT的英文小写模型的分词器，然后将一个示例句子分解成tokens，并打印出这些tokens及其总数。在BERT模型中，一些常见的英文单词可能被直接保留为一个token，而复杂或不常见的词则可能被分解为更小的单位。
通过这种方式，你可以对任何文本使用相应的分词器来计算其token数量。不同的模型和分词器可能会有不同的分词结果，因此token的计算也会有所不同。



generated_ids = model.generate(
input_ids=input_ids, max_new_tokens=512, do_sample=True,
top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|eot_id|>')[0]
) 这段代码是做什么的

这段代码是使用一个预训练的生成式模型生成文本。让我解释一下每个参数的作用：

1.model.generate()：这是一个生成文本的方法，通常由深度学习模型（如GPT、GPT-2、GPT-3等）提供。它接受一些输入，然后生成相应的文本。
2.input_ids：这是模型的输入，通常是已经经过编码的文本序列。模型会基于这些输入生成接下来的文本。
3.max_new_tokens：指定要生成的最大新token数。在这个例子中，设置为512，意味着最多生成512个新的token。
4.do_sample：这是一个布尔值，指示是否使用采样来生成文本。如果设置为True，则模型将根据概率分布随机选择下一个token；如果设置为False，则模型将根据最高概率选择下一个token。
5.top_p：用于采样的一个参数，指定在采样时保留的概率累积总和。在这个例子中，设置为0.9，意味着模型将选择概率累积达到0.9的token。
6.temperature：用于控制采样的参数，调整模型生成文本的多样性。较低的温度会导致更加保守和可预测的文本，而较高的温度会导致更加多样化和随机的文本。
7.repetition_penalty：用于惩罚重复token出现的参数。它有助于确保生成的文本更加多样化，避免重复性太高。
8.eos_token_id：表示结束标记的token的ID。在这个例子中，使用了一个特殊的方法来获取token的ID，通常是由tokenizer提供的。

综合起来，这段代码使用一个预训练的生成式模型，基于给定的输入，生成新的文本。生成的文本的长度最多为512个token，采用了采样的方式，同时控制了文本的多样性、重复性等因素。


datetime.datetime.now().strftime 是一个用于格式化日期时间的 Python 方法。它允许你将日期时间对象转换为指定格式的字符串。以下是一个简单的示例，展示了如何使用这个方法：
import datetime

# 获取当前日期时间
current_datetime = datetime.datetime.now()

# 将日期时间对象格式化为字符串
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 打印格式化后的日期时间字符串
print("Formatted Datetime:", formatted_datetime)

在这个例子中，strftime 方法的参数 "%Y-%m-%d %H:%M:%S" 是一个格式化字符串，它指定了日期时间的输出格式。具体来说：

1.%Y 表示四位数的年份
2.%m 表示月份（01-12）
3.%d 表示一个月中的某一天（01-31）
4.%H 表示小时（24小时制，00-23）
5.%M 表示分钟（00-59）
6.%S 表示秒（00-59）

你可以根据需要修改这个格式化字符串，以获得你想要的日期时间格式。