# 模型量化步骤linux
- 1、从git官网下载llama.cpp,git clone https://github.com/ggerganov/llama.cpp
- 2、打开llama.cpp，文件夹：cd llama.cpp
- 3、创建build 文件夹：mkdir build
- 4、打开build文件夹：cd build
- 5、在当前目录的上一级目录中寻找 CMakeLists.txt 文件，并使用该文件进行项目的配置和生成：cmake ..
- 6、构建程序：cmake --build . --config Release
- 7、返回llama.cpp文件夹：cd ..
- 8、量化模型：python convert-hf-to-gguf.py Path_To_Qwen，量化千问模型需要使用convert-hf-to-gguf，其他模型使用convert.py。这里会将模型量化为16PF版的。
- 9、压缩模型：一般将模型压缩为q8_0即可，需要先进入build文件后：执行指令 bin/quantize  /root/autodl-tmp/fineturn/qwen  /root/autodl-tmp/fineturn/gguf/Qwen-q8_0.gguf q8_0。一共三个参数，1、需要压缩的模型地址，2、压缩后存储的模型地址及名称，3、压缩的比例
