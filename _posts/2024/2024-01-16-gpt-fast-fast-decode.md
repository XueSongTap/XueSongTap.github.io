---
layout: article
title: gpt-fast 预测性推理 speculative decode 自测
tags: gpt LLM pytorch
---


## gpt-fast 


### 参考：
https://github.com/pytorch-labs/gpt-fast

https://pytorch.org/blog/accelerating-generative-ai-2/

https://mp.weixin.qq.com/s/QlpyjnkuNKGe_KP2Ut0Fgg


### 环境配置

```
git clone git@github.com:pytorch-labs/gpt-fast.git
```

官方没有指定，docker容器，选择拉取最新的pytorch/pytorch 验证环境符合要求
shm-size 需要设置大一些，否则torch.dymno 会有 no space left 报错



**截至2024.1.16, gpt-fast 这种必须要求pytorch-nightly** 
```
docker pull pytorch/pytorch:latest
docker run -it  --gpus all --ipc host \
                --name yxc.gpt-fast.cu121 \
                --shm-size=32G \
                -v /data0/yxc01841111/code:/code -v /data0/yxc01841111/models:/models \
                nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
```
```
pip install sentencepiece huggingface_hub
```



### 模型转换


 python scripts/convert_hf_checkpoint.py --checkpoint_dir /models/llama2/llama-2-7b-chat-hf/llama-2-7b-chat-hf


### 模型量化

python quantize.py --checkpoint_path /models/llama2/llama-2-7b-chat-hf/llama-2-7b-chat-hf/model.pth --mode int8


### 推理汇总


| 实验编号 | DRAFT_MODEL            | DRAFT_MODEL量化 | speculate_k | MODEL                   | MODEL量化 | 显卡型号     | 显卡数量nproc_per_node | 开启compile | prompt             | compile_prefill | 平均tokens/sec | memory used GB | Bandwidth achieved GB/s(典型值) | Mean Accepted:  | max_new_tokens | num_samples | temperature | pytorch版本                |
|------|------------------------|---------------|-------------|-------------------------|---------|----------|--------------------|-----------|--------------------|-----------------|--------------|----------------|------------------------------|-----------------|----------------|-------------|-------------|--------------------------|
| 1    | \                      | \             | \           | llama-2-70b-chat-hf     | int8    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 48.8         | 10.42          | 458.26                       | \               | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 2    | \                      | \             | \           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 43.73        | 18.47          | 796.17                       | \               | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 3    | llama-2-7b-chat-hf     | int8          | 5           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | y               | 32.6         | 20.27          | 587.95                       | 3.495831505     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 4    | llama-2-7b-chat-hf     | int8          | 4           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 31.92        | 20.03          | 564.33                       | 2.96131528      | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 5    | llama-2-7b-chat-hf     | int8          | 6           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 32.03        | 20.03          | 616.35                       | 4.739251815     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 6    | llama-2-7b-chat-hf     | int8          | 5           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 33.45        | 19.77          | 619.82                       | 3.495831505     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 7    | llama-2-7b-chat-hf     | int8          | 8           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 32.03        | 20.03          | 578.85                       | 4.739251815     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 8    | llama-2-7b-chat-hf     | int8          | 1           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 23.5         | 20.03          | 415.47                       | 0.883639059     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 9    | llama-2-7b-chat-hf     | bf16          | 5           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 32.86        | 20.75          | 610.06                       | 3.495614035     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 10   | llama-2-70b-chat-hf    | int8          | 5           | llama-2-70b-chat-hf     | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 21.06        | 28.32          | 392.27                       | 4.944636678     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 11   | llama-2-7b-chat-hf     | int8          | 5           | llama-2-7b-chat-hf      | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 52.63        | 4.4            | 109.02                       | 4.832951945     | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 12   | \                      | \             | \           | llama-2-7b-chat-hf      | bf16    | A100 40g | 8                  | y         | def quicksort(arr) | n               | 189.23       | 2.58           | 406.56                       | \               | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 13   | CodeLlama-7b-Python-hf | int4.g32      | 6           | CodeLlama-34b-Python-hf | bf16    | A100 40g | 1                  | y         | def quicksort(arr) | n               | OOM          | OOM            | OOM                          | OOM             | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 14   | CodeLlama-7b-Python-hf | int4.g32      | 6           | CodeLlama-34b-Python-hf | int8    | A100 40g | 1                  | y         | def quicksort(arr) | n               | 29.68        | 41.68          | 1111.16                      | 4.11083499      | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
| 15   | \                      | \             | \           | CodeLlama-34b-Python-hf | int8    | A100 40g | 1                  | y         | def quicksort(arr) | n               | 34.13        | 35.41          | 1160.83                      | \               | 200            | 50          | 0           | 2.3.0.dev20231219+cu121  |
|      |                        |               |             |                         |         |          |                    |           |                    |                 |              |                |                              |                 |                |             |             |                          |


### 结论

目前发现 7b 模型针对70b 不管是尝试了不同的量化方式，还是设置不同的speculate_k，预测性推理都是负优化, 暂时无足够显存（A100 80g） 来跑官方demo 


### 方向

咨询了改进的可能性


#### 1. prompt 优化

对特定prompt 有要求，

 def quicksort(arr) 

这种输出代码的prompt, 不一定适合预测性推理


#### 2.更小的draft model

llama 最小7b 规模，怀疑还是不够小，

可以尝试opt，小模型比较多，大模型也够大
