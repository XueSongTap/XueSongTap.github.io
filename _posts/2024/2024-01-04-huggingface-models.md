---
layout: article
title: HuggingFace 模型调用
tags: HuggingFace LLM transformer
---


## HuggingFace模型调用



### 模型下载
可以从官网下载，出于国内网络连接问题，也可使用镜像网站（非官方）下载


镜像地址： https://aliendao.cn/ ， 用--repo_id指定对应模型的名称即可下载：


```bash
$ pip install huggingface_hub
$ wget http://61.133.217.142:20800/download/model_download.py
# 比如下载THUDM/chatglm-6b
$ python model_download.py --mirror --repo_id THUDM/chatglm-6b

=> 所有文件保存到dataroot/models/THUDM/chatglm-6b目录下
```


### 模型文件解析
以下载的THUDM/chatglm-6b文件们为例

```bash
【1】config.json	 => 通过'auto_map'指明模型对应python文件的class，通过'architectures'指定模型class
  "architectures": [
    "ChatGLMModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_chatglm.ChatGLMConfig",
    "AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
    "AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"
  },

【2】modeling_chatglm.py => 模型推理主入口是ChatGLMModel，依赖下面的configuration_chatglm去加载模型config，依赖quantization去做量化
【3】configuration_chatglm.py
【4】quantization.py

【5】tokenizer_config.json   => 指明tokenizer来自tokenization_chatglm.py里的ChatGLMTokenizer
  "auto_map": {
    "AutoTokenizer": [
      "tokenization_chatglm.ChatGLMTokenizer",
      null
      ]
  }
【6】tokenization_chatglm.py

【7】pytorch_model.bin.index.json  => 指定下面的weight bin文件对应到模型的哪个layer
【8】pytorch_model-00001-of-00007.bin	
    pytorch_model-00002-of-00007.bin	
    pytorch_model-00003-of-00007.bin
    pytorch_model-00004-of-00007.bin
    pytorch_model-00005-of-00007.bin	
    pytorch_model-00006-of-00007.bin
    pytorch_model-00007-of-00007.bin


【9】MODEL_LICENSE
【10】README.md

```

### 模型文件使用
Huggingface所开源的python库transformers实现了对上述模型文件的使用


使用： 详见https://www.cnblogs.com/chenhuabin/p/16997607.html


三大件AutoModel， AutoTokenizer， AutoConfig： AutoModel为模型，AutoTokenizer和AutoConfig类似AutoModel的输入参数。 AutoTokenizer把promt文字转成token值作为模型输入，AutoConfig指定模型mha多少个头，decorder循环多少层等等。
AutoModel加载config.json里的"AutoModel"和pytorch_model.bin.index.json， AutoConfig加载config.json里的"AutoConfig"，AutoTokenizer加载tokenizer_config.json
```bash

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
tokens = tokenizer("今天天气真好")
print(tokens)

config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config.num_hidden_layers=5 #可以修改config

model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True)
print(model)

output = model(**tokenizer("今天天气真好", return_tensors="pt"))

print(output)

```

使用自带config

```bash
# transformers自带很多模型的config，这里相当于加载了自带的而非下载的configuration_bert.py
# 这样就不需要使用AutoConfig.from_pretrained去加载了
from transformers import BertConfig
config = BertConfig()
print(config)
```


### HuggingFace模型调用自定义kernel实现

这是个伪命题，当我们下载使用HF模型，调用的是HF的模型执行，也就是pytorch原生的执行。 如果想使用自己定义的kernel，就需要把pytorch的模型构建中使用的方法替换成自定义的方法，这无异于重新构建了一遍模型。

在自研LLM框架中，就是重新构建了模型，但使用了来自HuggingFace下载的模型配置(decorder layer num,  mha head num等)和权重