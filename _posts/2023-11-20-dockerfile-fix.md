---
layout: articles
title: Dockerfile 构建 替换国内源
tags: docker
---


修改dockerfile 用国内源：

替换ubuntu源：
```bash
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list
```

替换pip源
```bash
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

