---
layout: articles
title: vscode remote-ssh 连接失败 glibc版本问题
tags: python
---

## 问题描述
在VSCode更新到1.86版本后，一些之前运行正常的Remote-SSH连接突然无法使用。这个问题主要出现在Ubuntu 18.04和更早版本的Linux系统上，而在较新的系统如Debian 12上则不受影响。

## 报错

更新1.86完成后，原来好好的 Remote SSH 再也连不上了，提示 GLIBC 版本太低
```bash
Warning: Missing GLIBC >= 2.28! from /lib/x86_64-linux-gnu/libc-2.27.so
Error: Missing required dependencies. Please refer to our FAQ https://aka.ms/vsc
code-remote/faq/old-linux for additional information.
```
## 分析

这个错误明确表示VSCode 1.86版本要求目标系统的GLIBC版本至少为2.28，而Ubuntu 18.04使用的是GLIBC 2.27。这是因为：
- VSCode 1.86更新了其底层依赖项，要求更高版本的系统库
- Ubuntu 18.04作为2018年发布的长期支持版本，使用的是较旧的GLIBC版本
- 较新的Linux发行版（如Debian 12）默认包含更新的GLIBC版本，因此不受影响

## 解决
目前最直接的解决方法是：

- 降级VSCode版本：退回到VSCode 1.85版本，该版本与旧版GLIBC兼容
- 关闭自动更新：在VSCode的Preferences(设置)中禁用自动更新功能，防止再次升级到不兼容的版本