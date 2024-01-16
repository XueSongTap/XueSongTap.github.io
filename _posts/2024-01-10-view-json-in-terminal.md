---
layout: articles
title: 终端预览大json文件方式
tags: json preview less terminal
---

## 背景

查看较大 json文件，vscode-ssh 太大也无法查看


## 解决
安装
```
yum install jq
```


```
jq '.' filename.json | less
```
这里的 '.' 是一个简单的 jq 过滤器，代表将整个输入JSON文件作为输出。


但是这样的话，失去了jq 自带的json 高亮，使用下面命令：
```
jq -C '.' filename.json | less -R
```


这里的-C选项告诉jq输出颜色化的JSON，而less命令的-R选项则允许显示ANSI颜色转义序列。

