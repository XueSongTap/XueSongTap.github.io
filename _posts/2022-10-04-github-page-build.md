---
title: github pages 搭建
tags: TeXt
---

## 搭建流程
创建github.io结尾仓库，然后找一个jekly模板导入仓库，fork，或者下载再上传都行

博客内容直接用md写，静态模板会帮助转换成html网页，语法就参照markdown 语法，有部分区别直接参考模板文档即可


## bug 相关

### github actions 报错
pages build and deployment: Some jobs were not successful


### 细节报错
![Jekyll_build_error](/img/221004/Jekyll_build_error.png)


### 删除相关代码

已经用代码块包裹，还是不行，删除后流水线恢复正常