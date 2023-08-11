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

    Rendering: _posts/2023-07-27-drive-dev-advise.md/#excerpt
  Pre-Render Hooks: _posts/2023-07-27-drive-dev-advise.md/#excerpt
  Rendering Markup: _posts/2023-07-27-drive-dev-advise.md/#excerpt
         Rendering: _posts/2023-07-29-override-implement.md
  Liquid Exception: Liquid syntax error (line 127): Variable '{{1}' was not properly terminated with regexp: /\}\}/ in /github/workspace/_posts/2023-08-02-cpp-guide-namespace.md
/usr/local/bundle/gems/liquid-4.0.4/lib/liquid/block_body.rb:136:in `raise_missing_variable_terminator': Liquid syntax error (line 127): Variable '{{1}' was not properly terminated with regexp: /\\}\\}/ (Liquid::SyntaxError)


### 删除相关代码

已经用代码块包裹，还是不行，删除后流水线恢复正常