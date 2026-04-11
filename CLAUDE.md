# AGENTS.md

这个仓库的主要用途是记录和维护个人博客文章，不是一个需要做通用前端、Node.js 或 npm 开发的项目。

在这个仓库里工作的 agent，默认应当优先处理文档编辑、文章结构整理和配图管理。不要因为仓库里存在 Jekyll 主题文件、`package.json` 或模板脚手架，就主动去做 JS、npm、主题模板相关修改，除非用户明确提出这类需求。

## 仓库定位

- 主要内容位于 `_posts/`。
- 大多数用户请求都应优先理解为：修改 Markdown 文章及其引用图片。
- `_layouts`、`_includes`、`_sass`、`assets`、`docs`、`test` 等目录属于主题或演示环境，通常不要主动修改。

## 文章规范

- 文章存放路径采用 `_posts/YYYY/MM/`。
- 文件名格式为 `YYYY-MM-DD-slug.md`。
- 每篇文章开头使用简洁的 YAML front matter，形式如下：

```yaml
---
layout: article
title: 文章标题
tags: 标签
---
```

- 除非用户明确要求新增字段，否则保持现有 front matter 风格。
- 普通文章默认使用 `layout: article`。
- 中文文章标题直接使用中文，不要无故改成英文。
- `tags` 可以是单个标签，也可以是与现有文章一致的空格分隔标签列表。

## 图片规范

- 文章配图优先放在 `img/YYYY/MM/DD/` 下，并与文章日期对应。
- Markdown 图片引用使用站点根路径，例如：

```md
![说明](/img/2025/12/30/example.png)
```

- 为某篇文章新增图片时，优先复用该文章对应日期目录。
- 不要主动重构旧图片目录，除非用户明确要求整理。

## 编辑原则

- 优先做小范围、局部性的文章修改。
- 在没有明确要求重写的情况下，尽量保持作者现有的结构、术语和表达风格。
- 如果要优化文章，优先关注这些方面：
  - 错别字和明显病句
  - 表达是否更清楚
  - 小节结构是否更顺
  - 公式和配图说明是否完整
  - 文中图片引用是否一致

- 不要引入与当前文章无关的重构。
- 不要主动修改站点主题、npm 工具链或 Jekyll 配置，除非用户明确要求。

## 默认行为

- 在需求不明确时，默认把任务理解为“文档相关任务”。
- 优先修改 Markdown 文章和补充图片引用，而不是改模板或构建链路。
- 如果一个请求存在歧义，在动 `_posts` 以外的渲染或构建文件前，先确认是否真的需要修改那些内容。

## 文章 URL 规则

Jekyll 生成的博客文章 URL 遵循以下规则：

```
_posts/YYYY/MM/YYYY-MM-DD-slug.md  →  https://xuesongtap.github.io/YYYY/MM/DD/slug.html
```

例如：
- `_posts/2025/12/2025-12-29-comm.md` → `https://xuesongtap.github.io/2025/12/29/comm.html`
- `_posts/2025/12/2025-12-30-tensor-parallel-comm.md` → `https://xuesongtap.github.io/2025/12/30/tensor-parallel-comm.html`
