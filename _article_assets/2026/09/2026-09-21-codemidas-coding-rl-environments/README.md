# CodeMidas 论文配图

对应文章：[博客正文](../../../../_posts/2026/09/2026-09-21-codemidas-coding-rl-environments.md)。
正文图片：[img/2026/09/21](../../../../img/2026/09/21/)。

来源：CodeMidas: Scaling Agentic Coding RL Environments from Code Itself，arXiv:2609.22068v1，2026-09-18。
原 PDF：https://arxiv.org/pdf/2609.22068v1 。图片版权归论文作者所有；此处保留原图内容用于论文讲解，正文注明来源。

- `source/codemidas-v1.pdf`：原论文，固定为 v1。
- `source/crops.json`：唯一裁切配置，页码从 1 开始，矩形为 PDF point 坐标，原点在左上角。
- Figure 1 / 第 4 页：数据管线，`codemidas-pipeline.png`。
- Figure 5 / 第 7 页：外部评测效果，`codemidas-benchmark-gains.png`。
- Figure 8 / 第 9 页：任务数量与质量消融，`codemidas-data-quality.png`。
- `exports/`：216 dpi 白底 PNG，重建时同步到正文图片目录。
- `build/`：目检用整页渲染，Git 忽略。

依赖：Python 3、PyMuPDF。安装 `python3 -m pip install pymupdf` 后，运行：

```sh
python3 render.py
```

脚本通过自身位置定位仓库，不依赖运行目录。调整裁切范围时修改 `source/crops.json`，重建后检查文字、坐标轴和图例是否完整。
