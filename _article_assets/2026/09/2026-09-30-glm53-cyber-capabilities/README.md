# GLM-5.3 网络安全报告解读配图

对应文章：[从 Anthropic 的网络安全报告看 GLM-5.3](../../../../_posts/2026/09/2026-09-30-glm53-cyber-capabilities.md)。

这里保存三张机构原图及无损 PNG 导出。未重新绘制、裁切、改写标签或改变数据；仅转换文件格式，并将透明背景合成到白底。文章中保留中文图解与来源。版权归原机构；保存文件不代表取得额外授权。

## 文件与来源

| 文件名（不含扩展名） | 原图 | 来源 |
| --- | --- | --- |
| `glm53-exploitation-budget` | Anthropic Figure 2 | [报告](https://www.anthropic.com/research/glm-5-3-and-the-spread-of-advanced-cyber-capabilities) / [原图 CDN](https://www-cdn.anthropic.com/images/4zrzovbb/website/ab29550606b0df3c6c070f4c65b758aa18bdecbb-1280x1588.webp) |
| `glm53-harmful-task-engagement` | Anthropic Figure 5 | [报告](https://www.anthropic.com/research/glm-5-3-and-the-spread-of-advanced-cyber-capabilities) / [原图 CDN](https://www-cdn.anthropic.com/images/4zrzovbb/website/a8b70c8d729325cc988af66416f7c0811d2097a5-1280x1257.webp) |
| `glm53-caisi-benchmarks` | CAISI/NIST Figure 2 | [报告](https://www.nist.gov/news-events/news/2026/09/caisis-assessment-zais-glm-53-cyber-capabilities) / [原图](https://www.nist.gov/sites/default/files/styles/1400_x_1400_limit/public/images/2026/09/17/GLM-5.3%20Cyber%20Performance.png.webp?itok=zZwJYwsT) |

取得日期：2026-09-30。Anthropic 的站点 CDN 直连失败，因此使用 Sanity 的同项目、同数据集、同资产 ID 下载：`https://cdn.sanity.io/images/4zrzovbb/website/`。下载后已目检图号对应的内容。

- `source/`：唯一输入，原始 WebP 文件。
- `exports/`：可复用 PNG 导出。
- `build/`：本地检查材料，Git 忽略。
- 正文图片目录：[img/2026/09/30](../../../../img/2026/09/30/)。

## 重建

依赖 Python 3 与 Pillow，当前制作环境使用 Pillow 12.2.0。在仓库根目录运行：

```sh
python3 _article_assets/2026/09/2026-09-30-glm53-cyber-capabilities/render.py
```

脚本根据自身路径定位仓库，离线读取原图，同步更新 `exports/` 和正文图片。更新输入后重建，并目检三张最终 PNG 的文字、完整边界和颜色。
