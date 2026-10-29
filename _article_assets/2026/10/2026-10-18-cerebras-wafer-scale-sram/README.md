# Cerebras 片上 SRAM 文章的图片与资料记录

对应文章：[_posts/2026/10/2026-10-18-cerebras-wafer-scale-sram.md](../../../../_posts/2026/10/2026-10-18-cerebras-wafer-scale-sram.md)。

原笔记保留于 `/Users/yxc/code/own_log/2026/10/2026-10-01-cerebras-wafer-scale-sram.md`。本次按 2026-10-18 新文章日期整理到博客，没有修改原笔记。六张配图来自原笔记的 `assets/images/2026-10-01-cerebras-sram/`，逐字节复制，未重绘、未增加裁剪。

## 正文图片

| 图号 | 博客图片（`img/2026/10/18/`） | 原图片 | 来源 |
| --- | --- | --- | --- |
| 1 | `cerebras-core-to-wafer.png` | `03-core-to-wafer.png` | IEEE Micro 2023，PDF 第 7 页，Figure 7 |
| 2 | `cerebras-core-physical-layout.png` | `01-core-physical-layout.png` | 同上，第 3 页，Figure 2 |
| 3 | `cerebras-local-sram-banks.png` | `02-local-sram-banks.png` | 同上，第 4 页，Figure 3 |
| 4 | `cerebras-mesh-fabric.png` | `04-mesh-fabric.png` | 同上，第 7 页，Figure 8 |
| 5 | `cerebras-gpu-vs-wse-memory.png` | `05-gpu-vs-wse-memory.png` | Cerebras 2024 推理介绍的官网示意图 |
| 6 | `cerebras-prefill-decode.png` | `06-prefill-decode.png` | Cerebras 2026 CS-4 介绍的官网示意图 |

[source/sources.json](source/sources.json) 保留原始获取日期、来源 URL、PDF 页码、裁剪参数、像素尺寸及 SHA-256，并补充博客文件映射。图 1–4 属于 WSE-2，文章通过图注区分代际。图 5–6 保留官网原始设计与颜色。版权归原作者／权利人，正文图注和参考文献保留来源。

## 恢复与来源复现

最直接的恢复方式是按上表从原笔记图片目录复制文件并改名，无需绘图工具。最终图片的 SHA-256 必须与 `source/sources.json` 对应记录一致。

若从原始来源重新获取，图 1–4 需下载 JSON 中的论文 PDF，用 Poppler `pdftoppm -scale-to 2400` 渲染指定页，再按 `crop_pixels_xywh` 截取图区；图 5–6 从 JSON 中完整的 `asset_url` 获取，保留 CDN 的缩放参数。裁剪可用 Pillow，坐标为 `(x, y, x + width, y + height)`。不同渲染器版本可能改变像素，因此原笔记保留的 PNG 与校验值是本次复用版本的依据。

本次没有新增自制绘图，也没有需要维护的绘图源或导出版本。临时渲染与检查产物如需保存，应置于 `build/`，不进入发布图片目录。

## 本次核查带来的正文更新

- 加入 Sean Lie 的 [Hot Chips 2024 官方演示](https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf)：第 3 页是 WSE-3 整片规格，第 29 页直接披露每核 48 kB SRAM 和 512 B local cache。替换原笔记从总容量反推每核容量的间接说明；不将 WSE-2 的八 bank 结构直接套用于 WSE-3。
- 21 PB/s 沿用 2024 资料的近似值；2026 CS-4 公告对照表列 21.6 PB/s，已在正文参考 [10] 交代。
- 区分 WSE-3 Turbo 单晶圆 44 GB／43.2 PB/s 与三晶圆 CS-4 系统。
- AWS／AMD 描述为已公布的合作架构，CS-6 描述为路线图；不将预期性能或上线计划改写成已验证部署。
- GEMV 的算术强度、6T 晶体管数量及 70B 容量均为正文明确给出假设的手算。
