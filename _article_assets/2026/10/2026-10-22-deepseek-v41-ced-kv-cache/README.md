# DeepSeek CED 与 CSA2 配图

对应文章：[DeepSeek V4.1：拆分 Transformer，压缩 KV Cache](../../../../_posts/2026/10/2026-10-22-deepseek-v41-ced-kv-cache.md)。

- `source/`：唯一可编辑的 TikZ 源码。
- `exports/`：PDF、SVG、透明 PNG。
- `build/`：编译日志与中间文件，Git 忽略。
- 正文白底 PNG 位于 `../../../../img/2026/10/22/`。

重建依赖：Python 3、TeX Live（XeLaTeX、ctex/Fandol、TikZ）、Poppler（pdftoppm、pdftocairo）。在本目录执行 `python3 render.py`，脚本从自身路径寻找仓库，不依赖固定的用户目录。

图 1 展示 decoder 的 m=1 全局 KV 由 encoder 输出投影而来，以及每层继续生成主 query 的路径。矩阵几何单位为 0.28 cm；示意轴 N=8、d=6、d_kv=4，矩阵面分别为 8×6、6×4、8×4。图中省略归一化与位置变换，不代表物理内存布局。

图 2 展示 Full 建立数值缓存，评分→整数 Top-k 地址→gather→选中数值→本层注意力的链条。三种模式说明的是状态来源；流程的共同读取步骤适用于所有模式，Reuse 直接使用已有地址而不重算评分。S 的 1、4、6 是示意地址，不代表实际模型 Top-512 的取值。

来源：依据 [DeepSeek-V4.1-Flash 技术报告 v1，§2.2–2.3](https://arxiv.org/html/2609.19969v1) 与 own_log 中已有视频讲义重绘。图中未使用视频截图或外部图像。

发布正文只引用白底 PNG；此目录保留制作文件，不作为 Jekyll collection 发布。
