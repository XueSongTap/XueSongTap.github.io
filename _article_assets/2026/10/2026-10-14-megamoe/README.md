# MegaMoE：EP token 流动与通信计算重叠 · 配图

对应文章：[MegaMoE：EP token 流动与通信计算重叠](../../../../_posts/2026/10/2026-10-14-megamoe.md)。

来源：2026-10-14 的 own_log MegaMoE 笔记配图。迁入博客后，后续文章配图修改以本目录 `source/` 为准；不与笔记目录双向同步。正文保留白底 PNG，制作文件仅在本目录维护。

## 文件

| 图 | PNG 预览 | 可编辑源文件 | 矢量文件 |
|---|---|---|---|
| 一个 token 在 EP 中的路径 | [PNG](../../../../img/2026/10/14/01-ep-token-flow.png) | [TikZ](source/01-ep-token-flow.tex) | [SVG](exports/01-ep-token-flow.svg) / [PDF](exports/01-ep-token-flow.pdf) |
| 专家内部张量形状 | [PNG](../../../../img/2026/10/14/02-expert-tensor-shapes.png) | [TikZ](source/02-expert-tensor-shapes.tex) | [SVG](exports/02-expert-tensor-shapes.svg) / [PDF](exports/02-expert-tensor-shapes.pdf) |
| 通信与计算重叠 | [PNG](../../../../img/2026/10/14/03-communication-compute-overlap.png) | [TikZ](source/03-communication-compute-overlap.tex) | [SVG](exports/03-communication-compute-overlap.svg) / [PDF](exports/03-communication-compute-overlap.pdf) |

`source/common.tex` 提供统一的字体、颜色和矩阵绘制方法；`exports/` 保存每张图的 PDF、SVG 和透明 PNG，`build/` 保存编译中间文件与日志（由 `.gitignore` 排除）。没有复制外部参考图片，全部为本文生成的教学示意。

## 形状与语义核对

图 1 假设 EP=2、总专家数 E=4、Top-k=2、专家内 TP=1，只追踪一个源 token，不展示其他 token、padding 或 shared expert。

| 符号 | 类型 / 每项语义 | 形状 | 生产者 → 消费者 |
|---|---|---|---|
| `s_0` | 连续路由分数；第 e 项是专家 e 的分数 | [1,E] | 路由打分 → TopK |
| `S_0` | 整数专家 ID，范围 0…E-1；第 j 项是选中槽位 j 的专家 | [1,k] | TopK → Dispatch / owner 映射 |
| `a_0` | 浮点路由权重；第 j 项与 S_0[j] 配对 | [1,k] | 路由权重处理 → 加权专家贡献 |
| `x_0` | 完整 hidden 输入行 | [1,D] | 源 GPU → 选中专家 |
| `o_0,j` | 槽位 j 的未加权专家输出 | [1,D] | 专家 → 源 GPU 的逻辑加权归并 |
| `y_0` | 按所选专家槽位求和后的输出 | [1,D] | Combine → 下一层 |

例子中 `S_0=(2,1)` 对应 `a_0=(0.6,0.4)`，权重和为 1 只是本例数值设定，不宣称所有路由实现均采用相同归一化。无重复 ID、无 padding。连续专家分组对应 GPU 0={0,1}、GPU 1={2,3}。

图 2 的矩阵均为浮点逻辑值，忽略量化与物理布局，固定一个专家 e：

| 张量 | 形状 | 轴语义 / 操作 |
|---|---|---|
| X_e | [M_e,D] | 收到的 token 行 × hidden 通道 |
| W_1,e | [D,2H] | 收缩 D；输出 gate/up 两组通道 |
| Z_e | [M_e,2H] | Linear 1 输出 |
| G_e、U_e、A_e | [M_e,H] | 沿 2H 通道 split；逐元素 SwiGLU |
| W_2,e | [H,D] | 收缩 H；还原 hidden 通道 |
| O_e | [M_e,D] | 专家输出 |

逻辑归并为 `y_t = sum_j a_tj * f_(S_tj)(x_t)`。当前代码把 a 乘到 SwiGLU 激活后再进行 Linear 2；数学线性等价及量化差异在笔记中说明。

## 几何核对

矩阵面均以行数决定高度、列数决定宽度。以下为各图独立的符号到物理长度映射；名义长度只用于解释轴结构，不反映真实模型的数值大小。

| 图 | 行轴长度 | 列轴长度 |
|---|---|---|
| 图 1 | 单 token 行=0.8 cm | E=3.2 cm、k=1.6 cm、D=3.2 cm |
| 图 2 | M_e=1.2 cm、D=2.4 cm、H=1.8 cm | D=2.4 cm、H=1.8 cm、2H=3.6 cm |
| 图 3 | 每条任务条高=0.55 cm | 一个时间单位 τ=1.5 cm |

- 图 1 的 x、两个专家输出和 y 同为 [1,D]，尺寸完全相同；ID 使用离散槽位，不用数值热图表示整数 ID。
- 图 2 两次乘法中的 D / H 收缩边长度相等；相同形状张量在所有阶段保持相同尺寸。
- G、U 两个 H 宽度精确拼成 2H，只有 partition 复合对象内部共享边界。
- 图 3 分阶段与流水线时间轴使用同一单位；9τ 和 5τ 独立手算核对。
- 图 3 的 Return 仅表示贡献回传，最终归并另有同步；所有时间均为理想教学假设。

## 复现

依赖：TeX Live 的 XeLaTeX、ctex/Fandol、TikZ、standalone，以及 Poppler 的 `pdftoppm` / `pdftocairo`。

```bash
python3 _article_assets/2026/10/2026-10-14-megamoe/render.py
```

以上命令在博客仓库根目录运行；脚本根据自身位置解析所有路径，也支持其他工作目录。修改 `source/` 后运行脚本，将更新正文的白底 PNG 和 `exports/` 中的 PDF、SVG、透明 PNG。PNG 为 180 DPI，适合 Markdown 阅读；SVG/PDF 适合放大与复用。

验证包括：源码语义与形状核对、编译错误及缺字检查、PNG 目检，以及公式、标签、连接线和版面间距检查。渲染后逐图检查最新版本，不以编译通过替代目检。

## 源码依据

示意图的技术依据与正文一致，固定到 DeepGEMM 版本 `1fc9c66975ee322d628b3288534ca0e9ce12a64d`：

- [SM100 MegaMoE 内核](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)。
- [Scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)。
- [Buffer layout](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/layout/mega_moe.cuh)。
