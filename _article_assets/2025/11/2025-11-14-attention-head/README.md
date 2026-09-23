# 注意力头共享关系图

对应文章：[注意力头设计](../../../../_posts/2025/11/2025-11-14-attention-head.md)，第 2 节。
正文图片：[白底 PNG](../../../../img/2025/11/14/attention-head-sharing.png)。

## 文件与重建

- `source/attention-head-sharing.tex`：唯一可编辑 TikZ 源码；中文使用 `ctex` 的 Fandol 字体。
- `exports/`：PDF、SVG、透明 PNG；正文仅引用白底 PNG。
- `build/`：编译日志、中间产物与白底检查图，已忽略。
- `render.py`：依据脚本位置定位仓库，更新全部导出及正文图片。

依赖：Python 3、TeX Live（XeLaTeX、TikZ、ctex、standalone）、Poppler（pdftocairo）。
在任意工作目录运行：

```sh
python3 /path/to/XueSongTap.github.io/_article_assets/2025/11/2025-11-14-attention-head/render.py
```

## 语义与形状台账

这是一张头共享关系图；连线不是计算或内存复制操作。三个面板是三种替代结构，非串行阶段。

| 对象 | 类型及含义 | 完整形状 | 单头单 batch 图面 |
| --- | --- | --- | --- |
| Q | 浮点查询值；保持 4 个头 | B × 4 × Tq × d | Tq × d |
| K | 浮点键值；每组一份 | B × Hkv × Tkv × d | Tkv × d |
| V | 浮点值向量；每组一份 | B × Hkv × Tkv × d | Tkv × d |
| O | 浮点输出；每个 Q 头分别计算 | B × 4 × Tq × d | 未画 |
| h | 整数 Q 头编号 | 标量，0 ≤ h < Hq | 下标 |
| g(h) | 整数 KV 头编号 | 标量，0 ≤ g < Hkv | 共享连接 |
| M | 加性 mask；允许位置为 0、屏蔽位置为负无穷 | Tq × Tkv，按 batch/head 广播 | 未画 |
| C_KV | 单层 K+V 理论存储字节数 | 标量 | 面板底部比例 |

采用等大连续分组：g(h) = floor(h / (Hq/Hkv))，Hq 必须被 Hkv 整除。
三个面板 Hkv=4、2、1；各自映射为 [0,1,2,3]、[0,0,1,1]、[0,0,0,0]。
每头收缩：(Tq × d)(d × Tkv) → Tq × Tkv；softmax 沿 key 位置轴；再乘 (Tkv × d) → Tq × d。
K/V 不必物理复制到每个 Q 头。缓存包含 K 和 V 两份，因此为 2 B Tkv Hkv d s；不计 allocator、分页元数据、量化 scales 等额外存储。

## 几何与视觉约定

- 所有 d 边宽 0.5 cm，Tq 边高 0.55 cm，Tkv 边高 1.10 cm，跨面板复用；示意边长不代表实际数值比例。
- K 和 V 是独立同形状对象，用中性括号标出一组；各组之间不拼成矩阵。
- Q/K/V 分别使用蓝/橙/紫三种固定角色色。头数按实际示例逐个画出，不省略。
- 顶部公式，三种结构的头面板，底部统一维度/对象/机制说明；无署名或水印。

## 来源与检查

原创绘图；结构参考 Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*：https://arxiv.org/abs/2305.13245 。
缓存比例由上述张量形状推导，不引用文章原有性能表格作为测量依据。
已检查 4/2/1 个 K/V 头、映射、形状、收缩轴和 1/1⁄2/1⁄4 缓存比例；渲染 PNG 后目检布局和文字；日志无 overfull/underfull box。

## 第 3 节：注意力连接模式图

- `source/attention-mask-patterns.tex`：第二张图的唯一可编辑 TikZ 源码。
- `exports/attention-mask-patterns.{pdf,svg}` 及透明 PNG：可复用导出。
- [正文白底图](../../../../img/2025/11/14/attention-mask-patterns.png)。
- `render.py` 现在遍历 `source/*.tex`，一次重建本篇所有图；每张图的命令日志单独保存。

### 语义与几何台账

固定单个 batch、单个 attention head，N=10、w=3（含自身）、G={0,5}。
A 是布尔可见性矩阵，形状 N×N；行 i 为 Query，列 j 为 Key，索引范围 0..9。
M 是加性 mask：允许处 0，禁止处负无穷；P 为 softmax 后概率，均与 A 同形状，但三者并非同一对象。
Q/K/V 形状均为 N×d；QKᵀ 沿 d 收缩得到 N×N，softmax 沿 j 归一化；PV 沿 Key 位置收缩得到 N×d。
此图只画 A，不将注意力概率当作二值 mask，也不将 M 中的数值 0 画成“不可见”。

可见位置规则：

1. 完整因果：j ≤ i。
2. 因果滑窗：0 ≤ i-j < 3。
3. 自定义局部＋全局：j ≤ i 且 (i-j < 3 或 i∈G 或 j∈G)。

第三种规则将全局行、全局列都裁到因果区域；并非 Longformer 的完整模型结构复刻。
蓝色表示基础允许连接，橙色仅表示第三图中超出局部窗口的新增连接；所有禁止位置留白。
每个格子间距 0.43 cm，实际色块边长 0.38 cm，三张矩阵图面统一 4.30×4.30 cm；只使用二值支持，不引入虚假的强弱色阶。

独立核对：完整因果 55 个可见位置；滑窗 27 个；局部＋全局 38 个（新增 11 个）。
第 8 行分别为 {0,1,2,3,4,5,6,7,8}、{6,7,8}、{0,5,6,7,8}；第 5 行在第三图为 {0,1,2,3,4,5}。
所有对角线均可见、所有上三角均为空，不会出现全屏蔽行。

机制参考：https://arxiv.org/abs/2004.05150 ，自定义示例和图形均为原创。
已按公式独立核对源文件中每个着色位置，渲染后目检形状、留白、文字间距与裁切。


## 第 5 节：MLA latent 解码路径

- 源码：`source/mla-latent-decode.tex`。
- 导出：`exports/mla-latent-decode.pdf`、SVG、透明 PNG。
- 正文白底图：`../../../../img/2025/11/14/mla-latent-decode.png`。
- 通过同一 `render.py` 重建。

### 形状与语义台账

行向量记法；单个 batch、当前 token t、头 h。T 是含当前位置的有效前缀长度。

| 对象 | 形状 | 语义 |
| --- | --- | --- |
| C | T×r | 归一化后的共享 KV latent，持久浮点缓存 |
| K^R | T×d_R | RoPE 后共享位置 key，持久浮点缓存 |
| q^C | 1×d_c | 当前头内容 query |
| U^K_h | r×d_c | 当前头内容 Key 投影权重 |
| q tilde | 1×r | q^C (U^K_h)^T |
| q^R | 1×d_R | 当前头旋转后的位置 query |
| p | 1×T | 两项分数相加并缩放后，沿历史位置归一化的概率 |
| z | 1×r | pC，临时聚合结果 |
| U^V_h | r×d_v | 当前头 Value 投影权重 |
| o | 1×d_v | 当前头输出 |

上下两排以连线传递同一个 p；下排箭头表示依次计算，不将中间量 z 与输出 o 写成同一个连等式。
上排两次收缩分别沿 r、d_R，结果均为 1×T；下排沿 T 聚合、沿 r 投影。
C 和 C^T 共享颜色，宽高互换。各轴的物理边长（cm）：1→0.24，T→2.4，r→1.6，d_R→0.7，d_v→1.0；相同形状在所有位置保持一致。
图中 C 和 K^R 只画矩阵整体，不用单元格色阶暗示真实数值。

### 技术依据与范围

- DeepSeek-V2 §2.1：https://arxiv.org/html/2405.04434v5#S2.SS1
- DeepSeek-V3 `MLA.forward` 的 absorb 路径：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
- 数字配置：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json

访问日期：2026-09-21。图为原创代数展开；源码的权重存储方向与正文行向量权重记法互为转置。基础缩放使用 (d_c+d_R)^(-1/2)，长上下文额外缩放未画。解码仅含有效前缀；prefill/padding 需 mask。
正文缓存算例按同精度 BF16、B=1、T=32768 手算，不包含分页或量化元数据，不是性能实测。展开缓存指每头分别存完整 K/V 的布局，不能当作与其他 GQA 模型的对比。
已用独立随机小矩阵比较显式展开与 latent 路径的 logits、概率、输出，双精度误差在 1e-12 内；缓存字节数、图面形状和最终渲染均已检查。
