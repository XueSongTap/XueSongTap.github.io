# ReGLU 双分支张量图

对应文章：[Gated Activation 与 ReGLU](../../../../_posts/2025/11/2025-11-10-gating-activation.md)，第 3 节。
正文图片：[白底 PNG](../../../../img/2025/11/10/reglu-flow.png)。

## 文件与重建

- `source/reglu-flow.tex`：唯一可编辑绘图源，TikZ + 中文 LaTeX。
- `exports/reglu-flow.pdf`、`exports/reglu-flow.svg`：矢量导出。
- `exports/reglu-flow-transparent.png`：透明底导出。
- `build/`：编译日志、中间产物及检查图，由 `.gitignore` 排除。
- `render.py`：由脚本位置定位仓库，更新所有导出及正文 PNG。

依赖：Python 3、TeX Live（XeLaTeX、standalone、ctex/Fandol、TikZ、amsmath、amssymb）、Poppler（pdftocairo）。
从仓库根目录运行：

```sh
python3 _article_assets/2025/11/2025-11-10-gating-activation/render.py
```

修改源文件后重新运行并目检 PNG，尤其检查两条跨排连接线、阶段标题、张量形状和零元素的对应关系。

## 形状与语义账本

采用行向量右乘约定，省略偏置，不展平 batch 或 sequence。无设备分片，全局形状就是图中形状。

| 对象 | 形状 | 语义与生成方式 |
|---|---|---|
| X | B × T × d | 实值输入；上排两次展示的是同一输入 |
| W₁、V | d × f | 两份独立的实值可训练权重，均沿 d 收缩 |
| A | B × T × f | ReLU(XW₁)，非负实值，白格表示示意零元素 |
| G | B × T × f | XV，实值线性投影，不是概率或布尔掩码 |
| H | B × T × f | A ⊙ G，对应元素相乘，无收缩或广播 |
| W₂ | f × d | 实值输出权重，沿 f 收缩 |
| Y | B × T × d | HW₂，保留 B、T 两轴 |

图不指定 FP32/BF16 等实现 dtype。B 为 batch size，T 为序列长度，d 为模型维度，f 为中间维度；均为正整数。

几何账本：T=1.2 cm、d=1.6 cm、f=2.4 cm；每个矩阵正面高对应倒数第二轴、宽对应末轴。带 B 的对象统一使用 0.14/0.28 cm 偏移的轮廓叠片。等形状对象使用同一绘图宏，W₂ 的 f 高度等于 A/G/H 的 f 宽度。格数仅用于结构示意，不给出实际配置或 f/d 的数值约束。

颜色按角色区分，深浅仅作示意；重复的 X、A、G 复用同一纹理种子。H 与 A 的已知零位置严格一致。连接箭头表示同一张量接续到下排，不表示跨卡通信或额外复制。所有计算均是逻辑张量运算，不对应线程布局、内存布局或性能测量。

布局采用独立公式区、上排两路投影、两条跨排接续通道、下排乘积与输出投影、底部三行说明。基准留白约 1 em，连接线在背景层，标签在前景层。

## 来源与校验

- Noam Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)，第 2 节，式 (5)、(6)。原论文用 ⊗ 表示逐元素乘，图中改用 ⊙ 避免和张量积混淆。
- 文章中的 W₁、V、W₂ 符号沿用不变；新增 A、G、H 仅为图解中间态。
- 已独立核对两次 d 收缩、逐元素乘的形状一致性、最后一次 f 收缩，以及 B、T 维度保持不变。
