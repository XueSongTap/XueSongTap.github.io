# Fast-WAM 文章附属文件

对应文章：[Fast-WAM：Wan 预训练怎样变成动作策略](../../../../_posts/2026/09/2026-09-30-fast-wam-wan-action-architecture.md)。正文图片：[attention mask](../../../../img/2026/09/30/fast-wam-attention-mask.png)、[论文架构图](../../../../img/2026/09/30/fast-wam-paper-architecture.png)、[论文三种范式对比](../../../../img/2026/09/30/fast-wam-paper-paradigms.png)。

`source/fast-wam-attention-mask.tex` 是唯一可编辑图源，重画论文 Figure 2(b) 的 Boolean 可见性关系，并标明推理时的缓存使用。图中的 token 数只作轴结构示意，不表示真实 token 数、物理存储、线程布局或 attention 权重。

`source/fast-wam-paper-architecture.tex` 和 `source/fast-wam-paper-paradigms.tex` 是两张原图的唯一裁剪配置，直接从 `source/inputs/fast-wam-v2.pdf` 引入原始 PDF 内容，保留原图文字、箭头和图例。没有重新绘制或添加图内标注，正文另写中文解读。PDF 和 SVG 导出保留原始矢量内容，正文使用 320 DPI 白底 PNG。

输入 PDF 来源：https://arxiv.org/pdf/2603.16666v2，2026-03-23，作者 Tianyuan Yuan、Zibin Dong、Yicheng Liu、Hang Zhao，CC BY 4.0。原件作为重建所需输入保存在附属目录，不依赖本次会话的临时路径。裁剪坐标单位为 PDF point，以原页面左上角为原点：Figure 2(a) 位于 PDF 第 5 页，范围 `(132,68,374,264)`；Figure 1 位于 PDF 第 2 页，范围 `(112,72,502,292)`。LaTeX `trim` 参数依次为左、下、右、上。

## 修改与重建

依赖：Python 3 标准库、TeX Live（XeLaTeX、ctex、Fandol 字体）、Poppler（pdftocairo）。在任意工作目录执行：

```bash
python3 /path/to/repo/_article_assets/2026/09/2026-09-30-fast-wam-wan-action-architecture/render.py
```

脚本按自身位置定位仓库，将白底 PNG 更新到正文图片目录，将 PDF、SVG、透明 PNG 更新到 `exports/`。编译日志和目检用 PNG 位于 Git 忽略的 `build/`。

## 来源与核对记录

- 论文：[arXiv:2603.16666v2](https://arxiv.org/html/2603.16666v2)，2026-03-23，§3.2、§4.1–4.3、Figures 1–2、Figure 4、Tables 1–2。mask 为自行重画；架构图与范式对比图为原图裁剪，正文标明出处和 CC BY 4.0 许可。
- 官方代码：[FastWAM](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)，核对 commit `7faa71108368fbb3b6885649f112af607427a2d4`，访问日期 2026-09-30。
- mask：`src/fastwam/models/wan22/fastwam.py::_build_mot_attention_mask`、`wan_video_dit.py::build_video_to_video_mask`。
- 初始化：`scripts/preprocess_action_dit_backbone.py`、`action_dit.py::ACTION_BACKBONE_SKIP_PREFIXES`。默认从 Wan 插值 action backbone，保留随机 action_encoder/head；proprio_encoder 另行随机初始化。
- shared attention / cache：`mot.py::_forward_joint_layer`、`prefill_video_cache_tensor`、`forward_action_with_video_cache_tensor`。
- 训练：`trainer.py::_apply_dit_only_train_mode`，video/action DiT 和 proprio encoder 可训练，VAE/T5 冻结；公开 task YAML 已使用 epoch 配置，不能直接当作论文的 20k/30k steps。
- 数据与权重：`configs/data/libero_2cam.yaml` 合并四个 suite；RoboTwin 为独立配置。HF 模型库查询 revision `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`，提供两个 benchmark 的独立 checkpoint 和各自 stats，另提供 LIBERO Optional IDM。

### 图的语义与几何

语义：`M` 为 Boolean 可见性；`B_M` 为加到 attention logits 的 0 / 负无穷 bias。行=Query，列=Key；颜色区分 video/action 的 Query，白格始终表示禁止。

形状：训练 `(N_0+N_f+H)^2`，推理 `(N_0+H)^2`。示意取 `N_0=2,N_f=4,H=2`，统一每个 token 的边长 0.6 cm；训练方阵 4.8 cm，推理方阵 2.4 cm，保留分块边长一致。允许块为 `CC, FC, FF, AC, AA`，删除未来组后为 `CC, AC, AA`。

正文中 98/120 个当前帧 token、294/360 个训练 video token 是按公开分辨率、VAE 压缩率与 patch size 的手算，不是运行测量。缓存只在同一个 action chunk 的去噪循环中复用；新观测需重新编码。
