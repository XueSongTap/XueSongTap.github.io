# DSpark 文章核实记录

核实日期：2026-10-23（Asia/Shanghai）。本文档保存写作依据与边界，不作为正文展示。

对应文章：`../../../../../_posts/2026/10/2026-10-23-dspark-confidence-scheduled-verification.md`。

## 固定来源

- DSpark 论文：<https://arxiv.org/html/2607.05147v1>。核实时版本记录只有 v1，提交日期 2026-07-06。
- DFlash：<https://arxiv.org/abs/2602.06036v2>，2026-05-28 修订。
- DeepInfer：固定 commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。
- 原始笔记来自 Zotero：DSpark `ACGMU3EU`（item 548，note 569）；DeepSeek-V4.1-Flash `9HVXCWIV`（item 510，note 570 等）。论文方法、因果条件和实验通过原文取证核对，源码结论通过固定 commit 核对。

## 概率与调度

论文 §3.1–3.2：

- 单个已抽中候选的接受概率为 `min(1, p(x)/q(x))`。
- Confidence 的监督标签是对本位置草稿采样求平均的接受概率 `1-TV(p,q)`；不能把这两个数直接等同。
- Head 输入含当前位置 hidden state 和前一个 token 的 embedding，不含当前位置候选本身。
- 前缀贡献 `a[r,j] = product(c[r,1:j])`，是调度器的估计值。
- `B = sum_r(1 + l[r])` 为主模型位置预算；`tau = sum_r(1 + sum_j a[r,j])` 为全 batch 完整步产出，含每请求一个 correction/bonus。
- 固定 B 的全局 top-K 在前缀概率单调、不破坏同请求位置顺序的条件下保留前缀；跨 B 的首次下降早停需要单峰条件才有全局最优保证。
- Appendix A 的 A/B 例子：目标 P(A)=0.7、草稿 Q(A)=0.5，仅在草稿为 A 时验证会把输出 P(A) 改成 0.85。
- 后续 confidence 会读取前一个候选，因此不能随意用整个当前块的后续信息回头决定前面候选是否验证。

## 统一时间步编号

按即将调度第 N 步编号：

- CPU 用 N−2 步 confidence 选总预算 B_N。
- Non-ZOS 可在 CPU 用 N−1 步信息分配长度。
- ZOS 下 GPU 在第 N 步 forward 前用 N−1 步设备侧 confidence 修正长度，总预算不变。
- 不写成“CPU N−2，CPU N−1，GPU N”：这是混用了 schedule(N) 与 schedule(N+1) 两套编号。

证据：

- [CPU 预算与分配](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L160-L272)
- [GPU 调用](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L191-L245)
- [GPU 累乘与排名](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py#L26-L81)

## 执行成本

- CPU 扫描预算，对每个 B 向上取到已捕获的 graph 档位，估计 `baseline_ms + slope_ms * graph_size + sampler_ms`，比较期望 token / 时间。
- `sampler_ms` 的计时包含验证采样和下一块 drafter，不只是 softmax。
- CUDA 路径按全局排名做 top-K，但采用并行 rank counting；不要写成调用全局排序 kernel。
- 公开代码有 confidence temperature 配置入口，未找到论文 STS 的生产校准产物，不能由默认温度 1.0 推出必然过度自信。

## 性能数据的三个口径

1. 论文离线实验主要比较 drafter 质量，不把接受长度提升直接当吞吐提升。
2. 论文线上实验为 DeepSeek-V4 Flash/Pro preview，完整 DSpark-5 对 MTP-1。Flash 80 tok/s/user SLA 下 +51% 为完整方案收益，不是单独裁剪收益。
3. V4.1 Q&A 固定 5 与变长验证：表头为 estimated throughput，由 EP64/DP64 benchmark 数据计算。原表如下，未在本机复测。

| 并发/rank | 固定 draft 验证数 | 变长 draft 验证数 | 固定 accepted length | 变长 accepted length | 固定估计吞吐 | 变长估计吞吐 | 增益 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|16|5|4.44|3.32|3.24|4891|4816|−1.5%|
|32|5|3.65|3.25|3.17|8964|9111|+1.6%|
|48|5|3.94|3.30|3.26|12032|12467|+3.6%|
|60|5|3.58|3.30|3.21|13730|14466|+5.4%|
|72|5|3.24|3.29|3.12|14962|15961|+6.7%|
|96|5|3.24|3.30|3.15|16817|18660|+11.0%|

吞吐单位：tok/s/rank。来源：[Q&A L249–256](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L249-L256)。

Q&A 未说明 accepted length 是否包含额外 target token；公开日志有不含 target 的 `accepted_tokens`，完整产出另有每请求的基础项。因此正文只沿用原表描述，不直接把 3.30 当作理论 tau，也不由它反推步耗时。

## 原创教学推导

- 前缀图使用自设概率 A=[.95,.90,.85,.80]、B=[.90,.70,.50,.40]、C=[.80,.40,.30,.20]，不是实验数据。
- K=6 时分配 A3/B2/C1，选中前缀概率和为 4.86175；基础产出 3，tau=7.86175；主模型位置 B=9。
- 边际式 `(tau+a)/(t+dt) > tau/t` 等价于 `a > (tau/t)*dt`，示例 tau=8、t=2 ms、dt=.1 ms，阈值为 .4。该算例不是硬件测量。

## 验证记录

- 数学、因果条件、论文实验范围与源码说明分别独立复核通过。
- 检查正文 front matter、公式分隔符、代码围栏、两处 post_url 和三张图片引用。
- 三张原创图经渲染目检，编辑源和 SVG/PDF 导出随文保留。
- 当前机器 `bundle check` 提示缺少 Gemfile 依赖，因此未完成 Jekyll 整站构建；未为本文改动站点工具链。
