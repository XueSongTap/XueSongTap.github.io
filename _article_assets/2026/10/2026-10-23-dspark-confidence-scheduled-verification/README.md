# DSpark 博客配图

对应文章：[`2026-10-23-dspark-confidence-scheduled-verification.md`](../../../../_posts/2026/10/2026-10-23-dspark-confidence-scheduled-verification.md)。

三张配图由 Python / Matplotlib 原创绘制，白底，无署名、水印或外部位图素材。`source/dspark_figures.py` 是唯一可编辑绘图源；`render.py` 只是根据自身位置加载依赖和执行源文件的入口。

## 文件与用途

| 正文 PNG | 内容 | 尺寸 |
|---|---|---|
| [`dspark-prefix-budget.png`](../../../../img/2026/10/23/dspark-prefix-budget.png) | 固定额外预算下，按前缀存活概率分配各请求的验证长度 | 2736 × 1602 |
| [`dspark-async-budget.png`](../../../../img/2026/10/23/dspark-async-budget.png) | CPU 定总预算、GPU 按较新 confidence 重新分配的逻辑依赖 | 2736 × 1494 |
| [`dspark-concurrency-gain.png`](../../../../img/2026/10/23/dspark-concurrency-gain.png) | 官方估计吞吐中 dynamic 相对 fixed5 的增益随并发变化 | 2304 × 1404 |

- `exports/`：与正文 PNG 同步生成的 SVG 和 PDF。
- `source/`：绘图代码、依赖清单，以及[论文/源码核实记录与口径更正](source/research-notes.md)。
- `build/`：局部 Python 依赖、渲染检查与数据核算文件，由 `.gitignore` 排除。
- 不应在正文中添加这些制作文件的下载链接。

## 数据与解释边界

### 1. 固定预算前缀选择

这是自定数据、自定预算的教学示例，**不是性能测量，也不是搜索得到的最优预算**。

条件接受概率如下，第二个位置开始的概率均以前缀被接受为条件。前缀存活概率 `a[j] = product(c[:j+1])`，因此不需要额外假设各位置独立。

```text
A: c = [0.95, 0.90, 0.85, 0.80]
B: c = [0.90, 0.70, 0.50, 0.40]
C: c = [0.80, 0.40, 0.30, 0.20]

A: a = [0.95, 0.855, 0.72675, 0.5814]
B: a = [0.90, 0.63, 0.315, 0.126]
C: a = [0.80, 0.32, 0.096, 0.0192]
```

给定额外预算 `K=6`，全局最大的 6 个 `a` 对应 A 的前 3 个、B 的前 2 个、C 的前 1 个位置。每行仍是连续前缀；3 个请求的基础 target 槽另计。总预算 `B=3+6=9`，选中 `a` 之和为 `4.86175`，期望产出 `τ=3+4.86175=7.86175`。图中的基础槽和箭头表示预算拆分及前缀顺序，不是硬件线程或实际内存布局。

### 2. 异步预算信息路径

所有步号统一表示调度第 `N` 步：

1. CPU 根据第 `N−2` 步的历史 confidence 和硬件耗时表，确定第 `N` 步的总预算 `B_N`。
2. GPU 在执行第 `N` 步前，根据第 `N−1` 步的最新 confidence，在**不改变 `B_N`** 的条件下重分配各请求的 draft 长度。
3. 最新 confidence 在设备侧用于重分配，不需要先同步给 CPU。第 `N` 步主模型验证之后，采样 / draft 产生新 confidence，供后续调度步使用。

图表示逻辑依赖，不按实际 CPU/GPU 耗时比例绘制。图中不存在用第 `N` 步结果安排同一第 `N` 步的循环依赖。

机制来源为 DeepInfer 官方部署问答：[`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)，固定 commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。图为解释机制而重新组织的原创示意。

### 3. 并发与吞吐增益

数据来源：[DeepInfer 官方部署问答，L249–256](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L249-L256)，固定 commit 同上。

配置为 `EP64 / DP64`，横轴为 `seqs / rank`；纵轴为 dynamic 相对 fixed5 的吞吐增益百分比：

```text
seqs/rank: [16, 32, 48, 60, 72, 96]
gain (%):  [-1.5, 1.6, 3.6, 5.4, 6.7, 11.0]
```

这些增益由官方 benchmark 吞吐数据计算，属于官方估计吞吐结果的重绘，**非作者复测**。60 并发工作点只做轻标注；连线只连接所列数据点，不表示对其他并发的测量或拟合结论。

## 重建

需要 Python 3.12+、Matplotlib 3.11.2、Pillow 13.0.0，以及可用中文字体。脚本自动按顺序寻找 macOS 的 STHeiti / PingFang / Arial Unicode 字体。当前实际使用 `/System/Library/Fonts/STHeiti Medium.ttc`；SVG 使用路径保存字形，PDF 嵌入字体。

从本目录执行：

```sh
PYTHON=/Users/yxc/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PYTHON" -m pip install --target build/python_packages -r source/requirements.txt
"$PYTHON" render.py
```

也可以在已经具备依赖的 Python 环境中直接运行 `python3 render.py`。脚本按自身路径解析文件，不依赖调用时的工作目录，会同步重建正文 PNG、SVG、PDF 及 `build/figure-manifest.json`。

## 检查

已逐张用图像查看工具目检正文 PNG 和 PDF 栅格预览，检查文字、公式、箭头、数据标注、遮挡与裁切；SVG 同步导出字形路径，并核对宽高与 viewBox。第一张的数值由代码断言核算。尺寸与计算结果记录于 `build/figure-manifest.json`。没有修改任何既有文章或配图。
