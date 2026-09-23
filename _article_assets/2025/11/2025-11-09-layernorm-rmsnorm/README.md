# LayerNorm / RMSNorm：token 行归约与广播

对应文章：[2025-11-09-layernorm-rmsnorm.md](../../../../_posts/2025/11/2025-11-09-layernorm-rmsnorm.md)。
正文白底图：[norm-token-rows.png](../../../../img/2025/11/09/norm-token-rows.png)。

- `source/norm-token-rows.tex`：唯一可编辑 TikZ 源文件，中文采用 ctex 的 Fandol 字体。
- `exports/`：PDF、SVG、白底 PNG 与透明背景 PNG。
- `build/`：XeLaTeX 中间产物与日志，Git 忽略。
- `render.py`：从自身路径定位仓库，更新所有导出和正文图片。

依赖 Python 3、TeX Live（XeLaTeX、ctex、TikZ、standalone）、Poppler（pdftocairo）。从任意目录执行：

```sh
python3 /path/to/repo/_article_assets/2025/11/2025-11-09-layernorm-rmsnorm/render.py
```

## 形状与语义

输入与输出是实数张量，逻辑形状为 B×T×D。作图显式将前两轴展开为 N=BT：每行对应一个 (b,t)，列对应特征 d。X、Y 的面宽高始终为 6×4 个示意单元；μ、v、q 为 N×1，面宽高为 1×4 个相同单元。每个单元轴长 0.32 cm。格子是轴结构示意，不表示具体数值、真实维数或硬件布局。

LayerNorm：μ=mean_D(X)，v=mean_D((X−μ)²)，Y=(X−μ)/sqrt(v+ε)·γ+β。
RMSNorm：q=mean_D(X²)，Y=X/sqrt(q+ε)·γ。
统计量沿 D 广播；γ、β 的形状为 D，沿 N 共享。输出将 N 恢复为 B×T。
实线是计算依赖或输入复用，虚线是统计量广播。图不规定内核数、物理复制或分布式通信次数。

## 来源与复核

依据以下官方定义绘制原创计算示意图：

- https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

已核对归约轴、方差定义、ε 位置、参数广播与等形状几何一致性；渲染 PNG 后目检公式、连接线、文字与裁切。修改源文件后应重新运行脚本并再次目检。
