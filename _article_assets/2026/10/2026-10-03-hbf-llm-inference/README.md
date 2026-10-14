# HBF inference article figures

Independent analytical illustration for the HBF inference article; not video data or a measured benchmark. White background, English labels, no signature.

Article: [高带宽闪存 HBF：更大的容量，能否让推理更便宜？](../../../../_posts/2026/10/2026-10-03-hbf-llm-inference.md).

The article also uses two selected video frames in `../../../../img/2026/10/03/`: `hbf-package-die-stack.jpg` (08:04) and `hbf-kv-offload-paths.jpg` (16:20), from Asianometry, [High Bandwidth Flash Is Coming](https://www.youtube.com/watch?v=3nTpW52nioI), 2026-09-24. They are cropped excerpts for discussion, rather than original drawings. The raw video, subtitles and selection records remain in the original lecture-note directory in own_log; they are not duplicated in the blog. The KV-offload image corresponds to Figure 3 in [HBF Sucks?](https://arxiv.org/html/2608.11668v4).

## Editable sources

- `source/parameters.json`: the sole editable parameter source (expert count, top-k, annotation batches, styling and output path).
- `source/render_chart.py`: the sole editable plotting implementation.
- `render.py`: path-aware entry point. The asset root sits at `_article_assets/YYYY/MM/article-slug/`; its `parents[3]` resolves the repository or staging root. No absolute staging path is embedded.
- `source/bandwidth-budget.json`: editable values and styling for the independent-stack bandwidth illustration.
- `source/render_bandwidth_budget.py`: the sole editable implementation for that illustration.
- `render_bandwidth.py`: path-aware entry point for the new bandwidth figure.

## Derivation and limitations

There are N=256 equal-size experts. Each token selects exactly k=8 distinct experts as a uniformly random k-subset; selections are independent across tokens. A given expert is missed by one token with probability 1-k/N, and by B tokens with probability (1-k/N)^B. Linearity of expectation gives E[U_B]/N = 1-(1-k/N)^B.

Equal expert sizes let expert-count coverage serve as a proxy for parameter-byte coverage. No claim is made about real router balance, correlated tokens, throughput, cache misses, HBF bandwidth, or actual bytes read. The curve denotes expected distinct experts across a batch, not simultaneous expert activation by one token. Expected expert counts can be fractional. Batch expansion does not alone prove that HBF is unsuitable.

Highlighted values:

| B | Expected expert share |
|---|---|
| 1 | 3.1250% |
| 8 | 22.4300% (approximately) |
| 32 | 63.7945% (approximately) |
| 128 | 98.2817% (approximately) |

## Rebuild

Standard dependency installation: `python3 -m pip install matplotlib`. Bundled Python may instead use an isolated dependency directory:

```sh
python3 -m pip install --target .render-deps matplotlib
python3 render.py
python3 render_bandwidth.py
```

Run these commands from this asset directory. `.render-deps` is ignored, is not a formal source asset, and must not be copied into the blog repository. The same script also works with matplotlib installed in the active Python environment.

## Outputs

- `exports/hbf-moe-working-set.svg`: vector export.
- `exports/hbf-moe-working-set.pdf`: vector export.
- `../../../../img/2026/10/03/hbf-moe-working-set.png`: final blog bitmap, regenerated from the sources above (2160 × 1350 at default settings).
- `exports/hbf-bandwidth-budget.svg` and `.pdf`: vector exports of the independent-stack and buffer illustration.
- `../../../../img/2026/10/03/hbf-bandwidth-budget.png`: final white-background bitmap (2520 × 1800).

## Independent-stack bandwidth illustration (2026-10-06)

Panel A uses an eight-way equal split of an 8 TB/s GPU memory-bandwidth budget. This is a conceptual data-flow illustration, not a claim about a specific physical package layout or a per-stack product rating. The article's B200/B300 nominal GPU and eight-GPU figures follow [NVIDIA HGX Enterprise Reference Architecture, Table 1](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html). Specific SKUs differ: the [Blackwell Ultra Datasheet, p. 5](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf) lists a 270 GB, 7.7 TB/s HGX B300 configuration, which is kept separate in the prose.

Panel B illustrates an optional fast-buffer path. Cache hits can reuse resident weights; HBF cache misses or buffer refills remain limited by the source. It does not imply that all HBF accesses require HBM staging. Panel C sums four independently connected HBF stacks. Each uses 512 GiB and a 3.072 TB/s effective interface target from [OCP HBF v0.7.0, Tables 2/3](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf); Section 4.3 and Figure 2 explicitly show a four-chipset host organization. The 2 TiB and 12.288 TB/s sum is arithmetic, not a shipping GPU specification. Host receive bandwidth, internal paths, balanced placement and thermal constraints must support the sum.

The figure's arrows represent data delivery. Increasing independent stacks/interfaces can raise aggregate bandwidth; adding NAND die layers inside one stack does not automatically raise its external bandwidth. The argument is about capacity and interface budgets, not an assertion that a high-capacity HBF stack cannot be placed beside another stack.

## Verification

Both final PNGs were visually inspected for label readability, annotation collisions, chart boundaries, formulas, assumptions and arrow direction. Mathematical values and capacities were also checked. The updated article separates modeled bandwidth, official product figures and arithmetic sums, and corrects the claim that batch size alone makes HBF less suitable than HBM.
