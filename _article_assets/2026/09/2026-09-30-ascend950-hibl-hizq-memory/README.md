# 昇腾 950 显存方案配图

对应文章：[昇腾 950 的两种显存方案](../../../../_posts/2026/09/2026-09-30-ascend950-hibl-hizq-memory.md)。

正文图片：[ascend950-memory-options.png](../../../../img/2026/09/30/ascend950-memory-options.png)。

- `source/memory-options.json`：图中文字、模块数量与规格的唯一可编辑数据源。
- `render.py`：Pillow 绘图与重建脚本，布局只在此处维护。
- `exports/ascend950-memory-options.png`：与正文同步的白底导出图。
- `build/`：检查图片或中间文件，Git 忽略。

依赖：Python 3、Pillow，以及 macOS 的 Hiragino Sans GB / STHeiti，或 Linux 的 Noto Sans CJK。字体路径可在脚本中配置。

重建命令（仓库根目录）：

```bash
python3 _article_assets/2026/09/2026-09-30-ascend950-hibl-hizq-memory/render.py
```

脚本按自身位置解析路径，可从其他工作目录运行。修改 JSON 数据或脚本布局后执行重建，再目检正文 PNG。

来源：

- 华为，[《昇腾 950 NPU 架构白皮书》](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)，第 3 章及表 3-1。核对日期 2026-09-30，下载文件 SHA-256：`ece3405e6a17fabdd462338fb94266558649a6407a2f28008403211387b3a927`。
- 华为，[2025 年全联接大会演讲稿](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)，两种显存的命名与场景对应。

图为原创抽象对比，不复刻白皮书封装布局。方块仅表示模块数量，双向箭头表示访存关系。整芯片容量、带宽是公开规格上限；每模块数据是基于同规格模块、带宽可均分假设的手算结果，不是测量，也不确认层数、接口位宽或物理拓扑。外部“白鹭／朱雀”代号未放入图中，以免与官方产品名混淆。
