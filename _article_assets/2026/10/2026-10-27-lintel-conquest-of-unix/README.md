# Lintel 征服 UNIX 文章配图与来源

对应文章：[_posts/2026/10/2026-10-27-lintel-conquest-of-unix.md](../../../../_posts/2026/10/2026-10-27-lintel-conquest-of-unix.md)。

正式图片：[img/2026/10/27/](../../../../img/2026/10/27/)，本篇使用文件名以 `lintel-` 开头的 36 张图。资料核对日期为 2026-10-27。

## 文件与重建

- `render.py`：全部 36 张图的唯一重建入口。它把 `source/frames/` 里的 1080p 整帧按 `CROPS` 表裁切后导出，并用 matplotlib 重绘 6 张图。
- `source/frames/`：从 1080p 视频源导出的整帧（1920×1080），未做裁切。裁切与导出统一由 `render.py` 的 `CROPS` 表控制。
- `source/cover.jpg`：视频封面（1280×720，YouTube maxres），同时导出为 `lintel-cover.jpg`。
- `source/notes.tex`：同一内容的 36 页 LaTeX 讲义，是本篇的写作底稿，包含图注与时间区间标注。
- `source/ksx5g5WvAIw.en.srt`、`source/transcript.txt`：人工英文字幕与带时间码的纯文本，用来核对图注中的时间区间。
- `exports/`：可复用导出（目前为空，需要时放 SVG）。
- `build/`：中间文件，由本目录 `.gitignore` 排除。

脚本根据自身位置解析博客仓库，可从任意工作目录运行：

```bash
uv run --isolated --no-project --with matplotlib --with pillow python \
  _article_assets/2026/10/2026-10-27-lintel-conquest-of-unix/render.py
```

macOS 使用 PingFang SC，Linux 使用 Noto Sans CJK SC；字体候选在 `render.py` 顶部。

## 图源

视频画面均来自 Asianometry《The Lintel Conquest of UNIX》（2026-10-25），仅作裁切与格式转换；图注末尾标注了对应画面在视频中的时间区间。视频中的截图本身多为当时的报纸、广告、新闻稿或网页存档。

| 正文图 | 文件 | 来源与处理 |
| --- | --- | --- |
| 封面 | `lintel-cover.jpg` | YouTube 官方封面 |
| 1 | `lintel-timeline.png` | 按视频正文的时间点重绘 |
| 2 | `lintel-hp-parisc-hpux.jpg` | 视频画面 00:02:40–00:02:45，惠普 PA-RISC 与 HP-UX 工作站 |
| 3 | `lintel-flywheel.png` | 按视频对高销量循环的叙述重绘 |
| 4 | `lintel-nt-sales-1993-1994.jpg` | 视频画面 00:04:44–00:04:51，IDC 统计 |
| 5 | `lintel-nt-sales-1993-1996.jpg` | 视频画面 00:07:04–00:07:11，IDC 统计 |
| 6 | `lintel-specint95-pentium-pro.jpg` | 视频画面 00:05:25–00:05:31，原图源 *Microprocessor Report* 1995-11-13 |
| 7 | `lintel-intel-p6-servers-infoworld.jpg` | 视频画面 00:06:09–00:06:16，原图源 InfoWorld 1995-05 |
| 8 | `lintel-intel-nsp-infoworld.jpg` | 视频画面 00:07:17–00:07:23，原图源 InfoWorld（NSP 与 WinHEC） |
| 9 | `lintel-nec-nt-mips-ad.jpg` | 视频画面 00:08:40–00:08:46，原图源 InfoWorld 1993-11-15 |
| 10 | `lintel-pentium-pro-16bit-headline.jpg` | 视频画面 00:10:36–00:10:41，原图源 *Daily Record* 1995-11 |
| 11 | `lintel-smp-support.png` | 按视频正文中各系统的 SMP 上限重绘 |
| 12 | `lintel-symbolics-lmi.jpg` | 视频画面 00:12:27–00:12:33 |
| 13 | `lintel-kernel-diagram.jpg` | 视频画面 00:13:33–00:13:38 |
| 14 | `lintel-torvalds-gpl-cio.jpg` | 视频画面 00:15:13–00:15:19，CIO 采访页面 |
| 15 | `lintel-redhat-box.jpg` | 视频画面 00:18:31–00:18:36，Red Hat 6.1 包装盒 |
| 16 | `lintel-price-ladder.png` | 按视频正文的三档价格重绘（对数刻度） |
| 17 | `lintel-linux-philadelphia-inquirer.jpg` | 视频画面 00:17:50–00:17:54，原图源 *Philadelphia Inquirer* 1996-05 |
| 18 | `lintel-oracle-linux-cnet.jpg` | 视频画面 00:25:05–00:25:09，原图源 CNET 1999-07，页面本身不是 1998 年的那篇报道 |
| 19 | `lintel-future-of-linux-panel.jpg` | 视频画面 00:25:16–00:25:21，原图源 *Linux Journal* #54 |
| 20 | `lintel-intel-redhat-press-release.jpg` | 视频画面 00:26:23–00:26:28，Intel 新闻稿 1998-09-29 |
| 21 | `lintel-dell-redhat-press-release.jpg` | 视频画面 00:27:08–00:27:12，Red Hat 新闻稿 1999-02-04 |
| 22 | `lintel-halloween-documents.jpg` | 视频画面 00:28:29–00:28:33，catb.org 存档 |
| 23 | `lintel-linux-myths-webpage.jpg` | 视频画面 00:28:52–00:29:10，微软 "Linux Myths" 页面存档 |
| 24 | `lintel-mindcraft-benchmark.jpg` | 视频画面 00:29:10–00:29:15，Mindcraft 报告 |
| 25 | `lintel-xeon-8way-profusion.jpg` | 视频画面 00:30:57–00:31:02，原图源 *Network World* 1999-03-22 |
| 26 | `lintel-amazon-linux-2001.jpg` | 视频画面 00:32:46–00:32:58，路透社稿 |
| 27 | `lintel-cost-per-user.png` | 按 E*Trade 案例的三组数字重绘 |
| 28 | `lintel-etrade-sun-e4500.jpg` | 视频画面 00:33:09–00:33:16 |
| 29 | `lintel-server-os-share-1995-2000.jpg` | 视频画面 00:32:21–00:32:27，IDC 数据 |
| 30 | `lintel-linux-2.4-announcement.jpg` | 视频画面 00:34:26–00:34:32，LKML 原邮件 |
| 31 | `lintel-linux-2.6-register.jpg` | 视频画面 00:38:53–00:38:58，The Register |
| 32 | `lintel-amd-opteron.jpg` | 视频画面 00:37:38–00:37:43 |
| 33 | `lintel-address-space.png` | 按正文的寻址上限重绘 |
| 34 | `lintel-active-directory.jpg` | 视频画面 00:40:04–00:40:08 |
| 35 | `lintel-unix-install-base-gartner.jpg` | 视频画面 00:41:41–00:41:46，Gartner 经 Network World 报道 |

重绘的 6 张图（时间线、飞轮、SMP 上限、价格阶梯、每用户成本、地址空间）属于本文解释材料，数值全部取自视频正文；其中 SMP 与价格两项在原视频里是文字叙述，图中标注的是叙述里的区间而非精确测量。

## 内容核对说明

- 视频把 NT 4.0 的 SMP 上限表述为 4–8 路、部分 UNIX 变体到 64 路，正文沿用这个区间，不写成单一数字。
- E*Trade 的每用户成本按表中硬件成本与用户数相除得到（250000/350、4000/180、8000/360），与原视频的口径一致；正文写"约 32 倍"是按 714/22 计算。
- 视频中的 "Microsoft went ballistic"、"buy in, get locked in" 等原话保留英文并附中文解释，不替换为转述。
- 截图中出现的第三方报纸、广告和网页版权归原作者；本文仅在评论与说明的范围内引用，未做二次修改。
