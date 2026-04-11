---
layout: article
title: VSCode 无法读取代理环境变量的解决方案
tags: tools
---

在 macOS 上开发时，常见的做法是把代理配置写在 `~/.zshrc` 里：

```bash
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
```

在终端里运行 `curl` 或其他工具时一切正常，但在 VSCode 的插件中，代理失效，比如codex cli/claude code 这种 影响很大，




## 根本原因

`~/.zshrc` 只在**交互式 shell 启动时**才会被 source。

当你从 Dock 或 Finder 点击打开 VSCode 时，macOS 启动的是一个**非交互式进程**，不会走 `~/.zshrc` 的加载流程，所以里面定义的环境变量对 VSCode 进程及其子进程完全不可见。



## 解决方案：改用 `~/.zshenv`

zsh 有另一个配置文件 `~/.zshenv`，它对**所有** zsh 进程都生效，无论是否交互式、是否登录 shell。

把代理配置迁移到 `~/.zshenv`：

```bash
# ~/.zshenv
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
```

修改完成后，用 `Cmd+Q` **完全退出** VSCode（仅关闭窗口不够），再重新打开，代理配置即可生效。



## zsh 配置文件加载顺序小结

| 文件 | 触发时机 |
|||
| `~/.zshenv` | 所有 zsh 进程（含非交互式） |
| `~/.zprofile` | 登录 shell |
| `~/.zshrc` | 交互式 shell |
| `~/.zlogin` | 登录 shell（在 zshrc 之后） |

对于需要在所有场景下生效的环境变量（如代理、`PATH` 补充），`~/.zshenv` 是最合适的位置。



## 注意事项

`~/.zshenv` 加载时机非常早，且对所有子进程都有影响。建议只把**无副作用的环境变量**放在这里，避免放入会触发网络请求、依赖其他工具的初始化逻辑。



## 参考

- [zsh 官方文档：Startup/Shutdown Files](https://zsh.sourceforge.io/Doc/Release/Files.html)
- [VSCode 文档：Environment Variables in Integrated Terminal](https://code.visualstudio.com/docs/terminal/profiles#_terminal-profiles)
- [VSCode 文档：Launching from the command line](https://code.visualstudio.com/docs/setup/mac#_launching-from-the-command-line)
