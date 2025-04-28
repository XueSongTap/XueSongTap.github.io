---
layout: articles
title: spdlog 和 GLOG 的核心区别
tags: log
---


# 总结：spdlog 和 GLOG 的核心区别

| 维度 | GLOG (Google Logging) | spdlog |
|:---|:---|:---|
| 项目起源 | Google 内部老项目（2000年代初） | 轻量、高性能现代日志库（C++11以后） |
| 库大小 | 很重、依赖复杂（GFLAGS、线程库） | 单头文件版也有，非常轻量 |
| 初始化 | 必须全局初始化（InitGoogleLogging） | 直接即用，支持单独 logger、多 logger |
| 依赖 | gflags、pthread、甚至 glogd/glibc 特定版本 | 只依赖 C++11 标准库 (`<thread>`, `<mutex>`) |
| 性能 | 不算快（特别是大量日志场景） | 极快，百万级 QPS 轻松支持 |
| 灵活性 | 比较死板，所有日志走统一配置 | 支持每个模块/子系统有独立 logger，灵活配置格式、输出位置 |
| 格式化 | 只能 `<<` 流式拼接（像 `cout`） | 支持 `{}` 样式格式化（`fmtlib`），又快又安全 |
| 多线程 | 支持，但锁比较重 | 原生多线程优化，甚至有 async 模式（异步打日志） |
| 日志输出 | 只能写本地文件，旋转不灵活 | 文件、控制台、多文件输出、自定义sink，支持滚动、时间切割 |
| 日志等级 | 固定（INFO、WARNING、ERROR、FATAL） | 等级灵活，还可以自定义 logger |
| 崩溃日志 | 有 built-in 崩溃 dump（Fatal 日志后 core dump） | 需要自己加 hook，但也支持 |
| 社区维护 | 维护较少，功能很久没更新 | 活跃开发，持续支持新特性（比如 `fmt::compile`, `async_sink`） |

---

# 为什么很多项目从 GLOG 迁移到 spdlog？

总结就是：

✅ **spdlog 更快**：特别是大规模高并发写日志场景（比如服务端、存储系统、分布式系统）；

✅ **spdlog 更轻量**：GLOG 要拉很多库，编译很慢，出错概率高；spdlog 一个头文件就能跑；

✅ **spdlog 更好用**：`{}` 格式化远比 `<<` 写代码舒服，少出 bug，代码更美观、清晰；

✅ **spdlog 更灵活**：想要 file+console、多 logger、不同模块独立打日志？glog 很难，spdlog很简单。

✅ **spdlog 社区活跃**：一直在适配 C++17、C++20、C++23的新特性，而 GLOG 早停滞了。

✅ **spdlog 支持异步日志**（异步 sink），可以大幅降低日志打点对主线程性能的影响；GLOG 必须同步锁。

✅ **更好的可控性**：spdlog 支持自定义 format，比如加上 timestamp、thread id、source file name，非常容易。

---

# Kvrocks 为什么要换？

🔵 Kvrocks 是高性能、分布式 KV 系统：

- 对日志性能很敏感（特别是 replication, checkpoint 时大量同步数据）
- 对日志格式灵活性要求高（调试同步错误，定位问题需要精确控制日志内容）
- 需要轻量依赖，减少编译、部署、交付复杂度

所以，**为了性能、简洁、现代化开发体验，换成 spdlog 是自然选择**。

---

# 小总结一句话：

> **GLOG 是老派重型卡车，spdlog 是现代高铁：更快、更轻、更顺滑。**

---

要不要我顺便也给你补一张更直观的图？比如「spdlog 核心组件」对比「GLOG 组件」？  
可以一秒钟看清 spdlog 是怎么比 glog 架构简洁的，要的话告诉我！🚀