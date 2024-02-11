---
layout: articles
title: kallsyms 调试工具，找到内核调用函数及模块
tags: cuda tensor gpu nvidia
---

## 问题
Linux 内核：能否在编译或者运行环境中找出 a 模块调用的某个函数来源于另外哪个模块吗？

## 解决方案：kallsyms


kallsyms 是 Linux 内核中一个重要的调试工具，它允许用户在内核运行时查看和解析内核符号。

kallsyms 的主要功能是提供内核符号的地址和名称，这些符号包括函数、变量、结构体、枚举等。用户可以通过使用 kallsyms 来查找内核中特定符号的地址，或者通过解析内核符号的地址来确定符号的名称。

### kallsyms 的使用


只需要在内核命令行中添加 `kallsyms=1` 即可。添加 `kallsyms=1` 后，内核将在启动时生成一个符号表，并将符号表存储在 `/proc/kallsyms` 文件中。

用户可以通过使用 `cat` 命令来查看 `/proc/kallsyms` 文件的内容，如下所示：

```
$ cat /proc/kallsyms
0000000000000000 T __crc32c_8bytes
0000000000000000 T __crc32c_16bytes
0000000000000000 T __crc32c_32bytes
0000000000000000 T __crc32c_64bytes
0000000000000000 T __crc32c_128bytes
```

从上面的输出中，我们可以看到内核中的一些符号，包括 `__crc32c_8bytes`、`__crc32c_16bytes`、`__crc32c_32bytes`、`__crc32c_64bytes` 和 `__crc32c_128bytes`。这些符号都是 CRC32C 算法的实现，它们分别用于计算 8 字节、16 字节、32 字节、64 字节和 128 字节数据的 CRC32C 校验和。

除了使用 `cat` 命令来查看 `/proc/kallsyms` 文件的内容外，用户还可以使用 `kallsyms` 命令来解析内核符号的地址。`kallsyms` 命令的用法如下：

```
$ kallsyms <address>
```

例如，我们可以使用 `kallsyms` 命令来解析 `0xffffffff81000000` 地址的符号，如下所示：

```
$ kallsyms 0xffffffff81000000
0xffffffff81000000 t __crc32c_128bytes
```

从上面的输出中，我们可以看到 `0xffffffff81000000` 地址对应的符号是 `__crc32c_128bytes`。

kallsyms 是一个非常有用的调试工具，它可以帮助用户在内核运行时查看和解析内核符号。通过使用 kallsyms，用户可以快速找到内核中特定符号的地址，或者通过解析内核符号的地址来确定符号的名称。