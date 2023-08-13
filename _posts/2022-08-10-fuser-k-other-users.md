---
layout: articles
title: linux板端如何查找其他用户并踢掉
tags: linux fuser kill
---

### fuser 命令
`fuser` 命令用于查找和操作正在使用指定文件或文件系统的进程。它提供了一种查找和终止进程的方式，以释放文件或文件系统资源。


```shell
fuser [options] <file or directory>
```

其中，`options` 是可选的命令选项，`<file or directory>` 是要查询的文件或目录的路径。

一些常用的 `fuser` 命令选项包括：

- `-k`：终止正在使用文件或目录的进程。
- `-m`：指定文件系统类型，仅查找指定类型的文件系统上的进程。
- `-n <namespace>`：指定命名空间，仅查找指定命名空间中的进程。
- `-v`：显示详细的进程信息。

以下是一些示例用法：

1. 查找正在使用文件 `/path/to/file` 的进程：

   ```
   fuser /path/to/file
   ```

2. 终止正在使用文件 `/path/to/file` 的进程：

   ```
   fuser -k /path/to/file
   ```

3. 查找正在使用目录 `/path/to/directory` 的进程：

   ```
   fuser /path/to/directory
   ```

4. 终止正在使用目录 `/path/to/directory` 的进程：

   ```
   fuser -k /path/to/directory
   ```

### 实践常用
```
fuser -k /dev/pts/0
```
其他用户都是 /dev/pts/0  进行登陆的
