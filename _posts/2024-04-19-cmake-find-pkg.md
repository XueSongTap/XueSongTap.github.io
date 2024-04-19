---
layout: articles
title: cmake find_package() 处理方法
tags: cpp cmake find_package
---


在 cmakelist.txt 文件中，find_package() 命令用于查找并加载外部库的设置。它并不直接管理依赖包，而是依赖于外部预设的模块或配置来找到这些库。



他找的路径是 `cmake/module/XX.cmake`

因此，如果是单独的项目，要进行编译，可以设置path 

```cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake:${CMAKE_MODULE_PATH}")
```
这样的话，会去项目的cmake 目录找