---
layout: articles
title: cmake的使用
tags: cmake cpp
---
# cmake的使用

## cmake 主要规定了什么

cmake版本
```cmake
cmake_minimum_required(VERSION 3.10) 
```

工程名字
```cmake
project(CalculateSqrt) 
```

构建工程的源文件
```cmake
add_executable(CalculateSqrt hello.cxx) 
```

设置cpp的版本
```cpp
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED True) 

```

项目编译需要include的路径
```cpp
target_include_directories(CalculateSqrt PUBLIC 
                           "${PROJECT_BINARY_DIR}" 
                           ) 
```

添加链接库
```cmake
# 使用特定的源码为项目增加lib 
add_library(MathFunctions mysqrt.cpp) 
```

