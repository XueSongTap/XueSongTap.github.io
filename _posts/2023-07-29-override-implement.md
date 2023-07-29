---
layout: articles
title: c++ 的overload实现
tags: overload c++ 重载 
---

## 引入：

c++ 相比c有一个新特性就是overload 又名函数重载，是c++静态多态或者编译多态的实现

但是c不行，


c程序在汇编过程中，编译器会收集全局符号并生成全局符号表

符号表即，讲符合与其地址一一对应的表哥称为符号表，

在汇编的过程中我们生成了多个符号表，但最后我们只能有一个符号表，所以在链接过程中要对符号表进行合并。在合并的过程中发现同一个函数出现了两次

c++这种重载是如何实现的呢？

## name mangling/name decoration

编译器通过函数名和参数类型识别重载的函数，针对参数列表对每一个函数标识符进行专门编码，

例如:
```cpp
int add(int a, int b){return a+b;}
double add(double a, double b){return a + b;}
```

反汇编：
```cpp
00000000000008da <_Z3addii>:
00000000000008ee <_Z3adddd>:
```
## extern "C"的情况 


当一些代码被放入 extern “C” 块时，C++ 编译器确保函数名是未修改的

## 参考文档

https://en.wikipedia.org/wiki/Name_mangling