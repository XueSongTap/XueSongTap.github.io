---
layout: articles
title: c++ guide 头文件
tags: c++ guide header
---

头文件相关的c++ guide 

## 头文件要Self-contained 


### 头文件要能够自给自足，即 self contained， 头文件本身以来的其他头文件，需要全部包含

也就是说，需要保证 在包含该头文件后，出于易维护性考虑，可以不需要引入其它头文件，就可以保证编译通过

例如
```cpp
// foo.h
#ifndef FOO_H_
#define FOO_H_
// print_str 接口中的入参是 string 类型
// 所以要求在 这加上  头文件引用
#include <string>
void print_str(const std::string& input);
#endif  // A_H_
```

保证引用foo.h 的地方不用添加 #include <string>


### 模板和内联函数的定义和声明放在一个文件中

模板是遵循谁用谁生成的原则，如果声明和定义分散在不同文件，那么在链接过程中就会报错

### 只引用直接使用的头文件，不要传递依赖

例如，`foo.cc` 使用了`bar.h`, `foo.cc`应该直接包含`bar.h`, 不能因为`foo.h` 有包含`bar.h` 而省略

## 头文件的保护

### #prama once 防止多重包含

## 前置声明

### 尽可能避免前置声明

虽然前置声明可以避免不必要的重复编译，但是
- 会隐藏依赖关系
- 不利于库的后续更新，函数和模板的前置声明阻断了哭的所有者对api的修改，例如扩大参数类型，添加具有默认值的模板参数或者迁移命名空间
- 针对std::的正向声明符号会产生未定义的行为


例如：

```cpp
// B.h
class B {};
class C: B {}; // 类C继承自类B

// XXX.cpp
class B;
void func(B*);
void func(void*);

// 如果使用前置声明了B，此时会编译类B，但类C继承自类B，而类C却未编译。
// 导致test函数内调用的是 func(void*)，这就已经改变了代码含义了
void test(C* ob) { func(ob) };
```

通常来说，你都不需要主动去写class A这种前置声明。include能编译通过的时候都不要去写前置声明，

应该仅当出现了头文件循环依赖导致编译失败的时候，才去考虑去写前置声明！


参考：

https://zhuanlan.zhihu.com/p/386400840

## 内联函数

### 函数10行或者更少，考虑定义为内联函数

编译器会把内联函数展开，并非函数调用

即使被声明inline，虚函数和递归函数通常不是内联的

优点：函数体小，运行函数高校

滥用内联将导致程序变得更慢. 内联可能使目标代码量或增或减, 这取决于内联函数的大小. 内联非常短小的存取函数通常会减少代码大小, 但内联一个相当大的函数将戏剧性的增加代码大小.

### 递归函数不应该是inline

递归的展开无法像循环一样，递归层数位置，编译器不支持

### 虚函数不应该设置成内联属性

虚函数被函数调用，确实可以inline

但是大部分虚函数调用是通过对象的指针完成，这类行为无法被`inline` 

### inline 要避免循环或者switch 除非大部分情况下循环或switch不被执行

提交分支的inline 函数会导致难以预测分支，因为每个分支的实例都是独立的


并且 switch 意味着每个展开的地方函数展开有个跳转表，对代码段空间浪费比较大


## #include 的路径和顺序相关

### 顺序为：相关头文件，c库，cpp库，其他库.h， 本项目内的.h, 每个区域用空行隔开

例如：

```cpp
// 相关头文件
#include <base/foo.h>

//c库
#include "stdio.h"

//cpp 库
#include <iostream>

// 其他库的 .h
#include <glog/logging.h>

// 本项目内的 .h
#include "utils/xxx.h"
```

### 避免使用 . 或者.. 目录

### C库 cpp库 和第三方库用<>

### 条件包含#ifdef 的引入应该放在最后面
