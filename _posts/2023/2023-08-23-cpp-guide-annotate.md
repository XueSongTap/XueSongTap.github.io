---
layout: article
title: C++最佳实践-注释相关
tags: c++ guide annotation
---

## 注释风格
### 对于作为接口暴露的头文件，需要尽可能使用doxygen工具的注释语法

### 使用 `//` ，注释总是统一在代码上面
不要块注释，不方便cr


## 文件注释

### 头文件加入版权公告

## class注释
### 类的定义都要附带一份注释, 描述类的功能和用法
多个线程访问的话要注意记录多线程使用的规则和不变量

## function注释
### 函数声明处的注释描述函数功能
1. 功能注释应以该功能的隐含主语书写，并应以动词短语开头；例如，“Opens the file”，而不是“Open the file”
2. 在编写函数重写（override）相关注释时，请关注重写本身的细节
3. 构造函数析构函数记录对参数所做的操作
4. 详细记录：
    - 输入输出
    - 类成员函数方法调用后是否还有其参数的引用，以及是否释放
    - 函数分配这是否必须释放内存
    - 函数的使用方式是否对性能有影响


### 定义出的注释描述函数的实现
## 变量注释
### 某些情况下要额外注释


## 实现注释
### 代码中巧妙地，晦涩，重要的地方加以注释
## 标点 拼写 语法相关
### 注意标点, 拼写和语法; 写的好的注释比差的要易读的多.
## TODO注释相关

### cpplint TODO(xiaochuan.ye): 
格式：大写TODO，使用圆括号，然后冒号，再空格，圆括号里的内容是姓名。冒号后需要写明后续action。
```C++
// TODO(xiaochuan.ye): Use a "*" here for concatenation operator.
```

### release 版本不能有TODO
## 弃用注释相关
### 通过弃用注释（DEPRECATED comments）以标记某接口点已弃用.

格式：大写`DEPRECATED` ，使用圆括号，然后冒号，再空格，圆括号里的内容是是姓名。
```C++
// DEPRECATED(xiaochuan.ye):new interface changed to bool IsTableFull(const Table& t)
bool IsTableFull();
```
