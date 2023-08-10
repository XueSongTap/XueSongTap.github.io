---
layout: articles
title: c++ guide 命名相关
tags: c++ guide name
---


## 通用命名规则

首选将缩写当做为单个单词，例如 `StartRpc()` 而不是 `StartRPC()` 。

参考： https://www.educative.io/collection/page/10370001/5111386763952128/6533026123087872
### repo 风格保证相对统一

- **文件命名** 
全部小写，单词间用"_" 链接，例如`file_name.h`

定义`FooBar` 类型的文件： `foo_bar.h` 和 `foo_bar.cc` 

- **类型命名** 

单词首字母大写，如 MyClass
 ```cpp
class MyClass {};  // class|

struct MyStruct {};  // struct

typedef int64_t TimeStamp;  // type alias

using TimeStamp = int64_t;  // type alias

template <typename Item>  //  type template parameter
class Container {};
```

- 变量

全部小写，单词间以"_"连接，如：`param_var`

- 类成员变量

全部小写，单词间以"_"连接，以"_"结尾，如： `mem_var_`

- 结构体成员变量

【建议】全部小写，单词间以"_"连接，如： mem_var

- 静态变量

在成员变量的命名规则基础上，添加前缀s_，如：`s_mem_var_`

- 函数

单词首字母大写，对于缩写，值对首字母大写，如：`DoFpn` 而非 `DoFPN`

- 常量和枚举

枚举名命名规则同类型命名。以“k”开头，单词首字母大写，如：kEnumName，与宏的命名区分开

- 宏
全部大写，单词间使用"_"连接，如MACRO_NAME


- 命名空间

全部小写，单词间以"_"连接。顶级命名空间需要基于项目名

避免使用与已知顶级名称空间相同的嵌套名称空间


首选唯一的项目标识符（websearch:：index，websearch:：index_util）而不是像websearch::util这样容易发生冲突的名称

使用文件名生成唯一的内部名称（websearch:：index:：frobber_internal用于frobber.h