---
layout: articles
title: modern cpp 中auto 和decltype 的区别
tags: cpp auto decltype
---


## 区别


在现代C++中，`auto`和`decltype`是用于类型推导的关键字，它们有不同的用途和行为。

1. `auto`关键字：
   - `auto`用于自动推导变量的类型。编译器会根据变量的初始化表达式推导出变量的类型，并将其替换为推导得到的类型。
   - `auto`适用于大多数情况，特别是在使用迭代器、范围循环和模板编程时非常有用。
   - `auto`不能用于函数参数、类成员变量和非静态成员函数的返回类型（C++14之前）。

示例：
```cpp
auto x = 42;  // 推导为int类型
auto name = "John";  // 推导为const char*类型
std::vector<int> vec = {1, 2, 3};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    // 使用auto推导迭代器类型
    std::cout << *it << " ";
}
```

2. `decltype`关键字：
   - `decltype`用于获取表达式的类型，而不是变量的类型。
   - `decltype`在编译时计算表达式的类型，并将其作为关键字所在位置的类型。
   - `decltype`可以用于变量声明、函数返回类型声明和模板编程中。

示例：
```cpp
int x = 42;
decltype(x) y = x;  // 推导为int类型

template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;  // 推导为a和b相加的类型
}

int main() {
    int a = 5;
    double b = 3.14;
    auto result = add(a, b);  // 推导为double类型
    std::cout << result << std::endl;
    return 0;
}
```

总结：
- `auto`用于自动推导变量的类型。
- `decltype`用于获取表达式的类型。
- `auto`适用于大多数情况，而`decltype`在需要根据表达式类型进行操作时更有用。



## 实现上的区别

`auto`和`decltype`在底层实现推导的依据上有一些区别。

1. `auto`的推导依据：
   - `auto`关键字通过编译器根据变量初始化表达式的类型进行推导。
   - 编译器在编译时分析变量的初始化表达式，并根据表达式的类型来确定变量的类型。
   - `auto`的推导是基于初始化表达式的静态类型，不考虑运行时的值。

示例：
```cpp
int x = 42;
auto a = x;  // 推导为int类型
auto b = 3.14;  // 推导为double类型
```

2. `decltype`的推导依据：
   - `decltype`关键字通过编译器根据表达式的类型进行推导。
   - 编译器在编译时分析表达式的类型，并将`decltype`关键字所在位置替换为表达式的类型。
   - `decltype`的推导是基于表达式的静态类型，不考虑运行时的值。

示例：
```cpp
int x = 42;
decltype(x) a = x;  // 推导为int类型
decltype(3.14) b = 3.14;  // 推导为double类型
```

需要注意的是，`decltype`的推导结果可能包含类型修饰符（如`const`、`volatile`等），以及引用类型（根据表达式是否为左值或右值决定）。

总结：
- `auto`的推导依据是变量初始化表达式的静态类型。
- `decltype`的推导依据是表达式的静态类型。
- `auto`和`decltype`都是在编译时进行类型推导，不考虑运行时的值。