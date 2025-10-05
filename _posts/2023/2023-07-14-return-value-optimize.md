---
layout: articles
title: 深入解析C++返回值优化(RVO)和具名返回值优化(NRVO)的实现机制、编译器优化策略与最佳实践
tags: cpp
---
## 返回值优化是什么

返回值优化（Return value optimization，缩写为RVO）是C++的一项编译优化技术。即删除保持函数返回值的临时对象。这可能会省略两次复制构造函数，即使复制构造函数有副作用。[1] [2]

## 函数的返回机制

函数返回值的传递分为两种情况：

1. 当返回的对象大小不超过8字节时，通过寄存器（eax edx）返回


2. 当返回的对象大小大于8字节时，通过栈返回。此处需要注意的时候，如果返回的是struct或者class对象，即使其大小不大于8字节，也是通过栈返回的。


在通过栈返回的时候，栈上会有一块空间来保存函数的返回值。

当函数结束的时候，会把要返回的对象拷贝到这块区域，对于内置类型是直接拷贝，类类型的话是调用拷贝构造函数。这块区域又称为函数返回的临时对象。

## 示例代码

```cpp
class Obj {
 public:
  Obj() { // 构造函数
    std::cout << "in Obj() " << " " << this << std::endl;
  }
  
  Obj(int n) {
    std::cout << "in Obj(int) " << " " << this << std::endl;
  }

  Obj(const Obj &obj) { // 拷贝构造函数
    std::cout << "in Obj(const Obj &obj) " << &obj << " " << this << std::endl;
  }

  Obj &operator=(const Obj &obj) { // 赋值构造函数

    std::cout << "in operator=(const Obj &obj)" << std::endl;
    return *this;
  }

  ~Obj() { // 析构函数
    std::cout << "in ~Obj() " << this << std::endl;
  }
int n;
};
```
## 禁用优化

```
-fno-elide-constructors
```

```cpp
Obj fun() {
  Obj obj;
  // do sth;
  return obj;
}

int main() {
  Obj obj = fun();
  std::cout << "&obj is " << &obj << std::endl;
  return 0;
}
```

### 禁用优化
```
in Obj()  0x7fff0b5d955c
in Obj(const Obj &obj) 0x7fff0b5d955c 0x7fff0b5d958c
in ~Obj() 0x7fff0b5d955c
in Obj(const Obj &obj) 0x7fff0b5d958c 0x7fff0b5d9588
in ~Obj() 0x7fff0b5d958c
&obj is 0x7fff0b5d9588
in ~Obj() 0x7fff0b5d9588
```

调用构造函数，生成对象

调用拷贝构造函数，生成临时对象

析构第1步生成的对象

调用拷贝构造函数，将第2步生成的临时变量拷贝到main()函数中的局部对象obj中

调用析构函数，释放第2步生成的临时对象

调用析构函数，释放main()函数中的obj局部对象

### 优化

```
in Obj()  0x7ffe3fb781bc
&obj is 0x7ffe3fb781bc
in ~Obj() 0x7ffe3fb781bc
```
编译器对函数返回值优化的方式分为RVO和NRVO(自c++11开始引入)，在后面的文章中，我们将对该两种方式进行详细分析。

在此需要说明的是，因为自C++11起才引入了NRVO，而NRVO针对的是具名函数对象返回，而C++11之前的RVO相对NRVO来说，是一种URVO(未具名返回值优化)


## RVO Return Value Optimization

RVO优化针对的是返回一个未具名对象，也就是说RVO的功能是消除函数返回时创建的临时对象。

那么，编译器优化后与优化前相比，减少了2次拷贝构造函数以及两次析构函数。

编译器明确知道函数会返回哪一个局部对象，那么编译器会把存储这个局部对象的地址和存储返回临时对象的地址进行复用，也就是说避免了从局部对象到临时对象的拷贝操作，这就是RVO。

## NRVO

具名返回值优化(Named Return Value Optimization)

为RVO的一个变种，也是一种编译器对于函数返回值优化的方式。此特性从C++11开始支持，与RVO的不同之处在于函数返回的临时值是具名的。

NRVO与RVO的区别是返回的对象是具名的，既然返回的对象是具名的，那么对象是在return语句之前就构造完成。
## 原理

返回值优化的原理是将返回一个类对象的函数的返回值当做该函数的参数来处理

### RVO原理
RVO优化的原理是消除函数返回时产生的一次临时对象

编译器会将返回值函数的原型进行调整，编译器启用RVO优化，fun()函数会变成如下：

```cpp
void fun(Obj &_obj) {
  Obj obj(1);
  _obj.Obj::Obj(obj); // 拷贝构造函数
  return;
}
```
而main函数内的调用则会变成：

```cpp
int main() {
  Obj obj; // 仅定义不构造
  fun(obj);
  return 0;
}
```
但是仍然存在一次拷贝构造
### NRVO原理
```cpp
void fun(Obj &_obj) {
   _obj.Obj::Obj(1);
}
```

```cpp
int main() {
  Obj obj;
  fun(obj);
  
  return 0;
}
```

fun函数经过优化之后，去掉了RVO优化遗留的拷贝构造问题，达到了优化目标。
## 优化失败

### 运行时依赖(根据不同的条件分支，返回不同变量)
```cpp
Obj fun(bool flag) {
  Obj o1;
  Obj o2;
  if (flag) {
    return o1;
  }
  return o2;
}

int main() {
  Obj obj = fun(true);
  return 0;
}
```

不知道该返回哪个


### 返回全局变量

当返回的对象不是在函数内创建的时候，是无法执行返回值优化的。

### 返回函数参数

与返回全局变量类似，当返回的对象不是在函数内创建的时候，是无法执行返回值优化的。


### 存在赋值行为
(N)RVO只能在从返回值创建对象时发送，在现有对象上使用operator=而不是拷贝/移动构造函数，这样是不会进行RVO操作的。

### 使用std::move()返回


```cpp
Obj fun() {
  Obj obj;
  return std::move(obj);
}

int main() {
  Obj obj = fun();
  return 0;
}
```


```bash
in Obj()  0x7ffe7d4d1720
in Obj(const Obj &&obj)
in ~Obj() 0x7ffe7d4d1720
0x7ffe7d4d1750
in ~Obj() 0x7ffe7d4d1750
```

std::move()返回相比，使用std::move()返回增加了一次拷贝构造调用和一次析构调用
## 参考文献

https://zh.wikipedia.org/wiki/%E8%BF%94%E5%9B%9E%E5%80%BC%E4%BC%98%E5%8C%96

https://mp.weixin.qq.com/s?__biz=MzAxNDI5NzEzNg==&mid=2651170823&idx=1&sn=f5f05e10da2c6c707c7c121001cc319e&chksm=80647958b713f04e7b90edae6281215039f2fe062c50096db30aee2e28d79952632be784e167&scene=132#wechat_redirect


https://seineo.github.io/C-%E8%BF%94%E5%9B%9E%E5%80%BC%E4%BC%98%E5%8C%96.html

https://www.yhspy.com/2019/09/01/C-%E7%BC%96%E8%AF%91%E5%99%A8%E4%BC%98%E5%8C%96%E4%B9%8B-RVO-%E4%B8%8E-NRVO/

https://www.bilibili.com/video/BV1wM41177up/