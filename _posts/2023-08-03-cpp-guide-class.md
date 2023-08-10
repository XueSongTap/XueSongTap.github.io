---
layout: articles
title: c++ guide 类相关
tags: c++ guide class
---


## 构造函数的职责

### 不要再构造函数中调用虚函数，也不要在无法报出错误的时候进行可能失败的初始化

构造函数不应该调用虚函数，考虑一个工厂函数或者`Init()`方法

### 缺省实现的构造函数，操作符或者析构函数等，要用`=default`,或者`=delete`显式说明

`=default` 显式要求编译器生成函数的一个默认版本，对于构造函数而言（尤其是拷贝/移动构造函数中），可以减轻编码负担。

此外，使用=default还可以显示指定构造函数的权限，以及保持类的特性（如trivial、POD）。

## 隐式类型转换相关

### 不要定义隐式类型转换，对于转换运算符和单参数构造函数，用`explict` 关键字

不要提供隐式类型转换，可以用`ToFloat()` 等函数替代

隐式类型转换允许某一个类型的对象被用于另一种类型的位置，例如把`int` 传递给需要`double`类型的参数


除了语言定义的隐式转换，还可以通过在类定义中添加合适的成员定义自己需要的转换

在源类型定义隐式类型转换，可以通过目的类型名的类型转换运算符实现，例如`operator bool()`

在目的类型中定义隐式转换，通过源类型作为唯一参数的构造函数实现

例如

```c++
std::shared_ptr<int> ptr = func();
if (ptr) {
  // do something
}
```

优点：
- 可以简单替代函数重载
- 有时候目的类名一目了然，可以让类型可用性和表达性更强

缺点：
- 隐式类型转换会让隐藏类型不匹配错误，用户意识不到
- 函数重载的时候让程序难以阅读理解
- 单参数的构造函数可能会被无意用作隐式类型转换

## 拷贝与移动相关

### 如果需要拷贝 或者移动，要先显式写出来 （自行实现或者定义=default）, 否则就显式禁用

### 如果显式定义了拷贝/移动构造函数，就要同时定义相应的拷贝/移动构造符


例子
```C++
class rule_of_five {
  char* cstring;  // raw pointer used as a handle to a dynamically-allocated
                  // memory block
                  // 用作动态分配内存块的原始指针句柄
 public:
  rule_of_five(const char* s = "") : cstring(nullptr) {
    if (s) {
      // 分配内存
      std::size_t n = std::strlen(s) + 1;
      cstring = new char[n];       // allocate
      // 填充数据
      std::memcpy(cstring, s, n);  // populate
    }
  }
  // 析构函数
  ~rule_of_five() {
    delete[] cstring;  // deallocate
  }
  // 拷贝构造符
  rule_of_five(const rule_of_five& other)  // copy constructor
      : rule_of_five(other.cstring) {}

  rule_of_five(rule_of_five&& other) noexcept  // move constructor
      : cstring(std::exchange(other.cstring, nullptr)) {}

  // copy assignment
  rule_of_five& operator=(const rule_of_five& other) {
    return *this = rule_of_five(other);
  }

  // move assignment
  rule_of_five& operator=(rule_of_five&& other) noexcept {
    std::swap(cstring, other.cstring);
    return *this;
  }
  //或者,合并两个赋值运算符实现
  // alternatively, replace both assignment operators with
  //  rule_of_five& operator=(rule_of_five other) noexcept
  //  {
  //      std::swap(cstring, other.cstring);
  //      return *this;
  //  }
};
```
### 为了防止出现切片，避免给基类提供公共复制运算符或者复制/移动构造函数，如果基类就是要实现可以被复制，提供一个公共的`virtual Clone()`方法 和一个受到保护的复制构造函数，派生类可以用该构造函数实现

#### 什么是切片问题

切片问题（Slicing Problem）指的是在面向对象编程中，当通过基类指针或引用操作派生类对象时，只能访问到基类部分的成员和方法，而无法访问到派生类特有的成员和方法的情况。

具体来说，当将一个派生类对象赋值给基类对象或通过基类指针或引用指向派生类对象时，如果使用基类的拷贝构造函数或赋值运算符，那么只会复制基类部分的数据，派生类部分的数据将会被丢失。这就是切片问题的本质。

例如，考虑如下的基类 `Animal` 和派生类 `Dog`：

```cpp
class Animal {
public:
    std::string name;
};

class Dog : public Animal {
public:
    std::string breed;
};
```

如果我们使用基类指针来操作派生类对象：

```cpp
Dog dog;
dog.name = "Buddy";
dog.breed = "Labrador";

Animal* animalPtr = &dog;
```

在这种情况下，`animalPtr` 是一个指向 `Dog` 对象的基类指针。如果我们尝试访问 `animalPtr` 的成员：

```cpp
std::cout << animalPtr->name << std::endl;   // 输出: "Buddy"
std::cout << animalPtr->breed << std::endl;  // 错误！无法访问派生类特有的成员
```

我们只能访问到基类 `Animal` 的成员 `name`，而无法访问到派生类 `Dog` 特有的成员 `breed`。这是因为基类指针只能看到基类部分的成员和方法，而无法访问派生类特有的成员和方法。

为了避免切片问题，可以使用虚函数和多态性来实现运行时的动态绑定。通过在基类中声明虚函数，并在派生类中重写这些虚函数，可以实现在运行时根据对象的实际类型来调用相应的成员函数，而不是仅仅调用基类的成员函数。这样就能够正确地操作派生类对象，而不会发生切片问题。


#### 可能的构造问题
如下，d2 对象的构造过程中，只调用到了派生类的拷贝构造函数， 并没有调用到 基类的拷贝构造函数。

正确的构造过程，应该是调用派生类的拷贝构造函数并且调用基类的拷贝构造函数
```C++
class Base {
 public:
  Base() { std::cout << "Base Default Constructor" << std::endl; }
  Base(const Base&) { std::cout << "Base Copy Constructor" << std::endl; }
};

class Drived : public Base {
 public:
  Drived() { std::cout << "Drived Default Constructor" << std::endl; }
  Drived(const Drived& d) {
    std::cout << "Drived Copy Constructor" << std::endl;
  }
};

int main(void) {
  Drived d1;      // 输出 ：Base Default Constructor
                  //       Drived Default Constructor
  Drived d2(d1);  // 输出 ： Base Default Constructor //
                  // 调用了基类的默认构造函数而不是拷贝构造
                  //         Drived Copy Constructor
}
```
一个简单的解决办法如下，这本身并不难，但是可能会造成非常难定位的Bug，因此十分建议，禁用基类的拷贝构造或移动构造函数。
```C++
Drived(const Drived& d) : Base(d) {
  std::cout << "Drived Copy Constructor" << std::endl;
}
```

#### clone 方法
如果基类的拷贝难以避免时，也非常建议使用 public virtual clone 方法 应付多态的使用场景。
```C++
class B {
 public:
  virtual B* clone() = 0;
  B() = default;
  virtual ~B() = default;
  B(const B&) = delete;
  B& operator=(const B&) = delete;
};

class D : public B {
 public:
  D* clone() override;
  ~D() override;
};
```

`clone()` 方法是一种常见的实现对象拷贝的方式，特别适用于多态的场景。它通过创建一个新对象，并将原始对象的状态复制到新对象中，返回一个指向新对象的基类指针。

在提供的代码示例中，`B` 是一个抽象基类，其中声明了纯虚函数 `clone()`。派生类 `D` 继承了 `B` 并实现了 `clone()` 方法。

下面是一个使用 `clone()` 方法的示例：

```cpp
B* createCopy(const B* obj) {
  return obj->clone();
}

int main() {
  D* d = new D();
  B* copy = createCopy(d);

  // 使用拷贝得到的对象进行操作
  // ...

  delete copy;
  delete d;
}
```

在这个示例中，我们首先创建了一个 `D` 类的对象 `d`。然后，我们调用 `createCopy()` 函数，并将 `d` 的指针作为参数传递给它。`createCopy()` 函数内部调用了 `clone()` 方法，并返回一个指向新对象的基类指针。

接下来，我们可以使用返回的拷贝对象 `copy` 进行操作，无论是基类的方法还是派生类的方法，都可以通过 `copy` 进行调用。



## struct与class相关
### 仅当有数据成员的时候（POD类型） 时候应该用struct，其他一律用class

### 结构体不应该有private 成员变量，变量间不应该有隐含的关联，否则用户直接访问这些变量会破坏这种关联

### 模板编程中，对于无状态类型，例如traits，模板元函数和functor, 可以使用struct而不是class

## Structs Pairs Tuples


### 尽可能的使用struct而不是pair和tuple

struct 可以有自己的名字

## 继承相关

### 使用组合常常比使用继承更合理，如果使用继承的话， 定义public 继承

如果确实需要 protected  或者private 方式进行继承

```C++
class Fly {};
class Animal {};
class Bird : public Animal {
 private:
  Fly flyable_;  // 组合方式实现
}
```
### 只有很严格的满足 "is-a" 的情况下才考虑使用继承

过于复杂的继承关系，影响代码的可读性和维护性

如果结构稳定，继承较为浅，可以使用继承，其他的考虑用组合的方式替代继承

设计模式使用组合关系：
1. 装饰者模式 decorator pattern
2. 策略模式 strategy pattern
3. 组合模式 composite pattern

设计模式使用继承关系：
1. 模板模式

参考： Effective C++ item 32: Make Sure Public Inheritance Models "is-a"。

### 显式重写的虚函数要用override,  重写虚函数不用加virtual 关键字

标记为override的析构函数，如果不是对基类函数的重载的话，编译检查会报错


## 多重继承相关

### 只允许：最多只有一个基类是非抽象类，其他基类都是以 Interface 为后缀的纯接口类

多重继承，最显著的麻烦是二义性

### 对于多重继承，把stateless class 写在继承体前面，利用ECO (Empty Class Optimization) 优化内存占用

```cpp
class Empty{
public:
    void print() {std::cout  << "I am Empty class" << std::endl;}
};

class NotEbo{
    int i;
    Empty e;
};

class ebo :public{
    int i;
}

std::cout << sizeof(notEbo) << std::endl; // 8 内存对齐
std::cout << sizeof(ebo) << std::endl; // 4
```

## 运算符重载相关

cpp 允许用`operator` 关键字 对内建运算符进行重载定义

优点：
- 重载运算符更简单易懂
缺点：
- 操作符运算重载难度大
- 运算符定义在类的外部，引入风险


### 除了少数特定环境外，不要重载运算符，也不要创建用户定义的字面量 （operator”“）


不要可以避免定义运算符重载

更倾向于定义，== ， =, << 而不是定义 Equals(), CopyFrom(), PrintTo()

也可以考虑自定义类型比较器

### 不要重载& || ，"" 不要引入用户定义字面量

### 仅当重载运算符的含义显而易见并且与相应的内置运算符号一致的时候，才定义重载运算符

例如 将 | 用作按位或， 而不是shell央视的管道，不与内置运算符的语法违背

## 存取控制相关

### 将所有数据成员声明为 private， 除非是static const 类型的长远

不要让一个数据成员 成为 `public`，所有人都可以读写访问它，可以使用函数进行访问


对于message难以遵循的， 可以用`mutable_xxx()` 和 `xxx()` 进行区分或者`set_xxx() ` 和`xxx()`

## 声明顺序相关

### 相似的声明放在仪器，public部分放在最前面，后跟 protected， 然后是private， 省略为空的部分，用空行将不同组的声明隔开

习惯的约定

public 通常修饰成员函数放在最前面，方便阅读

protected   一般用来继承，放在中间，

private 一般是保护数据成员的，一般不用数据担心

### 声明的每个部分，按照：类型、常量、工厂函数、构造函数和赋值运算符、析构函数、所有其他方法、数据成员的顺序

