---
layout: articles
title: C++编程最佳实践：语法规范与代码优化指南
tags: cpp
---


## 所有权与指针相关

### 动态分配出的对象最好有单一且固定的所有主，并且通过智能指针传递所有权，优先考虑 unique_ptr 

```C++
// perfer owning pointer over raw in virtual c'tor function
class Investment {
  // ...
};
class Stock : public Investment {
  // ...
};
class Bond : public Investment {
  // ...
};
class RealEstate : public Investment {
  // ...
};

template <typename... Ts>
std::unique_ptr<Investment> makeInvestment(Ts&&... params);

auto pInvestment = makeInvestment(args);
```
### 优先使用 std::make_unique 和 std::make_share 而不是new 

使用 std::make_unique 和 std::make_shared ：

1. 更安全：使用 std::make_unique 和 std::make_shared 创建智能指针可以避免资源泄漏和内存泄漏的风险。这些函数会自动管理内存资源，并在发生异常时正确释放资源，避免资源泄漏的问题。

2. 更高效：使用 std::make_shared 可以提高内存使用效率。它可以在一次内存分配中同时创建对象和共享指针的控制块，减少了内存分配的开销，并且可以减少内存碎片化的问题。

3. 更简洁：使用 std::make_unique 和 std::make_shared 可以使代码更加简洁和易读。这些函数可以自动推导模板参数，避免了手动指定类型的繁琐，并且代码更加清晰明了。
## 右值引用相关

### 仅在特定情况下的使用右值引用
1. 移动构造函数和移动赋值操作
2. 完美转发，std::forward
3. 定义重载

示例：


```cpp
// r value in move c'tor
// Move constructor.移动构造函数
struct MemoryBlock {
  MemoryBlock(MemoryBlock&& other) noexcept;
  MemoryBlock& operator=(MemoryBlock&& other) noexcept;
}

// r value in perfect forwarding 完美转发
template <typename T>
void print(T&& obj) {
  print_impl(std::forward<T>(obj));
}

// define both L & R value version when performance maters 函数虫子啊
template <typename T>
void array<T>::push_back(const T& someth);
template <typename T>
void array<T>::push_back(T&& someth);
```

## 友元相关

### 使用合理的友元类和友元函数

友元通常应该在一个文件里定义，不用在另一个文件中查找类的私有成员的用法

友元扩展但不打破类的封装边界，大多数类应该仅通过公共成员与其他类交互


## 异常相关

### 不建议使用c++异常 （争议）

使用异常会对程序性能造成影响，但是代码可读性好

https://www.zhihu.com/question/22889420
## nonexcept 相关

### 当`nonexcept` 有用且正确的时候指定 `noexcept`

`nonxecept` 说明符用于指定函数是否抛出异常，如果抛出i长，程序通过`std::terminate` 崩溃，编译时候检查，如果不一场，返回true

```cpp
struct MemoryBlock {
    MemoryBlock(MemoryBlock&& other);
}

std::vector<MemoryBlock> blocks;
MemoryBlock a_block{1024};
blocks.push_back(
    std::move(a_block));  // 这里调用的是拷贝构造而不是移动构造，因为移动构造不是nonexcept
```
## 运行时类型识别相关
### 尽量不要用RTTI
`dynamic_cast` 最坏的情况下比 `reinterpret_cast` 版本慢了 16 倍，同时每个对象会增加 `typeinfo` 的空间

参考:

https://tinodidriksen.com/2010/04/cpp-dynamic-cast-performance/


| **g++ 4.4.1**                          | **Ticks** | **Relative Factor** |
| -------------------------------------- | --------- | ------------------- |
| dynamic_cast level1-to-base success    | 6008176   | 1.00                |
| dynamic_cast level2-to-base success    | 7009504   | 1.17                |
| dynamic_cast level2-to-level1 success  | 36055776  | 6.00                |
| dynamic_cast level3-to-base success    | 6008176   | 1.00                |
| dynamic_cast level3-to-level1 success  | 43178320  | 7.19                |
| dynamic_cast level3-to-level2 success  | 36056256  | 6.00                |
| dynamic_cast onebase-to-twobase fail   | 27042336  | 4.50                |
| dynamic_cast onelevel1-to-twobase fail | 33051232  | 5.50                |
| dynamic_cast onelevel2-to-twobase fail | 39096128  | 6.51                |
| dynamic_cast onelevel3-to-twobase fail | 99445824  | 16.55               |
| dynamic_cast same-type-base success    | 7009392   | 1.17                |
| dynamic_cast same-type-level1 success  | 30931008  | 5.15                |
| dynamic_cast same-type-level2 success  | 30442688  | 5.07                |
| dynamic_cast same-type-level3 success  | 30478432  | 5.07                |
| member variable + reinterpret_cast     | 8013216   | 1.33                |
| reinterpret_cast known-type            | 6008160   | 1.00                |
| virtual function + reinterpret_cast    | 11017248  | 1.83                |

运行时查询对象的类型通常意味着设计的问题，有缺陷

RTTI 有合理的用途，但是容易被滥用，单元测试时候可以自由使用RTTI

考虑其他选项来查询类型：
1. 虚拟方法，根据特定子类类型执行不同代码，工作在对象本身内
2. 工作属于对象之外，考虑使用double-dispatch的解决方案


当程序的逻辑保证一个基类的实例实际上是一个特定派生类的实例的时候，可以在对象上自由使用 `dynamic_cast`，这种情况下，通常可以用`static_cast` 替代

采用访问者模式：
```cpp
struct AnimalVisitor {
  virtual void Visit(struct Cat *) = 0;
  virtual void Visit(struct Dog *) = 0;
};

struct ReactVisitor : AnimalVisitor {
  ReactVisitor(struct Person *p) : person{p} {}
  void Visit(struct Cat *c);
  void Visit(struct Dog *d);
  struct Person *person = nullptr;
};
struct Animal {
  virtual std::string name() = 0;
  virtual void Visit(struct AnimalVisitor *visitor) = 0;
};
struct Cat : Animal {
  std::string name() { return "Cat"; }
  void Visit(AnimalVisitor *visitor) {
    visitor->Visit(this);
  }  // 2nd dispatch <<---------
};
struct Dog : Animal {
  std::string name() { return "Dog"; }
  void Visit(AnimalVisitor *visitor) {
    visitor->Visit(this);
  }  // 2nd dispatch <<---------
};
struct Person {
  void ReactTo(Animal *_animal) {
    ReactVisitor visitor{this};
    _animal->Visit(&visitor);  // 1st dispatch <<---------
  }
  void RunAwayFrom(Animal *_animal) {
    std::cout << "Run Away From " << _animal->name() << std::endl;
  }
  void TryToPet(Animal *_animal) {
    std::cout << "Try To Pet " << _animal->name() << std::endl;
  }
};
// Added Visitor Methods
void ReactVisitor::Visit(Cat *c) {  // Finally comes here <<-------------
  person->TryToPet(c);
}
void ReactVisitor::Visit(Dog *d) {  // Finally comes here <<-------------
  person->RunAwayFrom(d);
}

int main() {
  Person p;
  for (auto &&animal : std::vector<Animal *>{new Dog, new Cat}) {
    p.ReactTo(animal);
  }
  return 0;
}
```
访问者模式用于在不修改被访问对象的前提下，定义对对象的新操作。

在这个示例中，有三个类：`Animal`、`Cat` 和 `Dog`。`Animal` 是一个抽象基类，定义了两个纯虚函数 `name()` 和 `Visit()`。`Cat` 和 `Dog` 是 `Animal` 的派生类，实现了这两个函数。

接下来，定义了一个访问者接口 `AnimalVisitor`，其中包含了两个纯虚函数 `Visit()`，分别接受 `Cat*` 和 `Dog*` 类型的参数。

然后，定义了一个具体的访问者 `ReactVisitor`，它继承自 `AnimalVisitor`。`ReactVisitor` 中包含了一个指向 `Person` 类对象的指针，并在构造函数中初始化。`ReactVisitor` 实现了 `Visit()` 函数，分别处理 `Cat*` 和 `Dog*` 类型的对象。

`Person` 类中定义了三个函数：`ReactTo()`、`RunAwayFrom()` 和 `TryToPet()`。`ReactTo()` 函数接受一个 `Animal*` 类型的参数，然后创建一个 `ReactVisitor` 对象，并调用传入的 `Animal` 对象的 `Visit()` 函数，实现双重分派（double dispatch）。

在 `Visit()` 函数中，根据对象的实际类型，调用 `AnimalVisitor` 的相应 `Visit()` 函数。这样，根据具体的访问者类型和被访问对象的类型，会调用正确的操作函数。

在 `main()` 函数中，创建了一个存放 `Animal*` 类型指针的向量，并依次调用 `Person` 对象的 `ReactTo()` 函数，实现了对不同类型动物的不同反应。


访问者模式体现了双重分派（double dispatch）的思想。在传统的单分派中，函数的调用取决于调用者的类型。但是在某些情况下，我们需要根据两个对象的类型来确定调用的函数，这就是双重分派的需求。

在访问者模式中，首先定义了一个抽象的访问者接口（`AnimalVisitor`），其中包含了多个纯虚函数，每个函数对应一个被访问对象的类型。然后，被访问对象（`Animal` 的派生类）实现了一个接受访问者的函数（`Visit()`），并在该函数中调用访问者的相应函数。

当访问者需要访问被访问对象时，首先创建一个具体的访问者对象（`ReactVisitor`），将其传递给被访问对象的 `Visit()` 函数。在 `Visit()` 函数内部，根据被访问对象的实际类型，调用访问者的相应函数。这样，通过两次分派，确定了最终调用的函数。

访问者模式的优点是能够在不修改被访问对象的前提下，定义新的操作。通过将操作封装在访问者中，可以实现对被访问对象的多种不同操作，而无需修改被访问对象的类层次结构。这样可以提高代码的可维护性和扩展性。

总结来说，访问者模式通过双重分派的思想，将操作封装在访问者中，实现了对不同类型对象的不同操作，同时保持了对象结构的稳定性和可扩展性。
## 类型转换相关


### 使用 c++的类型转换，static_cast<int>() 而不是 (int)x
1. 用 `static_cast` 替代C语言风格的值转换, 或某个类指针需要明确的向上转换为父类指针时。
2. 用 `const_cast` 去掉 `const` 限定符。
3. 用 `reinterpret_cast` 指针类型和整型或其它指针之间进行不安全的相互转换. （可能比较危险）
## 流相关

### 只在记录日志的时候使用流


## 前置自增自减相关

### 对于迭代器和其他模板对象使用 前缀的 ++i 的自增、自减
```cpp
std::vector v{1, 2, 3};
// perfer ++iter over iter++
for (auto& iter = v.begin(); iter != v.end(); ++iter) {
  // ...
}
```
## const相关

### 所有可能情况下使用const
1. 如果函数保证不会修改通过引用或者指针传递的参数，用const
2. 对于通过值传递的参数，不用const
3. 把方法声明置为常量，除非他改变对象的逻辑状态
4. 所有const操作应该可以安全相互并发调用

### 使用const的位置，使用const XXX & 类型的形式



## constexpr 相关

### c++11起，用constexpr 来定义真正的常量或者实现常量的初始化

参考：https://en.cppreference.com/w/cpp/language/constexpr


constexpr definitions enable a more robust specification of the constant parts of an interface. Use constexpr to specify true constants and the functions that support their definitions. Avoid complexifying function definitions to enable their use with constexpr. Do not use constexpr to force inlining.

使用constexpr来指定真正的常量和支持其定义的函数，可以使接口的常量部分更加稳定。避免复杂化函数定义以便与constexpr一起使用。不要使用constexpr来强制内联。

示例：

```cpp
#include <iostream>
#include <stdexcept>
// C++11 constexpr functions use recursion rather than iteration
// (C++14 constexpr functions may use local variables and loops)
// constexpr函数：代码中定义了一个constexpr函数factorial，用于计算阶乘。
//constexpr函数在编译时求值，并且可以用于编译时常量的计算。
//在C++11中，constexpr函数使用递归而不是循环来实现。在C++14中，constexpr函数可以使用局部变量和循环。
constexpr int factorial(int n) { return n <= 1 ? 1 : (n * factorial(n - 1)); }
// literal class
//字面量类（Literal Class）：
// 定义了一个字面量类conststr，用于表示常量字符串。是一种特殊的类，
// 它的对象可以在编译时求值，并且可以用于编译时常量的定义。conststr类中的成员函数都被声明为constexpr，以支持在编译时进行求值。
class conststr {
  const char* p;
  std::size_t sz;

 public:
  template <std::size_t N>
  constexpr conststr(const char (&a)[N]) : p(a), sz(N - 1) {}

  // constexpr functions signal errors by throwing exceptions
  // in C++11, they must do so from the conditional operator ?:
  // constexpr函数中的异常处理：在conststr类的成员函数operator[]中，通过抛出异常来处理越界访问。
  constexpr char operator[](std::size_t n) const {
    return n < sz ? p[n] : throw std::out_of_range("");
  }
  constexpr std::size_t size() const { return sz; }
};

// C++11 constexpr functions had to put everything in a single return statement
// (C++14 doesn't have that requirement)

//在C++11中，constexpr函数必须使用条件运算符?:来抛出异常。
//在C++14中，不再有这个限制。
constexpr std::size_t countlower(conststr s, std::size_t n = 0,
                                 std::size_t c = 0) {
  return n == s.size()                ? c
         : 'a' <= s[n] && s[n] <= 'z' ? countlower(s, n + 1, c + 1)
                                      : countlower(s, n + 1, c);
}

// output function that requires a compile-time constant, for testing
//使用constexpr的输出函数：代码中定义了一个模板结构constN，用于在编译时输出常量。
//通过将模板参数设置为constexpr表达式的值，可以在编译时输出结果。
template <int n>
struct constN {
  constN() { std::cout << n << '\n'; }
};

int main() {
  std::cout << "4! = ";
  constN<factorial(4)> out1;  // computed at compile time

  volatile int k = 8;  // disallow optimization using volatile
  std::cout << k << "! = " << factorial(k) << '\n';  // computed at run time

  std::cout << "the number of lowercase letters in \"Hello, world!\" is ";
  constN<countlower("Hello, world!")> out2;  // implicitly converted to conststr
}
```
constexpr关键字用于指定在编译时求值的常量表达式和函数。它可以用于计算编译时常量、定义字面量类和进行编译时的错误处理。

通过在编译时进行求值，可以提高程序的性能和效率，并在一些特定的场景下实现更灵活的编程。
## 整形相关

### 推荐使用 <stdint.h> 中长度精确的整型
c++ 内建的，仅有 `int`, 如果需要不同大小的变量，使用上述头文件中的长度精确的整形， `int16_t` 或者`int64_t`等

而且运算的时候可能会溢出，可以考虑直接用更大的类型

无符号整数适用于表示位域和模运算


无符号整数和符号整数直接比较会引发的问题：
```C++
#include <iostream>
int main() {
  int i = -1;
  unsigned int j = 1;
  if (i < j)  // ops! compare between signed and unsigned
    std::cout << " i is less than j";
  else
    std::cout << " i is greater than j";
  return 0;
}

// Output: i is greater than j
```
## 64位移植性相关

### 对64 位和32位 系统优化，处理打印、比较、结构体对齐的时候额外注意


1. 可移植的printf()转换说明符：
   - `<cinttypes>`头文件中的PRI宏（例如`PRId64`、`PRIu32`）提供了用于整数typedef的可移植转换说明符。
   - 使用这些宏可能会很繁琐，并且在每种情况下都要求使用它们可能并不实际。
   - 如果可能的话，建议避免或升级依赖于printf系列的API。
   - 相反，考虑使用支持类型安全数值格式化的库，如`StrCat`或`Substitute`，或者使用`std::ostream`进行格式化输出。
   - 但是，请注意对于标准位宽typedef（例如`int64_t`、`uint64_t`、`int32_t`、`uint32_t`），PRI宏是唯一可移植的指定转换的方式。

2. sizeof(void *)与sizeof(int)的区别：
   - `void*`和`int`的大小在某些平台上可能不同。
   - 如果需要一个与指针大小相匹配的整数类型，请使用`<cstdint>`中的`intptr_t`。

3. 结构体对齐和存储在磁盘上：
   - 默认情况下，具有`int64_t`或`uint64_t`成员的结构体/类在64位系统上以8字节对齐。
   - 在32位和64位代码之间共享此类结构体时，需要确保在两种架构上进行相同的结构体打包。
   - 大多数编译器提供了改变结构体对齐的方法：
     - 对于gcc，可以使用`__attribute__((packed))`。
     - MSVC提供了`#pragma pack()`和`__declspec(align())`。

## 预处理宏相关

### 使用宏时候谨慎，尽量用inline，enum和常量代替
如果使用宏：
1. 不要在.h 使用宏
2. #在使用宏之前定义，然后立即取消

```C++
int limit(int height) {
#define MAX_HEIGHT 720
  return std::max(height, MAX_HEIGHT);
#undef MAX_HEIGHT
}
```
3. 在用自己的宏替换现有宏之前，不要只取消定义它；相反，选择一个可能是唯一的名称
4. 不要使用##来生成函数/类/变量名。
5. 可以用constexpr替换宏
```C++
// macro can be replaced with constexpr
#define PI 3.14

// constexpr version
constexpr auto PI = 3.14;

// macro version
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// constexpr version
template <typename T1, typename T2>
constexpr auto MAX(T1 a, T2 b) {
  return a > b ? a : b;
}
```
6. 不鼓励从头文件中导出宏(即在头文件中定义宏，不在头文件末尾之前取消定义宏)
## 0 nullptr和NULL相关
### 整数用`0`， 实数用`0.0`, 指针用`nullptr`, 字符串用`\0`

c++11 往后用`nullptr`

c++03 用`NULL`

字符 (串) 用 '\0', 不仅类型正确而且可读性好。


### 对于float，初始化要加上f，例如 float f =0.1f

## sizeof相关

### 用sizeof(varname) 替代sizeof(type)

使用 sizeof(varname) 是因为当代码中变量类型改变时会自动更新

## 类型推导auto相关
### 建议用auto 局部变量 绕过繁琐的类型名，可读性好

## 类模板参数推导相关 class template argument deduction

 
cpp11 不支持，待更新

https://google.github.io/styleguide/cppguide.html#CTAD




## 指定初始化函数相关 Designated Initializers

c++11 不支持，待更新


https://google.github.io/styleguide/cppguide.html#Designated_initializers


## Lambda 表达式相关



### 适当使用lambda比倒是，别用默认捕获，所有捕获方式显式写出来

1. 仅当lambda的生存期明显短于任何潜在捕获的时候，才用默认的引用捕获 ([&])
2. 使用默认值捕获 ([=])仅仅作为短lambda绑定几个变量的一种方法，变量必须一目了然
3. 仅使用捕获来实际捕获封闭范围中的变量。不要使用带有初始值设定项的捕获来引入新名称，或实质性地更改现有名称的含义

## 模板编程相关

### 不要使用复杂的模板编程


## warning相关
### 不要忽略 warning -Werror 编译选项强制要求修复warning

