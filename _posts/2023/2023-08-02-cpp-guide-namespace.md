---
layout: article
title: c++ 作用域/命名空间相关最佳实践
tags: c++
---


## 命名空间相关

### 在.cc 文件内部使用匿名空间或者static声明

### 禁用 using namespace ...  可以使用 using type alias

### 禁用 inline namespace

内联空间的名字可以被外部使用，和命名空间作用相违背


### 不要在头文件中定义空间别名

头文件中引入的namespace 都会成为公开api的一部分


### src中，namespace 应该包裹住除了#include 宏定义和其他namespace中前置声明以外的所有代码

### 不要在std:: 空间声明任何东西，包括前置声明

做法的结果未被定义， 不好移植，要直接引用头文件

## 内部连接性相关

### .cc 中定义一个不需要被外部引用的变量的时候，需要放在匿名空间或者static， 但.h 中不要这么做


```c++
// *.cpp
static ...

namespace {
...
}  // namespace
```
将不需要被外部引用的变量放在匿名命名空间中可以限制其作用域只在当前文件中，避免了对其他文件的命名冲突，并且可以隐藏变量的实现细节，提高代码的封装性。

将变量声明为static后，它只能在当前文件中访问，无法被其他文件引用

头文件通常用于声明公共接口和共享的全局变量，将变量放在匿名命名空间或声明为static会导致其只能在当前文件中使用，违反了头文件的设计目的。


## 非成员函数 静态成员函数和全局函数相关


### 用静态成员函数或者命名空间的非成员函数，不用裸的全局函数


裸的全局函数容易引起污染全局使用域

```c++
void Function1();  // bad
namespace myproject {
void Function2();  // good
}

class MyClass {
public:
  static void staticFunc() {}
};

MyClass::staticFunc(); // good
```

### 函数直接置于命名空间，不要用类的静态方法进行模拟

选择
```c++
namespace myproject {
namespace foo_bar {  // 优先使用命名空间
void Function1();
void Function2();
}  // namespace foo_bar
}  // namespace myproject
```
而非
```c++
namespace myproject {
class FooBar {  // FooBar 只有两个静态成员方法，与类并不强相关
 public:
  static void Function1();
  static void Function2();
};
}  // namespace myproject
```

## 局部变量相关

### 函数变量尽量在最小作用域内

比如 while for循环直接定义在声明里

### 如果在循环中的局部变量是个类类型，把这个对象移到外面更高校

避免重复的构造和析构

选择
```c++
Foo f;  // 构造函数和析构函数只调用 1 次
for (int i = 0; i < 1000000; ++i) {
  f.DoSomething(i);
}
```
而非
```c++
// 低效的实现
for (int i = 0; i < 1000000; ++i) {
  Foo f;  // 构造函数和析构函数分别调用 1000000 次!
  f.DoSomething(i);
}
```

### 不要定义静态存储周期的非 trivial 析构的类对象

非平凡（non-trivial）特性的类对象是指具有自定义的构造函数、析构函数、拷贝构造函数或移动构造函数的类对象。这些特性使得类对象具有一些自定义的行为或资源管理方式，而不仅仅是使用默认的构造函数和析构函数。

静态存储周期定义对象的析构销毁顺序是未被定义的，多线程等环境下，喜感函数non trivial的话，可能会在析构函数中访问一个已经在其他线程或者编译单元中释放了的对象或者资源

(同一个编译单元是明确，即： 静态初始化优先于动态初始化，初始化顺序按照声明顺序，但是不同的编译单元之间初始化顺序和销毁顺序不明确)

如何全局变量的声明是constexpr,就是满足要求的

```cpp
const int kNum = 10;  // allowed

void foo() {
//kArray是一个具有静态存储周期的常量 std::array<int, 3>。
//它使用了 constexpr 修饰符，这意味着在编译时就可以计算出它的值，并且它的析构函数是平凡的。
  static const char* const kMessages[] = {"hello", "world"};  // allowed
}

// allowed: constexpr guarantees trivial destructor
constexpr std::array<int, 3> kArray = {{1, 2, 3}};

// bad: non-trivial destructor
//std::string作为类模板，它可能具有非平凡的析构函数。
const std::string kFoo = "foo";

// bad for the same reason, even though kBar is a reference (the
// rule also applies to lifetime-extended temporary objects)
//即使 kBar 是一个引用，constexpr 的规则也适用于被延长生命周期的临时对象。
const std::string& kBar = StrCat("a", "b", "c");

```

注：
- `constexpr` 析构函数之所以被保证为平凡的（trivial），是因为 `constexpr` 成员函数在编译时必须是可求值的（evaluable）。为了满足这个要求，`constexpr` 析构函数必须满足以下条件：

1. 析构函数的函数体为空。
2. 析构函数只能直接或间接地调用其他平凡析构函数。
3. 析构函数不能有任何虚函数调用。
4. 析构函数不能有任何基类或成员对象的非平凡析构函数调用。


### 禁止使用含有副作用的函数初始化POD全局变量

多编译单元的静态变量执行的构造和析构顺序不明确，会让代码难以移植

对于POD（Plain Old Data）类型的全局变量，可以使用constexpr或者在运行时使用函数来初始化。constexpr函数在编译时求值，可以保证静态变量的初始化在编译期间完成。
#### POD 是什么
POD（Plain Old Data）是C++中的一个概念，它指的是一种简单的数据类型，没有非平凡的构造函数、析构函数和虚函数，并且可以通过内存拷贝进行复制。

POD类型通常是基本数据类型（如整数、浮点数）、C风格的结构体和数组等。它们的特点是可以直接进行内存拷贝，不需要特殊的构造或析构过程。

下面是一个POD类型的全局变量的示例：

```cpp
#include <iostream>

struct Point {
  int x;
  int y;
};

Point globalPoint = {1, 2};

int main() {
  std::cout << "Global point: (" << globalPoint.x << ", " << globalPoint.y << ")" << std::endl;
  return 0;
}
```

在这个示例中，`Point` 是一个简单的结构体，它只包含两个整数成员 `x` 和 `y`。`globalPoint` 是一个POD类型的全局变量，它被初始化为 `{1, 2}`。在 `main` 函数中，我们可以直接访问并打印 `globalPoint` 的成员。




#### 错误代码示例

以下是一个错误的代码示例，其中使用了含有副作用的函数来初始化POD全局变量，并且在多个编译单元中使用了这些全局变量：

**File1.cpp:**

```cpp
#include <iostream>

// 含有副作用的函数
int getValue() {
  static int value = 0;
  return ++value;
}

// 使用含有副作用的函数初始化POD全局变量
int globalValue = getValue();

void printGlobalValue() {
  std::cout << "Global value: " << globalValue << std::endl;
}
```

**File2.cpp:**

```cpp
#include "File1.cpp"

int main() {
  printGlobalValue();
  return 0;
}
```

在这个例子中，`getValue()` 函数是一个含有副作用的函数，它在每次调用时会递增一个静态局部变量 `value` 的值，并返回递增后的值。然后，全局变量 `globalValue` 在初始化时调用了 `getValue()` 函数，导致 `globalValue` 的值不再是一个简单的POD类型，而是受到 `getValue()` 函数副作用的影响。

问题出在多编译单元的静态变量执行的构造和析构顺序不明确。在这个例子中，无法确定 `File1.cpp` 和 `File2.cpp` 中的静态变量 `globalValue` 的初始化顺序。这可能导致在 `main` 函数中调用 `printGlobalValue()` 时，`globalValue` 的值不是预期的结果。
### 函数静态局部变量可以用动态初始化，但是不鼓励对静态类成员变量或者命名空间范围内的变量用静态初始化

静态局部变量是在函数内部定义的变量，但其生命周期延长到整个程序的执行期间。静态局部变量可以使用动态初始化，也就是在运行时使用表达式进行初始化，而不仅仅是使用常量表达式。

静态类成员变量是类中被声明为静态的成员变量，它们在类的所有对象之间共享。静态类成员变量应该避免使用静态初始化，因为静态初始化的顺序在不同的编译单元中是不确定的，这可能导致不同编译单元中的静态成员变量的初始化顺序不一致。

命名空间范围内的变量是在命名空间中定义的变量，它们在命名空间范围内可见。与静态类成员变量类似，命名空间范围内的变量也应该避免使用静态初始化，以避免不同编译单元中的初始化顺序问题。
#### 静态初始化例子
当静态类成员变量或命名空间范围内的变量使用静态初始化时，它们的初始化顺序在不同编译单元中是不确定的。以下是一个示例代码，展示了这种情况下可能出现的问题：

**File1.cpp:**

```cpp
#include <iostream>

namespace MyNamespace {
    // 命名空间范围内的静态变量使用静态初始化
    int staticVariable = getValue();

    int getValue() {
        return 5;
    }
}

void printStaticVariable() {
    std::cout << "Static variable: " << MyNamespace::staticVariable << std::endl;
}
```

**File2.cpp:**

```cpp
#include "File1.cpp"

int getValue() {
    return 10;
}

int main() {
    printStaticVariable();
    return 0;
}
```

在这个例子中，`File1.cpp` 中的命名空间 `MyNamespace` 内有一个静态变量 `staticVariable`，它使用静态初始化，并调用了 `getValue()` 函数来初始化。然后，`File2.cpp` 中也定义了一个 `getValue()` 函数，返回不同的值。

问题在于，由于静态初始化的顺序不确定，`MyNamespace::staticVariable` 的初始化可能在 `File2.cpp` 中的 `getValue()` 函数定义之前发生，导致 `staticVariable` 的值为 5 而不是预期的 10。因此，当调用 `printStaticVariable()` 函数时，输出的结果可能会出乎意料。

为了避免这种问题，应该避免对静态类成员变量或命名空间范围内的变量使用静态初始化，而是使用动态初始化或其他方式来确保初始化顺序的一致性。
#### 动态初始化例子
下面是一个示例代码，展示了函数内的静态局部变量使用动态初始化的情况：

```cpp
#include <iostream>

int getValue() {
    // 模拟一些复杂的计算过程
    std::cout << "Performing complex calculation..." << std::endl;
    return 42;
}

void myFunction() {
    // 静态局部变量使用动态初始化
    static int staticVariable = getValue();

    std::cout << "Static variable: " << staticVariable << std::endl;
}

int main() {
    myFunction();
    myFunction();

    return 0;
}
```

在这个例子中，`myFunction()` 是一个函数，内部有一个静态局部变量 `staticVariable`。它使用动态初始化，调用了 `getValue()` 函数来获取初始化值。

当调用 `myFunction()` 时，静态局部变量 `staticVariable` 只会在第一次调用时进行初始化，后续的调用不会再触发初始化过程。这是因为静态局部变量的生命周期延长到整个程序的执行期间，而不是每次函数调用时都重新初始化。

运行上述代码，输出将会是：

```
Performing complex calculation...
Static variable: 42
Static variable: 42
```

可以看到，在第一次调用 `myFunction()` 时，静态局部变量 `staticVariable` 进行了动态初始化，执行了复杂的计算过程。而在后续的调用中，静态局部变量的值保持不变，不会重新初始化。

这样的动态初始化可以用于在函数内部保存一些需要计算或获取的值，并且只在第一次调用时进行初始化，后续调用可以直接使用已经计算好的值，提高了程序的效率。
### 最佳实践

#### 对于去哪句使用的字符串，用char 或者char*

#### 如果需要一个静态固定的容器，例如查找表，不用map, set,  尝试用std::array, std::array<std::pair>

线性查找够用，尽量让数据经过排序，并且用二分查找


#### 如果必须要使用动态容器，并且希望在程序的整个生命周期内保持容器的存在，而不需要手动删除它。可以使用函数局部静态指针或引用来实现这一点

在某些情况下，可能需要创建动态容器，并且希望在程序的整个生命周期内保持容器的存在，而不需要手动删除它。可以使用函数局部静态指针或引用来实现这一点。

```cpp
#include <iostream>
#include <vector>

std::vector<int>* createDynamicContainer() {
    static std::vector<int>* dynamicContainer = new std::vector<int>();
    return dynamicContainer;
}

void addToDynamicContainer(int value) {
    std::vector<int>* dynamicContainer = createDynamicContainer();
    dynamicContainer->push_back(value);
}

void printDynamicContainer() {
    std::vector<int>* dynamicContainer = createDynamicContainer();
    for (int value : *dynamicContainer) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    addToDynamicContainer(1);
    addToDynamicContainer(2);
    addToDynamicContainer(3);

    printDynamicContainer();

    return 0;
}
```

在这个例子中，`createDynamicContainer()` 函数是一个工厂函数，用于创建动态容器。它使用函数局部静态指针 `dynamicContainer` 来持有容器的地址，并在第一次调用时进行初始化。后续调用将直接返回指向容器的指针。

`addToDynamicContainer()` 函数用于向动态容器中添加元素。它通过调用 `createDynamicContainer()` 获取容器的指针，并使用指针进行元素的添加操作。

`printDynamicContainer()` 函数用于打印动态容器中的元素。它也通过调用 `createDynamicContainer()` 获取容器的指针，并使用指针进行元素的遍历和打印操作。

在 `main()` 函数中，我们演示了向动态容器中添加元素，并打印容器中的所有元素。

通过使用函数局部静态指针 `dynamicContainer`，我们可以在程序的整个生命周期内保持容器的存在，而不需要手动删除它。这对于需要在多个函数之间共享容器，并且希望容器在程序的整个执行过程中保持不变的情况非常有用


## threadlocal 变量相关

`thread_local`对象针对每个线程都有一个实例

定义成thread_local的变量：
- 命名空间下的全局变量
- 类的`static` 变量
- 本地资源

优点：
- 避免资源竞争

缺点：
- 访问`thread_local`变量可能会触发不可预测且无法控制的其他代码的执行；
- 占用的内存随线程数量的增加而增加
- 普通类成员无法定义成 `thread local`；
- 对于同一个线程内该雷多的多个对象都会共享一个变量实例，并且只会在第一次执行的时候初始化


### 函数中的`thread_local`没有安全问题，可以不受限制访问


当一个函数内部使用 `thread_local` 变量时，每个线程都会有自己的变量副本，并且可以在不受限制的情况下访问和修改该变量。以下是一个示例：

```cpp
#include <iostream>
#include <thread>

void threadFunction()
{
    thread_local int threadVar = 0; // 使用 thread_local 声明一个线程局部变量
    
    // 在每个线程中递增变量的值
    for (int i = 0; i < 5; ++i)
    {
        threadVar++; // 修改线程局部变量的值
        std::cout << "Thread ID: " << std::this_thread::get_id() << ", ThreadVar: " << threadVar << std::endl;
    }
}

int main()
{
    std::thread t1(threadFunction);
    std::thread t2(threadFunction);

    t1.join();
    t2.join();

    return 0;
}
```

在上面的示例中，我们创建了两个线程 `t1` 和 `t2`，它们都调用了 `threadFunction` 函数。在 `threadFunction` 函数内部，我们使用 `thread_local` 关键字声明了一个整数变量 `threadVar`。每个线程都会有自己的 `threadVar` 变量副本，并且可以自由地访问和修改它。

运行该程序时，您会看到每个线程的输出都显示了线程ID和相应的 `threadVar` 值。由于每个线程都有自己的变量副本，它们可以独立地递增变量的值，而不会相互干扰。

### 类或者命名空间范围内的`thread_local`变量必须用编译时候的常量进行初始化，也就是说他们没有动态初始化

## 如果有其他方案可以实现类似thread_local功能，优先使用其他的