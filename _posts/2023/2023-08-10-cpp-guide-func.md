---
layout: article
title: C++函数设计最佳实践
tags: c++ guide function
---


## 参数顺序相关

### 函数的参数顺序为： input ->  input/output -> output


### 函数参数不超过5个

两种情况可能导致参数过多：
1. 函数功能过于庞大， 可以拆分成多个小函数，方便单元测试和代码调试
2. 隐藏了一个潜在的类，包含了这些参数


### c++函数的输出一般通过返回值 提供，有时候通过输出参数提供

倾向于使用返回值。可读性好，否则的话按照引用返回，避免返回指针，除非他可以为null



1. 第一个方法 `MakePizza(Pizza& pizza)` 通过引用参数传递 `pizza` 对象，并返回一个布尔值来表示制作披萨的结果。如果 `busy` 为假，则将 `pizza` 对象的属性设置为指定的值，并返回 `true` 表示制作成功。否则，返回 `false` 表示制作失败。

这种方法的问题在于，如果制作失败，仍然需要构造一个 `Pizza` 对象并传递给 `pizza` 引用参数。这可能会导致不必要的对象构造开销。

2. 第二个方法 `std::unique_ptr<Pizza> MakePizza()` 返回一个 `std::unique_ptr` 智能指针，它指向通过堆分配创建的 `Pizza` 对象。如果制作成功，则返回指向新创建的 `Pizza` 对象的指针。如果制作失败，则返回一个空指针。

这种方法避免了制作失败时的对象构造开销，但仍然需要进行堆分配和使用智能指针来管理内存。这可能会引入额外的开销和复杂性。

3. 第三个方法 `std::optional<Pizza> MakePizza()` 返回一个 `std::optional` 对象，其中包含一个 `Pizza` 对象。如果制作成功，则返回一个包含设置好属性的 `Pizza` 对象的 `std::optional`。如果制作失败，则返回一个空的 `std::optional`。

这种方法是推荐的，因为它避免了制作失败时的对象构造和堆分配开销。通过返回一个 `std::optional` 对象，我们可以明确地表示制作是否成功，并且不需要额外的内存管理。

总的来说，第三种方法是最好的选择，因为它在制作失败时避免了对象构造和堆分配开销，并提供了明确的制作结果表示。

```C++
// Not recommand - if fails, we still need construct pizza
//通过引用参数传递 `pizza` 对象，并返回一个布尔值来表示制作披萨的结果。
//如果 `busy` 为假，则将 `pizza` 对象的属性设置为指定的值，并返回 `true` 表示制作成功。否则，返回 `false` 表示制作失败。
//如果制作失败，仍然需要构造一个 Pizza 对象并传递给 pizza 引用参数。这可能会导致不必要的对象构造开销。
bool MakePizza(Pizza& pizza) {
  if (!busy) {
    pizza.size = 6;
    pizza.flavor = "Spicy";
    return true;
  } else {
    return false;
  }
}

// Better, no constructor when failed, but heap allocation if success
// std::unique_ptr<Pizza> MakePizza() 返回一个 std::unique_ptr 智能指针，它指向通过堆分配创建的 Pizza 对象。如果制作成功，则返回指向新创建的 Pizza 对象的指针。如果制作失败，则返回一个空指针。
//这种方法避免了制作失败时的对象构造开销，但仍然需要进行堆分配和使用智能指针来管理内存。这可能会引入额外的开销和复杂性。
std::unique_ptr<Pizza> MakePizza();

// Recommand - if failed, no constructor and heap allocation overhead
// std::optional<Pizza> MakePizza() 返回一个 std::optional 对象，其中包含一个 Pizza 对象。如果制作成功，则返回一个包含设置好属性的 Pizza 对象的 std::optional。如果制作失败，则返回一个空的 std::optional。
//这种方法是推荐的，因为它避免了制作失败时的对象构造和堆分配开销。通过返回一个 std::optional 对象，我们可以明确地表示制作是否成功，并且不需要额外的内存管理。
std::optional<Pizza> MakePizza() {
  if (!busy) {
    return Pizza{6, "Spicy"};   
  } else {
    return {}
  }
}
```
### 仅作为输入的参数通常应该是值或者常量引用，仅作为输出或者输入/输出的参数应该是引用（不能为null）

`std::optional（since c++17）` 作为可选值，或者使用常量指针
```cpp
// Exception in setter/getter/constructor, if removable, pass by value
// 如果在设置器（setter）、获取器（getter）或构造函数中可能抛出异常，并且这些异常可以被处理并从调用方移除，
// 则可以通过值传递参数。这意味着在函数内部使用传递的参数的副本。
//例如，void Person::Person(std::string name) 构造函数接受一个 std::string 类型的参数，并将其移动到 name_ 成员变量中。
void Person::Person(std::string name) : name_(std::move(name)) {}
```

### 避免定义生命周期长于函数调用的常量引用参数

因为常量引用参数可以绑定临时变量

可以找到消除生存期需求的方法（例如复制参数） 或者用const指针传递


在 StringHolder 类的构造函数中，使用了一个常量引用参数 const string& val。这意味着该参数可以绑定到一个临时变量，例如 "abc"s。然而，问题在于，构造函数将该引用存储为成员变量 val_，并且该成员变量的生命周期长于构造函数的调用。

在示例中，当构造 StringHolder 对象 holder 时，使用了一个临时的 string 对象 "abc"s。然而，一旦构造函数完成，临时对象就会被销毁，而成员变量 val_ 将成为悬空引用（dangling reference）。

当调用 holder.get() 时，尝试返回成员变量 val_ 的引用，但该引用已经无效，因为它指向一个已销毁的对象。这将导致未定义行为（Undefined Behavior）。
```C++
// Don't do this 错误示例
class StringHolder {
 public:
  // The input `val` must live as long as this object, not just the call to
  // this c'tor
  StringHolder(const string& val) : val_(val) {}

  const string& get() { return val_; }

 private:
  const string& val_;
};

StringHolder holder("abc"s);
std::cout << holder.get();  // boom, UB. The string temporary has already
                            // been destroyed, the reference is dangling.
```
如何解决：

1. 复制参数`StringHolder(string val) : val_(val) {}`
2. 使用 const 指针：将构造函数的参数改为 const string* val，并在构造函数中对指针进行适当的处理。这样，构造函数将接受一个指向 const string 的指针，并负责处理指针的生命周期。`StringHolder(const string* val) : val_(val ? *val : "") {}`


## 简短函数相关

要编写简短，凝练的函数

### 编写函数的时候，要综合考虑函数的 圈复杂度，可执行行数，函数调用数，嵌套深度等指标

长函数有时候合适的，但是超过70行，可以优先考虑分解，长函数可能会让难以修改

参考意见：

| **Item** | **Desc**       | **Criteria** |
| -------- | -------------- | ------------ |
| STCDN    | 注释与代码比率 | > 0.2        |
| STCYC    | 圈复杂度       | < 15         |
| STXLN    | 可执行行数     | < 70         |
| STSUB    | 函数调用数     | < 10         |
| STMIF    | 嵌套深度       | < 5          |
| STPTH    | 估计静态路径数 | < 250        |


## 引用参数相关

### 所有按照引用传递的输入参数，如果是只读`const`

不会改变的参数，用const修饰，保证函数声明的准确性

## 函数重载相关

### 使用函数重载，必须易懂，知道究竟调用的是哪一个


## 缺省参数相关

### 只允许在非虚函数中使用缺省参数（缺省参数是在编译的时候绑定的），并且必须保证缺省参数的值始终一致

```C++
// Don't use default parameter on virtual function 虚函数不要用缺省参数
class Foo {
 public:
  virtual void doIt(int x = 1) { std::cout << "Doing foo: " << x << std::endl; }
};

class Bar : public Foo {
 public:
  virtual void doIt(int x = 0) { std::cout << "Doing bar " << x << std::endl; }
};

int main() {
  Bar bar;
  Foo* fooPtr = &bar;
  fooPtr->doIt();  // Doing bar: 1
}
```

## 函数返回类型后置语法相关

### 只有在常规写法 （返回类型前置）不便于书写或者不便于阅读时候使用返回类型后置的语法

## goto 相关

### 尽量避免使用goto
goto statement会破坏程序的结构，debug难度增加
