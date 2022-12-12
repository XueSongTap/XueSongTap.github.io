---
layout: articles
title: cpp 面试问题
tags: cpp interview
---

## cpp 中struct 和class 区别

### struct 来自于c
首先struc是c里面的保留的， 定义struct之后就是定义了个一种数据类型，

在c中是不能定义成员函数

c++的struct得到了扩充

1. struct 可以包括成员函数了

2. struct 可以实现继承

3. struct 可以实现多态

### cpp中的区别

1. 默认的继承访问权：class 默认是private， struct默认是public

2. 默认的访问权限：class 默认是private， struct默认是public

3. class 可以用于定义模板参数

### 具体使用场景

如果只包含一些变量的结构，或者只包含一些plain old data的时候，算用struct， 比如定义向量

```cpp
struct Vec2{
    float x, y;
    void Add(const Vec2& other){
        x += other.x;
        y += other.y;
    }
};
```

实现复杂功能的话，选用class


## static 在类或者结构体外部的使用

在类/结构体外部的函数使用static：

意味着定义的函数和变量只对他生命所在的cpp单元是可见的


例如这样
```cpp
//a.cpp中
int s_var = 5;

//main.cpp中
int s_var = 10;
int main(){
    std::cout << s_var << std::endl;
}
```
编译错误
```shell
acs@VM-16-15-ubuntu:~/cpp_cherno/21$ g++ main.cpp a.cpp 
/usr/bin/ld: /tmp/ccfo6KGd.o:(.data+0x0): multiple definition of `s_var'; /tmp/cczuL69c.o:(.data+0x0): first defined here
collect2: error: ld returned 1 exit status
```

用static声明：
```cpp
//a.cpp中
static int s_var = 5;

//main.cpp中
int s_var = 10;
int main(){
    std::cout << s_var << std::endl;
}

//输出10
```

用extern声明：
```cpp
//a.cpp中
int s_var = 5;

//main.cpp中
extern int s_var;
int main(){
    std::cout << s_var << std::endl;
}

//输出5
```


## C++类和结构体中的static

静态方法不能访问非静态变量

静态方法没有类实例

本质上你在类里写的每个非静态方法都会获得当前的类实例作为参数（this指针）

静态变量在编译的时候存储在静态存储的地方，也就是说定义过程在编译的时候完成，所以一定要在类外进行定义

静态成员变量是所有实例共享的


报错，static变量x没有在类外定义
```cpp
#include <iostream>
using namespace std;

struct Entity
{
    static int x;

    void print()
    {
        cout << x << endl;
    }
};

int main()
{
    Entity e1;
    e1.x = 1;
    cin.get();
}
```

进行类外定义：
```cpp
#include <iostream>
using namespace std;

struct Entity
{
    static int x;

    void print()
    {
        cout << x << endl;
    }
};

int Entity::x;

int main()
{
    Entity e1;
    e1.x = 1;
    cin.get();
}
```


实例对象是共享的：
```cpp
#include <iostream>
using namespace std;

struct Entity
{
    static int x;

    void print()
    {
        cout << x << endl;
    }
};

int Entity::x;

int main()
{
    Entity e1;
    e1.x = 1;

    Entity e2;
    e2.x = 2;

    e1.print();
    e2.print();

    cin.get();
}
//输出
2
2
```

静态方法也要在类外声明，并且传入参数

```cpp
#include <iostream>
using namespace std;

struct Entity
{
    int x;

    static void print()
    {
        cout << x << endl;  // 报错，不能访问到非静态变量x
    }
};
//在类外面写一个print()函数
static void print(Entity e)
{
 cout << x << endl;  // 可以运行
}

int main()
{
    Entity e1;
    e1.x = 1;

    Entity e2;
    e2.x = 2;

    e1.print();
    e2.print();

    cin.get();
}
```


## 局部local static

局部作用域中static可以声明一个变量，重点是变量的生命周期和作用域

local static 的变量生命周期贯穿整个程序，但是他的作用域仅限当前

```cpp
#include <iostream>

using namespace std;

void Function() {
    static int i = 0;
    i ++;
    std::cout << i << std:: endl;
}

int main() {
    Function();
    Function();
    Function();
}

//输出 1 2 3
```

效果类似于在函数外面进行声明， 但是这种情况下变量i任何时候都可以直接访问、

```cpp
#include <iostream>
using namespace std;

void Function() {
    i ++;
    std::cout << i << std:: endl;
}

int main() {
    Function();
    //i = 10;
    Function();
    Function();
}
```

## 虚函数

虚函数可以在子类中重写方法

步骤：

1. 定义基类, 定义`virtual`函数

2. 定义派生类， 派生类实现 `virtual`函数

3. 声明基类指针，指向派生类，调用`virtual`函数，虽然是基类指针，但是调用到的是派生类实现的函数


```cpp
//基类
class Entity
{
public:
    virtual std::string GetName() {return "Entity";} //第一步，定义基类，声明基类函数为 virtual的。
};

//派生类
class Player : public Entity
{
private: 
    std::string m_Name; 
public: 
    Player(const std::string& name):m_Name (name) {} 
    //第二步，定义派生类(继承基类)，派生类实现了定义在基类的 virtual 函数。
    std::string GetName() override {return m_Name;}  //C++11新标准允许给被重写的函数用"override"关键字标记,增强代码可读性。
};

void printName(Entity* entity){
    std::cout << entity -> GetName() << std::endl;
}

int main(){
    Entity* e = new Entity();
    printName(e); 
    //第三步，声明基类指针，并指向派生类，调用`virtual`函数，此时虽然是基类指针，但调用的是派生类实现的基类virtual函数。Entity* p = new Player("cherno");也可以
    Player* p = new Player("cherno"); 
    printName(p); 
}
//输出
//Entity
//cherno
```


## cpp纯虚函数

### 作用

1. 纯虚函数使得派生类必须实现基类的虚函数

2. 含有纯虚函数的类称为抽象类，不能直接生成对象

3. 接口的作用

### 声明方法：

基类纯虚函数方法后面 `=0`

```cpp
virtual void funtion()=0;
virtual std::string GetName() = 0;
```

```cpp

#include <iostream>

using namespace std;

//接口
class Printable {
public:
    virtual std::string GetClassName() = 0;
};

class Entity: public Printable{
public:
    virtual std::string GetName() {return "Entity";}
    std::string GetClassName() override {return "Entity";}
};

class Player: public Entity{
private:
    std::string m_Name;
public:
    Player(const std::string& name):m_Name(name) {}
    std::string GetName() override{return m_Name;}
    std::string GetClassName() override {return "Player";}
};

void Print(Printable* obj) {
    std::cout << obj -> GetClassName() << std::endl;
}


int main() {
    Entity* e = new Entity();
    Player* p = new Player("cherno");

    Print(e);
    Print(p);
}

//Entity
//Player
```

## cpp const


### const 作用
const 首先作用在左边，如果左边没有东西，那就作用在右边

也可以说他是伪关键字，他是一个承诺

### const指针

