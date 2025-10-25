---
layout: article
title: 全面整理C++性能优化技巧，包括编译优化、代码规范、内存管理等关键实践要点
tags: c++ 性能
---

## cmakelist编译的时候打开wall

对一些语法进行检查 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")


## 类的初始化列表要按照类定义的顺序执行初始化

## 有返回值的函数一定要返回

## size_t和int类型的比较消除

static_cast<int>(XX)


## 在h文件中，用前向声明替换 头文件 对象定义成指针或引用，在cpp包含头文件

前向声明的好处 1 加快编译的速度 2 编译的时候避免互相依赖，减少头文件的暴露，在做接口实现的时候是必须的


##  指针统一用智能指针，不用裸指针进行new delete操作


## 提供单例的宏对象展开


```cpp
#define DECLARE_SINGLETON(classname)                                      \
 public:                                                                  \
  static classname *Instance(bool create_if_needed = true) {              \
    static classname *instance = nullptr;                                 \
    if (!instance && create_if_needed) {                                  \
      static std::once_flag flag;                                         \
      std::call_once(flag,                                                \
                     [&] { instance = new (std::nothrow) classname(); }); \
    }                                                                     \
    return instance;                                                      \
  }                                                                       \
                                                                          \
  static void CleanUp() {                                                 \
    auto instance = Instance(false);                                      \
    if (instance != nullptr) {                                            \
      CallShutdown(instance);                                             \
    }                                                                     \
  }                                                                       \
                                                                          \
 private:                                                                 \
  classname();                                                            \
  DISALLOW_COPY_AND_ASSIGN(classname)
```

## 工厂函数用模板管理，返回对象用智能指针管理


## 缓冲队列无锁化


## 如果定义的是类的指针，没赋值，默认是nullptr，必须指定一个地址，所以这样的定义是否合理