---
layout: articles
title: cpp17 新特性string_view 
tags: cpp17 新特性 string string_view 浅拷贝 字符串
---

### 引入 

单独的头文件`<string_view>`


### 实现
只读的，指针+ 长度, 不拥有内容

immutable, light weight (pointer + size), does NOT own the content.

```cpp
struct string_view {
    const char* data;
    size_t length;
}
```
[libstdc++-v3/include/std/string_view](https://github.com/gcc-mirror/gcc/blob/fab08d12b40ad637c5a4ce8e026fb43cd3f0fad1/libstdc%2B%2B-v3/include/std/string_view#L4)

### 使用
most interfaces remain the same as std::string, e.g. [], begin, at, back, data 大部分操作和std::string  一致 

**findsubstr O(1)** substr 操作不需要深拷贝

```cpp
//例如递归操作的时候，之前需要传入常引用，再传入l, r 避免拷贝substr 的情况
void solve(const std::string& str, int l, int r) {
    // solve(str, l, r)
    //do something

}
// 用string_view 可以直接取substr，不用担心创建额外拷贝
void solve(std::string_view sv) {
    // solve(sv.substr(s, n))
    // do something
}
```

remove_prefx O(1)// pop_front(n) 也是不需要修改内存

remove_suffx O(1)// pop_back(n)

### 限制

immutable 的， 无法追加字符或者字符串拼接的方式

声明周期问题，string_view 的生命周期内，确保原数据是不被修改的, not mutable


```cpp
std::string s3{"abcdefg"};
std::string_view sv3 = s3;
s3 = "123" // change data
std::cout << sv3 << std::endl; //undefined
```

输出 ([b站视频](https://www.bilibili.com/video/BV1iV411C769/))：
```
123<0x00>efg
```
或者是 ([自己试验](https://godbolt.org/z/eKaTqdcaq) 可能是编译器版本区别 gcc12.2):
```
123dfg
```

s3 内存没有销毁, sv3 拿的是原始s3的长度，就全部都打印出来了

### 参考文献


https://zh.cppreference.com/w/cpp/string/basic_string_view