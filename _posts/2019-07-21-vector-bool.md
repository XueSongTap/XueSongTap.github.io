---
layout: articles
title: stl "vector<bool>" 特殊优化
tags: c++ stl vector bool 优化 位运算
---


### vector<bool> 特性
std::vector<bool> 是 std::vector 对类型 bool 为空间提效的特化。

std::vector<bool> 中对空间提效的行为（以及它是否有优化）是实现定义的。一种潜在优化涉及到 vector 的元素联合，使得每个元素占用一个单独的位，而非 sizeof(bool) 字节。

std::vector<bool> 表现类似 std::vector ，但为节省空间，它：

不必作为连续数组存储元素

暴露类 std::vector<bool>::reference 为访问单个位的方法。尤其是，此类型的类为 operator[] 以值返回。

不使用 std::allocator_traits::construct 构造位值。

不保证同一容器中的不同元素能由不同线程同时修改。
[api文档](https://gcc.gnu.org/onlinedocs/gcc-4.6.2/libstdc++/api/a00740.html)


### 底层实现

[单独的libstdc++-v3/include/bits/stl_bvector.h实现](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/stl_bvector.h)

主要是存了bit pointer
![stl_bvector](/img/190721/vector_bool.png)

### vector<bool> 的空间优化

#### vector<bool> 的capacity
vector<bool> 如果是空的：

```cpp
std::vector<bool> data;
std::cout << data.size() << " " << data.capacity() << std::endl;
```
```
0 0
```

如果存入1个数字
```cpp
std::vector<bool> data;
data.push_back(1);
std::cout << data.size() << " " << data.capacity() << std::endl;
```
```
1 64
```
存入3个数字：

```cpp
std::vector<bool> data;
data.push_back(1);
data.push_back(2);
data.push_back(3);
std::cout << data.size() << " " << data.capacity() << std::endl;
```
```
3 64
```

可以看到，从存入第一个数字开始，capacity直接到了64

这是因为vector<bool> 用unsigned long 8字节按位存储bit映射成bool，也就是说分配一个
unsigned long 够64 个bool 用，所以第一次分配capacity直接到64
#### vector<int> 的capacity 变化
相反如果是vector<int> 情况下
```cpp
std::vector<int> data;
data.push_back(1);
std::cout << data.size() << " " << data.capacity() << std::endl;
```
```
1 1
```

存入3个int
```cpp
std::vector<bool> data;
data.push_back(1);
data.push_back(2);
data.push_back(3);
std::cout << data.size() << " " << data.capacity() << std::endl;
```
```
3 4
```

### vector<bool>取值
```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<bool> data;
    data.push_back(1);
    data.push_back(2);
    data.push_back(3);

    // auto &front = data.front();
    std::cout << data.size() << " " << data.capacity() << " " << data.front() <<std::endl;
}
```

输出data.front() 是正常的， 但是如果打开注释，取左值引用，编译器会报错, vector<int> 没有这样的报错

```shell
<source>: In function 'int main()':
<source>:10:29: error: cannot bind non-const lvalue reference of type 'std::_Bit_reference&' to an rvalue of type 'std::vector<bool>::reference'
   10 |     auto &front = data.front();
      |     
```
所以用operator[]的时候，正常容器返回的应该是一个对应元素的引用，但是对于vector< bool>实际上访问的是一个”proxy reference”而不是一个”true reference”，返回的是”std::vector< bool>:reference”类型的对象。

因为bit你是无法访问到地址的


#### 正确取值方式
```cpp
auto d = data[0];
```

修改d会影响到data

```cpp
bool a = data[0];
```

过程中其实包含了bit转换到bool的过程

#### std::_Bit_reference
下面来看看怎么将一个bool类型变量映射到_Bit_type中的一个bit，这由类  std::_Bit_reference 实现的。

类 std::_Bit_reference  是 std::vector<bool> 中的基本存储单位。

比如，std::vector<bool>的 operator[]函数返回值类型就是std::_Bit_reference，而不是 bool 类型  

实现如下：
```cpp
 struct _Bit_reference
  {
    _Bit_type* _M_p;
    _Bit_type  _M_mask;

    _Bit_reference(_Bit_type *__x, _Bit_type __y)
    : _M_p(__x), _M_mask(__y) {}

    _Bit_reference() noexcept : _M_p(0), _M_mask(0) {}
    _Bit_reference(const _Bit_reference &) = default;
      
    ///@brief 隐式转成 bool
    ///       bool state = vb[1]; 会触发此函数
    operator bool() const noexcept
    { return !!(*_M_p & _M_mask); }

    ///@brief 将 _M_p 的 _M_mask 位，设置为 __x 状态
    ///        vb[1] = true; 会触发此函数
    _Bit_reference& operator=(bool __x) noexcept
    {
      if (__x)
        *_M_p |= _M_mask;  // 1
      else
        *_M_p &= ~_M_mask;
      return *this;
    }
      
    // @brief 这个函数实际上调用了：
    //   1. 先调用了 operator bool() const noexcept
    //   2. 在调用了 _Bit_reference& operator=(bool __x) noexcept
    _Bit_reference& operator=(const _Bit_reference &__x) noexcept
    { return *this = bool(__x); }

    bool operator==(const _Bit_reference &__x) const
    { return bool(*this) == bool(__x); }

    bool operator<(const _Bit_reference &__x) const
    { return !bool(*this) && bool(__x); }

    void flip() noexcept
    { *_M_p ^= _M_mask; }
  };
```

### 参考

https://zh.cppreference.com/w/cpp/container/vector_bool

https://mp.weixin.qq.com/s/pkHfGjw8by8g1sbWhWKysw

https://blog.0xzhang.com/posts/%E8%B8%A9%E5%9D%91vector.html

https://www.youtube.com/watch?v=OP9IDIeicZE

https://blog.csdn.net/haolexiao/article/details/56837445