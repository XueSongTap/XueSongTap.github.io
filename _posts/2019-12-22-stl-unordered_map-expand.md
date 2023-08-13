---
layout: articles
title: c++ stl unordered_map 的扩容机制
tags: c++ stl unordered_map hash 哈希 哈希冲突 扩容
---

## gcc的扩容机制


https://github.com/gcc-mirror/gcc/blob/master/libstdc++-v3/src/c++11/hashtable_c++0x.cc#L104

gcc 的做法是按 growth_factor (=2) 来扩容，
## 参考

https://www.cnblogs.com/lygin/p/16572018.html

https://www.zhihu.com/question/60570937