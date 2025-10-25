---
layout: article
title: C++ STL unordered_map的扩容原理
tags: cpp stl unordered_map hash
---

## gcc的扩容机制


https://github.com/gcc-mirror/gcc/blob/master/libstdc++-v3/src/c++11/hashtable_c++0x.cc#L104

gcc 的做法是按 growth_factor (=2) 来扩容，

```cpp
    if (__n_elt + __n_ins > _M_next_resize)
      {
	// If _M_next_resize is 0 it means that we have nothing allocated so
	// far and that we start inserting elements. In this case we start
	// with an initial bucket size of 11.
	double __min_bkts
	  = std::max<std::size_t>(__n_elt + __n_ins, _M_next_resize ? 0 : 11)
	  / (double)_M_max_load_factor;
	if (__min_bkts >= __n_bkt)
	  return { true,
	    _M_next_bkt(std::max<std::size_t>(__builtin_floor(__min_bkts) + 1,
					      __n_bkt * _S_growth_factor)) };

	_M_next_resize
	  = __builtin_floor(__n_bkt * (double)_M_max_load_factor);
	return { false, 0 };
      }
    else
      return { false, 0 };

```
## 参考

https://www.cnblogs.com/lygin/p/16572018.html

https://www.zhihu.com/question/60570937