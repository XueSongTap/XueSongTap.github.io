---
layout: article
title: JavaScript与C++无缝集成：以stdlib.js的Node-API为例
tags: cpp node.js  javascript node-api
---

## 0 背景介绍

stdlib.js 是一个综合性的 JavaScript 库，提供了多种数学和统计功能，提供了js调用c/cpp的实现


## 1 重构issue

在 [Issue #1528](https://github.com/stdlib-js/stdlib/issues/1528) 中，提出了对 `@stdlib/blas/ext/base/snansum` 函数进行重构的建议。主要目标包括：

- 从 C++ 插件接口迁移到 C 插件接口
- 统一代码风格
- 简化实现方式
- 遵循项目约定（详见 Issue #788）


[PR #2227](https://github.com/stdlib-js/stdlib/pull/2227)

已经做了合入


## 2 基于Node-API的实现

核心实现在 lib/node_modules/@stdlib/blas/ext/base/snansum/src/addon.c
```cpp
#include "stdlib/blas/ext/base/snansum.h"
#include <node_api.h>
#include "stdlib/napi/export.h"
#include "stdlib/napi/argv.h"
#include "stdlib/napi/argv_double.h"
#include "stdlib/napi/argv_int64.h"
#include "stdlib/napi/argv_strided_float32array.h"
#include "stdlib/napi/create_double.h"

/**
* Add-on namespace.
*/
static napi_value addon(napi_env env, napi_callback_info info) {
    // 1. 参数获取和验证
    STDLIB_NAPI_ARGV(env, info, argv, argc, 3);
    
    // 2. 类型转换
    STDLIB_NAPI_ARGV_INT64(env, N, argv, 0);
    STDLIB_NAPI_ARGV_INT64(env, stride, argv, 2);
    
    // 3. 数组处理
    STDLIB_NAPI_ARGV_STRIDED_FLOAT32ARRAY(env, X, N, stride, argv, 1);
    
    // 4. 调用核心函数并返回结果
    STDLIB_NAPI_CREATE_DOUBLE(env, 
        stdlib_strided_snansum(N, X, stride), v);
    return v;
}

STDLIB_NAPI_MODULE_EXPORT_FCN( addon )
```

### 2.1 调用顺序

```
JavaScript 代码
↓
Node.js
↓
Node-API 层
↓
C/C++ 代码
```
### 2.2 关键组件

1. **napi_env**
```c
napi_value addon(napi_env env, napi_callback_info info) {
    // env 包含了 Node-API 调用的上下文信息
}
```
代表了 JavaScript 运行时环境
用于所有 Node-API 调用
管理异常处理和资源清理

2. napi_value
```
napi_value result;
napi_create_double(env, 42.0, &result);
```


### 2.3 数据类型转换

js 数据和 c/cpp 数据类型不太一样，传入的时候都需要做好转换


以下是示例
#### 2.3.1 基本类型转换

```c
// JavaScript Number 转 C double
double value;
napi_get_value_double(env, args[0], &value);

// C double 转 JavaScript Number
napi_value result;
napi_create_double(env, value, &result);
```
#### 2.3.2 复杂类型转换


##### 2.3.2.1 数组转换:


```c
// JavaScript Array 转 C 数组
float* data;
size_t length;
napi_get_arraybuffer_info(env, args[0], (void**)&data, &length);

// C 数组转 JavaScript TypedArray
napi_value array;
napi_create_typedarray(env, napi_float32_array, length, buffer, 0, &array);
```

##### 2.3.2.2 对象转换

```c
// 创建 JavaScript 对象
napi_value obj;
napi_create_object(env, &obj);

// 设置对象属性
napi_set_named_property(env, obj, "key", value);
```

#### TODO

还有一些 内存管理，引用计数、线程安全、错误处理，这次pr 没有涉及到，待深入探索


## 参考：

https://nodejs.github.io/node-addon-examples/about/what/


https://stdlib.io/