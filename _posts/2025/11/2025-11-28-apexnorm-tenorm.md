---
layout: article
title: ApexRMSNorm vs TENorm：bf16 大 weight 精度坑
tags: RMSNorm Precision
---

bf16 下使用 **ApexRMSNorm**，一旦 weight/gamma/scale 放大，输出标准差看似稳定，但均值会随机机漂移；换成 Transformer Engine 的 **TENorm** 就恢复稳定。
## 1 RMSNorm

RMSNorm 只做均方根归一化，不减均值、也没有 `beta` 偏置：

$$
y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

- `d` 为 hidden size，`γ` 逐通道缩放，通常初始化为 1。
- 由于不减均值，输出是否“居零”高度依赖舍入误差；`γ` 越大，误差越容易被放大。

## 2 现象

在 bf16 下把 `γ` 放大（例如 >3.5，或按长度线性增大）：

- **ApexRMSNorm**：重复前向，同一批输入标准差稳定，均值漂移
- **TENorm**：均值与标准差都稳定，和 torch 原生fp32 一致


## 3 代码路径对比

### ApexRMSNorm（BF16）

归一化值先被截成输出 dtype（bf16）再乘权重：

```
ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
```

`c_invvar * curr` 先量化一次到 bf16，再与 bf16 `gamma` 相乘并再次量化。bf16 只有 7 位尾数，放大后的 `gamma` 会把两次量化误差按比例放大，于是均值漂移；标准差仍稳定，因为 `invvar` 计算在 fp32。


参考 


https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu#L317
```c
template <typename T, typename U, typename V>
__device__ void cuApplyLayerNorm_(V* __restrict__ output_vals, U* __restrict__ mean, U* __restrict__ invvar,
                                  const T* __restrict__ vals, const int n1, const int n2, const U epsilon,
                                  const V* __restrict__ gamma, const V* __restrict__ beta, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, rms_only);

    const T* lvals = vals + i1 * n2;
    V* ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != nullptr && (beta != nullptr || rms_only)) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}
```


### Transformer Engine TENorm

归一化和缩放都在 `compute_t`（静态断言 float）里完成，只在写出时做一次 dtype 转换：

```
compute_t temp_output = gamma * y_ij;
z[...] = output_t(temp_output);
```

没有双重 bf16 量化，同样的输入/权重输出均值稳定。参考 



https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_kernels.cuh#L110
```c
        compute_t g_ij = gamma[it].data.elt[jt];
        if (params.zero_centered_gamma) {
          g_ij += 1;
        }
        compute_t temp_output = g_ij * y_ij;

        if (requires_amax) {
          __builtin_assume(amax >= 0);
          if (params.fp8_out) {
            // For fp8_out, keep amax on pre-scale compute_t
            amax = fmaxf(amax, fabsf(temp_output));
          } else {
            // Otherwise compute amax on the value converted to output_t (e.g., bf16)
            output_t out_t_val = output_t(temp_output);
            amax = fmaxf(amax, fabsf(compute_t(out_t_val)));
          }
        }
        if (params.fp8_out) {
          temp_output = temp_output * scale;
        }

        z[it].data.elt[jt] = output_t(temp_output);
```



## 4 心得

- bf16 归一化对权重尺度很敏感，缩放最好留在 fp32 路径，避免把舍入误差放大到可见水平。  
- TENorm 的实现更谨慎，适合在大 gamma、长序列、高 hidden 的极端形态下更稳定
