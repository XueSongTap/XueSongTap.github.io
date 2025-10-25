---
layout: article
title: Eigen与OpenCV矩阵运算性能对比与最佳实践
tags: eigen opencv
---

# Eigen Vs CV

## 矩阵基本运算

### 差异

OpenCV在x86 & arm上Mat矩阵的用float进行存储，但是计算时高于float，计算，再截断
VS
Eigen使用Matrix定义是指定的类型进行存储和计算

1. Eigen MatrixXf随机矩阵乘法与c++实现float矩阵乘法京都一直
2. OpenCV 随机矩阵乘法与C++ & Eigen 不一致
3. OpenCV与C++ float 强转成double 相乘再强转成foat 京都一直
4. OpenCV与Eigen MatrixXf cast到double进行乘法计算后强转成float精度移植

### 策略

1. Eigen使用Float进行计算（长期推荐），预计计算效率会比OpenCV计算高（当前评测中性能提升的来源之一，同时能够比较充分的利用Neon向量处理加速）

2. 将所有Eigen Matrix基本运算，使用Eigen MatrixXf cast到double进行乘法计算后强转成float精度策略，模拟OpenCV的计算过程。最大程度减少基本运算带来的数值差异，但是会影响计算速度。Neon向量运算可能会慢1倍（TBD）

3. 将涉及的Eigen Matrix基本运算，转换回OpenCV进行计算，使用OpenCV的算子计算后转回Eigen数据结构。最大程度消除数值差异（CR中使用本策略）

4. Eigen使用Double计算，输出与Baseline也会存在较大差别，会导致，预计计算与存储消耗均会一定程度提升。UKF及其派生类内存占用翻倍（不推荐）实验发现，如果Eigen FixValue UKF使用Double计算，也会导致结果产生一定差异（相对明显）。


## Cholesky Decomposition数值
### 差异
Baseline中使用OpenCV提供的cv::hal::Cholesky32f(L, lstep, lsize, NULL, 0, 0)做Cholesky分解

Eigen中使用也提供了Cholesky LLT实现，但是两者实现数值对比有差异（单元素矩阵就会有差异）。


### 策略

1. 使用OpenCV中提供的Cholesky Decomposition算法，会存在Eigen-CV矩阵转换。可以消除Eigen-CV数值差异，会引入不必要转换开销，且CV LLT计算较Eigen慢（CR中使用本策略）

2. 使用Eigen提供的LLT算法。在FixValue UKF场景比较，Eigen LLT分解数值精度在一些场景比OpenCV准确。（比较方法：将LL*相乘，与原阵作差，长期推荐）


## 矩阵逆运算数值

### 差异

OpenCV当前使用SVD（奇异值分解算法）做矩阵逆运算，支持对非方阵进行广义逆运算，对矩阵形状没有要求。
Eigen没有基于奇异值分解矩阵逆运算实现。仅提供默认inverse进行逆矩阵运算。（UKF开源实现直接使用了inverse）
### 策略


1. 直接使用Eigen提供的inverse函数进行矩阵逆运算。预计性能较好，且稳定性及可靠性优秀。(长期推荐)


2. 基于Eigen SVD实现奇异值分解逆运算。预计会最大程度消除逆运算数值差异；会有额外性能开销，有可靠性&数值验证成本。（CR中使用本策略， 用SVD）


## 矩阵Gemm运算数值

### 差异

可以被写成这种类型的乘法


OpenCV在X86 & Arm上Mat矩阵是使用float进行存储的，但是计算时会使用高于float的计算单元进行计算，然后截断成float。Eigen则会实际使用Matrix定义时指定的类型进行存储与计算。

当算式中包含可以被化简成上述形式的表达式时，OpenCV在处理matExpr的时候会使用gemm运算，而不是进行单步运算。调用opencv gemm与opencv单步运算结果存在数值差异。

根据Eigen官网描述https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html，

Eigen也有gemm实现。实验发现，Eigen gemm实现与单步计算数值结果一致。
OpenCV单步计算数值与Eigen数值一致。

### 策略

1. Eigen直接使用进行计算Gemm计算（长期推荐），预计计算效率会比OpenCV计算高，但是由于开发过程中需要进行一致性比较，故没有使用此方案。（相关代码以注释形式提交，方便后续开发评测使用）


2. 将包含乘加运算的Eigen Matrix表达式，转换成Opencv进行计算，计算后结果再转回Eigen。从而消除这种基本运算带来的数值差异，从而继续推进一致性比较。（本CR采用此策略）
由于Opencv和本身单步计算存在数值差异，故没有使用Eigen模拟OpenCV的计算过程。


## 矩阵标量计算

### 差异

使用cv::sum对矩阵中元素进行求和运算(对应eigen matrix.sum())，以及矩阵对标量的除法计算（matrix/scalar），cv与Eigen存在数值差异

### 策略


使用Eigen矩阵表达式直接进行标量计算，性能较优（长期推荐）


将Eigen MatrixXf转会OpenCV进行计算，转换会有不必要内存拷贝带来性能损失，OpenCV计算数值差异较double计算的差别比Eigen计算差别大，计算完成再转成Eigen MatrixXf。（一致性版本CR中使用本策略）

## cv::scaleAdd与Eigen矩阵加法表达式
cv::scaleAdd底层调用DAXPY算法，与Eigen矩阵加法表达式存在数值差异。


使用Eigen矩阵表达式直接进行标量加法计算，性能较优（长期推荐）

将Eigen MatrixXf转会OpenCV进行计算，转换会有不必要内存拷贝带来性能损失，，计算完成再转成Eigen MatrixXf。（一致性版本CR中使用本策略）


# Eigen 性能为什么好

## 并行计算

## 多核并行
OpenMp 线性加速

## SIMD
使用向量化，在单个周期内对多组数据进行向量级并行

## Cache
最大化Cache命中率，对于单个计算获得5-10倍访存性能提升

列优先存储访问【OpenCV行优先】


## 模板表达式分解+延迟计算

对表达式进行分解，能置标记位设置标记位，最小化实际计算
```cpp
Matrix m1, m2, m3;
m3 = m1 + m2 + m3; 

m2 = m3.transponse()
m3.transponse()
m3 = m3.transponseInPla()
 
// 传统C++矩阵预算
tmp1 = m1 + m2; 
tmp2 = tmp1 + m3; 
m3 = tmp2; 
// 3个循环
// 2个临时变量
// 8*m*n次内存计算
```

```cpp
// c++运算符重载，定义加法类
Sum operator+(const Matrix& A, const Matrix& B) { return Sum(A, B); }

template class Sum {
  const type_of_A& A;
  const type_of_B& B;
};
```


## 充分利用编译时信息
TODO