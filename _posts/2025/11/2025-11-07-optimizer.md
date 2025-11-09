---
layout: article
title: 优化器
tags: optimizer
---


## 1  Stochastic Gradient Descent (SGD)

### 1.1 更新公式：

$$
\theta_{t+1} = \theta_t - \eta , \nabla_\theta L(\theta_t)
$$

其中：

| 符号                          | 含义                  |
| --------------------------- | ------------------- |
| $\theta_t$                  | 第 $t$ 步的参数（weights） |
| $\eta$                      | 学习率（learning rate）  |
| $\nabla_\theta L(\theta_t)$ | 当前参数下损失函数的梯度        |

**含义：**
每一步沿梯度反方向更新参数，步长由学习率控制。

### 1.2 含动量（Momentum）的 SGD

$$
v_t = \mu v_{t-1} - \eta \nabla_\theta L(\theta_t) \
$$
 
$$
\theta_{t+1} = \theta_t + v_t
$$

其中 $\mu$ 是动量系数，通常在 0.9 左右。

动量项相当于对历史梯度做指数平滑，使更新更稳定、收敛更快


## 2 AdaGrad（Adaptive Gradient）


### 2.1 基本思想

AdaGrad 在 SGD 的基础上引入 **自适应学习率**。


它为每个参数单独维护一个历史梯度平方的累积量，并根据该量调整每个维度的步长。
参考论文：[Duchi et al., 2011, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)


### 2.2 更新公式：

$$
\begin{aligned}
g_t &= \nabla_\theta L(\theta_t) 
\end{aligned}
$$


$$
\begin{aligned}
G_t &= G_{t-1} + g_t \odot g_t
\end{aligned}
$$


$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{aligned}
$$

其中：

| 符号         | 含义                     |
| ---------- | ---------------------- |
| $g_t$      | 当前梯度                   |
| $G_t$      | 累积的历史梯度平方（逐元素）         |
| $\epsilon$ | 防止除零的微小常数（如 $10^{-8}$） |
| $\odot$    | 表示逐元素相乘或除              |


### 2.3 直观解释：

* 如果某个参数的梯度一直很大，那么 $G_t$ 会很快增大，分母变大 → 学习率自动变小；
* 如果某个参数的梯度长期很小，那么学习率保持较大
* 结果是 **学习率随参数维度自动调整**，从而加速稀疏特征的学习、抑制剧烈变化


## 3 Pytorch的实现

### 3.1 简版 SGD（无动量、无权衰减）

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super(SGD, self).__init__(params, dict(lr=lr))
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                p.data -= lr * grad
```

* 维护 `param_groups`，每组有一个学习率 `lr`。
* `step()` 等价于：`w ← w - lr * grad`。
* 这是最朴素的 SGD，不含 momentum / weight decay / 梯度裁剪等

### 3.2 简版 AdaGrad（历史梯度平方缩放）

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)            # g2_t = g2_{t-1} + grad_t^2
                state["g2"] = g2
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

* 为每个参数维护一个同形状的累加器 `g2`，存历史梯度平方和。
* 更新规则：$w ← w - lr \cdot grad / \sqrt{g2 + \epsilon}$；

* 梯度频繁的维度步长被缩小，实现自适应学习率；
* 只需 1 份优化器状态（每参数 1 份 `g2`）。


## 4 优化器之间的关系梳理




| 优化器          | 关键机制               | 特点                   |
| ------------ | ------------------ | -------------------- |
| **Momentum** | SGD + 梯度的指数滑动平均（动量） | 平滑更新轨迹，减少震荡          |
| **AdaGrad**  | SGD + 梯度平方的累积平均（自适应分量）| 稀疏特征学习快，但学习率单调变小     |
| **RMSProp**  | AdaGrad 的改进，使用**指数滑动平均**跟踪梯度平方| 缓解 AdaGrad 学习率衰减过快问题 |
| **Adam**     | RMSProp + Momentum | 同时平滑一阶与二阶矩，含偏置校正     |



Adam 实际上综合了动量和平滑自适应两种思想，是目前应用最广的优化器


## 5 `optimizer()` 中的训练流程

```python
B = 2; D = 4; num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers).to(get_device())

optimizer = AdaGrad(model.parameters(), lr=0.01)

x = torch.randn(B, D, device=get_device())
y = torch.tensor([4., 5.], device=get_device())

pred_y = model(x)
loss = F.mse_loss(pred_y, y)
loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

* `B` 为 batch size，`D` 为输入维度；
* 构造模型与优化器
* 前向计算、损失、反向传播得到梯度。
* `optimizer.step()` 按各自规则更新参数
* `zero_grad(set_to_none=True)` 将 `p.grad` 置为 `None`（更省内存，也避免与下一步梯度相加）

> 备注：`model.state_dict()` 的检查用于对比更新前后权重是否改变，以及观察优化器状态（如 AdaGrad 的 `g2`）


## 6 优化器显存与 FLOPs 估算


```python
## 权重：
num_parameters = (D * D * num_layers) + D      # 权重 + 偏置（示例里按每层 D×D，再加输出 D）

## 激活值：
num_activations = B * D * num_layers          # 每层激活缓存

## 梯度
num_gradients = num_parameters                 # 每个参数一份梯度

## 优化器
num_optimizer_states = num_parameters          # AdaGrad：每参数 1 份 g2

## 总
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)  # float32，单位：字节
flops = 6 * B * num_parameters                 # 单步训练 FLOPs，一阶近似（前向×2 + 反向×4）
```

要点：

* AdaGrad 的优化器状态量 = 参数量（1 倍）
* 若是 Adam，优化器状态通常≈2 倍参数量（m 和 v 两份）。
* `6 × B × #params` 来自经典近似：前向约 `2 × B × #params`，反向约 `4 × B × #params`。
* 真正的 Transformer 需要细分 QKV、注意力、MLP 两层线性以及归一化等，但主导仍是矩阵乘法。

参考：

* [Blog: *Memory usage in Transformer training*](https://erees.dev/transformer-memory/)
* [Blog: *Transformer FLOPs Estimation*](https://www.adamcasson.com/posts/transformer-flops)