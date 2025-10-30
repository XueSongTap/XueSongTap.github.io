---
layout: article
title: 分布式模型并行
tags: distribute
---

## 1 模型并行训练概念（Model Parallel Training）
**模型并行（Model Parallelism）**：将模型的计算（前向传播 / 反向传播 / 参数更新）分布到多个 GPU 上执行。


![model parallel](/img/2025/10/model_parallel.png)

## 2 流水线PP并行


### 2.1 朴素模型并行（Naïve Model Parallel）

最原始的模型并行方式是：将模型的不同层分布到不同 GPU 上执行。

![naive pp](/img/2025/10/naive_pp.png)


![alt text](/img/2025/10/naive_pp_comm.png)


参考: [https://siboehm.com/articles/22/pipeline-parallel-training](https://siboehm.com/articles/22/pipeline-parallel-training)

在模型定义的时候，针对于层数进行拆分
```py
  class ModelParallelResNet50(ResNet):
      def __init__(self, *args, **kwargs):
          super(ModelParallelResNet50, self).__init__(
              Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

          self.seq1 = nn.Sequential(
              self.conv1,
              self.bn1,
              self.relu,
              self.maxpool,
              self.layer1,
              self.layer2
          ).to('cuda:0')

          self.seq2 = nn.Sequential(
              self.layer3,
              self.layer4,
              self.avgpool,
          ).to('cuda:1')

          self.fc.to('cuda:1')
        
      def forward(self, x):
          x = self.seq2(self.seq1(x).to('cuda:1'))
          return self.fc(x.view(x.size(0), -1))      
```
#### 局限性
* **GPU 利用率低**：同一时刻只有一个 GPU 在计算，其余处于空闲
* **计算与通信无法重叠**：传输中间结果时 GPU 处于空闲状态
* **内存需求高**：首个 GPU 需缓存所有激活值，直到整个 batch 完成



### 2.2 GPipe：经典流水线并行方案


#### 2.2.1 核心思想
**核心思想**：将一个 batch 拆分成多个 **micro-batch**，以流水线方式在不同 GPU 间流动。

- 一个 batch 的大小通常由 GPU 的显存容量以及单个样本在前向/反向传播中所需的显存决定，
通常尽可能大，以充分利用 GPU 显存。

- 一个 micro-batch 则可以更小


参考论文：
[1] Huang, Yanping, et al. *"Gpipe: Efficient training of giant neural networks using pipeline parallelism."* NeurIPS 2019.


#### 2.2.2 梯度检查点


减少流水线并行的显存开销，其实主要是重计算，用计算换显存



**前向传播阶段**：每个 GPU 仅保存输出激活值。
**反向传播阶段**：每个阶段重新计算自己的前向函数 $F_k$

	
##### 2.2.2.1 Vanilla Backprop（普通反向传播）

显存占用：O(n)

- 因为在前向传播时，网络中每一层的中间激活值（activation）都被保存下来

- 在反向传播计算梯度时直接使用这些缓存的激活

- 所以显存随层数线性增长

计算量：O(n)

- 每层在 forward 计算一次，在 backward 计算一次，因此总体是线性复杂度

优点： 快（无需重复计算）
缺点： 占用大量显存，难以在大模型上训练。

##### 2.2.2.2 Memory-poor Backprop（节省显存的反向传播 / 梯度检查点）

显存占用：O(1)

- 在前向传播阶段不保存所有中间激活，只保留少量（例如 checkpoint 的节点输出）。
- 在反向传播时，需要**重新计算（recompute）**部分前向结果。
- 因此显存占用几乎与层数无关。

计算量：O(n²)
- 因为反向传播时要多次重新执行前向计算（每次重算部分子网络的 forward），
- 导致计算开销从线性上升为平方级别。


##### 2.2.2.3 对比
| 算法类型                 | 显存占用   | 计算量      | 特点      |
| -------------------- | ------ | -------- | ------- |
| **Vanilla Backprop** | $O(n)$ | $O(n)$   | 快但显存高   |
| **Checkpointing**    | $O(1)$ | $O(n^2)$ | 显存低但需重算 |


参考；

https://github.com/cybertronai/gradient-checkpointing

Chen, Tianqi, et al. "Training deep nets with sublinear memory cost." arXiv preprint arXiv:1604.06174 (2016).


### 2.3 PP 并行弊端分析

![pp bubble](/img/2025/10/pp_bubble.png)

- **等待时间（Bubble）与内存需求**： 为了计算梯度（后向传播），每个阶段需要其对应的前向传播过程中产生的激活值，即填充/排空 阶段的 GPU空间

- **In-Flight）的微批次**： 由于pipeline 的填充和排空（filling and draining），在任何给定时间点，某些微批次可能已经完成了前向传播（生成了激活值），但尚未开始或完成后向传播。这些就是“in-flight”的微批次， 反向传播未开始的前向激活需保留在显存中

- **巨大的内存开销：**  必须在 GPU 内存中保留所有这些“in-flight”微批次产生的全部激活值，直到它们相应的后向传播完成并释放内存为止。如果“in-flight”微批次的数量很大（如示例中的 8 个），则所需的内存会显著增加，甚至可能超过设备的可用内存，这就是主要的内存瓶颈

> 总结：开 PP 虽能加速，但内存与调度负担显著增加

### 2.4 1F1B Flush 优化策略

![PP 1F1B Flush](/img/2025/10/pp_1f1b_flush.png)

**思路**：在前向传播还未全部完成时，提前启动反向传播（即“尽早 backward”）

**效果**：显著减少 in-flight 激活值数量。
例如：若 $1B = 2F$，则最大仅有 4 个微批次同时驻留显存，而 GPipe 为 8 个



### 2.5 更进一步优化
![pp chunk & stage ](/img/2025/10/pp_chunk_stage.png)


#### 2.5.1 chunking 模型层


* **更细粒度的并行**： 允许在模型内部实现更细粒度的并行


* **更好的负载均衡**： 当模型层计算量不均匀时，通过切块可以将工作更均匀地分配给不同的设备，从而减少**瓶颈（bottleneck）**阶段，提高整体吞吐量。

* **更小的管道气泡**： 较小的块意味着前向和后向传播的操作单元更小，有助于减少管道气泡（Pipeline Bubble）或空闲时间

#### 2.5.2 阶段交错（Interleaving Stages）

在传统的管道并行中，不同的模型阶段（Stages）通常分配给不同的物理设备（GPU 0 负责 Stage 1，GPU 1 负责 Stage 2 等）。交错阶段指的是让同一个物理设备（例如 GPU 0）负责多个不连续的阶段（例如 Stage 1 和 Stage 4）。

- 缓解设备差异： 如果某些 GPU 性能较差，可以将计算量较小的阶段或交错分配给它们，以更好地利用所有硬件资源。

- 解决管道气泡： 可以在一个设备处理 Stage A 的后向传播时，同时处理 Stage B 的前向传播，从而更好地填补管道中的空闲时间，进一步减少等待，提高硬件利用率

参考论文：
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, Narayanan et al. (SC 2021)


### 2.6 GPipe 实现示例


仓库地址：[https://github.com/siboehm/shallowspeed](https://github.com/siboehm/shallowspeed)
```py
class GPipeSchedule(Schedule):
    """
    GPipe调度器：实现经典的“先全F后全B”管道并行策略
    """
    def steps(self):
        # 步骤 0: 梯度清零
        yield [ZeroGrad()]

        # STAGE 1: FWD all μBatches 对所有微批次 (μBatches) 进行前向传播 (FWD)
        for mubatch_id in range(self.num_micro_batches):
            yield self.steps_FWD_mubatch(mubatch_id)

        # STAGE 2: BWD all μBatches 对所有微批次 (μBatches) 进行后向传播 (BWD)，注意是逆序进行
        for mubatch_id in reversed(range(self.num_micro_batches)):
            yield from self.steps_BWD_mubatch(mubatch_id)

        # updating the weights is the last step of processing any batch 更新权重
        yield [OptimizerStep()]


    def steps_BWD_mubatch(self, mubatch_id):
        """定义单个微批次的后向传播步骤 (BWD)"""
        cmds = []
        # 如果是最后一个阶段（模型的尾部）：
        if self.is_last_stage:
            # 加载目标值（标签），用于计算损失和起始梯度
            cmds.append(LoadMuBatchTarget(mubatch_id=mubatch_id, buffer_id=0))
        else:
            # 接收来自下游阶段（模型更深层）的梯度（损失对输出的梯度）

            cmds.append(RecvOutputGrad(buffer_id=0))

        # 如果是第一个微批次（通常是最后一个 BWD 步骤）：
        if self.is_first_mubatch(mubatch_id):
            # interleaved backprop & AllReduce during last μBatch of BWD
            # 在最后一个微批次的 BWD 过程中，交错执行反向传播和 AllReduce
            cmds.append(BackwardGradAllReduce(buffer_id=0, mubatch_id=mubatch_id))
        else:
            # 执行反向传播，并将梯度累加起来（Accumulation）
            cmds.append(BackwardGradAcc(buffer_id=0, mubatch_id=mubatch_id))

        # 如果不是第一个阶段（模型的头部）：
        if not self.is_first_stage:
            # 将计算出的梯度（损失对输入的梯度）发送给上游阶段（模型更浅层）
            cmds.append(SendInputGrad(buffer_id=0))
        yield cmds

    def steps_FWD_mubatch(self, mubatch_id):
        """定义单个微批次的前向传播步骤 (FWD)"""
        cmds = []
        # 如果是第一个阶段（模型的头部）：
        if self.is_first_stage:
            # 加载输入数据（特征）
            cmds.append(LoadMuBatchInput(buffer_id=0, mubatch_id=mubatch_id))
        else:
            # 接收来自上游阶段（模型更浅层）的激活值
            cmds.append(RecvActivations(buffer_id=0))
        # 执行前向传播计算
        cmds.append(Forward(buffer_id=0, mubatch_id=mubatch_id))
        # the last stage just discards the output of its `forward()` pass since
        # it's not necessary for running BWD. The last stage just needs the target values
        # (loaded from disk) and the activations (cached inside the `Module`s) for BWD.
        # 如果不是最后一个阶段：
        # 将输出的激活值发送给下游阶段（模型更深层）
        # 注意：最后一个阶段会丢弃其 forward() 输出，因为它只需要 target 值和内部缓存的激活值
        if not self.is_last_stage:
            cmds.append(SendActivations(buffer_id=0))
        return cmds
```

### 2.7 PyTorch 官方 PP 实现

PyTorch 提供原生 API：
`torch.distributed.pipelining`
#### 2.7.1 两个阶段

#### 2.7.1 两个阶段

1. **构建阶段（PipelineStage）**

   * 手动划分模型层。
   * 或使用自动划分工具。

2. **调度阶段（PipelineSchedule）**

   * 执行微批次调度。




#### 2.7.2 模型构建示例


```py
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(...)

        # Using a ModuleDict lets us delete layers without affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(...)

        self.output = nn.Linear(...)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output
```
#### 2.7.3 管道与调度示例



```py
from torch.distributed.pipelining import PipelineStage


with torch.device("meta"):
    assert num_stages == 2, "This is a simple 2-stage example"

    # we construct the entire model, then delete the parts we do not need for this stage
    # in practice, this can be done using a helper function that automatically divides up layers across stages.
    model = Transformer()

    if stage_index == 0:
        # prepare the first stage model
        del model.layers["1"]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # prepare the second stage model
        model.tok_embeddings = None
        del model.layers["0"]

    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
```


管道构建，拆分stage 后实例化 PipelineStage
```py
from torch.distributed.pipelining import ScheduleGPipe
# Create a schedule
schedule = ScheduleGPipe(stage, n_microbatches)
# Input data (whole batch)
x = torch.randn(batch_size, in_dim, device=device)
# Run the pipeline with input `x` # `x` will be divided into microbatches automatically
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()
```


参考文档：[https://pytorch.org/docs/main/distributed.pipelining.html](https://pytorch.org/docs/main/distributed.pipelining.html)

## 3 张量并行（Tensor Parallelism, TP）


**核心思想**：矩阵按列 / 行拆分，分布到多个 GPU 计算。


![TP for FFN ](/img/2025/10/tp_for_FFN.png)

* TP 关注模型中 **大矩阵乘法** 的拆分 (例如FFN)
* 某些操作（如 LayerNorm、Dropout、Residual）不宜拆分，因通信开销过高



- TP 主要关注于拆分模型中最大的计算部分（如 FFN 中的大矩阵乘法）。然而，模型中还有一些操作，例如：层归一化 (Layer Norm) (Dropout)残差连接 (Residual Connections)

- 这些操作的计算量相对较小，且通常涉及输入张量的所有维度。如果尝试拆分它们，所需的通信开销（例如 All-gather 或 All-reduce）往往会超过节省的计算时间

在 TP 中，
* 每个 GPU 仅保存部分权重（如 $A_1, A_2, B_1, B_2$）。
* 各自计算对应梯度并独立更新。
* 不需要与其他 GPU 同步所有梯度（这与数据并行 DP 的做法不同）。



## 4 混合并行（Hybrid Parallelism）


### 4.1 PP + TP 混合（Model Parallel）


![alt text](/img/2025/10/pp_tp.png)




如图，先拆分PP，在PP里面拆TP


* **节点内优先 TP**（利用 NVLink 加速通信）。
* **节点间采用 PP**（激活值传输量小），通讯开销比较小，减少显存负担


节点内优先 TP，TP依赖nvlink通讯，加速计算

节点间用PP，通过传输少量激活值，通讯开销比较小，减少显存负担



### 4.2 DP + MP （DP + PP + TP）混合并行

1. 先确定模型并行规模
$$ M = tp_size \times pp_size $$

确保模型拆成M份后，不会超出单个GPU的内存


模型并行是一个节省显存的策略

2. DP，扩展加速

一旦确定好 M， 剩余的GPU显存用于DP

数据并行的核心作用是缩短训练时间。它通过将整个批次拆分到不同的模型副本上并行计算， 而有效增加训练的 GPU 总数，提升吞吐量
