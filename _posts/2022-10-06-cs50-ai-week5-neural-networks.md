---
layout: articles
title: week5-neural-networks
tags: cs50
---




# Neural Networks

AI neural networks are inspired by neuroscience. In the brain, neurons are cells that are connected to each other, forming networks. Each neuron is capable of both receiving and sending electrical signals. Once the electrical input that a neuron receives crosses some threshold, the neuron activates, thus sending its electrical signal forward.

An Artificial Neural Network is a mathematical model for learning inspired by biological neural networks. Artificial neural networks model mathematical functions that map inputs to outputs based on the structure and parameters of the network. In artificial neural networks, the structure of the network is shaped through training on data.

When implemented in AI, the parallel of each neuron is a unit that’s connected to other units. For example, like in the last lecture, the AI might map two inputs, x₁ and x₂, to whether it is going to rain today or not. Last lecture, we suggested the following form for this hypothesis function: h(x₁, x₂) = w₀ + w₁x₁ + w₂x₂, where w₁ and w₂ are weights that modify the inputs, and w₀ is a constant, also called bias, modifying the value of the whole expression.

#### Activation Functions 激活函数

To use the hypothesis function to decide whether it rains or not, we need to create some sort of threshold based on the value it produces.

需要创造阈值

One way to do this is with a step function, which gives 0 before a certain threshold is reached and 1 after the threshold is reached.

阶梯状函数

Another way to go about this is with a logistic function, which gives as output any real number from 0 to 1, thus expressing graded confidence in its judgment.

Logistic 函数

Another possible function is Rectified Linear Unit (ReLU), which allows the output to be any positive value. If the value is negative, ReLU sets it to 0.

ReLU 函数


#### Neural Network Structure 神经网络结构




#### Gradient Descent 梯度下降


#### Multilayer Neural Networks 多层神经网络


#### Backpropagation


#### Overfitting 过拟合



#### TensorFlow


