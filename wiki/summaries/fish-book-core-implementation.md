---
title: 核心实现
tags:
  - summary
  - source
  - code
source_type: code
source_url: raw/code/fish-book-practices
updated: 2026-04-24
---

# 核心实现

## 关键模块

- `libs/functions.py`：`sigmoid`、`relu`、`tanh`、`softmax`、交叉熵、数值梯度。
- `libs/layers.py`：`ReluLayer`、`SigmoidLayer`、`AffineLayer`、`SoftmaxWithLossLayer`、`BatchNormalizationLayer`、`DropoutLayer`、`ConvolutionLayer`、`PoolingLayer`。
- `libs/network.py`：抽象 `Network` 与可配置的 `NeuralNet`。
- `libs/optimizer.py`：`SGD`、`Momentum`、`AdaGrad`、`Adam`。
- `libs/trainer.py`：统一训练循环。
- `dataset/mnist.py`：下载、缓存、归一化、one-hot、展平或保留 4D 张量。

## 实现风格

- 不依赖深度学习框架，主要基于 NumPy。
- 代码显式实现前向传播、反向传播和梯度更新，适合教学。
- 同一主题常同时保留朴素实现和高效实现，例如卷积既有直观版本也有 `im2col` 版本。

## 关键观察

- `NeuralNet` 支持激活函数切换、He/Xavier 初始化、BatchNorm、Dropout、权重衰减。
- `Trainer` 把 mini-batch 抽样、梯度计算、优化器更新、每 epoch 评估串成统一流程。
- `ConvolutionLayer` 与 `PoolingLayer` 的高效版本依赖 `im2col`/`col2im`。
- `MNIST` 数据加载器决定了全连接网络和 CNN 的输入形态差异。

## 对应页面

- [[entities/NeuralNet|NeuralNet]]
- [[entities/Trainer|Trainer]]
- [[entities/MNIST|MNIST]]
- [[concepts/训练循环与优化器|训练循环与优化器]]
- [[concepts/权重初始化与正则化|权重初始化与正则化]]
- [[concepts/Batch Normalization|Batch Normalization]]
- [[concepts/卷积神经网络|卷积神经网络]]
