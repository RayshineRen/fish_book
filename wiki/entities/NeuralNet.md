---
title: NeuralNet
tags:
  - entity
  - model
updated: 2026-04-24
---

# NeuralNet

## 定位

`NeuralNet` 是 `libs/network.py` 中的核心全连接网络实现。

## 可配置项

- `hidden_size_list`
- `activation`：`relu` 或 `sigmoid`
- `weight_scale`：`he`、`xavier` 或常数
- `use_batchnorm`
- `weight_decay_lambda`
- `use_dropout`

## 训练相关接口

- `predict()`
- `loss()`
- `accuracy()`
- `numerical_gradient()`
- `gradient()`

## 设计意义

- 它把书中第 3 章到第 6 章的大部分内容集中到一个对象里。
- 它与 [[entities/Trainer|Trainer]] 解耦，因此不同实验只需要改超参数或层配置。

## 相关页面

- [[concepts/神经网络前向传播|神经网络前向传播]]
- [[concepts/误差反向传播|误差反向传播]]
- [[concepts/训练循环与优化器|训练循环与优化器]]
