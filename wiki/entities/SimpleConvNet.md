---
title: SimpleConvNet
tags:
  - entity
  - model
  - cnn
updated: 2026-04-24
---

# SimpleConvNet

## 定位

`SimpleConvNet` 出现在 `t21-简单的卷积网络实现.ipynb`，是鱼书 CNN 章节的最小可训练实现。

## 网络结构

`conv -> relu -> pool -> affine -> relu -> affine -> softmax`

## 关键参数

- 卷积核数量 `filter_num`
- 卷积核大小 `filter_size`
- 步长 `stride`
- 填充 `pad`
- 隐藏层大小 `hidden_size`

## 与其他组件的关系

- 使用 [[entities/MNIST|MNIST]] 的 4D 输入。
- 复用 `ConvolutionLayer`、`PoolingLayer`、`AffineLayer`、`SoftmaxWithLossLayer`。
- 继续使用 [[entities/Trainer|Trainer]] 做训练。

## 相关页面

- [[concepts/卷积神经网络|卷积神经网络]]
- [[concepts/鱼书代码架构|鱼书代码架构]]
