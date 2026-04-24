---
title: MNIST
tags:
  - entity
  - dataset
updated: 2026-04-24
---

# MNIST

## 定位

MNIST 是鱼书中最核心的数据集，几乎所有训练实验都围绕它展开。

## 在本仓库中的实现

- 数据加载器：`dataset/mnist.py`
- 训练数据规模：`60000`
- 测试数据规模：`10000`
- 默认图像尺寸：`1 x 28 x 28`

## 支持的变体

- `normalize=True`：把像素缩放到 `0~1`。
- `one_hot_label=True`：把标签转成 one-hot。
- `flatten=True/False`：决定输入是向量还是 4D 张量。

## 对不同模型的影响

- [[entities/NeuralNet|NeuralNet]] 默认更适合 `flatten=True`。
- [[entities/SimpleConvNet|SimpleConvNet]] 必须使用 `flatten=False`。

## 相关页面

- [[summaries/fish-book-core-implementation|核心实现]]
- [[concepts/卷积神经网络|卷积神经网络]]
