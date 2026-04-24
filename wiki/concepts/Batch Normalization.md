---
title: Batch Normalization
tags:
  - concept
  - chapter-6
updated: 2026-04-24
---

# Batch Normalization

## 目标

Batch Normalization 试图稳定每一层的输入分布，从而加快训练并改善深层网络的可训练性。

## 在本仓库中的实现

- `libs/layers.py` 提供 `BatchNormalizationLayer`。
- `NeuralNet` 在每个隐藏层的 `Affine` 之后可选插入 `BatchNorm`。
- `t17-增加BatchNorm改善学习效率.ipynb` 和 `t20` 给出了实验比较。

## 实现要点

- 训练时计算当前 batch 的均值和方差。
- 保存 `running_mean` 和 `running_var`，供测试阶段使用。
- 引入可学习参数 `gamma` 与 `beta` 做缩放和平移。

## 与其他技巧的关系

- 它不是正则化的替代品。
- 它常与 ReLU、He 初始化、Adam 等一起使用。
- 对 sigmoid 网络，它能一定程度缓解训练困难，但不能完全消除结构性问题。

## 相关页面

- [[concepts/权重初始化与正则化|权重初始化与正则化]]
- [[entities/NeuralNet|NeuralNet]]
- [[summaries/fish-book-examples|示例与练习]]
