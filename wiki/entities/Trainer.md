---
title: Trainer
tags:
  - entity
  - training
updated: 2026-04-24
---

# Trainer

## 定位

`Trainer` 负责驱动训练过程，是本仓库实验复用度最高的对象之一。

## 主要职责

- 管理 epoch、iteration 和 batch。
- 调用网络的 `gradient()`。
- 调用优化器的 `update()`。
- 记录 `train_loss_list`、`train_acc_list`、`test_acc_list`。

## 设计价值

- 统一了全连接网络与 CNN 的训练流程。
- 使实验差异主要收敛到“模型配置”和“优化参数”上。

## 相关页面

- [[entities/NeuralNet|NeuralNet]]
- [[entities/SimpleConvNet|SimpleConvNet]]
- [[concepts/训练循环与优化器|训练循环与优化器]]
