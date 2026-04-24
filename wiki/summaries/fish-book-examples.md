---
title: 示例与练习
tags:
  - summary
  - source
  - notebooks
source_type: code
source_url: raw/code/fish-book-practices/example
updated: 2026-04-24
---

# 示例与练习

## 脚本部分

- `t01.py`：最简单的 AND 感知机。
- `t02.py`：用 AND、OR、NAND 组合出 XOR，体现多层感知机突破线性不可分。
- `t03-激活函数实现.py`：step、sigmoid、ReLU。
- `t05-简单的神经网络实现.py`：三层网络前向传播。
- `t06-手写数字数据集.py`、`t07-加载模型进行手写数字识别.py`：MNIST 与推理。

## Notebook 主线

- `t08` 到 `t13`：损失函数、导数、梯度、数值微分、反向传播。
- `t14` 到 `t19`：优化器、初始化、BatchNorm、过拟合与超参数优化。
- `t20`：把优化器、Dropout、BatchNorm、权重衰减等组合起来做系统实验。
- `t21`：实现 `SimpleConvNet`，训练一个 `conv -> relu -> pool -> affine -> relu -> affine -> softmax` 网络。
- `sigmoid梯度消失问题.ipynb`：针对 sigmoid 深层网络退化做额外分析。

## 最有价值的补充信息

- `t20` 说明仅靠 Adam 并不自动解决泛化问题，往往还需要正则化或归一化。
- `t21` 说明 CNN 版本的数据输入要设置 `flatten=False`，并把样本组织成 `(N, C, H, W)`。
- 多个 notebook 实际上把书中的第 6、7 章重新做成了可执行实验。

## 对应页面

- [[concepts/学习地图|学习地图]]
- [[concepts/感知机|感知机]]
- [[concepts/损失函数与梯度|损失函数与梯度]]
- [[concepts/误差反向传播|误差反向传播]]
- [[concepts/训练循环与优化器|训练循环与优化器]]
- [[entities/SimpleConvNet|SimpleConvNet]]
