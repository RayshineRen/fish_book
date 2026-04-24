---
title: 鱼书 Wiki
tags:
  - wiki
  - fish-book
  - index
updated: 2026-04-24
---

# 鱼书 Wiki

这个知识库围绕《[[entities/深度学习入门：基于Python的理论与实现|深度学习入门：基于Python的理论与实现]]》组织，目标是把 `raw/` 中的书籍、代码、notebook、数据集和仓库说明整理成一个可持续维护的 wiki。

## 如何使用

- 先读 [[concepts/学习地图|学习地图]]，了解从感知机到 CNN 的主线。
- 再读 [[concepts/鱼书代码架构|鱼书代码架构]]，把书中概念和 `raw/code/` 的实现对应起来。
- 需要追溯来源时，从下方 `源摘要` 进入。
- 变更历史统一记录在 [[log]]，本仓库按你的要求保留单文件 `log.md`。

## 概念

- [[concepts/知识库组织原则|知识库组织原则]]
- [[concepts/学习地图|学习地图]]
- [[concepts/感知机|感知机]]
- [[concepts/神经网络前向传播|神经网络前向传播]]
- [[concepts/损失函数与梯度|损失函数与梯度]]
- [[concepts/误差反向传播|误差反向传播]]
- [[concepts/训练循环与优化器|训练循环与优化器]]
- [[concepts/权重初始化与正则化|权重初始化与正则化]]
- [[concepts/Batch Normalization|Batch Normalization]]
- [[concepts/卷积神经网络|卷积神经网络]]
- [[concepts/鱼书代码架构|鱼书代码架构]]

## 实体

- [[entities/深度学习入门：基于Python的理论与实现|深度学习入门：基于Python的理论与实现]]
- [[entities/斋藤康毅|斋藤康毅]]
- [[entities/MNIST|MNIST]]
- [[entities/NeuralNet|NeuralNet]]
- [[entities/Trainer|Trainer]]
- [[entities/SimpleConvNet|SimpleConvNet]]

## 源摘要

- [[summaries/karpathy-llm-wiki-gist|Karpathy 的 llm-wiki gist]]
- [[summaries/lewislulu-llm-wiki-skill|Lewislulu 的 llm-wiki-skill]]
- [[summaries/fish-book-pdf|鱼书 PDF]]
- [[summaries/fish-book-repo-overview|仓库总览]]
- [[summaries/fish-book-core-implementation|核心实现]]
- [[summaries/fish-book-examples|示例与练习]]
- [[summaries/fish-book-external-study-notes|外部学习笔记汇总]]
- [[summaries/fish-book-to-yolov5-roadmap|鱼书到 YOLOv5 的实践路径]]

## 导航建议

- 按章节学：[[concepts/学习地图|学习地图]] -> [[concepts/感知机|感知机]] -> [[concepts/神经网络前向传播|神经网络前向传播]] -> [[concepts/损失函数与梯度|损失函数与梯度]] -> [[concepts/误差反向传播|误差反向传播]] -> [[concepts/训练循环与优化器|训练循环与优化器]] -> [[concepts/卷积神经网络|卷积神经网络]]
- 按实现学：[[entities/MNIST|MNIST]] -> [[entities/NeuralNet|NeuralNet]] -> [[entities/Trainer|Trainer]] -> [[entities/SimpleConvNet|SimpleConvNet]]
- 按来源查证：先看 `源摘要`，再跳到概念页和实体页。

## 当前覆盖范围

- 已覆盖 `raw/pdfs/fish-book.pdf` 的章节骨架与主题。
- 已覆盖 `raw/code/fish-book-practices/` 的 README、核心库、示例脚本和主要 notebook。
- 已覆盖 `raw/docs/` 新增的 Chapter 3/4/5/7 学习笔记，并把其中的解释性内容回填到对应概念页。
- 已覆盖 `raw/docs/` 中“鱼书 -> YOLOv5 -> 复试包装”的实践路线，并作为外围实践材料整理。
- 第 8 章“深度学习应用与未来”在 `raw/code/` 中没有对应完整实现，目前只保留为书籍层面的主题线索。
