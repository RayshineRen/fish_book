---
title: 仓库总览
tags:
  - summary
  - source
  - repo
source_type: repo
source_url: raw/
updated: 2026-04-24
---

# 仓库总览

## 顶层来源

- `raw/docs/README.md`：说明 `raw/docs/` 用于存放鱼书相关文档。
- `raw/docs/` 新增外部学习笔记，补充了第 3、4、5、7 章的解释性材料与一条“鱼书到项目实战”的学习路线。
- `raw/pdfs/README.md`：说明 `raw/pdfs/` 用于存放鱼书 PDF。
- `raw/code/fish-book-practices/README.md`：最完整的仓库说明，给出了项目结构、学习路径和示例文件用途。

## 代码仓库结构

- `dataset/`：MNIST 数据读取与预处理。
- `libs/`：激活函数、层实现、网络类、优化器、训练器、卷积辅助函数。
- `example/`：从逻辑门到 CNN 的脚本与 notebook。
- `models/`：示例权重文件。
- `screen-short/`：配图资源，供 notebook 解释和可视化使用。

## 仓库表达的学习路径

- 第一阶段：感知机、激活函数、简单前向传播。
- 第二阶段：损失函数、导数、梯度。
- 第三阶段：数值梯度与误差反向传播。
- 第四阶段：优化器、参数初始化、BatchNorm。
- 第五阶段：过拟合、Dropout、超参数优化。
- 第六阶段：卷积神经网络。

## 相关页面

- [[concepts/学习地图|学习地图]]
- [[concepts/鱼书代码架构|鱼书代码架构]]
- [[summaries/fish-book-core-implementation|核心实现]]
- [[summaries/fish-book-examples|示例与练习]]
- [[summaries/fish-book-external-study-notes|外部学习笔记汇总]]
- [[summaries/fish-book-to-yolov5-roadmap|鱼书到 YOLOv5 的实践路径]]
