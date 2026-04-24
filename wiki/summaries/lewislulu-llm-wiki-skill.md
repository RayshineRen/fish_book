---
title: Lewislulu 的 llm-wiki-skill
tags:
  - summary
  - methodology
  - skill
source_type: github
source_url: https://github.com/lewislulu/llm-wiki-skill
updated: 2026-04-24
---

# Lewislulu 的 llm-wiki-skill

## 结构偏好

- 推荐把 wiki 分成 `summaries/`、`concepts/`、`entities/` 三层。
- 强调先做来源摘要，再逐步沉淀概念页与实体页。
- 鼓励每个页面只承载单一主题，并通过内部链接形成网络。

## 方法偏好

- 摘要页要说明来源、关键信息、局限和与其他页面的关系。
- 概念页应该跨来源整合，而不是被某一个原始文件绑定。
- 实体页适合收纳相对稳定的对象，例如作者、数据集、模型类、关键模块。
- 页面要持续演进，允许随着原始资料的补充而重构。

## 对当前仓库的应用

- `raw/pdfs/fish-book.pdf` 适合先进入 [[summaries/fish-book-pdf|鱼书 PDF]]。
- `raw/code/fish-book-practices/libs/`、`dataset/` 和 `example/` 先进入 [[summaries/fish-book-core-implementation|核心实现]] 与 [[summaries/fish-book-examples|示例与练习]]。
- 从这些摘要页再沉淀出 `[[concepts/鱼书代码架构|鱼书代码架构]]`、`[[concepts/训练循环与优化器|训练循环与优化器]]`、`[[concepts/卷积神经网络|卷积神经网络]]` 等主题页。

## 保留与调整

- 本仓库吸收了该 skill 的三层结构。
- 同时保留单文件 `log.md`，因为这是你的显式要求，也是 Karpathy 原始提议的一部分。

## 相关页面

- [[summaries/karpathy-llm-wiki-gist|Karpathy 的 llm-wiki gist]]
- [[concepts/知识库组织原则|知识库组织原则]]
