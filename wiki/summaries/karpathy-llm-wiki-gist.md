---
title: Karpathy 的 llm-wiki gist
tags:
  - summary
  - methodology
  - llm-wiki
source_type: gist
source_url: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
updated: 2026-04-24
---

# Karpathy 的 llm-wiki gist

## 核心思想

- 把 `raw/` 视为原始资料区，把 `wiki/` 视为可读、可导航、可持续演进的知识层。
- 允许知识库不完整，但要求它随着阅读和整理不断增长。
- `index.md` 负责导航，`log.md` 负责记录知识库如何被扩展和修正。
- 页面对内容做“重写与重组”，而不是机械复制原文。

## 对当前仓库的启发

- 本项目需要把鱼书 PDF、配套代码、notebook 和数据集统一到一个入口里。
- `index.md` 应该首先回答“从哪里开始读、如何交叉跳转”。
- `log.md` 不只是流水账，而是知识库维护轨迹。
- 页面之间要互相链接，让“书中的概念”和“代码里的类/函数”能来回跳转。

## 采用方式

- 保留 `wiki/index.md` 与 `wiki/log.md` 作为固定入口。
- 用 `summaries/` 承担“来源摘要”的职责。
- 用 `concepts/` 承担“主题归纳”的职责。
- 用 `entities/` 承担“书、作者、数据集、关键实现对象”的职责。

## 相关页面

- [[concepts/知识库组织原则|知识库组织原则]]
- [[summaries/lewislulu-llm-wiki-skill|Lewislulu 的 llm-wiki-skill]]
