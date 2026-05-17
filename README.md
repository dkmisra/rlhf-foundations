# RLHF Foundations: Learn RL Fundamentals on your Macbook/Linux

This repository contains the core implementation of recent RL algorithms designed for RLHF, that you can run locally on your Macbook and Linux. It also _simulates_ some issues that arise in real-world like mismatch between inference and training, and how it is specially bad with MoE models. The goal here is to help people new to this area get familiarize with core concept with the excess engineering that is required for large-scale multi-node async-RL experiments.

Focus is on simplicity and readability. The repository is not designed for large-scale GPU experiments, but more for researchers and engineers who want to learn core concepts in RLHF.

An accompanying PDF covering fundamental concept is planned for release in near future.

# What it contains

The repository contains the following core packages. 

## RL Algorithms

Contains implementation of following algorithms:

- GRPO: [Paper](https://arxiv.org/pdf/2402.03300)
- GSPO: [Paper](https://arxiv.org/pdf/2507.18071)
- DAPO: [Paper](https://arxiv.org/pdf/2503.14476)
- CISPO: [Paper](https://arxiv.org/pdf/2506.13585)
- TIS: [Blog](https://fengyao.notion.site/off-policy-rl)
- MIS: [Blog](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)

## Environments

Contains two simple environments: 

- Dyck language: the agent has to complete a partial list of unbalanced parenthesis.
- Block arangement: given 

## LLMs

Simple readable implementation of transformer-based LLM with Multi-head attention, MoE, and RoPE.  

## Utils

Utils for visualization, tokenization, etc.

# Issues and Citations

Feel free to raise issues and PR. If this is useful, you can cite it below: