# 🌌 EigenTruth

[![Status: MVP Alpha](https://img.shields.io/badge/Status-MVP_Alpha-orange.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![Model: HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow.svg)]()
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

*(English version follows the Chinese section)*

> **核心公理**："幻觉不是知识的缺失，而是表征流向的几何偏离。"

**EigenTruth** 是一个基于**几何动力学**与**表征工程 (RepE)** 的大模型幻觉治理 PyTorch 工具库。

非侵入式、即插即用（类似 PEFT），纯 PyTorch 实现。

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔬 **Sherman-Morrison 在线更新** | 增量构建协方差逆矩阵，无需全量重新计算 |
| 📏 **马氏距离监测** | 实时检测隐状态偏离真值流形的程度 |
| 🌀 **庞加莱球映射** | 将高维表征投射到双曲空间，捕捉层次化语义结构 |
| 📊 **双曲语义熵 (HSE)** | 量化生成过程中的语义发散程度 |
| 🎯 **对比引导 (Contrastive Steering)** | 当检测到幻觉时，自动注入真假对比修正向量 |
| 🛡️ **数值稳定** | FP32 内部计算、epsilon 正则、批处理安全 (Batch-safe) |

---

## 🏗️ 系统架构

```text
[ Input Prompt ]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              EigenTruth Wrapper (nn.Module)              │
│                                                         │
│  ┌─────────────────┐    [1. 离线真值流形基建 (Warmup)]     │
│  │ LLM Transformer │ <── 提取事实与谬误流形(Contrastive)  │
│  │                 │                                     │
│  │                 │    [2. 实时真值探针 Hook]              │
│  │                 │ <── 拦截特定层 Hidden States           │
│  │                 │ ──> 计算马氏距离 (Sherman-Morrison)   │
│  │                 │ <── 注入激活引导向量 (Steering)        │
│  └─────────────────┘                                     │
│       │                                                  │
│       ▼                                                  │
│  [3. 庞加莱映射器]                                         │
│  基于滑动窗口计算双曲语义熵 (HSE)                           │
│  判定逻辑崩塌并输出高危预警                                 │
└─────────────────────────────────────────────────────────┘
       │
       ▼
[ Output Token + 幻觉预警 ]
```

---

## 🚀 快速开始

### 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/catamitez0-maker/EigenTruth.git

# 开发模式安装
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
pip install -e .[dev]
```

### 使用示例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eigentruth.models.wrapper import EigenTruthWrapper

# 1. 加载原生基座模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 穿戴 EigenTruth 物理装甲 (在倒数第8层挂载探针，干预强度0.5)
safe_model = EigenTruthWrapper(
    model=base_model,
    target_layer_idx=-8,
    steering_lambda=0.5,
    mahalanobis_threshold=5.0
)

# 3. 探针冷启动 (预热真实与错误流形，用于提取对比方向)
fact_dataset = ["The Earth revolves around the Sun."]
false_dataset = ["The Sun revolves around the Earth."]
safe_model.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)

# 4. 安全受控的文本生成 (自带双曲熵监测与实时纠偏)
inputs = tokenizer("Did you know that", return_tensors="pt")
outputs = safe_model.generate(**inputs, max_new_tokens=10, do_sample=False)

print(tokenizer.decode(outputs[0]))
# 若逻辑发散，控制台将自动打印：
# [EigenTruth] ⚠️ 检测到深层语义发散 (HSE > 阈值)，系统可能正在产生幻觉！
```

---

## 🤝 贡献与许可
欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。
License: [Apache License 2.0](LICENSE)

<br>
<hr>
<br>

> **Core Axiom**: "Hallucination is not an absence of knowledge, but a geometric deviation of representational flow."

**EigenTruth** is an LLM hallucination governance PyTorch toolkit based on **Geometric Dynamics** and **Representation Engineering (RepE)**.

Non-intrusive, plug-and-play (similar to PEFT), and implemented purely in PyTorch.

---

## ✨ Core Features

| Feature | Description |
|------|------|
| 🔬 **Sherman-Morrison Updates** | Incremental construction of the inverse covariance matrix without full recalculation. |
| 📏 **Mahalanobis Distance** | Real-time monitoring of hidden state deviations from the truth manifold. |
| 🌀 **Poincaré Mapping** | Projects high-dimensional representations into hyperbolic space to capture hierarchical semantic structures. |
| 📊 **Hyperbolic Semantic Entropy (HSE)** | Quantifies the degree of semantic divergence during generation. |
| 🎯 **Contrastive Steering** | Automatically injects a true/false contrastive correction vector when hallucinations are detected. |
| 🛡️ **Numerical Stability** | Internal FP32 computation, epsilon regularization, and batch-safe parallel execution. |

---

## 🏗️ System Architecture

```text
[ Input Prompt ]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              EigenTruth Wrapper (nn.Module)              │
│                                                         │
│  ┌─────────────────┐    [1. Offline Truth Manifold Warmup] │
│  │ LLM Transformer │ <── Extract Fact/Fallacy Manifolds    │
│  │                 │                                     │
│  │                 │    [2. Real-time Truth Probe Hook]    │
│  │                 │ <── Intercept layer Hidden States     │
│  │                 │ ──> Compute Mahalanobis Distance      │
│  │                 │ <── Inject Activation Steering Vector │
│  └─────────────────┘                                     │
│       │                                                  │
│       ▼                                                  │
│  [3. Poincaré Mapper]                                      │
│  Compute Hyperbolic Semantic Entropy (HSE) via window      │
│  Evaluate logical collapse and trigger high-risk alerts    │
└─────────────────────────────────────────────────────────┘
       │
       ▼
[ Output Token + Hallucination Alert ]
```

---

## 🚀 Quick Start

### Installation

```bash
# Install via GitHub
pip install git+https://github.com/catamitez0-maker/EigenTruth.git

# Install for Development
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
pip install -e .[dev]
```

### Usage Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eigentruth.models.wrapper import EigenTruthWrapper

# 1. Load the base model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Equip the EigenTruth armor (attach probe to the -8th layer, intervention strength 0.5)
safe_model = EigenTruthWrapper(
    model=base_model,
    target_layer_idx=-8,
    steering_lambda=0.5,
    mahalanobis_threshold=5.0
)

# 3. Warmup the truth probe (precompute the truth and false manifolds for contrastive direction)
fact_dataset = ["The Earth revolves around the Sun."]
false_dataset = ["The Sun revolves around the Earth."]
safe_model.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)

# 4. Safe and controlled text generation (with built-in HSE monitoring and real-time steering)
inputs = tokenizer("Did you know that", return_tensors="pt")
outputs = safe_model.generate(**inputs, max_new_tokens=10, do_sample=False)

print(tokenizer.decode(outputs[0]))
# If semantic divergence occurs, the console will print:
# [EigenTruth] ⚠️ 检测到深层语义发散 (HSE > 阈值)，系统可能正在产生幻觉！
```

---

## 🤝 Contributing & License
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
License: [Apache License 2.0](LICENSE)
