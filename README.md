# 🌌 EigenTruth

[![Status: MVP Alpha](https://img.shields.io/badge/Status-MVP_Alpha-orange.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![Model: HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow.svg)]()
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

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
| 🎯 **激活引导 (Steering)** | 当检测到幻觉时，自动注入修正向量 |
| 🛡️ **数值稳定** | FP32 内部计算、epsilon 正则、梯度隔离 |

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
│  │ LLM Transformer │ <── 提取事实质心(Mean)与协方差(Cov)   │
│  │                 │                                     │
│  │                 │    [2. 实时真值探针 Hook]              │
│  │                 │ <── 拦截特定层 Hidden States           │
│  │                 │ ──> 计算马氏距离 (Sherman-Morrison)   │
│  │                 │ <── 注入激活引导向量 (Steering)        │
│  └─────────────────┘                                     │
│       │                                                  │
│       ▼                                                  │
│  [3. 庞加莱映射器]                                         │
│  计算双曲语义熵 (HSE)                                      │
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
pip install git+https://github.com/EigenTruth/EigenTruth.git

# 开发模式安装
git clone https://github.com/EigenTruth/EigenTruth.git
cd EigenTruth
pip install -e .[dev]
```

### 使用示例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eigentruth.models.wrapper import EigenTruthWrapper

# 1. 加载原生基座模型
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

# 2. 穿戴 EigenTruth 物理装甲 (在倒数第10层挂载探针，干预强度0.1)
safe_model = EigenTruthWrapper(
    model=base_model,
    target_layer_idx=-10,
    steering_lambda=0.1,
    mahalanobis_threshold=15.0
)

# 3. 探针冷启动 (预热真实流形，传入绝对正确的事实数据)
fact_dataset = ["光速是每秒299792458米。", "地球是圆的。"]
safe_model.warmup(fact_dataset, tokenizer)

# 4. 安全受控的文本生成 (自带双曲熵监测与实时纠偏)
inputs = tokenizer("量子力学证明了灵魂的存在，因为", return_tensors="pt").to("cuda")
outputs = safe_model.generate(**inputs, max_new_tokens=100, temperature=1.5)

print(tokenizer.decode(outputs[0]))
# 若逻辑发散，控制台将自动打印：
# [EigenTruth] ⚠️ 检测到深层语义发散 (HSE > 阈值)，系统可能正在产生幻觉！
```

---

## 📂 项目结构

```
eigentruth/
├── src/
│   └── eigentruth/
│       ├── __init__.py
│       ├── core/
│       │   └── math_engine.py       # 防崩溃数学引擎
│       ├── intervention/
│       │   └── hooks.py             # 动态探针与 Hook 系统
│       └── models/
│           └── wrapper.py           # HuggingFace 顶层封装器
├── tests/
│   ├── test_math_engine.py          # 数学引擎测试 (31 tests)
│   ├── test_hooks.py                # Hook 系统测试 (17 tests)
│   └── test_wrapper.py              # 封装器测试 (14 tests)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🧪 测试

```bash
# 运行全量测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=eigentruth --cov-report=term-missing

# 代码质量检查
ruff check src/
```

---

## 📖 API 参考

### `EigenTruthWrapper`

```python
EigenTruthWrapper(
    model: nn.Module,           # HuggingFace CausalLM 模型
    target_layer_idx: int = -10,       # 探针挂载层索引
    steering_lambda: float = 0.1,      # 引导强度 (0=纯监测)
    mahalanobis_threshold: float = 15.0,  # 距离阈值
    hse_warning_threshold: float = 5.0,   # HSE 预警阈值
)
```

| 方法 | 说明 |
|------|------|
| `warmup(fact_dataset, tokenizer)` | 用事实语料构建真值流形 |
| `generate(**kwargs)` | 受控文本生成（自动监测+干预） |
| `get_diagnostics()` | 获取运行时诊断信息 |
| `detach_probe()` | 移除探针，恢复原始模型 |

### 底层数学 API

```python
from eigentruth.core.math_engine import (
    TruthManifold,              # 真值流形数据结构
    sherman_morrison_update,    # 在线协方差逆更新
    mahalanobis_distance,       # 马氏距离
    poincare_map,               # 庞加莱球映射
    hyperbolic_semantic_entropy,  # 双曲语义熵
)
```

---

## 🤝 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 License

[Apache License 2.0](LICENSE)
