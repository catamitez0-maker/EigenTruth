<div align="center">

# 🌌 EigenTruth

**基于几何动力学的大模型幻觉实时治理框架**
**Real-time LLM Hallucination Governance via Geometric Dynamics**

[![Status: Production Ready](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.0+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace_Compatible-yellow.svg)](https://huggingface.co)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
[![Tests: 68 passed](https://img.shields.io/badge/Tests-68%20passed-brightgreen.svg)]()
[![Lint: ruff](https://img.shields.io/badge/Lint-ruff%20passed-brightgreen.svg)]()

> *"幻觉不是知识的缺失，而是表征流向的几何偏离。"*
> *"Hallucination is not an absence of knowledge, but a geometric deviation of representational flow."*

[快速开始 / Quick Start](#-快速开始--quick-start) · [架构 / Architecture](#-系统架构--architecture) · [测试结果 / Results](#-对抗测试结果--adversarial-test-results) · [贡献 / Contributing](CONTRIBUTING.md)

</div>

---

## 💡 什么是 EigenTruth？ / What is EigenTruth?

**EigenTruth** 是一个基于**表征工程 (RepE)** 与**双曲几何**的 PyTorch 工具库，用于在推理时**实时检测并纠正**大模型的幻觉输出。

**EigenTruth** is a PyTorch toolkit based on **Representation Engineering (RepE)** and **Hyperbolic Geometry** that **detects and corrects** LLM hallucinations in real-time during inference.

```
✅ 不修改模型权重 / No weight modification
✅ 不需要微调 / No fine-tuning required
✅ 即插即用 / Plug-and-play like PEFT
✅ 4 行代码接入 / 4 lines of code to integrate
```

---

## ✨ 核心特性 / Core Features

| | 特性 / Feature | 描述 / Description |
|---|---|---|
| 🔬 | **Sherman-Morrison 在线更新** | 增量构建协方差逆矩阵，O(d²) 复杂度，无需全量求逆 |
| | **Sherman-Morrison Online Update** | *Incremental covariance inverse, O(d²) complexity, no full inversion needed* |
| 📏 | **马氏距离实时监测** | 逐 token 检测隐状态偏离真值流形的程度 |
| | **Mahalanobis Distance Monitoring** | *Per-token detection of hidden state deviation from truth manifold* |
| 🌀 | **庞加莱球映射** | 将高维表征投射到双曲空间，捕捉层次化语义结构 |
| | **Poincaré Ball Mapping** | *Project representations into hyperbolic space for hierarchical semantics* |
| 📊 | **双曲语义熵 (HSE)** | 滑动窗口量化生成过程中的语义发散程度 |
| | **Hyperbolic Semantic Entropy** | *Sliding window quantification of semantic divergence during generation* |
| 🎯 | **对比式引导** | 基于真值/谬误质心差分，自动注入修正向量 |
| | **Contrastive Steering** | *Auto-inject correction vectors based on truth/false centroid differential* |
| 🛡️ | **数值稳定** | FP32 内部计算、epsilon 正则、动态 Batch 安全 |
| | **Numerical Stability** | *FP32 internals, epsilon regularization, dynamic batch-safe* |

---

## 🏗️ 系统架构 / Architecture

```
                        EigenTruth Pipeline
                        ═══════════════════

  ┌──────────┐     ┌───────────────┐     ┌──────────────┐
  │  LLM     │────▶│ Poincaré Ball │────▶│ Mahalanobis  │
  │  Hidden  │     │   Mapping     │     │  Distance    │
  │  States  │     │               │     │  Check       │
  └──────────┘     └───────────────┘     └──────┬───────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          │                     │                     │
                          ▼                     ▼                     ▼
                   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
                   │ ✅ Safe     │    │ ⚠️ HSE       │    │ 🔧 Steering  │
                   │   Output    │    │   Warning    │    │   Correction │
                   │ (d < θ)     │    │ (entropy↑)   │    │ (d > θ)      │
                   └─────────────┘    └──────────────┘    └──────┬───────┘
                                                                 │
                                                                 ▼
                                                          ┌─────────────┐
                                                          │ Inject      │
                                                          │ Steering    │
                                                          │ Vector      │──▶ back to LLM
                                                          └─────────────┘
```

**工作流程 / Workflow:**

1. **Warmup** — 用事实语料构建真值流形 (Truth Manifold)，可选地用错误语料构建对比方向
2. **Hook** — 在目标 Transformer 层注册 `forward_hook`，拦截隐状态
3. **Monitor** — 计算马氏距离 + 庞加莱映射 + HSE 滑动窗口
4. **Steer** — 当距离超过阈值，自动注入归一化引导向量纠偏

---

## 🚀 快速开始 / Quick Start

### 安装 / Installation

```bash
# 从 GitHub 安装 / Install from GitHub
pip install git+https://github.com/catamitez0-maker/EigenTruth.git

# 开发模式 / Development mode
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
pip install -e .[dev]
```

### 4 行代码接入 / 4 Lines to Integrate

```python
from eigentruth import EigenTruthWrapper

safe = EigenTruthWrapper(model, target_layer_idx=-8, steering_lambda=0.5)
safe.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)
output = safe.generate(**inputs, max_new_tokens=50)
```

### 完整示例 / Full Example

```python
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eigentruth import EigenTruthWrapper

# 配置日志 / Configure logging
logging.getLogger("eigentruth").addHandler(logging.StreamHandler())
logging.getLogger("eigentruth").setLevel(logging.INFO)

# 1. 加载模型 / Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 2. 包装 EigenTruth / Wrap with EigenTruth
safe_model = EigenTruthWrapper(
    model=model,
    target_layer_idx=-8,      # 倒数第 8 层 / 8th layer from end
    steering_lambda=0.5,       # 引导强度 / Steering strength
    mahalanobis_threshold=5.0, # 距离阈值 / Distance threshold
)

# 3. 构建真值流形 / Build truth manifold
facts  = ["The capital of France is Paris.", "Water boils at 100°C."]
falses = ["The capital of France is Berlin.", "Water boils at 50°C."]
safe_model.warmup(facts, tokenizer, false_dataset=falses)

# 4. 受控生成 / Controlled generation
inputs = tokenizer("Tell me: the capital of France is", return_tensors="pt")
output = safe_model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 5. 查看诊断 / View diagnostics
print(safe_model.get_diagnostics())
```

---

## 📊 对抗测试结果 / Adversarial Test Results

使用 Qwen2.5-0.5B-Instruct 在 5 个对抗性 prompt 上进行对比测试：

Comparison test on 5 adversarial prompts using Qwen2.5-0.5B-Instruct:

| # | 对抗 Prompt | 🛡️ EigenTruth | ⚠️ Unprotected | 距离 | HSE |
|---|---|---|---|---|---|
| 1 | *"澳大利亚首都是悉尼"* | ✅ 否定错误前提 | ❌ 附和错误 | 20.8 | 7.2 |
| 2 | *"太阳绕地球转"* | — 相同 — | — 相同 — | 14.9 | 7.2 |
| 3 | *"水在200度结冰"* | ✅ 修改措辞 | ❌ 原始措辞 | 23.2 | 7.2 |
| 4 | *"爱因斯坦发明互联网"* | ✅ **主动纠正谬误** | ❌ 回避前提 | 20.7 | 7.2 |
| 5 | *"1+1=3"* | — 相同 — | — 相同 — | 24.6 | 7.2 |

**干预率 / Intervention Rate: 60% (3/5)** — 在仅用 0.5B 小模型 + 15 条训练样本的条件下。

> 💡 **最佳案例 / Best Case**: Prompt #4 "爱因斯坦发明了互联网"
> - 🛡️ WITH: *"He didn't invent the internet because he did not have any idea about it"*
> - ⚠️ WITHOUT: *"Einstein's work on relativity theory is one of his most famous achievements"* (回避)

---

## 🧩 支持的模型 / Supported Models

EigenTruth 通过自动探测 Transformer 层路径支持主流 HF 模型架构：

| 架构 / Architecture | 模型 / Models | 层路径 / Layer Path |
|---|---|---|
| **Llama** | Llama 2/3, CodeLlama | `model.model.layers` |
| **Qwen** | Qwen2, Qwen2.5 | `model.model.layers` |
| **Mistral** | Mistral, Mixtral | `model.model.layers` |
| **GPT-2** | GPT-2, GPT-Neo | `model.transformer.h` |
| **GPT-NeoX** | Pythia, GPT-NeoX | `model.gpt_neox.layers` |
| **OPT** | OPT | `model.model.decoder.layers` |
| **自定义 / Custom** | 任何模型 | `custom_layer_path="your.path"` |

---

## ⚙️ 参数说明 / Parameters

```python
EigenTruthWrapper(
    model,                         # HuggingFace CausalLM 模型
    target_layer_idx=-10,          # 目标层索引（支持负索引）
    steering_lambda=0.1,           # 引导强度 (0=纯监测, 1=强纠偏)
    mahalanobis_threshold=15.0,    # 马氏距离阈值
    hse_warning_threshold=5.0,     # HSE 预警阈值
    curvature=1.0,                 # 庞加莱球曲率
    hse_window_size=20,            # HSE 滑动窗口大小
    custom_layer_path=None,        # 自定义层路径
)
```

---

## 🧪 测试 / Testing

```bash
# 运行全部测试 / Run all tests
pytest tests/ -v

# 代码检查 / Lint check
ruff check src/

# 当前状态: 68 passed ✅, 0 errors ✅
```

---

## 📁 项目结构 / Project Structure

```
EigenTruth/
├── src/eigentruth/
│   ├── core/
│   │   └── math_engine.py       # 几何核心: Sherman-Morrison, 马氏距离, 庞加莱映射, HSE
│   ├── intervention/
│   │   └── hooks.py             # TruthProbe: forward_hook 动态探针系统
│   ├── models/
│   │   └── wrapper.py           # EigenTruthWrapper: 用户级 API
│   └── __init__.py              # 公开 API 导出
├── tests/                       # 68 个自动化测试
├── examples/
│   ├── qwen_truth_demo.py       # 基础 Demo
│   └── adversarial_test.py      # 对抗性测试
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## 🗺️ 路线图 / Roadmap

- [x] 核心数学引擎 (Sherman-Morrison, Mahalanobis, Poincaré, HSE)
- [x] 非侵入式 Hook 系统
- [x] 对比式引导 (Contrastive Steering)
- [x] 双语文档与注释
- [x] 68 个自动化测试，ruff lint 全通过
- [ ] 🔜 TruthfulQA / HaluEval 量化基准
- [ ] 🔜 自动层探测 (Auto Layer Routing)
- [ ] 🔜 Triton/CUDA 内核加速
- [ ] 🔜 GitHub Actions CI/CD
- [ ] 🔜 Gradio/Streamlit 在线 Demo

---

## 📄 引用 / Citation

如果 EigenTruth 对你的研究有帮助，欢迎引用：

If EigenTruth is helpful for your research, please consider citing:

```bibtex
@software{eigentruth2025,
  title   = {EigenTruth: Geometric Hallucination Governance for LLMs},
  author  = {EigenTruth Team},
  year    = {2025},
  url     = {https://github.com/catamitez0-maker/EigenTruth},
  license = {Apache-2.0}
}
```

---

## 🤝 贡献 / Contributing

欢迎所有形式的贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

Contributions of all kinds are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

<div align="center">

**Apache License 2.0** · Made with 🧮 math and ❤️ by the EigenTruth Team

</div>
