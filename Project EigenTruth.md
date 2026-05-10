这是一份为您量身定制的、达到顶级开源项目标准的开发者 Markdown 文档。

---

Markdown

\# 🌌 Project EigenTruth: 架构规格与开发指南 (v1.0-MVP)

\[\!\[Status: MVP Alpha\](https://img.shields.io/badge/Status-MVP\_Alpha-orange.svg)\]()  
\[\!\[Framework: PyTorch\](https://img.shields.io/badge/Framework-PyTorch-red.svg)\]()  
\[\!\[Model: HuggingFace\](https://img.shields.io/badge/Model-HuggingFace-yellow.svg)\]()

\> **\*\*核心公理\*\***：“幻觉不是知识的缺失，而是表征流向的几何偏离。”  
\> **\*\*项目定位\*\***：基于几何动力学与表征工程（RepE）的大模型幻觉治理 PyTorch 工具库。  
\> **\*\*设计哲学\*\***：非侵入式、即插即用（类似 \`PEFT\`）、纯 PyTorch 实现（MVP阶段暂不涉及底层 Triton 算子重写）。

\---

\#\# 📑 1\. 系统架构蓝图

EigenTruth 采用非侵入式的 Wrapper 设计，完全解耦数学计算、模型拦截与用户接口：

\`\`\`text  
\[ Input Prompt \]   
       │  
       ▼  
┌─────────────────────────────────────────────────────────┐  
│              EigenTruth Wrapper (nn.Module)             │  
│                                                         │  
│  ┌─────────────────┐    \[1. 离线真值流形基建 (Warmup)\]    │  
│  │ LLM Transformer │ \<── 提取事实质心(Mean)与协方差(Cov)   │  
│  │                 │                                    │  
│  │                 │    \[2. 实时真值探针 Hook\]            │  
│  │                 │ \<── 拦截特定层 Hidden States          │  
│  │                 │ ──\> 计算马氏距离 (Sherman-Morrison)  │  
│  │                 │ \<── 注入激活引导向量 (Steering)       │  
│  └─────────────────┘                                    │  
│       │                                                 │  
│       ▼                                                 │  
│  \[3. 庞加莱映射器\]                                        │  
│  计算双曲语义熵 (HSE)                                     │  
│  判定逻辑崩塌并输出高危预警                                │  
└─────────────────────────────────────────────────────────┘  
       │  
       ▼  
\[ Output Token \]

## ---

**📂 2\. 目录结构与模块定义**

系统被严格划分为三个解耦模块，**开发必须严格按以下顺序推进，不可乱序**：

Plaintext

eigentruth/  
├── \_\_init\_\_.py  
├── core/  
│   └── math\_engine.py       \# 防崩溃数学引擎 (Sherman-Morrison更新 / 双曲测地线映射)  
├── intervention/  
│   └── hooks.py             \# 动态探针与挂载钩子 (PyTorch forward\_hook 拦截器)  
└── models/  
    └── wrapper.py           \# HF 顶层封装器 (接管 generate 与 warmup)

## ---

**🚧 3\. 工程避坑指南 (Engineering Guardrails)**

在编写具体代码时，必须严格遵守以下系统级防线（**极其重要**）：

1. **数值崩溃防线 (Numerical Drift)**：  
   * 在 sherman\_morrison\_update 中，连续的 FP16 矩阵相乘会迅速导致非正定。**内部累加运算必须强制转为 torch.float32 计算**，分母必须加上正则项 epsilon (1e-6) 防除零，计算完成后再转回原精度。  
2. **显存泄漏防线 (OOM Leak)**：  
   * Hook 内部截获的隐状态必须调用 .detach() 脱离计算图。  
   * 所有探针运算必须包裹在 @torch.no\_grad() 或 with torch.no\_grad(): 中，严禁将监控用的历史变量保留在显存中。  
3. **张量形状对齐 (Tensor Shape Alignment)**：  
   * HF 模型的 hidden\_states 通常是一个 Tuple。需提取实际张量（如 args\[0\]）。  
   * 自回归生成时，隐状态形状为 \[Batch, Seq\_len, Hidden\_dim\]。计算马氏距离和介入时，**只能切片操作最新生成的最后一个 Token，即 h\[:, \-1:, :\]**，并通过 .squeeze()/.unsqueeze() 对齐广播维度。

## ---

**💻 4\. 最终用户 API 示例 (Target UX)**

框架开发完毕后，用户应当能用以下代码无缝运行：

Python

import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from eigentruth.models.wrapper import EigenTruthWrapper

\# 1\. 加载原生基座模型  
model\_name \= "Qwen/Qwen2.5-0.5B"  
tokenizer \= AutoTokenizer.from\_pretrained(model\_name)  
base\_model \= AutoModelForCausalLM.from\_pretrained(model\_name, device\_map="cuda")

\# 2\. 穿戴 EigenTruth 物理装甲 (在倒数第10层挂载探针，干预强度0.1)  
safe\_model \= EigenTruthWrapper(  
    model=base\_model,  
    target\_layer\_idx=-10,  
    steering\_lambda=0.1,  
    mahalanobis\_threshold=15.0  
)

\# 3\. 探针冷启动 (预热真实流形，传入绝对正确的事实数据)  
fact\_dataset \= \["光速是每秒299792458米。", "地球是圆的。"\]  
safe\_model.warmup(fact\_dataset, tokenizer)

\# 4\. 安全受控的文本生成 (自带双曲熵监测与实时纠偏)  
inputs \= tokenizer("量子力学证明了灵魂的存在，因为", return\_tensors="pt").to("cuda")

\# 即使开启高温生成 (temperature=1.5)，推理轨迹也会被物理锚定  
outputs \= safe\_model.generate(\*\*inputs, max\_new\_tokens=100, temperature=1.5)

print(tokenizer.decode(outputs\[0\]))  
\# 若逻辑发散，控制台将自动打印：\[EigenTruth\] ⚠️ 检测到深层语义发散 (HSE \> 阈值)，系统可能正在产生幻觉！

## ---

**🤖 5\. 给 AI 编程助手的交互协议 (LLM Prompting Protocol)**

如果你是辅助开发的 AI（如 Claude）：

1. **请认真阅读并理解上述架构、模块定义和排雷指南**。  
2. **绝对禁止一次性写完所有代码**。我们需要采用敏捷开发的模式。  
3. 如果你已经完全理解了本项目的世界观，请**不要输出任何 Python 代码**，只需回复下方加粗的确认语：  
   **“✅ EigenTruth 架构规范已加载，雷区防线已确认。我是您的 AI 架构师，请主架构师下达 Phase 1 的开发指令。”**

