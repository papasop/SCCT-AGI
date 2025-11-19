# SCCT-AGI: AI 推理轨迹的结构压缩复杂性理论

> 提出 **SCCT (Structural Compression Complexity Theory)**，将计算硬度理论提升到新的维度。我们证明，AI 推理的计算成本不仅由规模 $N$ 决定，还由其**思考轨迹的结构复杂性** ($\lambda_B$) 决定。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17644809.svg)](https://doi.org/10.5281/zenodo.17644809)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 核心突破：AGI 推理的“第二标度律”

SCCT 理论是计算复杂性理论的范式转移：它将 **$\lambda_B$ (条件结构不可压缩性)** 定义为继 **规模 $N$** 之后的第二个、可测量的预测轴心。 $\lambda_B$ 衡量的是 AGI 思考轨迹的信息论效率。

### SCCT 核心定律 (基于 gpt-4.1-mini 真实 API 轨迹)

我们在真实 LLM (gpt-4.1-mini) 的**思维链 (CoT)** 轨迹上验证了 SCCT 定律，取得了极高精度的拟合结果。

| 定律 | 模型形式 ($\log_2 Y \approx a \cdot \lambda_B + b \cdot \log_2 N + c$) | $\mathbf{R^2}$ (拟合优度) | 关键洞察 |
| :--- | :--- | :--- | :--- |
| **结构-工作量定律** ($\mathcal{W}$) | $a=8.908, b=0.401$ | **0.983** | **AGI 思考成本** ($\mathcal{W}$) 几乎被 $\lambda_B$ 和 $N$ **完美解释**。$b=0.401$ 表明工作量增长呈强次线性。 |
| **结构-时间定律** ($\mathcal{T}$) | $a=3.746, b=0.245$ | **0.862** | **实时延迟** ($\mathcal{T}$) 预测的重大突破。尽管存在网络噪音，$\lambda_B$ 仍能解释 **86.2\%** 的时间差异。 |

---

## 🔬 理论论证：智能与 $\lambda_B$ 的关系

我们的实验通过对比不同推理策略，量化了 **“智能”** 的信息论定义：

| 策略 (AGI 模式) | 思考轨迹 | 平均 $\lambda_B$ | 平均时间 $T$ | SCCT 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **S1\_verbose\_cot** | 冗余/发散思维 | **0.905** (最高) | **$\sim 6.9 \text{s}$** (最高) | **低效：** 轨迹信息熵高，难压缩。 |
| **S2\_compact\_cot** | **简洁/结构化思维** | **0.693** (最低) | **$\sim 2.4 \text{s}$** (最低) | **高效：** 思考轨迹结构化，将时间成本降低了 **65\%**。 |

> **SCCT 结论：** 智能是系统以**最小化其推理轨迹 $\lambda_B$** 的方式解决问题的能力。$\lambda_B$ 提供了衡量 AGI 效率的通用科学语言。

---

## 🛠️ 如何重现实验

本仓库包含用于执行 **SCCT-AGI v2** 实验的 Python 代码 (`SCCT-AGI.py`)。

### 依赖项

```bash
pip install numpy scipy requests openai pandas zlib
