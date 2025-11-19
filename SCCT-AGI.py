# ============================================
# SCCT-AGI v2 · 真实 LLM 版（NeurIPS 论文验证用）
# - 使用 OpenAI 官方 Python SDK
# - 从 Colab Secret 中读取 OPENAI_API_KEY
# - 采集真实 LLM trace，计算 λ_B、回归结构–工作量/时间律
# ============================================

!pip install -q --upgrade openai matplotlib numpy pandas

import os
import time
import json
import math
import random
import re
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 为了可重复性（随机表达式生成用）
random.seed(2025)
np.random.seed(2025)

# ============================
# 0. 读取 OPENAI_API_KEY
# ============================

API_KEY = None

# 先尝试从 Colab Secret 读取（你刚刚配置的）
try:
    from google.colab import userdata
    API_KEY = userdata.get("OPENAI_API_KEY")
except Exception:
    API_KEY = None

# 如果 Secret 里没有，就再试试环境变量
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "未检测到 OPENAI_API_KEY。\n"
        "请在左侧侧边栏：钥匙图标(Secret) → 添加密钥：\n"
        "  名称: OPENAI_API_KEY\n"
        "  值:   你的 OpenAI 密钥\n"
        "然后重新运行本单元。"
    )

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# 你可以在这里改模型名称
MODEL_NAME = "gpt-4.1-mini"

print("✅ 已成功读取 OPENAI_API_KEY，当前模型:", MODEL_NAME)


# ============================
# 1. 压缩与 λ_B 定义（与论文一致）
# ============================

def C_t(s: bytes) -> int:
    """zlib level=9 压缩后的字节长度"""
    return len(zlib.compress(s, level=9))

SEP = b"||"
C_SEP = C_t(SEP)

def encode_problem(problem_obj) -> bytes:
    """Problem 统一编码（JSON + UTF-8）"""
    return json.dumps(problem_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")

def encode_trace(trace_text: str) -> bytes:
    """Trace_Token 编码（UTF-8 文本）"""
    return trace_text.encode("utf-8")

def lambda_B(problem_bytes: bytes, trace_bytes: bytes):
    """
    论文中的 λ_B 定义：
      C_p = C_t(problem)
      C_joint = C_t(problem || SEP || trace)
      C_cond = max(C_joint - C_p - C_SEP, 1)
      λ_B = C_cond / (C_p + C_cond)
    返回: (λ_B, C_cond, C_p)
    """
    C_p = C_t(problem_bytes)
    C_joint = C_t(problem_bytes + SEP + trace_bytes)
    C_cond_raw = C_joint - C_p - C_SEP
    C_cond = max(C_cond_raw, 1)
    lam = C_cond / (C_p + C_cond)
    return lam, C_cond, C_p


# ============================
# 2. 任务生成：随机算术表达式
# ============================

def random_expr(depth=3):
    """
    生成一个带 + - * / ^ 的算术表达式：
      depth 控制“块”的数量，可视为 N(x)
    """
    ops = ["+", "-", "*", "/"]
    parts = []
    for _ in range(depth):
        a = random.randint(2, 50)
        b = random.randint(2, 10)
        op = random.choice(ops)
        if random.random() < 0.3:
            # 随机插入幂运算
            term = f"({a}^{random.randint(2,3)})"
        else:
            term = f"{a}{op}{b}"
        parts.append(term)
    return " + ".join(parts)

def build_problem_set(num_problems=15, min_depth=3, max_depth=7):
    """
    构造一批算术题：
      - expr: 表达式字符串
      - N   : 运算块数量，作为规模参数
    """
    problems = []
    for i in range(num_problems):
        depth = random.randint(min_depth, max_depth)
        expr = random_expr(depth=depth)
        problems.append({
            "id": i,
            "expr": expr,
            "N": depth,
        })
    return problems

problems = build_problem_set()
print(f"生成问题数量: {len(problems)}")
print("示例:", problems[0])


# ============================
# 3. 调用真实 LLM：三种思考策略
# ============================

def call_llm(expr: str, strategy: str, model: str = MODEL_NAME):
    """
    使用真实 LLM 获取“思维轨迹”：
      S0_direct      : 只输出最终数字答案
      S1_verbose_cot : 极其啰嗦的 step-by-step
      S2_compact_cot : 精炼 CoT
    返回: (trace_text, wall_clock_T)
    """
    system_prompt = (
        "You are a precise arithmetic reasoning engine. "
        "You must evaluate the given expression exactly. "
        "The expression uses +, -, *, /, ^ for power, and parentheses."
    )

    if strategy == "S0_direct":
        user_prompt = (
            f"Solve the following arithmetic expression exactly:\n\n{expr}\n\n"
            "Respond with ONLY the final numeric result.\n"
            "- Do NOT show any intermediate steps.\n"
            "- Do NOT add explanations.\n"
            "- Output just the number (or a single signed float)."
        )
    elif strategy == "S1_verbose_cot":
        user_prompt = (
            f"Solve the following arithmetic expression exactly:\n\n{expr}\n\n"
            "Think step by step in great detail.\n"
            "- Explain each intermediate computation.\n"
            "- Use natural language paragraphs.\n"
            "- Be intentionally verbose and somewhat repetitive.\n"
            "- Around 300–600 words is OK.\n"
            "At the end, clearly state the final numeric result."
        )
    elif strategy == "S2_compact_cot":
        user_prompt = (
            f"Solve the following arithmetic expression exactly:\n\n{expr}\n\n"
            "Provide a brief but clear chain-of-thought:\n"
            "- 3–6 short bullet points or sentences.\n"
            "- Each step should be concise.\n"
            "Then give the final numeric result."
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    t1 = time.perf_counter()

    trace_text = resp.choices[0].message.content
    T = t1 - t0
    return trace_text, T


# ============================
# 4. 主实验：真实 LLM + λ_B
# ============================

def run_llm_scct_experiment(
    problems,
    strategies=("S0_direct", "S1_verbose_cot", "S2_compact_cot"),
):
    rows = []
    for prob in problems:
        expr = prob["expr"]
        N = prob["N"]
        prob_obj = {"type": "arith", "expr": expr, "N": N}
        prob_bytes = encode_problem(prob_obj)

        print(f"\n=== Problem #{prob['id']}: {expr} (N={N}) ===")

        for strat in strategies:
            print(f"  Strategy {strat} ... ", end="", flush=True)

            try:
                trace_text, T = call_llm(expr, strat)
            except Exception as e:
                print(f"[Error] {e}")
                row = {
                    "problem_id": prob["id"],
                    "expr": expr,
                    "N": N,
                    "strategy": strat,
                    "T_sec": np.nan,
                    "trace_len_bytes": 0,
                    "lambda_B": np.nan,
                    "C_cond": np.nan,
                    "C_p": C_t(prob_bytes),
                    "model": MODEL_NAME,
                    "error": str(e),
                }
                rows.append(row)
                continue

            trace_bytes = encode_trace(trace_text)
            lam, C_cond, C_p = lambda_B(prob_bytes, trace_bytes)

            row = {
                "problem_id": prob["id"],
                "expr": expr,
                "N": N,
                "strategy": strat,
                "T_sec": T,
                "trace_len_bytes": len(trace_bytes),
                "lambda_B": lam,
                "C_cond": C_cond,
                "C_p": C_p,
                "model": MODEL_NAME,
                "error": "",
            }
            rows.append(row)
            print(
                f"λ_B={lam:.4f}, T={T:.6f}s, "
                f"|trace|={len(trace_bytes)}, C_cond={C_cond}"
            )
    return pd.DataFrame(rows)

df_llm = run_llm_scct_experiment(problems)
print("\n=== 实验数据预览 ===")
print(df_llm.head())


# ============================
# 5. 标度律拟合：结构–时间
#   log2 T(x) ≈ a·λ_B + b·log2 N + c
# ============================

def regression_structure_time(df, strategy=None):
    d = df.copy()
    d = d[(d["T_sec"] > 0) & d["lambda_B"].notna()]
    if strategy is not None:
        d = d[d["strategy"] == strategy]
    if len(d) < 5:
        print(f"[Time] 样本太少 (len={len(d)})")
        return None
    
    d = d.copy()
    d["log2_T"] = np.log2(d["T_sec"])
    d["log2_N"] = np.log2(d["N"])

    X = np.vstack([
        d["lambda_B"].values,
        d["log2_N"].values,
        np.ones(len(d))
    ]).T
    y = d["log2_T"].values

    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef

    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot

    print("\n--- 结构–时间标度律拟合 ---")
    if strategy:
        print(f"策略: {strategy}")
    else:
        print("策略: ALL")
    print("log₂ T(x) ≈ a·λ_B + b·log₂ N + c")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, R² = {R2:.3f}")
    return {"a": a, "b": b, "c": c, "R2": R2, "data": d}

reg_time_all = regression_structure_time(df_llm)
reg_time_S0  = regression_structure_time(df_llm, "S0_direct")
reg_time_S1  = regression_structure_time(df_llm, "S1_verbose_cot")
reg_time_S2  = regression_structure_time(df_llm, "S2_compact_cot")


# ============================
# 6. 标度律拟合：结构–工作量
#   log2 W(x) ≈ a·λ_B + b·log2 N + c
# ============================

def regression_structure_work(df, strategy=None):
    d = df.copy()
    d = d[(d["trace_len_bytes"] > 0) & d["lambda_B"].notna()]
    if strategy is not None:
        d = d[d["strategy"] == strategy]
    if len(d) < 5:
        print(f"[Work] 样本太少 (len={len(d)})")
        return None

    d = d.copy()
    d["log2_W"] = np.log2(d["trace_len_bytes"])
    d["log2_N"] = np.log2(d["N"])

    X = np.vstack([
        d["lambda_B"].values,
        d["log2_N"].values,
        np.ones(len(d))
    ]).T
    y = d["log2_W"].values

    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef

    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot

    print("\n--- 结构–工作量标度律拟合 ---")
    if strategy:
        print(f"策略: {strategy}")
    else:
        print("策略: ALL")
    print("log₂ W(x) ≈ a·λ_B + b·log₂ N + c")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, R² = {R2:.3f}")
    return {"a": a, "b": b, "c": c, "R2": R2, "data": d}

reg_work_all = regression_structure_work(df_llm)
reg_work_S0  = regression_structure_work(df_llm, "S0_direct")
reg_work_S1  = regression_structure_work(df_llm, "S1_verbose_cot")
reg_work_S2  = regression_structure_work(df_llm, "S2_compact_cot")


# ============================
# 7. 各策略统计摘要（Table 用）
# ============================

print("\n=== 各策略 λ_B / W / C_cond 统计摘要 ===")
summary = (
    df_llm
    .groupby("strategy")[["lambda_B", "trace_len_bytes", "C_cond", "N"]]
    .agg(["mean", "std", "min", "max"])
)
print(summary)


# ============================
# 8. 图像：W vs λ_B（用于论文 Figure）
# ============================

plt.figure(figsize=(6, 4))
mask = df_llm["lambda_B"].notna() & (df_llm["trace_len_bytes"] > 0)
d_plot = df_llm[mask].copy()
d_plot["log2_W"] = np.log2(d_plot["trace_len_bytes"])

plt.scatter(d_plot["lambda_B"], d_plot["log2_W"])
plt.xlabel("λ_B (structural incompressibility)")
plt.ylabel("log₂ W(x) (trace length)")
plt.title("Structure–Work Law (All strategies, real LLM)")

coef_simple = np.polyfit(d_plot["lambda_B"], d_plot["log2_W"], 1)
lam_grid = np.linspace(d_plot["lambda_B"].min(), d_plot["lambda_B"].max(), 50)
log2W_fit = coef_simple[0] * lam_grid + coef_simple[1]
plt.plot(lam_grid, log2W_fit)

plt.tight_layout()
plt.show()


# ============================
# 9. 图像：W vs N（log₂W 对 log₂N）
# ============================

plt.figure(figsize=(6, 4))
plt.scatter(np.log2(d_plot["N"]), d_plot["log2_W"])
plt.xlabel("log₂ N (problem scale)")
plt.ylabel("log₂ W(x) (trace length)")
plt.title("Scaling of Work with Problem Size N (real LLM)")
plt.tight_layout()
plt.show()


# ============================
# 10. 导出 CSV（补充材料）
# ============================

csv_path = "scct_agi_llm_v2.csv"
df_llm.to_csv(csv_path, index=False)
print(f"\n已导出数据到: {csv_path}")

print("\n============================")
print("SCCT-AGI v2 真实 LLM 实验完成 ✅")
print("============================")

