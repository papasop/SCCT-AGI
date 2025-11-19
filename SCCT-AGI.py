# ============================================
# SCCT-AGI v1 论文验证版 Colab（离线 Mock 版）
# - 无需 API Key
# - 完整验证 SCCT-AGI mock 实验
# ============================================

import zlib
import json
import time
import math
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 为了可重复性（固定随机种子）
random.seed(2025)
np.random.seed(2025)

# ============================
# 1. 压缩与 λ_B 定义（与论文公式一致）
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
    C_cond = max(C_joint - C_p - C_SEP, 1)
    lam = C_cond / (C_p + C_cond)
    return lam, C_cond, C_p

# ============================
# 2. 本地“算式求值器” + 模拟 LLM 三种思考策略
# ============================

def eval_expr_python_style(expr: str) -> float:
    """
    把表达式里的 ^ 换成 **，然后用 Python eval 计算出数值。
    仅用于本地模拟，不暴露给外部。
    """
    safe_expr = expr.replace("^", "**")
    # 简单安全检查：只允许数字和运算符
    if not re.fullmatch(r"[0-9\+\-\*\/\^\(\)\.\s]+", expr.replace("^", "")):
        raise ValueError(f"非法字符出现在表达式中: {expr}")
    return eval(safe_expr, {"__builtins__": None}, {})

def build_mock_trace(problem_str: str, strategy: str) -> str:
    """
    根据 strategy 构造不同风格的“LLM 轨迹文本”。
    三种策略：
      S0_direct      : 只输出结果，无过程（最短 trace）
      S1_verbose_cot : 极其啰嗦的 step-by-step（最长 trace）
      S2_compact_cot : 精炼、有结构的 CoT（中等长度）
    """
    value = eval_expr_python_style(problem_str)

    if strategy == "S0_direct":
        # 模拟“直接回答，不给过程”
        return f"{value}"

    elif strategy == "S1_verbose_cot":
        # 非常啰嗦的 CoT：重复解释、分行叙述
        lines = []
        lines.append(f"我们来非常详细地计算这个表达式：{problem_str}")
        lines.append("第一步：按照运算优先级，先处理所有幂运算 (^)。")
        lines.append("第二步：在幂运算结果的基础上，计算所有乘法和除法。")
        lines.append("第三步：最后执行加减法，得到最终结果。")
        lines.append(f"经过严格的三步计算，这个算式的最终数值结果是：{value}。")
        # 人为增加冗余，模拟“思维散乱但正确”
        for i in range(8):
            lines.append(
                f"再次确认第 {i+1} 次：表达式 {problem_str} 的值的确等于 {value}，"
                "这里我们只是重复说明这一事实，以确保没有遗漏任何细节。"
            )
        return "\n".join(lines)

    elif strategy == "S2_compact_cot":
        # 紧凑 CoT：少量关键步骤
        return (
            f"对算式 {problem_str} 按照“先幂运算、再乘除、后加减”的顺序计算，"
            f"可以得到最终结果为 {value}。"
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def call_llm(problem_str: str, strategy: str):
    """
    Mock LLM 版本：
      - 不调用任何外部 API
      - 用本地 eval 计算数值
      - 用不同模板生成 Trace_Token
    """
    start = time.perf_counter()
    trace_text = build_mock_trace(problem_str, strategy)
    end = time.perf_counter()
    T = end - start
    return trace_text, T

# ============================
# 3. 生成一批算术任务（与日志一致）
# ============================

def random_expr(depth=3):
    """
    生成一个带 + - * / ^ 的算术表达式：
      depth 控制“块”的数量，可视为 N(x) 的 proxy
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
# 4. 主实验：Mock-LLM + λ_B
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
                    "model": "Mock-LLM",
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
                "model": "Mock-LLM",
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
# 7. 各策略统计摘要（写论文用的表）
# ============================

print("\n=== 各策略 λ_B / W / C_cond 统计摘要 ===")
summary = (
    df_llm
    .groupby("strategy")[["lambda_B", "trace_len_bytes", "C_cond", "N"]]
    .agg(["mean", "std", "min", "max"])
)
print(summary)

# 你可以在论文里挑一两列放成 Table:
#   strategy vs mean λ_B, mean trace_len_bytes, mean C_cond

# ============================
# 8. 图像：W vs λ_B（用于论文 Figure）
# ============================

# 仅简单散点+回归平面在 λ_B 方向的投影
plt.figure(figsize=(6, 4))
mask = df_llm["lambda_B"].notna() & (df_llm["trace_len_bytes"] > 0)
d_plot = df_llm[mask].copy()
d_plot["log2_W"] = np.log2(d_plot["trace_len_bytes"])

plt.scatter(d_plot["lambda_B"], d_plot["log2_W"])
plt.xlabel("λ_B (structural incompressibility)")
plt.ylabel("log₂ W(x) (trace length)")
plt.title("Structure–Work Law (All strategies, mock)")

# 画一条在 λ_B 上的简单线性拟合（忽略 N 项，只作为可视化）
coef_simple = np.polyfit(d_plot["lambda_B"], d_plot["log2_W"], 1)
lam_grid = np.linspace(d_plot["lambda_B"].min(), d_plot["lambda_B"].max(), 50)
log2W_fit = coef_simple[0] * lam_grid + coef_simple[1]
plt.plot(lam_grid, log2W_fit)

plt.tight_layout()
plt.show()

# ============================
# 9. 图像：W vs N（log₂W 对 log₂N，规模项）
# ============================

plt.figure(figsize=(6, 4))
plt.scatter(np.log2(d_plot["N"]), d_plot["log2_W"])
plt.xlabel("log₂ N (problem scale)")
plt.ylabel("log₂ W(x) (trace length)")
plt.title("Scaling of Work with Problem Size N (mock)")
plt.tight_layout()
plt.show()

# ============================
# 10. 导出 CSV（可作为论文补充材料）
# ============================

csv_path = "scct_agi_mock_v1.csv"
df_llm.to_csv(csv_path, index=False)
print(f"\n已导出数据到: {csv_path}")

print("\n============================")
print("SCCT-AGI v1 论文验证版实验完成 ✅")
print("============================")
