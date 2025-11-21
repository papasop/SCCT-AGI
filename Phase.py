# %% [markdown]
# # Paper II Colab: Kolmogorov Order Parameter, Phase Transitions and Landau Theory of Intelligence
#
# This notebook implements three experimental schemes:
# - (A) CDCL-only Structure–Time Law (phi -> log2 T)
# - (B) Unified CDCL + CoT scaling (optional, requires OpenAI API key)
# - (C) Landau-style "free energy" of the Kolmogorov order parameter φ(x)
#
# Author: Y.Y.N. Li (structure+comments adapted)


# %%
!pip install -q python-sat[gmpy] tabulate

# 如果你要跑 CoT（方案 B），取消下一行注释并安装 openai
# !pip install -q openai

# %%
import os
import time
import json
import gzip
import math
import random
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from pysat.formula import CNF
from pysat.solvers import Glucose3

# 如果要启用 CoT 实验，请取消下面两行注释，并设置 OPENAI_API_KEY 环境变量
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")


# %% [markdown]
# ## 0. 工具函数：随机 3-SAT、压缩器、序参量 φ(x)


# %%
def generate_random_3sat(n, alpha, rng=None):
    """
    生成随机 3-SAT CNF:
    - n: 变量个数
    - alpha: 子句/变量 比 (m ≈ alpha * n)
    返回: list of clauses, each clause is a list of literals (ints)
    """
    if rng is None:
        rng = random.Random()

    m = int(round(alpha * n))
    clauses = []
    for _ in range(m):
        # 选三个不同变量
        vars_ = rng.sample(range(1, n + 1), 3)
        clause = []
        for v in vars_:
            sign = rng.choice([1, -1])
            clause.append(sign * v)
        clauses.append(clause)
    return clauses


def cnf_to_json_problem(n, alpha, clauses):
    """
    生成 problem JSON 对象，用于压缩。
    """
    obj = {
        "type": "3-SAT",
        "n_vars": n,
        "alpha": float(alpha),
        "n_clauses": len(clauses),
        "clauses": clauses,
    }
    return obj


def cdcl_trace_to_json(n, alpha, model, stats):
    """
    生成 CDCL trace JSON 对象。
    """
    obj = {
        "engine": "CDCL",
        "n_vars": n,
        "alpha": float(alpha),
        "sat": model is not None,
        "model": model if model is not None else [],
        "stats": stats,
    }
    return obj


def gzip_compressed_len(obj) -> int:
    """
    使用 gzip 压缩 JSON 对象，返回字节长度。
    """
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(s)
    return len(buf.getvalue())


def compute_lambda_phi(K_problem: int, K_solution: int):
    """
    λ_K(x) 和 φ(x) = log(1 + λ_K(x)).
    """
    if K_problem <= 0:
        raise ValueError("K_problem must be positive.")
    lambda_k = (K_solution - K_problem) / K_problem
    phi = math.log(1.0 + lambda_k)
    return lambda_k, phi


# %% [markdown]
# ## 1. 运行 CDCL 相变扫描 (方案 A & C 的基础）
#
# - 生成 (n, α) 网格上的随机 3-SAT
# - 用 Glucose3 解
# - 记录:
#   - T_sec, log2 T_sec
#   - 压缩长度 K_problem, K_solution
#   - λ_K, φ
#   - CDCL 统计: conflicts, decisions, propagations


# %%
def solve_3sat_cdcl(n, alpha, instance_id, rng=None):
    if rng is None:
        rng = random.Random()

    clauses = generate_random_3sat(n, alpha, rng=rng)

    # 构造 PySAT CNF
    cnf = CNF()
    for cl in clauses:
        cnf.append(cl)

    solver = Glucose3()
    for cl in cnf.clauses:
        solver.add_clause(cl)

    t0 = time.perf_counter()
    sat = solver.solve()
    t1 = time.perf_counter()
    T_sec = t1 - t0
    log2T = math.log(T_sec, 2) if T_sec > 0 else float("-inf")

    model = solver.get_model() if sat else None

    # 部分 solver 版本没有这些属性，如果失败就给 None
    def safe_attr(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else None

    stats = {
        "conflicts": safe_attr(solver, "nof_conflicts"),
        "decisions": safe_attr(solver, "nof_decisions"),
        "propagations": safe_attr(solver, "nof_propagations"),
    }

    # JSON 编码 + 压缩长度
    problem_json = cnf_to_json_problem(n, alpha, clauses)
    trace_json = cdcl_trace_to_json(n, alpha, model, stats)

    K_problem = gzip_compressed_len(problem_json)
    K_solution = gzip_compressed_len(
        {"problem": problem_json, "solution_trace": trace_json}
    )

    lambda_k, phi = compute_lambda_phi(K_problem, K_solution)

    row = {
        "engine": "CDCL",
        "n": n,
        "alpha": float(alpha),
        "instance_id": instance_id,
        "T_sec": T_sec,
        "log2T": log2T,
        "K_problem": K_problem,
        "K_solution": K_solution,
        "lambda_K": lambda_k,
        "phi": phi,
        "conflicts": stats["conflicts"],
        "decisions": stats["decisions"],
        "propagations": stats["propagations"],
    }
    solver.delete()
    return row


def run_cdcl_phase_sweep(
    n_list=(20, 50, 100),
    alpha_list=(3.5, 4.0, 4.26, 4.5),
    n_instances_per_cell=5,
    seed=0,
):
    rng = random.Random(seed)
    rows = []
    for n in n_list:
        for alpha in alpha_list:
            for inst_id in range(n_instances_per_cell):
                row = solve_3sat_cdcl(n, alpha, inst_id, rng=rng)
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


# %%
print("=== Running CDCL Phase Sweep for Paper II ===")
df_cdcl = run_cdcl_phase_sweep(
    n_list=(20, 50, 100),
    alpha_list=(3.5, 4.0, 4.26, 4.5),
    n_instances_per_cell=5,
    seed=42,
)

print(f"CDCL 样本数: {len(df_cdcl)}\n")
print("--- CDCL 原始数据 (前 5 条) ---")
print(df_cdcl.head().to_markdown(index=True))

# 按 alpha 聚合，看相变行为
group_alpha = (
    df_cdcl.groupby("alpha")[["lambda_K", "phi", "log2T"]]
    .mean()
    .reset_index()
)
print("\n=== CDCL Phase Transition (by alpha) ===")
print(group_alpha.to_markdown(index=False))


# %% [markdown]
# ## 2. 方案 A：CDCL-only 结构–时间律
#
# 1. 单变量回归：log2T ≈ a * φ + b
# 2. 多变量回归：log2T ≈ a * φ + b * log2 n + c


# %%
def fit_linear_1d(x, y):
    """
    简单一维线性回归 y ≈ a x + b, 返回 a, b, R²
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    # a, b from polyfit
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def fit_linear_2d(X, y):
    """
    多元线性回归 y ≈ X @ beta (加常数项), 返回 beta, R²
    X: shape (n_samples, d)
    y: shape (n_samples,)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if len(X) < X.shape[1] + 1:
        return None, np.nan
    # 加截距
    X_design = np.hstack([X, np.ones((len(X), 1))])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


# %%
# 方案 A-1: 单变量 φ -> log2T
a1, b1, r2_1d = fit_linear_1d(df_cdcl["phi"], df_cdcl["log2T"])
print("\n=== 方案 A-1: CDCL-only Structure–Time Law (phi -> log2T) ===")
print(f"log2T ≈ {a1:.3f} * phi + {b1:.3f}")
print(f"R² = {r2_1d:.3f}")

# 方案 A-2: 多变量 φ, log2 n -> log2T
X = np.column_stack(
    [
        df_cdcl["phi"].values,
        np.log2(df_cdcl["n"].values),
    ]
)
beta, r2_2d = fit_linear_2d(X, df_cdcl["log2T"].values)
print("\n=== 方案 A-2: 多元结构–时间律 (phi, log2 n) ===")
if beta is not None:
    a_phi, a_logn, b0 = beta
    print(f"log2T ≈ {a_phi:.3f} * phi + {a_logn:.3f} * log2(n) + {b0:.3f}")
    print(f"R² (multi) = {r2_2d:.3f}")
else:
    print("样本不足，无法拟合多元回归。")

# 简单散点 + 拟合直线
plt.figure()
plt.scatter(df_cdcl["phi"], df_cdcl["log2T"], s=15)
phi_grid = np.linspace(df_cdcl["phi"].min(), df_cdcl["phi"].max(), 100)
plt.plot(phi_grid, a1 * phi_grid + b1)
plt.xlabel("phi (Kolmogorov order parameter)")
plt.ylabel("log2 T (CDCL runtime)")
plt.title("Scheme A: CDCL-only Structure–Time Law")
plt.show()


# %% [markdown]
# ## 3. 方案 B：CDCL + CoT 的统一序参量缩放
#
# - 可选：需要 OpenAI API Key，且模型要支持 CoT 输出。
# - 如果 `OPENAI_API_KEY` 未设置，就跳过 CoT，只保留 CDCL 数据结构。
#
# CoT 结构:
# - problem_json: 与 CDCL 一致（n, alpha, clauses）
# - solution_trace: 包含 CoT 文本、判定 SAT/UNSAT 等


# %%
def run_cot_on_formula(n, alpha, clauses, instance_id, engine_name="CoT"):
    """
    占位版本。如果你要真跑 CoT:
    1. 确保上面 import openai & 设置 openai.api_key
    2. 在这里实现一个调用，拿到 API 时间 T_sec + CoT 文本
    """
    if "openai" not in globals() or openai.api_key is None:
        raise RuntimeError("openai 未可用或未设置 API key。")

    # 构造自然语言描述 (非常简化)
    clause_strs = []
    for cl in clauses:
        lits = []
        for lit in cl:
            var = abs(lit)
            sign = "" if lit > 0 else "NOT "
            lits.append(f"{sign}x{var}")
        clause_strs.append("(" + " OR ".join(lits) + ")")
    problem_text = " AND ".join(clause_strs)
    prompt = (
        "You are a SAT solver. Given the following 3-SAT formula over variables x1..x"
        f"{n}: {problem_text}\n\n"
        "Decide if it is SAT or UNSAT. Think step by step and output your final answer clearly.\n"
    )

    t0 = time.perf_counter()
    # 这里用 ChatCompletion 旧接口示意，实际可换成新的 SDK 写法
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful SAT solver."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    t1 = time.perf_counter()
    T_sec = t1 - t0

    # 提取 CoT 文本
    cot_text = response["choices"][0]["message"]["content"]

    # 非严格解析 SAT/UNSAT，简单做个占位
    sat = "unsat" not in cot_text.lower()

    # problem + trace JSON
    problem_json = cnf_to_json_problem(n, alpha, clauses)
    trace_json = {
        "engine": engine_name,
        "n_vars": n,
        "alpha": float(alpha),
        "instance_id": instance_id,
        "cot_text": cot_text,
        "sat": sat,
    }

    K_problem = gzip_compressed_len(problem_json)
    K_solution = gzip_compressed_len(
        {"problem": problem_json, "solution_trace": trace_json}
    )
    lambda_k, phi = compute_lambda_phi(K_problem, K_solution)

    log2T = math.log(T_sec, 2) if T_sec > 0 else float("-inf")

    row = {
        "engine": engine_name,
        "n": n,
        "alpha": float(alpha),
        "instance_id": instance_id,
        "T_sec": T_sec,
        "log2T": log2T,
        "K_problem": K_problem,
        "K_solution": K_solution,
        "lambda_K": lambda_k,
        "phi": phi,
    }
    return row


def run_cot_phase_sweep(
    n=10,
    alpha_list=(3.5, 4.0, 4.26, 4.5),
    n_instances_per_alpha=2,
    seed=123,
):
    rng = random.Random(seed)
    rows = []
    for alpha in alpha_list:
        for inst_id in range(n_instances_per_alpha):
            clauses = generate_random_3sat(n, alpha, rng=rng)
            row = run_cot_on_formula(n, alpha, clauses, inst_id, engine_name="CoT")
            rows.append(row)
    return pd.DataFrame(rows)


# %%
print("\n==================================================")
print("=== Scheme B: Running CoT Phase Sweep (optional) ===")
print("==================================================")

df_cot = None
if "openai" in globals() and openai.api_key:
    try:
        df_cot = run_cot_phase_sweep(
            n=10,
            alpha_list=(3.5, 4.0, 4.26, 4.5),
            n_instances_per_alpha=2,
            seed=7,
        )
        print(f"CoT 样本数: {len(df_cot)}\n")
        print("--- CoT 原始数据 (前 5 条) ---")
        print(df_cot.head().to_markdown(index=True))
    except Exception as e:
        print("运行 CoT 时出现错误，跳过 CoT 实验：", e)
        df_cot = None
else:
    print("未检测到 openai 或 API key，跳过 CoT 实验。")


# %%
if df_cot is not None:
    # 合并 CDCL + CoT
    df_all = pd.concat([df_cdcl, df_cot], ignore_index=True)
else:
    df_all = df_cdcl.copy()

print("\n=== 方案 B：CDCL + CoT 统一结构–时间律 ===")
print(f"总样本数: {len(df_all)}")
print(df_all.groupby("engine")[["lambda_K", "phi", "log2T"]].agg(["mean", "std"]))


# %%
# 全体数据用 φ 拟合 log2T
a_all, b_all, r2_all = fit_linear_1d(df_all["phi"], df_all["log2T"])
print("\n=== Structure–Time Law (Unified CDCL + CoT, phi -> log2T) ===")
print(f"log2T ≈ {a_all:.3f} * phi + {b_all:.3f}")
print(f"R² = {r2_all:.3f}")

plt.figure()
for eng, sub in df_all.groupby("engine"):
    plt.scatter(sub["phi"], sub["log2T"], label=eng, s=20)
phi_grid = np.linspace(df_all["phi"].min(), df_all["phi"].max(), 200)
plt.plot(phi_grid, a_all * phi_grid + b_all)
plt.xlabel("phi (Kolmogorov order parameter)")
plt.ylabel("log2 T (runtime)")
plt.title("Scheme B: Unified Structure–Time Law (CDCL + CoT)")
plt.legend()
plt.show()


# %% [markdown]
# ## 4. 方案 C：Landau 自由能视角
#
# 基于经验分布 p(φ | α)，构造
# \[
#   F(\phi; \alpha) = -\log p(\phi | \alpha) + \text{const}.
# \]
# 这在 Landau 理论中扮演“有效自由能”的角色。
#
# 实现步骤：
# - 对每个 α，把 φ 样本投到一个统一的 φ 网格上，做核平滑或直方图估计。
# - 取 log 得到 F 曲线。
# - 比较不同 α 的 F(φ; α)，观察是否出现“单井/双井”等结构变化。


# %%
def estimate_free_energy(df, alpha_values, phi_grid=None, n_bins=20, eps=1e-6):
    """
    估计每个 alpha 条件下的 F(phi; alpha) ≈ -log p(phi | alpha)
    简单用直方图 + 对数。
    """
    if phi_grid is None:
        # 全局 φ 范围
        phi_min = df["phi"].min()
        phi_max = df["phi"].max()
        phi_grid = np.linspace(phi_min, phi_max, n_bins)
    else:
        phi_grid = np.asarray(phi_grid)

    fe_dict = {}
    for alpha in alpha_values:
        sub = df[df["alpha"] == alpha]
        if len(sub) < 2:
            continue
        phis = sub["phi"].values
        # 使用直方图估计密度
        hist, bin_edges = np.histogram(phis, bins=n_bins, density=True)
        # 计算 bin centers
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # 为了画图，我们把 hist 插值到 phi_grid 上（nearest）
        density = np.interp(phi_grid, centers, hist, left=0.0, right=0.0)
        # 避免 log(0)
        density = np.maximum(density, eps)
        F = -np.log(density)
        # 只在每个 alpha 内部 up to additive const，把最小值平移到 0
        F = F - F.min()
        fe_dict[alpha] = (phi_grid, F)
    return fe_dict


# %%
alpha_values = sorted(df_cdcl["alpha"].unique())
fe_dict = estimate_free_energy(df_cdcl, alpha_values, n_bins=30)

plt.figure()
for alpha in alpha_values:
    if alpha not in fe_dict:
        continue
    phi_grid, F = fe_dict[alpha]
    plt.plot(phi_grid, F, label=f"alpha={alpha}")
plt.xlabel("phi (Kolmogorov order parameter)")
plt.ylabel("F(phi; alpha)  (up to const)")
plt.title("Scheme C: Landau-like Free Energy of phi for CDCL")
plt.legend()
plt.show()


