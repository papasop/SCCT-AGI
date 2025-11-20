# ============================================================
#  Colab φ(x) 版：CDCL 相变 + LLM CoT + 统一 Structure–Time Law
#  你只需要做一件事：把下面这一行改成你的 OpenAI Key
# ============================================================

OPENAI_API_KEY = ""  # ← 在这里填你的 key（ 

# =========================
# 1. 安装依赖
# =========================
!pip install -q python-sat openai numpy pandas scikit-learn

# =========================
# 2. 导入库 & 基础工具
# =========================
import gzip
import json
import math
import time
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pysat.formula import CNF
from pysat.solvers import Glucose3

from openai import OpenAI

# 配置 OpenAI client
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_KEY_HERE":
    client = OpenAI(api_key=OPENAI_API_KEY)
    HAS_KEY = True
else:
    print("⚠️ 当前没有设置有效的 OPENAI_API_KEY，CoT 实验会被自动跳过。")
    client = None
    HAS_KEY = False

# =========================
# 3. 压缩 & 结构测量函数
# =========================
def gzip_len(s: str) -> int:
    """gzip 压缩长度，作为 Kolmogorov 风格复杂度代理"""
    return len(gzip.compress(s.encode("utf-8"), compresslevel=9))

def measure_phi(problem_obj, solution_obj):
    """
    对一个 (problem, solution_trace) 对，计算：
    - K_problem
    - K_solution
    - λ_K = (Ks - Kp) / Kp
    - φ(x) = log(1 + λ_K)  （新序参量）
    """
    problem_json = json.dumps(problem_obj, sort_keys=True)
    full_json    = json.dumps(
        {"problem": problem_obj, "solution_trace": solution_obj},
        sort_keys=True
    )
    Kp = gzip_len(problem_json)
    Ks = gzip_len(full_json)
    lam = (Ks - Kp) / Kp
    lam = max(lam, 0.0)  # 防守一下
    phi = math.log(1.0 + lam)
    return Kp, Ks, lam, phi

# =========================
# 4. 随机 3-SAT 生成 & CDCL 求解
# =========================
def random_3sat(n, alpha):
    """生成一个随机 3-SAT 实例（n 变量，m=alpha*n 子句）"""
    m = int(alpha * n)
    clauses = []
    for _ in range(m):
        vars_ = random.sample(range(1, n + 1), 3)
        lits = []
        for v in vars_:
            sign = random.choice([-1, 1])
            lits.append(sign * v)
        clauses.append(lits)
    problem_obj = {
        "n_vars": n,
        "alpha": alpha,
        "clauses": clauses,
    }
    return problem_obj

def solve_cdcl(problem_obj):
    """用 Glucose3 求解一个 3-SAT 实例，返回 (T_sec, solution_obj)"""
    cnf = CNF()
    for cl in problem_obj["clauses"]:
        cnf.append(cl)
    solver = Glucose3(bootstrap_with=cnf.clauses)
    start = time.perf_counter()
    sat = solver.solve()
    T_sec = time.perf_counter() - start

    model = solver.get_model() if sat else []
    num_decisions = solver.accum_stats().get("decisions", 0)
    num_conflicts = solver.accum_stats().get("conflicts", 0)
    solver.delete()

    solution_obj = {
        "sat": bool(sat),
        "model": model if model is not None else [],
        "num_decisions": int(num_decisions),
        "num_conflicts": int(num_conflicts),
    }
    return T_sec, solution_obj

# =========================
# 5. LLM CoT 求解器（新 API）
# =========================
COT_MODEL = "gpt-4o-mini"  # 你可以改成别的

def sat_to_text(problem_obj):
    """把 3-SAT 实例转成英文描述，方便喂给 LLM"""
    n = problem_obj["n_vars"]
    clauses = problem_obj["clauses"]
    lines = []
    lines.append(f"We have a Boolean formula in 3-CNF with {n} variables.")
    lines.append("Each clause is a disjunction of three literals (variable or its negation).")
    lines.append("Variables are numbered 1..{}; negative numbers mean negation.".format(n))
    lines.append("Here is the list of clauses:")
    for i, cl in enumerate(clauses):
        lines.append(f"Clause {i+1}: {cl}")
    lines.append(
        "Please decide whether the formula is SAT or UNSAT. "
        "If SAT, provide ONE satisfying assignment as a list of literals, "
        "and explain your reasoning step by step."
    )
    return "\n".join(lines)

def solve_with_cot(problem_obj):
    """调用 OpenAI Chat Completions 进行 CoT 求解，返回 (T_sec, solution_obj)"""
    if not HAS_KEY:
        raise RuntimeError("No valid OPENAI_API_KEY set; cannot run CoT.")

    prompt = sat_to_text(problem_obj)
    system_msg = (
        "You are a SAT solver. You solve 3-SAT formulas given as lists of "
        "integer clauses. Think step by step in natural language and then "
        "give a final SAT/UNSAT answer."
    )

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=COT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    T_sec = time.perf_counter() - start

    cot_text = resp.choices[0].message.content

    solution_obj = {
        "cot_text": cot_text,
        "sat_guess": None,        # 可以后续做解析，这里先占位
        "assignment_guess": None, # 占位
    }
    return T_sec, solution_obj

# =========================
# 6. CDCL 相变扫描（FAST）
# =========================
print("=== Running CDCL Phase Sweep (FAST) ===")

random.seed(0)
np.random.seed(0)

CDCL_RESULTS = []

# 可调参数：越小越快
N_LIST   = [20, 50, 100]           # 变量数
ALPHAS   = [3.5, 4.0, 4.26, 4.5]   # 相变扫描 α
INST_PER = 5                       # 每个 (n, α) 的样本数

inst_id = 0
for n in N_LIST:
    for alpha in ALPHAS:
        for k in range(INST_PER):
            problem = random_3sat(n, alpha)
            T_sec, sol = solve_cdcl(problem)
            Kp, Ks, lam, phi = measure_phi(problem, sol)

            CDCL_RESULTS.append({
                "engine": "CDCL",
                "n": n,
                "alpha": alpha,
                "instance_id": inst_id,
                "T_sec": T_sec,
                "log2T": math.log2(T_sec) if T_sec > 0 else -999.0,
                "K_problem": Kp,
                "K_solution": Ks,
                "lambda_K": lam,
                "phi": phi,
            })
            inst_id += 1

cdcl_df = pd.DataFrame(CDCL_RESULTS)
print("CDCL 样本数:", len(cdcl_df))
print(cdcl_df.head())

# 按 α 分组，看 φ 与 log2T 的相变趋势
cdcl_phase_stats = cdcl_df.groupby("alpha")[["lambda_K", "phi", "log2T"]].mean()
print("\n=== CDCL Phase Transition (by alpha) ===")
print(cdcl_phase_stats)

# =========================
# 7. LLM CoT 小样本（相变 + 高冗余相）
# =========================
COT_RESULTS = []

if HAS_KEY:
    print("\n=== Running LLM CoT Phase Sweep (Small Sample) ===")
    COT_N       = 10
    COT_ALPHAS  = [3.5, 4.0, 4.26, 4.5]
    COT_PER     = 2   # 每个 α 跑 2 个实例

    cot_inst_id = 0
    for alpha in COT_ALPHAS:
        for k in range(COT_PER):
            problem = random_3sat(COT_N, alpha)
            try:
                T_sec, sol = solve_with_cot(problem)
            except Exception as e:
                print(f"CoT 调用失败 (alpha={alpha}, k={k}): {e}")
                continue
            Kp, Ks, lam, phi = measure_phi(problem, sol)

            COT_RESULTS.append({
                "engine": "CoT",
                "n": COT_N,
                "alpha": alpha,
                "instance_id": cot_inst_id,
                "T_sec": T_sec,
                "log2T": math.log2(T_sec) if T_sec > 0 else -999.0,
                "K_problem": Kp,
                "K_solution": Ks,
                "lambda_K": lam,
                "phi": phi,
            })
            cot_inst_id += 1

    cot_df = pd.DataFrame(COT_RESULTS)
    print("CoT 样本数:", len(cot_df))
    if len(cot_df) > 0:
        print(cot_df.head())
else:
    print("\n⚠️ 跳过 CoT 实验：没有有效的 OPENAI_API_KEY。")
    cot_df = pd.DataFrame(columns=cdcl_df.columns)

# =========================
# 8. 统一 Structure–Time Law:  log2T ~ φ(x)
# =========================
print("\n=== Structure–Time Law (Unified CDCL + CoT) ===")
full_df = pd.concat([cdcl_df, cot_df], ignore_index=True)

# 过滤掉极端 / 错误值
full_df = full_df[np.isfinite(full_df["log2T"])]
full_df = full_df[full_df["T_sec"] > 0]
full_df = full_df[full_df["phi"] >= 0]

X = full_df[["phi"]].values
y = full_df["log2T"].values

reg = LinearRegression()
reg.fit(X, y)
alpha_T = reg.coef_[0]
gamma_T = reg.intercept_
R2 = reg.score(X, y)

print(f"log2T ≈ {alpha_T:.3f} * phi + {gamma_T:.3f}")
print(f"R² = {R2}")

# 分引擎看看 φ 统计 & 时间尺度
engine_stats = full_df.groupby("engine")[["lambda_K", "phi", "log2T"]].agg(["mean", "std", "min", "max"])
print("\n=== Engine-wise stats (CDCL vs CoT) ===")
print(engine_stats)

print("\n== DONE ==")
