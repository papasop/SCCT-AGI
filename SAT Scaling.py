# ============================================================
# SAT Scaling + SCCT (fixed λ_K) + LLM CoT (small n, throttled)
# ============================================================

!pip install python-sat openai tqdm scikit-learn -q

import time
import gzip
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from pysat.formula import CNF
from pysat.solvers import Glucose3
from tqdm.auto import tqdm

from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------- API Key ----------
API_KEY = ""  # TODO: 换成你的 sk-xxx

if not API_KEY or API_KEY == "YOUR_KEY_HERE":
    raise ValueError("请在代码中把 API_KEY 替换成你的 OpenAI 密钥（sk-xxx）")

client = OpenAI(api_key=API_KEY)
GPT_MODEL = "gpt-4.1-mini"

# ---------- 实验参数 ----------
N_LIST = [10, 14, 20, 50, 100, 200]
ALPHA_LIST = [3.5, 4.0, 4.26, 4.5]
NUM_INSTANCES_PER_CONFIG = 5

SMALL_N_THRESHOLD = 20         # n <= 20 才可能 CoT
ENABLE_COT = True              # 一键总开关
MAX_COT_CALLS = 6              # 整个实验里最多做 6 次 CoT 调用
cot_calls_done = 0             # 全局计数器

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------- Compression & λ_K helpers ----------
def compress_bytes(data: bytes) -> int:
    return len(gzip.compress(data))

def json_compress_len(obj: Any) -> int:
    s = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return compress_bytes(s)

def time_it(func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)

# λ_K = (K_solution - K_problem) / K_problem
def compute_lambda_K(K_problem: int, K_solution: int) -> float:
    if K_problem == 0:
        return float("nan")
    return (K_solution - K_problem) / K_problem

def safe_log2(x: float) -> float:
    if x <= 0:
        return float("nan")
    return math.log2(x)

# ---------- Random 3-SAT ----------
def generate_random_3sat(n_vars: int, alpha: float) -> CNF:
    m = int(alpha * n_vars)
    cnf = CNF()
    for _ in range(m):
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, n_vars)
            sign = random.choice([True, False])
            lit = var if sign else -var
            clause.add(lit)
        cnf.append(list(clause))
    return cnf

def cnf_to_dict(cnf: CNF, n_vars: int, alpha: float) -> Dict[str, Any]:
    return {
        "n_vars": n_vars,
        "alpha": alpha,
        "clauses": cnf.clauses,
    }

# ---------- Baseline Glucose3 ----------
@dataclass
class BaselineTrace:
    sat: bool
    model: List[int]
    num_decisions: int
    num_conflicts: int
    time_sec: float

def solve_with_glucose(cnf: CNF) -> BaselineTrace:
    solver = Glucose3(bootstrap_with=cnf.clauses)
    t0 = time.perf_counter()
    sat = solver.solve()
    t1 = time.perf_counter()
    model = solver.get_model() if sat else []
    stats = solver.accum_stats()
    num_decisions = stats.get("decisions", -1)
    num_conflicts = stats.get("conflicts", -1)
    solver.delete()
    return BaselineTrace(
        sat=sat,
        model=model,
        num_decisions=num_decisions,
        num_conflicts=num_conflicts,
        time_sec=(t1 - t0),
    )

# ---------- CNF -> text & LLM CoT ----------
def cnf_to_text(cnf_obj: Dict[str, Any]) -> str:
    lines = []
    n_vars = cnf_obj["n_vars"]
    alpha = cnf_obj["alpha"]
    lines.append(f"This is a 3-SAT formula with {n_vars} boolean variables.")
    lines.append(f"Clause/variable ratio α = {alpha:.2f}.")
    lines.append("Each clause is a disjunction (OR) of 3 literals.")
    lines.append(
        "Variables are x1..x{n}, literals are xi or ¬xi "
        "(represented as positive or negative integers)."
    )
    lines.append("Here are the clauses (each line is one clause, integers are literals):")
    for clause in cnf_obj["clauses"]:
        lines.append("  " + " ".join(str(lit) for lit in clause))
    return "\n".join(lines)

def call_gpt_cot_on_sat(cnf_obj: Dict[str, Any]) -> Dict[str, Any]:
    problem_text = cnf_to_text(cnf_obj)

    system_msg = (
        "You are a SAT reasoner. Solve small 3-SAT formulas exactly.\n"
        "Use only the minimal necessary step-by-step reasoning."
    )
    user_msg = (
        "Solve the following 3-SAT formula.\n"
        "1) Think step-by-step but keep the explanation concise (<150 tokens).\n"
        "2) At the end, clearly state SAT with an assignment, or UNSAT.\n\n"
        + problem_text
    )

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,  # 避免疯狂长 CoT
    )
    cot_text = resp.choices[0].message.content
    return {
        "cot_text": cot_text,
        "assignment": None,
    }

# ---------- Single-instance experiment ----------
@dataclass
class InstanceResult:
    n_vars: int
    alpha: float
    sat: bool
    # baseline
    T_base_sec: float
    decisions_base: int
    conflicts_base: int
    K_problem: int
    K_solution_base: int
    lambda_base: float
    log2T_base: float
    # CoT
    used_cot: bool
    T_cot_sec: float
    K_solution_cot: int
    lambda_cot: float
    log2T_cot: float

def run_single_instance(n_vars: int, alpha: float) -> InstanceResult:
    global cot_calls_done

    cnf = generate_random_3sat(n_vars, alpha)
    cnf_obj = cnf_to_dict(cnf, n_vars, alpha)

    K_problem = json_compress_len(cnf_obj)

    baseline_trace = solve_with_glucose(cnf)
    T_base = baseline_trace.time_sec

    solution_obj_base = {
        "sat": baseline_trace.sat,
        "model": baseline_trace.model,
        "num_decisions": baseline_trace.num_decisions,
        "num_conflicts": baseline_trace.num_conflicts,
    }
    K_solution_base = json_compress_len({
        "problem": cnf_obj,
        "solution_trace": solution_obj_base,
    })

    lambda_base = compute_lambda_K(K_problem, K_solution_base)
    log2T_base = safe_log2(T_base)

    used_cot = False
    T_cot_sec = float("nan")
    K_solution_cot = 0
    lambda_cot = float("nan")
    log2T_cot = float("nan")

    # 只在小 n 且没有超过 MAX_COT_CALLS 时启用 CoT
    if (
        ENABLE_COT
        and n_vars <= SMALL_N_THRESHOLD
        and cot_calls_done < MAX_COT_CALLS
    ):
        used_cot = True
        cot_calls_done += 1
        print(f"[CoT] n={n_vars}, alpha={alpha}, call #{cot_calls_done}")
        def run_cot():
            return call_gpt_cot_on_sat(cnf_obj)
        cot_result, T_cot = time_it(run_cot)
        T_cot_sec = T_cot

        solution_obj_cot = {
            "cot_text": cot_result["cot_text"],
            "assignment": cot_result["assignment"],
        }
        K_solution_cot = json_compress_len({
            "problem": cnf_obj,
            "cot_solution": solution_obj_cot,
        })
        lambda_cot = compute_lambda_K(K_problem, K_solution_cot)
        log2T_cot = safe_log2(T_cot_sec)

    return InstanceResult(
        n_vars=n_vars,
        alpha=alpha,
        sat=baseline_trace.sat,
        T_base_sec=T_base,
        decisions_base=baseline_trace.num_decisions,
        conflicts_base=baseline_trace.num_conflicts,
        K_problem=K_problem,
        K_solution_base=K_solution_base,
        lambda_base=lambda_base,
        log2T_base=log2T_base,
        used_cot=used_cot,
        T_cot_sec=T_cot_sec,
        K_solution_cot=K_solution_cot,
        lambda_cot=lambda_cot,
        log2T_cot=log2T_cot,
    )

# ---------- Main scaling loop ----------
all_results: List[InstanceResult] = []

for n in N_LIST:
    for alpha in ALPHA_LIST:
        print(f"=== n = {n}, alpha = {alpha} ===")
        for _ in tqdm(range(NUM_INSTANCES_PER_CONFIG)):
            res = run_single_instance(n, alpha)
            all_results.append(res)

results_serializable = [asdict(r) for r in all_results]

with open("sat_scct_results_fixed_lambdaK_throttled.json", "w") as f:
    json.dump(results_serializable, f, indent=2)

print("Saved results to sat_scct_results_fixed_lambdaK_throttled.json")
print("Total instances:", len(all_results))
print("Total CoT calls:", cot_calls_done)

# ---------- Regression: log2T vs λ_K ----------
df = pd.DataFrame(results_serializable)
mask = np.isfinite(df["log2T_base"]) & np.isfinite(df["lambda_base"])
df_base = df[mask].copy()

if len(df_base) == 0:
    print("No valid baseline data yet.")
else:
    X = df_base[["lambda_base", "n_vars"]].copy()
    X["log2n"] = np.log2(X["n_vars"])
    y = df_base["log2T_base"]

    reg = LinearRegression().fit(X[["lambda_base", "log2n"]], y)
    print("\nBaseline regression with fixed λ_K (throttled CoT):")
    print("log2T ≈ a * λ_K + b * log2(n) + c")
    print("a =", reg.coef_[0])
    print("b =", reg.coef_[1])
    print("c =", reg.intercept_)
    print("R^2 =", reg.score(X[["lambda_base", "log2n"]], y))

    print("\nGroup by alpha (mean λ_K, mean log2T):")
    grouped = df_base.groupby("alpha")[["lambda_base", "log2T_base"]].mean()
    display(grouped)

df_cot = df_base[df_base["used_cot"] == True]
if len(df_cot) > 0:
    print("\nCoT-enabled instances (n <= SMALL_N_THRESHOLD, limited):")
    display(df_cot[["n_vars", "alpha", "lambda_base", "lambda_cot",
                    "log2T_base", "log2T_cot"]].head(20))
else:
    print("\nNo CoT-enabled instances（可以调大 MAX_COT_CALLS 或扩大小 n 区间）.")
