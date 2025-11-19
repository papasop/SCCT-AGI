# ===============================================
# Structural Action vs Complexity: P (MST) vs NP (TSP)
# 版本 A：修复版（state 内含 problem，λ_K > 0）
# ===============================================

import time
import gzip
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from scipy import stats

# ---------- RNG ----------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- Compression helpers ----------
def compress_bytes(data: bytes) -> int:
    return len(gzip.compress(data))

def json_compress_len(obj: Any) -> int:
    s = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return compress_bytes(s)

def safe_log2(x: float) -> float:
    if x <= 0:
        return float("nan")
    return math.log2(x)

# ---------- Problem generator ----------
def generate_complete_graph(n: int, w_min: int = 1, w_max: int = 10) -> List[List[int]]:
    """Generate a symmetric complete graph with integer weights."""
    w = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            wij = random.randint(w_min, w_max)
            w[i][j] = wij
            w[j][i] = wij
    return w

# ---------- Structural action machinery ----------
def compute_lambda_step(K_problem: int, problem_obj: Any, solver_state: Any) -> float:
    """
    λ_K(t) = max( (K_state - K_problem) / K_problem, 0 )
    关键修复：K_state = C_t({problem, solver_state})
    这样才能保证 K_state >= K_problem，避免全 0。
    """
    state_obj = {
        "problem": problem_obj,     # 完整问题编码
        "solver_state": solver_state
    }
    K_state = json_compress_len(state_obj)
    if K_problem <= 0:
        return 0.0
    return max((K_state - K_problem) / K_problem, 0.0)

@dataclass
class RunResult:
    problem_type: str  # "MST" or "TSP"
    n: int
    instance_id: int

    T_sec: float
    steps: int
    action: float        # Σ λ_K(t)
    avg_lambda: float    # action / steps
    K_problem: int

# ---------- MST (P problem) with instrumented trajectory ----------
def prim_mst_structural(w: List[List[int]], instance_id: int) -> RunResult:
    """
    Prim's MST algorithm with structural trace.
    """
    n = len(w)
    problem_obj = {
        "type": "MST",
        "n": n,
        "weights": w,
    }
    K_problem = json_compress_len(problem_obj)

    in_tree = [False]*n
    in_tree[0] = True
    tree_edges: List[Tuple[int, int]] = []

    steps = 0
    total_action = 0.0

    t0 = time.perf_counter()
    # n-1 edges
    for _ in range(n-1):
        best = None
        best_w = None
        for u in range(n):
            if not in_tree[u]:
                continue
            for v in range(n):
                if in_tree[v]:
                    continue
                if best is None or w[u][v] < best_w:
                    best = (u, v)
                    best_w = w[u][v]
        u, v = best
        tree_edges.append((u, v))
        in_tree[v] = True

        # 记录一步的结构状态
        solver_state = {
            "edges": tree_edges[:],
            "last_edge": (u, v),
            "tree_size": sum(in_tree),
        }
        lam = compute_lambda_step(K_problem, problem_obj, solver_state)
        total_action += lam
        steps += 1

    t1 = time.perf_counter()
    T = t1 - t0
    avg_lambda = total_action / steps if steps > 0 else 0.0

    return RunResult(
        problem_type="MST",
        n=n,
        instance_id=instance_id,
        T_sec=T,
        steps=steps,
        action=total_action,
        avg_lambda=avg_lambda,
        K_problem=K_problem,
    )

# ---------- TSP (NP-complete) DFS + B&B with structural trace ----------
def tsp_bruteforce_structural(w: List[List[int]], instance_id: int) -> RunResult:
    """
    TSP solver by DFS + branch & bound, logging partial tours as states.
    Only safe for small n (<= 9).
    """
    n = len(w)
    problem_obj = {
        "type": "TSP",
        "n": n,
        "weights": w,
    }
    K_problem = json_compress_len(problem_obj)

    best_cost = float("inf")
    best_path = None

    steps = 0
    total_action = 0.0

    t0 = time.perf_counter()

    def dfs(path: List[int], cost_so_far: int, visited: List[bool]):
        nonlocal best_cost, best_path, steps, total_action

        # 记录当前状态（路径长度、cost、visited 模式）
        solver_state = {
            "path": path[:],
            "cost_so_far": cost_so_far,
            "path_len": len(path),
            "visited_mask": visited[:],
        }
        lam = compute_lambda_step(K_problem, problem_obj, solver_state)
        total_action += lam
        steps += 1

        # 剪枝
        if cost_so_far >= best_cost:
            return

        if len(path) == n:
            # 回到起点
            total_cost = cost_so_far + w[path[-1]][path[0]]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path[:]
            return

        last = path[-1]
        for nxt in range(n):
            if not visited[nxt]:
                visited[nxt] = True
                dfs(path + [nxt], cost_so_far + w[last][nxt], visited)
                visited[nxt] = False

    visited = [False]*n
    visited[0] = True
    dfs([0], 0, visited)

    t1 = time.perf_counter()
    T = t1 - t0
    avg_lambda = total_action / steps if steps > 0 else 0.0

    return RunResult(
        problem_type="TSP",
        n=n,
        instance_id=instance_id,
        T_sec=T,
        steps=steps,
        action=total_action,
        avg_lambda=avg_lambda,
        K_problem=K_problem,
    )

# ---------- Main experiment ----------
N_LIST_P = [10, 15, 20, 25, 30]   # MST: 可以大一点
N_LIST_NP = [6, 7, 8, 9]          # TSP: 控制规模
INSTANCES_PER_N = 5

results: List[RunResult] = []

print("=== P-side: MST (Prim) with structural action ===")
for n in tqdm(N_LIST_P):
    for inst in range(INSTANCES_PER_N):
        g = generate_complete_graph(n)
        res = prim_mst_structural(g, instance_id=inst)
        results.append(res)

print("\n=== NP-side: TSP (DFS + B&B) with structural action ===")
for n in tqdm(N_LIST_NP):
    for inst in range(INSTANCES_PER_N):
        g = generate_complete_graph(n)
        res = tsp_bruteforce_structural(g, instance_id=inst)
        results.append(res)

# ---------- DataFrame ----------
df = pd.DataFrame([asdict(r) for r in results])
df["log2T"] = df["T_sec"].apply(safe_log2)
df["log2n"] = np.log2(df["n"].astype(float))

print("\n=== Summary head ===")
display(df.head())

print("\n=== Basic group stats (by problem type) ===")
group_type = df.groupby("problem_type")[["T_sec", "steps", "action", "avg_lambda", "log2T"]].agg(["mean", "std"])
display(group_type)

print("\n=== P vs NP: Structural Action & Time (per n) ===")
display(
    df.pivot_table(
        index="n",
        columns="problem_type",
        values=["action", "T_sec", "steps", "avg_lambda"],
        aggfunc="mean"
    )
)

# ---------- Statistical comparison: MST vs TSP ----------
mst_actions = df[df["problem_type"]=="MST"]["action"].values
tsp_actions = df[df["problem_type"]=="TSP"]["action"].values

mst_log2T = df[df["problem_type"]=="MST"]["log2T"].values
tsp_log2T = df[df["problem_type"]=="TSP"]["log2T"].values

# Welch t-test for action
t_action, p_action = stats.ttest_ind(mst_actions, tsp_actions, equal_var=False)
t_time, p_time = stats.ttest_ind(mst_log2T, tsp_log2T, equal_var=False)

print("\n=== MST vs TSP: t-tests ===")
print(f"Structural action A: mean_MST={mst_actions.mean():.3e}, mean_TSP={tsp_actions.mean():.3e}")
print(f"  Welch t-test: t={t_action:.3f}, p={p_action:.3e}")
print(f"log2T: mean_MST={np.nanmean(mst_log2T):.3f}, mean_TSP={np.nanmean(tsp_log2T):.3f}")
print(f"  Welch t-test: t={t_time:.3f}, p={p_time:.3e}")

# ---------- Regression: log2T vs action + log2n ----------
mask = np.isfinite(df["log2T"]) & np.isfinite(df["action"])
df_reg = df[mask].copy()

X = df_reg[["action", "log2n"]].copy()
y = df_reg["log2T"].values

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
R2 = reg.score(X, y)

print("\n=== Global regression: log2T ~ action + log2n ===")
print("log2T ≈ a * Action + b * log2n + c")
print(f"a (Action coeff) = {reg.coef_[0]:.4f}")
print(f"b (log2n coeff)  = {reg.coef_[1]:.4f}")
print(f"c (intercept)    = {reg.intercept_:.4f}")
print(f"R^2              = {R2:.4f}")

# 也分别看 P / NP 内部回归
for ptype in ["MST", "TSP"]:
    df_sub = df_reg[df_reg["problem_type"]==ptype]
    if len(df_sub) < 3:
        continue
    X_sub = df_sub[["action", "log2n"]].copy()
    y_sub = df_sub["log2T"].values
    reg_sub = LinearRegression().fit(X_sub, y_sub)
    R2_sub = reg_sub.score(X_sub, y_sub)
    print(f"\n=== Regression within {ptype} ===")
    print("log2T ≈ a * Action + b * log2n + c")
    print(f"a = {reg_sub.coef_[0]:.4f}, b = {reg_sub.coef_[1]:.4f}, c = {reg_sub.intercept_:.4f}, R^2 = {R2_sub:.4f}")

