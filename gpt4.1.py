# ======================================================================
#   PLSA–LLM / CoT Structure–Time Law DEMO (V5)
#   Repeated Runs + Averages + φ_c + Temperature Analysis
#   Single-Cell Colab Version – just run this cell.
# ======================================================================

import time, json, gzip, math, numpy as np
import matplotlib.pyplot as plt
from getpass import getpass
from openai import OpenAI
import collections

print("=== PLSA – LLM / CoT Structure–Time Law Demo (V5: repeats + temp) ===")

# -----------------------------
# 1. ASK USER FOR GPT-4.1 KEY
# -----------------------------
key = getpass("Enter GPT-4.1 API KEY (blank = OFFLINE DEMO): ").strip()

ONLINE = (len(key) > 0)
client = None
if ONLINE:
    client = OpenAI(api_key=key)
    print("→ ONLINE mode: GPT-4.1 CoT reasoning enabled.")
else:
    print("→ OFFLINE mode: synthetic brain enabled (no LLM calls).")
    print("   (Using synthetic trace & time; good for形状测试，不是实测 GPT-4.1。)")

# -----------------------------
# 2. Compression helper
# -----------------------------
def clen(obj):
    """gzip compress length of JSON"""
    s = json.dumps(obj, ensure_ascii=False).encode("utf8")
    return len(gzip.compress(s))

# -----------------------------
# 3. Parametric L-task family
# -----------------------------
def make_chain_task(L, seed_base=1234):
    """
    构造一个长度为 L 的“思维链任务”：
    Start from x, 每一步做 +2 / *2 / -1，并要求 step-by-step 推理。
    返回 (task_prompt, true_val)。
    """
    import random
    random.seed(seed_base + L)
    x = random.randint(1, 9)

    ops = []
    descs = []
    for i in range(L):
        op = random.choice(["+2", "*2", "-1"])
        ops.append(op)
        if op == "+2":
            descs.append(f"Step {i+1}: add 2.")
        elif op == "*2":
            descs.append(f"Step {i+1}: multiply by 2.")
        else:
            descs.append(f"Step {i+1}: subtract 1.")

    # 真值（可选检查用）
    val = x
    for op in ops:
        if op == "+2":
            val += 2
        elif op == "*2":
            val *= 2
        else:
            val -= 1

    description = "\n".join(descs)
    task = (
        f"Start from the number {x}.\n"
        f"{description}\n"
        f"After performing all {L} steps in order, "
        f"what is the final result? Explain your reasoning step by step."
    )
    return task, val

# -----------------------------
# 4. LLM call wrapper
# -----------------------------
def ask_llm(prompt, temperature):
    if not ONLINE:
        # OFFLINE: 伪造 trace & 时间，确保结构形状合理
        L_guess = prompt.count("Step ")
        base_len = 20 + 5 * L_guess
        temp_boost = int(10 * temperature)
        trace_len = base_len + temp_boost
        fake_trace = "Reasoning steps:\n" + "x " * trace_len

        import random
        random.seed(L_guess * 1000 + int(temperature * 100))
        noise = 1.0 + 0.3 * (random.random() - 0.5)
        T_fake = 0.3 * (2.0 ** (0.12 * L_guess)) * abs(noise)
        return fake_trace, T_fake

    start = time.time()
    r = client.chat.completions.create(
        model="gpt-4.1",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    dt = time.time() - start
    txt = r.choices[0].message.content
    return txt, dt

# -----------------------------
# 5. RUN EXPERIMENTS (repeats)
# -----------------------------
records = []

TEMPS     = [0.1, 0.3, 0.5, 0.7, 1.0]
L_VALUES  = [1, 2, 3, 4, 5, 7, 10, 13, 16]
N_REPEATS = 5   # 每个 (L,temp) 重复次数，可改为 3/10 等

print(f"\nControl parameter L values: {L_VALUES}")
print(f"Temperatures: {TEMPS}")
print(f"Repeats per (L,temp): {N_REPEATS}")

for t in TEMPS:
    print(f"\n=== Temperature {t} ===")
    for L in L_VALUES:
        for rep in range(N_REPEATS):
            task, true_val = make_chain_task(L)
            print(f"[T={t}] L={L}, rep={rep+1}/{N_REPEATS} ...", end="")
            trace, dt = ask_llm(task, temperature=t)

            # 结构压缩指标
            Kp = clen({"L": L, "problem": task})
            Ks = clen({"L": L, "problem": task, "trace": trace})
            lam = (Ks - Kp) / Kp
            phi = math.log(1.0 + lam)

            records.append({
                "L": L,
                "temp": t,
                "rep": rep,
                "task": task,
                "trace": trace,
                "time": float(dt),
                "lambda": float(lam),
                "phi": float(phi),
                "true_val": int(true_val),
            })
            print(" done.")

print("\n=== Finished all runs ===")

# -----------------------------
# 6. Aggregate stats by (L,temp)
# -----------------------------
stats = {}   # (L,temp) -> dict of means/stds
for L in L_VALUES:
    for t in TEMPS:
        subset = [r for r in records if r["L"] == L and r["temp"] == t]
        phis = np.array([x["phi"] for x in subset], dtype=float)
        Ts   = np.array([x["time"] for x in subset], dtype=float)
        stats[(L, t)] = {
            "mean_phi": phis.mean(),
            "std_phi":  float(phis.std(ddof=1)) if len(phis) > 1 else 0.0,
            "mean_T":   Ts.mean(),
            "std_T":    float(Ts.std(ddof=1)) if len(Ts) > 1 else 0.0,
            "n": len(subset),
        }

# 用均值样本做全局拟合
phis_mean = np.array([v["mean_phi"] for v in stats.values()], dtype=float)
Ts_mean   = np.array([v["mean_T"] for v in stats.values()], dtype=float)

ys = np.log2(Ts_mean + 1e-9)
A = np.vstack([phis_mean, np.ones_like(phis_mean)]).T
alpha, gamma = np.linalg.lstsq(A, ys, rcond=None)[0]
pred = alpha * phis_mean + gamma
R2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)

print("\n=== Global Structure–Time Law (mean over repeats) ===")
print(f"log2 T ≈ {alpha:.4f}·φ + {gamma:.4f}")
print(f"R² = {R2:.4f}")

# -----------------------------
# 6b. Piecewise φ_c estimation (on mean data)
# -----------------------------
def estimate_phi_c_piecewise(phis, Ts, min_points=4):
    """
    在 φ 轴上做两段线性拟合:
        左: log2 T = a1 φ + b1
        右: log2 T = a2 φ + b2
    用均值数据减少噪声。
    """
    N = len(phis)
    if N < 2 * min_points:
        return None

    order = np.argsort(phis)
    phi_sorted = phis[order]
    T_sorted   = Ts[order]
    y_sorted   = np.log2(T_sorted + 1e-9)

    best_RSS = None
    best_k   = None
    best_params = None

    for k in range(min_points - 1, N - min_points):
        phi_L = phi_sorted[:k+1]
        y_L   = y_sorted[:k+1]
        A_L = np.vstack([phi_L, np.ones_like(phi_L)]).T
        a1, b1 = np.linalg.lstsq(A_L, y_L, rcond=None)[0]
        pred_L = a1 * phi_L + b1
        RSS_L = np.sum((y_L - pred_L) ** 2)

        phi_R = phi_sorted[k+1:]
        y_R   = y_sorted[k+1:]
        A_R = np.vstack([phi_R, np.ones_like(phi_R)]).T
        a2, b2 = np.linalg.lstsq(A_R, y_R, rcond=None)[0]
        pred_R = a2 * phi_R + b2
        RSS_R = np.sum((y_R - pred_R) ** 2)

        RSS_tot = RSS_L + RSS_R
        if (best_RSS is None) or (RSS_tot < best_RSS):
            best_RSS = RSS_tot
            best_k = k
            best_params = (a1, b1, a2, b2)

    if best_k is None:
        return None

    phi_c = 0.5 * (phi_sorted[best_k] + phi_sorted[best_k + 1])
    a1, b1, a2, b2 = best_params

    return {
        "phi_c": float(phi_c),
        "k_index": int(best_k),
        "params_left":  (float(a1), float(b1)),
        "params_right": (float(a2), float(b2)),
        "best_RSS": float(best_RSS),
        "phi_sorted": phi_sorted,
        "y_sorted": y_sorted,
    }

pw = estimate_phi_c_piecewise(phis_mean, Ts_mean, min_points=4)

if pw is not None:
    print("\n=== Piecewise φ_c Estimation (on mean samples) ===")
    print(f"Estimated φ_c ≈ {pw['phi_c']:.4f}")
    (aL, bL) = pw["params_left"]
    (aR, bR) = pw["params_right"]
    print(f"Left segment : log2 T ≈ {aL:.4f}·φ + {bL:.4f}")
    print(f"Right segment: log2 T ≈ {aR:.4f}·φ + {bR:.4f}")
    if aR != 0:
        print(f"Slope ratio (right/left) ≈ {aR/aL:.2f}")
else:
    print("\n[WARN] Not enough points for piecewise φ_c estimation.")

# -----------------------------
# 7. PLOTS: φ–T phase diagram（均值）
# -----------------------------
plt.figure(figsize=(6,5))
# 按 L 上色
L_list_for_color = []
for (L, t) in stats.keys():
    L_list_for_color.append(L)
L_list_for_color = np.array(L_list_for_color)

scatter = plt.scatter(
    phis_mean, Ts_mean,
    c=L_list_for_color,
    cmap="viridis",
    s=60,
    edgecolors="k",
    alpha=0.8
)
cbar = plt.colorbar(scatter)
cbar.set_label("chain length L")

plt.xlabel("mean φ (log(1+λ_K))")
plt.ylabel("mean runtime T (s)")
plt.title("Structure–Time Phase Diagram (mean over repeats)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)

if pw is not None:
    phi_c = pw["phi_c"]
    plt.axvline(phi_c, color="red", linestyle="--", label=f"φ_c ≈ {phi_c:.3f}")
    plt.legend()

plt.show()

# -----------------------------
# 8. L 维度：<T>(L), <φ>(L)
# -----------------------------
by_L = collections.defaultdict(list)
for (L, t), v in stats.items():
    by_L[L].append(v)

L_list   = sorted(by_L.keys())
mean_phi_L = []
mean_T_L   = []

for L in L_list:
    phi_vals = [x["mean_phi"] for x in by_L[L]]
    T_vals   = [x["mean_T"] for x in by_L[L]]
    mean_phi_L.append(np.mean(phi_vals))
    mean_T_L.append(np.mean(T_vals))

plt.figure(figsize=(6,4))
plt.errorbar(L_list, mean_T_L, fmt="-o")
plt.xlabel("chain length L")
plt.ylabel("mean runtime T (s)")
plt.title("Mean Runtime vs Chain Length L (averaged over temps)")
plt.yscale("log")
plt.grid(True, ls="--", alpha=0.4)
plt.show()

plt.figure(figsize=(6,4))
plt.errorbar(L_list, mean_phi_L, fmt="-o")
plt.xlabel("chain length L")
plt.ylabel("mean φ")
plt.title("Mean φ vs Chain Length L (averaged over temps)")
plt.grid(True, ls="--", alpha=0.4)
plt.show()

# -----------------------------
# 9. Temperature dimension: <T>(temp), <φ>(temp) for each L
# -----------------------------
# 9a. 画 T(temp) 曲线（每条线一个 L）
plt.figure(figsize=(7,5))
for L in L_VALUES:
    mean_Ts_temp = []
    for t in TEMPS:
        v = stats[(L, t)]
        mean_Ts_temp.append(v["mean_T"])
    plt.plot(TEMPS, mean_Ts_temp, "-o", label=f"L={L}")

plt.xlabel("temperature")
plt.ylabel("mean runtime T (s)")
plt.title("Mean T vs temperature for each L")
plt.yscale("log")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.show()

# 9b. 画 φ(temp) 曲线（每条线一个 L）
plt.figure(figsize=(7,5))
for L in L_VALUES:
    mean_phis_temp = []
    for t in TEMPS:
        v = stats[(L, t)]
        mean_phis_temp.append(v["mean_phi"])
    plt.plot(TEMPS, mean_phis_temp, "-o", label=f"L={L}")

plt.xlabel("temperature")
plt.ylabel("mean φ")
plt.title("Mean φ vs temperature for each L")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.show()

# -----------------------------
# 10. 固定某个 L，看 φ–T 随 temperature 的平移/扩散
# -----------------------------
L_focus = max(L_VALUES)  # 你可以改成 7 / 10 等
phis_focus = []
Ts_focus   = []
temps_focus = []

for t in TEMPS:
    v = stats[(L_focus, t)]
    phis_focus.append(v["mean_phi"])
    Ts_focus.append(v["mean_T"])
    temps_focus.append(t)

plt.figure(figsize=(6,5))
sc = plt.scatter(phis_focus, Ts_focus, c=temps_focus, cmap="plasma", s=80, edgecolors="k")
cbar = plt.colorbar(sc)
cbar.set_label("temperature")

plt.xlabel(f"mean φ (L={L_focus})")
plt.ylabel(f"mean T (s) (L={L_focus})")
plt.title(f"φ–T vs temperature (L={L_focus})")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()

# -----------------------------
# 11. 简要打印部分均值数据
# -----------------------------
print("\n=== Mean stats by (L,temp) ===")
for L in L_VALUES:
    for t in TEMPS:
        v = stats[(L, t)]
        print(f"L={L:2d}, temp={t:.1f} -> "
              f"<φ>={v['mean_phi']:.4f} (σ={v['std_phi']:.4f}), "
              f"<T>={v['mean_T']:.4f}s (σ={v['std_T']:.4f}), n={v['n']}")
