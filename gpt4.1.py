# ======================================================================
#   PLSA–LLM / CoT Structure–Time Law DEMO (V3)
#   Single-Cell Colab Version – just run this cell.
# ======================================================================

import time, json, gzip, base64, math, numpy as np
import matplotlib.pyplot as plt
from getpass import getpass
from openai import OpenAI

print("=== PLSA – LLM / CoT Structure–Time Law Demo (V3) ===")

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

# -----------------------------
# 2. Compression helper
# -----------------------------
def clen(obj):
    """gzip compress length of JSON"""
    s = json.dumps(obj, ensure_ascii=False).encode("utf8")
    return len(gzip.compress(s))

# -----------------------------
# 3. Task set
# -----------------------------
TASKS = [
    "Given 5,7,12, does 5+7=12? Explain.",
    "Is graph [(1,2),(2,3),(3,1)] bipartite? Why?",
    "Solve SAT: (A or B) and (not A or C).",
    "Handshakes among 17 people?",
    "Solve 2x+5=19 step by step.",
    "How many shortest 3×3 grid paths exist?",
    "Does sequence 2,4,8,16 satisfy a linear recurrence?",
    "Is (p->q and q->r) equivalent to p->r?",
    "Derivative of x^3 - 5x?",
    "Why is √2 irrational?",
    "Give a short proof primes are infinite.",
    "If A→B and B false, what about A?",
]

# Add more slight-hard tasks
EXTRA = [
    "Is 3-SAT NP-complete? Explain briefly.",
    "Give BFS completeness argument.",
    "Counterexample to XOR distributivity.",
    "Does every tree have a bipartition? Prove.",
    "How many permutations of 7 items?",
    "Determinant of [[1,2],[3,4]]?",
    "Does every finite DAG have a topological ordering?",
    "Define what a group is in 3 lines.",
]

TASKS = TASKS + EXTRA   # total ~20 tasks

# -----------------------------
# 4. LLM call wrapper
# -----------------------------
def ask_llm(prompt, temperature):
    if not ONLINE:
        # offline synthetic trace length
        fake = "Reasoning steps: " + "x " * int(30 + temperature*50)
        return fake, len(fake)
    start = time.time()
    r = client.chat.completions.create(
        model="gpt-4.1", temperature=temperature,
        messages=[{"role":"user","content":prompt}],
        max_tokens=2000
    )
    dt = time.time() - start
    txt = r.choices[0].message.content
    return txt, dt

# -----------------------------
# 5. RUN EXPERIMENTS
# -----------------------------
records = []

TEMPS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]

print(f"Total tasks: {len(TASKS)}")
print("Temperatures:", TEMPS)

for t in TEMPS:
    print(f"\n=== Temperature {t} ===")
    for i, task in enumerate(TASKS, 1):
        print(f"[T={t}] running task {i}/{len(TASKS)} ...", end="")
        trace, dt = ask_llm(task, temperature=t)

        Kp = clen({"problem":task})
        Ks = clen({"problem":task, "trace":trace})
        lam = (Ks-Kp)/Kp
        phi = math.log(1+lam)

        records.append({
            "task":task,
            "temp":t,
            "trace":trace,
            "time":dt,
            "lambda":lam,
            "phi":phi,
        })
        print(" done.")

# -----------------------------
# 6. FIT STRUCTURE–TIME LAW
# -----------------------------
phis = np.array([r["phi"] for r in records])
Ts   = np.array([r["time"] for r in records])
ys   = np.log2(Ts + 1e-9)
A = np.vstack([phis, np.ones_like(phis)]).T
alpha, gamma = np.linalg.lstsq(A, ys, rcond=None)[0]
pred = alpha*phis + gamma
R2 = 1 - np.sum((ys-pred)**2)/np.sum((ys-ys.mean())**2)

print("\n=== Structure–Time Law (GPT-4.1) ===")
print(f"log2 T ≈ {alpha:.4f}·φ + {gamma:.4f}")
print(f"R² = {R2:.4f}")

# -----------------------------
# 7. PHASE PLOT
# -----------------------------
plt.figure(figsize=(6,5))
plt.scatter(phis, Ts, c=[r["temp"] for r in records], cmap="viridis", s=40)
plt.colorbar(label="temperature")
plt.xlabel("phi")
plt.ylabel("runtime T (s)")
plt.title("Structure–Time Phase Diagram (GPT-4.1)")
plt.grid(True)
plt.show()

# -----------------------------
# 8. PRINT RAW LOGS
# -----------------------------
print("\n=== RAW RECORDS ===")
for r in records:
    print("--------------------------------------")
    print(f"T={r['temp']}  φ={r['phi']:.4f}  λ={r['lambda']:.4f}  time={r['time']:.4f}")
    print("Task:", r["task"])
    print("--------------------------------------")
