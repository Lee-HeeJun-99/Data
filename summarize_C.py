import os, json, glob
import numpy as np

dirs = sorted([d for d in glob.glob("C_seed*") if os.path.isdir(d)])
accs = []
f1_arc = []
f1_loose = []
f1_norm = []
f1_over = []

for d in dirs:
    with open(os.path.join(d, "metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    accs.append(m["test_acc"])
    f1_norm.append(m["report"]["normal"]["f1-score"])
    f1_loose.append(m["report"]["loose"]["f1-score"])
    f1_arc.append(m["report"]["arc"]["f1-score"])
    f1_over.append(m["report"]["overcurrent"]["f1-score"])

def ms(x):
    x = np.array(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.0)

print("== Final Summary (mean ± std) ==")
print(f"Test Acc: {ms(accs)[0]:.4f} ± {ms(accs)[1]:.4f}")
print(f"F1 normal: {ms(f1_norm)[0]:.4f} ± {ms(f1_norm)[1]:.4f}")
print(f"F1 loose:  {ms(f1_loose)[0]:.4f} ± {ms(f1_loose)[1]:.4f}")
print(f"F1 arc:    {ms(f1_arc)[0]:.4f} ± {ms(f1_arc)[1]:.4f}")
print(f"F1 over:   {ms(f1_over)[0]:.4f} ± {ms(f1_over)[1]:.4f}")
