"""
Generate publication-quality charts from the Week 3 benchmark data.
"""
import json
import glob
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# --- Load latest results ---
def latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]

bench = json.load(open(latest("benchmark_quants_*.json")))
quality = json.load(open(latest("eval_quants_*.json")))
thermal = json.load(open(latest("thermal_test_*.json")))

# ============================================
# Chart 1: Latency + memory by quantization
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

quants = [b["label"] for b in bench]
tps = [b["median_gen_tps"] for b in bench]
mem = [b["peak_mem_gb"] for b in bench]
colors = ["#0071e3", "#5ac8fa", "#1d1d1f"]  # Apple-ish palette

ax1.bar(quants, tps, color=colors)
ax1.set_ylabel("Generation throughput (tokens/sec)")
ax1.set_title("Throughput by quantization", fontweight="bold")
for i, v in enumerate(tps):
    ax1.text(i, v + 2, f"{v:.0f}", ha="center", fontweight="bold")
ax1.set_ylim(0, max(tps) * 1.15)

ax2.bar(quants, mem, color=colors)
ax2.set_ylabel("Peak GPU memory (GB)")
ax2.set_title("Memory by quantization", fontweight="bold")
for i, v in enumerate(mem):
    ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
ax2.set_ylim(0, max(mem) * 1.15)

fig.suptitle(f"Llama-3.2-1B on MLX — Apple Silicon (M-series, fanless)",
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("chart_quant_perf.png", dpi=160, bbox_inches="tight")
print("  → chart_quant_perf.png")

# ============================================
# Chart 2: Quality by quantization
# ============================================
fig, ax = plt.subplots(figsize=(7, 4.5))

labels = [q["label"] for q in quality]
honesty = [q["honesty_rate"] * 100 for q in quality]
helpfulness = [q["help_rate"] * 100 for q in quality]

x = range(len(labels))
w = 0.35
ax.bar([i - w/2 for i in x], honesty, w, label="Honesty (out-of-corpus)", color="#0071e3")
ax.bar([i + w/2 for i in x], helpfulness, w, label="Helpfulness (in-corpus)", color="#5ac8fa")

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Pass rate")
ax.set_title("Eval quality by quantization", fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 110)
ax.legend(loc="lower right")

for i, (h, hp) in enumerate(zip(honesty, helpfulness)):
    ax.text(i - w/2, h + 1, f"{h:.0f}%", ha="center", fontsize=9)
    ax.text(i + w/2, hp + 1, f"{hp:.0f}%", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig("chart_quant_quality.png", dpi=160, bbox_inches="tight")
print("  → chart_quant_quality.png")

# ============================================
# Chart 3: Thermal curve
# ============================================
# Use the same cleaning logic as before — exclude post-gap data
log = thermal["log"]
clean = []
for i, r in enumerate(log):
    if i > 0 and r["t_seconds"] - log[i-1]["t_seconds"] > 10:
        break
    clean.append(r)

times_min = [r["t_seconds"] / 60 for r in clean]
tps_series = [r["tok_per_sec"] for r in clean]

# Rolling median for a smoothed line
window = 15
rolling = []
for i in range(len(tps_series)):
    lo = max(0, i - window // 2)
    hi = min(len(tps_series), i + window // 2 + 1)
    rolling.append(statistics.median(tps_series[lo:hi]))

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(times_min, tps_series, s=8, alpha=0.25, color="#0071e3", label="Individual iterations")
ax.plot(times_min, rolling, color="#1d1d1f", linewidth=2, label=f"Rolling median (window={window})")

peak = max(tps_series)
ax.axhline(peak, color="#888", linestyle="--", linewidth=1, alpha=0.5)
ax.text(times_min[-1], peak + 1, f"peak {peak:.0f} tok/s",
        ha="right", fontsize=9, color="#555")

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.set_title("Sustained throughput over 19 minutes — thermal characterization",
             fontweight="bold")
ax.legend(loc="lower left")
ax.set_ylim(50, 130)

fig.tight_layout()
fig.savefig("chart_thermal.png", dpi=160, bbox_inches="tight")
print("  → chart_thermal.png")

plt.close("all")
print("\nDone.")