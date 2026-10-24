# -*- coding: utf-8 -*-
"""Hình 2: accuracy on the leak-free test split vs single-thread CPU latency (log x), from benchmark_classifiers.py
output. Emphasis form: EfficientNet-B0 in the accent blue as one point with min-max bars over its runs (accuracy
over the seeds; CPU time over every EfficientNet-B0 checkpoint, which shares one architecture), other models in gray
with direct labels.

    python scripts/plot_classifier_benchmark.py OUT.png benchmark.json [more.json ...]
"""
import json
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ACCENT, MUTED, INK, INK2 = "#2a78d6", "#898781", "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
OTHERS = {"MobileNetV3-Small": (7, -3), "ResNet-18": (7, 6), "ResNet-50": (7, 2),
          "ConvNeXt-Tiny": (7, -9), "VGG19-BN": (-30, 9)}
DEPLOYED = "MobileNetV3-Large"   # live classifier (MOBILENET_CHECKPOINT in src/childsafe/config.py)

models = {m["name"]: m for f in sys.argv[2:] for m in json.load(open(f, encoding="utf-8"))["models"]}
seeds = [m for n, m in models.items() if n.startswith("EfficientNet-B0") and "bền" not in n]
same_arch = [m for n, m in models.items() if n.startswith("EfficientNet-B0")]

plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

for name, off in OTHERS.items():
    m = models[name]
    x, y = m["latency_cpu_1thread_ms"], 100 * m["accuracy"]
    ax.scatter([x], [y], s=48, color=MUTED, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.annotate(name, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK2)

lat = [m["latency_cpu_1thread_ms"] for m in same_arch]
acc = [100 * m["accuracy"] for m in seeds]
x, y = statistics.median(lat), statistics.mean(acc)
ax.errorbar([x], [y], xerr=[[x - min(lat)], [max(lat) - x]], yerr=[[y - min(acc)], [max(acc) - y]],
            fmt="o", color=MUTED, ms=7, mec=SURFACE, mew=1.5, elinewidth=1.2, capsize=3, zorder=3)
ax.annotate(f"EfficientNet-B0 ({len(seeds)} seed)", (x, y), xytext=(-104, -16),
            textcoords="offset points", fontsize=8.5, color=INK2)
m = models[DEPLOYED]
xd, yd = m["latency_cpu_1thread_ms"], 100 * m["accuracy"]
ax.scatter([xd], [yd], s=70, color=ACCENT, zorder=4, edgecolors=SURFACE, linewidths=1.5)
ax.annotate(f"{DEPLOYED} (đang dùng)", (xd, yd), xytext=(8, 2), textcoords="offset points", fontsize=8.5,
            color=INK, weight="bold")
ax.annotate(f"recall lớp nguy hiểm {100 * m['unsafe_recall']:.1f}%".replace(".", ","), (xd, yd), xytext=(8, -9),
            textcoords="offset points", fontsize=8, color=ACCENT)

ax.set_xscale("log")
ax.set_xlabel("Thời gian phân loại một ảnh trên CPU, 1 luồng (ms, thang log)", fontsize=9, color=INK2)
ax.set_ylabel("Độ chính xác trên tập kiểm thử (%)", fontsize=9, color=INK2)
ax.grid(True, which="major", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=INK2, labelsize=8.5)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(8, 1500)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor=SURFACE)
print("saved", sys.argv[1])
