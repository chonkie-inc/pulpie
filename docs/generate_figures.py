#!/usr/bin/env python3
"""Generate all blog figures for Introducing Hummingbird."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
RED = "#DC2626"
RED_LIGHT = "#FCA5A5"
GRAY = "#6B7280"
GRAY_LIGHT = "#D1D5DB"
PURPLE = "#7C3AED"
BG = "#FAFAFA"


def fig1_quality_vs_cost():
    """Scatter: quality (ROUGE-5) vs cost per 1B pages, log x-axis.
    Simplified: one point per method (best config), leader lines for labels."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.set_facecolor("white")
    ax.set_facecolor(BG)

    # One point per method, best realistic config
    # (name, cost, rouge5, color, marker, size)
    points = [
        ("Trafilatura",          10,     0.640, GRAY,       "s", 90),
        ("Readability",          10,     0.654, GRAY,       "s", 90),
        ("magic-html",           10,     0.714, GRAY,       "s", 90),
        ("Hummingbird 210M",     6_500,  0.864, BLUE,       "o", 160),
        ("Dripper 0.6B",         77_000, 0.854, RED,        "D", 130),
    ]

    for name, cost, rouge, color, marker, size in points:
        ax.scatter(cost, rouge, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidths=1.0)

    # Labels with leader lines (annotate with arrow)
    label_cfg = {
        "Trafilatura":      (3,    0.615, "right"),
        "Readability":      (3,    0.670, "right"),
        "magic-html":       (3,    0.730, "right"),
        "Hummingbird 210M": (1_800, 0.885, "center"),
        "Dripper 0.6B":     (200_000, 0.870, "center"),
    }
    for name, cost, rouge, color, marker, size in points:
        lx, ly, ha = label_cfg[name]
        ax.annotate(
            name, (cost, rouge), xytext=(lx, ly),
            fontsize=10.5, fontweight="bold" if "Hummingbird" in name else "normal",
            color=color,
            ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=GRAY_LIGHT, lw=0.8),
        )

    # Cost labels under Hummingbird and Dripper
    ax.text(6_500, 0.845, "$6.5K", fontsize=9, color=BLUE, ha="center", style="italic")
    ax.text(77_000, 0.840, "$77K", fontsize=9, color=RED, ha="center", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1B pages (USD)", fontsize=12)
    ax.set_ylabel("ROUGE-5 F1 (WebMainBench)", fontsize=12)
    ax.set_xlim(1.5, 250_000)
    ax.set_ylim(0.60, 0.90)

    ax.set_xticks([10, 100, 1_000, 10_000, 100_000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "~$0" if x <= 10 else f"${x/1000:.0f}K" if x >= 1000 else f"${int(x)}"
    ))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GRAY, markersize=8, label="Heuristic (CPU)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Hummingbird (encoder)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=RED, markersize=8, label="Dripper (decoder)"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=10,
              frameon=True, facecolor="white", edgecolor=GRAY_LIGHT)

    ax.set_title("Quality vs Cost of Web Content Extraction", fontsize=14, fontweight="bold", pad=12)
    fig.savefig("fig1_quality_vs_cost.png")
    plt.close(fig)
    print("Saved fig1_quality_vs_cost.png")


def fig2_cost_comparison():
    """Bar chart: cost per 1B pages, four configs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor("white")
    ax.set_facecolor(BG)

    labels = ["Hummingbird\n210M on L4", "Hummingbird\n210M on A100",
              "Dripper\n0.6B on A100", "Dripper\n0.6B on L4"]
    costs = [6_500, 9_700, 77_000, 105_000]
    colors = [BLUE, BLUE_LIGHT, RED_LIGHT, RED]

    bars = ax.bar(labels, costs, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
                f"${cost:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Comparison annotation: best Hummingbird vs best Dripper (realistic configs)
    ax.annotate("",
                xy=(0, 6_500), xycoords="data",
                xytext=(2, 77_000), textcoords="data",
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2))
    ax.text(1.0, 44_000, "~12x cheaper", fontsize=12, fontweight="bold",
            color=BLUE, ha="center")

    ax.set_ylabel("Cost per 1B pages (USD)", fontsize=12)
    ax.set_ylim(0, 125_000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:.0f}K"
    ))

    ax.set_title("Cost to Clean 1 Billion Web Pages", fontsize=14, fontweight="bold", pad=12)
    fig.savefig("fig2_cost_comparison.png")
    plt.close(fig)
    print("Saved fig2_cost_comparison.png")


def fig3_throughput():
    """Grouped bar chart: throughput on L4 vs A100."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor("white")
    ax.set_facecolor(BG)

    gpus = ["NVIDIA L4\n(300 GB/s, ~120 TFLOPS)", "NVIDIA A100\n(2,039 GB/s, 312 TFLOPS)"]
    hb_speeds = [15.1, 43.0]
    dr_speeds = [0.92, 5.38]
    ratios = ["16.4x", "8.0x"]

    x = np.arange(len(gpus))
    width = 0.3

    bars1 = ax.bar(x - width/2, hb_speeds, width, color=BLUE, label="Hummingbird 210M",
                   edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, dr_speeds, width, color=RED, label="Dripper 0.6B",
                   edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars1, hb_speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val}", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    for bar, val in zip(bars2, dr_speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val}", ha="center", fontsize=11, fontweight="bold", color=RED)

    # Ratio annotations
    for i, ratio in enumerate(ratios):
        mid_x = x[i]
        mid_y = hb_speeds[i] / 2
        ax.text(mid_x + width/2 + 0.15, mid_y, ratio, fontsize=11,
                fontweight="bold", color=GRAY, va="center")

    ax.set_ylabel("Pages / second", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(gpus)
    ax.set_ylim(0, 52)
    ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
              edgecolor=GRAY_LIGHT)

    ax.set_title("Throughput: Encoder vs Decoder on Different GPUs",
                 fontsize=14, fontweight="bold", pad=12)
    fig.savefig("fig3_throughput_by_gpu.png")
    plt.close(fig)
    print("Saved fig3_throughput_by_gpu.png")


def fig4_distillation():
    """Bar chart: distillation results with reference lines."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.set_facecolor("white")
    ax.set_facecolor(BG)

    models = ["Latte Large\n(2.1B)", "Latte Base\n(610M)", "Latte Small\n(210M)"]
    scores = [0.864, 0.849, 0.864]
    colors_bars = [BLUE, BLUE_LIGHT, BLUE]

    bars = ax.bar(models, scores, color=colors_bars, width=0.5,
                  edgecolor="white", linewidth=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{score:.3f}", ha="center", fontsize=12, fontweight="bold")

    # Reference lines
    ax.axhline(y=0.865, color=PURPLE, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.35, 0.8655, "DeepSeek V3.2", fontsize=9, color=PURPLE, va="bottom")

    ax.axhline(y=0.854, color=RED, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.35, 0.8545, "Dripper 0.6B", fontsize=9, color=RED, va="bottom")

    ax.set_ylabel("ROUGE-5 F1", fontsize=12)
    ax.set_ylim(0.840, 0.875)

    ax.set_title("Distillation: Smaller Can Be Better",
                 fontsize=14, fontweight="bold", pad=12)
    fig.savefig("fig4_distillation.png")
    plt.close(fig)
    print("Saved fig4_distillation.png")


if __name__ == "__main__":
    fig1_quality_vs_cost()
    fig2_cost_comparison()
    fig3_throughput()
    fig4_distillation()
    print("All figures generated.")
