import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from cpu_pipeline import run_cpu_pipeline
from gpu_pipeline import run_gpu_pipeline

DATA_DIR  = r"c:\Users\RUTIKA\Desktop\uber\Data"
NUM_FILES = 4          # ≈ 13 million rows — plenty to saturate both pipelines


def run_benchmark():
    print("=" * 56)
    print("      UBER FARE PREDICTION BENCHMARK  –  CPU vs GPU")
    print("=" * 56)

    # ── CPU ────────────────────────────────────────────────────────────────
    print("\n[1/2]  Running CPU Baseline …\n")
    cpu = run_cpu_pipeline(DATA_DIR, num_files=NUM_FILES)

    print("\n" + "─" * 56 + "\n")

    # ── GPU ────────────────────────────────────────────────────────────────
    print("[2/2]  Running GPU (CUDA) Optimised …\n")
    gpu = run_gpu_pipeline(DATA_DIR, num_files=NUM_FILES)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("                  RESULTS SUMMARY")
    print("=" * 56)
    rows = [
        ("Stage",               "CPU (s)",               "GPU (s)",  "Speedup"),
        ("Data Loading",        cpu['load_time'],         gpu['load_time'],  None),
        ("Preprocessing",       cpu['prep_time'],         gpu['prep_time'],  None),
        ("Training (200 ep)",   cpu['train_time'],        gpu['train_time'], None),
        ("TOTAL",               cpu['total_time'],        gpu['total_time'], None),
    ]
    for row in rows:
        label, c, g = row[0], row[1], row[2]
        if isinstance(c, str):
            print(f"  {'Stage':<22}{'CPU':>10}{'GPU':>10}{'Speedup':>10}")
            print("  " + "─" * 52)
        else:
            sp = c / g if g > 0 else float('inf')
            print(f"  {label:<22}{c:>10.2f}{g:>10.2f}{sp:>9.2f}x")

    train_speedup = cpu['train_time'] / gpu['train_time']
    print(f"\n  ➤  GPU Training is {train_speedup:.1f}× faster than CPU!\n")

    # ── Plot ───────────────────────────────────────────────────────────────
    stages     = ['Data Loading', 'Preprocessing', 'Training\n(200 epochs)']
    cpu_times  = [cpu['load_time'], cpu['prep_time'], cpu['train_time']]
    gpu_times  = [gpu['load_time'], gpu['prep_time'], gpu['train_time']]
    speedups   = [c / g for c, g in zip(cpu_times, gpu_times)]

    x     = np.arange(len(stages))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1a')

    # ── Left: Bar chart ────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')
    bars_cpu = ax1.bar(x - width/2, cpu_times, width, label='CPU  (NumPy)',
                       color='#4a9eff', edgecolor='#2979ff', linewidth=0.8)
    bars_gpu = ax1.bar(x + width/2, gpu_times, width, label='GPU  (CUDA)',
                       color='#00e676', edgecolor='#00c853', linewidth=0.8)

    ax1.set_ylabel('Time (seconds)', color='white', fontsize=12)
    ax1.set_title('Execution Time: CPU vs GPU', color='white', fontsize=14, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='white')
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#334155')
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white',
               fontsize=11)

    # value labels
    for bar in bars_cpu:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                 f'{h:.2f}s', ha='center', va='bottom', color='#93c5fd', fontsize=9)
    for bar in bars_gpu:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                 f'{h:.2f}s', ha='center', va='bottom', color='#86efac', fontsize=9)

    # ── Right: Speedup bar chart ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    colors = ['#facc15' if s >= 2 else '#fb923c' for s in speedups]
    sp_bars = ax2.bar(x, speedups, 0.45, color=colors, edgecolor='#78350f', linewidth=0.8)

    ax2.axhline(1.0, color='#ef4444', linestyle='--', linewidth=1.2, label='1× (no speedup)')
    ax2.set_ylabel('Speedup (CPU time / GPU time)', color='white', fontsize=12)
    ax2.set_title('GPU Speedup over CPU', color='white', fontsize=14, pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4, color='white')
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white',
               fontsize=11)

    for bar, sp in zip(sp_bars, speedups):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f'{sp:.1f}×', ha='center', va='bottom', color='white',
                 fontsize=12, fontweight='bold')

    fig.suptitle(
        'Uber Fare Prediction  ·  CPU vs GPU Performance Benchmark\n'
        f'(~{NUM_FILES * 3.3:.0f}M NYC Yellow Taxi trips · 200 Gradient Descent epochs)',
        color='white', fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = 'benchmark_results.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Chart saved → {out}\n")


if __name__ == "__main__":
    run_benchmark()
