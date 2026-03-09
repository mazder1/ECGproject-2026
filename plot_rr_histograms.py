"""
RR-interval histogram comparison across subjects per activity.

For each activity, plots overlapping histograms of RR-intervals for all
subjects, annotated with mean ± std. Also plots the inverse (instantaneous
HR in bpm). Saves one figure per activity to output/plots/.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Physiological plausibility filter: 30–200 bpm → 300–2000 ms
RR_MIN_MS = 300
RR_MAX_MS = 2000


def subject_label(stem: str) -> str:
    """Strip '_rr' suffix and clean up the filename for display."""
    return stem.replace("_rr", "").replace("-", " ").replace("_", " ")


def load_rr(path: pathlib.Path) -> np.ndarray:
    df = pd.read_csv(path)
    rr = df["rr_ms"].to_numpy()
    return rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]


def plot_activity(activity: str, rr_files: list[pathlib.Path], out_path: pathlib.Path):
    subjects = {subject_label(f.stem): load_rr(f) for f in rr_files}
    subjects = {k: v for k, v in subjects.items() if len(v) >= 5}
    n = len(subjects)
    if n == 0:
        return

    colors = cm.tab20(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Activity: {activity.title()}", fontsize=14, fontweight="bold")

    # --- Left panel: RR-interval histograms ---
    ax1 = axes[0]
    all_rr = np.concatenate(list(subjects.values()))
    bins = np.linspace(all_rr.min(), all_rr.max(), 35)

    for (name, rr), color in zip(subjects.items(), colors):
        mean, std = rr.mean(), rr.std()
        ax1.hist(rr, bins=bins, alpha=0.45, color=color, edgecolor="none",
                 label=f"{name}  μ={mean:.0f} σ={std:.0f} ms")

    ax1.set_xlabel("RR-interval (ms)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("RR-interval distribution", fontsize=12)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Right panel: Instantaneous HR (inverse of RR) histograms ---
    ax2 = axes[1]
    all_hr = 60_000 / all_rr
    bins_hr = np.linspace(all_hr.min(), all_hr.max(), 35)

    for (name, rr), color in zip(subjects.items(), colors):
        hr = 60_000 / rr
        mean_hr, std_hr = hr.mean(), hr.std()
        ax2.hist(hr, bins=bins_hr, alpha=0.45, color=color, edgecolor="none",
                 label=f"{name}  μ={mean_hr:.1f} σ={std_hr:.1f} bpm")

    ax2.set_xlabel("Instantaneous HR (bpm)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Instantaneous HR distribution (inverse of RR)", fontsize=12)
    ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}  ({n} subjects)")


def print_summary(activity: str, rr_files: list[pathlib.Path]):
    print(f"\n{'─'*55}")
    print(f"  {activity.upper()}")
    print(f"{'─'*55}")
    print(f"  {'Subject':<25} {'N':>5} {'Mean RR':>10} {'Std RR':>10} {'Mean HR':>10}")
    print(f"  {'':.<25} {'(beats)':>5} {'(ms)':>10} {'(ms)':>10} {'(bpm)':>10}")
    for f in rr_files:
        rr = load_rr(f)
        if len(rr) < 5:
            continue
        hr = 60_000 / rr
        name = subject_label(f.stem)
        print(f"  {name:<25} {len(rr):>5} {rr.mean():>10.1f} {rr.std():>10.1f} {hr.mean():>10.1f}")


def main():
    activity_dirs = sorted(
        d for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and d.name != "plots"
    )

    if not activity_dirs:
        print("No output data found. Run process_ecg.py first.")
        return

    for act_dir in activity_dirs:
        rr_files = sorted(act_dir.glob("*_rr.csv"))
        if not rr_files:
            continue
        activity = act_dir.name
        out_path = PLOTS_DIR / f"{activity.replace(' ', '_')}_histograms.png"
        plot_activity(activity, rr_files, out_path)
        print_summary(activity, rr_files)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
