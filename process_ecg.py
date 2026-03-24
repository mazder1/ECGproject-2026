"""
ECG R-peak detection and RR-interval extraction.

Iterates through all activity folders, detects R-peaks using NeuroKit2,
calculates RR-intervals, and saves results per subject.

Sampling rate: 512 Hz
Output: output/<activity>/<subject>_rr.csv
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import neurokit2 as nk

SAMPLING_RATE = 512
BASE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Folders to skip (non-activity)
SKIP_DIRS = {".git", "output", "__pycache__"}

# Preferred ECG lead for Shimmer device (Lead II equivalent)
SHIMMER_ECG_LEAD = "LL-RA"


def is_lfs_pointer(path: pathlib.Path) -> bool:
    """Return True if the file is still a Git LFS pointer (not real data)."""
    try:
        first_line = path.open().readline()
        return first_line.startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def find_ecg_column(df: pd.DataFrame) -> str:
    """Return the most likely ECG column name from a DataFrame."""
    cols = df.columns.tolist()

    # 1. Prefer the standard Lead II equivalent for Shimmer data
    for col in cols:
        if SHIMMER_ECG_LEAD in col:
            return col

    # 2. Any column with ECG/EKG in the name (but not status columns)
    for col in cols:
        col_lower = col.lower()
        if ("ecg" in col_lower or "ekg" in col_lower) and "status" not in col_lower:
            return col

    # 3. Generic keyword match
    hints = {"signal", "ch1", "channel", "lead", "millivolt"}
    for col in cols:
        if any(h in col.lower() for h in hints):
            return col

    # 4. Single numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    # 5. Highest variance among numeric columns
    if numeric_cols:
        return df[numeric_cols].var().idxmax()

    raise ValueError(f"Cannot identify an ECG column. Columns: {cols}")


def load_ecg(csv_path: pathlib.Path) -> np.ndarray:
    """Load ECG signal from a Shimmer CSV file (tab-separated, units on row 2)."""
    if is_lfs_pointer(csv_path):
        raise ValueError("File is a Git LFS pointer — run `git lfs pull` first")

    # Shimmer format: row 0 = "sep=\t", row 1 = headers, row 2 = units, row 3+ = data
    # Detect separator from first line
    first_line = csv_path.open().readline().strip().lower()
    if first_line.startswith('"sep=') or first_line.startswith("sep="):
        sep = "\t"
        skip = [0, 2]   # skip sep= line and units row
    else:
        # Fallback: try tab, comma, semicolon
        sep = None
        skip = []
        for s in ("\t", ",", ";"):
            try:
                test = pd.read_csv(csv_path, sep=s, nrows=5)
                if test.shape[1] > 1:
                    sep = s
                    break
            except Exception:
                continue
        if sep is None:
            raise ValueError("Could not determine separator")

    df = pd.read_csv(csv_path, sep=sep, skiprows=skip, header=0, low_memory=False)
    col = find_ecg_column(df)
    signal = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return signal


def process_file(csv_path: pathlib.Path, activity: str) -> pd.DataFrame | None:
    """Detect R-peaks and compute RR-intervals for one file."""
    subject = csv_path.stem  # filename without extension

    try:
        ecg = load_ecg(csv_path)
    except Exception as e:
        print(f"  [SKIP] {activity}/{csv_path.name}: load error — {e}")
        return None

    if len(ecg) < SAMPLING_RATE * 5:
        print(f"  [SKIP] {activity}/{csv_path.name}: signal too short ({len(ecg)} samples)")
        return None

    try:
        # Clean signal then find peaks
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE)
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE)
        r_peaks = info["ECG_R_Peaks"]
    except Exception as e:
        print(f"  [SKIP] {activity}/{csv_path.name}: peak detection failed — {e}")
        return None

    if len(r_peaks) < 2:
        print(f"  [SKIP] {activity}/{csv_path.name}: fewer than 2 R-peaks found")
        return None

    # RR-intervals in milliseconds
    rr_ms = np.diff(r_peaks) / SAMPLING_RATE * 1000

    # Timestamps of the second R-peak in each pair (ms from start)
    rr_times_ms = r_peaks[1:] / SAMPLING_RATE * 1000

    df_out = pd.DataFrame({
        "time_ms": rr_times_ms.round(3),    # time of beat (ms from recording start)
        "rr_ms": rr_ms.round(3),             # RR-interval (ms)
        "rr_s": (rr_ms / 1000).round(6),     # RR-interval (s)
        "hr_bpm": (60_000 / rr_ms).round(2), # instantaneous HR (bpm)
    })

    n_beats = len(r_peaks)
    mean_hr = df_out["hr_bpm"].mean()
    print(f"  [OK]   {activity}/{csv_path.name}: {n_beats} beats, mean HR {mean_hr:.1f} bpm")
    return df_out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    errors: list[str] = []
    total_ok = 0

    activity_dirs = sorted(
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )

    if not activity_dirs:
        print("No activity folders found.")
        sys.exit(1)

    for act_dir in activity_dirs:
        csv_files = sorted(act_dir.glob("*.csv"))
        if not csv_files:
            continue

        activity = act_dir.name
        print(f"\n=== {activity} ({len(csv_files)} files) ===")

        out_act_dir = OUTPUT_DIR / activity
        out_act_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in csv_files:
            df_rr = process_file(csv_path, activity)
            if df_rr is None:
                errors.append(f"{activity}/{csv_path.name}")
                continue

            out_path = out_act_dir / f"{csv_path.stem}_rr.csv"
            df_rr.to_csv(out_path, index=False)
            total_ok += 1

    print(f"\n{'='*50}")
    print(f"Done. {total_ok} files processed successfully.")
    if errors:
        print(f"{len(errors)} files skipped:")
        for e in errors:
            print(f"  - {e}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
