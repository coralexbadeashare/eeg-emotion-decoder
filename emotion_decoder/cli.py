"""Command line: EmotivPRO CSV (+ onsets) -> emotion weights per trial.

  python -m emotion_decoder RECORDING.csv                       # EEG only: a window every 3 s
  python -m emotion_decoder RECORDING.csv --every 2             # a window every 2 s
  python -m emotion_decoder RECORDING.csv --onsets ONSETS.csv   # optional: windows at stimulus onsets
  python -m emotion_decoder RECORDING.csv --markers             # optional: at the markers in the CSV
  python -m emotion_decoder trials.npy                          # (n, 30, 1408) windows already cut
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .pipeline import decode_csv, decode_trials, label_accuracy


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m emotion_decoder",
                                 description="5 emotion weights per trial from Emotiv FLEX2 EEG.")
    ap.add_argument("input", help="EmotivPRO CSV export of ONE recording session, or a .npy of trials (n, 30, 1408)")
    ap.add_argument("--every", type=float, default=3.0, help="EEG only (default): a window every N seconds")
    ap.add_argument("--onsets", help="optional: CSV with a column onset_unix (seconds, same clock as Timestamp); "
                                     "an optional column `emotion` is used to score accuracy")
    ap.add_argument("--markers", action="store_true", help="optional: windows at the markers inside the CSV")
    ap.add_argument("--out", default="emotion_weights.csv")
    ap.add_argument("--block", type=int, default=1, help="also average over the last N trials (more accurate)")
    ap.add_argument("--mode", choices=["model", "calibrated"], default="model")
    ap.add_argument("--sharp", type=float, default=1.0, help="> 1 adds visual contrast only")
    a = ap.parse_args(argv)

    try:
        if a.input.lower().endswith(".npy"):
            table, _ = decode_trials(np.load(a.input), block=a.block, mode=a.mode, sharp=a.sharp)
        else:
            table, _ = decode_csv(a.input, a.onsets, block=a.block, mode=a.mode, sharp=a.sharp,
                                  sliding_step=a.every, use_markers=a.markers)
    except ValueError as e:
        sys.exit(f"error: {e}")

    table.to_csv(a.out, index=False)
    n, n_flat = len(table), int(table["flat"].sum())
    print(f"{n} windows ({n_flat} flat, no prediction) -> {a.out}")
    if table.attrs.get("n_outside"):
        print(f"{table.attrs['n_outside']} onsets fell outside the recording and were skipped")
    if n - n_flat < 10:
        print("warning: fewer than 10 windows to calibrate the session alignment; weights will be noisy")
    acc = label_accuracy(table, [c[2:] for c in table if c.startswith("w_")])
    if acc:
        line = f"top emotion vs `{acc['label_column']}`: {acc['accuracy']:.1%} of {acc['n']} (chance {acc['chance']:.0%})"
        if "block_accuracy" in acc:
            line += f"; block of {a.block}: {acc['block_accuracy']:.1%}"
        print(line)
    cols = ["trial", "t_s"] + [c for c in table if c.startswith("w_")] + ["top_emotion"]
    print(table[[c for c in cols if c in table]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
