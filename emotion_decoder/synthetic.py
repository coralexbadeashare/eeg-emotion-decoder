"""Synthetic example recordings in EmotivPRO CSV format (the files in examples/).

They are NOT brain data. Background: coloured noise per channel, a 10 Hz posterior rhythm, a little
50 Hz line noise and an Emotiv-like DC offset. At every stimulus onset we add the model's own
average response for a chosen emotion (its class template, mapped back from aligned to channel
space), with a `gain` that controls how distinct the emotions are. Because the true emotion of
each trial is known, the examples are a positive control: the pipeline should recover it.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .model import EmotionModel, session_covariance, inverse_sqrt, N_BINS, BIN
from .preprocess import CHANNELS, SFREQ, ONSET, cut_trials

POSTERIOR = {"O1", "O2", "Oz", "PO9", "PO10", "P7", "P8", "Pz", "P3", "P4"}


def background(n: int, rng) -> np.ndarray:
    t = np.arange(n) / SFREQ
    spec = np.fft.rfft(rng.standard_normal((len(CHANNELS), n)), axis=1)
    spec /= np.maximum(np.fft.rfftfreq(n, 1 / SFREQ), 1.0) ** 0.5           # ~1/f power
    x = np.fft.irfft(spec, n=n, axis=1)
    x *= 8.0 / x.std(axis=1, keepdims=True)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * t / 20.0)
    for i, ch in enumerate(CHANNELS):
        x[i] += (6.0 if ch in POSTERIOR else 2.0) * env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    x += 1.5 * np.sin(2 * np.pi * 50.0 * t)
    mix = np.eye(len(CHANNELS)) + 0.3 * rng.standard_normal((len(CHANNELS), len(CHANNELS))) / np.sqrt(len(CHANNELS))
    return mix @ x                                                          # correlated channels, like real EEG


def make_session(emotions_seq, gap_s: float = 6.0, gain: float = 1.0, seed: int = 0,
                 start_unix: float = 1_780_000_000.0, model: EmotionModel | None = None):
    """-> (recording dict as io.read_recording returns it, onsets table with the true emotion)."""
    model = model or EmotionModel()
    rng = np.random.default_rng(seed)
    n_tr = len(emotions_seq)
    onset_s = 3.0 + gap_s * np.arange(n_tr) + rng.uniform(-0.3, 0.3, n_tr)
    n = int((onset_s[-1] + 7.0) * SFREQ)
    ts = start_unix + np.arange(n) / SFREQ
    eeg = background(n, rng)

    trials, _ = cut_trials(ts, eeg, start_unix + onset_s)
    R_half = np.linalg.inv(inverse_sqrt(session_covariance(trials)))      # aligned -> channel space
    T = model.templates()                                                  # (5, 30, 20) aligned units
    Tbar = T.mean(0)
    k_of = {e: i for i, e in enumerate(model.emotions)}
    for s, emo in zip(onset_s, emotions_seq):
        tpl = Tbar + gain * (T[k_of[emo]] - Tbar)                          # (30, 20)
        wave = np.repeat(tpl, BIN, axis=1)                                 # (30, 240), 0-937.5 ms
        i0 = int(np.searchsorted(ts, start_unix + s - 0.5)) + ONSET
        eeg[:, i0:i0 + N_BINS * BIN] += R_half @ wave
    eeg += 4200.0
    marker = np.zeros(n, int)
    for k, s in enumerate(onset_s):
        marker[int(np.searchsorted(ts, start_unix + s))] = k + 1
    rec = {"ts": ts, "eeg": eeg.astype(np.float32), "markers": ts[marker > 0], "marker_column": marker,
           "fs_estimate": float(SFREQ), "n_raw_columns": 32}
    onsets = pd.DataFrame({"onset_unix": np.round(start_unix + onset_s, 4), "emotion": list(emotions_seq)})
    return rec, onsets


def to_emotiv_csv(rec: dict, title: str = "synthetic example", with_markers: bool = True) -> bytes:
    ts, eeg, mk = rec["ts"], rec["eeg"], rec["marker_column"]
    out = io.StringIO()
    out.write(f"title:{title},start timestamp:{ts[0]:.6f},stop timestamp:{ts[-1]:.6f},"
              f"headset type:FLEX2,sampling rate:eeg_{SFREQ},note:SYNTHETIC - not brain data\n")
    cols = ["Timestamp"] + [f"EEG.{c}" for c in CHANNELS] + (["MarkerIndex"] if with_markers else [])
    out.write(",".join(cols) + "\n")
    body = np.column_stack([ts, eeg.T])
    for i in range(len(ts)):
        row = f"{body[i, 0]:.4f}," + ",".join(f"{v:.2f}" for v in body[i, 1:])
        if with_markers:
            row += "," + (str(mk[i]) if mk[i] else "")
        out.write(row + "\n")
    return out.getvalue().encode()
