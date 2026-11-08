"""One call from a recording to emotion weights. Used by the CLI, the Streamlit app and Python code.

    from emotion_decoder import decode_csv
    table, details = decode_csv("recording.csv", onsets_csv="onsets.csv", block=4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import read_recording, read_onsets_table
from .model import EmotionModel
from .preprocess import cut_trials, is_flat, sliding_onsets

LABEL_COLUMNS = ("emotion", "label", "true_emotion")


@dataclass
class Details:
    trials: np.ndarray                 # (n, 30, 1408) every trial cut, flat ones included
    onsets: np.ndarray                 # (n,) unix seconds
    flat: np.ndarray                   # (n,) bool
    proba: np.ndarray                  # (n, 5) NaN for flat trials
    model: EmotionModel
    recording: dict | None = None
    extra: pd.DataFrame = field(default_factory=pd.DataFrame)


def decode_trials(trials, onsets=None, block: int = 1, mode: str = "model", sharp: float = 1.0,
                  model: EmotionModel | None = None, extra: pd.DataFrame | None = None):
    """trials: (n, 30, 1408) of ONE session. The session alignment is computed from its non-flat trials."""
    model = model or EmotionModel()
    trials = np.asarray(trials, np.float32)
    n = len(trials)
    onsets = np.full(n, np.nan) if onsets is None else np.asarray(onsets, float)
    flat = is_flat(trials)
    good = np.where(~flat)[0]
    proba = np.full((n, len(model.emotions)), np.nan)
    if len(good):
        model.calibrate(trials[good])
        proba[good] = model.proba(trials[good])

    rows = []
    for j in range(n):
        row = {"trial": j + 1, "onset_unix": onsets[j], "flat": bool(flat[j])}
        if not flat[j]:
            w = model.to_weights(proba[j], mode, sharp)
            row.update({f"w_{k}": v for k, v in w.items()})
            row["top_emotion"] = max(w, key=w.get)
            if block > 1:
                prev = [i for i in good if i <= j][-block:]
                wb = model.to_weights(proba[prev].mean(0), mode, sharp)
                row.update({f"block{block}_{k}": v for k, v in wb.items()})
                row[f"block{block}_top"] = max(wb, key=wb.get)
        rows.append(row)
    table = pd.DataFrame(rows)
    if extra is not None and len(extra) == n:
        table = pd.concat([table, extra.reset_index(drop=True)], axis=1)
    t0 = np.nanmin(onsets) if np.isfinite(onsets).any() else np.nan
    table.insert(2, "t_s", (onsets - t0).round(3))
    return table, Details(trials, onsets, flat, proba, model, extra=extra if extra is not None else pd.DataFrame())


def decode_recording(rec: dict, onsets=None, onset_table: pd.DataFrame | None = None, sliding_step: float | None = 3.0,
                     use_markers: bool = False, block: int = 1, mode: str = "model", sharp: float = 1.0,
                     model: EmotionModel | None = None):
    """rec: output of io.read_recording. Default: EEG only, a window every `sliding_step` seconds.
    Optional: stimulus onsets (`onsets` / `onset_table` with column onset_unix) or, with
    use_markers=True, the markers stored in the file."""
    extra = None
    if onset_table is not None:
        onsets = onset_table["onset_unix"].to_numpy(float)
        extra = onset_table.drop(columns=["onset_unix"])
    if onsets is None:
        if use_markers:
            if not len(rec["markers"]):
                raise ValueError("this recording has no markers (EmotivPRO MarkerIndex column is empty)")
            onsets = rec["markers"]
        elif sliding_step:
            onsets = sliding_onsets(rec["ts"], sliding_step)
        else:
            raise ValueError("no windows: give sliding_step, onsets, or use_markers=True")
    onsets = np.asarray(onsets, float)
    trials, ok = cut_trials(rec["ts"], rec["eeg"], onsets)
    if len(ok) == 0:
        raise ValueError(f"none of the {len(onsets)} onsets has a full trial inside the recording "
                         "(a trial needs 0.5 s before and 5.0 s after the onset; same clock as Timestamp?)")
    ex = extra.iloc[ok].reset_index(drop=True) if extra is not None else None
    table, det = decode_trials(trials, onsets[ok], block, mode, sharp, model, ex)
    det.recording = rec
    table.attrs["n_onsets"] = len(onsets)
    table.attrs["n_outside"] = len(onsets) - len(ok)
    return table, det


def decode_csv(csv, onsets_csv=None, **kw):
    """Path (or file object) of an EmotivPRO CSV export -> (table, details)."""
    rec = read_recording(csv)
    table = read_onsets_table(onsets_csv) if onsets_csv is not None else None
    return decode_recording(rec, onset_table=table, **kw)


def label_accuracy(table: pd.DataFrame, emotions) -> dict | None:
    """If the onsets CSV carried a label column with the 5 emotion names, score the top emotion."""
    col = next((c for c in LABEL_COLUMNS if c in table), None)
    if col is None:
        return None
    lab = table[col].astype(str).str.strip().str.lower()
    names = {e.lower(): e for e in emotions}
    m = lab.isin(names) & ~table["flat"]
    if m.sum() == 0:
        return None
    out = {"label_column": col, "n": int(m.sum()), "chance": 1 / len(emotions),
           "accuracy": float((table.loc[m, "top_emotion"].str.lower() == lab[m]).mean())}
    bcol = next((c for c in table if c.startswith("block") and c.endswith("_top")), None)
    if bcol:
        out["block_accuracy"] = float((table.loc[m, bcol].str.lower() == lab[m]).mean())
    return out
