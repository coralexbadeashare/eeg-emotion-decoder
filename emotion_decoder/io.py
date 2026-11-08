"""Reading EEG recordings: EmotivPRO CSV exports and plain CSV files."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .preprocess import CHANNELS, SFREQ


class RecordingError(ValueError):
    pass


def _header_row(first_lines: list[str]) -> int:
    """EmotivPRO writes one metadata line before the column names; plain CSVs do not."""
    for i, line in enumerate(first_lines[:5]):
        cols = [c.strip() for c in line.split(",")]
        if "Timestamp" in cols or any(c in cols for c in (f"EEG.{CHANNELS[0]}", CHANNELS[0])):
            return i
    raise RecordingError("no header row with 'Timestamp' or the EEG channel names in the first 5 lines")


def read_recording(source) -> dict:
    """Read a recording from a path or a file-like object.

    Accepted columns: `EEG.<channel>` (EmotivPRO) or `<channel>`, for the 30 Emotiv FLEX2
    channels, in uV at 256 Hz. `Timestamp` (unix seconds) is optional; without it, time starts
    at 0. EmotivPRO marker columns (`MarkerIndex`, `MarkerValueInt`) are read when present.

    Returns dict(ts (N,), eeg (30, N) float32, markers (unix seconds), fs_estimate, n_raw_columns)
    """
    if hasattr(source, "read"):
        raw = source.read()
        raw = raw.encode() if isinstance(raw, str) else raw
        buf = io.BytesIO(raw)
    else:
        buf = open(source, "rb")                        # large files: never decoded as a whole
    head = [buf.readline().decode("utf-8", errors="replace").lstrip("\ufeff") for _ in range(5)]
    buf.seek(0)
    skip = _header_row(head)
    cols = [c.strip() for c in head[skip].split(",")]

    prefixed = all(f"EEG.{c}" in cols for c in CHANNELS)
    plain = all(c in cols for c in CHANNELS)
    if not (prefixed or plain):
        missing = [c for c in CHANNELS if f"EEG.{c}" not in cols and c not in cols]
        raise RecordingError(f"missing EEG channels: {', '.join(missing)} "
                             f"(the models need all 30 Emotiv FLEX2 channels)")
    eeg_cols = [f"EEG.{c}" if prefixed else c for c in CHANNELS]
    marker_cols = [c for c in ("MarkerIndex", "MarkerValueInt") if c in cols]
    use = (["Timestamp"] if "Timestamp" in cols else []) + eeg_cols + marker_cols

    df = pd.read_csv(buf, skiprows=skip, usecols=use, on_bad_lines="skip", low_memory=False, encoding="utf-8-sig")
    if not hasattr(source, "read"):
        buf.close()
    df = df.apply(pd.to_numeric, errors="coerce")
    if "Timestamp" in df:
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        ts = df["Timestamp"].to_numpy(float)
    else:
        ts = np.arange(len(df)) / SFREQ
    if len(df) == 0:
        raise RecordingError("no data rows")

    markers = np.zeros(0)
    if "MarkerIndex" in df:
        m = df["MarkerIndex"].fillna(0).to_numpy() != 0
        markers = ts[m]

    dt = np.diff(ts)
    fs = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else float("nan")
    return {"ts": ts, "eeg": df[eeg_cols].to_numpy(np.float32).T, "markers": markers,
            "fs_estimate": fs, "n_raw_columns": len(cols)}


def read_onsets_table(source) -> pd.DataFrame:
    """CSV of stimulus onsets: a column `onset_unix` (seconds, same clock as the recording's Timestamp)
    or, failing that, the first column. Other columns (e.g. `emotion`, `stimulus`) are kept and copied
    to the output; a column `emotion` with the 5 emotion names lets the tools score accuracy."""
    df = pd.read_csv(source)
    col = "onset_unix" if "onset_unix" in df else df.columns[0]
    df = df.rename(columns={col: "onset_unix"})
    df["onset_unix"] = pd.to_numeric(df["onset_unix"], errors="coerce")
    return df.dropna(subset=["onset_unix"]).reset_index(drop=True)


def read_onsets(source) -> np.ndarray:
    return read_onsets_table(source)["onset_unix"].to_numpy(float)
