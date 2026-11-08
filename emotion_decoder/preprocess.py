"""Cutting a recording into trials, exactly as the training trials were cut.

A trial = 30 channels x 1408 samples at 256 Hz: from 0.5 s before a stimulus onset to 5.0 s after.
Start = the first sample whose timestamp is at or after onset - 0.5 s. No filtering here.
"""
from __future__ import annotations

import numpy as np

SFREQ = 256
CHANNELS = ["Cz", "Fz", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5",
            "CP1", "P3", "P7", "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8",
            "P4", "CP2", "CP6", "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8"]
N_CH = len(CHANNELS)
PRE_S, POST_S = 0.5, 5.0
N_SAMPLES = 1408
ONSET, WIN = 128, 1024              # onset sample inside a trial; 4 s after it
FLAT_TOL = 1e-3


def cut_trials(ts: np.ndarray, eeg: np.ndarray, onsets) -> tuple[np.ndarray, np.ndarray]:
    """-> trials (n, 30, 1408) and the indices (into `onsets`) of the onsets that fit in the recording."""
    onsets = np.asarray(onsets, float)
    starts = np.searchsorted(ts, onsets - PRE_S)
    ok = np.where(starts + N_SAMPLES <= eeg.shape[1])[0]
    if len(ok) == 0:
        return np.zeros((0, N_CH, N_SAMPLES), np.float32), ok
    idx = starts[ok][:, None] + np.arange(N_SAMPLES)[None, :]
    return np.nan_to_num(eeg[:, idx]).transpose(1, 0, 2).astype(np.float32), ok


def channel_index(names) -> list[int]:
    """Positions of the 30 model channels (CHANNELS order) inside a list of channel names.
    Accepts "Cz" or EmotivPRO-style "EEG.Cz"; extra channels are ignored; missing ones raise."""
    clean = [str(n).strip().replace("EEG.", "") for n in names]
    missing = [c for c in CHANNELS if c not in clean]
    if missing:
        raise ValueError(f"missing EEG channel(s): {', '.join(missing)}. The model needs the 30 Emotiv FLEX2 "
                         f"channels {', '.join(CHANNELS)}; got {', '.join(clean)}")
    return [clean.index(c) for c in CHANNELS]


def select_channels(eeg, names) -> np.ndarray:
    """eeg (len(names), n) in any channel order, with extra channels -> (30, n) in CHANNELS order."""
    eeg = np.asarray(eeg)
    if eeg.ndim != 2 or eeg.shape[0] != len(names):
        hint = " (it looks transposed: pass (channels, samples))" if eeg.ndim == 2 and eeg.shape[1] == len(names) else ""
        raise ValueError(f"eeg has shape {eeg.shape} but {len(names)} channel names were given{hint}")
    return eeg[channel_index(names)]


MAX_MISSING_S = 0.05               # training trials with more missing time than this were excluded


def missing_seconds(ts: np.ndarray, onsets) -> np.ndarray:
    """Time lost to dropped packets in each window: data missing at its start (the first sample comes
    later than 0.5 s before the onset) plus gaps inside it (its 1408 samples span more than 5.5 s)."""
    want = np.asarray(onsets, float) - PRE_S
    starts = np.searchsorted(ts, want)
    late_start = ts[starts] - want                    # the window would begin after a gap (or before the recording)
    inside = ts[starts + N_SAMPLES - 1] - ts[starts] - (N_SAMPLES - 1) / SFREQ
    return np.maximum(late_start, 0.0) + np.maximum(inside, 0.0)


def is_flat(trials: np.ndarray) -> np.ndarray:
    """True where every channel is flat after onset (headset not recording)."""
    trials = np.asarray(trials)
    if len(trials) == 0:
        return np.zeros(0, bool)
    return trials[:, :, ONSET:ONSET + WIN].std(axis=2).max(axis=1) < FLAT_TOL


def sliding_onsets(ts: np.ndarray, step_s: float) -> np.ndarray:
    """Pseudo-onsets every step_s seconds (EXPERIMENTAL: the model was trained on stimulus onsets)."""
    first, last = ts[0] + PRE_S, ts[-1] - POST_S
    return np.arange(first, last, step_s) if last > first else np.zeros(0)
