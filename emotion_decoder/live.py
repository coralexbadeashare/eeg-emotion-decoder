"""Live decoding: a rolling EEG buffer + windows decoded as soon as they are complete.

Sources (all give (timestamps, eeg (30, n), marker_times) when polled):
  CsvTail      a CSV that another program keeps appending to (reads only the new lines)
  LslSource    a Lab Streaming Layer EEG stream (+ optional Markers stream); needs pylsl
  ReplaySource a recording replayed at real (or faster) speed, for demos without a headset

    dec = LiveDecoder(every=3.0, calib=10)
    src = CsvTail("C:/data/recording.csv")
    while True:
        dec.push(*src.poll()); new = dec.step()       # new: list of decoded windows
"""
from __future__ import annotations

import os
import time
import warnings
from collections import deque

import numpy as np

from .io import _header_row
from .model import EmotionModel, session_covariance, inverse_sqrt

ALIGN_WINDOWS = 200                # the session alignment uses the covariance of the last 200 windows
from .preprocess import CHANNELS, SFREQ, PRE_S, POST_S, N_SAMPLES, is_flat, channel_index

BUFFER_S = 120


class LiveDecoder:
    """Decode a live EEG stream into 5 emotion weights.

    every        a window every `every` seconds (ignored with use_markers=True)
    calib        the first `calib` windows only calibrate the session alignment (no output)
    block        block_weights = mean of the last `block` windows' probabilities
    use_markers  windows start at the marker times given to push() instead of every `every` s
    channel_names  the channel names of the eeg you will push, in their order (e.g. the labels of your
                 stream, "Cz" or "EEG.Cz"); the 30 model channels are then picked by name, extra
                 channels are ignored and missing ones raise. Without it, push() expects exactly the
                 30 channels in CHANNELS order.

    One LiveDecoder = one session (one headset placement): create a new one if the headset is re-placed.
    """

    def __init__(self, every: float = 3.0, calib: int = 10, block: int = 4, use_markers: bool = False,
                 channel_names=None, model: EmotionModel | None = None):
        self.every, self.calib, self.block, self.use_markers = every, calib, block, use_markers
        self.channel_names = list(channel_names) if channel_names is not None else None
        self._idx = channel_index(self.channel_names) if self.channel_names is not None else None
        self._warned: set[str] = set()
        self.model = model or EmotionModel()
        self.ts = deque(maxlen=BUFFER_S * SFREQ)
        self.x = deque(maxlen=BUFFER_S * SFREQ)
        self.pending: list[float] = []
        self.next_start: float | None = None
        self._covs: deque = deque(maxlen=ALIGN_WINDOWS)   # per-window covariances (bounded memory)
        self.n_windows = 0                                  # usable windows seen (calibration ones included)
        self.results: list[dict] = []
        self.n_samples = 0
        self.skipped = 0

    def _warn(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg, stacklevel=3)

    def _check(self, ts: np.ndarray, eeg: np.ndarray) -> np.ndarray:
        """Validate one pushed chunk; returns it as (30, n) float32 in CHANNELS order."""
        eeg = np.asarray(eeg, np.float32)
        n_exp = len(self.channel_names) if self.channel_names is not None else len(CHANNELS)
        if eeg.ndim != 2 or eeg.shape[0] != n_exp:
            hint = (" It looks transposed: pass eeg as (channels, samples)." if eeg.ndim == 2 and eeg.shape[1] == n_exp
                    else "" if self.channel_names is not None
                    else " Pass channel_names=... to LiveDecoder to select the 30 channels by name.")
            raise ValueError(f"eeg has shape {eeg.shape}; expected ({n_exp}, n_samples).{hint}")
        if eeg.shape[1] != len(ts):
            raise ValueError(f"{len(ts)} timestamps but {eeg.shape[1]} samples in eeg")
        if self._idx is not None:
            eeg = eeg[self._idx]
        if not np.isfinite(eeg).all():
            self._warn("nan", "the EEG contains NaN/inf values; they are replaced by 0.")
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        med = float(np.nanmedian(np.abs(eeg - np.nanmedian(eeg, axis=1, keepdims=True))))
        if 0 < med < 0.01:
            self._warn("units", "the EEG values are tiny: are they in volts? The model expects microvolts (multiply by 1e6).")
        return eeg

    def _check_rate(self):
        n = min(len(self.ts), 2 * SFREQ)
        if n >= SFREQ // 2 and "fs" not in self._warned:
            tail = np.fromiter((self.ts[k] for k in range(len(self.ts) - n, len(self.ts))), float)
            d = float(np.median(np.diff(tail)))
            fs = 1.0 / d if d > 0 else float("inf")
            if abs(fs - SFREQ) > 0.1 * SFREQ:
                self._warn("fs", f"the timestamps suggest {fs:.0f} Hz; the model expects {SFREQ} Hz "
                                 "(timestamps in seconds, one per sample).")

    def push(self, ts, eeg, markers=()):
        """Add new samples: ts (n,) in seconds, eeg (channels, n) in microvolts. markers: stimulus
        times (same clock as ts), used when use_markers=True. Call step() afterwards."""
        ts = np.asarray(ts, float).ravel()
        if len(ts):
            eeg = self._check(ts, eeg)
            self.ts.extend(ts); self.x.extend(eeg.T); self.n_samples += len(ts)
            self._check_rate()
        if self.use_markers:
            self.pending += [float(m) for m in markers]
        elif self.ts:
            if self.next_start is None:
                self.next_start = self.ts[0] + PRE_S
            while self.next_start + POST_S <= self.ts[-1] + self.every:
                self.pending.append(self.next_start); self.next_start += self.every

    @property
    def calibrated(self) -> bool:
        """True once more than `calib` windows were seen: the first `calib` only calibrate."""
        return self.n_windows > self.calib

    def step(self) -> list[dict]:
        """Decode every window that is now complete (5 s after its start). Returns the new results,
        each {"start", "weights", "block_weights", "proba", "decoded_at"}; all are kept in .results."""
        if not self.ts:
            return []
        ts = np.fromiter(self.ts, float)
        out = []
        for start in sorted(s for s in self.pending if s + POST_S <= ts[-1]):
            self.pending.remove(start)
            i0 = int(np.searchsorted(ts, start - PRE_S))
            if (i0 + N_SAMPLES > len(ts) or ts[i0] - (start - PRE_S) > 0.05
                    or ts[i0 + N_SAMPLES - 1] - ts[i0] - (N_SAMPLES - 1) / SFREQ > 0.05):   # dropped packets
                self.skipped += 1; continue
            win = np.stack([self.x[k] for k in range(i0, i0 + N_SAMPLES)], axis=1)
            if is_flat(win[None])[0]:
                self.skipped += 1; continue
            self._covs.append(session_covariance([win]))
            self.n_windows += 1
            self.model.W = inverse_sqrt(np.mean(self._covs, axis=0))
            if not self.calibrated:
                continue
            p = self.model.proba([win])[0]
            prev = [r["proba"] for r in self.results[-(self.block - 1):]] if self.block > 1 else []
            pb = np.mean(prev + [p], axis=0)
            r = {"start": start, "proba": p, "weights": self.model.to_weights(p),
                 "block_weights": self.model.to_weights(pb), "decoded_at": time.time()}
            self.results.append(r); out.append(r)
        return out

    def recent_eeg(self, seconds: float = 10.0):
        n = min(len(self.ts), int(seconds * SFREQ))
        if n == 0:
            return np.zeros(0), np.zeros((30, 0))
        ts = np.fromiter(list(self.ts)[-n:], float)
        return ts, np.stack(list(self.x)[-n:], axis=1)


class CsvTail:
    """Follow a CSV that keeps growing (EmotivPRO-style or plain). Reads only complete new lines."""

    def __init__(self, path: str, from_end: bool = False):
        self.path = os.path.expanduser(path.strip().strip('"'))
        self.pos = 0
        self.cols = None
        self.buf = ""
        self.rows = 0
        if from_end:                                     # only data written from now on
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(1 << 16)
                self._init_header(head)
                f.seek(0, os.SEEK_END)
                self.pos = f.tell()
            self.drop_partial = True
        else:
            self.drop_partial = False

    def _init_header(self, text: str):
        lines = [l.lstrip("\ufeff") for l in text.splitlines()]
        k = _header_row(lines)
        cols = [c.strip() for c in lines[k].split(",")]
        pre = all(f"EEG.{c}" in cols for c in CHANNELS)
        if not (pre or all(c in cols for c in CHANNELS)):
            raise ValueError("the CSV header does not contain the 30 FLEX2 channels (EEG.Cz ... EEG.F8)")
        self.idx = [cols.index(f"EEG.{c}" if pre else c) for c in CHANNELS]
        self.its = cols.index("Timestamp") if "Timestamp" in cols else None
        self.imk = cols.index("MarkerIndex") if "MarkerIndex" in cols else None
        self.cols = cols
        consumed = sum(len(l) + 1 for l in text.split("\n")[:k + 1])
        return consumed

    def poll(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"no such file: {self.path}")
        if os.path.getsize(self.path) < self.pos:          # truncated or replaced: start over
            self.pos, self.cols, self.buf, self.rows, self.drop_partial = 0, None, "", 0, False
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        self.buf += chunk
        if self.drop_partial:                            # we started mid-line: skip to the next line
            if "\n" not in self.buf:
                return np.zeros(0), np.zeros((30, 0)), []
            self.buf = self.buf.split("\n", 1)[1]
            self.drop_partial = False
        if self.cols is None:
            if self.buf.count("\n") < 2:
                return np.zeros(0), np.zeros((30, 0)), []
            used = self._init_header(self.buf)
            self.buf = self.buf[used:]
        if "\n" not in self.buf:
            return np.zeros(0), np.zeros((30, 0)), []
        complete, self.buf = self.buf.rsplit("\n", 1)
        ts, xs, mk = [], [], []
        for line in complete.split("\n"):
            p = line.split(",")
            try:
                x = [float(p[i]) for i in self.idx]
                t = float(p[self.its]) if self.its is not None else self.rows / SFREQ
            except (ValueError, IndexError):
                continue
            ts.append(t); xs.append(x); self.rows += 1
            if self.imk is not None and self.imk < len(p) and p[self.imk].strip() not in ("", "0"):
                mk.append(t)
        return np.array(ts), (np.array(xs, np.float32).T if xs else np.zeros((30, 0))), mk


class LslSource:
    def __init__(self, timeout: float = 5.0, markers: bool = False):
        import pylsl
        self.pylsl = pylsl
        found = pylsl.resolve_byprop("type", "EEG", timeout=timeout)
        if not found:
            raise RuntimeError("no LSL stream of type EEG found")
        self.inlet = pylsl.StreamInlet(found[0], max_buflen=60)
        info = self.inlet.info()
        labels, ch = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label").replace("EEG.", "")); ch = ch.next_sibling()
        if all(c in labels for c in CHANNELS):
            self.idx = channel_index(labels)
        elif info.channel_count() == 30:
            warnings.warn("the LSL stream has no channel labels: assuming the 30 channels are in CHANNELS order")
            self.idx = list(range(30))
        else:
            raise RuntimeError(f"cannot find the 30 FLEX2 channels in the stream labels: {labels}")
        self.name = info.name()
        self.mk = None
        if markers:
            m = pylsl.resolve_byprop("type", "Markers", timeout=timeout)
            if not m:
                raise RuntimeError("no LSL stream of type Markers found")
            self.mk = pylsl.StreamInlet(m[0])

    def poll(self):
        chunk, stamps = self.inlet.pull_chunk(timeout=0.0, max_samples=SFREQ * 30)
        ts = np.array(stamps) + self.inlet.time_correction() if stamps else np.zeros(0)
        x = np.array(chunk, np.float32)[:, self.idx].T if stamps else np.zeros((30, 0))
        mk = []
        if self.mk is not None:
            _, mts = self.mk.pull_chunk(timeout=0.0)
            mk = [t + self.mk.time_correction() for t in mts]
        return ts, x, mk


class ReplaySource:
    """Plays a recording (dict from io.read_recording) as if it were arriving now."""

    def __init__(self, rec: dict, speed: float = 1.0):
        self.rec, self.speed = rec, speed
        self.wall0 = time.time()
        self.i = 0

    def poll(self):
        t_rec = self.rec["ts"][0] + (time.time() - self.wall0) * self.speed
        j = int(np.searchsorted(self.rec["ts"], t_rec))
        ts, x = self.rec["ts"][self.i:j], self.rec["eeg"][:, self.i:j]
        mk = [m for m in self.rec["markers"] if self.rec["ts"][self.i] <= m < self.rec["ts"][max(j - 1, 0)]] if j > self.i else []
        self.i = j
        return ts, x, mk

    @property
    def finished(self):
        return self.i >= len(self.rec["ts"])
