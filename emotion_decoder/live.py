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
from collections import deque

import numpy as np

from .io import _header_row
from .model import EmotionModel, session_covariance, inverse_sqrt
from .preprocess import CHANNELS, SFREQ, PRE_S, POST_S, N_SAMPLES, is_flat

BUFFER_S = 120


class LiveDecoder:
    def __init__(self, every: float = 3.0, calib: int = 10, block: int = 4, use_markers: bool = False,
                 model: EmotionModel | None = None):
        self.every, self.calib, self.block, self.use_markers = every, calib, block, use_markers
        self.model = model or EmotionModel()
        self.ts = deque(maxlen=BUFFER_S * SFREQ)
        self.x = deque(maxlen=BUFFER_S * SFREQ)
        self.pending: list[float] = []
        self.next_start: float | None = None
        self.calib_windows: list[np.ndarray] = []
        self.results: list[dict] = []
        self.n_samples = 0
        self.skipped = 0

    def push(self, ts, eeg, markers=()):
        ts = np.asarray(ts, float)
        if len(ts):
            self.ts.extend(ts); self.x.extend(np.asarray(eeg, np.float32).T); self.n_samples += len(ts)
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
        return len(self.calib_windows) > self.calib

    def step(self) -> list[dict]:
        if not self.ts:
            return []
        ts = np.fromiter(self.ts, float)
        out = []
        for start in sorted(s for s in self.pending if s + POST_S <= ts[-1]):
            self.pending.remove(start)
            i0 = int(np.searchsorted(ts, start - PRE_S))
            if i0 + N_SAMPLES > len(ts) or ts[i0] - (start - PRE_S) > 0.05:
                self.skipped += 1; continue
            xs = list(self.x)
            win = np.stack(xs[i0:i0 + N_SAMPLES], axis=1)
            if is_flat(win[None])[0]:
                self.skipped += 1; continue
            self.calib_windows.append(win)
            self.model.W = inverse_sqrt(session_covariance(self.calib_windows[-200:]))
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
        lines = text.splitlines()
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
            raise FileNotFoundError(self.path)
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
            self.idx = [labels.index(c) for c in CHANNELS]
        elif info.channel_count() == 30:
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
