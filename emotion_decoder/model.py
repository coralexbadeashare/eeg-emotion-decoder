"""The 5-emotion model: Euclidean alignment + ERP shape + shrinkage LDA, in plain numpy.

Same computation as emotion_weights.py in eeg-stimulus-type-baselines (tests check it):

  calibrate(session trials)   R = mean spatial covariance of the session's trials (1-45 Hz, 0-4 s)
                              W = R^(-1/2)       -> removes headset-placement differences
  erp_features(W @ trial)     1-30 Hz band-pass, subtract the 0.5 s pre-stimulus mean,
                              keep 0-937.5 ms, average into 20 bins of 12 samples: 30 x 20 = 600
  proba                       standardise -> LDA scores -> softmax over the 5 emotions
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .preprocess import ONSET, WIN

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "emotion5_v2.json"
_SOS_COV = butter(4, [1, 45], btype="band", fs=256, output="sos")
_SOS_ERP = butter(4, [1, 30], btype="band", fs=256, output="sos")
N_BINS, BIN = 20, 12                       # 20 bins x 12 samples = 240 samples = 0-937.5 ms
BIN_MS = 1000.0 * BIN / 256


def session_covariance(trials) -> np.ndarray:
    """Mean spatial covariance (30 x 30) of the trials, 1-45 Hz, 0-4 s after onset."""
    Cs = []
    for e in trials:
        x = sosfiltfilt(_SOS_COV, np.asarray(e, np.float64), axis=1)[:, ONSET:ONSET + WIN]
        x = x - x.mean(1, keepdims=True)
        Cs.append(x @ x.T / x.shape[1])
    return np.mean(Cs, 0)


def inverse_sqrt(R: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(R)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def erp_filtered(e: np.ndarray) -> np.ndarray:
    """1-30 Hz, minus the pre-stimulus mean: the trial's evoked response, (30, 1408)."""
    x = sosfiltfilt(_SOS_ERP, np.asarray(e, np.float64), axis=1)
    return x - x[:, :ONSET].mean(1, keepdims=True)


def erp_features(e: np.ndarray) -> np.ndarray:
    """(30, 1408) aligned trial -> 600 = 30 channels x 20 time bins (0-937.5 ms)."""
    return erp_filtered(e)[:, ONSET:ONSET + N_BINS * BIN].reshape(30, N_BINS, BIN).mean(2).reshape(-1)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class EmotionModel:
    """
    m = EmotionModel()                       # loads models/emotion5_v2.json
    m.calibrate(session_trials)              # (n, 30, 1408) trials of ONE session, no labels needed
    m.weights(trial)                         # {"Joy": .., "Tenderness": .., ...}
    m.weights_block(last_trials)             # average over a block of trials: more accurate
    m.proba(trials)                          # (n, 5) probabilities
    """

    def __init__(self, path=MODEL_PATH):
        d = json.loads(Path(path).read_text())
        self.emotions = d["emotions"]
        self.T = float(d["temperature"])
        self.mean, self.scale = np.array(d["scaler_mean"]), np.array(d["scaler_scale"])
        self.coef, self.intercept = np.array(d["coef"]), np.array(d["intercept"])
        self.class_means_scaled = np.array(d["class_means_scaled"])
        self.info = {k: v for k, v in d.items() if not isinstance(v, list) or k == "emotions"}
        self.W = None

    # ---- session alignment
    def calibrate(self, session_trials) -> "EmotionModel":
        trials = list(session_trials)
        if len(trials) == 0:
            raise ValueError("calibrate() needs at least one trial of the session")
        self.W = inverse_sqrt(session_covariance(trials))
        return self

    def align(self, trial) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("call calibrate(session_trials) first: the model needs the session's alignment")
        return self.W @ np.asarray(trial, np.float64)

    # ---- features and scores
    def features(self, trials) -> np.ndarray:
        return np.stack([erp_features(self.align(e)) for e in trials])

    def scores(self, F) -> np.ndarray:
        return ((np.asarray(F, np.float64) - self.mean) / self.scale) @ self.coef.T + self.intercept

    def proba(self, trials) -> np.ndarray:
        return _softmax(self.scores(self.features(trials)))

    def proba_from_features(self, F) -> np.ndarray:
        return _softmax(self.scores(F))

    # ---- weights, as in emotion_weights.py
    def to_weights(self, p, mode: str = "model", sharp: float = 1.0) -> dict:
        """mode "model": the model's probabilities. "calibrated": divided by the temperature fitted on
        held-out data (honest but almost uniform). sharp > 1 only adds visual contrast."""
        z = np.log(np.clip(np.asarray(p, np.float64), 1e-9, 1))
        z = z / (self.T if mode == "calibrated" else 1.0) * sharp
        w = _softmax(z)
        return dict(zip(self.emotions, np.round(w, 4)))

    def weights(self, trial, mode="model", sharp=1.0) -> dict:
        return self.to_weights(self.proba([trial])[0], mode, sharp)

    def weights_block(self, trials, mode="model", sharp=1.0) -> dict:
        return self.to_weights(self.proba(trials).mean(0), mode, sharp)

    # ---- for explanations
    def templates(self) -> np.ndarray:
        """Average ERP feature of each emotion in training, (5, 30, 20), aligned units."""
        return (self.class_means_scaled * self.scale + self.mean).reshape(len(self.emotions), 30, N_BINS)
