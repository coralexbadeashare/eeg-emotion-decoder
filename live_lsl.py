"""WORK IN PROGRESS: live emotion weights from an Emotiv headset over Lab Streaming Layer (LSL).

Not yet tested with a real headset. Tested with tools/replay_csv_to_lsl.py, which replays a CSV as a
fake headset. Needs: pip install pylsl

How it works
  1. Finds an LSL stream of type EEG (EmotivPRO can stream to LSL; enable it in EmotivPRO's settings).
     Optional (--markers): a stream of type Markers carrying one marker per stimulus.
  2. Keeps the last 60 s of EEG. Picks the 30 FLEX2 channels by their labels in the stream description
     (EmotivPRO also streams counters, contact quality, ...).
  3. A window starts every --every seconds (default 3); each is decoded 5 s after its start. The first
     --calib windows only calibrate the session alignment (no output); the alignment keeps updating.
     With --markers, windows start at the stimulus markers instead.

  python live_lsl.py                        # EEG only, a window every 3 s
  python live_lsl.py --every 2 --calib 20
  python live_lsl.py --markers              # optional: windows at stimulus markers
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque

import numpy as np

from emotion_decoder.model import EmotionModel, session_covariance, inverse_sqrt
from emotion_decoder.preprocess import CHANNELS, SFREQ, PRE_S, POST_S, N_SAMPLES, is_flat


def find_stream(pylsl, kind: str, timeout: float):
    found = pylsl.resolve_byprop("type", kind, timeout=timeout)
    return pylsl.StreamInlet(found[0], max_buflen=60) if found else None


def channel_index(inlet) -> list[int]:
    info = inlet.info()
    labels, ch = [], info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label")); ch = ch.next_sibling()
    norm = [lab.replace("EEG.", "") for lab in labels]
    if all(c in norm for c in CHANNELS):
        return [norm.index(c) for c in CHANNELS]
    if info.channel_count() == len(CHANNELS):
        print("warning: no channel labels in the stream; assuming the FLEX2 order", CHANNELS)
        return list(range(len(CHANNELS)))
    raise SystemExit(f"cannot find the 30 FLEX2 channels among the stream's labels: {labels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=15, help="windows used only to calibrate the alignment")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--every", type=float, default=3.0, help="EEG only (default): a window every N s")
    ap.add_argument("--markers", action="store_true", help="optional: windows at a Markers stream instead")
    ap.add_argument("--out", default="live_weights.csv")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()
    import pylsl

    eeg_in = find_stream(pylsl, "EEG", a.timeout)
    if eeg_in is None:
        raise SystemExit("no LSL stream of type EEG found (is EmotivPRO streaming to LSL?)")
    idx = channel_index(eeg_in)
    mk_in = find_stream(pylsl, "Markers", a.timeout) if a.markers else None
    if a.markers and mk_in is None:
        raise SystemExit("--markers given but no LSL stream of type Markers found")
    print(f"EEG: {eeg_in.info().name()} @ {eeg_in.info().nominal_srate():g} Hz; "
          f"markers: {mk_in.info().name() if mk_in else f'every {a.every} s'}")

    model = EmotionModel()
    buf_t, buf_x = deque(maxlen=60 * SFREQ), deque(maxlen=60 * SFREQ)
    pending, calib, probs = [], [], []
    next_auto = None
    with open(a.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["onset_lsl"] + [f"w_{e}" for e in model.emotions] + [f"block{a.block}_{e}" for e in model.emotions])
        while True:
            chunk, stamps = eeg_in.pull_chunk(timeout=0.2)
            if stamps:
                off = eeg_in.time_correction()
                for s, t in zip(chunk, stamps):
                    buf_x.append(np.asarray(s, np.float32)[idx]); buf_t.append(t + off)
            if mk_in is not None:
                mks, mts = mk_in.pull_chunk(timeout=0.0)
                if mts:
                    off = mk_in.time_correction()
                    pending += [t + off for t in mts]
            elif buf_t:
                next_auto = next_auto or buf_t[0] + PRE_S
                while next_auto + POST_S < buf_t[-1]:
                    pending.append(next_auto); next_auto += a.every
            if not buf_t:
                continue
            ts = np.fromiter(buf_t, float)
            for onset in [o for o in pending if o + POST_S <= ts[-1]]:
                pending.remove(onset)
                i0 = int(np.searchsorted(ts, onset - PRE_S))
                if i0 + N_SAMPLES > len(ts) or ts[i0] - (onset - PRE_S) > 0.05:
                    print(f"skip stimulus at {onset:.2f}: EEG not in the buffer"); continue
                trial = np.stack(list(buf_x)[i0:i0 + N_SAMPLES], axis=1)
                if is_flat(trial[None])[0]:
                    print(f"skip stimulus at {onset:.2f}: flat EEG"); continue
                calib.append(trial)
                model.W = inverse_sqrt(session_covariance(calib))
                if len(calib) <= a.calib:
                    print(f"calibrating {len(calib)}/{a.calib}"); continue
                p = model.proba([trial])[0]; probs.append(p)
                w = model.to_weights(p); wb = model.to_weights(np.mean(probs[-a.block:], 0))
                wr.writerow([f"{onset:.3f}"] + list(w.values()) + list(wb.values())); fh.flush()
                top = max(wb, key=wb.get)
                print(time.strftime("%H:%M:%S"), " ".join(f"{k[:4]} {v:.2f}" for k, v in w.items()),
                      f"| last {a.block}: {top}")


if __name__ == "__main__":
    main()
