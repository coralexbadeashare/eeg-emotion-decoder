"""Replay a recording CSV as a fake headset over LSL (an EEG stream + a Markers stream), to test
live_lsl.py without a headset. Needs: pip install pylsl

  python tools/replay_csv_to_lsl.py examples/example_1_clear.csv
  python tools/replay_csv_to_lsl.py my.csv --onsets my_onsets.csv --speed 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emotion_decoder.io import read_recording, read_onsets          # noqa: E402
from emotion_decoder.preprocess import CHANNELS, SFREQ              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--onsets", help="onsets CSV, if the recording has no markers")
    ap.add_argument("--speed", type=float, default=1.0, help="replay faster than real time (timestamps stay real)")
    a = ap.parse_args()
    import pylsl

    rec = read_recording(a.csv)
    onsets = read_onsets(a.onsets) if a.onsets else rec["markers"]
    info = pylsl.StreamInfo("ReplayFLEX2", "EEG", len(CHANNELS), SFREQ, "float32", "replay-eeg")
    chs = info.desc().append_child("channels")
    for c in CHANNELS:
        chs.append_child("channel").append_child_value("label", f"EEG.{c}")
    eeg_out = pylsl.StreamOutlet(info, chunk_size=32)
    mk_out = pylsl.StreamOutlet(pylsl.StreamInfo("ReplayMarkers", "Markers", 1, 0, "string", "replay-mk"))
    print(f"streaming {rec['eeg'].shape[1] / SFREQ:.0f} s, {len(onsets)} markers, speed x{a.speed}")
    time.sleep(2.0)                                                # let consumers connect

    t0_lsl = pylsl.local_clock(); t0_rec = rec["ts"][0]; wall0 = time.time()
    mk_i, step = 0, 32
    for i in range(0, rec["eeg"].shape[1], step):
        stamps = t0_lsl + (rec["ts"][i:i + step] - t0_rec)
        for s, t in zip(rec["eeg"][:, i:i + step].T, stamps):
            eeg_out.push_sample(s.tolist(), t)
        while mk_i < len(onsets) and onsets[mk_i] <= rec["ts"][min(i + step, len(rec["ts"]) - 1)]:
            mk_out.push_sample([f"stim{mk_i + 1}"], t0_lsl + onsets[mk_i] - t0_rec); mk_i += 1
        lag = (rec["ts"][i] - t0_rec) / a.speed - (time.time() - wall0)
        if lag > 0:
            time.sleep(lag)
    print("done")


if __name__ == "__main__":
    main()
