"""Live emotion weights from an Emotiv headset over Lab Streaming Layer (LSL), in a terminal.

Not yet tested with a real headset; tested with tools/replay_csv_to_lsl.py (a CSV replayed as a fake
headset). Needs: pip install pylsl   (or: pip install "emotion-decoder[live] @ git+...")

  1. Finds an LSL stream of type EEG (EmotivPRO can stream to LSL) and picks the 30 FLEX2 channels
     by label. Optional (--markers): a stream of type Markers, one marker per stimulus.
  2. A window starts every --every seconds (default 3), or at every marker with --markers, and is
     decoded 5 s after its start. The first --calib windows only calibrate the session alignment.
  3. Prints the weights and appends them to --out.

  python live_lsl.py                        # EEG only, a window every 3 s
  python live_lsl.py --every 2 --calib 20
  python live_lsl.py --markers              # optional: windows at stimulus markers
"""
from __future__ import annotations

import argparse
import csv
import time

from emotion_decoder.live import LiveDecoder, LslSource


def main():
    ap = argparse.ArgumentParser(description="Live emotion weights from an LSL EEG stream.")
    ap.add_argument("--calib", type=int, default=15, help="windows used only to calibrate the alignment")
    ap.add_argument("--block", type=int, default=4, help="also average over the last N windows")
    ap.add_argument("--every", type=float, default=3.0, help="EEG only (default): a window every N s")
    ap.add_argument("--markers", action="store_true", help="optional: windows at a Markers stream instead")
    ap.add_argument("--out", default="live_weights.csv")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the streams")
    a = ap.parse_args()

    try:
        src = LslSource(timeout=a.timeout, markers=a.markers)
    except ImportError:
        raise SystemExit("pylsl is not installed: pip install pylsl")
    except RuntimeError as e:
        raise SystemExit(f"error: {e} (is EmotivPRO streaming to LSL?)")
    dec = LiveDecoder(every=a.every, calib=a.calib, block=a.block, use_markers=a.markers)
    print(f"EEG: {src.name}; windows {'at the markers' if a.markers else f'every {a.every:g} s'}; "
          f"the first {a.calib} calibrate the alignment. Ctrl+C to stop.")

    seen = 0
    with open(a.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["start_lsl"] + [f"w_{e}" for e in dec.model.emotions] + [f"block{a.block}_{e}" for e in dec.model.emotions])
        try:
            while True:
                dec.push(*src.poll())
                for r in dec.step():
                    wr.writerow([f"{r['start']:.3f}"] + list(r["weights"].values()) + list(r["block_weights"].values()))
                    fh.flush()
                    wb = r["block_weights"]
                    print(time.strftime("%H:%M:%S"), " ".join(f"{k[:4]} {v:.2f}" for k, v in r["weights"].items()),
                          f"| last {a.block}: {max(wb, key=wb.get)}")
                if not dec.calibrated and dec.n_windows != seen:
                    seen = dec.n_windows
                    print(f"calibrating {seen}/{a.calib}")
                time.sleep(0.2)
        except KeyboardInterrupt:
            print(f"\nstopped: {len(dec.results)} windows decoded, {dec.skipped} skipped -> {a.out}")


if __name__ == "__main__":
    main()
