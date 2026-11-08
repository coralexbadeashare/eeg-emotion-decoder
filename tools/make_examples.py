"""Regenerate the synthetic example files in examples/ (deterministic).

    python tools/make_examples.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emotion_decoder.synthetic import make_session, to_emotiv_csv          # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "examples"
BLOCKS = {
    1: ["Joy", "Serenity", "Melancholy", "Tenderness", "Nostalgia"],
    2: ["Melancholy", "Joy", "Nostalgia", "Serenity", "Tenderness"],
    3: ["Tenderness", "Melancholy", "Joy", "Nostalgia", "Serenity"],
}
SPEC = {   # gain: how distinct the emotions are; 3 = no markers in the CSV + two dropouts
    1: dict(gain=2.0, name="example_1_clear", markers=True),
    2: dict(gain=1.0, name="example_2_moderate", markers=True),
    3: dict(gain=0.6, name="example_3_hard_no_markers", markers=False),
}


def main():
    OUT.mkdir(exist_ok=True)
    for k, s in SPEC.items():
        seq = [e for e in BLOCKS[k] for _ in range(4)]                          # 5 blocks of 4 trials
        rec, onsets = make_session(seq, gain=s["gain"], seed=k)
        if k == 3:                                                              # headset dropouts
            for j in (6, 13):
                i0 = int(np.searchsorted(rec["ts"], onsets.onset_unix[j] - 0.5))
                rec["eeg"][:, i0:i0 + 1408] = 0.0
        (OUT / f"{s['name']}.csv").write_bytes(to_emotiv_csv(rec, s["name"], with_markers=s["markers"]))
        onsets.to_csv(OUT / f"{s['name']}_onsets.csv", index=False)
        print(f"{s['name']}: {len(seq)} trials, gain {s['gain']}, {len(rec['ts']) / 256:.0f} s")


if __name__ == "__main__":
    main()
