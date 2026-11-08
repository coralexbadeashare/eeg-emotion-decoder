# EEG emotion decoder

[![tests](https://github.com/coralexbadeashare/eeg-emotion-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/coralexbadeashare/eeg-emotion-decoder/actions/workflows/tests.yml)

Weights for **5 emotions** (Joy, Tenderness, Serenity, Nostalgia, Melancholy) for every image stimulus in
an **Emotiv FLEX2** EEG recording. Three ways to use it, all on a plain laptop (CPU, no GPU):

| | |
|---|---|
| **Streamlit app** | upload a CSV and watch every step, trial by trial: `streamlit run streamlit_app.py` |
| **Command line** | `python -m emotion_decoder recording.csv --onsets onsets.csv --out weights.csv` |
| **Python** | `from emotion_decoder import decode_csv` |
| **Live** (work in progress) | `python live_lsl.py`: from the headset through Lab Streaming Layer |

Pull requests are welcome from anyone; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```bash
git clone https://github.com/coralexbadeashare/eeg-emotion-decoder
cd eeg-emotion-decoder
pip install -r requirements.txt          # numpy, scipy, pandas, streamlit, plotly
```

## Try it on the examples

```bash
streamlit run streamlit_app.py
python -m emotion_decoder examples/example_1_clear.csv --block 4
python -m emotion_decoder examples/example_3_hard_no_markers.csv --onsets examples/example_3_hard_no_markers_onsets.csv
```

The three files in [examples/](examples/) are **synthetic**: background noise plus, at every stimulus, the
model's own average response for a known emotion. They show the file formats and let you check that the
pipeline recovers what was put in (100%, 90% and 78% of trials). They say nothing about accuracy on real EEG.

## Your own recordings

**1. The recording**: the CSV that EmotivPRO exports (the band-passed export, `*.md.bp.csv`, when there
are two). The tool reads the `Timestamp` column and the 30 columns `EEG.Cz` … `EEG.F8`. Everything else in
the file is ignored. A plain CSV with columns `Cz, Fz, …, F8` (µV, 256 Hz) also works. One file = one session
(one headset placement).

**2. When each stimulus appeared**. Either:
- markers inside the CSV (EmotivPRO's `MarkerIndex` column), used automatically; or
- an onsets CSV with a column `onset_unix`: stimulus onset in Unix seconds, on the same clock as the
  recording's `Timestamp` column. Extra columns are copied to the output. A column **`emotion`** holding
  the true emotion names makes the tools report accuracy.

```csv
onset_unix,emotion,stimulus
1778083701.512,Joy,img_014.jpg
1778083709.034,Melancholy,img_102.jpg
```

**3. Run**

```bash
python -m emotion_decoder rec.csv --onsets onsets.csv --block 4 --out weights.csv
```

The output has one row per stimulus: `w_Joy … w_Melancholy` (sum to 1), `top_emotion`, and with
`--block N` the same averaged over the last N trials (more accurate). `--mode calibrated` gives honest,
nearly uniform probabilities. Trials where the headset was not recording (flat) get no weights.

**From Python**

```python
from emotion_decoder import decode_csv, EmotionModel

table, details = decode_csv("rec.csv", "onsets.csv", block=4)

# or with trials you already cut: (n, 30, 1408), 256 Hz, from 0.5 s before to 5.0 s after each onset
m = EmotionModel().calibrate(session_trials)      # one session, no labels needed
m.weights(trial)                                  # {"Joy": 0.31, "Tenderness": 0.12, ...}
m.weights_block(last_four_trials)
```

## How it works

1. **Cut** each trial from 0.5 s before to 5.0 s after the stimulus: 1408 samples × 30 channels.
2. **Align the session** (Euclidean alignment): R = average spatial covariance of the session's trials
   (1–45 Hz, 0–4 s); every trial X becomes R^-½ X. This removes differences in headset placement between
   sessions and needs no labels, only the session's own EEG.
3. **Evoked response**: filter 1–30 Hz, subtract the 0.5 s before the stimulus, keep 0–940 ms and average
   it into 20 bins of 47 ms: 30 channels × 20 bins = 600 numbers.
4. **Shrinkage LDA**: distance of those 600 numbers to each emotion's average response, measured against
   trial-to-trial noise → 5 probabilities.

The model (`models/emotion5_v2.json`) is plain numbers exported from the scikit-learn checkpoint in
[eeg-stimulus-type-baselines](https://github.com/coralexbadeashare/eeg-stimulus-type-baselines)
(`checkpoints/emotion5`). The tests check that this code gives the same weights as the original; on the
844 training-set trials the difference is exactly 0.

## How good is it

Trained on **one participant** (370 image trials, Emotiv FLEX2, 256 Hz, EmotivPRO band-passed export, no
other cleaning). Top emotion, chance 20%:

| test | accuracy |
|---|---|
| recordings never seen in training (leave one recording out) | 26.3% (p = 0.0006) |
| held-out test recordings, untouched during development | 28.4% (p = 0.007) |
| average over a block of 2 / 8 trials | 28.6% / 33.4% |
| pictures never seen in training | 22.6% (p = 0.12, **not significant**) |
| videos, action images | chance |

So: above chance, but weak; it may partly reflect the specific pictures; it works for images only; the raw
probabilities are over-confident. Use the weights as an **EEG-driven control signal** (for example for
generative art), not as a measurement of someone's emotion. Other people and other headsets are untested.

## Live (work in progress)

`live_lsl.py` reads the EEG and a stimulus-marker stream from **Lab Streaming Layer**; EmotivPRO can stream
to LSL. The first `--calib` trials calibrate the session alignment; after that each stimulus is decoded 5 s
after its onset and printed. Untested with a real headset so far. It was tested with a fake headset that
replays a CSV:

```bash
pip install -r requirements-live.txt
python tools/replay_csv_to_lsl.py examples/example_1_clear.csv --speed 4   # terminal 1: fake headset
python live_lsl.py --calib 8                                               # terminal 2
```

Without a marker stream, `--every 6` decodes a window every 6 s (experimental: the model was trained on
stimulus onsets).

## Files

| | |
|---|---|
| `emotion_decoder/` | the package: `io` (CSV reading), `preprocess` (cutting), `model` (alignment, ERP, LDA), `pipeline`, `cli` |
| `streamlit_app.py` | the app |
| `live_lsl.py` | live decoding over LSL (work in progress) |
| `models/emotion5_v2.json` | the model |
| `examples/` | three synthetic recordings + onsets files |
| `tools/` | model export, example generation, CSV → LSL replay |
| `tests/` | `pytest -q` |

## License

MIT
