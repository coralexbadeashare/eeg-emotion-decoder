# EEG emotion decoder

[![tests](https://github.com/coralexbadeashare/eeg-emotion-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/coralexbadeashare/eeg-emotion-decoder/actions/workflows/tests.yml)

Weights for **5 emotions** (Joy, Tenderness, Serenity, Nostalgia, Melancholy) from **Emotiv FLEX2** EEG,
for a window every few seconds. Only the EEG is needed. All of it runs on a plain laptop (CPU, no GPU):

| | |
|---|---|
| **Streamlit app** | upload a CSV and watch every step, window by window: `streamlit run streamlit_app.py` |
| **Live** (in the app, run locally) | follow a CSV that is being written, an LSL stream, or a replayed example |
| **Command line** | `python -m emotion_decoder recording.csv --out weights.csv` |
| **Python** | `from emotion_decoder import decode_csv` |

Pull requests are welcome from anyone; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```bash
git clone https://github.com/coralexbadeashare/eeg-emotion-decoder
cd eeg-emotion-decoder
pip install -r requirements.txt          # numpy, scipy, pandas, streamlit, plotly
```

## Use it as a library (API)

Install it into any Python project (3.9+); it needs only numpy, scipy and pandas:

```bash
pip install "git+https://github.com/coralexbadeashare/eeg-emotion-decoder"
pip install "emotion-decoder[live] @ git+https://github.com/coralexbadeashare/eeg-emotion-decoder"   # + pylsl, for LSL
```

Or, without installing: copy the `emotion_decoder/` folder next to your code (the model is inside it).

**Input format everywhere:** EEG in µV at 256 Hz, the 30 Emotiv FLEX2 channels in this order:
`emotion_decoder.preprocess.CHANNELS` = Cz, Fz, F7, F3, FC1, C3, FC5, FT9, T7, CP5, CP1, P3, P7, PO9, O1, Pz,
Oz, O2, PO10, P8, P4, CP2, CP6, T8, FT10, FC6, C4, FC2, F4, F8. Timestamps in seconds (any clock, the same for
EEG and markers). Weights are `{"Joy", "Tenderness", "Serenity", "Nostalgia", "Melancholy"}` → float, summing to 1.

### Live: `LiveDecoder`

For a stream you already read yourself (PyQt, LSL, a device SDK, …):

```python
from emotion_decoder.live import LiveDecoder

dec = LiveDecoder(every=3.0,       # a window every 3 s (ignored with use_markers=True)
                  calib=10,        # the first 10 windows only calibrate the session alignment
                  block=4,         # block_weights = mean of the last 4 windows' probabilities
                  use_markers=False,
                  channel_names=None)  # optional: your stream's channel names (any order, extras ok)

# every time new samples arrive:
dec.push(timestamps, eeg)          # timestamps: (n,) seconds; eeg: (channels, n) µV
for r in dec.step():               # windows that became complete (5 s after their start)
    r["start"]                     # window start, same clock as timestamps
    r["weights"]                   # this window
    r["block_weights"]             # averaged over the last `block` windows
    r["proba"]                     # numpy array, model order

dec.calibrated                     # True once the alignment is calibrated
dec.n_windows                      # usable windows seen so far (calibration ones included)
dec.results                        # every decoded window so far
dec.skipped                        # windows skipped (flat signal or dropped packets)
```

```python
# your stream carries its own channels, in its own order (labels as your stream/device gives them):
my_channels = ["COUNTER", "EEG.Fz", "EEG.Cz", "EEG.F7", ..., "EEG.F8", "CQ.Overall"]
dec = LiveDecoder(every=3.0, calib=10, block=4, channel_names=my_channels)
dec.push(timestamps, eeg)          # eeg: (len(my_channels), n), rows in the same order as my_channels
```

`channel_names` picks the 30 model channels **by name** from whatever your stream carries ("Cz" or
EmotivPRO's "EEG.Cz", in any order, extra channels such as counters or contact quality are ignored).
Without it, `push` expects exactly the 30 channels in `CHANNELS` order. For arrays you hold yourself:
`select_channels(eeg, names)` does the same and returns `(30, n)`.

Optional, windows at stimulus onsets instead of every few seconds: create it with `use_markers=True` and pass
the onset times: `dec.push(timestamps, eeg, markers=[t_image])`. With LSL, `pylsl.local_clock()` at the moment
the image appears is on the same clock as the EEG stream's timestamps.

Ready-made sources (each `.poll()` returns `(timestamps, eeg, marker_times)` for `dec.push(*...)`):

| | |
|---|---|
| `LslSource(markers=False)` | the first LSL stream of type EEG (picks the 30 channels by label), optionally a Markers stream |
| `CsvTail(path, from_end=True)` | a CSV that another program keeps appending to |
| `ReplaySource(recording, speed=1.0)` | a recording replayed as if it were arriving now |

```python
from emotion_decoder.live import LiveDecoder, LslSource
src, dec = LslSource(), LiveDecoder()
while True:
    dec.push(*src.poll())
    for r in dec.step():
        print(r["block_weights"])
```

### Integration checklist

| check | what happens if it is wrong |
|---|---|
| EEG in **microvolts** (Emotiv's unit) | values in volts trigger a warning; every window would look flat and be skipped |
| **256 Hz**, one timestamp per sample, in **seconds** | another rate triggers a warning; the windows would have the wrong length |
| shape **(channels, samples)** | (samples, channels) raises an error that says it looks transposed |
| the **30 FLEX2 channels**, by name or in `CHANNELS` order | a missing channel raises an error listing it; use `channel_names=` for other orders |
| markers on the **same clock** as the EEG timestamps (e.g. `pylsl.local_clock()` with an LSL EEG stream) | windows would start at the wrong moment |
| **one `LiveDecoder` per session** (headset placement) | after re-placing the headset, create a new one so the alignment recalibrates |
| the same FLEX2 **electrode montage** as in training | other positions: the channel names will not match and the model does not apply |

Handled for you: flat windows (headset not recording) and windows with more than 50 ms of data missing
(dropped packets, including a window that would start inside a gap) are skipped and counted in
`dec.skipped`; NaN values are replaced by 0 with a warning; memory stays bounded in long sessions (the
alignment uses the last 200 windows). Errors are `ValueError`s with a message that says what to fix;
warnings are Python `UserWarning`s, shown once.

### Recordings: `decode_csv`

```python
from emotion_decoder import decode_csv, label_accuracy

table, details = decode_csv("recording.csv", block=4)                  # a window every 3 s
table, details = decode_csv("recording.csv", sliding_step=2.0)         # every 2 s
table, details = decode_csv("recording.csv", use_markers=True)         # at MarkerIndex markers
table, details = decode_csv("recording.csv", "onsets.csv", block=4)    # at the onsets in a CSV
```

`table` (pandas): one row per window: `trial`, `onset_unix`, `t_s`, `flat` (skipped), `skipped` (why),
`w_<Emotion>` ×5, `top_emotion`, and with `block > 1` also `block<N>_<Emotion>` and `block<N>_top`; extra columns of
the onsets CSV are copied. `details` holds the cut windows (`details.trials`, (n, 30, 1408)) and probabilities.
`label_accuracy(table, emotions)` scores `top_emotion` against an `emotion` column, if there is one.
`decode_recording(read_recording(path), ...)` does the same from an already-read recording, and
`decode_trials(trials)` from windows you cut yourself.

### Low level: `EmotionModel`

```python
from emotion_decoder import EmotionModel

m = EmotionModel()                     # the shipped model
m.calibrate(session_windows)           # (n, 30, 1408) windows of ONE session: 0.5 s before to 5.0 s after their start
m.weights(window)                      # one window -> {"Joy": .., ...}
m.weights_block(last_windows)          # mean over several windows
m.proba(windows)                       # (n, 5), order m.emotions
```

## Try it on the examples

```bash
streamlit run streamlit_app.py
python -m emotion_decoder examples/example_1_clear.csv --block 4                    # EEG only
python -m emotion_decoder examples/example_1_clear.csv --onsets examples/example_1_clear_onsets.csv
```

The three files in [examples/](examples/) are **synthetic**: background noise plus, at every stimulus, the
model's own average response for a known emotion. They show the file format, and with their onsets files
they let you check that the pipeline recovers what was put in (100%, 90% and 78% of trials). They say
nothing about accuracy on real EEG.

## Your own recordings

**1. The recording**: the CSV that EmotivPRO exports (the band-passed export, `*.md.bp.csv`, when there
are two). The tool reads the `Timestamp` column and the 30 columns `EEG.Cz` … `EEG.F8`. Everything else in
the file is ignored. A plain CSV with columns `Cz, Fz, …, F8` (µV, 256 Hz) also works. One file = one session
(one headset placement).

**2. Run**

```bash
python -m emotion_decoder rec.csv --block 4 --out weights.csv        # a window every 3 s (--every N)
```

The output has one row per window: `w_Joy … w_Melancholy` (sum to 1), `top_emotion`, and with `--block N`
the same averaged over the last N windows (steadier). Windows where the headset was not recording (flat)
get no weights.

**Optional: stimulus onsets.** If you know when each stimulus appeared, windows can start there instead:
`--markers` uses EmotivPRO's `MarkerIndex` column, `--onsets onsets.csv` a CSV with a column `onset_unix`
(Unix seconds, same clock as `Timestamp`). Extra columns are copied to the output; a column `emotion` with
the true emotion names makes the tools report accuracy.

```csv
onset_unix,emotion
1778083701.512,Joy
1778083709.034,Melancholy
```

**From Python**

```python
from emotion_decoder import decode_csv, EmotionModel

table, details = decode_csv("rec.csv", block=4)              # EEG only, a window every 3 s
table, details = decode_csv("rec.csv", "onsets.csv", block=4)  # optional onsets

# or with trials you already cut: (n, 30, 1408), 256 Hz, from 0.5 s before to 5.0 s after each onset
m = EmotionModel().calibrate(session_trials)      # one session, no labels needed
m.weights(trial)                                  # {"Joy": 0.31, "Tenderness": 0.12, ...}
m.weights_block(last_four_trials)
```

## Art view

The **🎨 Art view** switch (top right, in both modes) shows a generative sculpture whose five coloured strata
follow the five emotion weights: on a recording it plays the windows one after another (its own Play
button), live it moves to every new window as it is decoded.

## How it works

1. **Cut** windows of 5.5 s (1408 samples × 30 channels): one every few seconds, or from 0.5 s before
   to 5.0 s after each stimulus when onsets are given.
2. **Align the session** (Euclidean alignment): R = average spatial covariance of the session's trials
   (1–45 Hz, 0–4 s); every trial X becomes R^-½ X. This removes differences in headset placement between
   sessions and needs no labels, only the session's own EEG.
3. **Evoked response**: filter 1–30 Hz, subtract the 0.5 s before the stimulus, keep 0–940 ms and average
   it into 20 bins of 47 ms: 30 channels × 20 bins = 600 numbers.
4. **Shrinkage LDA**: distance of those 600 numbers to each emotion's average response, measured against
   trial-to-trial noise → 5 probabilities.

The model (`emotion_decoder/models/emotion5_v2.json`) is plain numbers exported from the scikit-learn checkpoint in
[eeg-stimulus-type-baselines](https://github.com/coralexbadeashare/eeg-stimulus-type-baselines)
(`checkpoints/emotion5`). The tests check that this code gives the same weights as the original; on the
844 training-set trials the difference is exactly 0.

## How good is it

Trained on **one participant** (370 image trials, Emotiv FLEX2, 256 Hz, EmotivPRO band-passed export, no
other cleaning). Top emotion, chance 20%, measured with windows starting at image onsets:

| test | accuracy |
|---|---|
| recordings never seen in training (leave one recording out) | 26.3% (p = 0.0006) |
| held-out test recordings, untouched during development | 28.4% (p = 0.007) |
| average over a block of 2 / 8 trials | 28.6% / 33.4% |
| pictures never seen in training | 22.6% (p = 0.12, **not significant**) |
| videos, action images | chance |

Windows at other moments (the default) have not been measured yet. So: above chance, but weak; it may partly reflect the specific pictures; it works for images only; the raw
probabilities are over-confident. Use the weights as an **EEG-driven control signal** (for example for
generative art), not as a measurement of someone's emotion. Other people and other headsets are untested.

## Live

Run the app on the computer that records, choose **Live** in the sidebar, then the source:

- **A CSV that is being written**: paste its path (Windows paths work as they are). The app reads only the
  new rows, every second. Whatever writes the file must append rows continuously (`Timestamp`,
  `EEG.Cz` … `EEG.F8`).
- **LSL stream**: EmotivPRO can stream EEG to Lab Streaming Layer (needs `pip install -r requirements-live.txt`).
- **Simulation**: replays an example as if it were arriving now, to try everything without a headset.

The first windows (10 by default, about 35 s) only calibrate the session alignment; after that, the weights
update with every new window. Untested with a real headset so far: reports and pull requests welcome.
The same from a terminal: `python live_lsl.py` (LSL), and `tools/replay_csv_to_lsl.py` replays a CSV as a
fake LSL headset.

## Files

| | |
|---|---|
| `emotion_decoder/` | the package: `io` (CSV reading), `preprocess` (cutting), `model` (alignment, ERP, LDA), `pipeline`, `cli`, `live` |
| `streamlit_app.py` | the app |
| `live_lsl.py` | live decoding over LSL from a terminal |
| `emotion_decoder/models/emotion5_v2.json` | the model (installed with the package) |
| `examples/` | three synthetic recordings + onsets files |
| `static/linetree_studio.html` | the generative sculpture shown by the app's 🎨 Art view (fed through its own EEG-playback functions by `emotion_decoder/art.py`; the only change is a small responsive-layout CSS block) |
| `tools/` | model export, example generation, CSV → LSL replay |
| `tests/` | `pytest -q` |

## License

MIT
