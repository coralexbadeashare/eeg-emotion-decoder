# Examples (synthetic, not brain data)

Each example is one 2-minute session in EmotivPRO's CSV format, with 20 image stimuli in 5 blocks of 4
(one emotion per block), and an `*_onsets.csv` with `onset_unix` and the true `emotion`.

The EEG is synthetic: coloured noise, a 10 Hz rhythm, 50 Hz line noise and an Emotiv-like offset, plus,
at every stimulus, the model's own average response for that trial's emotion. `gain` sets how distinct
the emotions are. Real EEG is far harder (26-28% on new recordings, chance 20%).

| file | gain | markers in CSV | top emotion correct |
|---|---|---|---|
| `example_1_clear.csv` | 2.0 | yes | 100% |
| `example_2_moderate.csv` | 1.0 | yes | 90% |
| `example_3_hard_no_markers.csv` | 0.6 | no: use the onsets file | 78% (2 trials flat: headset dropout) |

Regenerate with `python tools/make_examples.py`.
