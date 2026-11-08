"""EEG emotion decoder: 5 emotion weights from Emotiv FLEX2 EEG (alignment + ERP + shrinkage LDA)."""
from .model import EmotionModel
from .pipeline import decode_csv, decode_recording, decode_trials, label_accuracy
from .io import read_recording, read_onsets, read_onsets_table
from .preprocess import CHANNELS, select_channels, channel_index

__all__ = ["EmotionModel", "decode_csv", "decode_recording", "decode_trials", "label_accuracy",
           "read_recording", "read_onsets", "read_onsets_table",
           "CHANNELS", "select_channels", "channel_index"]
