"""Run: pytest -q   (these keep the decoder identical to the trained model)"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, sosfiltfilt

from emotion_decoder import EmotionModel, decode_csv, decode_trials, read_recording, label_accuracy
from emotion_decoder.cli import main as cli_main
from emotion_decoder.io import RecordingError
from emotion_decoder.preprocess import CHANNELS, cut_trials
from emotion_decoder.synthetic import make_session

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


# reference: emotion_weights.py from eeg-stimulus-type-baselines, the code the model shipped with
def _ref_whitener(epochs):
    sos = butter(4, [1, 45], btype="band", fs=256, output="sos")
    Cs = []
    for e in epochs:
        x = sosfiltfilt(sos, np.asarray(e, np.float64), axis=1)[:, 128:1152]; x = x - x.mean(1, keepdims=True)
        Cs.append(x @ x.T / x.shape[1])
    w, V = np.linalg.eigh(np.mean(Cs, 0))
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def _ref_erp(e):
    sos = butter(4, [1, 30], btype="band", fs=256, output="sos")
    x = sosfiltfilt(sos, np.asarray(e, np.float64), axis=1); x = x - x[:, :128].mean(1, keepdims=True)
    return x[:, 128:368].reshape(30, 20, 12).mean(2).reshape(-1)


@pytest.fixture(scope="module")
def session():
    rec, on = make_session(["Joy", "Serenity", "Melancholy", "Tenderness", "Nostalgia"] * 2, gain=1.0, seed=11)
    trials, _ = cut_trials(rec["ts"], rec["eeg"], on.onset_unix)
    return trials, on


def test_model_matches_sklearn_self_check():
    d = json.loads((ROOT / "models" / "emotion5_v2.json").read_text())
    m = EmotionModel()
    X = np.random.default_rng(d["self_check"]["seed"]).standard_normal((d["self_check"]["n"], 600))
    p = m.proba_from_features(X * m.scale + m.mean)          # self-check inputs are in scaled units
    np.testing.assert_allclose(p, d["self_check"]["proba"], atol=1e-10)


def test_alignment_and_features_match_reference(session):
    trials, _ = session
    m = EmotionModel().calibrate(trials)
    W = _ref_whitener(trials)
    np.testing.assert_allclose(m.W, W, atol=1e-10)
    np.testing.assert_allclose(m.features(trials), np.stack([_ref_erp(W @ e) for e in trials]), atol=1e-9)


def test_weights_sum_to_one_and_modes(session):
    trials, _ = session
    m = EmotionModel().calibrate(trials)
    for mode in ("model", "calibrated"):
        w = m.weights(trials[0], mode=mode)
        assert list(w) == ["Joy", "Tenderness", "Serenity", "Nostalgia", "Melancholy"]
        assert sum(w.values()) == pytest.approx(1, abs=1e-3)
    assert max(m.weights(trials[0], mode="calibrated").values()) <= max(m.weights(trials[0]).values()) + 1e-9


def test_block_is_mean_of_probabilities(session):
    trials, _ = session
    table, d = decode_trials(trials, block=3)
    w = d.model.to_weights(d.proba[2:5].mean(0))
    assert table.loc[4, "block3_Joy"] == pytest.approx(w["Joy"], abs=1e-4)


def test_uncalibrated_model_refuses(session):
    with pytest.raises(RuntimeError):
        EmotionModel().proba(session[0][:1])


@pytest.mark.parametrize("name,min_acc", [("example_1_clear", 0.95), ("example_2_moderate", 0.8)])
def test_examples_positive_control(name, min_acc):
    table, _ = decode_csv(EX / f"{name}.csv", EX / f"{name}_onsets.csv", block=4)
    acc = label_accuracy(table, EmotionModel().emotions)
    assert acc["n"] == 20 and acc["accuracy"] >= min_acc


def test_markers_equal_onsets_file():
    a, _ = decode_csv(EX / "example_1_clear.csv", use_markers=True)                  # markers in the CSV
    b, _ = decode_csv(EX / "example_1_clear.csv", EX / "example_1_clear_onsets.csv")
    np.testing.assert_allclose(a["w_Joy"], b["w_Joy"], atol=1e-3)


def test_example_3_needs_onsets_and_flags_dropouts():
    with pytest.raises(ValueError, match="no markers"):
        decode_csv(EX / "example_3_hard_no_markers.csv", use_markers=True)
    table, _ = decode_csv(EX / "example_3_hard_no_markers.csv", EX / "example_3_hard_no_markers_onsets.csv")
    assert table["flat"].sum() == 2 and table.loc[table.flat, "top_emotion"].isna().all()


def test_cli(tmp_path, capsys):
    out = tmp_path / "w.csv"
    cli_main([str(EX / "example_2_moderate.csv"), "--onsets", str(EX / "example_2_moderate_onsets.csv"),
              "--block", "4", "--out", str(out)])
    t = pd.read_csv(out)
    assert len(t) == 20 and {"w_Joy", "block4_top", "emotion"} <= set(t.columns)
    assert "top emotion vs `emotion`" in capsys.readouterr().out


def test_npy_input(tmp_path, session):
    p = tmp_path / "trials.npy"; np.save(p, session[0])
    cli_main([str(p), "--out", str(tmp_path / "w.csv")])
    assert len(pd.read_csv(tmp_path / "w.csv")) == len(session[0])


def test_reader_plain_csv_and_missing_channel():
    rows = "\n".join(",".join(["1.0"] * 30) for _ in range(10))
    rec = read_recording(io.BytesIO(f"{','.join(CHANNELS)}\n{rows}\n".encode()))
    assert rec["eeg"].shape == (30, 10)
    with pytest.raises(RecordingError, match="F8"):
        read_recording(io.BytesIO((",".join(["Timestamp"] + [f"EEG.{c}" for c in CHANNELS[:-1]]) + "\n").encode()))


def test_default_is_eeg_only_windows():
    table, d = decode_csv(EX / "example_3_hard_no_markers.csv")                      # no onsets, no markers
    rec = d.recording
    expected = len(np.arange(rec["ts"][0] + 0.5, rec["ts"][-1] - 5.0, 3.0))
    assert len(table) == expected and np.allclose(np.diff(d.onsets), 3.0, atol=1e-6)
    t2, _ = decode_csv(EX / "example_3_hard_no_markers.csv", sliding_step=2.0)
    assert len(t2) > len(table)


def test_cli_default_eeg_only(tmp_path):
    out = tmp_path / "w.csv"
    cli_main([str(EX / "example_1_clear.csv"), "--out", str(out)])
    t = pd.read_csv(out)
    assert len(t) > 30 and "emotion" not in t


def _grow(src_path, dst_path, n_parts):
    """Yield after writing each part of src to dst, like a recorder appending rows."""
    lines = Path(src_path).read_text().splitlines(keepends=True)
    head, body = lines[:2], lines[2:]
    step = len(body) // n_parts + 1
    with open(dst_path, "w") as f:
        f.writelines(head); f.flush(); yield
        for i in range(0, len(body), step):
            chunk = "".join(body[i:i + step])
            cut = len(chunk) // 2 if i + step < len(body) else len(chunk)   # also split a line in two
            f.write(chunk[:cut]); f.flush(); yield
            f.write(chunk[cut:]); f.flush(); yield


def test_live_csv_tail_markers_recovers_emotions(tmp_path):
    from emotion_decoder.live import LiveDecoder, CsvTail
    dst = tmp_path / "growing.csv"
    dec, tail = LiveDecoder(calib=8, block=1, use_markers=True), None
    for _ in _grow(EX / "example_1_clear.csv", dst, 25):
        tail = tail or CsvTail(str(dst))
        dec.push(*tail.poll()); dec.step()
    truth = pd.read_csv(EX / "example_1_clear_onsets.csv")["emotion"].tolist()
    got = [max(r["weights"], key=r["weights"].get) for r in dec.results]
    assert len(got) == 12 and np.mean([g == t for g, t in zip(got, truth[8:])]) >= 0.9
    assert tail.rows == 124 * 256 or abs(tail.rows - len(read_recording(EX / "example_1_clear.csv")["ts"])) == 0


def test_live_default_windows_every_3s(tmp_path):
    from emotion_decoder.live import LiveDecoder, CsvTail
    dst = tmp_path / "growing.csv"
    dec, tail = LiveDecoder(every=3.0, calib=5), None
    for _ in _grow(EX / "example_2_moderate.csv", dst, 10):
        tail = tail or CsvTail(str(dst))
        dec.push(*tail.poll()); dec.step()
    starts = [r["start"] for r in dec.results]
    assert len(dec.calib_windows) >= 35 and len(starts) == len(dec.calib_windows) - 5
    assert np.allclose(np.diff(starts), 3.0)


def test_windows_with_dropped_packets_are_skipped():
    from emotion_decoder import decode_recording
    rec = read_recording(EX / "example_2_moderate.csv")
    keep = np.ones(len(rec["ts"]), bool); keep[int(30.0 * 256):int(30.4 * 256)] = False      # 0.4 s lost
    rec2 = dict(rec, ts=rec["ts"][keep], eeg=rec["eeg"][:, keep])
    table, _ = decode_recording(rec2)
    hit = table[table.skipped.str.contains("missing")]
    assert 1 <= len(hit) <= 3 and hit.top_emotion.isna().all()
    assert (table.t_s[hit.index] > 23).all() and (table.t_s[hit.index] < 31).all()


def test_art_doc_reorders_to_the_sculpture_strata():
    from emotion_decoder import art
    w = {"Joy": 0.1, "Tenderness": 0.2, "Serenity": 0.3, "Nostalgia": 0.15, "Melancholy": 0.25}
    doc = art.build_doc([{"weights": w, "label": "serenity"}])
    t = doc["sessions"]["recording"][0]
    assert doc["classes"] == ["01_JOY", "02_TENDERNESS", "03_NOSTALGIA", "04_SERENITY", "05_MELANCHOLY"]
    assert t["proba"] == [0.1, 0.2, 0.15, 0.3, 0.25] and t["predicted"] == "04_SERENITY" and t["emotion"] == "04_SERENITY"
    html = art.page_html(doc)
    page = art.PAGE.read_text(encoding="utf-8"); i = page.rfind("</body>")
    assert html.count("Appended by the EEG emotion decoder") == 1
    assert html.startswith(page[:i]) and html.endswith(page[i:])                       # page itself unchanged
