"""EEG emotion decoder: upload an Emotiv recording, watch every step, get 5 emotion weights per trial.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from emotion_decoder import EmotionModel, decode_recording, read_recording, read_onsets_table, label_accuracy
from emotion_decoder.io import RecordingError
from emotion_decoder.model import session_covariance, erp_filtered, N_BINS, BIN_MS
from emotion_decoder.preprocess import CHANNELS, SFREQ, ONSET, N_SAMPLES

st.set_page_config(page_title="EEG emotion decoder", page_icon="🧠", layout="wide")

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Example 1 · clear (synthetic)": "example_1_clear",
    "Example 2 · moderate (synthetic)": "example_2_moderate",
    "Example 3 · hard, no markers, 2 dropouts (synthetic)": "example_3_hard_no_markers",
}
EMO_COLORS = {"Joy": "#E8A600", "Tenderness": "#E07AA0", "Serenity": "#3E8ED0",
              "Nostalgia": "#8A63C0", "Melancholy": "#4D5B6B"}
VIEW = ["Fz", "Cz", "Pz", "O1", "Oz", "O2"]
POSTERIOR = ["P3", "Pz", "P4", "O1", "Oz", "O2"]
REPO = "https://github.com/coralexbadeashare/eeg-emotion-decoder"


@st.cache_data(show_spinner="Reading the recording ...", max_entries=4)
def load_recording(key: str, _data: bytes) -> dict:
    return read_recording(io.BytesIO(_data))


@st.cache_data(show_spinner="Cutting trials, aligning the session, computing weights ...", max_entries=8)
def run(key: str, _rec: dict, _onsets: pd.DataFrame | None, sliding: float | None, block: int, mode: str, sharp: float):
    table, d = decode_recording(_rec, onset_table=_onsets, sliding_step=sliding, block=block, mode=mode, sharp=sharp)
    good = ~d.flat
    R = session_covariance(d.trials[good]) if good.any() else np.eye(30)
    W = d.model.W if d.model.W is not None else np.eye(30)
    return table, d.trials, d.onsets, d.flat, d.proba, R, W


def stacked(x, chans, t, sp=None, color="#444", width=0.9):
    sp = sp or 6.0 * float(np.median(np.std(x[[CHANNELS.index(c) for c in chans]], axis=-1)) or 1.0)
    traces = [go.Scattergl(x=t, y=x[CHANNELS.index(c)] - j * sp, mode="lines", line=dict(width=width, color=color),
                           showlegend=False, hovertemplate=f"{c}<extra></extra>") for j, c in enumerate(chans)]
    return traces, sp


# ------------------------------------------------------------------ sidebar
model = EmotionModel()
EMOS = model.emotions
st.sidebar.title("🧠 EEG emotion decoder")
st.sidebar.caption("Emotiv FLEX2 recording → trials → session alignment → evoked response → 5 emotion weights")

src = st.sidebar.radio("Recording", list(EXAMPLES) + ["Upload my CSV"])
onset_tbl = None
if src == "Upload my CSV":
    up = st.sidebar.file_uploader("EmotivPRO CSV export of one session", type=["csv"])
    data = up.getvalue() if up else None
    oup = st.sidebar.file_uploader("Stimulus onsets CSV (column onset_unix; optional column emotion)", type=["csv"],
                                   help="Needed when the recording has no markers. Same clock as the Timestamp column.")
    if oup:
        try:
            onset_tbl = read_onsets_table(oup)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read the onsets CSV: {e}")
else:
    name = EXAMPLES[src]
    data = (ROOT / "examples" / f"{name}.csv").read_bytes()
    use_truth = st.sidebar.checkbox("Use the example's onsets file (has the true emotions)", value=True)
    if use_truth:
        onset_tbl = read_onsets_table(ROOT / "examples" / f"{name}_onsets.csv")

st.sidebar.subheader("Weights")
block = st.sidebar.slider("Average over the last N trials", 1, 8, 4,
                          help="Blocks are more accurate on real data: 1 trial 26-28%, 2 trials 29%, 8 trials 33% (chance 20%).")
mode = st.sidebar.radio("Probabilities", ["model", "calibrated"], horizontal=True,
                        help="calibrated = divided by the temperature fitted on held-out data: honest, but almost uniform.")
sharp = st.sidebar.slider("Contrast (visual only)", 1.0, 5.0, 1.0, 0.5)
sliding = None
with st.sidebar.expander("Experimental"):
    if st.checkbox("No onsets? Use sliding windows", help="The model was trained on stimulus onsets; this is a rough continuous read-out."):
        sliding = st.slider("Window every (s)", 1.0, 6.0, 3.0, 0.5)
st.sidebar.markdown(f"---\n[Code, CLI and contributing]({REPO})")

# ------------------------------------------------------------------ header
st.title("EEG emotion decoder")
st.markdown(
    "For every stimulus in an Emotiv FLEX2 recording this app gives **weights for 5 emotions** "
    "(Joy, Tenderness, Serenity, Nostalgia, Melancholy). Four steps, shown below for each trial: "
    "**① cut** the 5.5 s around the stimulus · **② align** the session (removes differences in headset "
    "placement, no labels needed) · **③ measure the evoked response**: the brain's reply in the first "
    "940 ms, as 30 channels × 20 time bins · **④ compare** it with the average response of each emotion "
    "(shrinkage LDA) → 5 probabilities.")
with st.expander("How good is it? (read this)"):
    st.markdown("""
- Trained on **one participant**: 370 image trials, 30-channel Emotiv FLEX2, 256 Hz, EmotivPRO's band-passed
  export, no other cleaning.
- Top emotion on **recordings the model never saw: 26–28%** (chance 20%, p < 0.01). Averaging over 8 trials: 33%.
- On **pictures it never saw: 22.6%, not significant**; part of the signal may be the specific pictures.
- Only **image** stimuli: videos and action images score at chance.
- The probability values are over-confident; *calibrated* mode is honest and almost uniform.
- Use it as an **EEG-driven control signal** (e.g. for generative art), not as a measurement of emotion.
- The examples are **synthetic** (noise + the model's own average response for a known emotion), so the
  pipeline should recover their emotions far better than on real EEG. They show how it works, not how well.
""")

if data is None:
    st.info("Upload an EmotivPRO CSV export in the sidebar, or pick an example. Without markers in the file, "
            "also upload a CSV of stimulus onsets (column `onset_unix`).")
    st.stop()

key = hashlib.md5(data).hexdigest()
try:
    rec = load_recording(key, data)
except RecordingError as e:
    st.error(f"Could not read this file: {e}")
    st.stop()
okey = hashlib.md5(onset_tbl.to_csv().encode()).hexdigest() if onset_tbl is not None else "none"
try:
    table, trials, onsets, flat, proba, Rcov, W = run(f"{key}-{okey}", rec, onset_tbl, sliding, block, mode, sharp)
except ValueError as e:
    st.warning(str(e) + ". Tip: the sidebar's Experimental section has sliding windows.")
    st.stop()

n = len(table)
ts0 = rec["ts"][0]
t_tr = onsets - ts0
wcols = [f"w_{e}" for e in EMOS]
bcols = [f"block{block}_{e}" for e in EMOS] if block > 1 else wcols
acc = label_accuracy(table, EMOS)

c = st.columns(5)
c[0].metric("Duration", f"{(rec['ts'][-1] - ts0) / 60:.1f} min")
c[1].metric("Trials", n)
c[2].metric("Flat (no signal)", int(flat.sum()))
if acc:
    c[3].metric("Top emotion correct", f"{acc['accuracy']:.0%}", f"chance {acc['chance']:.0%}", delta_color="off")
    if "block_accuracy" in acc:
        c[4].metric(f"Block of {block} correct", f"{acc['block_accuracy']:.0%}")
else:
    c[3].metric("Session alignment from", f"{int((~flat).sum())} trials")
if (~flat).sum() < 10:
    st.warning("Fewer than 10 usable trials: the session alignment is estimated from very little data.")
if table.attrs.get("n_outside"):
    st.info(f"{table.attrs['n_outside']} onsets fall outside the recording and were skipped.")

# ------------------------------------------------------------------ overview
st.subheader("The whole session")
view = st.multiselect("Channels", CHANNELS, default=VIEW)
x = rec["eeg"] - np.median(rec["eeg"], axis=1, keepdims=True)
stride = max(1, x.shape[1] // 6000)
fig = go.Figure()
if view:
    tr, sp = stacked(x[:, ::stride], view, rec["ts"][::stride] - ts0, width=0.6, color="#666")
    for t_ in tr:
        fig.add_trace(t_)
    fig.update_yaxes(tickvals=[-j * sp for j in range(len(view))], ticktext=view)
for j in range(n):
    col = "#bbb" if flat[j] else EMO_COLORS[table.loc[j, "top_emotion"]]
    fig.add_vline(x=t_tr[j], line=dict(color=col, width=2))
fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time (s)")
st.plotly_chart(fig, width="stretch")

wf = go.Figure()
W_show = table[bcols].to_numpy(float)
for k, e in enumerate(EMOS):
    wf.add_trace(go.Bar(x=table["trial"], y=W_show[:, k], name=e, marker_color=EMO_COLORS[e]))
if acc:
    lab = table[acc["label_column"]].astype(str).str.capitalize()
    wf.add_trace(go.Scatter(x=table["trial"], y=[1.06] * n, mode="markers", name="true emotion", showlegend=False,
                            marker=dict(symbol="square", size=11, color=[EMO_COLORS.get(v, "#999") for v in lab]),
                            hovertext=lab, hoverinfo="text"))
wf.update_layout(barmode="stack", height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.12]),
                 xaxis_title="trial", yaxis_title="weight", legend=dict(orientation="h", y=1.15))
st.plotly_chart(wf, width="stretch")
st.caption(("Weights averaged over the last %d trials. " % block if block > 1 else "Single-trial weights. ")
           + ("Squares on top: the true emotion from the onsets file." if acc else "")
           + " Vertical lines above: each stimulus, coloured by its top emotion (grey = flat, no prediction).")


# ------------------------------------------------------------------ trial player
@st.fragment
def player():
    st.subheader("Inside one trial, step by step")
    if "trial" not in st.session_state or st.session_state.trial >= n:
        st.session_state.trial = 0
    if st.session_state.get("playing"):
        st.session_state.trial = (st.session_state.trial + 1) % n

    def nudge(d):
        st.session_state.trial = int(np.clip(st.session_state.trial + d, 0, n - 1))

    b = st.columns([1, 1, 1.4, 7])
    b[0].button("◀", on_click=nudge, args=(-1,), width="stretch")
    b[1].button("▶", on_click=nudge, args=(1,), width="stretch")
    playing = st.session_state.get("playing", False)
    if b[2].button("⏸ Pause" if playing else "⏵ Play", width="stretch"):
        st.session_state.playing = not playing
        st.rerun(scope="fragment")
    b[3].slider("Trial", 0, n - 1, key="trial", format="%d", label_visibility="collapsed")

    i = st.session_state.trial
    head = f"**Trial {i + 1} / {n}** · stimulus at **{int(t_tr[i] // 60)}:{t_tr[i] % 60:04.1f}**"
    if acc:
        head += f" · true emotion: **{table.loc[i, acc['label_column']]}**"
    st.markdown(head + (" · ⚠️ flat: the headset was not recording, no prediction" if flat[i] else
                        f" · top emotion: **{table.loc[i, 'top_emotion']}**"))

    tt = (np.arange(N_SAMPLES) - ONSET) / SFREQ
    seg = trials[i]
    left, right = st.columns(2)
    with left:
        st.markdown("**① Cut.** 0.5 s before the stimulus to 5.0 s after (1408 samples × 30 channels). "
                    "The model uses the shaded 0–940 ms.")
        f1 = go.Figure()
        tr, sp = stacked(seg - np.median(seg, axis=1, keepdims=True), VIEW, tt)
        for t_ in tr:
            f1.add_trace(t_)
        f1.add_vrect(x0=0, x1=0.9375, fillcolor="#3E8ED0", opacity=0.1, line_width=0)
        f1.add_vline(x=0, line=dict(color="#c0392b", width=1.5))
        f1.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="s from stimulus",
                         yaxis=dict(tickvals=[-j * sp for j in range(len(VIEW))], ticktext=VIEW))
        st.plotly_chart(f1, width="stretch", key="f1")

        st.markdown("**② Align the session.** The spatial covariance of the whole session (left) is "
                    "undone, R^-½ (right): every session ends up looking alike, whatever the headset placement.")
        a1, a2 = st.columns(2)
        Rn = Rcov / np.sqrt(np.outer(np.diag(Rcov), np.diag(Rcov)))
        A = W @ Rcov @ W.T
        for col, M, title in ((a1, Rn, "before (correlation)"), (a2, A, "after alignment")):
            h = go.Figure(go.Heatmap(z=M, x=CHANNELS, y=CHANNELS, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                     showscale=False, hovertemplate="%{y}–%{x}: %{z:.2f}<extra></extra>"))
            h.update_layout(height=230, margin=dict(l=0, r=0, t=24, b=0), title=dict(text=title, font=dict(size=12)),
                            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed"))
            col.plotly_chart(h, width="stretch", key=f"a_{title}")

    with right:
        if flat[i]:
            st.info("This trial is flat on every channel, so it gets no weights and is left out of the alignment.")
        else:
            st.markdown("**③ Evoked response.** Aligned trial, filtered 1–30 Hz, minus its pre-stimulus mean; "
                        "0–940 ms averaged into 20 bins of 47 ms → 30 × 20 = 600 numbers.")
            erp = erp_filtered(W @ seg.astype(np.float64))
            F = erp[:, ONSET:ONSET + N_BINS * 12].reshape(30, N_BINS, 12).mean(2)
            bins_ms = (np.arange(N_BINS) + 0.5) * BIN_MS
            h3 = go.Figure(go.Heatmap(z=F, x=np.round(bins_ms), y=CHANNELS, colorscale="RdBu_r", zmid=0,
                                      colorbar=dict(thickness=8), hovertemplate="%{y} · %{x} ms: %{z:.2f}<extra></extra>"))
            h3.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ms after stimulus",
                             yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
            st.plotly_chart(h3, width="stretch", key="f3")

            st.markdown("**④ Compare with each emotion's average response** (posterior channels shown), "
                        "then shrinkage LDA → probabilities.")
            T = model.templates()
            pi = [CHANNELS.index(c) for c in POSTERIOR]
            f4 = go.Figure()
            for k, e in enumerate(EMOS):
                f4.add_trace(go.Scatter(x=bins_ms, y=T[k][pi].mean(0), mode="lines", name=e,
                                        line=dict(color=EMO_COLORS[e], width=1.5, dash="dot")))
            f4.add_trace(go.Scatter(x=bins_ms, y=F[pi].mean(0), mode="lines+markers", name="this trial",
                                    line=dict(color="black", width=2.5)))
            f4.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ms after stimulus",
                             legend=dict(font=dict(size=9)))
            st.plotly_chart(f4, width="stretch", key="f4")

            p = table.loc[i, wcols].to_numpy(float)
            pb = table.loc[i, bcols].to_numpy(float)
            f5 = go.Figure()
            f5.add_trace(go.Bar(y=EMOS, x=p, orientation="h", name="this trial",
                                marker_color=[EMO_COLORS[e] for e in EMOS], text=[f"{v:.0%}" for v in p]))
            if block > 1:
                f5.add_trace(go.Bar(y=EMOS, x=pb, orientation="h", name=f"last {block} trials",
                                    marker_color="rgba(0,0,0,0.25)", text=[f"{v:.0%}" for v in pb]))
            f5.add_vline(x=0.2, line=dict(dash="dot", color="#999"))
            f5.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                             xaxis=dict(range=[0, 1], title="weight"), yaxis=dict(autorange="reversed"),
                             legend=dict(orientation="h", y=1.15))
            st.plotly_chart(f5, width="stretch", key="f5")
            st.caption("Dotted line: chance (20%).")

    if st.session_state.get("playing"):
        time.sleep(1.0)
        st.rerun(scope="fragment")


player()

# ------------------------------------------------------------------ table + downloads
st.subheader("Weights per trial")
st.download_button("Download weights (CSV)", table.to_csv(index=False).encode(), "emotion_weights.csv", "text/csv")
st.dataframe(table, width="stretch", height=260, hide_index=True)
with st.expander("Use it from code or the command line"):
    st.code("""pip install -r requirements.txt

# command line
python -m emotion_decoder my_recording.csv --onsets my_onsets.csv --block 4 --out weights.csv

# Python
from emotion_decoder import decode_csv
table, details = decode_csv("my_recording.csv", "my_onsets.csv", block=4)""", language="bash")
with st.expander("Live from the headset (work in progress)"):
    st.markdown(f"""`live_lsl.py` reads EEG and stimulus markers from **Lab Streaming Layer** (EmotivPRO can stream
to LSL) and prints weights a few seconds after each stimulus. It is untested with a real headset: see the
README in [the repository]({REPO}). `tools/replay_csv_to_lsl.py` replays a CSV as a fake headset for testing.""")
