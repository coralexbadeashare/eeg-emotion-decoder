"""EEG emotion decoder: upload an Emotiv recording, watch every step, get 5 emotion weights per window.

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
import streamlit.components.v1 as components

from emotion_decoder import EmotionModel, decode_recording, read_recording, read_onsets_table, label_accuracy
from emotion_decoder.io import RecordingError
from emotion_decoder import art
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


@st.cache_data(show_spinner="Cutting windows, aligning the session, computing weights ...", max_entries=8)
def run(key: str, _rec: dict, _onsets: pd.DataFrame | None, every: float, use_markers: bool, block: int, mode: str, sharp: float):
    table, d = decode_recording(_rec, onset_table=_onsets, sliding_step=every, use_markers=use_markers,
                                block=block, mode=mode, sharp=sharp)
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
st.sidebar.caption("Emotiv FLEX2 EEG → windows → session alignment → response shape → 5 emotion weights")

app_mode = st.sidebar.radio("Mode", ["Analyse a recording", "Live (run the app locally)"], horizontal=False)


def live_page():
    from emotion_decoder.live import LiveDecoder, CsvTail, LslSource, ReplaySource

    tl, tr_ = st.columns([5, 1])
    tl.title("EEG emotion decoder · live")
    art_on = tr_.toggle("🎨 Art view", key="art_live", help="Show the generative sculpture, driven live by the weights.")
    st.markdown("Weights for 5 emotions, updated every few seconds while the EEG arrives. Run the app on the "
                "computer that records (`streamlit run streamlit_app.py`) and pick where the EEG comes from.")
    src_kind = st.sidebar.radio("EEG comes from", ["A CSV that is being written", "LSL stream (EmotivPRO)",
                                                    "Simulation (replay an example)"])
    path, from_end, ex, speed = None, True, None, 1.0
    if src_kind == "A CSV that is being written":
        path = st.sidebar.text_input("Path of the CSV", placeholder=r"C:\Users\me\Documents\EmotivPRO\recording.csv",
                                     help="Windows paths work as they are. The program writing the file must add "
                                          "rows continuously (Timestamp + EEG.Cz ... EEG.F8).")
        from_end = st.sidebar.checkbox("Only new data (start at the end of the file)", value=True,
                                       help="Untick to replay the file from its beginning.")
    elif src_kind == "Simulation (replay an example)":
        ex = st.sidebar.selectbox("Example", list(EXAMPLES))
        speed = st.sidebar.select_slider("Speed", [1.0, 2.0, 4.0, 8.0], value=1.0)
    every = st.sidebar.slider("A window every (s)", 1.0, 6.0, 3.0, 0.5)
    calib = st.sidebar.slider("Calibration windows", 5, 40, 10,
                              help="The first windows only estimate the session alignment; weights start after them.")
    block = st.sidebar.slider("Average over the last N windows", 1, 8, 4)
    markers = st.sidebar.checkbox("Optional: windows at stimulus markers instead",
                                  help="CSV: the MarkerIndex column. LSL: a stream of type Markers. Simulation: the example's markers.")

    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("⏵ Start", type="primary", width="stretch"):
        try:
            if src_kind == "A CSV that is being written":
                if not path:
                    raise ValueError("enter the path of the CSV")
                source = CsvTail(path, from_end=from_end)
            elif src_kind == "LSL stream (EmotivPRO)":
                source = LslSource(markers=markers)
            else:
                source = ReplaySource(read_recording(ROOT / "examples" / f"{EXAMPLES[ex]}.csv"), speed)
            st.session_state.live = {"src": source, "dec": LiveDecoder(every, calib, block, markers), "on": True,
                                     "t0": time.time(), "err": None, "art_n": 0}
            art.clear_live()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not start: {e}")
    if c2.button("⏹ Stop", width="stretch") and "live" in st.session_state:
        st.session_state.live["on"] = False

    if art_on:
        components.html(art.page_html(live=True), height=900, scrolling=False)

    @st.fragment(run_every=1.0)
    def live_view():
        L = st.session_state.get("live")
        if not L:
            st.info("Choose the source in the sidebar and press Start.")
            return
        dec = L["dec"]
        if L["on"]:
            try:
                dec.push(*L["src"].poll()); dec.step(); L["err"] = None
            except Exception as e:  # noqa: BLE001
                L["err"] = str(e)
        if len(dec.results) != L.get("art_n", 0):                  # feed the sculpture
            r0 = dec.results[0]["start"] if dec.results else 0.0
            art.write_live([{"weights": r["block_weights"], "name": f"t {r['start'] - r0:.0f} s",
                             "kind": "" if dec.use_markers else "window"} for r in dec.results])
            L["art_n"] = len(dec.results)
        if L["err"]:
            st.warning(f"Waiting for data: {L['err']}")
        m = st.columns(4)
        m[0].metric("Status", "running" if L["on"] else "stopped")
        m[1].metric("EEG received", f"{dec.n_samples / SFREQ:.0f} s")
        m[2].metric("Calibration", f"{min(len(dec.calib_windows), dec.calib)} / {dec.calib}")
        m[3].metric("Windows decoded", len(dec.results))
        if not dec.n_samples and L["on"] and time.time() - L["t0"] > 5:
            st.warning("No EEG is arriving yet: is the recording running and the path/stream right?")
        if not dec.results:
            need = dec.calib * dec.every + 5
            st.info(f"Collecting EEG for the session alignment: the first weights appear after about {need:.0f} s.")
        else:
            r = dec.results[-1]
            wb = r["block_weights"]; top = max(wb, key=wb.get)
            left, right = st.columns([1, 2])
            with left:
                st.markdown(f"<div style='font-size:15px;color:#666'>now (last {dec.block} windows)</div>"
                            f"<div style='font-size:46px;font-weight:700;color:{EMO_COLORS[top]}'>{top}</div>",
                            unsafe_allow_html=True)
                fb = go.Figure(go.Bar(y=EMOS, x=[wb[e] for e in EMOS], orientation="h",
                                      marker_color=[EMO_COLORS[e] for e in EMOS], text=[f"{wb[e]:.0%}" for e in EMOS]))
                fb.add_vline(x=0.2, line=dict(dash="dot", color="#999"))
                fb.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1]),
                                 yaxis=dict(autorange="reversed"))
                st.plotly_chart(fb, width="stretch", key="live_bar")
            with right:
                hist = dec.results[-60:]
                t = [h["start"] - dec.results[0]["start"] for h in hist]
                fl = go.Figure()
                for e in EMOS:
                    fl.add_trace(go.Scatter(x=t, y=[h["block_weights"][e] for h in hist], mode="lines", name=e,
                                            line=dict(color=EMO_COLORS[e], width=2), stackgroup="w"))
                fl.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="s",
                                 yaxis=dict(range=[0, 1], title="weight"), legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fl, width="stretch", key="live_hist")
        ts, x = dec.recent_eeg(10.0)
        if len(ts):
            fe = go.Figure()
            tr, sp = stacked(x - np.median(x, axis=1, keepdims=True), VIEW, ts - ts[-1])
            for t_ in tr:
                fe.add_trace(t_)
            fe.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="last 10 s",
                             yaxis=dict(tickvals=[-j * sp for j in range(len(VIEW))], ticktext=VIEW))
            st.plotly_chart(fe, width="stretch", key="live_eeg")
        if dec.results:
            df = pd.DataFrame([{"t_s": round(h["start"] - dec.results[0]["start"], 2),
                                **{f"w_{k}": v for k, v in h["weights"].items()},
                                **{f"block_{k}": v for k, v in h["block_weights"].items()}} for h in dec.results])
            st.download_button("Download the weights so far (CSV)", df.to_csv(index=False).encode(),
                                "live_weights.csv", "text/csv", key=f"dl{len(df)}")

    live_view()


if app_mode.startswith("Live"):
    live_page()
    st.stop()

src = st.sidebar.radio("Recording", list(EXAMPLES) + ["Upload my CSV"])
if src == "Upload my CSV":
    up = st.sidebar.file_uploader("EmotivPRO CSV export (EEG)", type=["csv"])
    data = up.getvalue() if up else None
else:
    name = EXAMPLES[src]
    data = (ROOT / "examples" / f"{name}.csv").read_bytes()

st.sidebar.subheader("Windows")
every = st.sidebar.slider("A window every (s)", 1.0, 6.0, 3.0, 0.5,
                          help="Each window: 0.5 s before to 5.0 s after its start. Only the EEG is needed.")
onset_tbl, use_markers = None, False
if st.sidebar.checkbox("Optional: use stimulus onsets instead", value=False,
                       help="Place windows at the moments stimuli appeared: from markers in the CSV or an onsets CSV."):
    if src == "Upload my CSV":
        how = st.sidebar.radio("Onsets from", ["Markers in the CSV", "An onsets CSV"], horizontal=True)
        if how == "Markers in the CSV":
            use_markers = True
        else:
            oup = st.sidebar.file_uploader("Onsets CSV (column onset_unix; optional column emotion)", type=["csv"],
                                           help="Same clock as the Timestamp column of the EEG CSV.")
            if oup:
                try:
                    onset_tbl = read_onsets_table(oup)
                except Exception as e:  # noqa: BLE001
                    st.sidebar.error(f"Could not read the onsets CSV: {e}")
    else:
        onset_tbl = read_onsets_table(ROOT / "examples" / f"{name}_onsets.csv")
        st.sidebar.caption("Using the example's onsets file (it also has the true emotions).")

st.sidebar.subheader("Weights")
block = st.sidebar.slider("Average over the last N windows", 1, 8, 4,
                          help="Averaging makes the weights steadier (on stimulus-aligned real data: 1 window 26-28%, 8 windows 33%, chance 20%).")
mode, sharp = "model", 1.0
st.sidebar.markdown(f"---\n[Code, CLI and contributing]({REPO})")

# ------------------------------------------------------------------ header
tl, tr_ = st.columns([5, 1])
tl.title("EEG emotion decoder")
art_on = tr_.toggle("🎨 Art view", key="art_rec", help="Show the generative sculpture, played window by window.")
st.markdown(
    "From the EEG of an Emotiv FLEX2 recording, this app gives **weights for 5 emotions** "
    "(Joy, Tenderness, Serenity, Nostalgia, Melancholy) for a window every few seconds. Only the EEG CSV is "
    "needed. Four steps, shown below for each window: "
    "**① cut** 5.5 s of EEG · **② align** the session (removes differences in headset "
    "placement, no labels needed) · **③ measure the response shape**: the first "
    "940 ms of the window, as 30 channels × 20 time bins · **④ compare** it with the average response of each emotion "
    "(shrinkage LDA) → 5 probabilities.")

if data is None:
    st.info("Upload an EmotivPRO CSV export (EEG) in the sidebar, or pick an example.")
    st.stop()

key = hashlib.md5(data).hexdigest()
try:
    rec = load_recording(key, data)
except RecordingError as e:
    st.error(f"Could not read this file: {e}")
    st.stop()
okey = hashlib.md5(onset_tbl.to_csv().encode()).hexdigest() if onset_tbl is not None else "none"
try:
    table, trials, onsets, flat, proba, Rcov, W = run(f"{key}-{okey}", rec, onset_tbl, every, use_markers, block, mode, sharp)
except ValueError as e:
    st.warning(str(e))
    st.stop()

n = len(table)
ts0 = rec["ts"][0]
t_tr = onsets - ts0
wcols = [f"w_{e}" for e in EMOS]
bcols = [f"block{block}_{e}" for e in EMOS] if block > 1 else wcols
acc = label_accuracy(table, EMOS)

c = st.columns(5)
c[0].metric("Duration", f"{(rec['ts'][-1] - ts0) / 60:.1f} min")
c[1].metric("Windows", n)
c[2].metric("Skipped", int(flat.sum()), help="No signal (flat) or data missing inside the window (dropped packets); as in training, these get no weights.")
if acc:
    c[3].metric("Top emotion correct", f"{acc['accuracy']:.0%}", f"chance {acc['chance']:.0%}", delta_color="off")
    if "block_accuracy" in acc:
        c[4].metric(f"Block of {block} correct", f"{acc['block_accuracy']:.0%}")
else:
    c[3].metric("Session alignment from", f"{int((~flat).sum())} windows")
if (~flat).sum() < 10:
    st.warning("Fewer than 10 usable windows: the session alignment is estimated from very little data.")
if table.attrs.get("n_outside"):
    st.info(f"{table.attrs['n_outside']} onsets fall outside the recording and were skipped.")

# ------------------------------------------------------------------ art view
if art_on:
    rows = []
    lab_col = acc["label_column"] if acc else None
    for j in range(n):
        if flat[j]:
            continue
        rows.append({"weights": {e: float(table.loc[j, c]) for e, c in zip(EMOS, bcols)},
                     "label": str(table.loc[j, lab_col]) if lab_col else None,
                     "name": f"window {j + 1} · {t_tr[j]:.0f} s",
                     "kind": "window" if onset_tbl is None and not use_markers else ""})
    components.html(art.page_html(art.build_doc(rows, "recording")), height=900, scrolling=False)
    st.caption("The sculpture's five strata follow the five emotion weights" +
               (f" (averaged over the last {block} windows)" if block > 1 else "") +
               ". Use its Play button to step through the windows; switch the toggle off for the analysis.")
    st.stop()

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
if flat.any():
    wf.add_trace(go.Bar(x=table["trial"][flat], y=[1.0] * int(flat.sum()), name="skipped", marker_color="#d0d0d0",
                        hovertext=table["skipped"][flat], hoverinfo="text+x"))
wf.update_layout(barmode="stack", bargap=0, height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.12]),
                 xaxis_title="window", yaxis_title="weight", legend=dict(orientation="h", y=1.15))
st.plotly_chart(wf, width="stretch")
st.caption(("Weights averaged over the last %d windows. " % block if block > 1 else "Single-window weights. ")
           + ("Squares on top: the true emotion from the onsets file." if acc else "")
           + " Grey: skipped (no signal or data missing). Vertical lines above: the start of each window, coloured by its top emotion.")


# ------------------------------------------------------------------ trial player
@st.fragment
def player():
    st.subheader("Inside one window, step by step")
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
    b[3].slider("Window", 0, n - 1, key="trial", format="%d", label_visibility="collapsed")

    i = st.session_state.trial
    head = f"**Window {i + 1} / {n}** · starts at **{int(t_tr[i] // 60)}:{t_tr[i] % 60:04.1f}**"
    if acc:
        head += f" · true emotion: **{table.loc[i, acc['label_column']]}**"
    st.markdown(head + (f" · ⚠️ skipped ({table.loc[i, 'skipped']}): no prediction" if flat[i] else
                        f" · top emotion: **{table.loc[i, 'top_emotion']}**"))

    tt = (np.arange(N_SAMPLES) - ONSET) / SFREQ
    seg = trials[i]
    left, right = st.columns(2)
    with left:
        st.markdown("**① Cut.** 5.5 s of EEG: 0.5 s before the window start to 5.0 s after (1408 samples × 30 channels). "
                    "The model uses the shaded 0–940 ms.")
        f1 = go.Figure()
        tr, sp = stacked(seg - np.median(seg, axis=1, keepdims=True), VIEW, tt)
        for t_ in tr:
            f1.add_trace(t_)
        f1.add_vrect(x0=0, x1=0.9375, fillcolor="#3E8ED0", opacity=0.1, line_width=0)
        f1.add_vline(x=0, line=dict(color="#c0392b", width=1.5))
        f1.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="s from window start",
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
            st.info("This window is skipped (no signal, or data missing from dropped packets): no weights, "
                    "and it is left out of the alignment.")
        else:
            st.markdown("**③ Response shape.** Aligned window, filtered 1–30 Hz, minus its first 0.5 s mean; "
                        "0–940 ms averaged into 20 bins of 47 ms → 30 × 20 = 600 numbers.")
            erp = erp_filtered(W @ seg.astype(np.float64))
            F = erp[:, ONSET:ONSET + N_BINS * 12].reshape(30, N_BINS, 12).mean(2)
            bins_ms = (np.arange(N_BINS) + 0.5) * BIN_MS
            h3 = go.Figure(go.Heatmap(z=F, x=np.round(bins_ms), y=CHANNELS, colorscale="RdBu_r", zmid=0,
                                      colorbar=dict(thickness=8), hovertemplate="%{y} · %{x} ms: %{z:.2f}<extra></extra>"))
            h3.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ms after window start",
                             yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
            st.plotly_chart(h3, width="stretch", key="f3")

            st.markdown("**④ Compare with each emotion's average response shape** (posterior channels shown), "
                        "then shrinkage LDA → probabilities.")
            T = model.templates()
            pi = [CHANNELS.index(c) for c in POSTERIOR]
            f4 = go.Figure()
            for k, e in enumerate(EMOS):
                f4.add_trace(go.Scatter(x=bins_ms, y=T[k][pi].mean(0), mode="lines", name=e,
                                        line=dict(color=EMO_COLORS[e], width=1.5, dash="dot")))
            f4.add_trace(go.Scatter(x=bins_ms, y=F[pi].mean(0), mode="lines+markers", name="this window",
                                    line=dict(color="black", width=2.5)))
            f4.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ms after window start",
                             legend=dict(font=dict(size=9)))
            st.plotly_chart(f4, width="stretch", key="f4")

            p = table.loc[i, wcols].to_numpy(float)
            pb = table.loc[i, bcols].to_numpy(float)
            f5 = go.Figure()
            f5.add_trace(go.Bar(y=EMOS, x=p, orientation="h", name="this window",
                                marker_color=[EMO_COLORS[e] for e in EMOS], text=[f"{v:.0%}" for v in p]))
            if block > 1:
                f5.add_trace(go.Bar(y=EMOS, x=pb, orientation="h", name=f"last {block} windows",
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
st.subheader("Weights per window")
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
