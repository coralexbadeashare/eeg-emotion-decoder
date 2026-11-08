"""EEG emotion decoder: upload an Emotiv recording or follow it live, watch every step, get 5 emotion weights.

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
from emotion_decoder import art
from emotion_decoder.io import RecordingError
from emotion_decoder.model import session_covariance, erp_filtered, N_BINS, BIN_MS
from emotion_decoder.preprocess import CHANNELS, SFREQ, ONSET, N_SAMPLES

st.set_page_config(page_title="EEG emotion decoder", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/coralexbadeashare/eeg-emotion-decoder"
EXAMPLES = {
    "Example 1 · clear": "example_1_clear",
    "Example 2 · moderate": "example_2_moderate",
    "Example 3 · hard, no markers, 2 dropouts": "example_3_hard_no_markers",
}
# The sculpture's palette, a shade deeper so it reads on white. Strata order, bottom to top.
ORDER = ["Joy", "Tenderness", "Nostalgia", "Serenity", "Melancholy"]
EMO_COLORS = {"Joy": "#F0C232", "Tenderness": "#E795D3", "Nostalgia": "#2A7276",
              "Serenity": "#5DBFB0", "Melancholy": "#78BEE4"}
SKIP_COLOR = "#E3E5EA"
INK, INK2, MUTE, LINE = "#111528", "#4F5566", "#9AA0AD", "#EBECEF"
FONT = '-apple-system, BlinkMacSystemFont, Inter, "Segoe UI", "Helvetica Neue", sans-serif'
DIVERGING = [[0, "#3B6E8F"], [0.5, "#F7F7F8"], [1, "#C8553D"]]
VIEW = ["Fz", "Cz", "Pz", "O1", "Oz", "O2"]
POSTERIOR = ["P3", "Pz", "P4", "O1", "Oz", "O2"]
PLOT_CFG = {"displayModeBar": False, "responsive": True}

# ------------------------------------------------------------------ style
st.markdown(f"""
<style>
  :root {{ --ink:{INK}; --ink2:{INK2}; --mute:{MUTE}; --line:{LINE}; --soft:#F6F7F9; }}
  .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3,
  .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"], .stApp button p,
  .stApp [data-baseweb="select"] {{ font-family: {FONT}; }}
  .block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1480px; }}
  h1 {{ font-weight: 650 !important; letter-spacing: -0.025em; font-size: 2.05rem !important; margin-bottom: 0 !important; }}

  /* sidebar: collapsible, the collapse button always visible */
  section[data-testid="stSidebar"] {{ background: #F8F9FB; border-right: 1px solid var(--line); }}
  [data-testid="stSidebarCollapseButton"] {{ display: flex !important; visibility: visible !important; opacity: 1 !important; }}
  [data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] {{
      border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
  .side-brand {{ display:flex; align-items:center; gap:10px; margin: 0.2rem 0 0.2rem; }}
  .side-brand .dot {{ width:28px; height:28px; border-radius:50%; background:var(--ink); color:#fff; display:grid;
                      place-items:center; font-size:15px; flex:none; }}
  .side-brand b {{ font-size: 15px; font-weight: 650; color: var(--ink); letter-spacing: -0.01em; }}
  .side-brand small {{ display:block; color: var(--mute); font-size: 11.5px; margin-top: 1px; }}
  .side-label {{ font: 600 10.5px/1 {FONT}; letter-spacing: .08em; text-transform: uppercase; color: var(--mute);
                 margin: 1.15rem 0 0.45rem; }}
  .side-foot {{ margin-top: 1.4rem; padding-top: .8rem; border-top: 1px solid var(--line); font-size: 12px; color: var(--mute); }}
  .side-foot a {{ color: var(--ink2); text-decoration: none; }}

  /* header */
  .eyebrow {{ font: 600 11px/1 {FONT}; letter-spacing: .1em; text-transform: uppercase; color: var(--mute); margin-bottom: .45rem; }}
  .subtitle {{ color: var(--ink2); font-size: 14.5px; margin: .35rem 0 0; max-width: 860px; }}

  /* pipeline steps */
  .steps {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 1.1rem 0 .4rem; }}
  .step {{ border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px; background: #fff; }}
  .step .n {{ display:inline-grid; place-items:center; width:20px; height:20px; border-radius:50%; background:var(--ink);
              color:#fff; font-size:11px; font-weight:600; margin-right:7px; }}
  .step b {{ font-size: 13.5px; color: var(--ink); }}
  .step p {{ margin: 5px 0 0; font-size: 12.3px; color: var(--ink2); line-height: 1.4; }}
  @media (max-width: 900px) {{ .steps {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}

  /* section titles */
  .sec {{ margin: 1.9rem 0 .55rem; }}
  .sec h3 {{ font-size: 1.08rem !important; font-weight: 650 !important; margin: 0 !important; padding: 0 !important; letter-spacing: -0.01em; }}
  .sec p {{ margin: .15rem 0 0; color: var(--mute); font-size: 12.8px; }}

  /* metric cards */
  [data-testid="stMetric"] {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding: 12px 15px 10px; }}
  [data-testid="stMetricLabel"] p {{ font-size: 12px !important; color: var(--mute) !important; font-weight: 500; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem !important; font-weight: 600; color: var(--ink); letter-spacing: -0.02em; }}

  /* bordered containers = cards */
  [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px !important; border-color: var(--line) !important; }}
  .card-h {{ display:flex; align-items:center; gap:8px; margin-bottom: 2px; }}
  .card-h .n {{ display:inline-grid; place-items:center; width:20px; height:20px; border-radius:50%; background:var(--ink);
                color:#fff; font-size:11px; font-weight:600; flex:none; }}
  .card-h b {{ font-size: 13.5px; color: var(--ink); }}
  .stApp p.card-p {{ margin: 0 0 4px 28px !important; font-size: 12.3px !important; color: var(--ink2); line-height: 1.4; }}
  .stApp p.subtitle {{ font-size: 14.5px !important; }}

  /* emotion chips and badges */
  .chips {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size: 12.3px; color: var(--ink2); }}
  .chips i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
  .badge {{ display:inline-flex; align-items:center; gap:7px; padding: 3px 11px 3px 9px; border-radius: 999px;
            font-weight: 600; font-size: 13px; color: var(--ink); background: #fff; border: 1px solid var(--line); }}
  .badge i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .now {{ display:flex; flex-wrap:wrap; align-items:center; gap: 8px 12px; font-size: 13.5px; color: var(--ink2);
          padding: 10px 14px; border-radius: 12px; background: var(--soft); margin: .5rem 0 .8rem; }}
  .now b {{ color: var(--ink); }}
  .now .sep {{ color: var(--mute2, #C3C7CF); }}
  .now .ok {{ color:#2E7D5B; font-weight:600; }} .now .no {{ color:#C8553D; font-weight:600; }}
  .big-emo {{ font-size: 46px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; margin: 4px 0 10px; }}
  .muted {{ color: var(--mute); font-size: 12.5px; }}

  /* buttons, toggles */
  .stButton button, .stDownloadButton button {{ border-radius: 9px; font-weight: 500; }}
  div[data-testid="stCaptionContainer"] p {{ color: var(--mute); }}
  .foot {{ margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--mute); font-size: 12px; }}
  .foot a {{ color: var(--ink2); }}
</style>
""", unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="sec"><h3>{title}</h3>{f"<p>{sub}</p>" if sub else ""}</div>', unsafe_allow_html=True)


def card_head(n: str, title: str, text: str):
    st.markdown(f'<div class="card-h"><span class="n">{n}</span><b>{title}</b></div><p class="card-p">{text}</p>',
                unsafe_allow_html=True)


def chips() -> str:
    return '<div class="chips">' + "".join(f'<span><i style="background:{EMO_COLORS[e]}"></i>{e}</span>'
                                           for e in ORDER) + "</div>"


def badge(emo: str) -> str:
    return f'<span class="badge"><i style="background:{EMO_COLORS.get(emo, MUTE)}"></i>{emo}</span>'


def style(fig: go.Figure, height: int, legend: bool = False, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=11, color=INK2),
                      hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family=FONT, size=12, color=INK)),
                      showlegend=legend, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)),
                      **kw)
    fig.update_xaxes(gridcolor="#F0F1F4", zeroline=False, linecolor=LINE, ticks="", title_font=dict(size=11, color=MUTE))
    fig.update_yaxes(gridcolor="#F0F1F4", zeroline=False, linecolor=LINE, ticks="", title_font=dict(size=11, color=MUTE))
    return fig


def embed(html: str, height: int = 950):
    """The sculpture page, same-origin (the live view fetches /app/static/live_emotions.json)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                            # Streamlit < 1.52
        components.html(html, height=height, scrolling=False)


def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)


def stacked(x, chans, t, sp=None, color=INK2, width=0.8):
    sp = sp or 6.0 * float(np.median(np.std(x[[CHANNELS.index(c) for c in chans]], axis=-1)) or 1.0)
    traces = [go.Scattergl(x=t, y=x[CHANNELS.index(c)] - j * sp, mode="lines", line=dict(width=width, color=color),
                           showlegend=False, hovertemplate=f"{c}<extra></extra>") for j, c in enumerate(chans)]
    return traces, sp


def header(eyebrow: str, title: str, sub: str, toggle_key: str, toggle_help: str) -> bool:
    left, right = st.columns([5, 1.1], vertical_alignment="bottom")
    with left:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<p class="subtitle">{sub}</p>', unsafe_allow_html=True)
    return right.toggle("🎨 Art view", key=toggle_key, help=toggle_help)


@st.cache_data(show_spinner="Reading the recording …", max_entries=4)
def load_recording(key: str, _data: bytes) -> dict:
    return read_recording(io.BytesIO(_data))


@st.cache_data(show_spinner="Cutting windows, aligning the session, computing weights …", max_entries=8)
def run(key: str, _rec: dict, _onsets: pd.DataFrame | None, every: float, use_markers: bool, block: int):
    table, d = decode_recording(_rec, onset_table=_onsets, sliding_step=every, use_markers=use_markers, block=block)
    good = ~d.flat
    R = session_covariance(d.trials[good]) if good.any() else np.eye(30)
    W = d.model.W if d.model.W is not None else np.eye(30)
    return table, d.trials, d.onsets, d.flat, d.proba, R, W


# ------------------------------------------------------------------ sidebar
model = EmotionModel()
EMOS = model.emotions
sb = st.sidebar
sb.markdown('<div class="side-brand"><span class="dot">✳</span><div><b>EEG emotion decoder</b>'
            '<small>Emotiv FLEX2 → 5 emotion weights</small></div></div>', unsafe_allow_html=True)
sb.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
app_mode = sb.segmented_control("Mode", ["Recording", "Live"], default="Recording", key="mode",
                                label_visibility="collapsed") or "Recording"


def side_foot():
    sb.markdown(f'<div class="side-foot">Runs on CPU · <a href="{REPO}" target="_blank">code, CLI &amp; contributing</a></div>',
                unsafe_allow_html=True)


# ================================================================== LIVE
def live_page():
    from emotion_decoder.live import LiveDecoder, CsvTail, LslSource, ReplaySource

    sb.markdown('<div class="side-label">EEG source</div>', unsafe_allow_html=True)
    src_kind = sb.radio("EEG source", ["A CSV that is being written", "LSL stream (EmotivPRO)",
                                       "Simulation (replay an example)"], label_visibility="collapsed")
    path, from_end, ex, speed = None, True, None, 1.0
    if src_kind.startswith("A CSV"):
        path = sb.text_input("Path of the CSV", placeholder=r"C:\Users\me\Documents\EmotivPRO\recording.csv",
                             help="Windows paths work as they are. The recorder must append rows continuously "
                                  "(Timestamp + EEG.Cz … EEG.F8).")
        from_end = sb.checkbox("Only new data", value=True,
                               help="Start at the end of the file. Untick to replay it from the start.")
    elif src_kind.startswith("Simulation"):
        ex = sb.selectbox("Example", list(EXAMPLES))
        speed = sb.select_slider("Speed", [1.0, 2.0, 4.0, 8.0], value=1.0, format_func=lambda v: f"{v:g}×")
    sb.markdown('<div class="side-label">Windows</div>', unsafe_allow_html=True)
    every = sb.slider("A window every (s)", 1.0, 6.0, 3.0, 0.5)
    calib = sb.slider("Calibration windows", 5, 40, 10,
                      help="The first windows only estimate the session alignment; weights start after them.")
    block = sb.slider("Average over the last N windows", 1, 8, 4)
    markers = sb.checkbox("Use stimulus markers instead",
                          help="CSV: the MarkerIndex column. LSL: a stream of type Markers.")
    side_foot()

    art_on = header("Live · run locally", "Live decoding",
                    "Weights for five emotions, updated with every new window while the EEG arrives.",
                    "art_live", "Show the generative sculpture, driven live by the weights.")

    if src_kind.startswith("A CSV"):
        desc = f"Following <b>{path or '— enter a path in the sidebar'}</b>" + (" · new data only" if from_end else " · from the start")
    elif src_kind.startswith("LSL"):
        desc = "Listening to the first <b>LSL stream of type EEG</b>" + (" and Markers" if markers else "")
    else:
        desc = f"Replaying <b>{ex}</b> at {speed:g}×"
    with st.container(border=True):
        a, b, c = st.columns([5, 1, 1], vertical_alignment="center")
        a.markdown(f'<div style="font-size:13.5px;color:{INK2}">{desc}<br><span class="muted">a window every {every:g} s · '
                   f'{calib} calibration windows · average of the last {block}</span></div>', unsafe_allow_html=True)
        start = b.button("Start", type="primary", icon=":material/play_arrow:", width="stretch")
        stop = c.button("Stop", icon=":material/stop:", width="stretch")
    if start:
        try:
            if src_kind.startswith("A CSV"):
                if not path:
                    raise ValueError("enter the path of the CSV in the sidebar")
                source = CsvTail(path, from_end=from_end)
            elif src_kind.startswith("LSL"):
                source = LslSource(markers=markers)
            else:
                source = ReplaySource(read_recording(ROOT / "examples" / f"{EXAMPLES[ex]}.csv"), speed)
            st.session_state.live = {"src": source, "dec": LiveDecoder(every, calib, block, markers), "on": True,
                                     "t0": time.time(), "err": None, "art_n": 0}
            art.clear_live()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not start: {e}")
    if stop and "live" in st.session_state:
        st.session_state.live["on"] = False

    if art_on:
        embed(art.page_html(live=True))

    @st.fragment(run_every=1.0)
    def live_view():
        L = st.session_state.get("live")
        if not L:
            st.info("Choose the EEG source in the sidebar and press **Start**.", icon=":material/sensors:")
            return
        dec = L["dec"]
        if L["on"]:
            try:
                dec.push(*L["src"].poll()); dec.step(); L["err"] = None
            except Exception as e:  # noqa: BLE001
                L["err"] = str(e)
        if len(dec.results) != L.get("art_n", 0):                   # feed the sculpture
            r0 = dec.results[0]["start"] if dec.results else 0.0
            art.write_live([{"weights": r["block_weights"], "name": f"t {r['start'] - r0:.0f} s",
                             "kind": "" if dec.use_markers else "window"} for r in dec.results])
            L["art_n"] = len(dec.results)
        if L["err"]:
            st.warning(f"Waiting for data: {L['err']}", icon=":material/hourglass_top:")
        if not dec.n_samples and L["on"] and time.time() - L["t0"] > 5:
            st.warning("No EEG is arriving yet: is the recording running, and are the path / stream right?")

        m = st.columns(4)
        m[0].metric("Status", "Running" if L["on"] else "Stopped")
        m[1].metric("EEG received", f"{dec.n_samples / SFREQ:.0f} s")
        m[2].metric("Calibration", f"{min(dec.n_windows, dec.calib)} / {dec.calib}")
        m[3].metric("Windows decoded", len(dec.results))

        if not dec.results:
            done = min(dec.n_windows, dec.calib) / dec.calib
            st.progress(done, text=f"Calibrating the session alignment · first weights after about "
                                   f"{dec.calib * dec.every + 5:.0f} s of EEG")
        else:
            r = dec.results[-1]
            wb = r["block_weights"]
            top = max(wb, key=wb.get)
            left, right = st.columns([1, 2], gap="medium")
            with left, st.container(border=True):
                st.markdown(f'<div class="muted">Now · last {dec.block} windows</div>'
                            f'<div class="big-emo" style="color:{EMO_COLORS[top]}">{top}</div>', unsafe_allow_html=True)
                fb = go.Figure(go.Bar(y=ORDER, x=[wb[e] for e in ORDER], orientation="h",
                                      marker_color=[EMO_COLORS[e] for e in ORDER],
                                      text=[f"{wb[e]:.0%}" for e in ORDER], textposition="outside",
                                      cliponaxis=False))
                fb.add_vline(x=0.2, line=dict(dash="dot", color=MUTE, width=1))
                plot(style(fb, 215, xaxis=dict(range=[0, 1.08], tickformat=".0%"), bargap=0.35), "live_bar")
            with right, st.container(border=True):
                st.markdown('<div class="muted">Weights over time</div>', unsafe_allow_html=True)
                hist = dec.results[-60:]
                t = [h["start"] - dec.results[0]["start"] for h in hist]
                fl = go.Figure()
                for e in ORDER:
                    fl.add_trace(go.Scatter(x=t, y=[h["block_weights"][e] for h in hist], mode="lines", name=e,
                                            line=dict(color=EMO_COLORS[e], width=1.2), stackgroup="w"))
                plot(style(fl, 250, legend=True, xaxis_title="seconds",
                           yaxis=dict(range=[0, 1], tickformat=".0%")), "live_hist")

        ts, x = dec.recent_eeg(10.0)
        if len(ts):
            with st.container(border=True):
                st.markdown('<div class="muted">EEG · last 10 s</div>', unsafe_allow_html=True)
                fe = go.Figure()
                tr, sp = stacked(x - np.median(x, axis=1, keepdims=True), VIEW, ts - ts[-1])
                for t_ in tr:
                    fe.add_trace(t_)
                plot(style(fe, 210, xaxis_title="seconds",
                           yaxis=dict(tickvals=[-j * sp for j in range(len(VIEW))], ticktext=VIEW, showgrid=False)),
                     "live_eeg")
        if dec.results:
            df = pd.DataFrame([{"t_s": round(h["start"] - dec.results[0]["start"], 2),
                                **{f"w_{k}": v for k, v in h["weights"].items()},
                                **{f"block_{k}": v for k, v in h["block_weights"].items()}} for h in dec.results])
            st.download_button("Download the weights so far", df.to_csv(index=False).encode(), "live_weights.csv",
                               "text/csv", icon=":material/download:", key=f"dl{len(df)}")

    live_view()


if app_mode == "Live":
    live_page()
    st.stop()


# ================================================================== RECORDING
sb.markdown('<div class="side-label">Recording</div>', unsafe_allow_html=True)
src = sb.radio("Recording", list(EXAMPLES) + ["Upload my CSV"], label_visibility="collapsed",
               captions=["synthetic", "synthetic", "synthetic", "EmotivPRO CSV export"])
if src == "Upload my CSV":
    up = sb.file_uploader("EmotivPRO CSV export", type=["csv"], label_visibility="collapsed")
    data = up.getvalue() if up else None
else:
    name = EXAMPLES[src]
    data = (ROOT / "examples" / f"{name}.csv").read_bytes()

sb.markdown('<div class="side-label">Windows</div>', unsafe_allow_html=True)
every = sb.slider("A window every (s)", 1.0, 6.0, 3.0, 0.5,
                  help="Each window: 0.5 s before to 5.0 s after its start. Only the EEG is needed.")
onset_tbl, use_markers = None, False
if sb.checkbox("Use stimulus onsets instead", value=False,
               help="Place windows where stimuli appeared: markers in the CSV, or an onsets CSV."):
    if src == "Upload my CSV":
        how = sb.radio("Onsets from", ["Markers in the CSV", "An onsets CSV"], horizontal=True)
        if how == "Markers in the CSV":
            use_markers = True
        else:
            oup = sb.file_uploader("Onsets CSV (column onset_unix; optional column emotion)", type=["csv"],
                                   help="Same clock as the Timestamp column of the EEG CSV.")
            if oup:
                try:
                    onset_tbl = read_onsets_table(oup)
                except Exception as e:  # noqa: BLE001
                    sb.error(f"Could not read the onsets CSV: {e}")
    else:
        onset_tbl = read_onsets_table(ROOT / "examples" / f"{name}_onsets.csv")
        sb.caption("Using the example's onsets file, which also holds the true emotions.")
sb.markdown('<div class="side-label">Smoothing</div>', unsafe_allow_html=True)
block = sb.slider("Average over the last N windows", 1, 8, 4,
                  help="Averaging the probabilities of consecutive windows makes the weights steadier.")
side_foot()

art_on = header("EEG → five emotions", "EEG emotion decoder",
                "From an Emotiv FLEX2 recording, weights for five emotions (Joy, Tenderness, Nostalgia, Serenity "
                "and Melancholy) for a window every few seconds. Only the EEG is needed.",
                "art_rec", "Show the generative sculpture, played window by window.")
st.markdown("""
<div class="steps">
  <div class="step"><span class="n">1</span><b>Cut</b><p>5.5 s of EEG per window, 30 channels at 256 Hz.</p></div>
  <div class="step"><span class="n">2</span><b>Align the session</b><p>Undo the session's channel covariance, so headset placement stops mattering.</p></div>
  <div class="step"><span class="n">3</span><b>Response shape</b><p>The first 940 ms, filtered 1–30 Hz, as 30 channels × 20 time bins.</p></div>
  <div class="step"><span class="n">4</span><b>Compare</b><p>Shrinkage LDA against each emotion's average response gives 5 weights.</p></div>
</div>""", unsafe_allow_html=True)

if data is None:
    st.info("Upload an **EmotivPRO CSV export** in the sidebar, or pick an example. The file needs a `Timestamp` "
            "column and the 30 channels `EEG.Cz` … `EEG.F8` (µV, 256 Hz); everything else is ignored.",
            icon=":material/upload_file:")
    st.stop()

key = hashlib.md5(data).hexdigest()
try:
    rec = load_recording(key, data)
except RecordingError as e:
    st.error(f"Could not read this file: {e}", icon=":material/error:")
    st.stop()
except Exception as e:  # noqa: BLE001 - any other unreadable file: say so instead of a traceback
    st.error(f"Could not read this file ({type(e).__name__}): {e}", icon=":material/error:")
    st.stop()
if np.isfinite(rec["fs_estimate"]) and abs(rec["fs_estimate"] - SFREQ) > 0.1 * SFREQ:
    st.warning(f"The timestamps suggest {rec['fs_estimate']:.0f} Hz; the model expects {SFREQ} Hz.")
okey = hashlib.md5(onset_tbl.to_csv().encode()).hexdigest() if onset_tbl is not None else "none"
try:
    table, trials, onsets, flat, proba, Rcov, W = run(f"{key}-{okey}", rec, onset_tbl, every, use_markers, block)
except ValueError as e:
    st.warning(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not decode this recording ({type(e).__name__}): {e}", icon=":material/error:")
    st.stop()

n = len(table)
ts0 = rec["ts"][0]
t_tr = onsets - ts0
wcols = {e: f"w_{e}" for e in EMOS}
bcols = {e: (f"block{block}_{e}" if block > 1 else f"w_{e}") for e in EMOS}
acc = label_accuracy(table, EMOS)

c = st.columns(5)
c[0].metric("Duration", f"{(rec['ts'][-1] - ts0) / 60:.1f} min")
c[1].metric("Windows", n)
c[2].metric("Skipped", int(flat.sum()), help="No signal (flat) or data missing inside the window (dropped packets); "
                                              "as in training, these get no weights.")
if acc:
    c[3].metric("Top emotion correct", f"{acc['accuracy']:.0%}", f"chance {acc['chance']:.0%}", delta_color="off")
    if "block_accuracy" in acc:
        c[4].metric(f"Average of {block} correct", f"{acc['block_accuracy']:.0%}")
else:
    c[3].metric("Alignment from", f"{int((~flat).sum())} windows")
    c[4].metric("Window step", f"{every:g} s" if onset_tbl is None and not use_markers else "onsets")
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
        rows.append({"weights": {e: float(table.loc[j, bcols[e]]) for e in EMOS},
                     "label": str(table.loc[j, lab_col]) if lab_col else None,
                     "name": f"window {j + 1} · {t_tr[j]:.0f} s",
                     "kind": "window" if onset_tbl is None and not use_markers else ""})
    section("Sculpture", "The five strata follow the five weights" +
            (f", averaged over the last {block} windows" if block > 1 else "") +
            ". Play steps through the windows; switch Art view off for the analysis.")
    embed(art.page_html(art.build_doc(rows, "recording")))
    st.stop()

# ------------------------------------------------------------------ session overview
section("The session", "Each window's five weights, stacked like the sculpture's strata"
        + (f" (averaged over the last {block} windows)" if block > 1 else "") + ".")
with st.container(border=True):
    st.markdown(chips(), unsafe_allow_html=True)
    wf = go.Figure()
    for e in ORDER:
        wf.add_trace(go.Bar(x=table["trial"], y=table[bcols[e]], name=e, marker_color=EMO_COLORS[e], marker_line_width=0,
                            hovertemplate=f"window %{{x}}<br>{e} %{{y:.0%}}<extra></extra>"))
    if flat.any():
        wf.add_trace(go.Bar(x=table["trial"][flat], y=[1.0] * int(flat.sum()), name="skipped", marker_color=SKIP_COLOR,
                            marker_line_width=0, hovertext=table["skipped"][flat], hoverinfo="text+x"))
    if acc:
        lab = table[acc["label_column"]].astype(str).str.capitalize()
        wf.add_trace(go.Scatter(x=table["trial"], y=[1.05] * n, mode="markers", showlegend=False,
                                marker=dict(symbol="square", size=8, color=[EMO_COLORS.get(v, MUTE) for v in lab]),
                                hovertext="true: " + lab, hoverinfo="text"))
    plot(style(wf, 250, barmode="stack", bargap=0, xaxis_title="window",
               yaxis=dict(range=[0, 1.1 if acc else 1.0], tickformat=".0%")), "weights")
    view = st.multiselect("Channels", CHANNELS, default=VIEW, label_visibility="collapsed",
                          placeholder="Channels to show")
    x = rec["eeg"] - np.median(rec["eeg"], axis=1, keepdims=True)
    stride = max(1, x.shape[1] // 6000)
    fig = go.Figure()
    if view:
        tr, sp = stacked(x[:, ::stride], view, rec["ts"][::stride] - ts0, width=0.55, color="#8A90A0")
        for t_ in tr:
            fig.add_trace(t_)
        fig.update_yaxes(tickvals=[-j * sp for j in range(len(view))], ticktext=view, showgrid=False)
    for j in range(n):
        fig.add_vline(x=t_tr[j], line=dict(color=SKIP_COLOR if flat[j] else EMO_COLORS[table.loc[j, "top_emotion"]],
                                           width=1.6))
    plot(style(fig, 230, xaxis_title="time (s)"), "signal")
    st.caption("Below: the EEG, with a line at the start of every window in the colour of its top emotion"
               + (" (grey: skipped)" if flat.any() else "") + (". Squares on top: the true emotion." if acc else "."))


# ------------------------------------------------------------------ window player
@st.fragment
def player():
    section("Window by window", "The four steps for one window. Play walks through the session.")
    if "trial" not in st.session_state or st.session_state.trial >= n:
        st.session_state.trial = 0
    if st.session_state.get("playing"):
        st.session_state.trial = (st.session_state.trial + 1) % n

    def nudge(d):
        st.session_state.trial = int(np.clip(st.session_state.trial + d, 0, n - 1))

    b = st.columns([1, 1, 1.15, 7], vertical_alignment="center")
    b[0].button("Prev", icon=":material/chevron_left:", on_click=nudge, args=(-1,), width="stretch")
    b[1].button("Next", icon=":material/chevron_right:", on_click=nudge, args=(1,), width="stretch")
    playing = st.session_state.get("playing", False)
    if b[2].button("Pause" if playing else "Play", icon=":material/pause:" if playing else ":material/play_arrow:",
                   type="primary", width="stretch"):
        st.session_state.playing = not playing
        st.rerun(scope="fragment")
    b[3].slider("Window", 0, n - 1, key="trial", format="%d", label_visibility="collapsed")

    i = st.session_state.trial
    parts = [f"<b>Window {i + 1}</b> of {n}", f"starts at <b>{int(t_tr[i] // 60)}:{t_tr[i] % 60:04.1f}</b>"]
    if flat[i]:
        parts.append(f"skipped · {table.loc[i, 'skipped']}")
    else:
        parts.append("top emotion " + badge(table.loc[i, "top_emotion"]))
        if acc:
            true = str(table.loc[i, acc["label_column"]]).capitalize()
            ok = true == table.loc[i, "top_emotion"]
            parts.append(f"true {badge(true)} <span class='{'ok' if ok else 'no'}'>{'✓' if ok else '✗'}</span>")
    st.markdown('<div class="now">' + '<span class="sep">·</span>'.join(f"<span>{p}</span>" for p in parts) + "</div>",
                unsafe_allow_html=True)

    tt = (np.arange(N_SAMPLES) - ONSET) / SFREQ
    seg = trials[i]
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            card_head("1", "Cut", "5.5 s of EEG: 0.5 s before the window start to 5.0 s after. "
                                  "The model reads the shaded first 940 ms.")
            f1 = go.Figure()
            tr, sp = stacked(seg - np.median(seg, axis=1, keepdims=True), VIEW, tt)
            for t_ in tr:
                f1.add_trace(t_)
            f1.add_vrect(x0=0, x1=0.9375, fillcolor=EMO_COLORS["Nostalgia"], opacity=0.07, line_width=0)
            f1.add_vline(x=0, line=dict(color="#C8553D", width=1.2))
            plot(style(f1, 250, xaxis_title="seconds from window start",
                       yaxis=dict(tickvals=[-j * sp for j in range(len(VIEW))], ticktext=VIEW, showgrid=False)), "f1")
        with st.container(border=True):
            card_head("2", "Align the session", "The session's channel covariance (left) is undone, "
                                                "R<sup>-½</sup>: after alignment (right) every session looks alike.")
            a1, a2 = st.columns(2)
            Rn = Rcov / np.sqrt(np.outer(np.diag(Rcov), np.diag(Rcov)))
            for col, M, title in ((a1, Rn, "before · correlation"), (a2, W @ Rcov @ W.T, "after alignment")):
                h = go.Figure(go.Heatmap(z=M, x=CHANNELS, y=CHANNELS, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                                         showscale=False, hovertemplate="%{y} – %{x}: %{z:.2f}<extra></extra>"))
                with col:
                    st.markdown(f'<div class="muted" style="text-align:center">{title}</div>', unsafe_allow_html=True)
                    plot(style(h, 210, xaxis=dict(showticklabels=False, showgrid=False),
                               yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed")), f"a_{title}")
    with right:
        if flat[i]:
            with st.container(border=True):
                st.info("This window is skipped (no signal, or data missing from dropped packets): it gets no "
                        "weights and is left out of the alignment.", icon=":material/block:")
        else:
            erp = erp_filtered(W @ seg.astype(np.float64))
            F = erp[:, ONSET:ONSET + N_BINS * 12].reshape(30, N_BINS, 12).mean(2)
            bins_ms = (np.arange(N_BINS) + 0.5) * BIN_MS
            with st.container(border=True):
                card_head("3", "Response shape", "Aligned, filtered 1–30 Hz, minus the first 0.5 s; 0–940 ms "
                                                 "in 20 bins of 47 ms → 30 × 20 = 600 numbers.")
                lim = float(np.percentile(np.abs(F), 98)) or 1.0
                h3 = go.Figure(go.Heatmap(z=F, x=np.round(bins_ms), y=CHANNELS, colorscale=DIVERGING, zmid=0,
                                          zmin=-lim, zmax=lim, colorbar=dict(thickness=8, outlinewidth=0, len=0.9),
                                          hovertemplate="%{y} · %{x} ms: %{z:.2f}<extra></extra>"))
                plot(style(h3, 250, xaxis_title="ms after window start",
                           yaxis=dict(tickfont=dict(size=8), autorange="reversed", showgrid=False)), "f3")
            with st.container(border=True):
                card_head("4", "Compare → weights", "Each emotion's average response (dotted, posterior channels) "
                                                    "against this window; shrinkage LDA turns all 600 numbers into weights.")
                T = model.templates()
                pi = [CHANNELS.index(c) for c in POSTERIOR]
                f4 = go.Figure()
                for e in ORDER:
                    f4.add_trace(go.Scatter(x=bins_ms, y=T[EMOS.index(e)][pi].mean(0), mode="lines", name=e,
                                            line=dict(color=EMO_COLORS[e], width=1.6, dash="dot")))
                f4.add_trace(go.Scatter(x=bins_ms, y=F[pi].mean(0), mode="lines+markers", name="this window",
                                        line=dict(color=INK, width=2.2), marker=dict(size=4)))
                plot(style(f4, 190, legend=True, xaxis_title="ms after window start"), "f4")
                p = {e: float(table.loc[i, wcols[e]]) for e in EMOS}
                pb = {e: float(table.loc[i, bcols[e]]) for e in EMOS}
                f5 = go.Figure()
                f5.add_trace(go.Bar(y=ORDER, x=[p[e] for e in ORDER], orientation="h", name="this window",
                                    marker_color=[EMO_COLORS[e] for e in ORDER],
                                    text=[f"{p[e]:.0%}" for e in ORDER], textposition="outside", cliponaxis=False))
                if block > 1:
                    f5.add_trace(go.Bar(y=ORDER, x=[pb[e] for e in ORDER], orientation="h",
                                        name=f"last {block} windows", marker_color="rgba(17,21,40,0.16)",
                                        text=[f"{pb[e]:.0%}" for e in ORDER], textposition="outside",
                                        cliponaxis=False))
                f5.add_vline(x=0.2, line=dict(dash="dot", color=MUTE, width=1))
                plot(style(f5, 215, legend=block > 1, barmode="group", bargap=0.3,
                           xaxis=dict(range=[0, 1.12], tickformat=".0%")), "f5")
                st.caption("Dotted line: chance (20%).")

    if st.session_state.get("playing"):
        time.sleep(1.0)
        st.rerun(scope="fragment")


player()

# ------------------------------------------------------------------ table + downloads
section("Weights per window", "One row per window; download the full table as CSV.")
show = ["trial", "t_s", "top_emotion"] + [wcols[e] for e in ORDER]
if block > 1:
    show.append(f"block{block}_top")
if acc:
    show.append(acc["label_column"])
show.append("skipped")
cfg = {wcols[e]: st.column_config.ProgressColumn(e, min_value=0.0, max_value=1.0, format="%.2f") for e in ORDER}
cfg.update({"trial": st.column_config.NumberColumn("Window", format="%d"),
            "t_s": st.column_config.NumberColumn("Start (s)", format="%.1f"),
            "top_emotion": "Top emotion", f"block{block}_top": f"Top, last {block}", "skipped": "Skipped"})
if acc:
    cfg[acc["label_column"]] = "True emotion"
st.dataframe(table[[c for c in show if c in table]], column_config=cfg, width="stretch", height=300, hide_index=True)
d1, d2, _ = st.columns([1.2, 1.2, 4])
d1.download_button("Download CSV", table.to_csv(index=False).encode(), "emotion_weights.csv", "text/csv",
                   icon=":material/download:", width="stretch")
with d2.popover("Use from code / CLI", icon=":material/terminal:", width="stretch"):
    st.code("pip install -r requirements.txt\n\n"
            "# command line: a window every 3 s\n"
            "python -m emotion_decoder my_recording.csv --block 4 --out weights.csv\n\n"
            "# Python\nfrom emotion_decoder import decode_csv\n"
            "table, details = decode_csv(\"my_recording.csv\", block=4)", language="bash")
st.markdown(f'<div class="foot">EEG emotion decoder · open source, pull requests welcome · '
            f'<a href="{REPO}" target="_blank">GitHub</a></div>', unsafe_allow_html=True)
