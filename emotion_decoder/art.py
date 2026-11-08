"""Bridge to the generative sculpture (static/linetree_studio.html).

The page already plays back emotion predictions: its "EEG session" pane reads a JSON
  {"classes": ["01_JOY", "02_TENDERNESS", "03_NOSTALGIA", "04_SERENITY", "05_MELANCHOLY"],
   "sessions": {name: [{"trial_id", "proba": [5], "predicted", "confidence", "emotion"?}, ...]}}
and turns the five probabilities into the heights of its five coloured strata. We build that JSON
from the decoder's weights and hand it over through the page's own functions (eegAccept, eegShow)
with a small script appended to the page. The page's own code is not modified; it only
gained a responsive-layout CSS block so the sculpture keeps its space inside the app.

Live: the app rewrites static/live_emotions.json after every new window, and the appended script
polls it once a second and moves the sculpture to the newest window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "static" / "linetree_studio.html"
LIVE_JSON = ROOT / "static" / "live_emotions.json"
LIVE_URL = "/app/static/live_emotions.json"            # Streamlit static serving
ART_ORDER = ["Joy", "Tenderness", "Nostalgia", "Serenity", "Melancholy"]   # the page's strata, bottom to top
ART_CLASSES = ["01_JOY", "02_TENDERNESS", "03_NOSTALGIA", "04_SERENITY", "05_MELANCHOLY"]


def _trial(i: int, weights: dict, label: str | None = None, name: str = "", kind: str = "") -> dict:
    p = [float(weights[e]) for e in ART_ORDER]
    s = sum(p) or 1.0
    p = [v / s for v in p]
    k = int(np.argmax(p))
    t = {"trial_id": i, "proba": [round(v, 5) for v in p], "predicted": ART_CLASSES[k],
         "confidence": round(p[k], 5), "stimulus_filename": name, "stimulus_type": kind}
    if label and label.capitalize() in ART_ORDER:
        t["emotion"] = ART_CLASSES[ART_ORDER.index(label.capitalize())]
    return t


def build_doc(rows: list[dict], session: str = "recording") -> dict:
    """rows: dicts with `weights` ({emotion: weight}) and optional `label`, `name`, `kind`."""
    trials = [_trial(i, r["weights"], r.get("label"), r.get("name", ""), r.get("kind", ""))
              for i, r in enumerate(rows)]
    return {"classes": ART_CLASSES, "n_trials": len(trials), "seq": len(trials), "sessions": {session: trials}}


def write_live(rows: list[dict]) -> None:
    """Atomically rewrite the live JSON the page polls."""
    LIVE_JSON.parent.mkdir(exist_ok=True)
    tmp = LIVE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(build_doc(rows, "live")))
    os.replace(tmp, LIVE_JSON)


def clear_live() -> None:
    LIVE_JSON.unlink(missing_ok=True)


BRIDGE = """
<script>
/* Appended by the EEG emotion decoder: feeds the page through its own EEG-playback functions. */
(function(){
  const DOC = %(doc)s, LIVE = %(live)s;
  function load(doc){
    if(!EEG.pred){ eegAccept(doc, eegLoadPred, 'Predictions', doc.name || 'EEG emotion decoder'); }
    else { EEG.pred = doc; EEG.trials = eegBuildTrials(Object.keys(doc.sessions)[0]); }
  }
  function openPane(){ try { paneShow('eeg'); } catch(e){} }
  if(DOC){ load(DOC); openPane(); }
  if(LIVE){
    let seq = -1;
    setInterval(async () => {
      try {
        const r = await fetch(LIVE + '?t=' + Date.now(), {cache: 'no-store'});
        if(!r.ok) return;
        const doc = await r.json();
        if(doc.seq === seq || !doc.n_trials) return;
        const first = seq < 0; seq = doc.seq;
        load(doc); if(first) openPane();
        eegShow(EEG.trials.length - 1);
      } catch(e){}
    }, 1000);
  }
})();
</script>
"""


def page_html(doc: dict | None = None, live: bool = False) -> str:
    html = PAGE.read_text(encoding="utf-8")
    bridge = BRIDGE % {"doc": json.dumps(doc) if doc else "null", "live": json.dumps(LIVE_URL) if live else "null"}
    i = html.rfind("</body>")
    return html[:i] + bridge + html[i:] if i >= 0 else html + bridge
