"""Export the 5-emotion checkpoint (emotion_weights_v2.joblib, scikit-learn 1.7.2) to models/emotion5_v2.json.

The checkpoint comes from https://github.com/coralexbadeashare/eeg-stimulus-type-baselines
(checkpoints/emotion5). The JSON holds plain numbers only: the feature scaler, the shrinkage-LDA
discriminant (coef, intercept) and class means, the calibration temperature, and a self-check
(probabilities scikit-learn gives for a fixed random input), which tests/ compare against the
numpy model.

    python tools/export_model.py --joblib path/to/emotion_weights_v2.joblib
"""
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import sklearn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joblib", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models" / "emotion5_v2.json"))
    a = ap.parse_args()

    b = joblib.load(a.joblib)
    sc, lda = b["scaler"], b["lda"]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, len(sc.mean_)))                       # already in scaled units
    out = {
        "name": "emotion5_v2",
        "emotions": list(b["emotions"]),
        "pipeline": ("per-session Euclidean alignment (R^-1/2, R = mean 1-45 Hz spatial covariance, 0-4 s) "
                     "-> 1-30 Hz band-pass -> subtract the 0.5 s pre-stimulus mean -> 0-937.5 ms in 20 bins "
                     "of 12 samples x 30 channels = 600 -> StandardScaler -> shrinkage LDA (lsqr, auto, equal priors)"),
        "trained_on": "370 QC-passed image trials, one participant, 30-channel Emotiv FLEX2, 256 Hz, "
                      "EmotivPRO band-passed export (no other cleaning)",
        "evaluation": {"chance": 0.20, "leave_one_recording_out": 0.263, "held_out_recordings": 0.284,
                       "block_of_2": 0.286, "block_of_8": 0.334, "pictures_never_seen": 0.226},
        "source": "github.com/coralexbadeashare/eeg-stimulus-type-baselines (checkpoints/emotion5)",
        "sklearn_version": sklearn.__version__,
        "temperature": float(b["temperature"]),
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": lda.coef_.tolist(),
        "intercept": lda.intercept_.tolist(),
        "class_means_scaled": lda.means_.tolist(),
        "self_check": {"seed": 0, "n": 8, "scaled_input": True, "proba": lda.predict_proba(X).tolist()},
    }
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {out['emotions']}, coef {np.shape(lda.coef_)}, T={out['temperature']:.3f}")


if __name__ == "__main__":
    main()
