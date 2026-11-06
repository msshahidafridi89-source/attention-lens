"""Train and evaluate the attention model.

  python train.py                          # synthetic data (pipeline demo only)
  python train.py --real sessions/*.json   # real exported sessions with probe labels

Evaluation is subject-wise (GroupKFold): the model is always tested on people
it has not seen. Ablations drop one modality at a time.
NOTE: synthetic-data scores demonstrate the pipeline, NOT real validity.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from attention_model import FEATURES, GROUPS, MODEL_PATH

LOOKBACK_S = 10


def synth(n_subjects=40, n_windows=80, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subjects):
        gaze_bias, yaw_bias, blink_base = rng.normal(0, .03), rng.normal(0, 3), rng.normal(15, 3)
        state = 0
        for _ in range(n_windows):
            if rng.random() < (0.12 if state == 0 else 0.25):
                state = 1 - state
            m = state
            row = {
                "face_ratio": np.clip(rng.normal(0.97 - 0.05 * m, 0.06), 0, 1),
                "blink_rate": max(0, rng.normal(blink_base + 2 * m, 6)),
                "perclos": np.clip(rng.normal(0.03 + 0.04 * m, 0.04), 0, 1),
                "gaze_off_ratio": np.clip(rng.normal(0.12 + 0.12 * m + gaze_bias, 0.12), 0, 1),
                "gaze_std": max(0, rng.normal(0.15 + 0.06 * m, 0.08)),
                "saccade_rate": max(0, rng.normal(1.5 + 0.7 * m, 1.0)),
                "yaw_abs": max(0, rng.normal(7 + 4 * m + yaw_bias, 6)),
                "pitch_abs": max(0, rng.normal(5 + 2 * m, 4)),
                "head_motion": max(0, rng.normal(4 + 1.5 * m, 3)),
                "mouth_open": np.clip(rng.normal(0.02 + 0.03 * m, 0.04), 0, 1),
                "brow_furrow": np.clip(rng.normal(0.1 + 0.02 * m, 0.08), 0, 1),
                "lean_dev": max(0, rng.normal(0.06 + 0.03 * m, 0.05)),
                "tab_hidden": float(rng.random() < 0.03 * (1 + 2 * m)) * rng.uniform(0.2, 1),
                "interaction_rate": float(rng.poisson(0.2 + 0.2 * m)),
            }
            X.append([row[k] for k in FEATURES]); y.append(m); g.append(s)
    return np.array(X), np.array(y), np.array(g)


def load_real(paths):
    X, y, g = [], [], []
    for gi, p in enumerate(paths):
        sess = json.loads(Path(p).read_text())
        for w in sess["windows"]:
            for pr in sess["probes"]:
                if pr["t"] - LOOKBACK_S < w["t"] <= pr["t"]:
                    X.append([w["features"].get(k, 0.0) for k in FEATURES])
                    y.append(int(pr["mind_wandering"])); g.append(gi)
    return np.array(X), np.array(y), np.array(g)


def cv_auc(model, X, y, g):
    n = min(5, len(set(g)))
    p = cross_val_predict(model, X, y, groups=g, cv=GroupKFold(n), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*")
    a = ap.parse_args()
    if a.real:
        files = [f for pat in a.real for f in glob.glob(pat)]
        X, y, g = load_real(files)
        source = f"real:{len(set(g))}_sessions"
    else:
        X, y, g = synth()
        source = "synthetic"
    if len(set(y)) < 2 or len(set(g)) < 2:
        raise SystemExit("Need at least 2 sessions and both classes in the labels.")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150)
    metrics = {"source": source, "n_windows": int(len(y)), "positive_rate": float(y.mean()),
               "auroc": {}, "ablation_drop": {}}
    metrics["auroc"]["logistic_all"] = cv_auc(lr, X, y, g)
    metrics["auroc"]["boosting_all"] = cv_auc(gb, X, y, g)
    for name, cols in GROUPS.items():
        keep = [i for i, f in enumerate(FEATURES) if f not in cols]
        metrics["ablation_drop"][name] = metrics["auroc"]["logistic_all"] - cv_auc(lr, X[:, keep], y, g)

    print(json.dumps(metrics, indent=2))
    lr.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": lr, "source": source}, MODEL_PATH)
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()