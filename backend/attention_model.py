"""Attention (mind-wandering risk) model.

Features are computed in the browser per 5-second window and sent here.
If a trained model exists in models/, it is used; otherwise a transparent
heuristic is used so the demo always works.
"""
from pathlib import Path
import math

FEATURES = [
    "face_ratio", "blink_rate", "perclos", "gaze_off_ratio", "gaze_std", "saccade_rate",
    "yaw_abs", "pitch_abs", "head_motion", "mouth_open", "brow_furrow", "lean_dev",
    "tab_hidden", "interaction_rate",
]
DEFAULTS = {"face_ratio": 1.0}

# Modality groups, used for ablations in train.py
GROUPS = {
    "gaze": ["gaze_off_ratio", "gaze_std", "saccade_rate"],
    "head_posture": ["yaw_abs", "pitch_abs", "head_motion", "lean_dev"],
    "eyes": ["blink_rate", "perclos"],
    "face": ["face_ratio", "mouth_open", "brow_furrow"],
    "behaviour": ["tab_hidden", "interaction_rate"],
}

MODEL_PATH = Path(__file__).parent / "models" / "attention_model.joblib"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def heuristic_risk(f: dict) -> float:
    z = (
        -2.6
        + 2.5 * (1.0 - f["face_ratio"])
        + 2.4 * f["gaze_off_ratio"]
        + 1.2 * min(f["gaze_std"], 1.0)
        + 0.15 * min(f["saccade_rate"], 6.0)
        + 1.5 * min(f["yaw_abs"] / 30.0, 1.5)
        + 1.0 * min(f["pitch_abs"] / 30.0, 1.5)
        + 0.8 * min(f["head_motion"] / 20.0, 1.5)
        + 2.0 * f["perclos"]
        + 0.15 * min(abs(f["blink_rate"] - 15.0) / 15.0, 2.0)
        + 1.5 * f["mouth_open"]
        + 1.2 * min(f["lean_dev"], 1.0)
        + 3.0 * f["tab_hidden"]
        + 0.3 * f["interaction_rate"]
    )
    return _sigmoid(z)


class AttentionModel:
    def __init__(self):
        self.pipeline = None
        self.source = "heuristic"
        if MODEL_PATH.exists():
            import joblib
            bundle = joblib.load(MODEL_PATH)
            self.pipeline = bundle["pipeline"]
            self.source = bundle.get("source", "trained")

    def predict(self, features: dict) -> tuple[float, str]:
        f = {k: float(features.get(k, DEFAULTS.get(k, 0.0))) for k in FEATURES}
        if self.pipeline is None:
            return heuristic_risk(f), self.source
        x = [[f[k] for k in FEATURES]]
        return float(self.pipeline.predict_proba(x)[0][1]), self.source