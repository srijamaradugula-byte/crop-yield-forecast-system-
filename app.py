"""CropCast AI - Flask app.
Uses trained models from models/ if present, else the simulator fallback."""
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import simulator

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
app = Flask(__name__)

# ---- field definitions (shared by form + validation) ------------------------
DISTRICTS = ["Eluru", "West Godavari", "Krishna", "Guntur", "Prakasam", "Kurnool", "Anantapur", "Chittoor"]
FIELDS = [
    {"k": "days", "l": "Days after sowing", "t": "n", "v": 62, "r": [0, 150], "req": True},
    {"k": "crop", "l": "Crop", "t": "s", "o": list(simulator.CROPS)},
    {"k": "district", "l": "District", "t": "s", "o": DISTRICTS},
    {"k": "season", "l": "Season", "t": "s", "o": ["Kharif", "Rabi"]},
    {"k": "soil", "l": "Soil type", "t": "s", "o": ["Clay loam", "Black cotton", "Red sandy", "Alluvial"]},
    {"k": "area", "l": "Area (acres)", "t": "n", "v": 3.5, "r": [0.1, 500], "req": True},
    {"k": "irr", "l": "Irrigation", "t": "s", "o": ["Canal", "Borewell", "Rainfed"]},
    {"k": "seed", "l": "Seed quality", "t": "s", "o": ["Certified", "Farm-saved"]},
    {"k": "delay", "l": "Sowing delay (days)", "t": "n", "v": 0, "r": [-30, 60]},
    {"k": "N", "l": "Nitrogen (kg/acre)", "t": "n", "v": 50, "r": [0, 250]},
    {"k": "P", "l": "Phosphorus (kg/acre)", "t": "n", "v": 26, "r": [0, 150]},
    {"k": "K", "l": "Potash (kg/acre)", "t": "n", "v": 22, "r": [0, 150]},
    {"k": "n30", "l": "NDVI at 30 days", "t": "n", "v": "", "r": [0, 1]},
    {"k": "n60", "l": "NDVI at 60 days", "t": "n", "v": 0.71, "r": [0, 1]},
    {"k": "n90", "l": "NDVI at 90 days", "t": "n", "v": "", "r": [0, 1]},
    {"k": "rain", "l": "Rainfall (mm)", "t": "n", "v": "", "r": [0, 3000]},
    {"k": "pest", "l": "Pest incidence (%)", "t": "n", "v": "", "r": [0, 100]},
]

# ---- optional trained models -------------------------------------------------
STAGE_FILES = {0: "stage_0.joblib", 30: "stage_30.joblib", 60: "stage_60.joblib", 90: "stage_90.joblib"}
MODELS, INTERVALS = {}, {}


def load_models():
    """Load models/stage_*.joblib and models/intervals.json if they exist."""
    try:
        import joblib
    except ImportError:
        return
    for st, fn in STAGE_FILES.items():
        f = MODEL_DIR / fn
        if f.exists():
            MODELS[st] = joblib.load(f)
    f = MODEL_DIR / "intervals.json"
    if f.exists():
        INTERVALS.update(json.loads(f.read_text()))


load_models()


def predict_with_model(p):
    """TODO: align the feature columns with your stage_models.py, then the
    trained model is used automatically. Until then the simulator answers."""
    st = simulator.stage_of(int(p["days"]))
    if st not in MODELS:
        return None
    import pandas as pd
    # Column names/order must match what you trained with:
    X = pd.DataFrame([p])
    y = float(MODELS[st].predict(X)[0])
    # INTERVALS example shape: {"Paddy": {"60": [-2.1, 2.3]}, ...} (residual quantiles)
    q = INTERVALS.get(p["crop"], {}).get(str(st))
    lo, hi = (y + q[0], y + q[1]) if q else (y * (1 - simulator.REL[st]), y * (1 + simulator.REL[st]))
    price = simulator.CROPS[p["crop"]]["p"]
    r2, rmse, mape = simulator.ACC[st]
    tot = y * p["area"]
    return {"stage": st, "yield": round(y, 2), "low": round(max(lo, 0), 2), "high": round(hi, 2),
            "total": round(tot, 1), "price": price, "revenue": round(tot * price),
            "r2": r2, "rmse": rmse, "mape": mape, "engine": "trained-model"}


def parse(form):
    """Validate input; returns (values, error)."""
    v = {}
    for f in FIELDS:
        k = f["k"]
        raw = form.get(k)
        if f["t"] == "s":
            if raw not in f["o"]:
                return None, f"{f['l']} is invalid."
            v[k] = raw
            continue
        if raw is None or str(raw).strip() == "":
            if f.get("req"):
                return None, f"{f['l']} is required."
            v[k] = None
            continue
        try:
            x = float(raw)  # accepts 62.5 too (no int() crash)
        except ValueError:
            return None, f"{f['l']} must be a number."
        lo, hi = f["r"]
        if not lo <= x <= hi:
            return None, f"{f['l']} must be between {lo} and {hi}."
        v[k] = x
    return v, None


@app.route("/")
def index():
    return render_template("index.html", fields=FIELDS, engine="trained-model" if MODELS else "simulator")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or request.form
    v, err = parse(data)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(predict_with_model(v) or simulator.predict(v))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(MODELS)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
