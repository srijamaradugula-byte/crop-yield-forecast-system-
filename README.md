# CropCast AI – Per-Acre Yield Forecasting

Stage-wise (0/30/60/90 days) yield forecasting with a 90% interval, total production and indicative revenue.

## Contents
| Path | Purpose |
|---|---|
| `docs/index.html` | Static browser demo (simulator). Hosted on GitHub Pages. |
| `app.py`, `simulator.py`, `templates/` | Flask app. Uses trained models in `models/` if present, else the simulator. |
| `models/` | Put `stage_0.joblib`, `stage_30.joblib`, `stage_60.joblib`, `stage_90.joblib`, `intervals.json` here. |

> The data is synthetic. The online demo uses a simplified simulator; the trained gradient boosting models run in the Flask app.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

## Deploy
**GitHub Pages (static demo):** Settings → Pages → Branch `main`, folder `/docs`.

**Render (Flask):** New → Web Service → connect repo. Build `pip install -r requirements.txt`, start `gunicorn app:app` (or use `render.yaml`).
Pin the same scikit-learn version you trained with so the joblib files load.

## Using your trained models
Open `predict_with_model()` in `app.py` and make the DataFrame columns match your training features. Until then the simulator answers.
