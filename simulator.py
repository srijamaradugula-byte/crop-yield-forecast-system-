"""Python port of the browser simulator (NDVI -> stress -> yield).
Used as a fallback when trained models are not present in models/."""
import math

CROPS = {
    "Paddy": {"b": 27, "p": 2183},
    "Maize": {"b": 25, "p": 2225},
    "Chilli": {"b": 11.5, "p": 13000},
    "Groundnut": {"b": 8.6, "p": 6783},
    "Cotton": {"b": 8, "p": 7121},
}
REL = {0: 0.43, 30: 0.22, 60: 0.13, 90: 0.13}  # 90% interval half-width (fraction of yield)
ACC = {  # stage: (R2, RMSE, MAPE%)
    0: (0.784, 2.69, 27.6),
    30: (0.928, 1.55, 14.6),
    60: (0.961, 1.15, 9.6),
    90: (0.962, 1.13, 9.7),
}


def clip(x, a, b):
    return min(b, max(a, x))


def stage_of(days):
    return 0 if days < 30 else 30 if days < 60 else 60 if days < 90 else 90


def predict(p):
    """p: dict with days, crop, area, seed, irr, delay, n30, n60, n90, pest (None allowed)."""
    st = stage_of(math.floor(p["days"]))
    cp = CROPS[p["crop"]]
    n30, n60, n90 = p.get("n30"), p.get("n60"), p.get("n90")
    first = lambda *xs: next((x for x in xs if x is not None), None)
    nd = (first(n90, n60, n30) if st == 90 else
          first(n60, n30) if st == 60 else
          n30 if st == 30 else None)

    if nd is not None:
        stress = clip((nd - 0.18) / (0.62 * 0.92), 0.05, 1.1) ** (1 / 0.7)
    else:
        stress = 0.58
    if st == 0 or nd is None:
        stress *= ((1.04 if p["seed"] == "Certified" else 0.94)
                   * (1 - 0.006 * max(p.get("delay") or 0, 0))
                   * (0.9 if p["irr"] == "Rainfed" else 1))
    if st >= 60 and p.get("pest") is not None:
        stress *= 1 - 0.0004 * p["pest"]

    y = cp["b"] * stress
    half = y * REL[st]
    lo, hi = max(y - half, 0), y + half
    tot = y * p["area"]
    r2, rmse, mape = ACC[st]
    return {
        "stage": st, "yield": round(y, 2), "low": round(lo, 2), "high": round(hi, 2),
        "total": round(tot, 1), "price": cp["p"], "revenue": round(tot * cp["p"]),
        "r2": r2, "rmse": rmse, "mape": mape, "engine": "simulator",
    }
