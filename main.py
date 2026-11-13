"""Quantile forecast for Polymarket's Next-CEO event: per-market volatility bands blended with an
empirical change-quantile model. Flat markets get degenerate quantiles, so quiet minutes are free.
"""
import json
import math
import os
import statistics

IN, OUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input"), os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
Q = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
HS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
K, WIN, MULT = 0.4, 240, 0.5
SHRINK, TAIL, BLEND, FLOOR, HORIZON = 0.2, 0.35, 0.2, 0.001, 240
W_VOL = 0.8


def clean(prices):
    out = [p for p in prices if isinstance(p, (int, float)) and math.isfinite(p)]
    return [min(1.0, max(0.0, p)) for p in out]


def qtl(v, q):
    i = q * (len(v) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo)


def vol_rows(prices):
    """Bands from the market's own recent volatility; flat markets collapse to the last price."""
    x = prices[-1]
    recent = prices[-WIN:]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    moving = [abs(d) for d in diffs if d != 0]
    if not moving:
        return [[round(x, 6)] * 5 for _ in range(HORIZON)]
    sigma = max(statistics.pstdev(diffs), statistics.median(moving) * MULT)
    rows = []
    for h in range(1, HORIZON + 1):
        spread = K * sigma * math.sqrt(h)
        rows.append(sorted(min(1.0, max(0.0, round(x + z * spread, 6))) for z in Z))
    return rows


def change_table(prices):
    tab = {}
    for h in HS:
        if len(prices) > h + 5:
            d = sorted(prices[i + h] - prices[i] for i in range(len(prices) - h))
            tab[h] = [qtl(d, q) for q in Q]
    return tab or {1: [0.0] * 5}


def rows_for(x, tab, shrink, floor):
    """Build horizon rows from empirical change quantiles and a volatility floor.
    Each row is shrunk, blended, clamped to [0, 1], and rounded."""
    keys = sorted(tab)
    rows = []
    for h in range(1, HORIZON + 1):
        lo = max([k for k in keys if k <= h], default=keys[0])
        hi = min([k for k in keys if k >= h], default=keys[-1])
        w = 0 if lo == hi else (h - lo) / (hi - lo)
        emp = [tab[lo][i] if lo == hi else tab[lo][i] + (tab[hi][i] - tab[lo][i]) * w for i in range(5)]
        band = [z * floor * math.sqrt(h) for z in Z]
        mix = [emp[i] * shrink[i] * (1 - BLEND) + band[i] * BLEND for i in range(5)]
        row = [x + min(mix[0], band[0]), x + min(mix[1], band[1]), x + mix[2],
               x + max(mix[3], band[3]), x + max(mix[4], band[4])]
        rows.append(sorted(min(1.0, max(0.0, round(v, 6))) for v in row))
    return rows


def empirical_rows(prices):
    """Quantiles of the market's own h-step changes, shrunk, with a small volatility floor."""
    x = prices[-1]
    tab = change_table(prices)
    floor = FLOOR * math.sqrt(max(1e-6, x * (1 - x)))
    shrink = [TAIL, TAIL, SHRINK, TAIL, TAIL]
    return rows_for(x, tab, shrink, floor)


def forecast(prices):
    prices = clean(prices)
    if not prices:
        return [[0.45, 0.475, 0.5, 0.525, 0.55] for _ in range(HORIZON)]
    a = vol_rows(prices)
    x = prices[-1]
    tab = change_table(prices)
    floor = FLOOR * math.sqrt(max(1e-6, x * (1 - x)))
    shrink = [TAIL, TAIL, SHRINK, TAIL, TAIL]
    b = rows_for(x, tab, shrink, floor)
    rows = []
    for h in range(HORIZON):
        blended = [a[h][i] * W_VOL + b[h][i] * (1 - W_VOL) for i in range(5)]
        rows.append(sorted(min(1.0, max(0.0, round(v, 6))) for v in blended))
    return rows


def load_case(path):
    try:
        with open(path) as f:
            return json.load(f).get("markets") or []
    except Exception:
        return []


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sorted(n for n in os.listdir(IN) if n.endswith(".in")):
        markets = load_case(os.path.join(IN, name))
        out = [{"id": m["id"], "q": forecast(m.get("prices") or [])} for m in markets]
        with open(os.path.join(OUT, name[:-3] + ".out"), "w") as f:
            f.write(json.dumps({"forecasts": out}, separators=(",", ":")))


if __name__ == "__main__":
    main()
