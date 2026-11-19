"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI trophy's earn rate
on Exophase at the case target, as five quantiles (5/25/50/75/95%).

Model (standard library only, deterministic, no network):
  rate_i(t) = E_i(t) / P(t), where P is Exophase's players-tracked count and
  E_i the number of tracked players who earned trophy i.  The two move for
  different reasons — P jumps when Exophase refreshes in a flood of new
  players, E_i grows as players reach that trophy — so they are extrapolated
  separately and combined, instead of extrapolating the rate directly.

  TREND mode (history moves): fit log P and log E_i against x = ln(1 + hours
  since the first read) over the last 24 h of reads (least squares), read both
  at the target, rate = E / P.  The log-time axis damps long-horizon
  extrapolation the way the house curve baseline does for rates.  Slopes are
  shrunk toward the cross-trophy median where a trophy's own fit is thin.

  FLAT mode (every read identical — launch night): no trend is visible, but
  the page is stale: Exophase had not refreshed for hours while the game
  released, so the next refresh dilutes every rate with new players.  Prior
  calibrated on the launch-night refresh the rules describe (638 -> 3,868
  players at 04:30) and on the live page fetched 2026-11-19 12:36 UTC:
  arrivals accumulate about linearly after release, P*(t) ~= P_last *
  (1 + 1.12 * t) with t hours since release, and a new player's chance of
  having earned trophy i is c_i * r_i, c_i growing with playtime and with
  how fast incumbents earned that trophy (its launch-night rate relative to
  the median).  The 100% trophy stays 100%.

Spread: logit-normal around the median, sigma from a random-walk term
(0.5 per root-day, as the house baseline) plus fit/prior uncertainty.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tuned constants (dev: backtest vs live Exophase page 2026-11-19 12:36 UTC
#     + practice refresh family; see ../gta6-trophy-dev/score.py) ---
SIGMA_RW = 0.30        # logit sigma per sqrt-day, random-walk term
# (dev: trend-sim median logit error ~0.32 at 4.5 d; D1 backtest median
#  error 0.12 at 0.38 d — total sigma ~0.64 at the final horizon fits
#  better than the house baseline's 0.5/sqrt-day)
SIGMA_PRIOR = 0.30    # extra logit sigma in FLAT mode (refresh uncertainty)
ARRIVAL_RATE = 0.80   # P multiplier slope: P*(t) = P_last * (1 + ARRIVAL_RATE * t)
C_PER_HOUR = 0.008    # new-player earn fraction per hour of playtime
C_GAMMA = 1.0         # c_i scales with (r_i / median r) ** C_GAMMA
C_CAP = 0.85          # cap on the new-player earn fraction
TREND_WINDOW_H = 24.0  # hours of history the trend fits use
SLOPE_SHRINK = 0.5    # shrink per-trophy E slopes toward the median slope
MAX_LOG_MOVE = 2.5    # cap on |log E| / |log P| move from last read


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg):
    return 100 / (1 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def linfit(xs, ys):
    """Least-squares y = a + b x. Returns (a, b, resid_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx == 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / n) if n > 2 else 0.0
    return a, b, sd


def median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the scored five")
    if not case["history"]:
        raise ValueError("no history")
    trophies = case["trophies"]
    n = len(trophies)
    hist = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    first = parse_time(hist[0]["at"])

    # Hours since first read, per read; x = ln(1 + h).
    hours = [(parse_time(r["at"]) - first).total_seconds() / 3600.0 for r in hist]
    xs = [math.log1p(max(h, 0.0)) for h in hours]
    x_target = math.log1p(max((target - first).total_seconds() / 3600.0, 0.0))
    # Reads inside the trend window.
    t_last = hours[-1]
    win = [k for k in range(len(hist)) if t_last - hours[k] <= TREND_WINDOW_H]
    if len(win) < 3:
        win = list(range(len(hist)))

    players = [r["players_tracked"] for r in hist]
    rates = [r["rates"] for r in hist]
    last_rates = rates[-1]
    p_last = players[-1]

    flat = (len(set(players)) == 1
            and len(set(tuple(r) for r in rates)) == 1)

    medians = []
    sigmas = []
    if flat:
        # --- FLAT mode: launch-dilution prior. ---
        release = case.get("game", {}).get("release_date", "2026-11-19")
        rel = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
        t_hours = max((target - rel).total_seconds() / 3600.0, 0.0)
        g = 1.0 + ARRIVAL_RATE * t_hours
        p_star = p_last * g
        r_med = median([r for r in last_rates if r is not None])
        c_base = min(C_CAP, C_PER_HOUR * t_hours)
        sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                          + SIGMA_PRIOR ** 2)
        for i in range(n):
            r = last_rates[i]
            if r is None:
                r = 0.0
            if r >= 99.99:
                medians.append(100.0)
                sigmas.append(0.05)
                continue
            e_old = r * p_last / 100.0
            c_i = min(C_CAP, c_base * (max(r, 0.01) / max(r_med, 0.01)) ** C_GAMMA)
            e_new = c_i * r * (p_star - p_last) / 100.0
            med = 100.0 * (e_old + e_new) / p_star
            medians.append(clamp(med))
            sigmas.append(sigma)
    else:
        # --- TREND mode: separate log-time fits for P and each E_i. ---
        xw = [xs[k] for k in win]
        # log P fit.
        lp = [math.log(max(players[k], 1)) for k in win]
        _, b_p, sd_p = linfit(xw, lp)
        log_p_star = math.log(max(p_last, 1)) + max(
            -MAX_LOG_MOVE, min(MAX_LOG_MOVE, b_p * (x_target - xs[-1])))
        p_star = math.exp(log_p_star)
        # Per-trophy log E fits.
        slopes = []
        fits = []
        for i in range(n):
            pts = [(xw[j], math.log(max((rates[win[j]][i] or 0.0)
                                        * players[win[j]] / 100.0, 0.05)))
                   for j in range(len(win)) if rates[win[j]][i] is not None]
            if len(pts) < 3:
                fits.append(None)
                slopes.append(0.0)
                continue
            a, b, sd = linfit([p[0] for p in pts], [p[1] for p in pts])
            fits.append((a, b, sd))
            slopes.append(b)
        med_slope = median(slopes)
        for i in range(n):
            r = last_rates[i]
            if r is None:
                r = 0.0
            if r >= 99.99:
                medians.append(100.0)
                sigmas.append(0.05)
                continue
            e_last = max(r * p_last / 100.0, 0.05)
            fit = fits[i]
            if fit is None:
                b_i = med_slope
                sd_i = 0.3
            else:
                _, b_raw, sd_i = fit
                b_i = med_slope + SLOPE_SHRINK * (b_raw - med_slope)
            dlog = max(-MAX_LOG_MOVE,
                       min(MAX_LOG_MOVE, b_i * (x_target - xs[-1])))
            e_star = e_last * math.exp(dlog)
            medians.append(clamp(100.0 * e_star / p_star))
            sig = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                            + (sd_i * 0.5) ** 2 + (sd_p * 0.5) ** 2)
            sigmas.append(max(0.15, min(2.0, sig)))

    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        qs = [round4(inv_logit(lg + z * sigmas[i])) for z in Z]
        qs = [min(100.0, max(0.0, q)) for q in qs]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
