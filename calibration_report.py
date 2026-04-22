"""
calibration_report.py
---------------------
Analyze accumulated predictions vs actuals to find systematic biases.

Reads all predictions/YYYY-MM-DD.json files that have results attached.

Usage:
  python calibration_report.py              # print report
  python calibration_report.py --send       # print + send to Telegram
  python calibration_report.py --days 14    # last 14 days only (default 30)
"""

import json
import argparse
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

PREDICTIONS_DIR = Path("predictions")
CONFIG_FILE     = Path("config.json")


def load_config() -> dict:
    defaults = {"telegram_token": "", "chat_id": ""}
    if CONFIG_FILE.exists():
        try:
            defaults.update(json.load(open(CONFIG_FILE)))
        except Exception:
            pass
    return defaults


def send_telegram(message: str, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  Telegram failed: {e}")
        return False


def load_results(days: int = 30) -> tuple[list, list]:
    """Load pitcher and batter results from the last N days."""
    cutoff    = datetime.now() - timedelta(days=days)
    pit_rows  = []
    bat_rows  = []

    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        try:
            date = datetime.strptime(f.stem, "%Y-%m-%d")
        except ValueError:
            continue
        if date < cutoff:
            continue

        try:
            data = json.load(open(f))
        except Exception:
            continue

        date_str = f.stem

        for r in data.get("pitcher_results", []):
            if r.get("pred_k") is not None and r.get("actual_k") is not None:
                r["date"] = date_str
                pit_rows.append(r)

        for r in data.get("batter_results", []):
            if r.get("actual_h") is not None:
                r["date"] = date_str
                bat_rows.append(r)

    return pit_rows, bat_rows


# ── PITCHER CALIBRATION ─────────────────────────────────

def analyze_pitchers(rows: list) -> dict:
    if not rows:
        return {}

    diffs   = [r["actual_k"] - r["pred_k"] for r in rows]
    abs_err = [abs(d) for d in diffs]
    hits    = [abs(d) <= 1.5 for d in diffs]

    # Accuracy by K range
    low  = [r for r in rows if r["pred_k"] <= 4.0]
    mid  = [r for r in rows if 4.0 < r["pred_k"] < 6.5]
    high = [r for r in rows if r["pred_k"] >= 6.5]

    def range_stats(subset):
        if not subset:
            return None
        ds = [r["actual_k"] - r["pred_k"] for r in subset]
        return {
            "n":      len(subset),
            "hit":    sum(abs(d) <= 1.5 for d in ds),
            "mae":    round(np.mean([abs(d) for d in ds]), 2),
            "bias":   round(np.mean(ds), 2),   # positive = we under-predict
        }

    # Systematic per-pitcher bias
    pitcher_errs = defaultdict(list)
    for r in rows:
        pitcher_errs[r["pitcher"]].append(r["actual_k"] - r["pred_k"])

    over_pred  = []  # we consistently over-predicted
    under_pred = []  # we consistently under-predicted
    for name, errs in pitcher_errs.items():
        if len(errs) < 2:
            continue
        avg = np.mean(errs)
        if avg <= -2.0:
            over_pred.append((name, round(avg, 1), len(errs)))
        elif avg >= 2.0:
            under_pred.append((name, round(avg, 1), len(errs)))

    over_pred.sort(key=lambda x: x[1])
    under_pred.sort(key=lambda x: x[1], reverse=True)

    return {
        "n":          len(rows),
        "hit_pct":    round(sum(hits) / len(hits) * 100, 1),
        "mae":        round(np.mean(abs_err), 2),
        "bias":       round(np.mean(diffs), 2),
        "low":        range_stats(low),
        "mid":        range_stats(mid),
        "high":       range_stats(high),
        "over_pred":  over_pred[:5],
        "under_pred": under_pred[:5],
    }


# ── BATTER CALIBRATION ──────────────────────────────────

def analyze_batters(rows: list) -> dict:
    if not rows:
        return {}

    def stat_stats(key_pred, key_actual, key_hit):
        subset = [r for r in rows if r.get(key_pred) is not None and r.get(key_actual) is not None]
        if not subset:
            return None
        diffs = [r[key_actual] - r[key_pred] for r in subset]
        return {
            "n":      len(subset),
            "hit":    sum(r.get(key_hit, False) for r in subset),
            "mae":    round(np.mean([abs(d) for d in diffs]), 3),
            "bias":   round(np.mean(diffs), 3),  # positive = we under-predict
        }

    h_stats  = stat_stats("pred_h",  "actual_h",  "hit_h")
    tb_stats = stat_stats("pred_tb", "actual_tb", "hit_tb")
    hr_stats = stat_stats("pred_hr", "actual_hr", "hit_hr")

    # HR calibration: group by pred_hr bucket, see actual HR rate
    hr_buckets = defaultdict(list)
    for r in rows:
        pred = r.get("pred_hr")
        actual = r.get("actual_hr")
        if pred is None or actual is None:
            continue
        bucket = round(pred * 10) / 10   # round to nearest 0.1
        hr_buckets[bucket].append(int(actual >= 1))

    hr_cal = []
    for bucket in sorted(hr_buckets):
        vals = hr_buckets[bucket]
        if len(vals) >= 3:
            hr_cal.append((bucket, round(np.mean(vals) * 100, 1), len(vals)))

    # Batters we consistently miss on H
    batter_errs = defaultdict(list)
    for r in rows:
        if r.get("pred_h") is not None and r.get("actual_h") is not None:
            batter_errs[r["batter"]].append(r["actual_h"] - r["pred_h"])

    over_h  = []
    under_h = []
    for name, errs in batter_errs.items():
        if len(errs) < 2:
            continue
        avg = np.mean(errs)
        if avg <= -0.4:
            over_h.append((name, round(avg, 2), len(errs)))
        elif avg >= 0.4:
            under_h.append((name, round(avg, 2), len(errs)))

    over_h.sort(key=lambda x: x[1])
    under_h.sort(key=lambda x: x[1], reverse=True)

    return {
        "n":          len(rows),
        "h":          h_stats,
        "tb":         tb_stats,
        "hr":         hr_stats,
        "hr_cal":     hr_cal,
        "over_h":     over_h[:4],
        "under_h":    under_h[:4],
    }


# ── REPORT BUILDER ──────────────────────────────────────

def build_report(pit: dict, bat: dict, days: int) -> str:
    lines = [f"<b>📊 MLB Model Calibration — Last {days} Days</b>\n"]

    # ── Pitchers ──
    if pit:
        bias_str = f"{pit['bias']:+.2f} K/game avg" if pit["bias"] else ""
        lines.append(f"<b>⚾ Pitcher K Model</b>")
        lines.append(f"   {pit['n']} predictions · {pit['hit_pct']}% within 1.5 K · MAE {pit['mae']} K")
        if bias_str:
            direction = "OVER-predicting" if pit["bias"] < -0.3 else ("UNDER-predicting" if pit["bias"] > 0.3 else "well-calibrated")
            lines.append(f"   Overall bias: {bias_str} ({direction})\n")

        # Accuracy by K range
        lines.append("   <i>Accuracy by prediction range:</i>")
        for label, stats in [("Low ≤4 K", pit["low"]), ("Mid 4-6.5 K", pit["mid"]), ("High ≥6.5 K", pit["high"])]:
            if stats:
                bias = f"bias {stats['bias']:+.1f}" if abs(stats['bias']) > 0.3 else "unbiased"
                lines.append(f"   {label}: {stats['hit']}/{stats['n']} ({round(stats['hit']/stats['n']*100)}%) MAE {stats['mae']} · {bias}")
        lines.append("")

        if pit["over_pred"]:
            lines.append("   <i>Pitchers we consistently over-predict:</i>")
            for name, avg, n in pit["over_pred"]:
                lines.append(f"   ⬇️ {name}: avg {avg:+.1f} K/start ({n} starts)")
            lines.append("")

        if pit["under_pred"]:
            lines.append("   <i>Pitchers we consistently under-predict:</i>")
            for name, avg, n in pit["under_pred"]:
                lines.append(f"   ⬆️ {name}: avg {avg:+.1f} K/start ({n} starts)")
            lines.append("")
    else:
        lines.append("<b>⚾ Pitcher K Model</b>\n   Not enough data yet.\n")

    # ── Batters ──
    if bat:
        lines.append(f"<b>🏏 Batter Model</b>")
        lines.append(f"   {bat['n']} batter-games analyzed")

        for label, stats, thresh in [
            ("H",  bat["h"],  0.50),
            ("TB", bat["tb"], 0.45),
            ("HR", bat["hr"], 0.65),
        ]:
            if stats:
                pct    = stats["hit"] / stats["n"] * 100
                grade  = "✅" if pct >= thresh * 100 else "⚠️"
                bias_d = "over-predicting" if stats["bias"] < -0.1 else ("under-predicting" if stats["bias"] > 0.1 else "calibrated")
                lines.append(f"   {grade} {label}: {stats['hit']}/{stats['n']} ({pct:.0f}%) · MAE {stats['mae']} · {bias_d}")
        lines.append("")

        # HR calibration
        if bat["hr_cal"]:
            lines.append("   <i>HR % calibration (pred% → actual hit rate):</i>")
            for pred_p, actual_p, n in bat["hr_cal"]:
                diff = actual_p - pred_p * 100
                arrow = "⬆️" if diff > 10 else ("⬇️" if diff < -10 else "≈")
                lines.append(f"   pred {int(pred_p*100)}% → actual {actual_p}% {arrow}  (n={n})")
            lines.append("")

        if bat["over_h"]:
            lines.append("   <i>Batters we over-predict on H:</i>")
            for name, avg, n in bat["over_h"]:
                lines.append(f"   ⬇️ {name}: avg {avg:+.2f} H/game ({n} games)")
            lines.append("")

        if bat["under_h"]:
            lines.append("   <i>Batters we under-predict on H:</i>")
            for name, avg, n in bat["under_h"]:
                lines.append(f"   ⬆️ {name}: avg {avg:+.2f} H/game ({n} games)")
    else:
        lines.append("<b>🏏 Batter Model</b>\n   Not enough data yet.")

    lines.append(f"\n⏰ {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Model Calibration Report")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze")
    parser.add_argument("--send", action="store_true",  help="Send report to Telegram")
    args   = parser.parse_args()
    config = load_config()

    print(f"\n📊 Calibration Report — Last {args.days} Days\n")

    pit_rows, bat_rows = load_results(days=args.days)
    print(f"  {len(pit_rows)} pitcher results loaded")
    print(f"  {len(bat_rows)} batter results loaded")

    if not pit_rows and not bat_rows:
        print("\n  No results data found yet — run daily_results.py for a few days first.")
        return

    pit_stats = analyze_pitchers(pit_rows)
    bat_stats = analyze_batters(bat_rows)

    report = build_report(pit_stats, bat_stats, args.days)

    import re
    print("\n" + re.sub(r"<[^>]+>", "", report))

    if args.send:
        ok = send_telegram(report, config["telegram_token"], config["chat_id"])
        print("\n  Sent to Telegram ✅" if ok else "\n  Telegram send failed ⚠️")


if __name__ == "__main__":
    main()
