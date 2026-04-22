"""
weekly_summary.py
-----------------
Reads the past 7 days of predictions from predictions/YYYY-MM-DD.json,
fetches actual pitcher K results from MLB Stats API, and sends a Telegram
recap showing how the predictions did for the week.

Run manually or schedule for Sunday night:
  python weekly_summary.py              # last 7 days
  python weekly_summary.py --no-send    # print only
  python weekly_summary.py --days 14    # last 14 days
"""

import json
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PREDICTIONS_DIR = Path("predictions")
CONFIG_FILE     = Path("config.json")


def load_config() -> dict:
    defaults = {"telegram_token": "", "chat_id": ""}
    if CONFIG_FILE.exists():
        saved = json.load(open(CONFIG_FILE))
        defaults.update(saved)
    return defaults


def send_telegram(message: str, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        print("  Add telegram_token and chat_id to config.json")
        return False
    try:
        r = requests.post(
            "https://api.telegram.org/bot{}/sendMessage".format(token),
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        print("  Sent to Telegram")
        return True
    except Exception as e:
        print("  Telegram failed: {}".format(e))
        return False


# ──────────────────────────────────────────────
# LOAD PREDICTIONS FOR DATE RANGE
# ──────────────────────────────────────────────

def load_week_predictions(days: int = 7) -> list:
    """
    Load pitcher predictions from the last N days.
    Returns list of dicts: {date, pitcher, pred_k, actual_k, ...}
    """
    today    = datetime.now().date()
    all_rows = []

    for i in range(days):
        date = today - timedelta(days=i + 1)  # yesterday and back
        date_str  = date.strftime("%Y-%m-%d")
        filepath  = PREDICTIONS_DIR / "{}.json".format(date_str)

        if not filepath.exists():
            continue

        try:
            data = json.load(open(filepath))
        except Exception:
            continue

        pitchers = data.get("pitchers", [])
        results  = {r["pitcher"]: r for r in data.get("pitcher_results", [])}

        for p in pitchers:
            name   = p.get("pitcher", "")
            pred_k = p.get("pred_k_total")
            result = results.get(name)

            if result is None:
                # Try last-name fuzzy match
                for rname, rval in results.items():
                    if name.split()[-1].lower() == rname.split()[-1].lower():
                        result = rval
                        break

            actual_k = result["actual_k"] if result else None

            all_rows.append({
                "date":      date_str,
                "pitcher":   name,
                "home_team": p.get("home_team", ""),
                "away_team": p.get("away_team", ""),
                "pred_k":    pred_k,
                "actual_k":  actual_k,
                "ip":        result.get("ip") if result else None,
            })

    return all_rows


# ──────────────────────────────────────────────
# FETCH MISSING ACTUALS FROM MLB STATS API
# ──────────────────────────────────────────────

def fetch_game_pks(date_str: str) -> list:
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": date_str},
            timeout=10,
        )
        r.raise_for_status()
        pks = []
        for date in r.json().get("dates", []):
            for game in date.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final":
                    pks.append(game["gamePk"])
        return pks
    except Exception:
        return []


def fetch_starter_results(game_pk: int) -> list:
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/game/{}/boxscore".format(game_pk),
            timeout=10,
        )
        r.raise_for_status()
        box = r.json()
    except Exception:
        return []

    results = []
    for side in ["home", "away"]:
        team_data = box.get("teams", {}).get(side, {})
        pitchers  = team_data.get("pitchers", [])
        if not pitchers:
            continue
        starter_id = pitchers[0]
        player     = team_data.get("players", {}).get("ID{}".format(starter_id), {})
        name       = player.get("person", {}).get("fullName", "")
        stats      = player.get("stats", {}).get("pitching", {})
        so         = stats.get("strikeOuts")
        ip         = stats.get("inningsPitched")
        if name and so is not None:
            results.append({"name": name, "SO": int(so), "IP": ip})
    return results


def fill_missing_actuals(rows: list) -> list:
    """
    For any row missing actual_k, try to fetch it from MLB Stats API.
    Groups by date to avoid redundant schedule lookups.
    """
    by_date = {}
    for row in rows:
        if row["actual_k"] is None and row["pred_k"] is not None:
            by_date.setdefault(row["date"], []).append(row)

    for date_str, date_rows in by_date.items():
        print("  Fetching actuals for {}...".format(date_str))
        pks = fetch_game_pks(date_str)
        actuals = {}
        for pk in pks:
            for res in fetch_starter_results(pk):
                actuals[res["name"]] = res

        for row in date_rows:
            name   = row["pitcher"]
            result = actuals.get(name)
            if result is None:
                for aname, ares in actuals.items():
                    if name.split()[-1].lower() == aname.split()[-1].lower():
                        result = ares
                        break
            if result:
                row["actual_k"] = result["SO"]
                row["ip"]       = result.get("IP")

    return rows


# ──────────────────────────────────────────────
# SCORE + ANALYSE
# ──────────────────────────────────────────────

def score_rows(rows: list) -> dict:
    """Compute weekly accuracy stats."""
    with_results = [r for r in rows if r["actual_k"] is not None and r["pred_k"] is not None]

    if not with_results:
        return {}

    diffs    = [abs(r["actual_k"] - r["pred_k"]) for r in with_results]
    hits     = [d <= 1.5 for d in diffs]
    raw_diffs = [r["actual_k"] - r["pred_k"] for r in with_results]

    # Per-pitcher summary (aggregate across multiple starts)
    by_pitcher = {}
    for r in with_results:
        p = r["pitcher"]
        by_pitcher.setdefault(p, []).append({
            "pred":   r["pred_k"],
            "actual": r["actual_k"],
            "diff":   r["actual_k"] - r["pred_k"],
        })

    pitcher_summaries = []
    for name, starts in by_pitcher.items():
        avg_pred   = np.mean([s["pred"]   for s in starts])
        avg_actual = np.mean([s["actual"] for s in starts])
        avg_diff   = np.mean([s["diff"]   for s in starts])
        n          = len(starts)
        n_hit      = sum(abs(s["diff"]) <= 1.5 for s in starts)
        pitcher_summaries.append({
            "name":       name,
            "n":          n,
            "avg_pred":   round(avg_pred, 1),
            "avg_actual": round(avg_actual, 1),
            "avg_diff":   round(avg_diff, 1),
            "hit_rate":   round(n_hit / n * 100),
        })

    # Sort: biggest miss first (for learning)
    pitcher_summaries.sort(key=lambda x: abs(x["avg_diff"]), reverse=True)

    return {
        "n_total":   len(with_results),
        "n_hits":    sum(hits),
        "hit_rate":  round(sum(hits) / len(hits) * 100, 1),
        "mae":       round(np.mean(diffs), 2),
        "bias":      round(np.mean(raw_diffs), 2),  # positive = we under-predict
        "pitchers":  pitcher_summaries,
    }


# ──────────────────────────────────────────────
# BUILD MESSAGE
# ──────────────────────────────────────────────

def build_weekly_message(rows: list, stats: dict, days: int) -> str:
    today    = datetime.now()
    end_date = (today - timedelta(days=1)).strftime("%b %d")
    start_date = (today - timedelta(days=days)).strftime("%b %d")

    lines = [
        "<b>Weekly K Prediction Report</b>",
        "<i>{} – {}</i>\n".format(start_date, end_date),
    ]

    if not stats:
        lines.append("No results found for this period.")
        return "\n".join(lines)

    n_total  = stats["n_total"]
    n_hits   = stats["n_hits"]
    hit_rate = stats["hit_rate"]
    mae      = stats["mae"]
    bias     = stats["bias"]

    bias_str = "we over-predict" if bias < 0 else "we under-predict"
    bias_str = "{} by {:.1f} K on avg".format(bias_str, abs(bias))

    lines.append(
        "Overall: <b>{}/{}</b> within 1.5 Ks ({:.0f}%)".format(
            n_hits, n_total, hit_rate
        )
    )
    lines.append("Avg error: <b>{} Ks</b>".format(mae))
    lines.append("Bias: <i>{}</i>\n".format(bias_str))

    # Per-pitcher breakdown — top 10 by volume, sorted by miss size
    lines.append("<b>Pitcher Breakdown</b>")
    for p in stats["pitchers"][:10]:
        diff_str  = "{:+.1f}".format(p["avg_diff"])
        n_str     = "{}x".format(p["n"]) if p["n"] > 1 else ""
        hit_str   = "{}%".format(p["hit_rate"])
        lines.append(
            "• <b>{}</b> {}\n"
            "  Pred: {}  Actual: {}  Diff: {}  Hit: {}".format(
                p["name"], n_str,
                p["avg_pred"], p["avg_actual"], diff_str, hit_str,
            )
        )

    # Best and worst calls
    with_results = [r for r in rows if r["actual_k"] is not None and r["pred_k"] is not None]
    if with_results:
        sorted_by_diff = sorted(with_results, key=lambda x: abs(x["actual_k"] - x["pred_k"]))
        best  = sorted_by_diff[:3]
        worst = sorted_by_diff[-3:][::-1]

        lines.append("\n<b>Best Calls</b>")
        for r in best:
            diff = r["actual_k"] - r["pred_k"]
            lines.append("  ✅ {} ({}) — Pred: {} Actual: {} ({:+.1f})".format(
                r["pitcher"], r["date"], r["pred_k"], r["actual_k"], diff
            ))

        lines.append("\n<b>Biggest Misses</b>")
        for r in worst:
            diff = r["actual_k"] - r["pred_k"]
            lines.append("  ❌ {} ({}) — Pred: {} Actual: {} ({:+.1f})".format(
                r["pitcher"], r["date"], r["pred_k"], r["actual_k"], diff
            ))

    lines.append("\n{}".format(today.strftime("%I:%M %p ET")))
    return "\n".join(lines)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly K Prediction Summary")
    parser.add_argument("--days",    type=int, default=7, help="Days to look back (default 7)")
    parser.add_argument("--no-send", action="store_true",  help="Print only, skip Telegram")
    args   = parser.parse_args()
    config = load_config()

    print("\nWeekly Summary — last {} days\n".format(args.days))

    # Load saved predictions + any already-fetched results
    print("  Loading predictions...")
    rows = load_week_predictions(args.days)
    if not rows:
        print("  No prediction files found in predictions/")
        return
    print("  {} pitcher entries loaded".format(len(rows)))

    # Fill in missing actuals from MLB Stats API
    missing = sum(1 for r in rows if r["actual_k"] is None and r["pred_k"] is not None)
    if missing:
        print("  {} entries missing actuals — fetching from MLB API...".format(missing))
        rows = fill_missing_actuals(rows)

    # Score
    stats = score_rows(rows)

    # Print table
    with_results = [r for r in rows if r["actual_k"] is not None and r["pred_k"] is not None]
    print("\n" + "=" * 65)
    if with_results:
        print("  {:<25} {:<12} {:>7} {:>8} {:>7}".format(
            "PITCHER", "DATE", "PRED K", "ACTUAL K", "DIFF"
        ))
        print("  " + "-" * 58)
        for r in sorted(with_results, key=lambda x: x["date"]):
            diff     = r["actual_k"] - r["pred_k"]
            hit_str  = "✓" if abs(diff) <= 1.5 else "✗"
            print("  {:<25} {:<12} {:>7.1f} {:>8} {:>+7.1f} {}".format(
                r["pitcher"], r["date"],
                r["pred_k"], r["actual_k"], diff, hit_str
            ))
        print("  " + "-" * 58)
        if stats:
            print("  Hit rate: {}/{} ({:.0f}%)   MAE: {} Ks   Bias: {:+.2f}".format(
                stats["n_hits"], stats["n_total"], stats["hit_rate"],
                stats["mae"], stats["bias"]
            ))
    else:
        print("  No completed results found for this period.")
    print("=" * 65 + "\n")

    message = build_weekly_message(rows, stats, args.days)

    if not args.no_send:
        send_telegram(message, config["telegram_token"], config["chat_id"])
    else:
        import re
        print("--- MESSAGE PREVIEW ---")
        print(re.sub(r"<[^>]+>", "", message))


if __name__ == "__main__":
    main()
