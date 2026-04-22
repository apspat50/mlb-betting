"""
daily_results.py
----------------
Fetch actual game results, compare to saved predictions, send Telegram recap.

Run after games finish (~11 PM ET or next morning).

Usage:
  python daily_results.py              # check today's results
  python daily_results.py --date 2025-04-09   # specific date
  python daily_results.py --no-send    # print only
"""

import json
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PREDICTIONS_DIR = Path("predictions")
CONFIG_FILE     = Path("config.json")


# ======================================================
# CONFIG + TELEGRAM
# ======================================================

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


# ======================================================
# LOAD PREDICTIONS FILE
# ======================================================

def load_predictions(date_str: str) -> dict:
    filepath = PREDICTIONS_DIR / f"{date_str}.json"
    if not filepath.exists():
        print("  No predictions file found for {}".format(date_str))
        return {}
    return json.load(open(filepath))


# ======================================================
# FETCH ACTUAL RESULTS FROM MLB STATS API
# ======================================================

def fetch_game_pks(date_str: str) -> list:
    """Get list of game PKs for a date."""
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
                status = game.get("status", {}).get("abstractGameState", "")
                if status == "Final":
                    pks.append(game["gamePk"])
        return pks
    except Exception as e:
        print("  Schedule fetch failed: {}".format(e))
        return []


def fetch_pitcher_results(game_pk: int) -> list:
    """
    Fetch boxscore for one game.
    Returns list of dicts: {name, team, SO, IP, result} for each starting pitcher.
    """
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/game/{}/boxscore".format(game_pk),
            timeout=10,
        )
        r.raise_for_status()
        box = r.json()
    except Exception as e:
        print("  Boxscore fetch failed for game {}: {}".format(game_pk, e))
        return []

    results = []
    for side in ["home", "away"]:
        team_data = box.get("teams", {}).get(side, {})
        team_name = team_data.get("team", {}).get("name", "")
        pitchers  = team_data.get("pitchers", [])

        if not pitchers:
            continue

        # First pitcher = starter
        starter_id = pitchers[0]
        players    = team_data.get("players", {})
        player_key = "ID{}".format(starter_id)
        player     = players.get(player_key, {})

        name  = player.get("person", {}).get("fullName", "Unknown")
        stats = player.get("stats", {}).get("pitching", {})
        so    = stats.get("strikeOuts", None)
        ip    = stats.get("inningsPitched", None)
        er    = stats.get("earnedRuns", None)
        note  = stats.get("note", "")  # e.g. "W", "L", "ND"

        results.append({
            "name":  name,
            "team":  team_name,
            "SO":    int(so) if so is not None else None,
            "IP":    ip,
            "er":    int(er) if er is not None else None,
            "note":  note,
        })

    return results


def fetch_all_results(date_str: str) -> dict:
    """
    Fetch actual pitching results for all completed games on date_str.
    Returns dict: pitcher_name -> result dict
    """
    pks = fetch_game_pks(date_str)
    if not pks:
        print("  No completed games found for {}".format(date_str))
        return {}

    print("  Fetching results for {} completed games...".format(len(pks)))
    all_results = {}
    for pk in pks:
        for res in fetch_pitcher_results(pk):
            if res["name"]:
                all_results[res["name"]] = res

    return all_results


def fetch_batter_results(date_str: str) -> dict:
    """
    Fetch actual batter stats (H, TB, HR) from boxscores.
    Returns dict: player_name -> {H, TB, HR}
    """
    pks = fetch_game_pks(date_str)
    if not pks:
        return {}

    all_batters = {}
    for pk in pks:
        try:
            r = requests.get(
                "https://statsapi.mlb.com/api/v1/game/{}/boxscore".format(pk),
                timeout=10,
            )
            r.raise_for_status()
            box = r.json()
        except Exception:
            continue

        for side in ["home", "away"]:
            team_data = box.get("teams", {}).get(side, {})
            batters   = team_data.get("batters", [])
            players   = team_data.get("players", {})

            for batter_id in batters:
                player = players.get("ID{}".format(batter_id), {})
                name   = player.get("person", {}).get("fullName", "")
                stats  = player.get("stats", {}).get("batting", {})
                if not name or not stats:
                    continue

                ab = int(stats.get("atBats",   0) or 0)
                if ab == 0:
                    continue  # pinch runner, etc.

                h  = int(stats.get("hits",     0) or 0)
                d  = int(stats.get("doubles",  0) or 0)
                t  = int(stats.get("triples",  0) or 0)
                hr = int(stats.get("homeRuns", 0) or 0)
                tb = h + d + 2 * t + 3 * hr  # singles=1, doubles=2, triples=3, hr=4

                all_batters[name] = {"H": h, "TB": tb, "HR": hr}

    return all_batters


# ======================================================
# COMPARE + SCORE
# ======================================================

def compare_predictions(predictions: list, actuals: dict) -> list:
    """
    Match each prediction to actual result.
    Returns list of comparison dicts.
    """
    compared = []
    for pred in predictions:
        name   = pred.get("pitcher", "")
        actual = actuals.get(name)

        if actual is None:
            # Try partial name match
            for aname, ares in actuals.items():
                if name.split()[-1].lower() == aname.split()[-1].lower():
                    actual = ares
                    break

        pred_k  = pred.get("pred_k_total")
        actual_k = actual["SO"] if actual and actual["SO"] is not None else None

        if pred_k is not None and actual_k is not None:
            diff   = actual_k - pred_k
            pct_err = abs(diff) / max(pred_k, 1) * 100
            hit    = abs(diff) <= 1.5   # within 1.5 Ks = "hit"
        else:
            diff     = None
            pct_err  = None
            hit      = None

        compared.append({
            "pitcher":   name,
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "pred_k":    pred_k,
            "actual_k":  actual_k,
            "diff":      round(diff, 1) if diff is not None else None,
            "pct_err":   round(pct_err, 1) if pct_err is not None else None,
            "hit":       hit,
            "ip":        actual["IP"] if actual else None,
            "game_note": actual["note"] if actual else None,
        })

    return compared


# ======================================================
# BATTER COMPARE + MESSAGE
# ======================================================

def compare_batter_predictions(batter_preds: list, actuals: dict) -> list:
    compared = []
    for pred in batter_preds:
        name   = pred.get("batter", "")
        actual = actuals.get(name)

        if actual is None:
            for aname, ares in actuals.items():
                if name.split()[-1].lower() == aname.split()[-1].lower():
                    actual = ares
                    break

        pred_h  = pred.get("pred_h")
        pred_tb = pred.get("pred_tb")
        pred_hr = pred.get("pred_hr")

        actual_h  = actual["H"]  if actual else None
        actual_tb = actual["TB"] if actual else None
        actual_hr = actual["HR"] if actual else None

        def _diff(p, a): return round(a - p, 2) if p is not None and a is not None else None
        def _hit_h(d):  return abs(d) <= 0.5 if d is not None else None
        def _hit_tb(d): return abs(d) <= 1.0 if d is not None else None
        def _hit_hr(d): return abs(d) <= 0.3 if d is not None else None

        dh  = _diff(pred_h,  actual_h)
        dtb = _diff(pred_tb, actual_tb)
        dhr = _diff(pred_hr, actual_hr)

        compared.append({
            "batter":    name,
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "pred_h":    pred_h,   "actual_h":  actual_h,  "diff_h":  dh,  "hit_h":  _hit_h(dh),
            "pred_tb":   pred_tb,  "actual_tb": actual_tb, "diff_tb": dtb, "hit_tb": _hit_tb(dtb),
            "pred_hr":   pred_hr,  "actual_hr": actual_hr, "diff_hr": dhr, "hit_hr": _hit_hr(dhr),
        })
    return compared


def build_batter_results_message(compared: list, date_str: str) -> str:
    date_fmt = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
    lines = ["\n<b>Results: MLB Batter Props — {}</b>".format(date_fmt)]

    has  = [c for c in compared if c["actual_h"] is not None]
    if not has:
        lines.append("No batter results found yet.")
        return "\n".join(lines)

    h_hits  = sum(1 for c in has if c["hit_h"])
    tb_hits = sum(1 for c in has if c["hit_tb"])
    hr_hits = sum(1 for c in has if c["hit_hr"])
    n       = len(has)

    lines.append(
        "H: <b>{}/{}</b> ({:.0f}%)  TB: <b>{}/{}</b> ({:.0f}%)  HR: <b>{}/{}</b> ({:.0f}%)\n".format(
            h_hits, n, h_hits/n*100,
            tb_hits, n, tb_hits/n*100,
            hr_hits, n, hr_hits/n*100,
        )
    )

    has.sort(key=lambda x: x.get("pred_tb") or 0, reverse=True)
    for c in has[:15]:
        def _fmt(pred, actual, hit):
            if pred is None or actual is None:
                return "n/a"
            diff = actual - pred
            mark = "✅" if hit else "❌"
            return "{} (pred {:.1f}) {}".format(actual, pred, mark)

        lines.append(
            "<b>{}</b>\n"
            "   {} @ {}\n"
            "   H: {}  TB: {}  HR: {}\n".format(
                c["batter"],
                c["away_team"], c["home_team"],
                _fmt(c["pred_h"],  c["actual_h"],  c["hit_h"]),
                _fmt(c["pred_tb"], c["actual_tb"], c["hit_tb"]),
                _fmt(c["pred_hr"], c["actual_hr"], c["hit_hr"]),
            )
        )
    return "\n".join(lines)


# ======================================================
# BUILD TELEGRAM MESSAGE
# ======================================================

def build_results_message(compared: list, date_str: str) -> str:
    date_fmt = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
    lines = ["<b>Results: MLB Pitcher K Props — {}</b>".format(date_fmt)]

    has_results = [c for c in compared if c["actual_k"] is not None]
    no_results  = [c for c in compared if c["actual_k"] is None]

    if not has_results:
        lines.append("No completed game results found yet.")
        return "\n".join(lines)

    # Sort by predicted Ks descending
    has_results.sort(key=lambda x: x["pred_k"] or 0, reverse=True)

    hits   = sum(1 for c in has_results if c["hit"])
    total  = len(has_results)
    hit_pct = hits / total * 100 if total > 0 else 0

    diffs  = [abs(c["diff"]) for c in has_results if c["diff"] is not None]
    mae    = round(np.mean(diffs), 2) if diffs else None

    lines.append("Score: <b>{}/{}</b> within 1.5 Ks ({:.0f}%)".format(hits, total, hit_pct))
    if mae is not None:
        lines.append("Avg error: <b>{} Ks</b>\n".format(mae))

    for c in has_results:
        pred_k   = c["pred_k"]
        actual_k = c["actual_k"]
        diff     = c["diff"]

        if diff is None:
            diff_str = "n/a"
        elif diff > 0:
            diff_str = "+{:.1f} ✅".format(diff) if c["hit"] else "+{:.1f} ❌".format(diff)
        elif diff < 0:
            diff_str = "{:.1f} ✅".format(diff) if c["hit"] else "{:.1f} ❌".format(diff)
        else:
            diff_str = "0.0 ✅"

        ip_str   = " ({} IP)".format(c["ip"]) if c["ip"] else ""
        note_str = " [{}]".format(c["game_note"]) if c["game_note"] else ""

        lines.append(
            "<b>{}</b>{}{}\n"
            "   {} @ {}\n"
            "   Pred: {} K  Actual: <b>{} K</b>  {}\n".format(
                c["pitcher"], ip_str, note_str,
                c["away_team"], c["home_team"],
                pred_k, actual_k, diff_str,
            )
        )

    if no_results:
        lines.append("<i>No result: {}</i>".format(
            ", ".join(c["pitcher"] for c in no_results)
        ))

    lines.append("{}".format(datetime.now().strftime("%I:%M %p ET")))
    return "\n".join(lines)


# ======================================================
# SAVE RESULTS BACK TO FILE
# ======================================================

def save_results(compared_pitchers: list, date_str: str,
                 compared_batters: list = None):
    filepath = PREDICTIONS_DIR / f"{date_str}.json"
    existing = {}
    if filepath.exists():
        try:
            existing = json.load(open(filepath))
        except Exception:
            existing = {}

    existing["pitcher_results"]    = compared_pitchers
    if compared_batters is not None:
        existing["batter_results"] = compared_batters
    existing["results_fetched_at"] = datetime.now().isoformat()

    json.dump(existing, open(filepath, "w"), indent=2, default=str)
    print("  Results saved → {}".format(filepath))


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Daily Results Check")
    parser.add_argument("--date",    default=None, help="Date YYYY-MM-DD (default today)")
    parser.add_argument("--no-send", action="store_true")
    args   = parser.parse_args()
    config = load_config()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print("\nResults Check — {}\n".format(date_str))

    # Load predictions
    preds_data = load_predictions(date_str)
    pitcher_preds = preds_data.get("pitchers", [])
    if not pitcher_preds:
        print("  No pitcher predictions found for {}.".format(date_str))
        print("  Run daily_picks.py first.")
        return
    print("  {} pitcher predictions loaded".format(len(pitcher_preds)))

    # Fetch actual results
    print("\n  Fetching actual game results from MLB Stats API...")
    actuals = fetch_all_results(date_str)
    if not actuals:
        print("  No results available yet — games may still be in progress.")
        return
    print("  {} starters with results found".format(len(actuals)))

    # Compare pitchers
    compared = compare_predictions(pitcher_preds, actuals)

    # Batter results
    batter_preds    = preds_data.get("batters", [])
    compared_batters = []
    if batter_preds:
        print("\n  Fetching batter results from boxscores...")
        batter_actuals  = fetch_batter_results(date_str)
        compared_batters = compare_batter_predictions(batter_preds, batter_actuals)
        print("  {} batter results found".format(
            sum(1 for c in compared_batters if c["actual_h"] is not None)
        ))

    save_results(compared, date_str, compared_batters if batter_preds else None)

    # Print pitcher table
    has_results = [c for c in compared if c["actual_k"] is not None]
    print("\n" + "=" * 60)
    if has_results:
        print("  {:<25} {:>7} {:>8} {:>7}".format("PITCHER", "PRED K", "ACTUAL K", "DIFF"))
        print("  " + "-" * 50)
        for c in sorted(has_results, key=lambda x: x["pred_k"] or 0, reverse=True):
            diff_str = "{:+.1f}".format(c["diff"]) if c["diff"] is not None else "n/a"
            hit_str  = "✓" if c.get("hit") else "✗"
            print("  {:<25} {:>7.1f} {:>8} {:>7} {}".format(
                c["pitcher"], c["pred_k"] or 0, c["actual_k"], diff_str, hit_str
            ))
        hits  = sum(1 for c in has_results if c["hit"])
        total = len(has_results)
        diffs = [abs(c["diff"]) for c in has_results if c["diff"] is not None]
        print("  " + "-" * 50)
        print("  Hit rate: {}/{} ({:.0f}%)   MAE: {:.2f} Ks".format(
            hits, total, hits / total * 100 if total else 0,
            np.mean(diffs) if diffs else 0
        ))
    print("=" * 60)

    # Print batter table
    has_batters = [c for c in compared_batters if c["actual_h"] is not None]
    if has_batters:
        print("\n  {:<25} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
            "BATTER", "pH", "aH", "pTB", "aTB", "pHR", "aHR"
        ))
        print("  " + "-" * 58)
        for c in has_batters[:15]:
            print("  {:<25} {:>6.1f} {:>6} {:>6.1f} {:>6} {:>6.2f} {:>6}  {}{}{}".format(
                c["batter"],
                c["pred_h"]  or 0, c["actual_h"]  or 0,
                c["pred_tb"] or 0, c["actual_tb"] or 0,
                c["pred_hr"] or 0, c["actual_hr"] or 0,
                "H✓" if c["hit_h"]  else "H✗",
                " TB✓" if c["hit_tb"] else " TB✗",
                " HR✓" if c["hit_hr"] else " HR✗",
            ))
    print()

    pitcher_msg = build_results_message(compared, date_str)
    batter_msg  = build_batter_results_message(compared_batters, date_str) if compared_batters else ""

    if not args.no_send:
        send_telegram(pitcher_msg, config["telegram_token"], config["chat_id"])
        if batter_msg:
            send_telegram(batter_msg, config["telegram_token"], config["chat_id"])
    else:
        import re
        clean = lambda m: re.sub(r"<[^>]+>", "", m)
        print("--- PITCHER MESSAGE ---")
        print(clean(pitcher_msg))
        if batter_msg:
            print("--- BATTER MESSAGE ---")
            print(clean(batter_msg))


if __name__ == "__main__":
    main()
