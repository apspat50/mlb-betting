"""
daily_picks.py
--------------
Daily props picker — runs every morning and sends to Telegram.

What it does:
  1. Fetches today's MLB games and starting pitchers (ESPN, free)
  2. Runs pitcher K props model on each starter
  3. Shows predicted K/9 rate — compare to your FanDuel line manually
  4. Sends to Telegram

Setup (one time):
  1. Create Telegram bot via @BotFather → /newbot
  2. Copy token, message your bot "hello"
  3. Visit https://api.telegram.org/botTOKEN/getUpdates to get chat_id
  4. Add both to config.json

Usage:
  python daily_picks.py              # today's picks + send to Telegram
  python daily_picks.py --no-send    # print only, don't send
  python daily_picks.py --test       # test Telegram connection
  python daily_picks.py --date 2025-04-15  # specific date
"""

import json
import warnings
import argparse
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

CONFIG_FILE = Path("config.json")
MODELS_DIR  = Path("saved_models") / "props"
DATA_DIR    = Path("props_data")


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════

def load_config() -> dict:
    defaults = {
        "telegram_token": "",
        "chat_id":        "",
        "bankroll":       1000,
        "min_edge":       0.06,
    }
    if CONFIG_FILE.exists():
        saved = json.load(open(CONFIG_FILE))
        defaults.update(saved)
    else:
        json.dump(defaults, open(CONFIG_FILE, "w"), indent=2)
        print("  ✅ Created config.json — add your Telegram token and chat_id")
    return defaults


# ══════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════

def send_telegram(message: str, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        print("  ⚠️  Add telegram_token and chat_id to config.json")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        print("  ✅ Sent to Telegram")
        return True
    except Exception as e:
        print(f"  ⚠️  Telegram failed: {e}")
        return False


def test_telegram(config: dict):
    msg = (f"🤖 MLB Props Bot connected!\n"
           f"Time: {datetime.now().strftime('%I:%M %p')}\n"
           f"Ready to send daily pitcher K predictions.")
    ok = send_telegram(msg, config["telegram_token"], config["chat_id"])
    if ok:
        print("  ✅ Check your phone!")
    else:
        print("  ❌ Failed — check config.json")


# ══════════════════════════════════════════════
# ESPN GAME SCHEDULE (free, no API key)
# ══════════════════════════════════════════════

def fetch_todays_games(date_str: str = None) -> pd.DataFrame:
    """Fetch today's MLB schedule and probable starters from ESPN."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            params={"dates": date_str},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠️  ESPN failed: {e}")
        return pd.DataFrame()

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        teams = comp.get("competitors", [])
        home  = next((t for t in teams if t.get("homeAway") == "home"), {})
        away  = next((t for t in teams if t.get("homeAway") == "away"), {})

        # Game time
        try:
            gt = datetime.strptime(event.get("date",""), "%Y-%m-%dT%H:%MZ")
            gt_et = gt - timedelta(hours=4)
            time_str = gt_et.strftime("%I:%M %p ET").lstrip("0")
        except:
            time_str = "TBD"

        # Probable starters
        home_sp = "TBD"
        away_sp = "TBD"
        for comp2 in comp.get("competitors", []):
            for prob in comp2.get("probables", []):
                name = prob.get("athlete", {}).get("displayName", "TBD")
                if comp2.get("homeAway") == "home":
                    home_sp = name
                else:
                    away_sp = name

        games.append({
            "time":      time_str,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_sp":   home_sp,
            "away_sp":   away_sp,
            "venue":     comp.get("venue", {}).get("fullName", ""),
            "status":    event.get("status", {}).get("type", {}).get("description", ""),
        })

    return pd.DataFrame(games) if games else pd.DataFrame()


# ══════════════════════════════════════════════
# PITCHER K PREDICTIONS
# ══════════════════════════════════════════════

def run_k_predictions(games: pd.DataFrame) -> list:
    """
    Run pitcher strikeout model on today's starters.
    Returns list of prediction dicts.
    """
    model_path = MODELS_DIR / "pitcher_strikeouts.pkl"
    if not model_path.exists():
        print("  ⚠️  No K model found. Run: python props_model.py --train first")
        return []

    try:
        model_pkg = joblib.load(model_path)
    except Exception as e:
        print(f"  ⚠️  Could not load K model: {e}")
        return []

    try:
        from props_model import (
            build_pitcher_k_features,
            load_pitching_logs,
            load_team_batting_stats,
            predict_stat,
        )
        from statcast_logs import fetch_todays_pitcher_starts

        seasons  = list(range(2014, datetime.now().year + 1))
        pit_logs = load_pitching_logs(seasons)
        team_bat = load_team_batting_stats(seasons)

        # Get all pitchers scheduled today
        todays_pitchers = []
        for _, g in games.iterrows():
            if g.get("home_sp") and g["home_sp"] != "TBD":
                todays_pitchers.append(g["home_sp"])
            if g.get("away_sp") and g["away_sp"] != "TBD":
                todays_pitchers.append(g["away_sp"])

        # Fetch only recent starts for today's pitchers — fast, ~30 seconds
        print(f"  Fetching recent Statcast starts for {len(todays_pitchers)} pitchers...")
        start_logs = fetch_todays_pitcher_starts(todays_pitchers, days_back=45)

        if start_logs.empty:
            print("  ⚠️  No Statcast data — using season averages")
            start_logs = None

    except Exception as e:
        print(f"  ⚠️  Could not load pitcher data: {e}")
        return []

    predictions = []
    today = datetime.now()

    for _, game in games.iterrows():
        for pitcher, opp_team, home_team, role in [
            (game["home_sp"], game["away_team"], game["home_team"], "Home"),
            (game["away_sp"], game["home_team"], game["home_team"], "Away"),
        ]:
            if not pitcher or pitcher == "TBD":
                continue

            try:
                feats        = build_pitcher_k_features(
                    pitcher, today, pit_logs, opp_team, team_bat, home_team,
                    start_logs=start_logs,
                )
                # predict_stat returns per-start K rate (e.g. 6.5 Ks per start)
                pred_per_start = predict_stat(model_pkg, feats)

                # Sanity check — if value > 20 it's a season total not per-start
                # Convert it: avg pitcher makes ~30 starts, divide to get per-start
                if pred_per_start > 20:
                    pred_per_start = pred_per_start / 30.0

                # K/9 rate for display
                avg_ip = max(feats.get("p_ip_L3", 5.5) or 5.5, 1.0)
                pred_k9 = pred_per_start / avg_ip * 9

                predictions.append({
                    "pitcher":       pitcher,
                    "role":          role,
                    "home_team":     game["home_team"],
                    "away_team":     game["away_team"],
                    "time":          game["time"],
                    "opp_team":      opp_team,
                    "pred_k9":       round(pred_k9, 1),
                    "pred_k_total":  round(pred_per_start, 1),
                    "avg_ip":        round(avg_ip, 1),
                    "form_z":        round(feats.get("p_form_z", 0) or 0, 2),
                    "opp_kpct":      round(feats.get("opp_kpct", 0.22) or 0.22, 3),
                })
            except Exception as e:
                continue

    return predictions


# ══════════════════════════════════════════════
# BUILD MESSAGE
# ══════════════════════════════════════════════

def build_message(games: pd.DataFrame,
                   predictions: list,
                   config: dict) -> str:
    today    = datetime.now().strftime("%B %d, %Y")
    bankroll = config.get("bankroll", 1000)

    lines = [f"<b>⚾ MLB Pitcher K Props — {today}</b>"]

    if predictions:
        lines.append(f"\n<b>🔮 K Predictions ({len(predictions)} starters)</b>")
        lines.append("<i>Compare predicted Ks to your FanDuel line</i>\n")

        # Sort by predicted K total descending
        predictions.sort(key=lambda x: x["pred_k_total"], reverse=True)

        for p in predictions:
            # Form indicator
            form = p.get("form_z", 0)
            if form > 1.5:
                form_str = " 🔥 hot"
            elif form < -1.5:
                form_str = " 🥶 cold"
            else:
                form_str = ""

            # Opponent K vulnerability
            opp_k = p.get("opp_kpct", 0.22)
            if opp_k > 0.26:
                opp_str = " (opp Ks a lot)"
            elif opp_k < 0.18:
                opp_str = " (opp patient)"
            else:
                opp_str = ""

            lines.append(
                f"📊 <b>{p['pitcher']}</b> ({p['role']}){form_str}"
                f"\n   {p['away_team']} @ {p['home_team']} | {p['time']}"
                f"\n   Predicted: <b>{p['pred_k_total']} Ks</b> "
                f"({p['pred_k9']} K/9 × {p['avg_ip']} IP){opp_str}"
                f"\n   👉 Check FanDuel K line — bet OVER if line is lower\n"
            )
    else:
        lines.append("\n❌ No predictions available")
        lines.append("Make sure props model is trained:")
        lines.append("python props_model.py --train")

    # Games summary
    if not games.empty:
        lines.append(f"\n<b>📅 {len(games)} Games Today</b>")
        for _, g in games.iterrows():
            lines.append(
                f"  {g['time']:>9}  "
                f"{g['away_team'][:14]} @ {g['home_team'][:14]}"
            )

    lines.append(f"\n⏰ {datetime.now().strftime('%I:%M %p ET')}")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Daily MLB Props Picks")
    parser.add_argument("--test",    action="store_true",
                        help="Test Telegram connection")
    parser.add_argument("--no-send", action="store_true",
                        help="Print picks, don't send to Telegram")
    parser.add_argument("--date",    default=None,
                        help="Date YYYY-MM-DD, default today")
    args   = parser.parse_args()
    config = load_config()

    if args.test:
        test_telegram(config)
        return

    print(f"\n⚾  Daily Props Picks — {datetime.now().strftime('%B %d, %Y')}\n")

    # 1. Get today's games
    print("  Fetching today's games...")
    games = fetch_todays_games(args.date)
    if games.empty:
        print("  No games today.")
        return
    print(f"  ✅ {len(games)} games  |  "
          f"{sum(1 for _,g in games.iterrows() if g['home_sp'] != 'TBD')} "
          f"probable starters found")

    # 2. Run K predictions
    print("\n  Running pitcher K model...")
    predictions = run_k_predictions(games)
    print(f"  ✅ {len(predictions)} predictions generated")

    # 3. Print predictions
    print(f"\n{'═'*55}")
    if predictions:
        predictions_sorted = sorted(predictions,
                                     key=lambda x: x["pred_k_total"],
                                     reverse=True)
        print(f"  {'PITCHER':<25} {'MATCHUP':<30} {'PRED Ks':>8} {'K/9':>6}")
        print(f"  {'-'*55}")
        for p in predictions_sorted:
            matchup = f"{p['away_team'][:12]} @ {p['home_team'][:12]}"
            form = " 🔥" if p.get("form_z",0) > 1.5 else (" 🥶" if p.get("form_z",0) < -1.5 else "")
            print(f"  {p['pitcher']:<25} {matchup:<30} "
                  f"{p['pred_k_total']:>7.1f} {p['pred_k9']:>5.1f}{form}")
    print(f"{'═'*55}\n")

    # 4. Build and send message
    message = build_message(games, predictions, config)

    if not args.no_send:
        send_telegram(message, config["telegram_token"], config["chat_id"])
    else:
        print("(--no-send: Telegram skipped)")


if __name__ == "__main__":
    main()
