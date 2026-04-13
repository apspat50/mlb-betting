"""
daily_picks.py

Daily props picker -- runs every morning and sends to Telegram.

What it does:
1. Fetches today's MLB games and starting pitchers (ESPN, free)
2. Fetches recent Statcast per-start data for each pitcher
3. Predicts Ks using Statcast rolling averages (sc_k_L3, sc_k_L5)
4. Sends predictions to Telegram -- compare to FanDuel manually

Usage:
  python daily_picks.py            # today's picks + send to Telegram
  python daily_picks.py --no-send  # print only, don't send
  python daily_picks.py --test     # test Telegram connection
  python daily_picks.py --date 2025-04-15  # specific date
"""

import json
import subprocess
import warnings
import argparse
import requests

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")

CONFIG_FILE = Path("config.json")
MODELS_DIR  = Path("saved_models") / "props"
DATA_DIR    = Path("props_data")


# ======================================================
# CONFIG
# ======================================================

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
        print("  Created config.json -- add your Telegram token and chat_id")
    return defaults


# ======================================================
# TELEGRAM
# ======================================================

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


def test_telegram(config: dict):
    msg = (
        "MLB Props Bot connected!\n"
        "Time: {}\n"
        "Ready to send daily pitcher K predictions.".format(
            datetime.now().strftime("%I:%M %p")
        )
    )
    ok = send_telegram(msg, config["telegram_token"], config["chat_id"])
    if ok:
        print("  Check your phone!")
    else:
        print("  Failed -- check config.json")


# ======================================================
# ESPN GAME SCHEDULE (free, no API key)
# ======================================================

def fetch_todays_games(date_str=None) -> pd.DataFrame:
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
        print("  ESPN failed: {}".format(e))
        return pd.DataFrame()

    games = []
    for event in data.get("events", []):
        comp  = event.get("competitions", [{}])[0]
        teams = comp.get("competitors", [])
        home  = next((t for t in teams if t.get("homeAway") == "home"), {})
        away  = next((t for t in teams if t.get("homeAway") == "away"), {})

        try:
            gt       = datetime.strptime(event.get("date", ""), "%Y-%m-%dT%H:%MZ")
            gt_et    = gt - timedelta(hours=4)
            time_str = gt_et.strftime("%I:%M %p ET").lstrip("0")
        except Exception:
            time_str = "TBD"

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


# ======================================================
# PITCHER K PREDICTIONS -- STATCAST FIRST
# ======================================================

def run_k_predictions(games: pd.DataFrame) -> list:
    """
    Predict strikeouts for today's starters using Statcast rolling averages.

    Priority order:
      1. sc_k_L3 + sc_k_L5  -- avg Ks per start from last 3/5 Statcast starts
      2. sc_k_season         -- Statcast season average per start
      3. Trained model       -- fallback only, season-avg based
    """
    try:
        from props_model import (
            build_pitcher_k_features,
            load_pitching_logs,
            load_team_batting_stats,
            predict_stat,
        )
        from statcast_logs import fetch_todays_pitcher_starts, fetch_mlb_k_logs_for_pitchers
    except Exception as e:
        print("  Could not import props_model: {}".format(e))
        return []

    # Season logs for opponent context (K%, wRC+, etc.)
    try:
        seasons  = list(range(2014, datetime.now().year + 1))
        pit_logs = load_pitching_logs(seasons)
        team_bat = load_team_batting_stats(seasons)
    except Exception as e:
        print("  Could not load season logs: {}".format(e))
        pit_logs = pd.DataFrame()
        team_bat = pd.DataFrame()

    # Load model as fallback only
    model_pkg  = None
    model_path = MODELS_DIR / "pitcher_strikeouts.pkl"
    if model_path.exists():
        try:
            model_pkg = joblib.load(model_path)
        except Exception:
            pass

    # Collect today's pitchers
    todays_pitchers = []
    for _, g in games.iterrows():
        if g.get("home_sp") and g["home_sp"] != "TBD":
            todays_pitchers.append(g["home_sp"])
        if g.get("away_sp") and g["away_sp"] != "TBD":
            todays_pitchers.append(g["away_sp"])

    # Fetch recent Statcast starts -- primary data source
    print("  Fetching recent Statcast starts for {} pitchers...".format(
        len(todays_pitchers)
    ))
    start_logs = fetch_todays_pitcher_starts(todays_pitchers, days_back=60)

    if start_logs.empty:
        print("  No Statcast data found -- will use model fallback only")
        start_logs = None
    else:
        print("  Statcast data loaded for {} pitchers".format(
            start_logs["name"].nunique()
        ))

    # Fetch accurate K counts from MLB Stats API (official source)
    print("  Fetching accurate K counts from MLB Stats API...")
    mlb_k_logs = fetch_mlb_k_logs_for_pitchers(todays_pitchers)

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
                feats = build_pitcher_k_features(
                    pitcher, today, pit_logs, opp_team, team_bat, home_team,
                    start_logs=start_logs,
                )

                # -- PATCH K COUNTS WITH ACCURATE MLB STATS API DATA --
                # MLB game logs have 100% correct K counts; Statcast aggregation
                # can miscount due to pitch-by-pitch edge cases.
                mlb_logs = mlb_k_logs.get(pitcher)
                if mlb_logs is not None and not mlb_logs.empty:
                    past = mlb_logs[mlb_logs["game_date"] < today].copy()
                    if len(past) >= 3:
                        feats["sc_k_L3"]    = float(past["SO"].tail(3).mean())
                        feats["sc_k_L5"]    = float(past["SO"].tail(5).mean())
                        feats["sc_k_season"]= float(past["SO"].mean())
                        feats["sc_ip_L3"]   = float(past["IP"].tail(3).mean())
                        feats["sc_k_std"]   = float(past["SO"].tail(10).std())

                # -- STATCAST-FIRST PREDICTION --
                sc_k_L3     = feats.get("sc_k_L3")
                sc_k_L5     = feats.get("sc_k_L5")
                sc_k_season = feats.get("sc_k_season")
                data_source = "unknown"

                if sc_k_L3 is not None and sc_k_L5 is not None:
                    # Base: weighted rolling K average (recent weighted heavier)
                    base_k = sc_k_L3 * 0.6 + sc_k_L5 * 0.4

                    # SwStr% adjustment vs league avg (~10.5%)
                    # Higher whiff rate = pitcher is missing more bats than history shows
                    sc_swstr = feats.get("sc_swstr_L3")
                    if sc_swstr and not np.isnan(float(sc_swstr)):
                        swstr_factor = float(sc_swstr) / 0.105
                        # Dampen: don't let swstr move prediction more than ±12%
                        swstr_adj = max(0.88, min(1.12, 0.5 + swstr_factor * 0.5))
                    else:
                        swstr_adj = 1.0

                    # Opponent K% adjustment vs league avg (~22.2%)
                    opp_kpct_val = feats.get("opp_kpct", 0.222) or 0.222
                    opp_adj = max(0.90, min(1.10, float(opp_kpct_val) / 0.222))

                    # Velocity trend: losing velo → fewer Ks
                    velo_trend = feats.get("sc_velo_trend", 0) or 0
                    velo_adj = 1.0 + np.clip(float(velo_trend) * 0.015, -0.06, 0.06)

                    pred_per_start = base_k * swstr_adj * opp_adj * velo_adj
                    data_source    = "Statcast (L3={:.1f} L5={:.1f} SwStr={:.1%} OppK={:.1%})".format(
                        sc_k_L3, sc_k_L5, sc_swstr or 0.105, opp_kpct_val
                    )

                elif sc_k_season is not None:
                    pred_per_start = sc_k_season
                    data_source    = "Statcast season avg ({:.1f})".format(sc_k_season)

                elif model_pkg is not None:
                    pred_per_start = predict_stat(model_pkg, feats)
                    if pred_per_start > 20:
                        season_ip = feats.get("p_ip_L3")
                        if season_ip and season_ip > 50:
                            gs_estimate = max(round(season_ip / 5.5), 1)
                        else:
                            gs_estimate = 32
                        pred_per_start = pred_per_start / gs_estimate
                    data_source = "model fallback (no Statcast)"

                else:
                    print("    {}: no data -- skipping".format(pitcher))
                    continue

                print("    {}: {} -> {:.1f} K/start".format(
                    pitcher, data_source, pred_per_start
                ))

                avg_ip = feats.get("sc_ip_L3")
                if not avg_ip or avg_ip > 10 or avg_ip < 1:
                    avg_ip = 5.5
                avg_ip  = round(float(avg_ip), 1)
                pred_k9 = pred_per_start / avg_ip * 9

                form_z   = feats.get("sc_form_z") or feats.get("p_form_z") or 0
                opp_kpct = feats.get("opp_kpct", 0.22) or 0.22

                predictions.append({
                    "pitcher":      pitcher,
                    "role":         role,
                    "home_team":    game["home_team"],
                    "away_team":    game["away_team"],
                    "time":         game["time"],
                    "opp_team":     opp_team,
                    "pred_k9":      round(pred_k9, 1),
                    "pred_k_total": round(pred_per_start, 1),
                    "avg_ip":       round(avg_ip, 1),
                    "form_z":       round(form_z, 2),
                    "opp_kpct":     round(opp_kpct, 3),
                    "data_source":  data_source,
                })

            except Exception as e:
                print("    {}: error -- {}".format(pitcher, e))
                continue

    return predictions


# ======================================================
# BUILD MESSAGE
# ======================================================

def build_message(games: pd.DataFrame, predictions: list, config: dict) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    lines = ["<b>MLB Pitcher K Props -- {}</b>".format(today)]
    lines.append("<i>Statcast rolling averages -- compare to FanDuel manually</i>\n")

    if predictions:
        predictions.sort(key=lambda x: x["pred_k_total"], reverse=True)

        for p in predictions:
            form = p.get("form_z", 0)
            if form > 1.5:
                form_str = " HOT"
            elif form < -1.5:
                form_str = " COLD"
            else:
                form_str = ""

            opp_k = p.get("opp_kpct", 0.22)
            if opp_k > 0.26:
                opp_str = " (opp Ks a lot)"
            elif opp_k < 0.18:
                opp_str = " (opp patient)"
            else:
                opp_str = ""

            lines.append(
                "<b>{}</b> ({}){}\n"
                "   {} @ {} | {}\n"
                "   Predicted: <b>{} Ks</b> ({} K/9 x {} IP){}\n".format(
                    p["pitcher"], p["role"], form_str,
                    p["away_team"], p["home_team"], p["time"],
                    p["pred_k_total"], p["pred_k9"], p["avg_ip"], opp_str,
                )
            )
    else:
        lines.append("No predictions available.")
        lines.append("Statcast data may not be ready yet for today's starters.")

    if not games.empty:
        lines.append("<b>{} Games Today</b>".format(len(games)))
        for _, g in games.iterrows():
            lines.append("  {:>9} {} @ {}".format(
                g["time"], g["away_team"][:14], g["home_team"][:14]
            ))

    lines.append("\n{}".format(datetime.now().strftime("%I:%M %p ET")))
    return "\n".join(lines)


# ======================================================
# SAVE PREDICTIONS + GIT PUSH
# ======================================================

def save_pitcher_predictions(predictions: list, date_str: str) -> Path:
    """Save pitcher predictions to predictions/YYYY-MM-DD.json."""
    filepath = PREDICTIONS_DIR / f"{date_str}.json"

    # Load existing file so batter predictions aren't overwritten
    existing = {}
    if filepath.exists():
        try:
            existing = json.load(open(filepath))
        except Exception:
            existing = {}

    existing["date"]     = date_str
    existing["pitchers"] = predictions

    json.dump(existing, open(filepath, "w"), indent=2, default=str)
    print("  Saved predictions → {}".format(filepath))
    return filepath


def git_push_predictions(filepath: Path):
    """Commit and push the predictions file to GitHub."""
    try:
        subprocess.run(["git", "add", str(filepath)], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", "predictions: {}".format(filepath.stem)],
            capture_output=True, text=True
        )
        if "nothing to commit" in result.stdout:
            print("  No changes to commit")
            return
        subprocess.run(
            ["git", "push", "-u", "origin", "HEAD"],
            check=True, capture_output=True
        )
        print("  Pushed predictions to GitHub")
    except subprocess.CalledProcessError as e:
        print("  Git push failed: {}".format(e))


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Daily MLB Props Picks")
    parser.add_argument("--test",    action="store_true", help="Test Telegram connection")
    parser.add_argument("--no-send", action="store_true", help="Print picks, skip Telegram")
    parser.add_argument("--date",    default=None,        help="Date YYYY-MM-DD, default today")
    args   = parser.parse_args()
    config = load_config()

    if args.test:
        test_telegram(config)
        return

    print("\nDaily Props Picks -- {}\n".format(datetime.now().strftime("%B %d, %Y")))

    print("  Fetching today's games...")
    games = fetch_todays_games(args.date)
    if games.empty:
        print("  No games today.")
        return
    starters = sum(1 for _, g in games.iterrows() if g["home_sp"] != "TBD")
    print("  {} games | {} probable starters found".format(len(games), starters))

    print("\n  Running pitcher K predictions (Statcast)...")
    predictions = run_k_predictions(games)
    print("  {} K predictions generated".format(len(predictions)))

    print("\n" + "=" * 60)
    if predictions:
        predictions_sorted = sorted(
            predictions, key=lambda x: x["pred_k_total"], reverse=True
        )
        print("  {:<25} {:<28} {:>8} {:>6}".format(
            "PITCHER", "MATCHUP", "PRED Ks", "K/9"
        ))
        print("  " + "-" * 55)
        for p in predictions_sorted:
            matchup = "{} @ {}".format(p["away_team"][:11], p["home_team"][:11])
            form    = " HOT" if p.get("form_z", 0) > 1.5 else (
                      " COLD" if p.get("form_z", 0) < -1.5 else "")
            print("  {:<25} {:<28} {:>7.1f} {:>5.1f}{}".format(
                p["pitcher"], matchup, p["pred_k_total"], p["pred_k9"], form
            ))
    print("=" * 60 + "\n")

    date_str = (
        args.date if args.date
        else datetime.now().strftime("%Y-%m-%d")
    )
    filepath = save_pitcher_predictions(predictions, date_str)
    git_push_predictions(filepath)

    message = build_message(games, predictions, config)

    if not args.no_send:
        send_telegram(message, config["telegram_token"], config["chat_id"])
    else:
        print("(--no-send: Telegram skipped)")
        import re
        print("\n--- MESSAGE PREVIEW ---")
        print(re.sub(r"<[^>]+>", "", message))


if __name__ == "__main__":
    main()
