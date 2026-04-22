"""
daily_batter_picks.py
---------------------
Daily batter props predictions — H, TB, HR for top hitters.
Runs separately from pitcher picks at noon ET when lineups are posted.

Usage:
  python daily_batter_picks.py              # today's predictions
  python daily_batter_picks.py --no-send    # print only
  python daily_batter_picks.py --test       # test Telegram
"""

import json
import subprocess
import warnings
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")

CONFIG_FILE = Path("config.json")


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
    return defaults


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


def fetch_todays_games() -> pd.DataFrame:
    """Fetch today's MLB schedule from ESPN."""
    from daily_picks import fetch_todays_games as _fetch
    return _fetch()


def get_actual_lineups(games: pd.DataFrame) -> dict:
    """
    Fetch actual posted lineups from MLB Stats API.
    Returns dict of team -> list of batter names in batting order.
    Only works when lineups are posted (~2-3 hours before game time).
    """
    import requests

    lineups = {}
    date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        # Get today's schedule with game PKs
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId":     1,
                "date":        date_str,
                "hydrate":     "lineups",
                "season":      date_str[:4],
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        for date in data.get("dates", []):
            for game in date.get("games", []):
                lineups_data = game.get("lineups", {})
                if not lineups_data:
                    continue

                for side in ["homePlayers", "awayPlayers"]:
                    players = lineups_data.get(side, [])
                    if not players:
                        continue

                    team_name = (
                        game.get("teams", {})
                        .get("home" if side == "homePlayers" else "away", {})
                        .get("team", {})
                        .get("name", "")
                    )

                    # Filter to batters only (exclude pitcher at end)
                    batters = []
                    for p in players:
                        pos = p.get("primaryPosition", {}).get("code", "")
                        if pos != "1":  # 1 = pitcher
                            name = p.get("fullName", "")
                            if name:
                                batters.append(name)

                    if team_name and batters:
                        lineups[team_name] = batters[:3]  # top 3

    except Exception as e:
        print(f"  ⚠️  Lineup fetch failed: {e}")

    return lineups


def run_batter_predictions(games: pd.DataFrame) -> list:
    """
    Run H, TB, HR predictions for top 3 batters per team.
    Uses actual posted lineups when available.
    """
    from statcast_batters import (
        fetch_todays_batter_games,
        predict_batter_props,
        get_top_batters_for_team,
    )

    today    = datetime.now()
    date_str = today.strftime("%Y-%m-%d")

    # Try actual lineups first
    print("  Fetching today's lineups from MLB API...")
    lineups = get_actual_lineups(games)

    if lineups:
        print(f"  ✅ Got actual lineups for {len(lineups)} teams")
    else:
        print("  ⚠️  Lineups not posted yet — using roster fallback")

    # Collect all batters
    all_batters  = []
    batter_info  = {}

    for _, game in games.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        for team in [home, away]:
            # Use actual lineup if available, else roster fallback
            if team in lineups:
                batters = lineups[team]
            else:
                batters = get_top_batters_for_team(team, date_str, n=3)

            opp_sp   = game.get("away_sp","") if team == home else game.get("home_sp","")
            opp_hand = "L" if "L" in str(opp_sp) else "R"

            for b in batters:
                if b not in all_batters:
                    all_batters.append(b)
                batter_info[b] = {
                    "team":      team,
                    "home_team": home,
                    "away_team": away,
                    "time":      game.get("time",""),
                    "opp_hand":  opp_hand,
                }

    if not all_batters:
        print("  ⚠️  No batters found")
        return []

    # Fetch recent game logs
    print(f"  Fetching recent Statcast data for {len(all_batters)} batters...")
    game_logs = fetch_todays_batter_games(all_batters, days_back=30)

    if game_logs.empty:
        print("  ⚠️  No batter game logs found")
        return []

    # Generate predictions
    predictions = []
    for batter, info in batter_info.items():
        preds = predict_batter_props(
            batter, today, game_logs,
            opp_pitcher_hand = info["opp_hand"],
            home_team        = info["home_team"],
        )
        if not preds or preds.get("n_games", 0) < 5:
            continue

        predictions.append({
            "batter":    batter,
            "team":      info["team"],
            "home_team": info["home_team"],
            "away_team": info["away_team"],
            "time":      info["time"],
            "pred_h":    preds.get("pred_h"),
            "pred_tb":   preds.get("pred_tb"),
            "pred_hr":   preds.get("pred_hr"),
            "hr_pct":    preds.get("hr_pct"),
            "form_z":    preds.get("form_z", 0),
            "hot":       preds.get("hot", 0),
            "slump":     preds.get("slump", 0),
            "exit_velo": preds.get("exit_velo"),
            "barrel":    preds.get("barrel"),
            "n_games":   preds.get("n_games", 0),
        })

    return predictions


def build_message(predictions: list, games: pd.DataFrame) -> str:
    """Build Telegram message for batter props."""
    today = datetime.now().strftime("%B %d, %Y")
    lines = [f"<b>🏏 MLB Batter Props — {today}</b>"]
    lines.append("<i>Compare predictions to your FanDuel lines</i>\n")

    if not predictions:
        lines.append("❌ No predictions available\n(Lineups may not be posted yet)")
        lines.append(f"\n⏰ {datetime.now().strftime('%I:%M %p ET')}")
        return "\n".join(lines)

    # ── Top H/TB picks (sorted by TB) ──
    preds_sorted = sorted(
        predictions,
        key=lambda x: x.get("pred_tb") or 0,
        reverse=True
    )

    for p in preds_sorted[:12]:
        form_str = " 🔥" if p.get("hot") else (" 🥶" if p.get("slump") else "")
        hr_pct   = p.get("hr_pct")
        hr_str   = f"{hr_pct}%" if hr_pct is not None else "?"

        lines.append(
            f"🏏 <b>{p['batter']}</b>{form_str}"
            f"\n   {p['away_team']} @ {p['home_team']} | {p['time']}"
            f"\n   H: <b>{p['pred_h']}</b>  "
            f"TB: <b>{p['pred_tb']}</b>  "
            f"HR: <b>{hr_str}</b>\n"
        )

    # ── HR Picks section ──
    hr_picks = sorted(
        [p for p in predictions if (p.get("hr_pct") or 0) >= 20],
        key=lambda x: x.get("hr_pct") or 0,
        reverse=True
    )

    if hr_picks:
        lines.append("─" * 30)
        lines.append("<b>🏠 HR Picks — Tonight's Best Bets</b>")
        lines.append("<i>% = model's estimated HR probability</i>\n")

        for p in hr_picks[:8]:
            pct = p.get("hr_pct", 0)
            barrel = p.get("barrel")
            barrel_str = f" · barrel {barrel*100:.0f}%" if barrel else ""

            if pct >= 35:
                icon = "🔥"
            elif pct >= 25:
                icon = "⚡"
            else:
                icon = "📊"

            lines.append(
                f"{icon} <b>{p['batter']}</b> — <b>{pct}%</b>{barrel_str}"
                f"\n   {p['away_team']} @ {p['home_team']} | {p['time']}\n"
            )

    lines.append(f"⏰ {datetime.now().strftime('%I:%M %p ET')}")
    return "\n".join(lines)


def save_batter_predictions(predictions: list, date_str: str) -> Path:
    """Merge batter predictions into predictions/YYYY-MM-DD.json."""
    filepath = PREDICTIONS_DIR / f"{date_str}.json"

    existing = {}
    if filepath.exists():
        try:
            existing = json.load(open(filepath))
        except Exception:
            existing = {}

    existing["date"]    = date_str
    existing["batters"] = predictions

    json.dump(existing, open(filepath, "w"), indent=2, default=str)
    print("  Saved batter predictions → {}".format(filepath))
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


def main():
    parser = argparse.ArgumentParser(description="Daily Batter Props")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--test",    action="store_true")
    args   = parser.parse_args()
    config = load_config()

    if args.test:
        send_telegram(
            f"🏏 Batter Props Bot connected!\n{datetime.now().strftime('%I:%M %p')}",
            config["telegram_token"], config["chat_id"]
        )
        return

    print(f"\n🏏  Daily Batter Props — {datetime.now().strftime('%B %d, %Y')}\n")

    print("  Fetching today's games...")
    games = fetch_todays_games()
    if games.empty:
        print("  No games today.")
        return
    print(f"  ✅ {len(games)} games found")

    print("\n  Running batter predictions...")
    predictions = run_batter_predictions(games)
    print(f"  ✅ {len(predictions)} batter predictions")

    # Print summary — sorted by TB
    print(f"\n{'═'*62}")
    if predictions:
        print(f"  {'BATTER':<25} {'H':>5} {'TB':>6} {'HR%':>6}")
        print(f"  {'-'*52}")
        for p in sorted(predictions, key=lambda x: x.get("pred_tb") or 0, reverse=True)[:12]:
            form  = " 🔥" if p.get("hot") else (" 🥶" if p.get("slump") else "")
            hr_pct = p.get("hr_pct")
            hr_str = f"{hr_pct}%" if hr_pct is not None else "  ?"
            print(f"  {p['batter']:<25} "
                  f"{p['pred_h']:>5.2f} "
                  f"{p['pred_tb']:>6.2f} "
                  f"{hr_str:>6}{form}")

    # Print HR picks section — sorted by HR%
    hr_picks = sorted(
        [p for p in predictions if (p.get("hr_pct") or 0) >= 15],
        key=lambda x: x.get("hr_pct") or 0,
        reverse=True
    )
    if hr_picks:
        print(f"\n  {'─'*52}")
        print(f"  🏠 TOP HR PICKS")
        print(f"  {'─'*52}")
        for p in hr_picks[:10]:
            barrel_str = f" barrel {p['barrel']*100:.0f}%" if p.get("barrel") else ""
            print(f"  {p['batter']:<25} {p['hr_pct']:>3}%{barrel_str}")
    print(f"{'═'*62}\n")

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = save_batter_predictions(predictions, date_str)
    git_push_predictions(filepath)

    message = build_message(predictions, games)

    if not args.no_send:
        send_telegram(message, config["telegram_token"], config["chat_id"])
    else:
        print("(--no-send: Telegram skipped)")


if __name__ == "__main__":
    main()
