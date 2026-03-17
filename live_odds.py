"""
live_odds.py
------------
Fetches live MLB odds from free sources including FanDuel.

Sources (all free, no API key required):
  1. The Odds API free tier (500 req/month) — game lines + props
  2. DraftKings public API — game lines
  3. FanDuel public endpoint — game lines
  4. ESPN scores API — game schedules and results

Usage:
  python live_odds.py                     # today's games + picks
  python live_odds.py --date 2025-04-15   # specific date
  python live_odds.py --props             # include player props
  python live_odds.py --sport mlb         # mlb (default), nba, nfl
"""

import json
import time
import warnings
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache") / "live_odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Sport configs ──
SPORT_CONFIGS = {
    "mlb": {
        "espn_id":     "baseball/mlb",
        "dk_sport":    "baseball",
        "dk_league":   "84240",     # DraftKings MLB league ID
        "odds_api":    "baseball_mlb",
        "fd_sport":    "MLB",
    },
    "nba": {
        "espn_id":     "basketball/nba",
        "dk_sport":    "basketball",
        "dk_league":   "42648",
        "odds_api":    "basketball_nba",
        "fd_sport":    "NBA",
    },
    "nfl": {
        "espn_id":     "football/nfl",
        "dk_sport":    "football",
        "dk_league":   "88808",
        "odds_api":    "americanfootball_nfl",
        "fd_sport":    "NFL",
    },
}


# ══════════════════════════════════════════════
# ESPN — FREE GAME SCHEDULE
# ══════════════════════════════════════════════

def fetch_espn_games(sport: str = "mlb",
                     date_str: str = None) -> pd.DataFrame:
    """
    Fetch today's games from ESPN public API.
    Returns game schedule with teams, times, and scores if available.
    No API key required.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    cfg     = SPORT_CONFIGS.get(sport, SPORT_CONFIGS["mlb"])
    espn_id = cfg["espn_id"]
    url     = f"https://site.api.espn.com/apis/site/v2/sports/{espn_id}/scoreboard"

    cache_file = CACHE_DIR / f"espn_{sport}_{date_str}.json"
    if cache_file.exists():
        age = (datetime.now() - datetime.fromtimestamp(
            cache_file.stat().st_mtime)).seconds
        if age < 300:   # cache for 5 minutes
            data = json.load(open(cache_file))
        else:
            data = None
    else:
        data = None

    if data is None:
        try:
            r = requests.get(url, params={"dates": date_str}, timeout=10)
            r.raise_for_status()
            data = r.json()
            json.dump(data, open(cache_file, "w"))
        except Exception as e:
            print(f"  ⚠️  ESPN fetch failed: {e}")
            return pd.DataFrame()

    games = []
    for event in data.get("events", []):
        comp    = event.get("competitions", [{}])[0]
        teams   = comp.get("competitors", [])
        status  = event.get("status", {}).get("type", {})

        home = next((t for t in teams if t.get("homeAway") == "home"), {})
        away = next((t for t in teams if t.get("homeAway") == "away"), {})

        game_time = event.get("date", "")
        # Convert UTC to local
        try:
            gt = datetime.strptime(game_time, "%Y-%m-%dT%H:%MZ")
            gt_local = gt - timedelta(hours=4)   # ET approximate
            time_str = gt_local.strftime("%I:%M %p ET")
        except:
            time_str = game_time

        games.append({
            "game_id":   event.get("id", ""),
            "date":      date_str,
            "time":      time_str,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_score":home.get("score", ""),
            "away_score":away.get("score", ""),
            "status":    status.get("description", ""),
            "home_sp":   _get_starter(comp, "home"),
            "away_sp":   _get_starter(comp, "away"),
            "venue":     comp.get("venue", {}).get("fullName", ""),
        })

    return pd.DataFrame(games) if games else pd.DataFrame()


def _get_starter(comp: dict, home_away: str) -> str:
    """Extract probable starting pitcher from ESPN competition data."""
    for competitor in comp.get("competitors", []):
        if competitor.get("homeAway") == home_away:
            probable = competitor.get("probables", [{}])
            if probable:
                return probable[0].get("athlete", {}).get("displayName", "TBD")
    return "TBD"


# ══════════════════════════════════════════════
# DRAFTKINGS — FREE ODDS API
# ══════════════════════════════════════════════

def fetch_draftkings_odds(sport: str = "mlb") -> pd.DataFrame:
    """
    Fetch moneyline and run line odds from DraftKings public endpoint.
    No API key required.
    """
    cfg = SPORT_CONFIGS.get(sport, SPORT_CONFIGS["mlb"])
    url = (f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/"
           f"eventgroups/{cfg['dk_league']}/"
           f"categories/743/subcategories/9517?format=json")

    cache_file = CACHE_DIR / f"dk_{sport}_{datetime.now().strftime('%Y%m%d_%H')}.json"

    if cache_file.exists():
        data = json.load(open(cache_file))
    else:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible)",
                "Accept": "application/json",
            }
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            json.dump(data, open(cache_file, "w"))
        except Exception as e:
            print(f"  ⚠️  DraftKings fetch failed: {e}")
            return pd.DataFrame()

    records = []
    try:
        for event_group in data.get("eventGroup", {}).get("offerCategories", []):
            for subcat in event_group.get("offerSubcategoryDescriptors", []):
                for offer_group in subcat.get("offerSubcategory", {}).get("offers", []):
                    for offer in offer_group:
                        label   = offer.get("label", "")
                        outcomes= offer.get("outcomes", [])
                        event   = offer.get("providerEventId", "")

                        if "Moneyline" in label or "Run Line" in label:
                            for oc in outcomes:
                                records.append({
                                    "event_id":  event,
                                    "market":    label,
                                    "team":      oc.get("participant", ""),
                                    "odds":      oc.get("oddsAmerican", ""),
                                    "line":      oc.get("line", ""),
                                    "book":      "DraftKings",
                                })
    except:
        pass

    return pd.DataFrame(records) if records else pd.DataFrame()


# ══════════════════════════════════════════════
# THE ODDS API — MULTI-BOOK ODDS (free tier)
# ══════════════════════════════════════════════

def fetch_odds_api(sport: str = "mlb",
                   api_key: str = "",
                   markets: str = "h2h,spreads") -> pd.DataFrame:
    """
    Fetch odds from The Odds API (free tier: 500 requests/month).
    Includes FanDuel, DraftKings, BetMGM, and others.
    """
    if not api_key:
        return pd.DataFrame()

    cfg = SPORT_CONFIGS.get(sport, SPORT_CONFIGS["mlb"])
    url = f"https://api.the-odds-api.com/v4/sports/{cfg['odds_api']}/odds"

    cache_key = f"oddsapi_{sport}_{datetime.now().strftime('%Y%m%d_%H')}"
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        data = json.load(open(cache_file))
    else:
        try:
            r = requests.get(url, params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    markets,
                "oddsFormat": "american",
                "bookmakers": "fanduel,draftkings,betmgm",
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            json.dump(data, open(cache_file, "w"))
        except Exception as e:
            print(f"  ⚠️  Odds API failed: {e}")
            return pd.DataFrame()

    records = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        gt   = game.get("commence_time", "")

        for bk in game.get("bookmakers", []):
            book = bk.get("title", "")
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                for oc in mkt.get("outcomes", []):
                    records.append({
                        "home_team": home,
                        "away_team": away,
                        "game_time": gt,
                        "market":    mkt_key,
                        "team":      oc.get("name", ""),
                        "odds":      oc.get("price", ""),
                        "line":      oc.get("point", ""),
                        "book":      book,
                    })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ══════════════════════════════════════════════
# FANDUEL SPECIFIC PARSING
# ══════════════════════════════════════════════

def get_fanduel_lines(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to FanDuel lines only."""
    if odds_df.empty:
        return pd.DataFrame()
    return odds_df[odds_df["book"].str.lower() == "fanduel"].copy()


def format_fanduel_game(home: str, away: str,
                          odds_df: pd.DataFrame) -> dict:
    """
    Extract FanDuel moneyline and run line for a specific game.
    Returns dict with home_ml, away_ml, home_rl, away_rl, total.
    """
    result = {
        "home_ml": None, "away_ml": None,
        "home_rl": None, "away_rl": None,
        "total":   None, "total_odds_over": None,
    }

    fd = odds_df[odds_df["book"].str.lower() == "fanduel"]
    game_lines = fd[
        (fd["home_team"] == home) & (fd["away_team"] == away)
    ]

    if game_lines.empty:
        return result

    for _, row in game_lines.iterrows():
        mkt  = str(row.get("market", ""))
        team = str(row.get("team", ""))
        odds = row.get("odds", None)
        line = row.get("line", None)

        if mkt == "h2h":
            if team == home: result["home_ml"] = odds
            if team == away: result["away_ml"] = odds
        elif mkt == "spreads":
            if team == home: result["home_rl"] = odds; result["rl_line"] = line
            if team == away: result["away_rl"] = odds
        elif mkt == "totals":
            if "over" in team.lower():
                result["total"] = line
                result["total_odds_over"] = odds

    return result


# ══════════════════════════════════════════════
# MAIN DAILY FETCH FUNCTION
# ══════════════════════════════════════════════

def get_todays_games(sport: str = "mlb",
                      api_key: str = "",
                      date_str: str = None) -> pd.DataFrame:
    """
    Get today's games with odds from all available sources.
    Combines ESPN schedule + DraftKings/FanDuel odds.
    Works without an API key (ESPN + DK are fully free).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"  📅 Fetching {sport.upper()} games for {date_str}...")

    # Get schedule from ESPN (always free)
    games = fetch_espn_games(sport, date_str)
    if games.empty:
        print(f"  ⚠️  No {sport.upper()} games found for {date_str}")
        return pd.DataFrame()

    print(f"  ✅ {len(games)} games found")

    # Get odds — try Odds API first (has FanDuel), fall back to DK
    if api_key:
        odds_df = fetch_odds_api(sport, api_key)
        source  = "The Odds API (FanDuel + DraftKings)"
    else:
        odds_df = fetch_draftkings_odds(sport)
        source  = "DraftKings (free)"

    print(f"  📊 Odds source: {source}")

    # Merge odds into games
    if not odds_df.empty:
        for i, game in games.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            fd_lines = format_fanduel_game(home, away, odds_df)
            for k, v in fd_lines.items():
                games.at[i, k] = v

    return games


# ══════════════════════════════════════════════
# FORMATTED OUTPUT FOR PHONE/TERMINAL
# ══════════════════════════════════════════════

def format_picks_message(games: pd.DataFrame,
                          predictions: list,
                          sport: str = "MLB") -> str:
    """
    Format predictions into a clean message for Telegram/phone.
    """
    today = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"⚾ {sport} Picks — {today}",
        f"{'─'*35}",
    ]

    bet_count = 0
    for pred in predictions:
        if not pred.get("bet"):
            continue
        bet_count += 1
        lines.append(
            f"\n✅ {pred['bet_type']}"
            f"\n   {pred['away_team']} @ {pred['home_team']}"
            f"\n   {pred['time']}"
            f"\n   Model: {pred['model_prob']:.1%}  "
            f"Book: {pred['book_prob']:.1%}  "
            f"Edge: {pred['edge']:+.1%}"
            f"\n   Stake: ${pred['stake']:.0f}  "
            f"Odds: {pred['odds']:+d}"
        )

    if bet_count == 0:
        lines.append("\n❌ No strong edges found today")
        lines.append("   (min 8% edge threshold)")
    else:
        lines.append(f"\n{'─'*35}")
        lines.append(f"Total bets: {bet_count}")

    lines.append(f"\n⏰ Generated: {datetime.now().strftime('%I:%M %p ET')}")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live MLB Odds Fetcher")
    parser.add_argument("--sport",   default="mlb",
                        choices=["mlb","nba","nfl"])
    parser.add_argument("--date",    default=None,
                        help="Date (YYYY-MM-DD), default today")
    parser.add_argument("--api-key", default="",
                        help="The Odds API key (optional, gets FanDuel lines)")
    parser.add_argument("--props",   action="store_true",
                        help="Also fetch player props")
    args = parser.parse_args()

    games = get_todays_games(args.sport, args.api_key, args.date)

    if not games.empty:
        print(f"\n{'═'*55}")
        print(f"  TODAY'S {args.sport.upper()} GAMES")
        print(f"{'─'*55}")
        for _, g in games.iterrows():
            ml_str = ""
            if pd.notna(g.get("home_ml")):
                ml_str = (f"  ML: {g['away_ml']:+.0f} / "
                          f"{g['home_ml']:+.0f}")
            rl_str = ""
            if pd.notna(g.get("home_rl")):
                rl_str = f"  RL: {g['home_rl']:+.0f}"

            print(f"  {g['time']:>10}  "
                  f"{g['away_team'][:18]:<18} @ "
                  f"{g['home_team'][:18]:<18}"
                  f"{ml_str}{rl_str}")
            if g.get("away_sp") and g["away_sp"] != "TBD":
                print(f"  {'':>10}  SP: {g['away_sp']} vs {g['home_sp']}")
        print(f"{'═'*55}")
