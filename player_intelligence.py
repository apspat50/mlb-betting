"""
player_intelligence.py
----------------------
Tracks player trades, slumps, and position changes using the MLB Stats API.
All data is free, no API key required.

Features added per game:
  TRADES:
    - home/away_team_trade_disruption  : how many key players acquired in last 30 days
    - home/away_traded_player_adj      : career stats of newly acquired players vs replaced
    - home/away_roster_stability       : % of lineup that's been on team 30+ days

  SLUMPS (2 std dev threshold):
    - home/away_lineup_slump_score     : how many starters are in a statistical slump
    - home/away_slump_severity         : avg deviation below career mean for slumping players
    - home/away_hot_streak_score       : how many starters are running hot (2 std above)
    - home/away_net_form_score         : hot players minus slumping players

  POSITION CHANGES:
    - home/away_position_change_count  : players playing out of primary position
    - home/away_defense_disruption     : estimated defensive impact of position changes

MLB Stats API endpoints used:
  /api/v1/transactions     - all trades, DFA, callups
  /api/v1/people/{id}/stats - player rolling stats
  /api/v1/teams/{id}/roster - current roster with positions
"""

import time
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:
    raise ImportError("Run: pip install requests")

CACHE_DIR = Path("cache") / "mlb_api"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"

# How long to cache different data types
CACHE_DAYS = {
    "transactions": 1,    # refresh daily
    "roster":       1,
    "player_stats": 7,    # career/season stats don't change much
    "game_log":     1,
}


# ══════════════════════════════════════════════
# MLB STATS API HELPERS
# ══════════════════════════════════════════════

def _cache_path(key: str) -> Path:
    safe = hashlib.md5(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{safe}.json"


def _is_fresh(path: Path, max_days: int) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    return age < max_days


def api_get(endpoint: str, params: dict = None,
            cache_key: str = None, cache_days: int = 1) -> Optional[dict]:
    """GET from MLB Stats API with caching."""
    cache_key = cache_key or (endpoint + str(sorted((params or {}).items())))
    cp = _cache_path(cache_key)

    if _is_fresh(cp, cache_days):
        with open(cp) as f:
            return json.load(f)

    url = f"{MLB_API}{endpoint}"
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        with open(cp, "w") as f:
            json.dump(data, f)
        time.sleep(0.2)   # be polite to the API
        return data
    except Exception as e:
        return None


# ══════════════════════════════════════════════
# TEAM ID LOOKUP
# ══════════════════════════════════════════════

# MLB team name → Stats API team ID
TEAM_TO_MLB_ID = {
    "Arizona Diamondbacks":  109, "Atlanta Braves":        144,
    "Baltimore Orioles":     110, "Boston Red Sox":        111,
    "Chicago Cubs":          112, "Chicago White Sox":     145,
    "Cincinnati Reds":       113, "Cleveland Indians":     114,
    "Cleveland Guardians":   114, "Colorado Rockies":      115,
    "Detroit Tigers":        116, "Houston Astros":        117,
    "Kansas City Royals":    118, "Los Angeles Angels":    108,
    "Los Angeles Dodgers":   119, "Miami Marlins":         146,
    "Milwaukee Brewers":     158, "Minnesota Twins":       142,
    "New York Mets":         121, "New York Yankees":      147,
    "Oakland Athletics":     133, "Philadelphia Phillies": 143,
    "Pittsburgh Pirates":    134, "San Diego Padres":      135,
    "Seattle Mariners":      136, "San Francisco Giants":  137,
    "St. Louis Cardinals":   138, "Tampa Bay Rays":        139,
    "Texas Rangers":         140, "Toronto Blue Jays":     141,
    "Washington Nationals":  120,
}


def get_team_id(team_name: str) -> Optional[int]:
    return TEAM_TO_MLB_ID.get(team_name)


# ══════════════════════════════════════════════
# TRANSACTIONS (trades, DFA, callups)
# ══════════════════════════════════════════════

def fetch_transactions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch all MLB transactions between two dates.
    Returns DataFrame with columns:
      date, player_id, player_name, from_team, to_team, transaction_type
    """
    cache_key = f"transactions_{start_date}_{end_date}"
    data = api_get(
        "/transactions",
        params={"startDate": start_date, "endDate": end_date, "sportId": 1},
        cache_key=cache_key,
        cache_days=CACHE_DAYS["transactions"]
    )

    if not data or "transactions" not in data:
        return pd.DataFrame()

    records = []
    for t in data["transactions"]:
        try:
            records.append({
                "date":             t.get("date", ""),
                "effective_date":   t.get("effectiveDate", t.get("date", "")),
                "player_id":        t.get("person", {}).get("id"),
                "player_name":      t.get("person", {}).get("fullName", ""),
                "from_team_id":     t.get("fromTeam", {}).get("id"),
                "from_team":        t.get("fromTeam", {}).get("name", ""),
                "to_team_id":       t.get("toTeam", {}).get("id"),
                "to_team":          t.get("toTeam", {}).get("name", ""),
                "transaction_type": t.get("typeDesc", ""),
                "description":      t.get("description", ""),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    return df


def build_transaction_lookup(seasons: list) -> dict:
    """
    Build a complete transaction history for all given seasons.
    Returns dict: {team_id: [list of transactions affecting this team]}
    """
    all_transactions = []
    for season in seasons:
        print(f"    Fetching transactions {season}...")
        df = fetch_transactions(f"{season}-01-01", f"{season}-12-31")
        if not df.empty:
            all_transactions.append(df)

    if not all_transactions:
        return {}

    combined = pd.concat(all_transactions, ignore_index=True)
    combined = combined.dropna(subset=["player_id", "effective_date"])
    combined = combined.sort_values("effective_date")

    # Index by to_team_id for fast lookup
    lookup = {}
    for _, row in combined.iterrows():
        for team_id in [row.get("to_team_id"), row.get("from_team_id")]:
            if pd.notna(team_id):
                tid = int(team_id)
                if tid not in lookup:
                    lookup[tid] = []
                lookup[tid].append(row.to_dict())

    return lookup


def get_recent_trades(transaction_lookup: dict, team_name: str,
                      game_date, lookback_days: int = 30) -> dict:
    """
    Get trades/acquisitions for a team in the last N days before the game.
    Returns features about roster disruption.
    """
    result = {
        "trade_disruption":  0,   # number of significant acquisitions
        "roster_stability":  1.0, # 1.0 = fully stable, 0.0 = everyone is new
        "days_since_trade":  30,  # days since last trade (30+ = no recent trades)
    }

    team_id = get_team_id(team_name)
    if not team_id or team_id not in transaction_lookup:
        return result

    game_date = pd.Timestamp(game_date)
    cutoff    = game_date - timedelta(days=lookback_days)

    transactions = transaction_lookup[team_id]
    recent = [
        t for t in transactions
        if pd.notna(t.get("effective_date"))
        and cutoff <= pd.Timestamp(t["effective_date"]) < game_date
        and t.get("to_team_id") == team_id  # incoming only
        and any(kw in str(t.get("transaction_type", "")).lower()
                for kw in ["trade", "claim", "purchase", "selected"])
    ]

    if recent:
        result["trade_disruption"] = len(recent)
        result["roster_stability"] = max(0.0, 1.0 - len(recent) / 5.0)
        most_recent = max(pd.Timestamp(t["effective_date"]) for t in recent)
        result["days_since_trade"] = (game_date - most_recent).days

    return result


# ══════════════════════════════════════════════
# PLAYER GAME LOGS — for slump detection
# ══════════════════════════════════════════════

def fetch_player_game_log(player_id: int, season: int) -> pd.DataFrame:
    """
    Fetch game-by-game batting stats for a player in a given season.
    Used to detect slumps vs career average.
    """
    cache_key = f"gamelog_{player_id}_{season}"
    data = api_get(
        f"/people/{player_id}/stats",
        params={
            "stats":   "gameLog",
            "group":   "hitting",
            "season":  season,
            "sportId": 1,
        },
        cache_key=cache_key,
        cache_days=CACHE_DAYS["game_log"]
    )

    if not data:
        return pd.DataFrame()

    splits = []
    try:
        for split_group in data.get("stats", []):
            for split in split_group.get("splits", []):
                s = split.get("stat", {})
                splits.append({
                    "date":    split.get("date", ""),
                    "avg":     float(s.get("avg", 0) or 0),
                    "obp":     float(s.get("obp", 0) or 0),
                    "slg":     float(s.get("slg", 0) or 0),
                    "ops":     float(s.get("ops", 0) or 0),
                    "hits":    int(s.get("hits", 0) or 0),
                    "atBats":  int(s.get("atBats", 0) or 0),
                    "homeRuns":int(s.get("homeRuns", 0) or 0),
                    "rbi":     int(s.get("rbi", 0) or 0),
                    "strikeOuts": int(s.get("strikeOuts", 0) or 0),
                })
    except Exception:
        return pd.DataFrame()

    if not splits:
        return pd.DataFrame()

    df = pd.DataFrame(splits)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_player_career_stats(player_id: int) -> dict:
    """Fetch career batting averages for a player — used as the baseline."""
    cache_key = f"career_{player_id}"
    data = api_get(
        f"/people/{player_id}/stats",
        params={"stats": "career", "group": "hitting", "sportId": 1},
        cache_key=cache_key,
        cache_days=30
    )

    if not data:
        return {}

    try:
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                s = split.get("stat", {})
                return {
                    "career_avg": float(s.get("avg", 0.250) or 0.250),
                    "career_obp": float(s.get("obp", 0.320) or 0.320),
                    "career_slg": float(s.get("slg", 0.400) or 0.400),
                    "career_ops": float(s.get("ops", 0.720) or 0.720),
                    "career_k_rate": (float(s.get("strikeOuts", 0) or 0) /
                                      max(float(s.get("atBats", 1) or 1), 1)),
                }
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════
# SLUMP DETECTION
# 2 standard deviations below rolling mean = slump
# 2 standard deviations above = hot streak
# ══════════════════════════════════════════════

def detect_player_form(player_id: int, season: int,
                        game_date, window: int = 15) -> dict:
    """
    Detect if a player is in a slump or hot streak as of game_date.

    Method:
      1. Get last `window` games before game_date
      2. Calculate rolling OPS mean and std
      3. Compare current last-7-game OPS to rolling baseline
      4. Flag slump if > 2 std devs below, hot streak if > 2 std above

    Returns:
      form_score:   positive = hot, negative = slumping, 0 = neutral
      slump:        True if statistically slumping
      hot_streak:   True if statistically hot
      ops_z_score:  z-score of recent OPS vs rolling baseline
    """
    result = {
        "form_score":  0.0,
        "slump":       False,
        "hot_streak":  False,
        "ops_z_score": 0.0,
        "recent_ops":  np.nan,
        "career_ops":  np.nan,
    }

    game_log = fetch_player_game_log(player_id, season)
    if game_log.empty or len(game_log) < 7:
        return result

    game_date = pd.Timestamp(game_date)
    past = game_log[game_log["date"] < game_date].tail(window)

    if len(past) < 7:
        return result

    # Baseline: all games in the window
    baseline_ops = past["ops"].replace(0, np.nan).dropna()
    if len(baseline_ops) < 5:
        return result

    mean_ops = baseline_ops.mean()
    std_ops  = baseline_ops.std()

    # Recent form: last 7 games
    recent = past.tail(7)
    recent_ops = recent["ops"].replace(0, np.nan).dropna().mean()

    if std_ops < 0.001:   # avoid division by zero
        return result

    z_score = (recent_ops - mean_ops) / std_ops

    result["ops_z_score"]  = round(z_score, 3)
    result["recent_ops"]   = round(recent_ops, 3)
    result["career_ops"]   = round(mean_ops, 3)
    result["form_score"]   = round(z_score, 3)

    # 2 std dev threshold
    result["slump"]      = z_score < -2.0
    result["hot_streak"] = z_score > 2.0

    return result


# ══════════════════════════════════════════════
# ROSTER & POSITION CHANGES
# ══════════════════════════════════════════════

def fetch_team_roster(team_id: int, date_str: str) -> pd.DataFrame:
    """Fetch a team's active roster on a given date."""
    cache_key = f"roster_{team_id}_{date_str}"
    data = api_get(
        f"/teams/{team_id}/roster",
        params={"rosterType": "active", "date": date_str, "season": date_str[:4]},
        cache_key=cache_key,
        cache_days=CACHE_DAYS["roster"]
    )

    if not data or "roster" not in data:
        return pd.DataFrame()

    records = []
    for p in data["roster"]:
        records.append({
            "player_id":       p.get("person", {}).get("id"),
            "player_name":     p.get("person", {}).get("fullName", ""),
            "position":        p.get("position", {}).get("abbreviation", ""),
            "position_type":   p.get("position", {}).get("type", ""),
            "status":          p.get("status", {}).get("description", ""),
            "jersey_number":   p.get("jerseyNumber", ""),
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


def detect_position_changes(team_name: str, game_date,
                             lookback_days: int = 14) -> dict:
    """
    Detect players playing out of their primary position.
    Uses roster data to see if position assignments have changed recently.

    Returns:
      position_change_count: number of players in non-primary positions
      defense_disruption:    estimated defensive impact (0-1 scale)
    """
    result = {"position_change_count": 0, "defense_disruption": 0.0}

    team_id = get_team_id(team_name)
    if not team_id:
        return result

    game_date   = pd.Timestamp(game_date)
    current_str = game_date.strftime("%Y-%m-%d")
    prior_str   = (game_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    current = fetch_team_roster(team_id, current_str)
    prior   = fetch_team_roster(team_id, prior_str)

    if current.empty or prior.empty:
        return result

    # Find players who changed positions
    merged = current.merge(
        prior[["player_id", "position"]].rename(columns={"position": "prior_position"}),
        on="player_id", how="left"
    )
    merged["position_changed"] = (
        merged["position"].notna() &
        merged["prior_position"].notna() &
        (merged["position"] != merged["prior_position"])
    )

    changes = merged["position_changed"].sum()
    result["position_change_count"] = int(changes)

    # Defensive positions weighted by importance
    # C, SS, CF are hardest to replace — weight them more
    defensive_weights = {"C": 1.0, "SS": 0.9, "CF": 0.8, "2B": 0.7,
                          "3B": 0.6, "LF": 0.4, "RF": 0.4, "1B": 0.3, "DH": 0.0}
    disruption = 0.0
    for _, row in merged[merged["position_changed"]].iterrows():
        disruption += defensive_weights.get(row.get("prior_position", ""), 0.3)

    result["defense_disruption"] = min(1.0, disruption / 3.0)  # normalize 0-1
    return result


# ══════════════════════════════════════════════
# LINEUP FORM AGGREGATOR
# Combines all player-level signals into team-level features
# ══════════════════════════════════════════════

def get_lineup_form(team_name: str, game_date, season: int,
                    transaction_lookup: dict,
                    top_n_batters: int = 6) -> dict:
    """
    Aggregate player-level form signals for the top batters on a team.
    Uses top N batters by PA from pybaseball data if available,
    otherwise fetches directly from MLB Stats API.

    Returns a dict of team-level form features.
    """
    team_id = get_team_id(team_name)
    result  = {
        "lineup_slump_score":    0.0,   # count of players slumping
        "lineup_hot_score":      0.0,   # count of players hot
        "net_form_score":        0.0,   # hot minus slumping
        "avg_ops_z_score":       0.0,   # avg z-score across lineup
        "slump_severity":        0.0,   # avg deviation for slumping players
        "trade_disruption":      0.0,
        "roster_stability":      1.0,
        "days_since_trade":      30,
        "position_change_count": 0,
        "defense_disruption":    0.0,
    }

    if not team_id:
        return result

    # ── Trade features ──
    trade_feats = get_recent_trades(transaction_lookup, team_name, game_date)
    result.update(trade_feats)

    # ── Position change features ──
    pos_feats = detect_position_changes(team_name, game_date)
    result.update(pos_feats)

    # ── Player form — fetch roster and check each player ──
    date_str = pd.Timestamp(game_date).strftime("%Y-%m-%d")
    roster   = fetch_team_roster(team_id, date_str)

    if roster.empty:
        return result

    # Focus on batters (not pitchers)
    batters = roster[~roster["position"].isin(["P", "RP", "SP"])].head(top_n_batters)

    z_scores      = []
    slump_count   = 0
    hot_count     = 0
    slump_z_total = 0.0

    for _, player in batters.iterrows():
        pid  = player.get("player_id")
        if not pid:
            continue

        form = detect_player_form(int(pid), season, game_date)
        z    = form.get("ops_z_score", 0.0)

        if not np.isnan(z):
            z_scores.append(z)
            if form.get("slump"):
                slump_count   += 1
                slump_z_total += abs(z)
            if form.get("hot_streak"):
                hot_count += 1

    if z_scores:
        result["lineup_slump_score"] = float(slump_count)
        result["lineup_hot_score"]   = float(hot_count)
        result["net_form_score"]     = float(hot_count - slump_count)
        result["avg_ops_z_score"]    = float(np.mean(z_scores))
        result["slump_severity"]     = (float(slump_z_total / slump_count)
                                        if slump_count > 0 else 0.0)

    return result


# ══════════════════════════════════════════════
# TRADED PLAYER STAT CARRY-OVER
# When a player is traded, carry their stats with them
# ══════════════════════════════════════════════

def get_traded_player_context(player_id: int, trade_date,
                               season: int) -> dict:
    """
    For a recently traded player, compute:
      - Their stats before the trade (at old team)
      - Whether they're likely in an adjustment period (first 2 weeks)
      - Career stats as the stable baseline

    This lets the model know: "this team just acquired a .285 hitter
    but he's only been here 5 days — adjustment penalty applies"
    """
    result = {
        "pre_trade_ops":       np.nan,
        "career_ops":          np.nan,
        "days_since_acquired": 30,
        "in_adjustment":       False,   # True = first 14 days at new team
    }

    career = fetch_player_career_stats(player_id)
    result["career_ops"] = career.get("career_ops", np.nan)

    game_log = fetch_player_game_log(player_id, season)
    if game_log.empty:
        return result

    trade_date = pd.Timestamp(trade_date)
    pre_trade  = game_log[game_log["date"] < trade_date].tail(15)

    if not pre_trade.empty:
        result["pre_trade_ops"] = pre_trade["ops"].replace(0, np.nan).dropna().mean()

    result["days_since_acquired"] = 30   # will be set by caller
    result["in_adjustment"]       = False  # will be set by caller

    return result


# ══════════════════════════════════════════════
# MAIN FEATURE BUILDER
# Called once per game to get all player intelligence features
# ══════════════════════════════════════════════

def build_player_features(home_team: str, away_team: str,
                           game_date, season: int,
                           transaction_lookup: dict) -> dict:
    """
    Build all player intelligence features for a single game.
    This is the main entry point called from the feature pipeline.
    """
    game_date = pd.Timestamp(game_date)

    home_feats = get_lineup_form(home_team, game_date, season, transaction_lookup)
    away_feats = get_lineup_form(away_team, game_date, season, transaction_lookup)

    features = {}

    for prefix, feats in [("home", home_feats), ("away", away_feats)]:
        for k, v in feats.items():
            features[f"{prefix}_{k}"] = v

    # Matchup edges (home advantage over away)
    features["form_edge"]            = (home_feats["net_form_score"] -
                                        away_feats["net_form_score"])
    features["slump_edge"]           = (away_feats["lineup_slump_score"] -
                                        home_feats["lineup_slump_score"])
    features["stability_edge"]       = (home_feats["roster_stability"] -
                                        away_feats["roster_stability"])
    features["defense_disruption_edge"] = (away_feats["defense_disruption"] -
                                            home_feats["defense_disruption"])

    return features


# ══════════════════════════════════════════════
# BATCH BUILDER — attaches features to full DataFrame
# ══════════════════════════════════════════════

def attach_player_intelligence(df: pd.DataFrame,
                                transaction_lookup: dict,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Attach player intelligence features to every game in the DataFrame.
    Processes games in order — API calls are cached so subsequent runs are fast.

    Args:
        df:                   game DataFrame (one row per game)
        transaction_lookup:   from build_transaction_lookup()
        verbose:              print progress

    Returns:
        DataFrame with additional player intelligence columns
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    n     = len(df)
    rows  = []
    cache = {}   # in-memory cache for repeated team/date combinations

    for i, (_, game) in enumerate(df.iterrows()):
        if verbose and i % 100 == 0:
            pct = (i + 1) / n * 100
            print(f"    Player intelligence: {pct:.0f}% ({i+1}/{n})", end="\r")

        season = game["date"].year
        key_h  = (game["home_team"], str(game["date"].date()), season)
        key_a  = (game["away_team"], str(game["date"].date()), season)

        # Use cache to avoid re-fetching same team on same day
        if key_h not in cache:
            cache[key_h] = get_lineup_form(
                game["home_team"], game["date"], season, transaction_lookup
            )
        if key_a not in cache:
            cache[key_a] = get_lineup_form(
                game["away_team"], game["date"], season, transaction_lookup
            )

        home_f = cache[key_h]
        away_f = cache[key_a]

        row = game.to_dict()
        for prefix, feats in [("pi_home", home_f), ("pi_away", away_f)]:
            for k, v in feats.items():
                row[f"{prefix}_{k}"] = v

        row["pi_form_edge"]      = home_f["net_form_score"]  - away_f["net_form_score"]
        row["pi_slump_edge"]     = away_f["lineup_slump_score"] - home_f["lineup_slump_score"]
        row["pi_stability_edge"] = home_f["roster_stability"] - away_f["roster_stability"]
        row["pi_defense_edge"]   = away_f["defense_disruption"] - home_f["defense_disruption"]
        rows.append(row)

    if verbose:
        print()   # newline after progress

    out = pd.DataFrame(rows)
    print(f"  ✅ Player intelligence attached. "
          f"Non-null rate: {out[PLAYER_INTEL_COLS].notna().mean().mean():.1%}")
    return out


# ══════════════════════════════════════════════
# FEATURE COLUMN LIST
# ══════════════════════════════════════════════

PLAYER_INTEL_COLS = [
    # Home team form
    "pi_home_lineup_slump_score", "pi_home_lineup_hot_score",
    "pi_home_net_form_score", "pi_home_avg_ops_z_score",
    "pi_home_slump_severity",
    # Away team form
    "pi_away_lineup_slump_score", "pi_away_lineup_hot_score",
    "pi_away_net_form_score", "pi_away_avg_ops_z_score",
    "pi_away_slump_severity",
    # Trade / roster stability
    "pi_home_trade_disruption", "pi_home_roster_stability",
    "pi_home_days_since_trade",
    "pi_away_trade_disruption", "pi_away_roster_stability",
    "pi_away_days_since_trade",
    # Position changes
    "pi_home_position_change_count", "pi_home_defense_disruption",
    "pi_away_position_change_count", "pi_away_defense_disruption",
    # Matchup edges
    "pi_form_edge", "pi_slump_edge",
    "pi_stability_edge", "pi_defense_edge",
]
