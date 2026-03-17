"""
stats_fetcher.py
----------------
Pulls real advanced pitcher and team stats from pybaseball (FanGraphs data).
All stats are season-level and cached locally to avoid repeat downloads.

Stats this adds that the model didn't have before:
  PITCHER: ERA, FIP, xFIP, WHIP, K/9, BB/9, K%, BB%, HR/9, SwStr%, LOB%
  TEAM:    wRC+, OBP, SLG, wOBA, K%, BB%, ISO, BABIP, FIP, ERA

Install: pip install pybaseball
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Team name → FanGraphs abbreviation mapping ──
TEAM_TO_FG = {
    "Arizona Diamondbacks":    "ARI",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CHW",
    "Cincinnati Reds":         "CIN",
    "Cleveland Indians":       "CLE",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KCR",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "OAK",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SDP",
    "Seattle Mariners":        "SEA",
    "San Francisco Giants":    "SFG",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TBR",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSN",
}

# SBR pitcher name format: JLESTER-L → search "Lester"
def parse_pitcher_last_name(sbr_name: str) -> str:
    """Extract searchable last name from SBR format like JLESTER-L."""
    if pd.isna(sbr_name) or sbr_name == "TBD":
        return None
    # Remove handedness suffix
    name = sbr_name.split("-")[0]
    # First char is first initial, rest is last name
    if len(name) > 1:
        return name[1:].title()
    return name.title()


# ══════════════════════════════════════════════
# TEAM BATTING STATS
# ══════════════════════════════════════════════

def fetch_team_batting(seasons: list) -> pd.DataFrame:
    """
    Pull team batting stats from FanGraphs via pybaseball.
    Key stats: wRC+, OBP, SLG, wOBA, K%, BB%, ISO, BABIP
    """
    cache_file = CACHE_DIR / f"team_batting_{'_'.join(map(str, seasons))}.csv"
    if cache_file.exists():
        print(f"  📂 Team batting loaded from cache")
        return pd.read_csv(cache_file)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        print("  ⚠️  pybaseball not installed. Run: pip install pybaseball")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            print(f"  ⬇️  Downloading team batting {season}...")
            df = pyb.team_batting(season)
            df["season"] = season
            frames.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  Could not fetch team batting {season}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache_file, index=False)
    print(f"  ✅ Team batting saved ({len(result)} rows)")
    return result


def fetch_team_pitching(seasons: list) -> pd.DataFrame:
    """
    Pull team pitching stats from FanGraphs via pybaseball.
    Key stats: ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9
    """
    cache_file = CACHE_DIR / f"team_pitching_{'_'.join(map(str, seasons))}.csv"
    if cache_file.exists():
        print(f"  📂 Team pitching loaded from cache")
        return pd.read_csv(cache_file)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        print("  ⚠️  pybaseball not installed.")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            print(f"  ⬇️  Downloading team pitching {season}...")
            df = pyb.team_pitching(season)
            df["season"] = season
            frames.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  Could not fetch team pitching {season}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache_file, index=False)
    print(f"  ✅ Team pitching saved ({len(result)} rows)")
    return result


# ══════════════════════════════════════════════
# PITCHER STATS
# ══════════════════════════════════════════════

def fetch_pitcher_stats(seasons: list, min_ip: float = 20.0) -> pd.DataFrame:
    """
    Pull individual pitcher stats from FanGraphs.
    Filters to pitchers with at least min_ip innings pitched.
    Key stats: ERA, FIP, xFIP, WHIP, K/9, BB/9, K%, BB%, SwStr%, LOB%
    """
    cache_file = CACHE_DIR / f"pitcher_stats_{'_'.join(map(str, seasons))}.csv"
    if cache_file.exists():
        print(f"  📂 Pitcher stats loaded from cache")
        return pd.read_csv(cache_file)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        print("  ⚠️  pybaseball not installed.")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            print(f"  ⬇️  Downloading pitcher stats {season}...")
            # qual=1 means minimum 1 IP — we filter ourselves
            df = pyb.pitching_stats(season, qual=1)
            df["season"] = season
            if "IP" in df.columns:
                df = df[df["IP"] >= min_ip]
            frames.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  Could not fetch pitcher stats {season}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache_file, index=False)
    print(f"  ✅ Pitcher stats saved ({len(result)} rows)")
    return result


# ══════════════════════════════════════════════
# MERGE INTO GAME FEATURES
# ══════════════════════════════════════════════

def build_team_stat_lookup(team_batting: pd.DataFrame,
                            team_pitching: pd.DataFrame) -> dict:
    """
    Build a lookup dict: (team_fg_abbrev, season) -> stat dict
    Used to attach season-level stats to each game row.
    """
    lookup = {}

    # Batting stats we want
    bat_cols = ["wRC+", "OBP", "SLG", "wOBA", "BB%", "K%", "ISO", "BABIP"]

    if not team_batting.empty:
        for _, row in team_batting.iterrows():
            team = str(row.get("Team", ""))
            season = int(row.get("season", 0))
            key = (team, season)
            entry = {}
            for col in bat_cols:
                if col in row.index:
                    val = row[col]
                    # FanGraphs stores K% and BB% as decimals like 0.22
                    # normalize to percentage
                    if col in ["K%", "BB%"] and isinstance(val, float) and val < 1:
                        val = val * 100
                    entry[f"bat_{col.replace('%','pct').replace('/','_').replace('+','plus')}"] = val
            lookup[key] = lookup.get(key, {})
            lookup[key].update(entry)

    # Pitching stats we want
    pit_cols = ["ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9"]

    if not team_pitching.empty:
        for _, row in team_pitching.iterrows():
            team = str(row.get("Team", ""))
            season = int(row.get("season", 0))
            key = (team, season)
            entry = {}
            for col in pit_cols:
                if col in row.index:
                    entry[f"pit_{col.replace('/','_').replace('%','pct')}"] = row[col]
            lookup[key] = lookup.get(key, {})
            lookup[key].update(entry)

    return lookup


def build_pitcher_lookup(pitcher_stats: pd.DataFrame) -> dict:
    """
    Build lookup: (last_name_lower, season) -> stat dict
    Pitcher names in SBR are last-name only so we match on that.
    """
    lookup = {}
    if pitcher_stats.empty:
        return lookup

    p_cols = ["ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "K%", "BB%", "SwStr%", "LOB%"]

    for _, row in pitcher_stats.iterrows():
        name = str(row.get("Name", "")).strip()
        season = int(row.get("season", 0))
        if not name:
            continue
        # Store by last name (lowercase) for fuzzy matching with SBR format
        last = name.split()[-1].lower()
        key = (last, season)
        entry = {}
        for col in p_cols:
            if col in row.index and pd.notna(row[col]):
                safe = col.replace("/", "_").replace("%", "pct").replace("+", "plus")
                entry[f"sp_{safe}"] = row[col]
        # If multiple pitchers have same last name, keep the one with more IP
        if key not in lookup or row.get("IP", 0) > lookup[key].get("_ip", 0):
            entry["_ip"] = row.get("IP", 0)
            lookup[key] = entry

    return lookup


def get_team_advanced_stats(team_lookup: dict, team_name: str,
                             season: int, prefix: str) -> dict:
    """Look up season-level advanced stats for a team."""
    fg_abbrev = TEAM_TO_FG.get(team_name, "")
    key = (fg_abbrev, season)
    stats = team_lookup.get(key, {})
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def get_pitcher_advanced_stats(pitcher_lookup: dict, sbr_name: str,
                                season: int, prefix: str) -> dict:
    """Look up season-level advanced stats for a starting pitcher."""
    last = parse_pitcher_last_name(sbr_name)
    if not last:
        return {}
    key = (last.lower(), season)
    stats = pitcher_lookup.get(key, {})
    # Remove internal _ip key
    return {f"{prefix}_{k}": v for k, v in stats.items() if not k.startswith("_")}


# ══════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════

def load_advanced_stats(seasons: list) -> tuple:
    """
    Download (or load from cache) all advanced stats for given seasons.
    Returns (team_lookup, pitcher_lookup) dicts ready for feature merging.
    """
    print("\n  📊 Loading advanced stats from FanGraphs (via pybaseball)...")

    team_batting  = fetch_team_batting(seasons)
    team_pitching = fetch_team_pitching(seasons)
    pitcher_stats = fetch_pitcher_stats(seasons)

    team_lookup    = build_team_stat_lookup(team_batting, team_pitching)
    pitcher_lookup = build_pitcher_lookup(pitcher_stats)

    print(f"  ✅ Team stat entries:    {len(team_lookup)}")
    print(f"  ✅ Pitcher stat entries: {len(pitcher_lookup)}")

    return team_lookup, pitcher_lookup


# New feature columns added by advanced stats
ADVANCED_TEAM_COLS = [
    "home_bat_wRCplus", "home_bat_OBP", "home_bat_SLG",
    "home_bat_wOBA", "home_bat_Kpct", "home_bat_BBpct",
    "home_bat_ISO", "home_bat_BABIP",
    "home_pit_ERA", "home_pit_FIP", "home_pit_xFIP",
    "home_pit_WHIP", "home_pit_K_9", "home_pit_BB_9",
    "away_bat_wRCplus", "away_bat_OBP", "away_bat_SLG",
    "away_bat_wOBA", "away_bat_Kpct", "away_bat_BBpct",
    "away_bat_ISO", "away_bat_BABIP",
    "away_pit_ERA", "away_pit_FIP", "away_pit_xFIP",
    "away_pit_WHIP", "away_pit_K_9", "away_pit_BB_9",
]

ADVANCED_PITCHER_COLS = [
    "home_sp_ERA", "home_sp_FIP", "home_sp_xFIP",
    "home_sp_WHIP", "home_sp_K_9", "home_sp_BB_9",
    "home_sp_Kpct", "home_sp_BBpct", "home_sp_SwStrpct",
    "away_sp_ERA", "away_sp_FIP", "away_sp_xFIP",
    "away_sp_WHIP", "away_sp_K_9", "away_sp_BB_9",
    "away_sp_Kpct", "away_sp_BBpct", "away_sp_SwStrpct",
]

ADVANCED_EDGE_COLS = [
    "adv_wRC_edge", "adv_FIP_edge", "adv_xFIP_edge",
    "adv_sp_FIP_edge", "adv_sp_K9_edge", "adv_sp_SwStr_edge",
]

ALL_ADVANCED_COLS = ADVANCED_TEAM_COLS + ADVANCED_PITCHER_COLS + ADVANCED_EDGE_COLS
