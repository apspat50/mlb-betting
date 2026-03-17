"""
matchup_features.py
-------------------
Builds advanced matchup and situational features from:
  1. Your existing SBR odds files (innings, dates, pitcher handedness)
  2. pybaseball (individual batter stats, platoon splits, bullpen)

Features added:
  - Pitcher handedness & platoon splits (L/R matchup advantage)
  - Bullpen ERA/performance (built from late-inning scores)
  - Recent pitcher form (last 3 starts vs season avg)
  - Days of rest
  - Home/away streak (travel fatigue)
  - Early vs late inning run scoring patterns
  - Head to head record this season
  - Individual batter stats (top of lineup quality)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════
# 1. PITCHER HANDEDNESS (from SBR name format)
# ══════════════════════════════════════════════

def extract_handedness(sbr_name: str) -> str:
    """Extract L or R from pitcher name like JLESTER-L or CSALE-L."""
    if pd.isna(sbr_name) or "-" not in str(sbr_name):
        return "U"  # unknown
    return str(sbr_name).split("-")[-1].strip().upper()


# ══════════════════════════════════════════════
# 2. PLATOON SPLITS (from game results)
# Built entirely from your existing odds data.
# Tracks how each team scores vs LHP vs RHP.
# ══════════════════════════════════════════════

def build_platoon_splits(df: pd.DataFrame, window: int = 20) -> dict:
    """
    For each team, track rolling runs scored against LHP vs RHP separately.
    Returns dict: team -> DataFrame indexed by date with:
      - avg_vs_lhp_L20: average runs scored in last 20 games vs lefty starters
      - avg_vs_rhp_L20: average runs scored in last 20 games vs righty starters
      - lhp_games_L20:  how many of last 20 were vs lefties (sample size)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Extract pitcher handedness
    df["home_hand"] = df["home_pitcher"].apply(extract_handedness)
    df["away_hand"] = df["away_pitcher"].apply(extract_handedness)

    teams = set(df["home_team"]) | set(df["away_team"])
    platoon_stats = {}

    for team in teams:
        records = []

        # Games where team is home (faces away pitcher)
        home_g = df[df["home_team"] == team].copy()
        home_g["runs"]         = home_g["home_score"]
        home_g["opp_hand"]     = home_g["away_hand"]
        home_g["team"]         = team

        # Games where team is away (faces home pitcher)
        away_g = df[df["away_team"] == team].copy()
        away_g["runs"]         = away_g["away_score"]
        away_g["opp_hand"]     = away_g["home_hand"]
        away_g["team"]         = team

        combined = pd.concat([
            home_g[["date", "runs", "opp_hand"]],
            away_g[["date", "runs", "opp_hand"]]
        ]).sort_values("date").reset_index(drop=True)

        # Rolling stats vs LHP
        vs_lhp = combined[combined["opp_hand"] == "L"]["runs"].copy()
        vs_rhp = combined[combined["opp_hand"] == "R"]["runs"].copy()

        # Re-index to fill in all dates
        combined["is_vs_lhp"] = (combined["opp_hand"] == "L").astype(float)
        combined["runs_vs_lhp"] = combined["runs"].where(combined["opp_hand"] == "L")
        combined["runs_vs_rhp"] = combined["runs"].where(combined["opp_hand"] == "R")

        combined[f"avg_vs_lhp_L{window}"] = (
            combined["runs_vs_lhp"].shift(1)
            .rolling(window, min_periods=3).mean()
        )
        combined[f"avg_vs_rhp_L{window}"] = (
            combined["runs_vs_rhp"].shift(1)
            .rolling(window, min_periods=3).mean()
        )
        combined[f"lhp_games_L{window}"] = (
            combined["is_vs_lhp"].shift(1)
            .rolling(window, min_periods=3).sum()
        )

        # Platoon advantage score (positive = team better vs LHP)
        combined["platoon_adv"] = (
            combined[f"avg_vs_lhp_L{window}"] - combined[f"avg_vs_rhp_L{window}"]
        )

        platoon_stats[team] = combined.set_index("date")

    return platoon_stats


def get_platoon_feature(platoon_stats: dict, team: str,
                        date, opp_pitcher_hand: str) -> dict:
    """
    Get platoon advantage for a team facing a specific pitcher handedness.
    Returns dict of features.
    """
    result = {
        "avg_vs_lhp": np.nan,
        "avg_vs_rhp": np.nan,
        "platoon_adv": np.nan,
        "platoon_edge_this_game": np.nan,  # runs expected vs today's pitcher hand
    }

    if team not in platoon_stats:
        return result

    ts   = platoon_stats[team]
    past = ts[ts.index <= date]
    if len(past) == 0:
        return result

    row = past.iloc[-1]
    result["avg_vs_lhp"]  = row.get("avg_vs_lhp_L20", np.nan)
    result["avg_vs_rhp"]  = row.get("avg_vs_rhp_L20", np.nan)
    result["platoon_adv"] = row.get("platoon_adv", np.nan)

    # Edge this specific game based on today's pitcher handedness
    if opp_pitcher_hand == "L":
        result["platoon_edge_this_game"] = row.get("avg_vs_lhp_L20", np.nan)
    elif opp_pitcher_hand == "R":
        result["platoon_edge_this_game"] = row.get("avg_vs_rhp_L20", np.nan)

    return result


# ══════════════════════════════════════════════
# 3. BULLPEN PERFORMANCE (from late-inning scores)
# Built from inning-by-inning data in your SBR files.
# Late innings = 7th, 8th, 9th (bullpen territory)
# ══════════════════════════════════════════════

def build_bullpen_stats(df_raw: pd.DataFrame, window: int = 10) -> dict:
    """
    Build bullpen ERA proxy from late-inning run prevention.
    Uses the raw SBR DataFrame (before pairing into one game per row)
    to access individual inning columns.

    Returns dict: team -> DataFrame with rolling bullpen stats.
    """
    # We need the raw inning columns — check if they exist
    inning_cols = [str(i) for i in range(1, 10)]
    has_innings  = all(c in df_raw.columns for c in ["7th", "8th"])

    # Handle both numeric and named inning columns
    col_7  = "7th" if "7th" in df_raw.columns else "7"
    col_8  = "8th" if "8th" in df_raw.columns else "8"
    col_9  = "9th" if "9th" in df_raw.columns else "9"
    col_e1 = "1st" if "1st" in df_raw.columns else "1"
    col_e2 = "2nd" if "2nd" in df_raw.columns else "2"
    col_e3 = "3rd" if "3rd" in df_raw.columns else "3"

    if col_7 not in df_raw.columns:
        return {}  # No inning data available

    df_raw = df_raw.copy()
    df_raw["date_parsed"] = pd.to_datetime(df_raw["date_parsed"])

    def safe_num(x):
        try:
            v = float(str(x).replace("x", "0"))
            return v if not np.isnan(v) else 0
        except:
            return 0

    for c in [col_7, col_8, col_9, col_e1, col_e2, col_e3]:
        df_raw[c] = df_raw[c].apply(safe_num)

    # Late innings = bullpen; early innings = starter
    df_raw["late_runs"]  = df_raw[col_7] + df_raw[col_8] + df_raw[col_9]
    df_raw["early_runs"] = df_raw[col_e1] + df_raw[col_e2] + df_raw[col_e3]

    teams = df_raw["team_name"].unique()
    bullpen_stats = {}

    for team in teams:
        team_rows = df_raw[df_raw["team_name"] == team].sort_values("date_parsed").copy()

        # Runs allowed in late innings = bullpen gave up
        # For home team rows: late_runs = runs they scored (offense)
        # We want runs the OPPOSING bullpen gave up = our late inning scoring
        team_rows["bullpen_allowed"] = team_rows["late_runs"]  # runs this team scored late
        team_rows["starter_allowed"] = team_rows["early_runs"]

        team_rows[f"bullpen_avg_L{window}"] = (
            team_rows["bullpen_allowed"].shift(1)
            .rolling(window, min_periods=3).mean()
        )
        team_rows[f"starter_avg_L{window}"] = (
            team_rows["starter_allowed"].shift(1)
            .rolling(window, min_periods=3).mean()
        )
        team_rows["late_early_ratio"] = (
            team_rows[f"bullpen_avg_L{window}"] /
            (team_rows[f"starter_avg_L{window}"] + 0.1)
        )

        bullpen_stats[team] = team_rows.set_index("date_parsed")

    return bullpen_stats


# ══════════════════════════════════════════════
# 4. RECENT PITCHER FORM (last 3 vs season avg)
# ══════════════════════════════════════════════

def build_pitcher_form(df: pd.DataFrame) -> dict:
    """
    Compare pitcher's last 3 starts vs their season average.
    Positive = pitcher is on a hot streak (below their avg ERA)
    Negative = pitcher is struggling (above their avg ERA)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    pitcher_records = {}

    for _, game in df.iterrows():
        for pitcher, runs_allowed in [
            (game["home_pitcher"], game["away_score"]),
            (game["away_pitcher"], game["home_score"]),
        ]:
            if pd.isna(pitcher) or pitcher == "TBD":
                continue
            if pitcher not in pitcher_records:
                pitcher_records[pitcher] = []
            pitcher_records[pitcher].append({
                "date":         game["date"],
                "runs_allowed": float(runs_allowed),
            })

    pitcher_form = {}
    for pitcher, records in pitcher_records.items():
        p = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        p["season_avg_era"]    = p["runs_allowed"].shift(1).expanding().mean()
        p["recent_avg_L3"]     = p["runs_allowed"].shift(1).rolling(3,  min_periods=2).mean()
        p["recent_avg_L5"]     = p["runs_allowed"].shift(1).rolling(5,  min_periods=3).mean()
        # Form score: negative = pitcher is better recently than season avg (hot)
        p["form_vs_season"]    = p["recent_avg_L3"] - p["season_avg_era"]
        p["form_trend"]        = p["recent_avg_L3"] - p["recent_avg_L5"]  # getting better or worse
        p["start_num"]         = range(len(p))

        pitcher_form[pitcher]  = p.set_index("date")

    return pitcher_form


def get_pitcher_form(pitcher_form: dict, pitcher: str, date) -> dict:
    result = {
        "form_vs_season": np.nan,
        "form_trend":     np.nan,
        "recent_avg_L3":  np.nan,
        "season_avg_era": np.nan,
    }
    if pitcher not in pitcher_form:
        return result
    past = pitcher_form[pitcher]
    past = past[past.index <= date]
    if len(past) < 3:
        return result
    row = past.iloc[-1]
    for k in result:
        result[k] = row.get(k, np.nan)
    return result


# ══════════════════════════════════════════════
# 5. DAYS OF REST & TRAVEL
# ══════════════════════════════════════════════

def build_rest_and_travel(df: pd.DataFrame) -> dict:
    """
    For each team, calculate:
      - days_rest: days since last game
      - home_streak: consecutive home games (positive) or away games (negative)
      - games_last_7: workload — how many games in last 7 days
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    teams = set(df["home_team"]) | set(df["away_team"])
    rest_stats = {}

    for team in teams:
        home_g = df[df["home_team"] == team][["date"]].copy()
        away_g = df[df["away_team"] == team][["date"]].copy()
        home_g["is_home"] = 1
        away_g["is_home"] = 0

        all_g = pd.concat([home_g, away_g]).sort_values("date").reset_index(drop=True)
        all_g["days_rest"] = all_g["date"].diff().dt.days.fillna(3).clip(1, 10)

        # Home streak: consecutive home or away games
        streaks = []
        current = 0
        for _, r in all_g.iterrows():
            if r["is_home"] == 1:
                current = max(1, current + 1)
            else:
                current = min(-1, current - 1)
            streaks.append(current)
        all_g["home_away_streak"] = streaks

        # Games in last 7 days (workload / fatigue)
        all_g["games_last_7"] = 0
        for i, row in all_g.iterrows():
            cutoff = row["date"] - pd.Timedelta(days=7)
            all_g.at[i, "games_last_7"] = len(all_g[
                (all_g["date"] >= cutoff) & (all_g["date"] < row["date"])
            ])

        rest_stats[team] = all_g.set_index("date")

    return rest_stats


def get_rest_features(rest_stats: dict, team: str, date) -> dict:
    result = {"days_rest": 3.0, "home_away_streak": 0, "games_last_7": 5}
    if team not in rest_stats:
        return result
    past = rest_stats[team]
    past = past[past.index <= date]
    if len(past) == 0:
        return result
    row = past.iloc[-1]
    result["days_rest"]        = row.get("days_rest", 3.0)
    result["home_away_streak"] = row.get("home_away_streak", 0)
    result["games_last_7"]     = row.get("games_last_7", 5)
    return result


# ══════════════════════════════════════════════
# 6. EARLY vs LATE INNING SCORING PATTERNS
# Built from inning-by-inning data
# ══════════════════════════════════════════════

def build_inning_patterns(df: pd.DataFrame, window: int = 10) -> dict:
    """
    Track how teams score across innings:
      - early_avg: avg runs scored in innings 1-3 (starter matchup)
      - middle_avg: avg runs scored in innings 4-6
      - late_avg: avg runs scored in innings 7-9 (bullpen)
      - comeback_rate: how often team scores after being down after 5
    """
    # Check for inning cols
    inning_map = {}
    for i, names in enumerate(["1st","2nd","3rd","4th","5th","6th","7th","8th","9th"], 1):
        for n in [names, str(i)]:
            if n in df.columns:
                inning_map[i] = n
                break

    if len(inning_map) < 6:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def safe_num(x):
        try:
            return float(str(x).replace("x","0"))
        except:
            return 0.0

    for i, col in inning_map.items():
        df[col] = df[col].apply(safe_num)

    teams = set(df["home_team"]) | set(df["away_team"])
    inning_stats = {}

    for team in teams:
        home_g = df[df["home_team"] == team].copy()
        away_g = df[df["away_team"] == team].copy()

        records = []
        for _, g in home_g.iterrows():
            early  = sum(safe_num(g.get(inning_map.get(i,""))) for i in [1,2,3])
            middle = sum(safe_num(g.get(inning_map.get(i,""))) for i in [4,5,6])
            late   = sum(safe_num(g.get(inning_map.get(i,""))) for i in [7,8,9] if i in inning_map)
            records.append({"date": g["date"], "early": early, "middle": middle, "late": late})

        for _, g in away_g.iterrows():
            early  = sum(safe_num(g.get(inning_map.get(i,""))) for i in [1,2,3])
            middle = sum(safe_num(g.get(inning_map.get(i,""))) for i in [4,5,6])
            late   = sum(safe_num(g.get(inning_map.get(i,""))) for i in [7,8,9] if i in inning_map)
            records.append({"date": g["date"], "early": early, "middle": middle, "late": late})

        p = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        for phase in ["early", "middle", "late"]:
            p[f"avg_{phase}_L{window}"] = (
                p[phase].shift(1).rolling(window, min_periods=3).mean()
            )

        # Late inning clutch score = late runs above average
        p["late_clutch"] = p[f"avg_late_L{window}"] - p[f"avg_early_L{window}"]

        inning_stats[team] = p.set_index("date")

    return inning_stats


def get_inning_features(inning_stats: dict, team: str, date) -> dict:
    result = {
        "avg_early_runs":  np.nan,
        "avg_middle_runs": np.nan,
        "avg_late_runs":   np.nan,
        "late_clutch":     np.nan,
    }
    if team not in inning_stats:
        return result
    past = inning_stats[team]
    past = past[past.index <= date]
    if len(past) == 0:
        return result
    row = past.iloc[-1]
    result["avg_early_runs"]  = row.get("avg_early_L10",  np.nan)
    result["avg_middle_runs"] = row.get("avg_middle_L10", np.nan)
    result["avg_late_runs"]   = row.get("avg_late_L10",   np.nan)
    result["late_clutch"]     = row.get("late_clutch",    np.nan)
    return result


# ══════════════════════════════════════════════
# 7. HEAD TO HEAD RECORD THIS SEASON
# ══════════════════════════════════════════════

def build_h2h(df: pd.DataFrame) -> dict:
    """
    Track head-to-head record between every pair of teams within a season.
    Returns dict: (home_team, away_team) -> DataFrame with rolling H2H win rate.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    df = df.sort_values("date")

    h2h = {}
    for _, game in df.iterrows():
        key = (game["home_team"], game["away_team"], game["season"])
        if key not in h2h:
            h2h[key] = []
        h2h[key].append({
            "date":     game["date"],
            "home_win": game["home_win"],
        })

    h2h_stats = {}
    for key, records in h2h.items():
        p = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        p["h2h_home_win_rate"] = p["home_win"].shift(1).expanding().mean()
        p["h2h_games_played"]  = range(len(p))
        h2h_stats[key] = p.set_index("date")

    return h2h_stats


def get_h2h_features(h2h_stats: dict, home_team: str,
                      away_team: str, date, season: int) -> dict:
    result = {"h2h_home_win_rate": np.nan, "h2h_games_played": 0}
    key = (home_team, away_team, season)
    if key not in h2h_stats:
        return result
    past = h2h_stats[key]
    past = past[past.index < date]  # strictly before this game
    if len(past) == 0:
        return result
    row = past.iloc[-1]
    result["h2h_home_win_rate"] = row.get("h2h_home_win_rate", np.nan)
    result["h2h_games_played"]  = int(row.get("h2h_games_played", 0))
    return result


# ══════════════════════════════════════════════
# 8. INDIVIDUAL BATTER STATS (via pybaseball)
# ══════════════════════════════════════════════

def fetch_top_batters(seasons: list, min_pa: int = 100) -> pd.DataFrame:
    """
    Pull individual batter stats from FanGraphs.
    We use team-level aggregations of their top batters.
    Key stats: wRC+, OBP, SLG, ISO, K%, BB%, wOBA, Hard%
    """
    cache_file = CACHE_DIR / f"batters_{'_'.join(map(str,seasons))}.csv"
    if cache_file.exists():
        print("  📂 Batter stats loaded from cache")
        return pd.read_csv(cache_file)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        print("  ⚠️  pybaseball not installed: pip install pybaseball")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            print(f"  ⬇️  Downloading batter stats {season}...")
            df = pyb.batting_stats(season, qual=min_pa)
            df["season"] = season
            frames.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  Could not fetch batter stats {season}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache_file, index=False)
    print(f"  ✅ Batter stats saved ({len(result)} players)")
    return result


def build_lineup_strength(batter_df: pd.DataFrame) -> dict:
    """
    Aggregate individual batter stats to team-season level.
    Takes the top 9 batters by PA for each team/season and averages their stats.
    This gives a better picture of lineup quality than just team totals.

    Returns dict: (team_fg_abbrev, season) -> aggregated lineup stats
    """
    from stats_fetcher import TEAM_TO_FG
    FG_TO_TEAM = {v: k for k, v in TEAM_TO_FG.items()}

    if batter_df.empty:
        return {}

    lineup_stats = {}
    stat_cols = ["wRC+", "OBP", "SLG", "ISO", "K%", "BB%", "wOBA"]
    stat_cols = [c for c in stat_cols if c in batter_df.columns]

    for (team, season), group in batter_df.groupby(["Team", "season"]):
        # Top 9 by PA = the core lineup
        top9 = group.nlargest(9, "PA") if "PA" in group.columns else group.head(9)
        entry = {}
        for col in stat_cols:
            vals = pd.to_numeric(top9[col], errors="coerce").dropna()
            if len(vals) > 0:
                safe = col.replace("%","pct").replace("+","plus").replace("/","_")
                entry[f"lineup_{safe}_mean"] = vals.mean()
                entry[f"lineup_{safe}_top3"] = vals.nlargest(3).mean()  # cleanup hitters

        # Depth score = ratio of top3 to bottom3 wRC+ (high = stacked at top)
        if "wRC+" in group.columns:
            wrc = pd.to_numeric(top9["wRC+"], errors="coerce").dropna()
            if len(wrc) >= 6:
                entry["lineup_depth_score"] = wrc.nlargest(3).mean() / (wrc.nsmallest(3).mean() + 1)

        lineup_stats[(team, season)] = entry

    return lineup_stats


# ══════════════════════════════════════════════
# 9. ASSEMBLE ALL MATCHUP FEATURES
# ══════════════════════════════════════════════

def build_all_matchup_features(df: pd.DataFrame,
                                df_raw: pd.DataFrame = None) -> pd.DataFrame:
    """
    Master function — builds all matchup features and attaches to game DataFrame.
    Call after build_features() in main.py.

    df:      game-level DataFrame (one row per game)
    df_raw:  raw SBR rows (V/H pairs, needed for bullpen/inning stats)
    """
    print("  Building platoon splits...")
    platoon_stats  = build_platoon_splits(df)

    print("  Building pitcher form...")
    pitcher_form   = build_pitcher_form(df)

    print("  Building rest & travel stats...")
    rest_stats     = build_rest_and_travel(df)

    print("  Building head-to-head records...")
    h2h_stats      = build_h2h(df)

    print("  Building inning patterns...")
    inning_stats   = build_inning_patterns(df)

    if df_raw is not None and not df_raw.empty:
        print("  Building bullpen stats from inning data...")
        bullpen_stats = build_bullpen_stats(df_raw)
    else:
        bullpen_stats = {}

    rows = []
    for _, game in df.iterrows():
        d         = pd.to_datetime(game["date"])
        home      = game["home_team"]
        away      = game["away_team"]
        season    = d.year
        home_hand = extract_handedness(game.get("home_pitcher", ""))
        away_hand = extract_handedness(game.get("away_pitcher", ""))

        row = game.to_dict()

        # ── Pitcher handedness ──
        row["home_pitcher_hand"] = 1 if home_hand == "L" else 0
        row["away_pitcher_hand"] = 1 if away_hand == "L" else 0
        row["same_hand_matchup"] = int(home_hand == away_hand)

        # ── Platoon splits ──
        # Home team faces away pitcher hand, away team faces home pitcher hand
        hp = get_platoon_feature(platoon_stats, home, d, away_hand)
        ap = get_platoon_feature(platoon_stats, away, d, home_hand)
        for k, v in hp.items():
            row[f"home_platoon_{k}"] = v
        for k, v in ap.items():
            row[f"away_platoon_{k}"] = v
        row["platoon_matchup_edge"] = (
            hp.get("platoon_edge_this_game", np.nan) -
            ap.get("platoon_edge_this_game", np.nan)
        )

        # ── Pitcher form ──
        hf = get_pitcher_form(pitcher_form, game.get("home_pitcher",""), d)
        af = get_pitcher_form(pitcher_form, game.get("away_pitcher",""), d)
        row["home_p_form_vs_season"] = hf["form_vs_season"]
        row["home_p_form_trend"]     = hf["form_trend"]
        row["home_p_recent_L3"]      = hf["recent_avg_L3"]
        row["away_p_form_vs_season"] = af["form_vs_season"]
        row["away_p_form_trend"]     = af["form_trend"]
        row["away_p_recent_L3"]      = af["recent_avg_L3"]
        # Combined form edge: negative = home pitcher is hotter
        row["pitcher_form_edge"] = (
            hf.get("form_vs_season", np.nan) - af.get("form_vs_season", np.nan)
        )

        # ── Rest & travel ──
        hr = get_rest_features(rest_stats, home, d)
        ar = get_rest_features(rest_stats, away, d)
        row["home_days_rest"]        = hr["days_rest"]
        row["home_home_streak"]      = hr["home_away_streak"]
        row["home_games_last_7"]     = hr["games_last_7"]
        row["away_days_rest"]        = ar["days_rest"]
        row["away_home_streak"]      = ar["home_away_streak"]
        row["away_games_last_7"]     = ar["games_last_7"]
        row["rest_advantage"]        = hr["days_rest"] - ar["days_rest"]
        row["fatigue_edge"]          = ar["games_last_7"] - hr["games_last_7"]

        # ── Inning patterns ──
        hi = get_inning_features(inning_stats, home, d)
        ai = get_inning_features(inning_stats, away, d)
        row["home_avg_early_runs"]   = hi["avg_early_runs"]
        row["home_avg_late_runs"]    = hi["avg_late_runs"]
        row["home_late_clutch"]      = hi["late_clutch"]
        row["away_avg_early_runs"]   = ai["avg_early_runs"]
        row["away_avg_late_runs"]    = ai["avg_late_runs"]
        row["away_late_clutch"]      = ai["late_clutch"]
        row["late_inning_edge"]      = (
            hi.get("avg_late_runs", np.nan) - ai.get("avg_late_runs", np.nan)
        )

        # ── Head to head ──
        h2h = get_h2h_features(h2h_stats, home, away, d, season)
        row["h2h_home_win_rate"]     = h2h["h2h_home_win_rate"]
        row["h2h_games_played"]      = h2h["h2h_games_played"]

        rows.append(row)

    out = pd.DataFrame(rows)
    non_null = out[MATCHUP_FEATURE_COLS].notna().mean().mean()
    print(f"  ✅ Matchup features built. Avg non-null rate: {non_null:.1%}")
    return out


# ── Feature column lists ──
MATCHUP_FEATURE_COLS = [
    # Handedness
    "home_pitcher_hand", "away_pitcher_hand", "same_hand_matchup",
    # Platoon
    "home_platoon_avg_vs_lhp", "home_platoon_avg_vs_rhp", "home_platoon_platoon_adv",
    "away_platoon_avg_vs_lhp", "away_platoon_avg_vs_rhp", "away_platoon_platoon_adv",
    "platoon_matchup_edge",
    # Pitcher form
    "home_p_form_vs_season", "home_p_form_trend", "home_p_recent_L3",
    "away_p_form_vs_season", "away_p_form_trend", "away_p_recent_L3",
    "pitcher_form_edge",
    # Rest & travel
    "home_days_rest", "home_home_streak", "home_games_last_7",
    "away_days_rest", "away_home_streak", "away_games_last_7",
    "rest_advantage", "fatigue_edge",
    # Inning patterns
    "home_avg_early_runs", "home_avg_late_runs", "home_late_clutch",
    "away_avg_early_runs", "away_avg_late_runs", "away_late_clutch",
    "late_inning_edge",
    # Head to head
    "h2h_home_win_rate", "h2h_games_played",
]
