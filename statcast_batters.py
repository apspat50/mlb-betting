"""
statcast_batters.py
-------------------
Fetches recent Statcast data for batters and builds rolling features
for predicting hits, total bases, and home runs.

Same approach as statcast_logs.py for pitchers:
  - Only fetches last 30 days for today's batters (fast, ~30 seconds)
  - Aggregates pitch-by-pitch to per-game stats
  - Builds rolling features: avg H/TB/HR, exit velo, hard hit %, splits

Key features:
  H/game last 7, 15 games
  TB/game last 7, 15 games
  HR/game last 15, 30 games
  Exit velocity trend (hard hit %)
  vs LHP vs RHP splits
  Recent form z-score (hot/cold)
  Ballpark factor
  Opposing pitcher handedness
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_DIR  = Path("props_data")
CACHE_DIR = Path("cache") / "statcast_bat"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum games before making predictions
MIN_GAMES = 7

# Cache for player ID lookups
_batter_id_cache = {}


# ══════════════════════════════════════════════
# PLAYER ID LOOKUP
# ══════════════════════════════════════════════

def _lookup_batter_via_mlb_api(name: str) -> int:
    """Primary batter ID lookup — handles unusual spellings pybaseball misses."""
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people/search",
            params={"names": name, "sportId": 1, "active": True},
            timeout=10,
        )
        r.raise_for_status()
        people = r.json().get("people", [])
        for p in people:
            pos = p.get("primaryPosition", {}).get("code", "")
            if pos != "1":  # prefer non-pitchers
                return int(p["id"])
        if people:
            return int(people[0]["id"])
    except Exception:
        pass
    return 0


def get_batter_id(name: str) -> int:
    """Look up a batter's MLBAM player ID. Tries MLB API first, then pybaseball."""
    if name in _batter_id_cache:
        return _batter_id_cache[name]

    pid = _lookup_batter_via_mlb_api(name)
    if pid > 0:
        _batter_id_cache[name] = pid
        return pid

    try:
        import pybaseball as pyb
        parts = name.strip().split()
        if len(parts) < 2:
            _batter_id_cache[name] = 0
            return 0

        result = pyb.playerid_lookup(parts[-1], parts[0])
        if result.empty:
            result = pyb.playerid_lookup(parts[-1])

        if not result.empty:
            pid = int(result.sort_values("mlb_played_last",
                                          ascending=False).iloc[0]["key_mlbam"])
            _batter_id_cache[name] = pid
            return pid
    except Exception:
        pass

    _batter_id_cache[name] = 0
    return 0


# ══════════════════════════════════════════════
# FETCH RECENT BATTER GAMES
# ══════════════════════════════════════════════

def fetch_recent_batter_games(batter_name: str,
                               days_back: int = 30) -> pd.DataFrame:
    """
    Fetch last N days of Statcast data for one batter.
    Aggregates to per-game stats.

    Per-game stats:
      H, TB, HR, RBI, AB, BB, K
      avg_exit_velo, hard_hit_pct, avg_launch_angle
      xBA, xSLG (expected stats)
    """
    pid = get_batter_id(batter_name)
    if pid == 0:
        return pd.DataFrame()

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    cache_key  = f"bat_{pid}_{end_dt.strftime('%Y%m%d')}"
    cache_file = CACHE_DIR / f"{cache_key}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    try:
        import pybaseball as pyb
        pyb.cache.enable()

        raw = pyb.statcast_batter(
            start_dt  = start_dt.strftime("%Y-%m-%d"),
            end_dt    = end_dt.strftime("%Y-%m-%d"),
            player_id = pid,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        games = aggregate_batter_to_games(raw, batter_name)

        if not games.empty:
            games.to_parquet(cache_file, index=False)

        time.sleep(0.5)
        return games

    except Exception as e:
        return pd.DataFrame()


def aggregate_batter_to_games(df: pd.DataFrame,
                               batter_name: str) -> pd.DataFrame:
    """
    Aggregate pitch-by-pitch Statcast data to one row per game.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    games = []
    for game_date, gdf in df.groupby("game_date"):
        if len(gdf) < 2:
            continue

        # Counting stats from events column
        events = gdf["events"].dropna()
        h      = events.isin(["single","double","triple","home_run"]).sum()
        hr     = (events == "home_run").sum()
        bb     = events.isin(["walk","hit_by_pitch"]).sum()
        k      = events.isin(["strikeout","strikeout_double_play"]).sum()
        ab     = events.isin([
            "single","double","triple","home_run",
            "field_out","strikeout","grounded_into_double_play",
            "force_out","fielders_choice","fielders_choice_out",
            "double_play","triple_play","strikeout_double_play",
        ]).sum()

        # Total bases
        tb = (
            events.isin(["single"]).sum() * 1 +
            events.isin(["double"]).sum() * 2 +
            events.isin(["triple"]).sum() * 3 +
            events.isin(["home_run"]).sum() * 4
        )

        # RBI
        rbi = pd.to_numeric(gdf.get("post_bat_score", pd.Series([0]*len(gdf))),
                             errors="coerce").fillna(0)
        rbi_total = max(0, rbi.diff().clip(lower=0).sum())

        # Exit velocity stats (batted balls only)
        batted = gdf[gdf["launch_speed"].notna()]
        exit_velo    = pd.to_numeric(batted["launch_speed"], errors="coerce").dropna()
        launch_angle = pd.to_numeric(batted["launch_angle"], errors="coerce").dropna()

        avg_exit_velo    = exit_velo.mean()    if len(exit_velo) > 0 else np.nan
        hard_hit_pct     = (exit_velo >= 95).sum() / len(exit_velo) \
                           if len(exit_velo) > 0 else np.nan
        avg_launch_angle = launch_angle.mean() if len(launch_angle) > 0 else np.nan
        barrel_pct       = (
            ((exit_velo >= 98) & (launch_angle.between(26, 30))).sum() /
            len(exit_velo)
        ) if len(exit_velo) > 0 else np.nan

        # xBA, xSLG if available
        xba  = pd.to_numeric(gdf.get("estimated_ba_using_speedangle",
                                      pd.Series([np.nan]*len(gdf))),
                              errors="coerce").mean()
        xslg = pd.to_numeric(gdf.get("estimated_slg_using_speedangle",
                                      pd.Series([np.nan]*len(gdf))),
                              errors="coerce").mean()

        # Pitcher handedness
        p_throws = gdf["p_throws"].mode().iloc[0] \
                   if "p_throws" in gdf.columns and not gdf["p_throws"].empty \
                   else "R"

        games.append({
            "game_date":       game_date,
            "name":            batter_name,
            "H":               int(h),
            "TB":              int(tb),
            "HR":              int(hr),
            "RBI":             int(rbi_total),
            "AB":              int(ab),
            "BB":              int(bb),
            "K":               int(k),
            "avg_exit_velo":   round(avg_exit_velo, 1) if pd.notna(avg_exit_velo) else np.nan,
            "hard_hit_pct":    round(hard_hit_pct, 3)  if pd.notna(hard_hit_pct) else np.nan,
            "avg_launch_angle":round(avg_launch_angle, 1) if pd.notna(avg_launch_angle) else np.nan,
            "barrel_pct":      round(barrel_pct, 3)    if pd.notna(barrel_pct) else np.nan,
            "xba":             round(xba, 3)            if pd.notna(xba) else np.nan,
            "xslg":            round(xslg, 3)           if pd.notna(xslg) else np.nan,
            "p_throws":        p_throws,
            "n_pitches":       len(gdf),
        })

    if not games:
        return pd.DataFrame()

    result = pd.DataFrame(games).sort_values("game_date").reset_index(drop=True)

    # Rolling averages
    for stat in ["H","TB","HR"]:
        result[f"{stat}_avg_7"]  = result[stat].rolling(7,  min_periods=1).mean()
        result[f"{stat}_avg_15"] = result[stat].rolling(15, min_periods=1).mean()

    return result


# ══════════════════════════════════════════════
# ROLLING FEATURE BUILDER
# ══════════════════════════════════════════════

def build_batter_rolling_features(batter_name: str,
                                   game_date,
                                   game_logs: pd.DataFrame,
                                   opp_pitcher_hand: str = "R") -> dict:
    """
    Build rolling features for a batter as of game_date.
    Only uses games BEFORE game_date (no leakage).

    Returns dict of features for H, TB, HR prediction.
    """
    f = {}

    if game_logs.empty:
        return f

    gd = pd.Timestamp(game_date)

    # Get this batter's games before today
    p = game_logs[
        (game_logs["name"] == batter_name) &
        (game_logs["game_date"] < gd)
    ].sort_values("game_date").copy()

    if len(p) < MIN_GAMES:
        return f

    # ── Hits rolling ──
    h = p["H"].astype(float)
    f["bat_h_L7"]      = h.tail(7).mean()
    f["bat_h_L15"]     = h.tail(15).mean()
    f["bat_h_season"]  = h.mean()
    f["bat_h_std_L15"] = h.tail(15).std()

    # ── Total Bases rolling ──
    tb = p["TB"].astype(float)
    f["bat_tb_L7"]     = tb.tail(7).mean()
    f["bat_tb_L15"]    = tb.tail(15).mean()
    f["bat_tb_season"] = tb.mean()

    # ── Home Runs rolling ──
    hr = p["HR"].astype(float)
    f["bat_hr_L15"]    = hr.tail(15).mean()
    f["bat_hr_L30"]    = hr.tail(30).mean()
    f["bat_hr_season"] = hr.mean()

    # ── Exit velocity ──
    ev = p["avg_exit_velo"].astype(float).dropna()
    if len(ev) >= 3:
        f["bat_exit_velo_L7"]   = ev.tail(7).mean()
        f["bat_exit_velo_trend"]= ev.tail(5).mean() - ev.tail(10).head(5).mean() \
                                   if len(ev) >= 10 else 0

    # ── Hard hit % ──
    hh = p["hard_hit_pct"].astype(float).dropna()
    if len(hh) >= 3:
        f["bat_hard_hit_L7"]  = hh.tail(7).mean()
        f["bat_hard_hit_L15"] = hh.tail(15).mean()

    # ── Barrel % ──
    bp = p["barrel_pct"].astype(float).dropna()
    if len(bp) >= 3:
        f["bat_barrel_L10"] = bp.tail(10).mean()

    # ── xBA / xSLG ──
    xba = p["xba"].astype(float).dropna()
    if len(xba) >= 3:
        f["bat_xba_L7"]  = xba.tail(7).mean()

    xslg = p["xslg"].astype(float).dropna()
    if len(xslg) >= 3:
        f["bat_xslg_L7"] = xslg.tail(7).mean()

    # ── Handedness splits ──
    vs_hand = p[p["p_throws"] == opp_pitcher_hand]
    if len(vs_hand) >= 5:
        f["bat_h_vs_hand_L20"]  = vs_hand["H"].tail(20).mean()
        f["bat_tb_vs_hand_L20"] = vs_hand["TB"].tail(20).mean()
        f["bat_hr_vs_hand_L20"] = vs_hand["HR"].tail(20).mean()
    f["bat_opp_hand"] = 1 if opp_pitcher_hand == "L" else 0

    # ── Form z-score (hot/cold streak) ──
    if len(h) >= 10:
        base_mean = h.iloc[:-5].mean()
        base_std  = h.iloc[:-5].std()
        recent    = h.tail(5).mean()
        if base_std > 0.01:
            f["bat_form_z"]   = (recent - base_mean) / base_std
            f["bat_slump"]    = int(f["bat_form_z"] < -2.0)
            f["bat_hot"]      = int(f["bat_form_z"] > 2.0)
        f["bat_h_trend"]      = h.tail(5).mean() - h.tail(10).head(5).mean()

    # ── AB per game (playing time) ──
    ab = p["AB"].astype(float)
    f["bat_ab_L7"] = ab.tail(7).mean()

    # ── Days since last game ──
    f["bat_days_rest"] = (gd - p["game_date"].iloc[-1]).days

    # ── Number of recent games ──
    f["bat_n_games"] = len(p)

    return f


# ══════════════════════════════════════════════
# TOP BATTERS PER TEAM (from MLB Stats API)
# ══════════════════════════════════════════════

def get_top_batters_for_team(team_name: str,
                              game_date: str,
                              n: int = 3) -> list:
    """
    Get the top N batters for a team by fetching today's lineup
    from the MLB Stats API. Falls back to roster if lineup not posted.

    Returns list of batter names.
    """
    import requests

    # Team name → MLB team ID mapping
    TEAM_IDS = {
        "New York Yankees":      147, "Boston Red Sox":       111,
        "Tampa Bay Rays":        139, "Toronto Blue Jays":    141,
        "Baltimore Orioles":     110, "Chicago White Sox":    145,
        "Cleveland Guardians":   114, "Cleveland Indians":    114,
        "Detroit Tigers":        116, "Kansas City Royals":   118,
        "Minnesota Twins":       142, "Houston Astros":       117,
        "Los Angeles Angels":    108, "Oakland Athletics":    133,
        "Seattle Mariners":      136, "Texas Rangers":        140,
        "Atlanta Braves":        144, "Miami Marlins":        146,
        "New York Mets":         121, "Philadelphia Phillies":143,
        "Washington Nationals":  120, "Chicago Cubs":         112,
        "Cincinnati Reds":       113, "Milwaukee Brewers":    158,
        "Pittsburgh Pirates":    134, "St. Louis Cardinals":  138,
        "Arizona Diamondbacks":  109, "Colorado Rockies":     115,
        "Los Angeles Dodgers":   119, "San Diego Padres":     135,
        "San Francisco Giants":  137,
    }

    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return []

    try:
        # Try to get today's lineup first
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        r   = requests.get(url, params={
            "rosterType": "active",
            "date":       game_date,
            "season":     game_date[:4],
        }, timeout=10)
        r.raise_for_status()
        data = r.json()

        batters = []
        for player in data.get("roster", []):
            pos = player.get("position", {}).get("type", "")
            if pos != "Pitcher":
                name = player.get("person", {}).get("fullName", "")
                if name:
                    batters.append(name)

        # Return top N (by roster order which roughly = batting order)
        return batters[:n]

    except Exception as e:
        return []


# ══════════════════════════════════════════════
# FETCH ALL TODAY'S BATTERS
# ══════════════════════════════════════════════

def fetch_todays_batter_games(batter_names: list,
                               days_back: int = 30) -> pd.DataFrame:
    """
    Fetch recent game logs for all batters scheduled today.
    Fast — only pulls data for the specific batters needed.
    """
    all_games = []
    n = len(batter_names)

    print(f"  Fetching recent games for {n} batters...")

    for i, name in enumerate(batter_names):
        if not name or name == "TBD":
            continue
        print(f"  [{i+1}/{n}] {name}...", end=" ")
        games = fetch_recent_batter_games(name, days_back)
        if not games.empty:
            all_games.append(games)
            print(f"{len(games)} games")
        else:
            print("no data")

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    return combined.sort_values(["name","game_date"]).reset_index(drop=True)


# ══════════════════════════════════════════════
# PREDICT H, TB, HR FOR A BATTER
# ══════════════════════════════════════════════

def predict_batter_props(batter_name: str,
                          game_date,
                          game_logs: pd.DataFrame,
                          opp_pitcher_hand: str = "R",
                          home_team: str = "",
                          park_factors: dict = None) -> dict:
    """
    Predict H, TB, HR for a batter tonight.

    Key design choices:
      H  — recent hot streaks matter (L7 weighted heavily), xBA stabilizes luck
      TB — season-weighted to avoid overreacting to variance, barrel% drives extra bases
      HR — season + career barrel% primary; HR is too rare for recent form to dominate
    """
    from props_model import PARK_FACTORS as DEFAULT_PARKS
    parks = park_factors or DEFAULT_PARKS

    feats = build_batter_rolling_features(
        batter_name, game_date, game_logs, opp_pitcher_hand
    )

    if not feats:
        return {}

    park_f = parks.get(home_team, 1.0)

    def safe(v):
        return float(v) if v is not None and not pd.isna(v) else None

    barrel   = safe(feats.get("bat_barrel_L10"))
    hard_hit = safe(feats.get("bat_hard_hit_L7"))

    # ── Hits ──
    # Hot streaks are real for H — weight L7 most heavily.
    # xBA corrects for luck (BABIP variance).
    h_l7  = safe(feats.get("bat_h_L7"))
    h_l15 = safe(feats.get("bat_h_L15"))
    h_szn = safe(feats.get("bat_h_season"))
    xba   = safe(feats.get("bat_xba_L7"))

    if h_l7 is not None and h_l15 is not None and h_szn is not None:
        h_base = h_l7 * 0.55 + h_l15 * 0.25 + h_szn * 0.20
    elif h_l7 is not None and h_szn is not None:
        h_base = h_l7 * 0.70 + h_szn * 0.30
    else:
        h_base = h_szn

    # xBA nudge — pulls prediction toward true talent, dampens streaks
    if xba is not None and h_base is not None:
        xh = xba * 3.8  # ~3.8 AB/game
        h_base = h_base * 0.80 + xh * 0.20

    # Platoon split — batters hit significantly better vs opposite hand
    vs_h = safe(feats.get("bat_h_vs_hand_L20"))
    if vs_h is not None and h_base is not None:
        h_base = vs_h * 0.50 + h_base * 0.50

    # Hard hit % boost — if a batter is squaring up balls, hits follow
    if hard_hit is not None and h_base is not None:
        hh_adj = 1.0 + np.clip((hard_hit - 0.37) * 0.8, -0.08, 0.12)
        h_base = h_base * hh_adj

    pred_h = h_base

    # ── Total Bases ──
    # More season weight than H because TB swings wildly on single HRs.
    # Barrel % and hard hit % are the key drivers of extra bases.
    tb_l7  = safe(feats.get("bat_tb_L7"))
    tb_l15 = safe(feats.get("bat_tb_L15"))
    tb_szn = safe(feats.get("bat_tb_season"))
    xslg   = safe(feats.get("bat_xslg_L7"))

    if tb_l7 is not None and tb_l15 is not None and tb_szn is not None:
        tb_base = tb_l7 * 0.30 + tb_l15 * 0.35 + tb_szn * 0.35
    elif tb_l7 is not None and tb_szn is not None:
        tb_base = tb_l7 * 0.50 + tb_szn * 0.50
    else:
        tb_base = tb_szn

    # xSLG stabilizes TB prediction against HR variance
    if xslg is not None and tb_base is not None:
        xtb = xslg * 3.8
        tb_base = tb_base * 0.75 + xtb * 0.25

    # Barrel % — primary driver of extra bases (league avg ~7%)
    # A batter at 15% barrel rate hits for way more TB per game
    if barrel is not None and tb_base is not None:
        barrel_adj = 1.0 + np.clip((barrel - 0.07) * 4.0, -0.15, 0.35)
        tb_base = tb_base * barrel_adj

    # Hard hit % — extra boost for batters making hard contact recently
    if hard_hit is not None and tb_base is not None:
        hh_adj = 1.0 + np.clip((hard_hit - 0.37) * 1.5, -0.12, 0.20)
        tb_base = tb_base * hh_adj

    # Platoon split for TB
    vs_tb = safe(feats.get("bat_tb_vs_hand_L20"))
    if vs_tb is not None and tb_base is not None:
        tb_base = vs_tb * 0.45 + tb_base * 0.55

    pred_tb = (tb_base * park_f) if tb_base is not None else None

    # ── Home Runs ──
    # HR is the hardest to predict per game — use heaviest season weighting.
    # Barrel % is the single best predictor. Hard hit % is secondary.
    # Recent 15-game rate is very noisy — don't overweight it.
    hr_l15 = safe(feats.get("bat_hr_L15"))
    hr_l30 = safe(feats.get("bat_hr_L30"))
    hr_szn = safe(feats.get("bat_hr_season"))

    if hr_l15 is not None and hr_l30 is not None and hr_szn is not None:
        hr_base = hr_l15 * 0.20 + hr_l30 * 0.35 + hr_szn * 0.45
    elif hr_l30 is not None and hr_szn is not None:
        hr_base = hr_l30 * 0.45 + hr_szn * 0.55
    elif hr_szn is not None:
        hr_base = hr_szn
    else:
        hr_base = None

    # Barrel % is king for HR — much heavier adjustment than TB
    if barrel is not None and hr_base is not None:
        barrel_adj = 1.0 + np.clip((barrel - 0.07) * 6.0, -0.25, 0.50)
        hr_base = hr_base * barrel_adj

    # Hard hit % secondary boost
    if hard_hit is not None and hr_base is not None:
        hh_adj = 1.0 + np.clip((hard_hit - 0.37) * 2.0, -0.15, 0.30)
        hr_base = hr_base * hh_adj

    # Platoon split for HR
    vs_hr = safe(feats.get("bat_hr_vs_hand_L20"))
    if vs_hr is not None and hr_base is not None:
        hr_base = vs_hr * 0.40 + hr_base * 0.60

    pred_hr = (hr_base * park_f) if hr_base is not None else None

    return {
        "pred_h":     round(pred_h, 2)  if pred_h  is not None else None,
        "pred_tb":    round(pred_tb, 2) if pred_tb is not None else None,
        "pred_hr":    round(pred_hr, 3) if pred_hr is not None else None,
        "form_z":     feats.get("bat_form_z", 0),
        "slump":      feats.get("bat_slump", 0),
        "hot":        feats.get("bat_hot", 0),
        "exit_velo":  feats.get("bat_exit_velo_L7"),
        "hard_hit":   feats.get("bat_hard_hit_L7"),
        "barrel":     barrel,
        "n_games":    feats.get("bat_n_games", 0),
        "feats":      feats,
    }


if __name__ == "__main__":
    print("Testing statcast_batters.py...\n")

    # Test with Aaron Judge
    print("Fetching Aaron Judge recent games...")
    games = fetch_recent_batter_games("Aaron Judge", days_back=30)

    if not games.empty:
        print(f"Found {len(games)} games")
        print(games[["game_date","H","TB","HR","avg_exit_velo","hard_hit_pct"]].to_string())

        preds = predict_batter_props(
            "Aaron Judge",
            datetime.now(),
            games,
            opp_pitcher_hand="L",
            home_team="New York Yankees",
        )
        print(f"\nPredictions for Aaron Judge tonight:")
        print(f"  Hits:        {preds.get('pred_h')}")
        print(f"  Total Bases: {preds.get('pred_tb')}")
        print(f"  Home Runs:   {preds.get('pred_hr')}")
        print(f"  Form:        {'🔥 Hot' if preds.get('hot') else '🥶 Cold' if preds.get('slump') else 'Normal'}")
    else:
        print("No data found")
