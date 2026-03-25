"""
props_model.py
--------------
Player proposition bet prediction model using regression approach.

Instead of predicting over/under on a book's line, we predict the
ACTUAL stat value. When you have a live line to bet against, compare
your predicted number to the book's line and bet the difference.

Example:
  Model predicts: Jacob deGrom gets 8.2 K's
  Book line:      6.5 K's
  Action:         Strong OVER bet (1.7K edge)

Advantages over line-based approach:
  - Train on all historical data back to 2010 (no API needed)
  - Independent of bookmaker pricing
  - Spots larger edges
  - Works even on new prop markets

Supported props:
  - Pitcher strikeouts (K props)
  - Batter hits
  - Batter total bases
  - Batter home runs

Usage:
  python props_model.py --train --seasons 2014 2015 2016 2017 2018 2019 2020 2021
  python props_model.py --walkforward
  python props_model.py --predict --date 2021-09-01
  python props_model.py --compare
"""

import warnings, time, argparse
from statcast_logs import (
    fetch_all_pitcher_logs, build_pitcher_rolling_features,
    load_pitcher_id_map, STATCAST_FEATURE_COLS,
)
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_DIR  = Path("cache") / "props"
MODELS_DIR = Path("saved_models") / "props"
DATA_DIR   = Path("props_data")
for d in [CACHE_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_statcast_pitcher_stats(seasons: list) -> pd.DataFrame:
    """
    Load Statcast pitcher stats from pybaseball (FanGraphs Statcast).
    Key columns: Name, season, SwStr%, Whiff%, CSW%, FB%, SL%, CH%,
                 vFA, Stuff+, Location+, Pitching+, xFIP, SIERA
    These are the best predictors of strikeout rate.
    """
    cache = DATA_DIR / f"statcast_pit_{'_'.join(map(str,seasons))}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        return pd.DataFrame()

    frames = []
    for s in seasons:
        print(f"  ⬇️  Statcast pitcher stats {s}...", end=" ")
        try:
            # pitching_stats with Statcast columns
            df = pyb.pitching_stats(s, qual=20)
            df["season"] = s
            frames.append(df)
            print(f"{len(df)} pitchers")
            time.sleep(1)
        except Exception as e:
            print(f"failed: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    # Standardize name
    for old in ["playerName","player_name"]:
        if old in result.columns and "Name" not in result.columns:
            result = result.rename(columns={old: "Name"})
    result.to_csv(cache, index=False)
    return result


# ── Park run environment factors ──
PARK_FACTORS = {
    "Colorado Rockies":      1.38,
    "Boston Red Sox":        1.10,
    "Chicago Cubs":          1.08,
    "Cincinnati Reds":       1.06,
    "Texas Rangers":         1.05,
    "Philadelphia Phillies": 1.04,
    "Baltimore Orioles":     1.03,
    "Atlanta Braves":        1.02,
    "New York Yankees":      1.01,
    "Detroit Tigers":        1.00,
    "Cleveland Indians":     0.99,
    "Cleveland Guardians":   0.99,
    "Minnesota Twins":       0.98,
    "Tampa Bay Rays":        0.97,
    "Kansas City Royals":    0.97,
    "Oakland Athletics":     0.96,
    "Los Angeles Dodgers":   0.96,
    "Arizona Diamondbacks":  0.96,
    "Pittsburgh Pirates":    0.95,
    "Seattle Mariners":      0.95,
    "San Diego Padres":      0.94,
    "Toronto Blue Jays":     0.94,
    "Miami Marlins":         0.93,
    "New York Mets":         0.93,
    "Houston Astros":        0.92,
    "Chicago White Sox":     0.98,
    "Milwaukee Brewers":     0.97,
    "St. Louis Cardinals":   0.96,
    "Washington Nationals":  0.97,
    "Los Angeles Angels":    1.00,
    "San Francisco Giants":  0.92,
}

PROP_CONFIGS = {
    "pitcher_strikeouts": {
        "label":      "Pitcher Strikeouts",
        "stat_col":   "SO",
        "log_type":   "pitching",
        "min_starts": 3,
        "typical_line_range": (4.0, 9.0),   # K/start range
        "bet_edge_threshold": 0.5,
    },
    "batter_hits": {
        "label":      "Batter Hits",
        "stat_col":   "H",
        "log_type":   "batting",
        "min_starts": 3,
        "typical_line_range": (0.5, 1.5),   # H/game range
        "bet_edge_threshold": 0.1,
    },
    "batter_total_bases": {
        "label":      "Batter Total Bases",
        "stat_col":   "TB",
        "log_type":   "batting",
        "min_starts": 3,
        "typical_line_range": (0.75, 2.5),  # TB/game range
        "bet_edge_threshold": 0.15,
    },
    "batter_home_runs": {
        "label":      "Batter Home Runs",
        "stat_col":   "HR",
        "log_type":   "batting",
        "min_starts": 3,
        "typical_line_range": (0.05, 0.45), # HR/game range
        "bet_edge_threshold": 0.05,
    },
}


# ══════════════════════════════════════════════
# PYBASEBALL DATA LOADING
# ══════════════════════════════════════════════

def load_pitching_logs(seasons: list) -> pd.DataFrame:
    """Load pitcher season stats using pybaseball.pitching_stats."""
    cache = DATA_DIR / f"pit_logs_{'_'.join(map(str,seasons))}.csv"
    if cache.exists():
        print("  📂 Pitching logs from cache")
        df = pd.read_csv(cache)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        raise ImportError("Run: pip install pybaseball")

    frames = []
    for s in seasons:
        print(f"  ⬇️  Pitching stats {s}...", end=" ")
        try:
            df = pyb.pitching_stats(s, qual=20)
            df["season"] = s
            df["Date"]   = pd.Timestamp(f"{s}-11-01")
            frames.append(df)
            print(f"{len(df)} pitchers")
            time.sleep(1)
        except Exception as e:
            print(f"failed: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    # Standardize name column
    for old in ["playerName", "player_name"]:
        if old in result.columns and "Name" not in result.columns:
            result = result.rename(columns={old: "Name"})
    result.to_csv(cache, index=False)
    print(f"  ✅ Pitching stats cached")
    return result


def load_batting_logs(seasons: list) -> pd.DataFrame:
    """Load batter season stats using pybaseball.batting_stats."""
    cache = DATA_DIR / f"bat_logs_{'_'.join(map(str,seasons))}.csv"
    if cache.exists():
        print("  📂 Batting logs from cache")
        df = pd.read_csv(cache)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        raise ImportError("Run: pip install pybaseball")

    frames = []
    for s in seasons:
        print(f"  ⬇️  Batting stats {s}...", end=" ")
        try:
            df = pyb.batting_stats(s, qual=50)
            df["season"] = s
            df["Date"]   = pd.Timestamp(f"{s}-11-01")
            frames.append(df)
            print(f"{len(df)} batters")
            time.sleep(1)
        except Exception as e:
            print(f"failed: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    # Standardize name column
    for old in ["playerName", "player_name"]:
        if old in result.columns and "Name" not in result.columns:
            result = result.rename(columns={old: "Name"})
    result.to_csv(cache, index=False)
    print(f"  ✅ Batting stats cached")
    return result


def load_team_batting_stats(seasons: list) -> pd.DataFrame:
    """Load season-level team batting stats for opponent context."""
    cache = DATA_DIR / f"team_bat_{'_'.join(map(str,seasons))}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        return pd.DataFrame()

    frames = []
    for s in seasons:
        try:
            df = pyb.team_batting(s)
            df["season"] = s
            frames.append(df)
            time.sleep(1)
        except:
            pass

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache, index=False)
    return result


def load_team_pitching_stats(seasons: list) -> pd.DataFrame:
    """Load season-level team pitching stats."""
    cache = DATA_DIR / f"team_pit_{'_'.join(map(str,seasons))}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    try:
        import pybaseball as pyb
        pyb.cache.enable()
    except ImportError:
        return pd.DataFrame()

    frames = []
    for s in seasons:
        try:
            df = pyb.team_pitching(s)
            df["season"] = s
            frames.append(df)
            time.sleep(1)
        except:
            pass

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(cache, index=False)
    return result


# ══════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════

def _safe(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def build_pitcher_k_features(pitcher: str,
                               game_date,
                               pitching_logs: pd.DataFrame,
                               opp_team_abbrev: str,
                               team_bat_stats: pd.DataFrame,
                               home_team: str,
                               start_logs: pd.DataFrame = None) -> dict:
    """
    Build features for pitcher strikeout prediction.

    Key factors:
      Own stats:  K/9 rolling, BB/9, HR/9, IP trend, form z-score
      Matchup:    Opponent team K%, BB%, wRC+ (lineup quality)
      Context:    Park factor, month, rest days, home/away
    """
    f = {}
    gd     = pd.Timestamp(game_date)
    season = gd.year

    # ── Pitcher's own rolling stats ──
    p = pitching_logs[
        (pitching_logs["Name"] == pitcher) &
        (pitching_logs["Date"] < gd)
    ].sort_values("Date").copy()

    if len(p) >= 3:
        so = _safe(p["SO"])
        ip = _safe(p["IP"]).clip(lower=0.1)
        bb = _safe(p["BB"])
        hr = _safe(p["HR"])
        er = _safe(p["ER"])

        k9 = so / ip * 9
        f["p_k9_L3"]         = k9.tail(3).mean()
        f["p_k9_L5"]         = k9.tail(5).mean()
        f["p_k9_L10"]        = k9.tail(10).mean()
        f["p_k9_season"]     = k9.mean()
        f["p_k9_std_L10"]    = k9.tail(10).std()       # consistency
        f["p_bb9_L5"]        = (_safe(p["BB"]) / ip * 9).tail(5).mean()
        f["p_hr9_L5"]        = (_safe(p["HR"]) / ip * 9).tail(5).mean()
        f["p_era_L5"]        = (er / ip * 9).tail(5).mean()
        f["p_ip_L3"]         = ip.tail(3).mean()       # workload trend
        f["p_ip_L3_vs_L10"]  = ip.tail(3).mean() - ip.tail(10).mean()
        f["p_starts"]        = len(p)

        # Form: is pitcher hotter or cooler than season average?
        if len(k9) >= 8:
            base_mean = k9.iloc[:-3].mean()
            base_std  = k9.iloc[:-3].std()
            recent    = k9.tail(3).mean()
            if base_std > 0.01:
                f["p_form_z"]  = (recent - base_mean) / base_std
                f["p_slump"]   = int(f["p_form_z"] < -2.0)
                f["p_hot"]     = int(f["p_form_z"] > 2.0)
            f["p_k_trend"]     = k9.tail(3).mean() - k9.tail(6).head(3).mean()

        # Days rest since last start
        f["p_days_rest"] = (gd - p["Date"].iloc[-1]).days

    # ── Opponent team's K vulnerability ──
    opp_last = opp_team_abbrev.split()[-1] if opp_team_abbrev.strip() else ""
    opp = team_bat_stats[
        (team_bat_stats["Team"].str.contains(
            opp_last, na=False, regex=False) if opp_last else pd.Series([False]*len(team_bat_stats)))
        & (team_bat_stats["season"] == season)
    ]
    if not opp.empty:
        row = opp.iloc[0]
        kpct = str(row.get("K%","22%")).replace("%","")
        bbpct= str(row.get("BB%","8%")).replace("%","")
        f["opp_kpct"]    = float(kpct)/100 if float(kpct) > 1 else float(kpct)
        f["opp_bbpct"]   = float(bbpct)/100 if float(bbpct) > 1 else float(bbpct)
        f["opp_wrc"]     = float(row.get("wRC+", 100) or 100)
        f["opp_obp"]     = float(row.get("OBP", 0.320) or 0.320)
        f["opp_iso"]     = float(row.get("ISO", 0.150) or 0.150)

    # ── Statcast / stuff metrics (if available in pitching_logs) ──
    # pybaseball pitching_stats includes these FanGraphs Statcast columns
    statcast_cols = {
        "SwStr%": "p_swstr",      # swinging strike % — best K predictor
        "Whiff%": "p_whiff",      # whiff rate on swings
        "CSW%":   "p_csw",        # called strike + whiff %
        "FB%":    "p_fb_pct",     # fastball usage
        "SL%":    "p_sl_pct",     # slider usage
        "CH%":    "p_ch_pct",     # changeup usage
        "vFA":    "p_velo",       # fastball velocity
        "Stuff+": "p_stuff_plus", # stuff+ (100=avg, higher=better)
        "xFIP":   "p_xfip",      # expected FIP
        "SIERA":  "p_siera",     # strikeout-based ERA estimator
        "K%":     "p_kpct",      # strikeout rate (target predictor)
        "BB%":    "p_bbpct",     # walk rate
    }
    if len(p) >= 1:
        latest = p.iloc[-1]  # most recent season row
        for src_col, feat_name in statcast_cols.items():
            if src_col in latest.index:
                val = latest[src_col]
                # Handle percentage strings like "22.1%"
                if isinstance(val, str):
                    val = val.replace("%","")
                try:
                    fval = float(val)
                    # Normalize percentages > 1 to 0-1 range
                    if "%" in src_col and fval > 1:
                        fval = fval / 100
                    f[feat_name] = fval
                except:
                    pass

        # Pitch mix diversity (entropy) — more varied = harder to predict = more Ks
        pitches = [f.get(f"p_{p}_pct", 0) or 0
                   for p in ["fb","sl","ch"]]
        pitches = [x for x in pitches if x > 0]
        if pitches:
            total = sum(pitches)
            if total > 0:
                probs = [x/total for x in pitches]
                entropy = -sum(p*np.log(p+1e-10) for p in probs)
                f["p_pitch_entropy"] = entropy

    # ── Statcast per-start rolling features ──
    if start_logs is not None and not start_logs.empty:
        sc_feats = build_pitcher_rolling_features(pitcher, gd, start_logs)
        f.update(sc_feats)

    # ── Game context ──
    f["ctx_park"]        = PARK_FACTORS.get(home_team, 1.0)
    f["ctx_month"]       = gd.month
    f["ctx_is_summer"]   = int(gd.month in [6, 7, 8])
    f["ctx_is_home"]     = int(pitcher in get_home_pitchers(home_team, gd,
                                                              pitching_logs))
    return f


def build_batter_features(batter: str,
                           game_date,
                           batting_logs: pd.DataFrame,
                           pitching_logs: pd.DataFrame,
                           opp_pitcher: str,
                           opp_pitcher_hand: str,
                           team_pit_stats: pd.DataFrame,
                           home_team: str,
                           prop_type: str) -> dict:
    """
    Build features for batter prop prediction.

    Key factors:
      Own stats:    rolling stat rate, form, slump/hot, vs LHP/RHP splits
      Pitcher:      hits allowed rate, HR allowed, K rate, ERA, pitch tendencies
      Matchup edge: batter ISO vs pitcher HR/9 (power matchup)
      Context:      park factor, month, rest, batting order position
    """
    f  = {}
    gd = pd.Timestamp(game_date)
    sc = PROP_CONFIGS[prop_type]["stat_col"]
    season = gd.year

    # ── Batter's own rolling stats ──
    b = batting_logs[
        (batting_logs["Name"] == batter) &
        (batting_logs["Date"] < gd)
    ].sort_values("Date").copy()

    if len(b) >= 7:
        stat = _safe(b.get(sc, pd.Series([0]*len(b))))
        ab   = _safe(b.get("AB", pd.Series([3]*len(b))), 3).clip(lower=0.1)
        h    = _safe(b.get("H",  pd.Series([0]*len(b))))
        hr   = _safe(b.get("HR", pd.Series([0]*len(b))))
        tb   = _safe(b.get("TB", pd.Series([0]*len(b))))

        f["bat_L7"]        = stat.tail(7).mean()
        f["bat_L15"]       = stat.tail(15).mean()
        f["bat_L30"]       = stat.tail(30).mean()
        f["bat_season"]    = stat.mean()
        f["bat_std_L15"]   = stat.tail(15).std()       # consistency
        f["bat_ab_L7"]     = ab.tail(7).mean()

        # Power metrics
        f["bat_iso_L15"]   = ((tb - h) / ab).tail(15).mean()
        f["bat_avg_L15"]   = (h / ab).tail(15).mean()
        f["bat_hr_L20"]    = hr.tail(20).mean()
        f["bat_tb_L15"]    = tb.tail(15).mean()

        # Recent form z-score
        if len(stat) >= 10:
            base = stat.iloc[:-7]
            rec  = stat.tail(7)
            if base.std() > 0.001:
                z = (rec.mean() - base.mean()) / base.std()
                f["bat_form_z"] = z
                f["bat_slump"]  = int(z < -2.0)
                f["bat_hot"]    = int(z > 2.0)

        # Handedness split — CRITICAL for hit/TB props
        # How does this batter perform specifically vs LHP vs RHP?
        if "opp_hand" in b.columns:
            mask = b["opp_hand"] == opp_pitcher_hand
            vs   = b[mask]
            if len(vs) >= 5:
                vs_stat = _safe(vs.get(sc, pd.Series([0]*len(vs))))
                f[f"bat_vs_{opp_pitcher_hand.lower()}hp_L20"] = vs_stat.tail(20).mean()
                f[f"bat_vs_{opp_pitcher_hand.lower()}hp_avg"] = vs_stat.mean()

    # ── Opposing pitcher's tendencies ──
    # Does this pitcher give up lots of hits/HR?
    p = pitching_logs[
        (pitching_logs["Name"] == opp_pitcher) &
        (pitching_logs["Date"] < gd)
    ].sort_values("Date").tail(10)

    if len(p) >= 3:
        ip   = _safe(p["IP"]).clip(lower=0.1)
        opp_h  = _safe(p.get("H",  pd.Series([0]*len(p))))
        opp_hr = _safe(p.get("HR", pd.Series([0]*len(p))))
        opp_bb = _safe(p.get("BB", pd.Series([0]*len(p))))
        opp_so = _safe(p.get("SO", pd.Series([0]*len(p))))
        opp_er = _safe(p.get("ER", pd.Series([0]*len(p))))

        f["opp_p_h9"]      = (opp_h  / ip * 9).mean()  # hits per 9 — higher = more hits for batter
        f["opp_p_hr9"]     = (opp_hr / ip * 9).mean()  # HR per 9
        f["opp_p_bb9"]     = (opp_bb / ip * 9).mean()  # walks
        f["opp_p_k9"]      = (opp_so / ip * 9).mean()  # K rate — high K = fewer hits
        f["opp_p_era"]     = (opp_er / ip * 9).mean()
        f["opp_p_hand"]    = 1 if opp_pitcher_hand == "L" else 0
        f["opp_p_starts"]  = len(p)

        # Key matchup edge: batter's power vs pitcher's HR allowed rate
        if "bat_iso_L15" in f:
            f["power_matchup"] = f["bat_iso_L15"] * f["opp_p_hr9"]

    # ── Opponent team pitching quality ──
    opp_team_pit = team_pit_stats[
        (team_pit_stats["season"] == season)
    ]
    # Just use league average if we can't match team exactly
    if not opp_team_pit.empty:
        f["opp_team_era"]  = float(opp_team_pit["ERA"].mean() or 4.0)
        f["opp_team_whip"] = float(opp_team_pit["WHIP"].mean() or 1.3)

    # ── Game context ──
    f["ctx_park"]        = PARK_FACTORS.get(home_team, 1.0)
    f["ctx_month"]       = gd.month
    f["ctx_is_summer"]   = int(gd.month in [6, 7, 8])
    f["ctx_is_home_bat"] = 1  # set per game in batch building

    return f


def get_home_pitchers(home_team: str, game_date,
                       pitching_logs: pd.DataFrame) -> list:
    """Best-effort lookup of pitchers associated with home team."""
    # This is a simplification — in production use MLB API roster
    return []


# ══════════════════════════════════════════════
# TRAINING DATASET BUILDER
# ══════════════════════════════════════════════

def build_training_dataset(prop_type: str,
                            batting_logs: pd.DataFrame,
                            pitching_logs: pd.DataFrame,
                            team_bat_stats: pd.DataFrame,
                            team_pit_stats: pd.DataFrame,
                            start_logs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a regression training dataset for one prop type.
    Each row = one player game appearance with:
      - Features built from all prior games (no leakage)
      - Target = actual stat value that game
    """
    cfg    = PROP_CONFIGS[prop_type]
    scol   = cfg["stat_col"]
    is_pit = cfg["log_type"] == "pitching"
    source = pitching_logs if is_pit else batting_logs

    if source.empty:
        print(f"  ⚠️  No {cfg['log_type']} logs available")
        return pd.DataFrame()

    # Season-level: one row per player per season
    # Target = per-game rate (total stat / games started)
    source = source.sort_values(["season", "Name"]).copy()
    all_seasons = sorted(source["season"].dropna().unique())
    records     = []

    print(f"  Building {cfg['label']} dataset across {len(all_seasons)} seasons...")

    for season in all_seasons:
        season_df = source[source["season"] == season]
        prior_df  = source[source["season"] < season]

        print(f"    Season {int(season)}: {len(season_df)} players", end="\r")

        for _, row in season_df.iterrows():
            player = str(row.get("Name", ""))
            if not player or player == "nan":
                continue

            # Per-game rate as target
            gs    = pd.to_numeric(row.get("GS", row.get("G", 1)), errors="coerce") or 1
            total = pd.to_numeric(row.get(scol, np.nan), errors="coerce")
            if pd.isna(total) or gs < 1:
                continue

            # For pitchers: if GS looks wrong (=1 when SO is high),
            # estimate GS from IP (avg ~5.5 IP per start)
            if is_pit and gs <= 2 and total > 10:
                ip = pd.to_numeric(row.get("IP", 0), errors="coerce") or 0
                if ip > 10:
                    gs = max(round(ip / 5.5), 1)

            # Enforce minimum realistic GS for season stats
            # Pitchers: min 5 starts, batters: min 20 games
            min_gs = 5 if is_pit else 20
            if gs < min_gs:
                continue

            actual = total / gs  # per-start or per-game rate

            gdate = pd.Timestamp(f"{int(season)}-07-01")
            team  = str(row.get("Team", ""))

            # Build features from prior seasons only (no leakage)
            prior_pit = prior_df if is_pit else (
                pitching_logs[pitching_logs["season"] < season]
                if not pitching_logs.empty else pd.DataFrame()
            )
            prior_bat = prior_df if not is_pit else (
                batting_logs[batting_logs["season"] < season]
                if not batting_logs.empty else pd.DataFrame()
            )

            if is_pit:
                sc_prior = start_logs[start_logs["season"] < season] \
                    if start_logs is not None and not start_logs.empty else None
                feats = build_pitcher_k_features(
                    player, gdate,
                    prior_pit if not prior_pit.empty else pitching_logs,
                    team, team_bat_stats, team,
                    start_logs=sc_prior,
                )
            else:
                feats = build_batter_features(
                    player, gdate,
                    prior_bat if not prior_bat.empty else batting_logs,
                    prior_pit if not prior_pit.empty else pitching_logs,
                    "", "R", team_pit_stats, team, prop_type
                )

            if not feats:
                continue

            rec = {
                "date":   gdate,
                "player": player,
                "season": int(season),
                "team":   team,
                "actual": round(actual, 4),
                "total":  total,
                "gs":     gs,
            }
            rec.update(feats)
            records.append(rec)

    print()
    if not records:
        print(f"  ⚠️  No records built for {prop_type}")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  ✅ {prop_type}: {len(df)} samples  "
          f"avg {scol}: {df['actual'].mean():.2f}  "
          f"std: {df['actual'].std():.2f}")

    cache_path = DATA_DIR / f"training_{prop_type}.csv"
    df.to_csv(cache_path, index=False)
    return df


# ══════════════════════════════════════════════
# REGRESSION MODEL TRAINING
# ══════════════════════════════════════════════

def get_feat_cols(df: pd.DataFrame) -> list:
    """
    Get feature columns — strictly exclude anything that leaks the answer.
    'total' = season stat total (IS the answer scaled by GS)
    'gs'    = games started (divides total to get actual — direct leakage)
    Only use prior-season rolling stats and context features.
    """
    LEAK_COLS = {
        "date","player","season","team","actual","total","gs",
        "G","GS","PA","AB",
    }
    SAFE_PREFIXES = ("p_","bat_","opp_","ctx_","power_","sc_")
    return [
        c for c in df.columns
        if c not in LEAK_COLS
        and any(c.startswith(pfx) for pfx in SAFE_PREFIXES)
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]


def load_statcast_logs(seasons: list,
                       pitching_logs: pd.DataFrame) -> pd.DataFrame:
    """Load or download per-start Statcast logs for all pitchers."""
    cache_file = DATA_DIR / f"all_starts_{'_'.join(map(str,seasons))}.parquet"
    if cache_file.exists():
        print("  📂 Per-start Statcast logs from cache")
        df = pd.read_parquet(cache_file)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df
    print("  Building pitcher ID map...")
    id_map = load_pitcher_id_map(pitching_logs)
    pitchers = pitching_logs["Name"].dropna().unique().tolist()
    return fetch_all_pitcher_logs(pitchers, id_map, seasons)


def train_prop_model(train_df: pd.DataFrame,
                     prop_type: str) -> dict:
    """
    Train a regression model to predict the actual stat value.
    Uses gradient boosting regressor + random forest + linear regression.
    """
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, r2_score
    import copy

    cfg       = PROP_CONFIGS[prop_type]
    feat_cols = get_feat_cols(train_df)

    if not feat_cols or len(train_df) < 50:
        print(f"  ⚠️  Not enough data: {len(train_df)} samples")
        return {}

    X = train_df[feat_cols].fillna(0).values
    y = train_df["actual"].values

    tscv  = TimeSeriesSplit(n_splits=4)
    label = cfg["label"]

    print(f"\n{'='*62}")
    print(f"  {label}  (regression — predicting actual value)")
    print(f"  Samples: {len(train_df)}   Features: {len(feat_cols)}")
    print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")
    print(f"{'─'*62}")
    print(f"  {'Model':<22} {'MAE':>8} {'R²':>8}  Folds MAE")
    print(f"{'─'*62}")

    model_defs = [
        ("XGBoost", XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=10, random_state=42, verbosity=0
        )),
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        )),
        ("Ridge Regression", Pipeline([
            ("s", StandardScaler()),
            ("m", Ridge(alpha=1.0))
        ])),
    ]

    trained = {}
    for name, m in model_defs:
        fold_maes = []
        for tr, te in tscv.split(X):
            mc = copy.deepcopy(m)
            mc.fit(X[tr], y[tr])
            preds = mc.predict(X[te])
            fold_maes.append(mean_absolute_error(y[te], preds))

        mean_mae = np.mean(fold_maes)
        # Train final on all data
        m.fit(X, y)
        trained[name] = m
        full_r2 = r2_score(y, m.predict(X))
        fold_str = "  ".join([f"{v:.2f}" for v in fold_maes])
        print(f"  {name:<22} {mean_mae:>8.3f} {full_r2:>8.3f}  [{fold_str}]")

    print(f"{'='*62}")

    # Feature importance from XGBoost
    xgb_model = trained["XGBoost"]
    if hasattr(xgb_model, "feature_importances_"):
        importance = pd.Series(
            xgb_model.feature_importances_, index=feat_cols
        ).sort_values(ascending=False)
        print(f"\n  Top 10 features for {label}:")
        for feat, imp in importance.head(10).items():
            bar = "█" * int(imp * 200)
            print(f"    {feat:<35} {imp:.4f}  {bar}")

    model_pkg = {
        "models":    trained,
        "feat_cols": feat_cols,
        "prop_type": prop_type,
        "target_mean": float(y.mean()),
        "target_std":  float(y.std()),
    }
    joblib.dump(model_pkg, MODELS_DIR / f"{prop_type}.pkl")
    print(f"\n  ✅ Model saved → {prop_type}.pkl")
    return model_pkg


def predict_stat(model_pkg: dict, feats: dict) -> float:
    """Predict the expected stat value for a player-game."""
    fc = model_pkg["feat_cols"]
    x  = np.array([[feats.get(c, 0) for c in fc]])
    preds = [m.predict(x)[0] for m in model_pkg["models"].values()]
    return float(np.mean(preds))


# ══════════════════════════════════════════════
# BETTING HELPERS
# ══════════════════════════════════════════════

def ml_to_dec(ml):
    if ml == 0: return 2.0
    return (ml/100)+1 if ml > 0 else (100/abs(ml))+1

def bk_prob(ml):
    ml = float(ml)
    if ml == 0: return 0.5
    return 100/(ml+100) if ml > 0 else abs(ml)/(abs(ml)+100)

def ev(prob, ml):
    return prob*(ml_to_dec(ml)-1) - (1-prob)

def kelly(prob, ml, k=0.25, cap=0.025):
    b = ml_to_dec(ml)-1
    f = (b*prob-(1-prob))/b
    return min(max(0.0, f*k), cap)

def edge_to_prob(predicted: float, line: float,
                  std: float, direction: str) -> float:
    """
    Convert predicted stat value to over/under probability.
    Uses normal distribution assumption around predicted value.
    """
    from scipy import stats
    if std <= 0: std = 1.0
    if direction == "over":
        return 1 - stats.norm.cdf(line, loc=predicted, scale=std)
    else:
        return stats.norm.cdf(line, loc=predicted, scale=std)


# ══════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════

def run_props_walkforward(prop_type: str,
                           batting_logs: pd.DataFrame,
                           pitching_logs: pd.DataFrame,
                           team_bat_stats: pd.DataFrame,
                           team_pit_stats: pd.DataFrame,
                           min_edge: float  = 0.06,
                           bankroll: float  = 1000.0) -> dict:
    """
    Season-level walk-forward validation for props regression model.

    Validates prediction ACCURACY across seasons — how well does the
    model predict per-game stat rates on unseen seasons?

    Betting simulation is excluded until real historical prop lines
    are available. Instead we report:
      - MAE per season (how far off our predictions are)
      - Direction accuracy (did we predict the right direction vs prior year?)
      - Estimated edge if book lines were near season average

    This is the honest version — no fake line generation.
    """
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score
    import copy

    cfg        = PROP_CONFIGS[prop_type]
    min_starts = cfg["min_starts"]

    cache_path = DATA_DIR / f"training_{prop_type}.csv"
    if cache_path.exists():
        print(f"  📂 {cfg['label']} training data from cache")
        full_df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        full_df = build_training_dataset(
            prop_type, batting_logs, pitching_logs,
            team_bat_stats, team_pit_stats
        )

    if full_df.empty:
        return {}

    feat_cols = get_feat_cols(full_df)
    if not feat_cols:
        print(f"  ⚠️  No valid features after leak removal.")
        return {}

    print(f"  Leak-free features: {len(feat_cols)}")

    full_df = full_df.sort_values("season").reset_index(drop=True)
    full_df[feat_cols] = full_df[feat_cols].fillna(0)

    all_seasons  = sorted(full_df["season"].unique())
    pred_seasons = all_seasons[min_starts:]

    model_defs = [
        ("xgb",   XGBRegressor(n_estimators=150, max_depth=4,
                                learning_rate=0.05, subsample=0.8,
                                min_child_weight=10, random_state=42,
                                verbosity=0)),
        ("rf",    RandomForestRegressor(n_estimators=100, max_depth=5,
                                         min_samples_leaf=15, random_state=42,
                                         n_jobs=-1)),
        ("ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
    ]

    print(f"\n  Season walk-forward: {cfg['label']}")
    print(f"  Training on: {all_seasons[:min_starts]}  Predicting: {pred_seasons}\n")
    print(f"  {'Season':<8} {'Players':>8} {'MAE':>8} {'R²':>8} {'Dir Acc':>9} {'Baseline MAE':>13}")
    print(f"  {'-'*60}")

    season_results = []
    all_preds, all_actuals = [], []

    for pred_season in pred_seasons:
        tr_df = full_df[full_df["season"] < pred_season]
        te_df = full_df[full_df["season"] == pred_season]

        if len(tr_df) < 10 or te_df.empty:
            continue

        X_tr = tr_df[feat_cols].values
        X_te = te_df[feat_cols].values
        y_tr = tr_df["actual"].values
        y_te = te_df["actual"].values

        try:
            models = {}
            for name, m in model_defs:
                mc = copy.deepcopy(m)
                mc.fit(X_tr, y_tr)
                models[name] = mc

            preds = np.mean([m.predict(X_te) for m in models.values()], axis=0)
        except Exception as e:
            print(f"  {pred_season}: failed — {e}")
            continue

        mae      = mean_absolute_error(y_te, preds)
        r2       = r2_score(y_te, preds)
        baseline = mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean()))

        # Direction accuracy: did model predict above/below prior season avg correctly?
        prior_mean = y_tr.mean()
        dir_correct = np.mean(
            (preds > prior_mean) == (y_te > prior_mean)
        )

        print(f"  {pred_season:<8} {len(te_df):>8} {mae:>8.3f} {r2:>8.3f} "
              f"{dir_correct:>8.1%} {baseline:>13.3f}")

        all_preds.extend(preds.tolist())
        all_actuals.extend(y_te.tolist())
        season_results.append({
            "season": pred_season,
            "n": len(te_df),
            "mae": mae,
            "r2": r2,
            "baseline_mae": baseline,
            "dir_acc": dir_correct,
        })

    print(f"  {'-'*60}")

    if all_preds:
        overall_mae = mean_absolute_error(all_actuals, all_preds)
        overall_r2  = r2_score(all_actuals, all_preds)
        overall_dir = np.mean([r["dir_acc"] for r in season_results])
        baseline_all= mean_absolute_error(all_actuals,
                                           np.full_like(all_actuals,
                                                        np.mean(all_actuals)))

        print(f"  {'Overall':<8} {len(all_preds):>8} {overall_mae:>8.3f} "
              f"{overall_r2:>8.3f} {overall_dir:>8.1%} {baseline_all:>13.3f}")
        print(f"\n  MAE improvement vs baseline: "
              f"{(baseline_all - overall_mae)/baseline_all*100:.1f}%")
        print(f"  Direction accuracy: {overall_dir:.1%} "
              f"(50% = random, >55% = useful signal)")
        print(f"\n  ℹ️  To backtest betting P/L, provide real historical")
        print(f"     prop lines via --fetch --api-key YOUR_KEY")
    else:
        overall_mae = overall_r2 = 0
        overall_dir = 0.5

    return {
        "prop_type":    prop_type,
        "seasons":      season_results,
        "overall_mae":  overall_mae if all_preds else 0,
        "overall_r2":   overall_r2  if all_preds else 0,
        "dir_accuracy": overall_dir if all_preds else 0,
        "total_bets":   0,
        "roi":          0,
        "net_profit":   0,
    }



# ══════════════════════════════════════════════
# COMPARISON: PROPS vs GAME MODEL
# ══════════════════════════════════════════════

def compare_vs_game_model(props_results: dict,
                           game_roi: float   = 79.3,
                           game_bets: int    = 159,
                           game_profit: float= 793.03):
    """Print comparison of props accuracy vs game model results."""
    print(f"\n{'═'*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'─'*70}")
    print(f"  {'Model':<30} {'Metric':<15} {'Value':>10}")
    print(f"{'─'*70}")
    print(f"  {'Game Model (ML+RL Mode B)':<30} {'ROI':<15} {game_roi:>+9.1f}%")
    print(f"  {'':<30} {'Bets':<15} {game_bets:>10}")
    print(f"  {'':<30} {'Win Rate':<15} {'~52.8%':>10}")
    print(f"{'─'*70}")

    for pt, r in props_results.items():
        if not r:
            continue
        label   = PROP_CONFIGS[pt]["label"]
        mae     = r.get("overall_mae", 0)
        r2      = r.get("overall_r2", 0)
        dir_acc = r.get("dir_accuracy", 0)
        seasons = r.get("seasons", [])
        n_seasons = len(seasons)

        print(f"  {label:<30} {'MAE':<15} {mae:>10.3f}")
        print(f"  {'':<30} {'R²':<15} {r2:>10.3f}")
        print(f"  {'':<30} {'Dir Accuracy':<15} {dir_acc:>9.1%}")
        print(f"  {'':<30} {'Seasons tested':<15} {n_seasons:>10}")
        print(f"{'─'*70}")

    print(f"\n  HOW TO READ PROPS ACCURACY:")
    print(f"  Direction accuracy > 55% = model has real predictive signal")
    print(f"  MAE improvement vs baseline = how much better than just using avg")
    print(f"  R² > 0.10 = meaningful signal (props are inherently noisy)")
    print(f"\n  Props P/L backtest requires real historical lines.")
    print(f"  Run: python props_model.py --fetch --api-key YOUR_KEY")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════
# LIVE PREDICTION (when you have a line to bet)
# ══════════════════════════════════════════════

def predict_prop_vs_line(prop_type: str,
                          player: str,
                          book_line: float,
                          game_date: str,
                          batting_logs: pd.DataFrame,
                          pitching_logs: pd.DataFrame,
                          team_bat_stats: pd.DataFrame,
                          team_pit_stats: pd.DataFrame,
                          home_team: str = "",
                          opp_team: str  = "",
                          opp_pitcher: str = "",
                          opp_hand: str    = "R",
                          bankroll: float  = 1000.0,
                          min_edge: float  = 0.06) -> dict:
    """
    Given a player, a prop type, and a book line,
    predict whether to bet over or under.

    This is the main function for live daily use.
    """
    from scipy import stats

    model_path = MODELS_DIR / f"{prop_type}.pkl"
    if not model_path.exists():
        return {"error": f"No model for {prop_type}. Run --train first."}

    model_pkg = joblib.load(model_path)
    cfg       = PROP_CONFIGS[prop_type]

    # Build features
    gd = pd.Timestamp(game_date)
    if cfg["log_type"] == "pitching":
        # Use live fetch — only pulls last 45 days for this pitcher
        from statcast_logs import fetch_recent_starts
        print(f"  Fetching recent starts for {player}...")
        start_logs = fetch_recent_starts(player, days_back=45)
        if not start_logs.empty:
            start_logs["name"] = player
        feats = build_pitcher_k_features(
            player, gd, pitching_logs, opp_team,
            team_bat_stats, home_team,
            start_logs=start_logs if not start_logs.empty else None,
        )
    else:
        feats = build_batter_features(
            player, gd, batting_logs, pitching_logs,
            opp_pitcher, opp_hand, team_pit_stats,
            home_team, prop_type
        )

    predicted = predict_stat(model_pkg, feats)
    std       = model_pkg.get("target_std", 2.0)

    # Probability of going over/under the book line
    p_over  = 1 - stats.norm.cdf(book_line, loc=predicted, scale=std)
    p_under = stats.norm.cdf(book_line, loc=predicted, scale=std)

    result = {
        "player":    player,
        "prop_type": prop_type,
        "book_line": book_line,
        "predicted": round(predicted, 2),
        "edge_raw":  round(predicted - book_line, 2),
    }

    for direction, prob, odds in [
        ("OVER",  p_over,  -110),
        ("UNDER", p_under, -110),
    ]:
        bp   = bk_prob(odds)
        edge = prob - bp
        result[f"{direction.lower()}_prob"] = round(prob, 4)
        result[f"{direction.lower()}_edge"] = round(edge, 4)

        if edge >= min_edge and ev(prob, odds) > 0:
            stake = bankroll * kelly(prob, odds)
            result["bet_direction"] = direction
            result["bet_prob"]      = round(prob, 4)
            result["bet_edge"]      = round(edge, 4)
            result["bet_stake"]     = round(stake, 2)
            result["recommendation"]= (
                f"✅ BET {direction} {book_line}  "
                f"model:{predicted:.1f}  "
                f"prob:{prob:.1%}  edge:{edge:+.1%}  "
                f"stake:${stake:.0f}"
            )

    if "recommendation" not in result:
        result["recommendation"] = (
            f"❌ No edge  model:{predicted:.1f}  "
            f"book:{book_line}  "
            f"over:{p_over:.1%}  under:{p_under:.1%}"
        )

    return result


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Player Props Model")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--compare",     action="store_true")
    parser.add_argument("--predict",     action="store_true")
    parser.add_argument("--seasons",     type=int, nargs="+",
                        default=list(range(2014, 2022)))
    parser.add_argument("--bankroll",    type=float, default=1000.0)
    parser.add_argument("--edge",        type=float, default=0.06)
    # For --predict mode
    parser.add_argument("--player",      type=str, default="")
    parser.add_argument("--prop",        type=str,
                        default="pitcher_strikeouts",
                        choices=list(PROP_CONFIGS.keys()))
    parser.add_argument("--line",        type=float, default=0.0)
    parser.add_argument("--date",        type=str,   default="2021-09-01")
    parser.add_argument("--home-team",   type=str,   default="")
    parser.add_argument("--opp-team",    type=str,   default="")
    parser.add_argument("--opp-pitcher", type=str,   default="")
    parser.add_argument("--opp-hand",    type=str,   default="R")
    args = parser.parse_args()

    if args.train:
        print("⚾  Training Player Props Models\n")
        batting  = load_batting_logs(args.seasons)
        pitching = load_pitching_logs(args.seasons)
        team_bat = load_team_batting_stats(args.seasons)
        team_pit = load_team_pitching_stats(args.seasons)

        print("\n  Loading Statcast per-start logs (downloads ~500MB on first run)...")
        start_logs = load_statcast_logs(args.seasons, pitching)

        for pt in PROP_CONFIGS:
            df = build_training_dataset(
                pt, batting, pitching, team_bat, team_pit,
                start_logs=start_logs if pt == "pitcher_strikeouts" else None
            )
            if not df.empty:
                train_prop_model(df, pt)

    elif args.walkforward:
        print("⚾  Props Walk-Forward Validation\n")
        batting  = load_batting_logs(args.seasons)
        pitching = load_pitching_logs(args.seasons)
        team_bat = load_team_batting_stats(args.seasons)
        team_pit = load_team_pitching_stats(args.seasons)

        results = {}
        for pt in PROP_CONFIGS:
            results[pt] = run_props_walkforward(
                pt, batting, pitching, team_bat, team_pit,
                min_edge=args.edge, bankroll=args.bankroll
            )

        if args.compare:
            compare_vs_game_model(results)
        else:
            # Print summary
            print(f"\n{'═'*60}")
            for pt, r in results.items():
                if r:
                    print(f"  {PROP_CONFIGS[pt]['label']:<28} "
                          f"ROI: {r['roi']:+.2f}%  "
                          f"P/L: ${r['net_profit']:+.2f}")
            print(f"{'═'*60}")

    elif args.compare:
        results = {}
        for pt in PROP_CONFIGS:
            path = MODELS_DIR / f"{pt}_walkforward.pkl"
            if path.exists():
                results[pt] = joblib.load(path)
        compare_vs_game_model(results)

    elif args.predict:
        if not args.player or not args.line:
            print("❌ Provide --player NAME --line 5.5 --prop pitcher_strikeouts")
        else:
            batting  = load_batting_logs(args.seasons)
            pitching = load_pitching_logs(args.seasons)
            team_bat = load_team_batting_stats(args.seasons)
            team_pit = load_team_pitching_stats(args.seasons)

            result = predict_prop_vs_line(
                args.prop, args.player, args.line, args.date,
                batting, pitching, team_bat, team_pit,
                home_team=args.home_team, opp_team=args.opp_team,
                opp_pitcher=args.opp_pitcher, opp_hand=args.opp_hand,
                bankroll=args.bankroll, min_edge=args.edge,
            )
            print(f"\n{result['recommendation']}")
            print(f"  Predicted: {result['predicted']}  "
                  f"Book line: {result['book_line']}  "
                  f"Raw edge: {result['edge_raw']:+.2f}\n")

    else:
        print("Usage:")
        print("  python props_model.py --train --seasons 2014 2015 2016 2017 2018 2019 2020 2021")
        print("  python props_model.py --walkforward")
        print("  python props_model.py --walkforward --compare")
        print("  python props_model.py --predict --player 'Jacob deGrom' "
              "--prop pitcher_strikeouts --line 7.5 --date 2021-09-01")
