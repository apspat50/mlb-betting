"""
statcast_logs.py
----------------
Fetches and processes Statcast pitch-by-pitch data aggregated to per-start level.

This replaces season-average pitching stats with game-by-game actuals,
giving the model real rolling features like:
  - Last 3 starts: avg velocity, SwStr%, K/start
  - Velocity trend (gaining or losing velo)
  - Spin rate trend
  - Pitch mix changes
  - Recent form vs season baseline

Data source: pybaseball.statcast_pitcher() — free, no API key needed
Download size: ~200-500MB per season, cached after first run

Key functions:
  fetch_pitcher_game_logs(seasons)   — download + cache per-start data
  build_pitcher_rolling_features(name, date, logs_df) — rolling features
  get_pitcher_id(name)               — look up MLBAM player ID
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR  = Path("props_data")
CACHE_DIR = Path("cache") / "statcast"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Minimum starts before using rolling features ──
MIN_STARTS = 3


# ══════════════════════════════════════════════
# PLAYER ID LOOKUP
# ══════════════════════════════════════════════

# Cache player ID lookups to avoid repeated API calls
_player_id_cache = {}

def get_pitcher_id(name: str) -> int:
    """
    Look up a pitcher's MLBAM player ID by name.
    Uses pybaseball's playerid_lookup function.
    Returns 0 if not found.
    """
    if name in _player_id_cache:
        return _player_id_cache[name]

    try:
        import pybaseball as pyb

        # Parse name — try "First Last" format
        parts = name.strip().split()
        if len(parts) < 2:
            return 0

        last  = parts[-1]
        first = parts[0]

        result = pyb.playerid_lookup(last, first)
        if result.empty:
            # Try fuzzy — just last name
            result = pyb.playerid_lookup(last)

        if not result.empty:
            # Get most recent player (highest key_mlbam)
            pid = int(result.sort_values("mlb_played_last",
                                          ascending=False).iloc[0]["key_mlbam"])
            _player_id_cache[name] = pid
            return pid

    except Exception as e:
        pass

    _player_id_cache[name] = 0
    return 0


def load_pitcher_id_map(pitching_logs: pd.DataFrame) -> dict:
    """
    Build a name → player_id mapping for all pitchers in the dataset.
    Batches the lookups to avoid hammering the API.
    """
    cache_file = DATA_DIR / "pitcher_id_map.pkl"
    if cache_file.exists():
        return joblib.load(cache_file)

    try:
        import pybaseball as pyb
    except ImportError:
        return {}

    names   = pitching_logs["Name"].dropna().unique().tolist()
    id_map  = {}
    n       = len(names)

    print(f"  Looking up player IDs for {n} pitchers...")

    for i, name in enumerate(names):
        if i % 50 == 0:
            print(f"    {i}/{n} lookups done", end="\r")
        pid = get_pitcher_id(name)
        if pid > 0:
            id_map[name] = pid
        time.sleep(0.1)   # be polite

    print(f"\n  ✅ Found IDs for {len(id_map)}/{n} pitchers")
    joblib.dump(id_map, cache_file)
    return id_map


# ══════════════════════════════════════════════
# STATCAST DATA FETCHING
# ══════════════════════════════════════════════

def fetch_pitcher_season_logs(player_id: int,
                               season: int) -> pd.DataFrame:
    """
    Fetch all pitches for one pitcher in one season from Statcast.
    Aggregates to per-start level.
    Returns DataFrame with one row per start.
    """
    cache_file = CACHE_DIR / f"pitcher_{player_id}_{season}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    try:
        import pybaseball as pyb
        pyb.cache.enable()

        df = pyb.statcast_pitcher(
            start_dt = f"{season}-03-01",
            end_dt   = f"{season}-11-30",
            player_id= player_id,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Aggregate to per-start
        starts = aggregate_to_starts(df, player_id, season)

        if not starts.empty:
            starts.to_parquet(cache_file, index=False)

        time.sleep(1)
        return starts

    except Exception as e:
        return pd.DataFrame()


def aggregate_to_starts(df: pd.DataFrame,
                         player_id: int,
                         season: int) -> pd.DataFrame:
    """
    Aggregate pitch-by-pitch Statcast data to one row per start.

    Per-start stats computed:
      SO, BB, HR, H, IP (estimated)
      avg_velo, max_velo, velo_std
      swstr_pct (swinging strikes / total pitches)
      csw_pct (called strike + whiff %)
      zone_pct (pitches in zone %)
      fb_pct, sl_pct, ch_pct, cu_pct (pitch mix)
      avg_spin_rate
      hard_hit_pct (exit velo > 95 mph)
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Identify starts vs relief (starts have many pitches early in game)
    # Group by game_date
    starts = []

    for game_date, game_df in df.groupby("game_date"):
        pitches = len(game_df)
        if pitches < 50:   # filter out relief appearances (starts = 70+ pitches typically)
            continue

        # Basic counting stats
        # Include strikeout_double_play as a K
        so  = game_df["events"].isin(["strikeout", "strikeout_double_play"]).sum()
        bb  = game_df["events"].isin(["walk", "hit_by_pitch"]).sum()
        hr  = (game_df["events"] == "home_run").sum()
        h   = game_df["events"].isin(
            ["single","double","triple","home_run"]
        ).sum()

        # IP estimate: count outs recorded (more accurate than bf/3)
        # Single-out events count 1, double-play events count 2
        single_out_events = [
            "field_out", "strikeout", "force_out", "fielders_choice_out",
            "sac_fly", "sac_bunt", "fielders_choice",
        ]
        double_out_events = [
            "double_play", "grounded_into_double_play", "strikeout_double_play",
            "sac_fly_double_play", "sac_bunt_double_play",
        ]
        outs = (
            game_df["events"].isin(single_out_events).sum() +
            game_df["events"].isin(double_out_events).sum() * 2 +
            (game_df["events"] == "triple_play").sum() * 3
        )
        ip  = max(outs / 3.0, 1.0)

        # Velocity
        velo = pd.to_numeric(game_df["release_speed"], errors="coerce").dropna()
        avg_velo = velo.mean()   if len(velo) > 0 else np.nan
        max_velo = velo.max()    if len(velo) > 0 else np.nan
        velo_std = velo.std()    if len(velo) > 0 else np.nan

        # SwStr% — swinging strikes / total pitches
        swinging = game_df["description"].isin(
            ["swinging_strike", "swinging_strike_blocked"]
        ).sum()
        swstr_pct = swinging / pitches if pitches > 0 else np.nan

        # CSW% — called strikes + whiffs / total pitches
        called_strikes = (game_df["description"] == "called_strike").sum()
        csw_pct = (called_strikes + swinging) / pitches if pitches > 0 else np.nan

        # Zone%
        in_zone = pd.to_numeric(game_df["zone"], errors="coerce")
        zone_pct = ((in_zone >= 1) & (in_zone <= 9)).sum() / pitches \
                   if pitches > 0 else np.nan

        # Pitch mix
        pitch_types = game_df["pitch_type"].value_counts(normalize=True)
        fb_pct = pitch_types.get("FF", 0) + pitch_types.get("FT", 0) + \
                 pitch_types.get("SI", 0) + pitch_types.get("FC", 0)
        sl_pct = pitch_types.get("SL", 0)
        ch_pct = pitch_types.get("CH", 0) + pitch_types.get("FS", 0)
        cu_pct = pitch_types.get("CU", 0) + pitch_types.get("KC", 0)

        # Spin rate
        spin = pd.to_numeric(game_df["release_spin_rate"], errors="coerce").dropna()
        avg_spin = spin.mean() if len(spin) > 0 else np.nan

        # Hard hit %
        exit_velo = pd.to_numeric(game_df["launch_speed"], errors="coerce").dropna()
        hard_hit_pct = (exit_velo >= 95).sum() / len(exit_velo) \
                       if len(exit_velo) > 0 else np.nan

        starts.append({
            "game_date":   game_date,
            "season":      season,
            "player_id":   player_id,
            "pitches":     pitches,
            "SO":          so,
            "BB":          bb,
            "HR":          hr,
            "H":           h,
            "IP":          round(ip, 1),
            "avg_velo":    round(avg_velo, 1) if pd.notna(avg_velo) else np.nan,
            "max_velo":    round(max_velo, 1) if pd.notna(max_velo) else np.nan,
            "velo_std":    round(velo_std, 2) if pd.notna(velo_std) else np.nan,
            "swstr_pct":   round(swstr_pct, 4) if pd.notna(swstr_pct) else np.nan,
            "csw_pct":     round(csw_pct, 4)   if pd.notna(csw_pct)   else np.nan,
            "zone_pct":    round(zone_pct, 4)  if pd.notna(zone_pct)  else np.nan,
            "fb_pct":      round(fb_pct, 4),
            "sl_pct":      round(sl_pct, 4),
            "ch_pct":      round(ch_pct, 4),
            "cu_pct":      round(cu_pct, 4),
            "avg_spin":    round(avg_spin, 0) if pd.notna(avg_spin) else np.nan,
            "hard_hit_pct":round(hard_hit_pct, 4) if pd.notna(hard_hit_pct) else np.nan,
        })

    if not starts:
        return pd.DataFrame()

    result = pd.DataFrame(starts).sort_values("game_date").reset_index(drop=True)

    # Add K/9 and BB/9
    result["k9"]  = result["SO"] / result["IP"].clip(lower=0.1) * 9
    result["bb9"] = result["BB"] / result["IP"].clip(lower=0.1) * 9

    return result


def fetch_all_pitcher_logs(pitcher_names: list,
                            id_map: dict,
                            seasons: list) -> pd.DataFrame:
    """
    Fetch per-start logs for all pitchers across all seasons.
    Main entry point for bulk data download.
    """
    cache_file = DATA_DIR / f"all_starts_{'_'.join(map(str,seasons))}.parquet"
    if cache_file.exists():
        print(f"  📂 Per-start logs from cache ({cache_file.name})")
        return pd.read_parquet(cache_file)

    all_starts = []
    total = len(pitcher_names) * len(seasons)
    done  = 0

    print(f"  Downloading Statcast per-start data...")
    print(f"  {len(pitcher_names)} pitchers × {len(seasons)} seasons "
          f"= {total} fetches")
    print(f"  ⏳ Estimated time: {total * 2 // 60} minutes\n")

    for pitcher in pitcher_names:
        pid = id_map.get(pitcher, 0)
        if pid == 0:
            done += len(seasons)
            continue

        for season in seasons:
            done += 1
            if done % 50 == 0:
                pct = done / total * 100
                print(f"  [{pct:.0f}%] {pitcher} {season}", end="\r")

            starts = fetch_pitcher_season_logs(pid, season)
            if not starts.empty:
                starts["name"] = pitcher
                all_starts.append(starts)

    print()

    if not all_starts:
        print("  ⚠️  No per-start data fetched")
        return pd.DataFrame()

    result = pd.concat(all_starts, ignore_index=True)
    result["game_date"] = pd.to_datetime(result["game_date"])
    result = result.sort_values(["name","game_date"]).reset_index(drop=True)

    result.to_parquet(cache_file, index=False)
    print(f"  ✅ Cached {len(result)} starts → {cache_file.name}")
    return result


# ══════════════════════════════════════════════
# ROLLING FEATURE BUILDER
# This is the core function called per game prediction
# ══════════════════════════════════════════════

def build_pitcher_rolling_features(pitcher_name: str,
                                    game_date,
                                    start_logs: pd.DataFrame,
                                    windows: list = [3, 5]) -> dict:
    """
    Build rolling per-start features for a pitcher as of game_date.
    Only uses starts BEFORE game_date (no leakage).

    Features include:
      Rolling K/start, K/9, SwStr%, CSW%, velocity
      Velocity trend (gaining or losing velo)
      Form score (recent vs season baseline)
      Days since last start
      Pitch mix stability

    Returns empty dict if insufficient history.
    """
    f = {}

    if start_logs.empty or "name" not in start_logs.columns:
        return f

    gd = pd.Timestamp(game_date)

    # Get this pitcher's starts before today
    p = start_logs[
        (start_logs["name"] == pitcher_name) &
        (start_logs["game_date"] < gd)
    ].sort_values("game_date").copy()

    if len(p) < MIN_STARTS:
        return f

    # ── K strikeouts per start ──
    so = p["SO"].astype(float)
    for w in windows:
        f[f"sc_k_L{w}"]    = so.tail(w).mean()
    f["sc_k_season"]       = so.mean()
    f["sc_k_std"]          = so.tail(10).std()

    # ── K/9 ──
    k9 = p["k9"].astype(float)
    for w in windows:
        f[f"sc_k9_L{w}"]   = k9.tail(w).mean()
    f["sc_k9_season"]      = k9.mean()

    # ── Swinging strike % — best K predictor ──
    swstr = p["swstr_pct"].astype(float)
    for w in windows:
        f[f"sc_swstr_L{w}"]= swstr.tail(w).mean()
    f["sc_swstr_season"]   = swstr.mean()

    # ── CSW% ──
    csw = p["csw_pct"].astype(float)
    for w in windows:
        f[f"sc_csw_L{w}"]  = csw.tail(w).mean()

    # ── Fastball velocity ──
    velo = p["avg_velo"].astype(float).dropna()
    if len(velo) >= MIN_STARTS:
        f["sc_velo_L3"]    = velo.tail(3).mean()
        f["sc_velo_season"]= velo.mean()
        # Velocity trend: positive = gaining, negative = losing
        if len(velo) >= 5:
            recent  = velo.tail(3).mean()
            earlier = velo.tail(6).head(3).mean()
            f["sc_velo_trend"] = recent - earlier

    # ── Spin rate trend ──
    spin = p["avg_spin"].astype(float).dropna()
    if len(spin) >= MIN_STARTS:
        f["sc_spin_L3"]    = spin.tail(3).mean()
        f["sc_spin_trend"] = spin.tail(3).mean() - spin.tail(6).head(3).mean() \
                             if len(spin) >= 6 else 0

    # ── Hard hit % ──
    hh = p["hard_hit_pct"].astype(float).dropna()
    if len(hh) >= MIN_STARTS:
        f["sc_hard_hit_L3"] = hh.tail(3).mean()

    # ── Pitch mix ──
    f["sc_fb_pct"]  = p["fb_pct"].tail(5).mean()
    f["sc_sl_pct"]  = p["sl_pct"].tail(5).mean()
    f["sc_ch_pct"]  = p["ch_pct"].tail(5).mean()

    # ── Form z-score: recent vs season baseline ──
    # How is this pitcher performing relative to his own average?
    if len(so) >= 8:
        baseline_mean = so.iloc[:-3].mean()
        baseline_std  = so.iloc[:-3].std()
        recent_mean   = so.tail(3).mean()
        if baseline_std > 0.1:
            f["sc_form_z"]    = (recent_mean - baseline_mean) / baseline_std
            f["sc_slump"]     = int(f["sc_form_z"] < -2.0)
            f["sc_hot"]       = int(f["sc_form_z"] > 2.0)
        f["sc_k_trend"]       = so.tail(3).mean() - so.tail(6).head(3).mean()

    # ── SwStr% trend ──
    if len(swstr.dropna()) >= 5:
        recent_swstr  = swstr.dropna().tail(3).mean()
        earlier_swstr = swstr.dropna().tail(6).head(3).mean()
        f["sc_swstr_trend"]   = recent_swstr - earlier_swstr

    # ── Days rest ──
    f["sc_days_rest"] = (gd - p["game_date"].iloc[-1]).days

    # ── Innings pitched trend (workload) ──
    ip = p["IP"].astype(float)
    f["sc_ip_L3"]          = ip.tail(3).mean()
    f["sc_ip_trend"]       = ip.tail(3).mean() - ip.tail(6).head(3).mean() \
                             if len(ip) >= 6 else 0

    # ── Number of starts available ──
    f["sc_n_starts"] = len(p)

    return f


# ══════════════════════════════════════════════
# FEATURE COLUMN LIST
# ══════════════════════════════════════════════

STATCAST_FEATURE_COLS = [
    # K per start rolling
    "sc_k_L3", "sc_k_L5", "sc_k_season", "sc_k_std",
    # K/9 rolling
    "sc_k9_L3", "sc_k9_L5", "sc_k9_season",
    # SwStr% — best K predictor
    "sc_swstr_L3", "sc_swstr_L5", "sc_swstr_season", "sc_swstr_trend",
    # CSW%
    "sc_csw_L3", "sc_csw_L5",
    # Velocity
    "sc_velo_L3", "sc_velo_season", "sc_velo_trend",
    # Spin
    "sc_spin_L3", "sc_spin_trend",
    # Hard hit
    "sc_hard_hit_L3",
    # Pitch mix
    "sc_fb_pct", "sc_sl_pct", "sc_ch_pct",
    # Form
    "sc_form_z", "sc_slump", "sc_hot", "sc_k_trend",
    # Context
    "sc_days_rest", "sc_ip_L3", "sc_ip_trend", "sc_n_starts",
]


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing statcast_logs.py...")

    # Test player ID lookup
    pid = get_pitcher_id("Jacob deGrom")
    print(f"deGrom player ID: {pid}")

    pid2 = get_pitcher_id("Gerrit Cole")
    print(f"Cole player ID: {pid2}")

    if pid > 0:
        print(f"\nFetching deGrom 2021 starts...")
        starts = fetch_pitcher_season_logs(pid, 2021)
        if not starts.empty:
            print(f"Found {len(starts)} starts")
            print(starts[["game_date","SO","IP","avg_velo","swstr_pct"]].head(5))

            # Test rolling features
            feats = build_pitcher_rolling_features(
                "Jacob deGrom",
                "2021-09-01",
                starts.rename(columns={"game_date":"game_date"}).assign(
                    name="Jacob deGrom"
                )
            )
            print(f"\nRolling features ({len(feats)}):")
            for k, v in sorted(feats.items()):
                print(f"  {k:<25} {v:.3f}" if isinstance(v, float) else
                      f"  {k:<25} {v}")


# ══════════════════════════════════════════════
# LIVE DAILY FETCH — only pulls recent starts
# for pitchers scheduled to pitch today
# ══════════════════════════════════════════════

def fetch_recent_starts(pitcher_name: str,
                         days_back: int = 45) -> pd.DataFrame:
    """
    Fetch only the last N days of Statcast data for one pitcher.
    Fast — pulls a few thousand rows instead of millions.
    Used for live daily predictions.

    Args:
        pitcher_name: Full name e.g. "Gerrit Cole"
        days_back:    How many days to look back (default 45 = ~7-8 starts)

    Returns:
        DataFrame with one row per recent start, same format as
        fetch_pitcher_season_logs() output.
    """
    from datetime import datetime, timedelta

    pid = get_pitcher_id(pitcher_name)
    if pid == 0:
        return pd.DataFrame()

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    # Check cache — don't re-fetch if we already got it today
    cache_key = f"recent_{pid}_{end_dt.strftime('%Y%m%d')}"
    cache_file = CACHE_DIR / f"{cache_key}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    try:
        import pybaseball as pyb
        pyb.cache.enable()

        raw = pyb.statcast_pitcher(
            start_dt = start_dt.strftime("%Y-%m-%d"),
            end_dt   = end_dt.strftime("%Y-%m-%d"),
            player_id= pid,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        starts = aggregate_to_starts(raw, pid, end_dt.year)
        starts["name"] = pitcher_name

        if not starts.empty:
            starts.to_parquet(cache_file, index=False)

        return starts

    except Exception as e:
        print(f"  ⚠️  Could not fetch recent starts for {pitcher_name}: {e}")
        return pd.DataFrame()


def fetch_todays_pitcher_starts(pitcher_names: list,
                                 days_back: int = 45) -> pd.DataFrame:
    """
    Fetch recent starts for all pitchers scheduled today.
    Runs in seconds — only pulls data for the 10-16 pitchers
    actually pitching today, not all 500+ in the league.

    Args:
        pitcher_names: List of pitcher names from ESPN schedule
        days_back:     How many days of history to fetch

    Returns:
        Combined DataFrame of recent starts for all pitchers,
        ready to pass to build_pitcher_rolling_features()
    """
    all_starts = []
    n = len(pitcher_names)

    print(f"  Fetching recent starts for {n} pitchers...")

    for i, name in enumerate(pitcher_names):
        if not name or name == "TBD":
            continue

        print(f"  [{i+1}/{n}] {name}...", end=" ")
        starts = fetch_recent_starts(name, days_back)

        if not starts.empty:
            all_starts.append(starts)
            print(f"{len(starts)} starts found")
        else:
            print("no data")

    if not all_starts:
        return pd.DataFrame()

    combined = pd.concat(all_starts, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    return combined.sort_values(["name", "game_date"]).reset_index(drop=True)


# ══════════════════════════════════════════════
# MLB STATS API — ACCURATE K COUNTS PER START
# Statcast pitch aggregation is unreliable for K counts.
# MLB Stats API game logs are the official source.
# ══════════════════════════════════════════════

def _parse_ip(ip_val) -> float:
    """
    Convert IP to decimal innings.
    MLB Stats API returns baseball notation (5.1 = 5⅓, 5.2 = 5⅔).
    If decimal part > 2, treat as regular float (already decimal).
    """
    try:
        s = str(ip_val).strip()
        if "." in s:
            whole, frac = s.split(".", 1)
            frac_int = int(frac[:1])  # only look at first decimal digit
            if frac_int <= 2:
                return int(whole) + frac_int / 3.0
            else:
                return float(s)  # already a decimal value
        return float(s)
    except Exception:
        return 0.0


def _fetch_mlb_game_log_season(pid: int, season: int) -> pd.DataFrame:
    """Fetch one season of game log splits from MLB Stats API."""
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people/{}/stats".format(pid),
            params={"stats": "gameLog", "group": "pitching", "season": season},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        splits = []
        for sg in data.get("stats", []):
            splits.extend(sg.get("splits", []))
        return splits
    except Exception:
        return []


def fetch_mlb_pitcher_game_logs(pitcher_name: str,
                                 season: int = None) -> pd.DataFrame:
    """
    Fetch accurate per-start stats from MLB Stats API game logs.
    When the current season has fewer than 5 starts (early season),
    also pulls the previous season to give the model enough history.

    Columns: game_date, name, SO, BB, H, HR, IP, pitches
    """
    from datetime import datetime

    if season is None:
        season = datetime.now().year

    pid = get_pitcher_id(pitcher_name)
    if pid == 0:
        return pd.DataFrame()

    def splits_to_rows(splits, name):
        rows = []
        for s in splits:
            stat      = s.get("stat", {})
            ip        = _parse_ip(stat.get("inningsPitched", "0"))
            n_pitches = int(stat.get("numberOfPitches", 0) or 0)
            # Only starts: IP >= 3 or 70+ pitches
            if ip < 3.0 and n_pitches < 70:
                continue
            rows.append({
                "game_date": pd.Timestamp(s.get("date", "")),
                "name":      name,
                "SO":        int(stat.get("strikeOuts",  0) or 0),
                "BB":        int(stat.get("baseOnBalls", 0) or 0),
                "H":         int(stat.get("hits",        0) or 0),
                "HR":        int(stat.get("homeRuns",    0) or 0),
                "IP":        round(ip, 2),
                "pitches":   n_pitches,
            })
        return rows

    # Current season
    rows = splits_to_rows(_fetch_mlb_game_log_season(pid, season), pitcher_name)

    # If fewer than 5 starts this season (early season), add previous season
    # so rolling averages have enough data to be meaningful
    if len(rows) < 5:
        prev_rows = splits_to_rows(
            _fetch_mlb_game_log_season(pid, season - 1), pitcher_name
        )
        # Append previous season rows before current (chronological order)
        rows = prev_rows + rows

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values("game_date").reset_index(drop=True)


def fetch_mlb_pitcher_season_stats(pid: int, season: int) -> dict:
    """
    Fetch season-aggregate pitching stats from MLB Stats API.
    Returns K/start, IP/start, GS from the full season — more reliable
    than summing individual game logs.
    """
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people/{}/stats".format(pid),
            params={"stats": "season", "group": "pitching",
                    "season": season, "gameType": "R"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        for sg in data.get("stats", []):
            splits = sg.get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            gs   = int(stat.get("gamesStarted", 0) or 0)
            so   = int(stat.get("strikeOuts",   0) or 0)
            ip   = _parse_ip(stat.get("inningsPitched", "0"))
            if gs == 0:
                return {}
            return {
                "gs":           gs,
                "so":           so,
                "ip":           ip,
                "k_per_start":  round(so / gs, 2),
                "ip_per_start": round(ip / max(gs, 1), 2),
                "k9":           round(so / max(ip, 0.1) * 9, 2),
            }
    except Exception:
        pass
    return {}


def fetch_mlb_season_stats_for_pitchers(pitcher_names: list,
                                         seasons: list = None) -> dict:
    """
    Batch fetch season stats for all today's pitchers.
    Returns dict: pitcher_name -> {season -> stats_dict}
    """
    from datetime import datetime
    if seasons is None:
        cur = datetime.now().year
        seasons = [cur, cur - 1]

    result = {}
    n = len(pitcher_names)
    for i, name in enumerate(pitcher_names):
        if not name or name == "TBD":
            continue
        pid = get_pitcher_id(name)
        if pid == 0:
            continue
        result[name] = {}
        for s in seasons:
            stats = fetch_mlb_pitcher_season_stats(pid, s)
            if stats:
                result[name][s] = stats
    return result


def fetch_mlb_k_logs_for_pitchers(pitcher_names: list,
                                   season: int = None) -> dict:
    """
    Batch fetch accurate game-log K data for all today's pitchers.
    Returns dict: pitcher_name -> DataFrame of starts
    """
    from datetime import datetime
    if season is None:
        season = datetime.now().year

    result = {}
    n = len(pitcher_names)
    for i, name in enumerate(pitcher_names):
        if not name or name == "TBD":
            continue
        print(f"  [{i+1}/{n}] MLB logs: {name}...", end=" ")
        logs = fetch_mlb_pitcher_game_logs(name, season)
        if not logs.empty:
            result[name] = logs
            print(f"{len(logs)} starts")
        else:
            print("no data")

    return result


def get_pitcher_features_live(pitcher_name: str,
                               game_date=None,
                               days_back: int = 45) -> dict:
    """
    Get rolling Statcast features for a pitcher using only
    recent starts fetched today. This is the main function
    for live daily predictions.

    Much faster than loading the full historical cache.
    """
    from datetime import datetime
    if game_date is None:
        game_date = datetime.now()

    starts = fetch_recent_starts(pitcher_name, days_back)
    if starts.empty:
        return {}

    return build_pitcher_rolling_features(pitcher_name, game_date, starts)
