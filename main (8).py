"""
main.py  v3
-----------
MLB Betting ML System — pitcher stats, expanded team stats, tuned Kelly.

Usage:
  python main.py --train    --odds mlb-odds-2021.xlsx
  python main.py --predict  --odds mlb-odds-2021.xlsx --date 2021-09-01
  python main.py --backtest --odds mlb-odds-2021.xlsx
"""

import argparse, warnings, os, re
from walk_forward import (
    run_walk_forward, run_all_modes,
    print_walk_forward_report, print_comparison,
    save_walk_forward_results, BASE_42_COLS,
)
from player_intelligence import (
    attach_player_intelligence, build_transaction_lookup,
    PLAYER_INTEL_COLS
)
from matchup_features import (
    build_all_matchup_features, fetch_top_batters, build_lineup_strength,
    MATCHUP_FEATURE_COLS
)
from stats_fetcher import (
    load_advanced_stats, get_team_advanced_stats, get_pitcher_advanced_stats,
    ALL_ADVANCED_COLS, ADVANCED_EDGE_COLS
)
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("Run: pip install xgboost")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    raise ImportError("Run: pip install scikit-learn")

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════

def _col(df, *candidates):
    for c in candidates:
        if c in df.columns: return c
    low = {x.lower().replace(" ", ""): x for x in df.columns}
    for c in candidates:
        if c.lower().replace(" ", "") in low:
            return low[c.lower().replace(" ", "")]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


def _unnamed(df, position: int):
    target = f"Unnamed: {position}"
    if target in df.columns: return target
    numbered = sorted([(int(c.split(": ")[1]), c)
                       for c in df.columns if c.startswith("Unnamed:")])
    return min(numbered, key=lambda x: abs(x[0] - position))[1]


def load_sbr_file(filepath: str, season: int = None) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\n❌ File not found: '{filepath}'\n")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, parse_dates=["date"])
        print(f"✅ Loaded {len(df)} games from {filepath}")
        return df

    try:
        import openpyxl  # noqa
    except ImportError:
        raise ImportError("Run: pip install openpyxl")

    raw = pd.read_excel(filepath)

    if season is None:
        match = re.search(r"(20\d{2})", filepath)
        season = int(match.group(1)) if match else 2021

    def parse_date(d):
        d = str(int(d)).zfill(4)
        return pd.Timestamp(year=season, month=int(d[:2]), day=int(d[2:]))

    raw["date_parsed"] = raw["Date"].apply(parse_date)

    team_map = {
        "PIT": "Pittsburgh Pirates",   "CUB": "Chicago Cubs",
        "ATL": "Atlanta Braves",       "PHI": "Philadelphia Phillies",
        "ARI": "Arizona Diamondbacks", "SDG": "San Diego Padres",
        "SDP": "San Diego Padres",     "LAD": "Los Angeles Dodgers",
        "COL": "Colorado Rockies",     "STL": "St. Louis Cardinals",
        "CIN": "Cincinnati Reds",      "MIL": "Milwaukee Brewers",
        "MIA": "Miami Marlins",        "NYM": "New York Mets",
        "WSH": "Washington Nationals", "WAS": "Washington Nationals",
        "SFG": "San Francisco Giants", "SFO": "San Francisco Giants",
        "CHW": "Chicago White Sox",    "CLE": "Cleveland Indians",
        "GUA": "Cleveland Guardians",  "DET": "Detroit Tigers",
        "KAN": "Kansas City Royals",   "KCR": "Kansas City Royals",
        "MIN": "Minnesota Twins",      "HOU": "Houston Astros",
        "OAK": "Oakland Athletics",    "SEA": "Seattle Mariners",
        "TEX": "Texas Rangers",        "LAA": "Los Angeles Angels",
        "NYY": "New York Yankees",     "BOS": "Boston Red Sox",
        "TOR": "Toronto Blue Jays",    "TBR": "Tampa Bay Rays",
        "TBA": "Tampa Bay Rays",       "BAL": "Baltimore Orioles",
    }
    raw["team_name"] = raw["Team"].map(team_map).fillna(raw["Team"])

    # Normalize neutral-site rows (N) to V/H based on odd/even rotation number
    raw = raw.copy()
    mask = raw["VH"] == "N"
    raw.loc[mask, "VH"] = raw.loc[mask, "Rot"].apply(
        lambda r: "V" if int(r) % 2 == 1 else "H"
    )

    visitor = raw[raw["VH"] == "V"].reset_index(drop=True)
    home    = raw[raw["VH"] == "H"].reset_index(drop=True)

    # Detect format: pre-2014 files have no run line column
    # and OU columns are at positions 18/20 instead of 22
    has_runline = any(c in home.columns for c in ["RunLine", "Run Line"])
    has_ou_22   = "Unnamed: 22" in home.columns

    df = pd.DataFrame({
        "date":            home["date_parsed"],
        "home_team":       home["team_name"],
        "away_team":       visitor["team_name"],
        "home_pitcher":    home["Pitcher"],
        "away_pitcher":    visitor["Pitcher"],
        "home_score":      pd.to_numeric(home["Final"],                                  errors="coerce"),
        "away_score":      pd.to_numeric(visitor["Final"],                               errors="coerce"),
        "home_ml":         pd.to_numeric(home["Close"],                                  errors="coerce"),
        "away_ml":         pd.to_numeric(visitor["Close"],                               errors="coerce"),
        "home_ml_open":    pd.to_numeric(home["Open"],                                   errors="coerce"),
        "away_ml_open":    pd.to_numeric(visitor["Open"],                                errors="coerce"),
        "run_line":        pd.to_numeric(home[_col(home, "RunLine", "Run Line")],        errors="coerce")
                           if has_runline else np.nan,
        "run_line_odds":   pd.to_numeric(home[_unnamed(home, 18)],                       errors="coerce")
                           if has_runline else np.nan,
        "total_line":      pd.to_numeric(home[_col(home, "CloseOU", "Close OU")],        errors="coerce"),
        "total_line_odds": pd.to_numeric(home[_unnamed(home, 22 if has_ou_22 else 20)],  errors="coerce"),
    })

    df["total_runs"]     = df["home_score"] + df["away_score"]
    df["home_win"]       = (df["home_score"] > df["away_score"]).astype(int)
    df["covered_spread"] = ((df["home_score"] - df["away_score"]) > 1.5).astype(int)
    df["went_over"]      = (df["total_runs"] > df["total_line"]).astype(int)
    df["season"]         = season

    def ml_to_prob(ml):
        ml = float(ml)
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

    df["home_impl_prob"] = df["home_ml"].apply(lambda x: ml_to_prob(x) if pd.notna(x) else np.nan)
    df["away_impl_prob"] = df["away_ml"].apply(lambda x: ml_to_prob(x) if pd.notna(x) else np.nan)
    total = df["home_impl_prob"] + df["away_impl_prob"]
    df["home_true_prob"] = df["home_impl_prob"] / total
    df["away_true_prob"] = df["away_impl_prob"] / total

    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} games from {filepath}")
    return df
def load_multiple_seasons(filepaths: list) -> pd.DataFrame:
    """
    Load and combine multiple SBR odds files into one DataFrame.
    Files are sorted chronologically before combining.
    Pitcher stats carry across seasons (same pitcher names persist year to year).
    """
    frames = []
    for fp in sorted(filepaths):  # sort so 2018 comes before 2019 etc.
        fp = fp.strip()
        if not fp:
            continue
        try:
            df = load_sbr_file(fp)
            frames.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {fp}: {e}")

    if not frames:
        raise ValueError("No valid files loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    # Drop duplicate games (same date + teams) in case of overlapping files
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
    print(f"\n📦 Combined {len(frames)} seasons: {len(combined)} total games")
    print(f"   Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
    return combined




# ══════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════

def build_pitcher_stats(df: pd.DataFrame) -> dict:
    """
    Build rolling pitcher stats from game results.
    Returns dict: pitcher_name -> DataFrame indexed by date with:
      - era_L5       : runs allowed per 9 innings proxy (runs allowed / starts * 9)
      - win_rate_L10 : win rate in last 10 starts
      - avg_runs_allowed_L5
      - start_count  : number of starts seen so far (for reliability)
    All stats are shifted to prevent leakage.
    """
    pitcher_games = {}

    for _, game in df.sort_values("date").iterrows():
        for pitcher, runs_allowed, won in [
            (game["home_pitcher"], game["away_score"], game["home_win"]),
            (game["away_pitcher"], game["home_score"], 1 - game["home_win"]),
        ]:
            if pd.isna(pitcher) or pitcher == "TBD":
                continue
            if pitcher not in pitcher_games:
                pitcher_games[pitcher] = []
            pitcher_games[pitcher].append({
                "date":         game["date"],
                "runs_allowed": runs_allowed,
                "won":          won,
            })

    pitcher_stats = {}
    for pitcher, games in pitcher_games.items():
        p = pd.DataFrame(games).sort_values("date").reset_index(drop=True)
        p["era_proxy_L5"]        = p["runs_allowed"].shift(1).rolling(5,  min_periods=1).mean()
        p["era_proxy_L10"]       = p["runs_allowed"].shift(1).rolling(10, min_periods=1).mean()
        p["win_rate_L10"]        = p["won"].shift(1).rolling(10, min_periods=1).mean()
        p["runs_allowed_std_L10"]= p["runs_allowed"].shift(1).rolling(10, min_periods=2).std()
        p["start_count"]         = range(len(p))
        pitcher_stats[pitcher]   = p.set_index("date")

    return pitcher_stats


def get_pitcher_feature(pitcher_stats, pitcher, date, col, default=np.nan):
    if pitcher not in pitcher_stats:
        return default
    past = pitcher_stats[pitcher]
    past = past[past.index <= date]
    if len(past) == 0 or col not in past.columns:
        return default
    val = past[col].iloc[-1]
    # Require at least 3 starts for reliability
    if past["start_count"].iloc[-1] < 1:
        return default
    return val


def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Team rolling stats ──
    teams = set(df["home_team"]) | set(df["away_team"])
    team_stats = {}

    for team in teams:
        home_g = df[df["home_team"] == team][["date", "home_score", "away_score"]].copy()
        away_g = df[df["away_team"] == team][["date", "home_score", "away_score"]].copy()
        home_g["scored"]  = home_g["home_score"]
        home_g["allowed"] = home_g["away_score"]
        home_g["is_home"] = 1
        away_g["scored"]  = away_g["away_score"]
        away_g["allowed"] = away_g["home_score"]
        away_g["is_home"] = 0

        combined = pd.concat([
            home_g[["date", "scored", "allowed", "is_home"]],
            away_g[["date", "scored", "allowed", "is_home"]]
        ]).sort_values("date").reset_index(drop=True)

        s = combined["scored"]
        a = combined["allowed"]

        combined[f"avg_scored_L{window}"]   = s.shift(1).rolling(window, min_periods=1).mean()
        combined[f"avg_allowed_L{window}"]  = a.shift(1).rolling(window, min_periods=1).mean()
        combined["avg_scored_L5"]           = s.shift(1).rolling(5,      min_periods=1).mean()
        combined["avg_allowed_L5"]          = a.shift(1).rolling(5,      min_periods=1).mean()
        combined["win_pct_L20"]             = (s > a).shift(1).rolling(20, min_periods=1).mean()
        combined["win_pct_L10"]             = (s > a).shift(1).rolling(10, min_periods=1).mean()
        # Scoring consistency (lower std = more consistent)
        combined["scored_std_L10"]          = s.shift(1).rolling(10, min_periods=2).std()
        combined["allowed_std_L10"]         = a.shift(1).rolling(10, min_periods=2).std()
        # Run differential rolling
        combined["run_diff"]                = s - a
        combined["avg_run_diff_L10"]        = combined["run_diff"].shift(1).rolling(10, min_periods=1).mean()
        # Home-specific splits
        home_only = combined[combined["is_home"] == 1].copy()
        home_only["home_avg_scored_L10"]    = home_only["scored"].shift(1).rolling(10, min_periods=1).mean()
        home_only["home_avg_allowed_L10"]   = home_only["allowed"].shift(1).rolling(10, min_periods=1).mean()
        combined = combined.merge(
            home_only[["date", "home_avg_scored_L10", "home_avg_allowed_L10"]],
            on="date", how="left"
        ).ffill()

        team_stats[team] = combined.set_index("date")

    # ── Pitcher rolling stats ──
    pitcher_stats = build_pitcher_stats(df)

    # ── Assemble per-game feature rows ──
    rows = []
    for _, game in df.iterrows():
        d, home, away = game["date"], game["home_team"], game["away_team"]

        def get_team(team, col):
            ts = team_stats.get(team)
            if ts is None: return np.nan
            past = ts[ts.index <= d]
            return past[col].iloc[-1] if len(past) > 0 and col in past.columns else np.nan

        row = game.to_dict()

        # Team stats — home
        for col in [f"avg_scored_L{window}", f"avg_allowed_L{window}",
                    "avg_scored_L5", "avg_allowed_L5", "win_pct_L20", "win_pct_L10",
                    "scored_std_L10", "allowed_std_L10", "avg_run_diff_L10",
                    "home_avg_scored_L10", "home_avg_allowed_L10"]:
            row[f"home_{col}"] = get_team(home, col)

        # Team stats — away
        for col in [f"avg_scored_L{window}", f"avg_allowed_L{window}",
                    "avg_scored_L5", "avg_allowed_L5", "win_pct_L20", "win_pct_L10",
                    "scored_std_L10", "allowed_std_L10", "avg_run_diff_L10"]:
            row[f"away_{col}"] = get_team(away, col)

        # Pitcher stats
        hp, ap = game.get("home_pitcher"), game.get("away_pitcher")
        for feat in ["era_proxy_L5", "era_proxy_L10", "win_rate_L10", "runs_allowed_std_L10"]:
            row[f"home_p_{feat}"] = get_pitcher_feature(pitcher_stats, hp, d, feat)
            row[f"away_p_{feat}"] = get_pitcher_feature(pitcher_stats, ap, d, feat)

        rows.append(row)

    out = pd.DataFrame(rows)

    # ── Derived matchup features ──
    out["run_diff_edge"]      = out[f"home_avg_scored_L{window}"] - out[f"away_avg_scored_L{window}"]
    out["defense_edge"]       = out[f"away_avg_allowed_L{window}"] - out[f"home_avg_allowed_L{window}"]
    out["projected_total"]    = out[f"home_avg_scored_L{window}"] + out[f"away_avg_scored_L{window}"]
    out["total_vs_line"]      = out["projected_total"] - out["total_line"]
    out["win_pct_edge"]       = out["home_win_pct_L20"] - out["away_win_pct_L20"]
    out["run_diff_edge_total"]= out["home_avg_run_diff_L10"] - out["away_avg_run_diff_L10"]
    out["ml_edge"]            = out["home_true_prob"] - out["away_true_prob"]
    # Pitcher matchup edge: lower ERA = better pitcher
    out["pitcher_era_edge"]   = out["away_p_era_proxy_L5"] - out["home_p_era_proxy_L5"]
    out["pitcher_win_edge"]   = out["home_p_win_rate_L10"] - out["away_p_win_rate_L10"]
    # Line movement signal
    out["home_ml_move"]       = out["home_ml"] - out["home_ml_open"]
    out["away_ml_move"]       = out["away_ml"] - out["away_ml_open"]

    # ── Player intelligence features (attached if --players flag used) ──
    for col in PLAYER_INTEL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    # ── Matchup & situational features (attached if --matchup flag used) ──
    for col in MATCHUP_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan

    # ── Advanced stats (attached if available via --advanced flag) ──
    # These are filled in by attach_advanced_stats() after build_features()
    # Placeholder columns so FEATURE_COLS can reference them
    for col in ALL_ADVANCED_COLS:
        if col not in out.columns:
            out[col] = np.nan

    return out


FEATURE_COLS = [
    # Team offense
    "home_avg_scored_L10", "home_avg_scored_L5",
    "away_avg_scored_L10", "away_avg_scored_L5",
    # Team defense
    "home_avg_allowed_L10", "home_avg_allowed_L5",
    "away_avg_allowed_L10", "away_avg_allowed_L5",
    # Win rates
    "home_win_pct_L20", "home_win_pct_L10",
    "away_win_pct_L20", "away_win_pct_L10",
    # Consistency
    "home_scored_std_L10", "home_allowed_std_L10",
    "away_scored_std_L10", "away_allowed_std_L10",
    # Run differential
    "home_avg_run_diff_L10", "away_avg_run_diff_L10",
    # Home splits
    "home_home_avg_scored_L10", "home_home_avg_allowed_L10",
    # Pitcher stats
    "home_p_era_proxy_L5", "home_p_era_proxy_L10",
    "home_p_win_rate_L10", "home_p_runs_allowed_std_L10",
    "away_p_era_proxy_L5", "away_p_era_proxy_L10",
    "away_p_win_rate_L10", "away_p_runs_allowed_std_L10",
    # Matchup edges
    "run_diff_edge", "defense_edge", "win_pct_edge",
    "run_diff_edge_total", "pitcher_era_edge", "pitcher_win_edge",
    # Totals
    "projected_total", "total_vs_line", "total_line",
    # Market signals
    "home_true_prob", "away_true_prob", "ml_edge",
    "home_ml_move", "away_ml_move",
]

# Extended feature set including advanced stats (used when --advanced flag is set)
FEATURE_COLS_ADVANCED  = FEATURE_COLS + ALL_ADVANCED_COLS
FEATURE_COLS_MATCHUP   = FEATURE_COLS + MATCHUP_FEATURE_COLS
FEATURE_COLS_FULL      = FEATURE_COLS + ALL_ADVANCED_COLS + MATCHUP_FEATURE_COLS
FEATURE_COLS_INTEL     = FEATURE_COLS + MATCHUP_FEATURE_COLS + PLAYER_INTEL_COLS
FEATURE_COLS_ALL       = FEATURE_COLS + ALL_ADVANCED_COLS + MATCHUP_FEATURE_COLS + PLAYER_INTEL_COLS


def attach_advanced_stats(df: pd.DataFrame,
                          team_lookup: dict,
                          pitcher_lookup: dict) -> pd.DataFrame:
    """
    Attach season-level FanGraphs stats to each game row.
    IMPORTANT: Uses PREVIOUS season stats to avoid leakage.
    e.g. a game on 2018-05-01 uses 2017 season stats as the baseline.
    After the All-Star break (~July 15) we blend in current season stats
    since enough games have been played to be meaningful.
    """
    from stats_fetcher import get_team_advanced_stats, get_pitcher_advanced_stats

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for _, game in df.iterrows():
        season     = game["date"].year
        month      = game["date"].month
        # Before All-Star break: use last season. After: blend current + last.
        # This mirrors how a real bettor would use available stats.
        prev_season = season - 1
        row = game.to_dict()

        # Primary stats: always previous season (fully known, no leakage)
        row.update(get_team_advanced_stats(team_lookup, game["home_team"], prev_season, "home"))
        row.update(get_team_advanced_stats(team_lookup, game["away_team"], prev_season, "away"))
        row.update(get_pitcher_advanced_stats(pitcher_lookup, game.get("home_pitcher"), prev_season, "home_sp"))
        row.update(get_pitcher_advanced_stats(pitcher_lookup, game.get("away_pitcher"), prev_season, "away_sp"))

        # After All-Star break (month >= 8): also pull current season stats
        # and overwrite if available — enough games played to be reliable
        if month >= 8:
            curr = {}
            curr.update(get_team_advanced_stats(team_lookup, game["home_team"], season, "home"))
            curr.update(get_team_advanced_stats(team_lookup, game["away_team"], season, "away"))
            curr.update(get_pitcher_advanced_stats(pitcher_lookup, game.get("home_pitcher"), season, "home_sp"))
            curr.update(get_pitcher_advanced_stats(pitcher_lookup, game.get("away_pitcher"), season, "away_sp"))
            # Only overwrite if current season data exists
            for k, v in curr.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)

    # Derived advanced matchup edges
    if "home_bat_wRCplus" in out.columns and "away_bat_wRCplus" in out.columns:
        out["adv_wRC_edge"]     = out["home_bat_wRCplus"]  - out["away_bat_wRCplus"]
    if "home_pit_FIP" in out.columns and "away_pit_FIP" in out.columns:
        out["adv_FIP_edge"]     = out["away_pit_FIP"]      - out["home_pit_FIP"]   # lower = better
        out["adv_xFIP_edge"]    = out["away_pit_xFIP"]     - out["home_pit_xFIP"]
    if "home_sp_FIP" in out.columns and "away_sp_FIP" in out.columns:
        out["adv_sp_FIP_edge"]  = out["away_sp_FIP"]       - out["home_sp_FIP"]
        out["adv_sp_K9_edge"]   = out["home_sp_K_9"]       - out["away_sp_K_9"]
    if "home_sp_SwStrpct" in out.columns and "away_sp_SwStrpct" in out.columns:
        out["adv_sp_SwStr_edge"]= out["home_sp_SwStrpct"]  - out["away_sp_SwStrpct"]

    print(f"  ✅ Advanced stats attached. Non-null rate: "
          f"{out[ALL_ADVANCED_COLS].notna().mean().mean():.1%}")
    return out


# ══════════════════════════════════════════════
# 3. TRAINING — MODEL COMPARISON + ENSEMBLE
# ══════════════════════════════════════════════

def get_candidate_models():
    """
    Returns all candidate models to compare.
    Each is a (name, model) tuple.
    XGBoost needs special handling for early stopping so it's separate.
    """
    return [
        ("XGBoost", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            gamma=1, reg_alpha=0.1, reg_lambda=1,
            eval_metric="logloss", random_state=42,
            early_stopping_rounds=25, verbosity=0,
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )),
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=0.1, max_iter=1000, random_state=42, solver="lbfgs",
            )),
        ])),
    ]


def compare_models(X: np.ndarray, y: np.ndarray,
                   bet_name: str, n_splits: int = 4) -> dict:
    """
    Run time-series cross-validation for every candidate model.
    Returns dict of {model_name: mean_auc} and prints a comparison table.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n{'='*68}")
    print(f"  {bet_name}")
    print(f"{'─'*68}")
    print(f"  {'Model':<22} {'Acc':>7} {'AUC':>7} {'Brier':>7}  Folds")
    print(f"{'─'*68}")

    results = {}
    for name, model in get_candidate_models():
        fold_acc, fold_auc, fold_brier = [], [], []

        for tr, te in tscv.split(X):
            import copy
            m = copy.deepcopy(model)
            # XGBoost needs eval_set for early stopping
            if isinstance(m, XGBClassifier):
                m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
            else:
                m.fit(X[tr], y[tr])

            preds = m.predict(X[te])
            probs = m.predict_proba(X[te])[:, 1]
            fold_acc.append(accuracy_score(y[te], preds))
            fold_auc.append(roc_auc_score(y[te], probs))
            fold_brier.append(brier_score_loss(y[te], probs))

        mean_acc   = np.mean(fold_acc)
        mean_auc   = np.mean(fold_auc)
        mean_brier = np.mean(fold_brier)
        results[name] = {"acc": mean_acc, "auc": mean_auc, "brier": mean_brier}

        fold_str = "  ".join([f"{a:.3f}" for a in fold_auc])
        print(f"  {name:<22} {mean_acc:>7.3f} {mean_auc:>7.3f} {mean_brier:>7.3f}  [{fold_str}]")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"{'─'*68}")
    print(f"  🏆 Best AUC: {best_name} ({results[best_name]['auc']:.3f})")
    print(f"{'='*68}")
    return results


def build_ensemble(X: np.ndarray, y: np.ndarray, comparison_results: dict) -> CalibratedClassifierCV:
    """
    Build a soft-voting ensemble from all candidate models.
    Models are weighted by their cross-validated AUC score —
    better models get more say in the final prediction.

    Soft voting = each model outputs a probability, they're averaged
    (weighted by AUC). This is more powerful than hard voting (majority rules)
    because it uses confidence, not just direction.
    """
    print("  Building weighted ensemble...")

    named_estimators = []
    weights = []

    for name, model in get_candidate_models():
        import copy
        short = name.lower().replace(" ", "_")
        m = copy.deepcopy(model)
        # XGBoost: remove early_stopping_rounds for final fit (no eval_set)
        if isinstance(m, XGBClassifier):
            m.set_params(early_stopping_rounds=None)
            m.fit(X, y, verbose=False)
        else:
            m.fit(X, y)
        model = m
        named_estimators.append((short, model))
        # Weight = AUC score from comparison (higher AUC = more weight)
        auc = comparison_results.get(name, {}).get("auc", 0.5)
        weights.append(max(auc - 0.48, 0.01))  # floor at 0.01, penalize near-random

    ensemble = VotingClassifier(
        estimators=named_estimators,
        voting="soft",
        weights=weights,
    )
    ensemble.fit(X, y)

    # Calibrate the ensemble probabilities
    calibrated = CalibratedClassifierCV(ensemble, method="isotonic", cv=5)
    calibrated.fit(X, y)

    weight_str = ", ".join([f"{e[0]}={w:.3f}" for e, w in zip(named_estimators, weights)])
    print(f"  Weights: {weight_str}")
    return calibrated


def run_training(odds_files, train_frac=0.70, use_advanced=False, use_matchup=False, use_players=False):
    print("\n⚾  MLB Betting ML — Training  (XGBoost + Random Forest + Logistic Regression)\n")

    if isinstance(odds_files, str):
        odds_files = [odds_files]

    if len(odds_files) == 1:
        df = load_sbr_file(odds_files[0])
    else:
        df = load_multiple_seasons(odds_files)

    print("Building features (may take 1-2 min for multiple seasons)...")
    df = build_features(df)

    feat_cols = FEATURE_COLS
    if use_advanced:
        seasons = sorted(df["date"].dt.year.unique().tolist())
        print("\n  Fetching advanced stats from FanGraphs...")
        team_lookup, pitcher_lookup = load_advanced_stats(seasons)
        df = attach_advanced_stats(df, team_lookup, pitcher_lookup)
        feat_cols = FEATURE_COLS_ADVANCED
        joblib.dump({"use_advanced": True,
                     "team_lookup": team_lookup,
                     "pitcher_lookup": pitcher_lookup},
                    MODELS_DIR / "advanced_stats.pkl")
        print(f"  Using {len(feat_cols)} features (base + advanced)\n")
    else:
        joblib.dump({"use_advanced": False}, MODELS_DIR / "advanced_stats.pkl")

    if use_matchup:
        print("\n  Building matchup & situational features...")
        df = build_all_matchup_features(df)

        # Individual batter lineup strength from pybaseball
        seasons = sorted(df["date"].dt.year.unique().tolist())
        print("  Fetching individual batter stats...")
        batter_df = fetch_top_batters(seasons)
        lineup_lk = build_lineup_strength(batter_df)
        joblib.dump({"use_matchup": True, "lineup_lk": lineup_lk},
                    MODELS_DIR / "matchup_cfg.pkl")

        # Attach lineup strength per game
        from stats_fetcher import TEAM_TO_FG
        def attach_lineup(row):
            for side, team in [("home", row["home_team"]), ("away", row["away_team"])]:
                fg  = TEAM_TO_FG.get(team, "")
                key = (fg, row["date"].year)
                for k, v in lineup_lk.get(key, {}).items():
                    row[f"{side}_{k}"] = v
            return row
        df = df.apply(attach_lineup, axis=1)

        # Lineup matchup edges
        if "home_lineup_wRCplus_mean" in df.columns:
            df["lineup_wrc_edge"]   = df["home_lineup_wRCplus_mean"] - df["away_lineup_wRCplus_mean"]
            df["lineup_obp_edge"]   = df["home_lineup_OBPmean"]      - df["away_lineup_OBPmean"] if "home_lineup_OBPmean" in df.columns else np.nan
            df["lineup_iso_edge"]   = df["home_lineup_ISOmean"]      - df["away_lineup_ISOmean"] if "home_lineup_ISOmean" in df.columns else np.nan

        if use_advanced:
            feat_cols = FEATURE_COLS_FULL
        else:
            feat_cols = FEATURE_COLS_MATCHUP
        print(f"  Using {len(feat_cols)} features (base + matchup)\n")
    else:
        joblib.dump({"use_matchup": False}, MODELS_DIR / "matchup_cfg.pkl")

    if use_players:
        print("\n  Building player intelligence features (trades, slumps, positions)...")
        seasons = sorted(df["date"].dt.year.unique().tolist())
        print("  Fetching MLB transaction history...")
        txn_lookup = build_transaction_lookup(seasons)
        df = attach_player_intelligence(df, txn_lookup)
        joblib.dump({"use_players": True, "txn_lookup": txn_lookup},
                    MODELS_DIR / "player_cfg.pkl")
        if use_advanced and use_matchup:
            feat_cols = FEATURE_COLS_ALL
        elif use_matchup:
            feat_cols = FEATURE_COLS_INTEL
        print(f"  Using {len(feat_cols)} features (+ player intelligence)\n")
    else:
        joblib.dump({"use_players": False}, MODELS_DIR / "player_cfg.pkl")

    # Fill nulls in matchup/situational features with sensible defaults
    # rather than dropping entire rows — early season games won't have
    # H2H history, platoon splits, etc. yet but are still valid training rows
    fill_defaults = {
        # Pitcher stats — fill with league average when pitcher has <1 start
        # League avg: ~0.5 runs/inning proxy, 50% win rate, 1.0 std dev
        "home_p_era_proxy_L5":      0.5, "home_p_era_proxy_L10":    0.5,
        "away_p_era_proxy_L5":      0.5, "away_p_era_proxy_L10":    0.5,
        "home_p_win_rate_L10":      0.5, "away_p_win_rate_L10":     0.5,
        "home_p_runs_allowed_std_L10": 1.0, "away_p_runs_allowed_std_L10": 1.0,
        "pitcher_era_edge":         0.0, "pitcher_win_edge":         0.0,
        # Team rolling stats — fill with 0 for first few games of season
        "home_avg_scored_L10":      4.5, "home_avg_scored_L5":       4.5,
        "away_avg_scored_L10":      4.5, "away_avg_scored_L5":       4.5,
        "home_avg_allowed_L10":     4.5, "home_avg_allowed_L5":      4.5,
        "away_avg_allowed_L10":     4.5, "away_avg_allowed_L5":      4.5,
        "home_win_pct_L10":         0.5, "home_win_pct_L20":         0.5,
        "away_win_pct_L10":         0.5, "away_win_pct_L20":         0.5,
        "home_avg_run_diff_L10":    0.0, "away_avg_run_diff_L10":    0.0,
        "home_scored_std_L10":      2.5, "away_scored_std_L10":      2.5,
        "home_allowed_std_L10":     2.5, "away_allowed_std_L10":     2.5,
        "home_home_avg_scored_L10": 4.5, "home_home_avg_allowed_L10":4.5,
        "run_diff_edge":            0.0, "defense_edge":             0.0,
        "win_pct_edge":             0.0, "run_diff_edge_total":      0.0,
        "projected_total":          9.0, "total_vs_line":            0.0,
        "home_ml_move":             0.0, "away_ml_move":             0.0,
        # Handedness — default unknown to 0
        "home_pitcher_hand": 0, "away_pitcher_hand": 0, "same_hand_matchup": 0,
        # Platoon — fill with overall team avg (0 = no edge known)
        "home_platoon_avg_vs_lhp": 0, "home_platoon_avg_vs_rhp": 0,
        "home_platoon_platoon_adv": 0, "away_platoon_avg_vs_lhp": 0,
        "away_platoon_avg_vs_rhp": 0, "away_platoon_platoon_adv": 0,
        "platoon_matchup_edge": 0,
        # Pitcher form — 0 = no known deviation from average
        "home_p_form_vs_season": 0, "home_p_form_trend": 0,
        "away_p_form_vs_season": 0, "away_p_form_trend": 0,
        "pitcher_form_edge": 0,
        # Rest — league average is ~3 days
        "home_days_rest": 3, "away_days_rest": 3,
        "rest_advantage": 0, "fatigue_edge": 0,
        "home_home_streak": 0, "away_home_streak": 0,
        "home_games_last_7": 5, "away_games_last_7": 5,
        # Inning patterns — fill with 0 (unknown)
        "home_avg_early_runs": 0, "home_avg_late_runs": 0, "home_late_clutch": 0,
        "away_avg_early_runs": 0, "away_avg_late_runs": 0, "away_late_clutch": 0,
        "late_inning_edge": 0,
        # H2H — no prior games = 0.5 win rate (no edge)
        "h2h_home_win_rate": 0.5, "h2h_games_played": 0,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Player intel defaults — fill all with 0
    for col in PLAYER_INTEL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Only require core base features to be non-null
    clean = df.dropna(subset=FEATURE_COLS).sort_values("date").copy()
    # Fill any remaining nulls in extended features with 0
    for col in feat_cols:
        if col in clean.columns:
            clean[col] = clean[col].fillna(0)

    if len(odds_files) > 1:
        last_year = clean["date"].dt.year.max()
        train = clean[clean["date"].dt.year < last_year]
        test  = clean[clean["date"].dt.year == last_year]
        print(f"\n📊 Strategy: train on {clean['date'].dt.year.min()}–{last_year-1}, test on {last_year}")
    else:
        split = int(len(clean) * train_frac)
        train = clean.iloc[:split]
        test  = clean.iloc[split:]

    print(f"   Train: {len(train)} games  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"   Test:  {len(test)} games   ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"   Features: {len(FEATURE_COLS)}")

    X_train = train[feat_cols].values
    joblib.dump(feat_cols, MODELS_DIR / "features.pkl")

    for bet_name, target, fname in [
        ("MONEYLINE  (home win)",         "home_win",       "moneyline"),
        ("RUN LINE   (home covers -1.5)", "covered_spread", "runline"),
    ]:
        y_train = train[target].values

        # Step 1: compare all models
        comp = compare_models(X_train, y_train, bet_name)

        # Step 2: build weighted ensemble
        print(f"\n  Ensemble for {bet_name}:")
        ensemble = build_ensemble(X_train, y_train, comp)
        joblib.dump(ensemble, MODELS_DIR / f"{fname}.pkl")
        joblib.dump(comp,     MODELS_DIR / f"{fname}_comparison.pkl")
        print(f"  ✅ Ensemble saved → {fname}.pkl\n")

    joblib.dump(train["date"].max(), MODELS_DIR / "train_cutoff.pkl")
    print(f"\n✅ All ensembles saved. Run --compare to see model comparison summary.\n")


# ══════════════════════════════════════════════
# 4. PREDICTIONS
# ══════════════════════════════════════════════

def american_to_decimal(ml):
    return (ml / 100) + 1 if ml > 0 else (100 / abs(ml)) + 1

def expected_value(prob, ml):
    dec = american_to_decimal(ml)
    return (prob * (dec - 1)) - (1 - prob)

def kelly_fraction(prob, ml, k=0.25):
    b = american_to_decimal(ml) - 1
    if b <= 0: return 0.0
    raw = (b * prob - (1 - prob)) / b
    # Tuned: quarter Kelly, capped at 3% per bet
    return max(0.0, min(raw * k, 0.03))

def book_prob(ml):
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def run_predictions(odds_file, date, bankroll=1000.0):
    print(f"\n⚾  MLB Betting ML — Predictions for {date}\n")
    try:
        ml_m  = joblib.load(MODELS_DIR / "moneyline.pkl")
        rl_m  = joblib.load(MODELS_DIR / "runline.pkl")
        feats = joblib.load(MODELS_DIR / "features.pkl")
    except FileNotFoundError:
        print("❌ No saved models. Run --train first.")
        return

    files = odds_file if isinstance(odds_file, list) else [odds_file]
    df    = load_multiple_seasons(files) if len(files) > 1 else load_sbr_file(files[0])
    print("Building features...")
    df    = build_features(df)
    for col, default in [
        ("home_pitcher_hand",0),("away_pitcher_hand",0),("same_hand_matchup",0),
        ("home_platoon_avg_vs_lhp",0),("home_platoon_avg_vs_rhp",0),
        ("away_platoon_avg_vs_lhp",0),("away_platoon_avg_vs_rhp",0),
        ("platoon_matchup_edge",0),("home_p_form_vs_season",0),
        ("away_p_form_vs_season",0),("pitcher_form_edge",0),
        ("home_days_rest",3),("away_days_rest",3),("rest_advantage",0),
        ("fatigue_edge",0),("home_home_streak",0),("away_home_streak",0),
        ("home_games_last_7",5),("away_games_last_7",5),
        ("home_avg_early_runs",0),("home_avg_late_runs",0),("home_late_clutch",0),
        ("away_avg_early_runs",0),("away_avg_late_runs",0),("away_late_clutch",0),
        ("late_inning_edge",0),("h2h_home_win_rate",0.5),("h2h_games_played",0),
    ]:
        if col in df.columns: df[col] = df[col].fillna(default)
    for col in feats:
        if col in df.columns: df[col] = df[col].fillna(0)
    games = df[df["date"] == pd.Timestamp(date)].dropna(subset=FEATURE_COLS)

    if games.empty:
        print(f"❌ No complete feature rows for {date}.")
        dates = df["date"].dt.date.unique()
        print(f"   Available range: {dates.min()} → {dates.max()}")
        return

    X        = games[feats].values
    ml_probs = ml_m.predict_proba(X)[:, 1]
    rl_probs = rl_m.predict_proba(X)[:, 1]

    print(f"\n{'─'*82}")
    print(f"  {'MATCHUP':<33} {'BET':<12} {'MODEL':>6} {'BOOK':>6} {'EDGE':>6} {'EV':>6}  ACTION")
    print(f"{'─'*82}")

    for i, (_, g) in enumerate(games.iterrows()):
        mu = f"{str(g['away_team'])[:14]} @ {str(g['home_team'])[:14]}"
        print(f"\n  {mu}  —  SP: {g.get('home_pitcher','?')} vs {g.get('away_pitcher','?')}")
        for label, prob, ml in [
            ("ML Home",  ml_probs[i],   g["home_ml"]),
            ("ML Away",  1-ml_probs[i], g["away_ml"]),
            ("Run Line", rl_probs[i],   g["run_line_odds"]),
            ]:
            if pd.isna(ml): continue
            bp   = book_prob(ml)
            edge = prob - bp
            ev   = expected_value(prob, ml)
            k    = kelly_fraction(prob, ml)
            act  = f"✅ BET ${bankroll*k:.0f}" if edge > 0.08 and ev > 0 else "❌ pass"
            print(f"    {label:<12} {prob:>5.1%} model  {bp:>5.1%} book  {edge:>+5.1%} edge  {ev:>+5.2f} EV   {act}")
    print(f"\n{'─'*82}")
    print(f"💰 Bankroll: ${bankroll:,.0f}  |  Kelly 25%, max 3% per bet  |  Min edge: 5%\n")


# ══════════════════════════════════════════════
# 5. BACKTEST
# ══════════════════════════════════════════════

def run_backtest(odds_file, bankroll=1000.0, min_edge=0.08):
    print("\n⚾  MLB Betting ML — Backtest\n")
    try:
        ml_m    = joblib.load(MODELS_DIR / "moneyline.pkl")
        rl_m    = joblib.load(MODELS_DIR / "runline.pkl")
        feats   = joblib.load(MODELS_DIR / "features.pkl")
        cutoff  = joblib.load(MODELS_DIR / "train_cutoff.pkl")
        adv_cfg     = joblib.load(MODELS_DIR / "advanced_stats.pkl")  if (MODELS_DIR / "advanced_stats.pkl").exists()  else {"use_advanced": False}
        matchup_cfg  = joblib.load(MODELS_DIR / "matchup_cfg.pkl")   if (MODELS_DIR / "matchup_cfg.pkl").exists()   else {"use_matchup": False}
        player_cfg   = joblib.load(MODELS_DIR / "player_cfg.pkl")    if (MODELS_DIR / "player_cfg.pkl").exists()    else {"use_players": False}
    except FileNotFoundError:
        print("❌ No saved models. Run --train first.")
        return

    files = odds_file if isinstance(odds_file, list) else [odds_file]
    df = load_multiple_seasons(files) if len(files) > 1 else load_sbr_file(files[0])
    print("Building features...")
    df = build_features(df)
    if adv_cfg.get("use_advanced"):
        print("  Attaching advanced stats...")
        df = attach_advanced_stats(df, adv_cfg["team_lookup"], adv_cfg["pitcher_lookup"])
    if matchup_cfg.get("use_matchup"):
        print("  Building matchup features...")
        df = build_all_matchup_features(df)
    if player_cfg.get("use_players"):
        print("  Building player intelligence features...")
        df = attach_player_intelligence(df, player_cfg["txn_lookup"])
    # Fill nulls in matchup/situational features with sensible defaults
    # rather than dropping entire rows — early season games won't have
    # H2H history, platoon splits, etc. yet but are still valid training rows
    fill_defaults = {
        # Pitcher stats — fill with league average when pitcher has <1 start
        # League avg: ~0.5 runs/inning proxy, 50% win rate, 1.0 std dev
        "home_p_era_proxy_L5":      0.5, "home_p_era_proxy_L10":    0.5,
        "away_p_era_proxy_L5":      0.5, "away_p_era_proxy_L10":    0.5,
        "home_p_win_rate_L10":      0.5, "away_p_win_rate_L10":     0.5,
        "home_p_runs_allowed_std_L10": 1.0, "away_p_runs_allowed_std_L10": 1.0,
        "pitcher_era_edge":         0.0, "pitcher_win_edge":         0.0,
        # Team rolling stats — fill with 0 for first few games of season
        "home_avg_scored_L10":      4.5, "home_avg_scored_L5":       4.5,
        "away_avg_scored_L10":      4.5, "away_avg_scored_L5":       4.5,
        "home_avg_allowed_L10":     4.5, "home_avg_allowed_L5":      4.5,
        "away_avg_allowed_L10":     4.5, "away_avg_allowed_L5":      4.5,
        "home_win_pct_L10":         0.5, "home_win_pct_L20":         0.5,
        "away_win_pct_L10":         0.5, "away_win_pct_L20":         0.5,
        "home_avg_run_diff_L10":    0.0, "away_avg_run_diff_L10":    0.0,
        "home_scored_std_L10":      2.5, "away_scored_std_L10":      2.5,
        "home_allowed_std_L10":     2.5, "away_allowed_std_L10":     2.5,
        "home_home_avg_scored_L10": 4.5, "home_home_avg_allowed_L10":4.5,
        "run_diff_edge":            0.0, "defense_edge":             0.0,
        "win_pct_edge":             0.0, "run_diff_edge_total":      0.0,
        "projected_total":          9.0, "total_vs_line":            0.0,
        "home_ml_move":             0.0, "away_ml_move":             0.0,
        # Handedness — default unknown to 0
        "home_pitcher_hand": 0, "away_pitcher_hand": 0, "same_hand_matchup": 0,
        # Platoon — fill with overall team avg (0 = no edge known)
        "home_platoon_avg_vs_lhp": 0, "home_platoon_avg_vs_rhp": 0,
        "home_platoon_platoon_adv": 0, "away_platoon_avg_vs_lhp": 0,
        "away_platoon_avg_vs_rhp": 0, "away_platoon_platoon_adv": 0,
        "platoon_matchup_edge": 0,
        # Pitcher form — 0 = no known deviation from average
        "home_p_form_vs_season": 0, "home_p_form_trend": 0,
        "away_p_form_vs_season": 0, "away_p_form_trend": 0,
        "pitcher_form_edge": 0,
        # Rest — league average is ~3 days
        "home_days_rest": 3, "away_days_rest": 3,
        "rest_advantage": 0, "fatigue_edge": 0,
        "home_home_streak": 0, "away_home_streak": 0,
        "home_games_last_7": 5, "away_games_last_7": 5,
        # Inning patterns — fill with 0 (unknown)
        "home_avg_early_runs": 0, "home_avg_late_runs": 0, "home_late_clutch": 0,
        "away_avg_early_runs": 0, "away_avg_late_runs": 0, "away_late_clutch": 0,
        "late_inning_edge": 0,
        # H2H — no prior games = 0.5 win rate (no edge)
        "h2h_home_win_rate": 0.5, "h2h_games_played": 0,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    # Only require core base features; fill extended features with 0
    clean = df.dropna(subset=FEATURE_COLS).sort_values("date")
    for col in feats:
        if col in clean.columns:
            clean[col] = clean[col].fillna(0)
    test  = clean[clean["date"] > cutoff].copy()

    print(f"\n   Training cutoff:  {cutoff.date()}")
    print(f"   Holdout period:   {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"   Holdout games:    {len(test)}\n")

    X    = test[feats].values
    ml_p = ml_m.predict_proba(X)[:, 1]
    rl_p = rl_m.predict_proba(X)[:, 1]

    bal = bankroll
    bets = wins = 0
    wagered = 0.0
    daily = {}

    for i, (_, g) in enumerate(test.iterrows()):
        day = str(g["date"].date())

        for prob, ml, outcome in [
            (ml_p[i],   g["home_ml"],        g["home_win"]),
            (1-ml_p[i], g["away_ml"],        1 - g["home_win"]),
            (rl_p[i],   g["run_line_odds"],   g["covered_spread"]),
        ]:
            if pd.isna(ml): continue
            bp   = book_prob(ml)
            edge = prob - bp
            ev   = expected_value(prob, ml)
            if edge < min_edge or ev <= 0: continue

            stake    = bal * kelly_fraction(prob, ml)
            wagered += stake
            bets    += 1
            profit   = stake * (american_to_decimal(ml) - 1) if outcome == 1 else -stake
            if outcome == 1: wins += 1
            bal += profit
            daily[day] = daily.get(day, 0) + profit

    win_rate = wins / bets * 100 if bets else 0
    roi      = (bal - bankroll) / bankroll * 100

    print(f"  Bets placed:      {bets}  (ML + Run Line only)")
    print(f"  Win rate:         {win_rate:.1f}%  (need 52.4% to profit at -110)")
    print(f"  ROI:              {roi:+.2f}%")
    print(f"  Starting balance: ${bankroll:,.2f}")
    print(f"  Final balance:    ${bal:,.2f}")
    print(f"  Net profit:       ${bal - bankroll:+,.2f}")

    if daily:
        profits = list(daily.values())
        winning_days = sum(1 for p in profits if p > 0)
        print(f"\n  Profitable days:  {winning_days}/{len(daily)}")
        print(f"  Best day:         ${max(profits):+.2f}")
        print(f"  Worst day:        ${min(profits):+.2f}")

    print()


# ══════════════════════════════════════════════
# 5b. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════

def run_walkforward(odds_files, bankroll=1000.0, min_edge=0.08,
                    min_train_games=15, use_matchup=False,
                    use_advanced=False, use_players=False,
                    mode="B"):
    """
    Season-aware walk-forward validation.

    mode: "A" | "B" | "C" | "all"
      A   = current season only (fresh start each year)
      B   = all seasons weighted by recency
      C   = base 42-feature honest baseline
      all = run all three and compare side by side
    """
    print("\n⚾  MLB Betting ML — Walk-Forward Validation\n")

    if isinstance(odds_files, str):
        odds_files = [odds_files]

    df = load_multiple_seasons(odds_files) if len(odds_files) > 1 else load_sbr_file(odds_files[0])
    print("Building base features...")
    df = build_features(df)

    # Apply optional feature layers
    if use_matchup:
        print("  Building matchup features...")
        df = build_all_matchup_features(df)

    if use_players:
        player_cfg = (joblib.load(MODELS_DIR / "player_cfg.pkl")
                      if (MODELS_DIR / "player_cfg.pkl").exists()
                      else {"use_players": False})
        if player_cfg.get("use_players"):
            print("  Attaching player intelligence...")
            df = attach_player_intelligence(df, player_cfg["txn_lookup"])

    # Fill nulls
    fill_defaults = {
        "home_p_era_proxy_L5": 0.5,  "home_p_era_proxy_L10": 0.5,
        "away_p_era_proxy_L5": 0.5,  "away_p_era_proxy_L10": 0.5,
        "home_p_win_rate_L10": 0.5,  "away_p_win_rate_L10":  0.5,
        "home_p_runs_allowed_std_L10": 1.0,
        "away_p_runs_allowed_std_L10": 1.0,
        "pitcher_era_edge": 0.0,     "pitcher_win_edge": 0.0,
        "home_ml_move": 0.0,         "away_ml_move": 0.0,
        "home_pitcher_hand": 0,      "away_pitcher_hand": 0,
        "same_hand_matchup": 0,      "platoon_matchup_edge": 0,
        "home_p_form_vs_season": 0,  "away_p_form_vs_season": 0,
        "pitcher_form_edge": 0,
        "home_days_rest": 3,         "away_days_rest": 3,
        "rest_advantage": 0,         "fatigue_edge": 0,
        "home_home_streak": 0,       "away_home_streak": 0,
        "home_games_last_7": 5,      "away_games_last_7": 5,
        "home_avg_early_runs": 0,    "home_avg_late_runs": 0,
        "away_avg_early_runs": 0,    "away_avg_late_runs": 0,
        "late_inning_edge": 0,
        "h2h_home_win_rate": 0.5,    "h2h_games_played": 0,
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)

    # Choose feature set
    if use_matchup and use_players:
        feat_cols = FEATURE_COLS_INTEL
    elif use_matchup:
        feat_cols = FEATURE_COLS_MATCHUP
    else:
        feat_cols = FEATURE_COLS   # base 42 by default

    # Only keep rows where base features are available
    clean = df.dropna(subset=FEATURE_COLS).sort_values("date").reset_index(drop=True)

    print(f"\n   Games:    {len(clean)}")
    print(f"   Features: {len(feat_cols)}")
    print(f"   Seasons:  {sorted(clean['date'].dt.year.unique().tolist())}")
    print(f"   Mode:     {mode.upper()}")

    m = mode.upper()
    if m not in ("A", "B", "C", "ALL"):
        print(f"❌ Unknown mode: {mode}. Use A, B, C, or all.")
        return

    if m == "ALL":
        all_results = run_all_modes(
            clean, feat_cols,
            min_train_games=min_train_games,
            min_edge=min_edge,
            bankroll=bankroll,
        )
        best = max(all_results, key=lambda k: all_results[k]["roi"])
        save_walk_forward_results(all_results[best], f"walk_forward_best_mode{best}.csv")
        print_walk_forward_report(all_results[best])
    else:
        results = run_walk_forward(
            clean, feat_cols,
            min_train_games=min_train_games,
            min_edge=min_edge,
            bankroll=bankroll,
            mode=m,
        )
        print_walk_forward_report(results)
        save_walk_forward_results(results, f"walk_forward_mode{m}.csv")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Betting ML System v4")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--compare",     action="store_true", help="Show model comparison table")
    parser.add_argument("--walkforward", action="store_true", help="Walk-forward: season-aware rolling backtest")
    parser.add_argument("--mode",        type=str, default="B",   help="Walk-forward mode: A, B, C, or all (default: B)")
    parser.add_argument("--advanced", action="store_true", help="Fetch real pitcher/team stats from FanGraphs")
    parser.add_argument("--matchup",  action="store_true", help="Add handedness, platoon, rest, H2H, batter stats")
    parser.add_argument("--players",  action="store_true", help="Add trade, slump, position change features (MLB Stats API)")
    parser.add_argument("--odds",     type=str, nargs="+", default=["mlb-odds-2021.xlsx"],
                        help="One or more SBR odds files")
    parser.add_argument("--date",     type=str, default="2021-09-01")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()

    if   args.train:       run_training(args.odds, use_advanced=args.advanced, use_matchup=args.matchup, use_players=args.players)
    elif args.predict:     run_predictions(args.odds, args.date, args.bankroll)
    elif args.backtest:    run_backtest(args.odds, args.bankroll)
    elif args.walkforward: run_walkforward(
        args.odds, args.bankroll,
        min_train_games=15,
        use_matchup=args.matchup,
        use_players=args.players,
        mode=args.mode,
    )
    elif args.compare:     run_compare()
    else:
        print("Usage:")
        print("Usage:")
        print("  --train       --odds FILE [FILES...]               Train ensemble models")
        print("  --train       --odds FILE [FILES...] --matchup     + platoon/rest/H2H/batter features")
        print("  --train       --odds FILE [FILES...] --advanced    + FanGraphs pitcher/team stats")
        print("  --walkforward --odds FILE [FILES...]               Game-by-game rolling backtest")
        print("  --walkforward --odds FILE [FILES...] --matchup     With matchup features")
        print("  --backtest    --odds FILE [FILES...]               Holdout backtest on last season")
        print("  --compare                                          Show model comparison table")
        print("  --predict     --odds FILE --date YYYY-MM-DD        Predict a specific date")
