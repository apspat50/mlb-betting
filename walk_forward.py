"""
walk_forward.py
---------------
Season-aware walk-forward validation with 3 modes:

  Mode A — current_only:
    Each season trains ONLY on that season's games so far.
    Pure fresh start every April. No stale cross-season patterns.

  Mode B — weighted_all:
    All prior seasons included but recent seasons weighted higher.
    current season = 1.0, prior = 0.5, 2 back = 0.25, 3+ = 0.125
    Balances continuity with recency.

  Mode C — base_42:
    Same as Mode A (current season only) but FORCES the 42-feature
    base model regardless of what features were trained with.
    This is the cleanest honest baseline.

All modes:
  - Predict starting after min_train_games within the current season
  - Retrain before every prediction day
  - Results cached per mode — re-runs are instant
  - Results broken down by season, bet type, and month
"""

import hashlib
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_DIR  = Path("cache")
MODELS_DIR = Path("saved_models")
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Hardcoded base 42 features for Mode C ──
BASE_42_COLS = [
    "home_avg_scored_L10", "home_avg_scored_L5",
    "away_avg_scored_L10", "away_avg_scored_L5",
    "home_avg_allowed_L10", "home_avg_allowed_L5",
    "away_avg_allowed_L10", "away_avg_allowed_L5",
    "home_win_pct_L20", "home_win_pct_L10",
    "away_win_pct_L20", "away_win_pct_L10",
    "home_scored_std_L10", "home_allowed_std_L10",
    "away_scored_std_L10", "away_allowed_std_L10",
    "home_avg_run_diff_L10", "away_avg_run_diff_L10",
    "home_home_avg_scored_L10", "home_home_avg_allowed_L10",
    "home_p_era_proxy_L5", "home_p_era_proxy_L10",
    "home_p_win_rate_L10", "home_p_runs_allowed_std_L10",
    "away_p_era_proxy_L5", "away_p_era_proxy_L10",
    "away_p_win_rate_L10", "away_p_runs_allowed_std_L10",
    "run_diff_edge", "defense_edge", "win_pct_edge",
    "run_diff_edge_total", "pitcher_era_edge", "pitcher_win_edge",
    "projected_total", "total_vs_line", "total_line",
    "home_true_prob", "away_true_prob", "ml_edge",
    "home_ml_move", "away_ml_move",
]


# ══════════════════════════════════════════════
# FAST PER-DAY MODEL (no CV, trains thousands of times)
# ══════════════════════════════════════════════

def build_window_models(X_train, y_ml, y_rl, sample_weights=None):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    models = {}
    for label, y in [("ml", y_ml), ("rl", y_rl)]:
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=300,
                                       random_state=42, solver="lbfgs")),
        ])
        rf = RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=20,
            max_features="sqrt", random_state=42, n_jobs=1,
        )
        xgb = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.07,
            subsample=0.8, min_child_weight=15,
            eval_metric="logloss", random_state=42, verbosity=0,
        )

        if sample_weights is not None:
            # XGBoost and RF support sample weights natively
            xgb.fit(X_train, y, sample_weight=sample_weights, verbose=False)
            rf.fit(X_train, y, sample_weight=sample_weights)
            # Logistic regression via pipeline — pass to clf step
            lr.fit(X_train, y,
                   **{"clf__sample_weight": sample_weights})
        else:
            xgb.fit(X_train, y, verbose=False)
            rf.fit(X_train, y)
            lr.fit(X_train, y)

        models[label] = (lr, rf, xgb)
    return models


def predict_probs(models, X_test):
    out = {}
    for label, (lr, rf, xgb) in models.items():
        p = (lr.predict_proba(X_test)[:, 1] +
             rf.predict_proba(X_test)[:, 1] +
             xgb.predict_proba(X_test)[:, 1]) / 3.0
        out[label] = p
    return out


# ══════════════════════════════════════════════
# SEASON WEIGHTING (Mode B)
# ══════════════════════════════════════════════

def get_season_weights(train_df, current_season):
    """
    Assign a sample weight to each training row based on how
    recent its season is relative to the current season.
    current = 1.0, 1 back = 0.5, 2 back = 0.25, 3+ = 0.125
    """
    weights = np.ones(len(train_df))
    seasons = train_df["date"].dt.year.values
    for i, s in enumerate(seasons):
        gap = current_season - s
        if gap == 0:
            weights[i] = 1.000
        elif gap == 1:
            weights[i] = 0.500
        elif gap == 2:
            weights[i] = 0.250
        else:
            weights[i] = 0.125
    return weights


# ══════════════════════════════════════════════
# BETTING HELPERS
# ══════════════════════════════════════════════

def american_to_decimal(ml):
    if ml == 0: return 2.0  # treat 0 as even money
    return (ml / 100) + 1 if ml > 0 else (100 / abs(ml)) + 1

def book_prob(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

def expected_value(prob, ml):
    dec = american_to_decimal(ml)
    return (prob * (dec - 1)) - (1 - prob)

def kelly_fraction(prob, ml, k=0.25):
    dec  = american_to_decimal(ml)
    b    = dec - 1
    q    = 1 - prob
    full = (b * prob - q) / b
    return min(max(0.0, full * k), 0.03)


# ══════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════

def _cache_key(df, feat_cols, min_train, mode):
    sig = (f"{len(df)}_{df['date'].min()}_{df['date'].max()}"
           f"_{len(feat_cols)}_{min_train}_{mode}")
    return hashlib.md5(sig.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════
# CORE WALK-FORWARD ENGINE
# ══════════════════════════════════════════════

def _run_season(season_df, feat_cols, all_prior_df,
                mode, min_train, min_edge, bankroll_ref,
                current_season, max_lookback=3):
    """
    Run walk-forward for a single season.
    Returns list of bet dicts and running balance delta.

    season_df:      games for this season only
    all_prior_df:   all games from prior seasons (used in mode B)
    bankroll_ref:   list with one float (mutable reference for balance)
    """
    season_df = season_df.sort_values("date").reset_index(drop=True)
    all_dates  = sorted(season_df["date"].dt.date.unique())

    # Find first prediction date (after min_train games into this season)
    if len(season_df) < min_train + 1:
        return []

    warmup_end    = season_df.iloc[min_train - 1]["date"].date()
    predict_dates = [d for d in all_dates if d > warmup_end]

    bets = []

    for pred_date in predict_dates:
        # ── Build training set based on mode ──
        season_train = season_df[season_df["date"].dt.date < pred_date]

        if mode == "A" or mode == "C":
            # Current season only
            train_df = season_train
            weights  = None

        elif mode == "B":
            # All prior seasons weighted by recency, capped at max_lookback seasons
            cutoff_season = current_season - max_lookback
            prior_capped  = all_prior_df[
                all_prior_df["date"].dt.year > cutoff_season
            ]
            train_df = pd.concat([prior_capped, season_train], ignore_index=True)
            weights  = get_season_weights(train_df, current_season)

        if len(train_df) < min_train:
            continue

        test_df = season_df[season_df["date"].dt.date == pred_date]
        if len(test_df) == 0:
            continue

        X_tr = train_df[feat_cols].fillna(0).values
        X_te = test_df[feat_cols].fillna(0).values
        y_ml = train_df["home_win"].values
        y_rl = train_df["covered_spread"].values

        try:
            models = build_window_models(X_tr, y_ml, y_rl, weights)
            preds  = predict_probs(models, X_te)
        except Exception:
            continue

        ml_probs = preds["ml"]
        rl_probs = preds["rl"]
        day_str  = str(pred_date)

        for j, (_, g) in enumerate(test_df.iterrows()):
            for prob, ml, outcome, btype in [
                (ml_probs[j],   g["home_ml"],       g["home_win"],       "ML_home"),
                (rl_probs[j],   g["run_line_odds"],  g["covered_spread"], "RL_home"),
                # ML_away disabled — model consistently loses on away moneylines
                # RL_home skipped automatically if run_line_odds is null (pre-2014 data)
            ]:
                if pd.isna(ml): continue
                bp   = book_prob(float(ml))
                edge = prob - bp
                ev   = expected_value(prob, float(ml))
                if edge < min_edge or ev <= 0: continue

                bal    = bankroll_ref[0]
                stake  = bal * kelly_fraction(prob, float(ml))
                won    = int(outcome) == 1
                profit = stake * (american_to_decimal(float(ml)) - 1) if won else -stake
                bankroll_ref[0] += profit

                bets.append({
                    "date":       day_str,
                    "season":     current_season,
                    "home_team":  g.get("home_team", ""),
                    "away_team":  g.get("away_team", ""),
                    "bet_type":   btype,
                    "prob":       round(prob, 4),
                    "book_prob":  round(bp, 4),
                    "edge":       round(edge, 4),
                    "ml":         float(ml),
                    "stake":      round(stake, 2),
                    "profit":     round(profit, 2),
                    "won":        won,
                    "balance":    round(bankroll_ref[0], 2),
                    "train_size": len(train_df),
                })

    return bets


def run_walk_forward(df, feat_cols, min_train_games=15,
                     min_edge=0.08, bankroll=1000.0,
                     mode="C", use_cache=True,
                     skip_first_season=True):
    """
    Season-aware walk-forward validation.

    mode:
      "A" — current season only (fresh start each year)
      "B" — all seasons weighted by recency
      "C" — base 42 features, current season only (honest baseline)

    Returns results dict with all bet records and summary stats.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Mode C forces base 42 feature set
    if mode == "C":
        available = [c for c in BASE_42_COLS if c in df.columns]
        feat_cols  = available
        print(f"  Mode C: forcing base {len(feat_cols)}-feature set")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cache check
    ck = _cache_key(df, feat_cols, min_train_games, mode + str(skip_first_season))
    cache_file = CACHE_DIR / f"wf_{mode}_{ck}.pkl"
    if use_cache and cache_file.exists():
        print(f"  📂 Walk-forward Mode {mode} loaded from cache")
        return joblib.load(cache_file)

    seasons = sorted(df["date"].dt.year.unique())
    first_season = seasons[0]

    mode_labels = {"A": "Current season only",
                   "B": "All seasons weighted",
                   "C": f"Base {len(feat_cols)}-feature (honest baseline)"}

    print(f"\n  Mode {mode}: {mode_labels[mode]}")
    print(f"  Seasons: {seasons}")
    if skip_first_season:
        print(f"  ⚠️  {first_season} used as training-only (no bets) — too little history")
    print(f"  Min warmup: {min_train_games} games per season")
    print(f"  Features: {len(feat_cols)}")
    print(f"  ⏳ Estimated time: 10-30 min. Cached after first run.\n")

    bankroll_ref = [bankroll]
    all_bets     = []

    for i, season in enumerate(seasons):
        season_df   = df[df["date"].dt.year == season].copy()
        prior_df    = df[df["date"].dt.year < season].copy()

        n_games = len(season_df)
        if n_games < min_train_games + 1:
            print(f"  Season {season}: skipped (only {n_games} games)")
            continue

        # Skip first season for betting — use it as training data only
        if skip_first_season and season == first_season:
            print(f"  Season {season}: {n_games} games — training only (no bets placed)")
            continue

        print(f"  Season {season}: {n_games} games — predicting...")
        season_bets = _run_season(
            season_df, feat_cols, prior_df,
            mode, min_train_games, min_edge,
            bankroll_ref, season, max_lookback=3,
        )

        all_bets.extend(season_bets)
        season_wins   = sum(1 for b in season_bets if b["won"])
        season_wr     = season_wins / len(season_bets) * 100 if season_bets else 0
        season_profit = sum(b["profit"] for b in season_bets)
        print(f"  Season {season} done:  {len(season_bets)} bets  "
              f"WR: {season_wr:.1f}%  "
              f"P/L: ${season_profit:+.2f}  "
              f"Balance: ${bankroll_ref[0]:,.2f}")

    total_bets = len(all_bets)
    wins       = sum(1 for b in all_bets if b["won"])
    final_bal  = bankroll_ref[0]

    results = {
        "mode":       mode,
        "bets":       all_bets,
        "total_bets": total_bets,
        "wins":       wins,
        "win_rate":   wins / total_bets * 100 if total_bets else 0,
        "roi":        (final_bal - bankroll) / bankroll * 100,
        "start_bal":  bankroll,
        "end_bal":    final_bal,
        "net_profit": final_bal - bankroll,
        "feat_cols":  feat_cols,
        "min_train":  min_train_games,
        "seasons":    seasons,
    }

    joblib.dump(results, cache_file)
    print(f"\n  ✅ Results cached → {cache_file.name}")
    return results


# ══════════════════════════════════════════════
# RUN ALL THREE MODES AND COMPARE
# ══════════════════════════════════════════════

def run_all_modes(df, feat_cols, min_train_games=15,
                  min_edge=0.08, bankroll=1000.0,
                  skip_first_season=True):
    results = {}
    for mode in ["A", "B", "C"]:
        print(f"\n{'═'*60}")
        print(f"  Running Mode {mode}...")
        print(f"{'═'*60}")
        results[mode] = run_walk_forward(
            df, feat_cols,
            min_train_games=min_train_games,
            min_edge=min_edge,
            bankroll=bankroll,
            mode=mode,
            skip_first_season=skip_first_season,
        )
    print_comparison(results)
    return results


# ══════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════

def print_walk_forward_report(results):
    mode  = results.get("mode", "?")
    bets  = results["bets"]
    mode_labels = {
        "A": "Current season only",
        "B": "All seasons weighted by recency",
        "C": f"Base {len(results['feat_cols'])}-feature honest baseline",
    }

    print(f"\n{'═'*62}")
    print(f"  WALK-FORWARD  Mode {mode}: {mode_labels.get(mode,'')}")
    print(f"  {results['min_train']}-game warmup per season  |  "
          f"Seasons: {results['seasons']}")
    print(f"{'═'*62}")
    print(f"  Total bets:       {results['total_bets']}")
    print(f"  Win rate:         {results['win_rate']:.1f}%  "
          f"(need 52.4% to profit)")
    print(f"  ROI:              {results['roi']:+.2f}%")
    print(f"  Starting balance: ${results['start_bal']:,.2f}")
    print(f"  Final balance:    ${results['end_bal']:,.2f}")
    print(f"  Net profit:       ${results['net_profit']:+,.2f}")

    if bets:
        df_b = pd.DataFrame(bets)

        # By season
        print(f"\n  ── By season ──")
        for yr, grp in df_b.groupby("season"):
            wr  = grp["won"].mean() * 100
            roi = (grp["profit"].sum() / grp["stake"].sum() * 100
                   if grp["stake"].sum() > 0 else 0)
            print(f"  {yr}   {len(grp):>5} bets   "
                  f"WR: {wr:.1f}%   ROI: {roi:+.1f}%   "
                  f"P/L: ${grp['profit'].sum():+.2f}")

        # By bet type
        print(f"\n  ── By bet type ──")
        for bt, grp in df_b.groupby("bet_type"):
            wr  = grp["won"].mean() * 100
            roi = (grp["profit"].sum() / grp["stake"].sum() * 100
                   if grp["stake"].sum() > 0 else 0)
            print(f"  {bt:<12}  {len(grp):>5} bets   "
                  f"WR: {wr:.1f}%   ROI: {roi:+.1f}%")

        # Daily summary
        daily = df_b.groupby("date")["profit"].sum()
        print(f"\n  Profitable days:  "
              f"{(daily > 0).sum()}/{len(daily)}")
        print(f"  Best day:         ${daily.max():+.2f}")
        print(f"  Worst day:        ${daily.min():+.2f}")

        # Monthly
        df_b["month"] = pd.to_datetime(df_b["date"]).dt.to_period("M")
        monthly = df_b.groupby("month")["profit"].sum()
        if len(monthly) > 1:
            print(f"  Best month:       "
                  f"{monthly.idxmax()}  (${monthly.max():+.2f})")
            print(f"  Worst month:      "
                  f"{monthly.idxmin()}  (${monthly.min():+.2f})")

    print(f"{'═'*62}\n")


def print_comparison(results: dict):
    """Side-by-side comparison of all three modes."""
    mode_labels = {
        "A": "Current season only    ",
        "B": "Weighted all seasons   ",
        "C": "Base 42 features (C)   ",
    }

    print(f"\n{'═'*72}")
    print(f"  WALK-FORWARD MODE COMPARISON")
    print(f"{'─'*72}")
    print(f"  {'Mode':<6} {'Description':<26} {'Bets':>6} "
          f"{'WR%':>7} {'ROI':>8} {'Net P/L':>10}")
    print(f"{'─'*72}")

    best_roi = max(r["roi"] for r in results.values())

    for mode, r in results.items():
        star = " ⭐" if r["roi"] == best_roi else "   "
        print(f"  {mode:<6} {mode_labels.get(mode,''):<26} "
              f"{r['total_bets']:>6} "
              f"{r['win_rate']:>6.1f}% "
              f"{r['roi']:>+7.2f}% "
              f"${r['net_profit']:>+9.2f}"
              f"{star}")

    print(f"{'─'*72}")
    winner = max(results, key=lambda k: results[k]["roi"])
    print(f"  Best mode: {winner} — {mode_labels.get(winner,'').strip()}")
    print(f"{'═'*72}\n")


def save_walk_forward_results(results, path="walk_forward_results.csv"):
    if not results.get("bets"):
        print("  No bets to save.")
        return
    df = pd.DataFrame(results["bets"])
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df)} bet records → {path}")
