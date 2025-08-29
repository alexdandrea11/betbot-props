# BetBot Props — Live NFL Player Props + Trainable Regression (probability-first)
# - LIVE odds & all matchups (The Odds API v4)
# - NFL Week -> Matchup selector (robust, with schedule fallback)
# - Uses /models regression if available (μ, σ). Otherwise, heuristics.
# - Ranks by hit probability (best side Over/Under)
# - Kelly staking, explanation panel, line-move radar

import os, json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import requests
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import norm, multivariate_normal
from joblib import load as joblib_load
import nfl_data_py as nfl  # schedules for week mapping

# --------------------------- Config & Paths ---------------------------
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
PICKS_LOG = DATA_DIR / "picks_log.csv"

UNDERS_BIAS_BPS = 200        # -2% to Over probs (optional)
MAX_KELLY_DEFAULT = 0.10
EXPOSURE_CAP_DEFAULT = 2.0

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

SPORT_KEY = "americanfootball_nfl"
REGIONS = "us"
ODDS_FMT = "american"

# Markets we’ll fetch
MARKET_KEYS = [
    "player_receptions",
    "player_reception_yds",
    "player_rush_yds",
    "player_pass_yds",
]
MARKET_LABEL = {
    "player_receptions": "Receptions",
    "player_reception_yds": "Rec Yds",
    "player_rush_yds": "Rush Yds",
    "player_pass_yds": "Pass Yds",
}
LABEL_TO_MK = {v:k for k,v in MARKET_LABEL.items()}

BOOK_LABEL = {
    "draftkings": "DK",
    "fanduel": "FD",
    "betmgm": "MGM",
    "pointsbetus": "PB",
    "caesars": "CZR",
}
BOOK_INV = {v:k for k,v in BOOK_LABEL.items()}

# --------------------------- Pricing helpers --------------------------
def american_to_prob(odds: int) -> float:
    return 100/(odds+100) if odds>0 else (-odds)/(-odds+100)

def prob_to_american(p: float) -> float:
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return np.nan
    return int(round(-100*p/(1-p))) if p>=0.5 else int(round(100*(1-p)/p))

def fair_price(p_over: float) -> float:
    return prob_to_american(p_over)

def decimal_payout(odds: int) -> float:
    return 1 + (odds/100.0) if odds>0 else 1 + (100/abs(odds))

def ev_from_prob(p_win: float, odds: int) -> float:
    payout_minus_stake = (odds/100.0) if odds>0 else (100/abs(odds))
    return p_win * payout_minus_stake - (1 - p_win)

# --------------------------- Risk & correlation -----------------------
def confidence_score(data_freshness: float, role_stability: float, line_dispersion: float, sample_depth: float) -> float:
    w = np.array([0.25,0.35,0.20,0.20])
    v = np.array([data_freshness, role_stability, line_dispersion, sample_depth])
    return float(np.clip((w*v).sum(), 0.0, 1.0))

def kelly_fraction(p_win: float, odds: int, max_kelly: float = MAX_KELLY_DEFAULT) -> float:
    b = decimal_payout(odds) - 1
    f_star = (b*p_win - (1-p_win)) / b
    return float(np.clip(f_star, 0.0, max_kelly))

_DEF_CORR = {
    ("same_player","rec_vs_recyds"): 0.60,
    ("teammates","recv_targets_share"): -0.25,
    ("qb_wr","passyds_vs_wr"): 0.30,
}
def heuristic_pair_corr(a: Dict, b: Dict) -> float:
    if a['player'] == b['player']:
        if {a['market'], b['market']} == {"Receptions","Rec Yds"}:
            return _DEF_CORR[("same_player","rec_vs_recyds")]
        return 0.3
    if a['team'] == b['team']:
        if a['role'] in {"WR","TE"} and b['role'] in {"WR","TE"}:
            return _DEF_CORR[("teammates","recv_targets_share")]
        if {a['role'], b['role']} == {"QB","WR"} and ("Pass Yds" in {a['market'], b['market']}):
            return _DEF_CORR[("qb_wr","passyds_vs_wr")]
    return 0.0
def correlation_matrix(legs: List[Dict]) -> np.ndarray:
    n = len(legs); R = np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            R[i,j] = R[j,i] = heuristic_pair_corr(legs[i], legs[j])
    return R
def joint_win_probability(p_list: List[float], R: np.ndarray) -> float:
    z = norm.ppf(p_list)
    return float(multivariate_normal.cdf(z, mean=np.zeros(len(p_list)), cov=R))

# --------------------------- CSV priors (heuristics) ------------------
def load_games_csv() -> pd.DataFrame:
    p = DATA_DIR/"sample_games.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=[
        "game_id","game","home","away","total","spread_home","week","roof","wind_kts"
    ])

def load_usage_csv() -> pd.DataFrame:
    p = DATA_DIR/"sample_usage.csv"
    if not p.exists():
        return pd.DataFrame(columns=[
            "player","team","role","targets_share","catch_rate","aDot","carries_share",
            "yards_per_carry","snap_share","recent_weight","sample_games","week"
        ])
    df = pd.read_csv(p)
    df['targets_share']   = pd.to_numeric(df.get('targets_share', np.nan), errors='coerce').clip(0.05, 0.40)
    df['catch_rate']      = pd.to_numeric(df.get('catch_rate', np.nan), errors='coerce').clip(0.45, 0.85)
    df['carries_share']   = pd.to_numeric(df.get('carries_share', np.nan), errors='coerce').clip(0.05, 0.80)
    df['yards_per_carry'] = pd.to_numeric(df.get('yards_per_carry', np.nan), errors='coerce').clip(3.2, 5.8)
    df['snap_share']      = pd.to_numeric(df.get('snap_share', np.nan), errors='coerce').fillna(0.6)
    df['recent_weight']   = pd.to_numeric(df.get('recent_weight', 0.6), errors='coerce').fillna(0.6)
    df['sample_games']    = pd.to_numeric(df.get('sample_games', 6), errors='coerce').fillna(6)
    df['aDot']            = pd.to_numeric(df.get('aDot', 9.0), errors='coerce').fillna(9.0)
    return df

_DEF_BASE_PLAYS = 126; _PLAYS_K_TOTAL = 0.8; _TOTAL_BASE = 44.0
_PASS_SLOPE = -0.8/7.0
POS_PRIORS = {'WR': 0.21, 'TE': 0.18, 'RB': 0.12}

def build_game_context(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=[
            "team","plays","pass_rate","rush_rate","game_id","game","week","weather_note"
        ])
    df = games_df.copy()
    df['combined_plays'] = _DEF_BASE_PLAYS + _PLAYS_K_TOTAL * (df['total'] - _TOTAL_BASE)
    df['home_pass_rate'] = 0.56 + _PASS_SLOPE * df['spread_home']
    df['away_pass_rate'] = 1 - df['home_pass_rate']
    df['home_rush_rate'] = 1 - df['home_pass_rate']
    df['away_rush_rate'] = 1 - df['away_pass_rate']
    fav_tilt = (df['spread_home'] < 0).astype(float) * 0.02
    df['home_plays'] = df['combined_plays'] * (0.5 + fav_tilt)
    df['away_plays'] = df['combined_plays'] - df['home_plays']
    def weather_note(g):
        roof = str(g.get('roof','open')).lower()
        wind = float(g.get('wind_kts', 0))
        if roof in ('closed','dome'): return "dome/closed roof"
        if wind >= 15: return f"wind {wind:.0f} kts"
        return "no major wind"
    rows = []
    for _, g in df.iterrows():
        rows += [
            {'team': g['home'],'plays': g['home_plays'],'pass_rate': g['home_pass_rate'],'rush_rate': g['home_rush_rate'],
             'game_id': g['game_id'],'game': g['game'],'week': g['week'],'weather_note': weather_note(g)},
            {'team': g['away'],'plays': g['away_plays'],'pass_rate': g['away_pass_rate'],'rush_rate': g['away_rush_rate'],
             'game_id': g['game_id'],'game': g['game'],'week': g['week'],'weather_note': weather_note(g)}
        ]
    return pd.DataFrame(rows)

def apply_usage_priors(usage_df: pd.DataFrame) -> pd.DataFrame:
    if usage_df.empty: return usage_df
    df = usage_df.copy()
    def blend(row):
        prior = POS_PRIORS.get(row.get('role',''), 0.15)
        w = float(row.get('recent_weight', 0.6))
        return w*float(row.get('targets_share',0.0)) + (1-w)*prior
    df['targets_share_adj'] = df.apply(blend, axis=1)
    df['carries_share_adj'] = df.get('carries_share', pd.Series(0.2, index=df.index))
    return df

def project_players_from_csv(week: int = 1) -> pd.DataFrame:
    usage = apply_usage_priors(load_usage_csv())
    games = build_game_context(load_games_csv())
    if usage.empty or games.empty:
        return pd.DataFrame(columns=[
            "player","team","role","targets_mean","catch_rate","receptions_mu",
            "rec_yards_mu","carries_mean","yards_per_carry","pass_rate","rush_rate",
            "snap_share","sample_games","game","week"
        ])
    ctx = games.copy()
    df = usage.merge(ctx, on='team', how='inner')
    df['team_pass_attempts'] = df['plays'] * df['pass_rate'] * 0.95
    df['team_rush_attempts'] = df['plays'] * df['rush_rate']
    df['targets_mean'] = df['team_pass_attempts'] * df['targets_share_adj']
    df['carries_mean'] = df['team_rush_attempts'] * df['carries_share_adj']
    df['receptions_mu'] = df['targets_mean'] * df['catch_rate']
    df['rec_yards_mu']  = df['receptions_mu'] * df['aDot'].clip(6,12) * 0.6
    df['rush_yards_mu'] = df['carries_mean'] * df['yards_per_carry']
    return df

# --------------------------- LIVE odds fetching -----------------------
def _odds_get(path: str, params: dict) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("Missing ODDS_API_KEY")
    url = f"https://api.the-odds-api.com{path}"
    q = {"apiKey": ODDS_API_KEY, **params}
    r = requests.get(url, params=q, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:250]}")
    return r.json()

@st.cache_data(ttl=60)
def fetch_events_live() -> pd.DataFrame:
    js = _odds_get(f"/v4/sports/{SPORT_KEY}/events", {"dateFormat": "iso"})
    rows = []
    for e in js:
        rows.append({
            "event_id": e.get("id"),
            "commence_time": e.get("commence_time"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "game": f"{e.get('away_team')} @ {e.get('home_team')}",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_dt"] = pd.to_datetime(df["commence_time"], errors="coerce")
        df["commence_date"] = df["commence_dt"].dt.date
        df["season_guess"] = df["commence_dt"].dt.year.where(df["commence_dt"].dt.month >= 3,
                                                             df["commence_dt"].dt.year - 1)
    return df

@st.cache_data(ttl=1800)
def build_week_windows(season: int) -> pd.DataFrame:
    # Robust to 'gameday' vs 'game_date'
    try:
        sch = nfl.import_schedules([season])
        date_col = "gameday" if "gameday" in sch.columns else ("game_date" if "game_date" in sch.columns else None)
        if date_col is None or "week" not in sch.columns or "season" not in sch.columns:
            return pd.DataFrame(columns=["season","week","week_start_date","week_end_date"])
        sch = sch[["season","week", date_col]].copy()
        sch[date_col] = pd.to_datetime(sch[date_col], errors="coerce")
        sch = sch.dropna(subset=[date_col, "week"])
        sch["week"] = sch["week"].astype(int)
        grp = sch.groupby(["season","week"], as_index=False)[date_col].agg(["min","max"]).reset_index()
        grp = grp.rename(columns={"min":"week_start", "max":"week_end"})
        grp["week_start_date"] = pd.to_datetime(grp["week_start"]).dt.date
        grp["week_end_date"]   = pd.to_datetime(grp["week_end"]).dt.date
        return grp[["season","week","week_start_date","week_end_date"]]
    except Exception:
        return pd.DataFrame(columns=["season","week","week_start_date","week_end_date"])

def simple_week_tagging(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty or "commence_time" not in events_df.columns:
        return events_df.assign(week=np.nan)
    out = events_df.copy()
    out["commence_dt"] = pd.to_datetime(out["commence_time"], errors="coerce")
    out["commence_date"] = out["commence_dt"].dt.floor("D")
    if out["commence_dt"].isna().all():
        return out.assign(week=np.nan)
    min_date = out["commence_date"].min()
    offset = int((min_date.weekday() - 3) % 7)  # Thu=3
    wk0 = pd.to_datetime(min_date) - pd.Timedelta(days=offset)
    out["week"] = ((pd.to_datetime(out["commence_date"]) - wk0).dt.days // 7) + 1
    return out

def map_events_to_weeks(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    if "commence_dt" not in events_df.columns:
        events_df = events_df.copy()
        events_df["commence_dt"] = pd.to_datetime(events_df["commence_time"], errors="coerce")
        events_df["commence_date"] = events_df["commence_dt"].dt.date
    try:
        season_guess = events_df["commence_dt"].dt.year.where(events_df["commence_dt"].dt.month >= 3,
                                                              events_df["commence_dt"].dt.year - 1)
        events_df = events_df.assign(season_guess=season_guess)
        joined_weeks = []
        for season in sorted(events_df["season_guess"].dropna().unique()):
            windows = build_week_windows(int(season))
            if windows.empty:
                return simple_week_tagging(events_df)
            e = events_df[events_df["season_guess"] == season].copy()
            w = windows.copy(); e["__k"] = 1; w["__k"] = 1
            j = e.merge(w, on="__k").drop(columns="__k")
            mask = (j["commence_date"] >= j["week_start_date"]) & (j["commence_date"] <= j["week_end_date"])
            j = j[mask]
            j["span"] = (pd.to_datetime(j["week_end_date"]) - pd.to_datetime(j["week_start_date"])).dt.days
            j = j.sort_values(["event_id","span"]).drop_duplicates("event_id", keep="first")
            joined_weeks.append(j[["event_id","week","season"]])
        if joined_weeks:
            return events_df.merge(pd.concat(joined_weeks, ignore_index=True), on="event_id", how="left")
        return simple_week_tagging(events_df)
    except Exception:
        return simple_week_tagging(events_df)

# --------------------------- Heuristic distributions ------------------
def receptions_prob_over(targets_mean: float, catch_rate: float, line: float, variance_mult: float = 1.0):
    mu = targets_mean * catch_rate
    var = targets_mean * catch_rate * (1 - catch_rate) * variance_mult
    sigma = float(np.sqrt(max(var, 1e-6)))
    return float(1 - norm.cdf(line + 0.5, loc=mu, scale=sigma))

def recyards_prob_over(receptions_mean: float, yards_per_rec: float, sigma_ypr: float, line: float, variance_mult: float = 1.0):
    mu = receptions_mean * yards_per_rec
    sigma = float(np.sqrt((receptions_mean * (sigma_ypr**2)) + 1e-6) * np.sqrt(variance_mult))
    return float(1 - norm.cdf(line, loc=mu, scale=sigma))

def rushyards_prob_over(carries_mean: float, ypc_mean: float, ypc_sd: float, line: float, variance_mult: float = 1.0):
    mu = carries_mean * ypc_mean
    sigma = float(np.sqrt(max(carries_mean,1.0)) * ypc_sd * np.sqrt(variance_mult))
    return float(1 - norm.cdf(line, loc=mu, scale=sigma))

def apply_bias(df: pd.DataFrame, prefer_unders: bool) -> pd.DataFrame:
    out = df.copy()
    if prefer_unders:
        glam = out['market'].isin(["Receptions","Rec Yds","Pass Yds","Anytime TD"])
        out.loc[glam, 'p_over'] = (out.loc[glam, 'p_over'] - UNDERS_BIAS_BPS/1e4).clip(0.01, 0.99)
    return out

# --------------------------- Regression model support -----------------
def load_reg_model(market_key: str):
    mpath = MODELS_DIR / f"props_reg_{market_key}.joblib"
    jpath = MODELS_DIR / f"props_reg_{market_key}.meta.json"
    if not mpath.exists() or not jpath.exists():
        return None, [], None
    model = joblib_load(mpath)
    meta = json.loads(jpath.read_text())
    feats = meta.get("features", [])
    sigma = float(meta.get("sigma", 20.0))
    return model, feats, sigma

def predict_p_over_from_reg(model, feats: List[str], sigma: float, row_dict: Dict, line_float: float) -> Tuple[float,float]:
    X = [row_dict.get(f, np.nan) for f in feats]
    mu = float(model.predict([X])[0])
    p_over = float(1 - norm.cdf(line_float, loc=mu, scale=float(sigma)))
    return p_over, mu

def enrich_for_reg_row(rr: pd.Series) -> Dict:
    d = {}
    team = rr.get("team", "")
    home_team = rr.get("home_team", "")
    d["is_home"] = 1 if team and team == home_team else 0
    for k in ["team_spread","total_line","days_rest",
              "receptions_l3","receptions_l5","receptions_l8",
              "targets_l5",
              "receiving_yards_l3","receiving_yards_l5","receiving_yards_l8",
              "rushing_yards_l3","rushing_yards_l5","rushing_yards_l8",
              "carries_l5",
              "passing_yards_l3","passing_yards_l5","passing_yards_l8",
              "receptions_allowed_l3","receptions_allowed_l5","receptions_allowed_l8",
              "receiving_yards_allowed_l3","receiving_yards_allowed_l5","receiving_yards_allowed_l8",
              "rushing_yards_allowed_l3","rushing_yards_allowed_l5","rushing_yards_allowed_l8",
              "passing_yards_allowed_l3","passing_yards_allowed_l5","passing_yards_allowed_l8",
              "h2h_mean_alltime"]:
        d[k] = float(rr.get(k, np.nan))
    return d

def player_key(name:str) -> str:
    if not isinstance(name,str) or not name.strip(): return ""
    parts = name.strip().split()
    return (parts[-1] + "_" + (parts[0][0] if parts else "")).lower()

# --------------------------- UI --------------------------------------
st.set_page_config(page_title="BetBot Props (Live + Trainable)", page_icon="🏈", layout="wide")
st.title("BetBot Props 🏈 — Probability-First Player Props")

col0, col1, col2, col3 = st.columns([1,1,1,2])
use_live = col0.toggle("Use LIVE odds", value=bool(ODDS_API_KEY))
prefer_unders = col1.checkbox("Prefer unders (book skew)", True)
model_mode = col2.selectbox("Model", ["Heuristic", "Regression (if available)"], index=0)
books_ui = col3.multiselect("Books (live)", ["DK","FD","MGM","PB","CZR"], default=["DK","FD"])
bookmaker_keys = [BOOK_INV[b] for b in books_ui if b in BOOK_INV]

st.sidebar.header("Risk Controls")
max_kelly = st.sidebar.slider("Kelly cap", 0.0, 0.5, MAX_KELLY_DEFAULT, 0.05)
exposure_cap = st.sidebar.slider("Max exposure per game (units)", 0.5, 5.0, EXPOSURE_CAP_DEFAULT, 0.5)

if use_live and not ODDS_API_KEY:
    st.warning("Missing ODDS_API_KEY. Add it in Settings → Variables & secrets.")
    use_live = False

# --------------------------- Data acquisition ------------------------
if use_live:
    try:
        events = fetch_events_live()
    except Exception as e:
        st.error(f"Could not fetch events: {e}")
        events = pd.DataFrame()

    if not events.empty:
        events = map_events_to_weeks(events)
        weeks = sorted([int(w) for w in events["week"].dropna().unique().tolist()])
    else:
        weeks = []

    topc1, topc2 = st.columns([1,2])
    week_choice = topc1.selectbox("NFL Week", ["All"] + weeks if weeks else ["All"], index=0)
    ev_view = events if week_choice == "All" else events[events["week"] == int(week_choice)]
    games_list = ev_view['game'].tolist() if not ev_view.empty else []
    sel_game = topc2.selectbox("Matchup", games_list) if games_list else None

    if sel_game:
        event_id = ev_view.loc[ev_view['game']==sel_game, 'event_id'].values[0]
        try:
            props = fetch_player_props_for_event(event_id, MARKET_KEYS, bookmaker_keys or None)
        except Exception as e:
            st.error(f"Could not fetch player props: {e}")
            props = pd.DataFrame()

        if props.empty:
            st.info("No live props returned for this game (maybe not posted yet for selected books).")
        else:
            priors = project_players_from_csv(week=1)
            priors['player_key'] = priors['player'].apply(player_key)
            props['player_key'] = props['player'].apply(player_key)
            props = props.merge(
                priors[['player_key','team','role','targets_mean','catch_rate','receptions_mu',
                        'rec_yards_mu','carries_mean','yards_per_carry','pass_rate','rush_rate',
                        'snap_share','sample_games','game']],
                on='player_key', how='left'
            )

            # totals/spreads
            try:
                total_line, spread_map = fetch_event_mainlines(event_id, bookmaker_keys or None)
            except Exception:
                total_line, spread_map = np.nan, {}
            props['total_line'] = float(total_line) if np.isfinite(total_line) else np.nan

            def team_spread_for_row(row):
                t = row.get('team', None)
                if t in spread_map: return float(spread_map[t])
                if row.get('home_team') in spread_map and t == row.get('home_team'): return float(spread_map[row['home_team']])
                if row.get('away_team') in spread_map and t == row.get('away_team'): return float(spread_map[row['away_team']])
                return np.nan
            props['team_spread'] = props.apply(team_spread_for_row, axis=1)

            # compute is_home for explanation
            def comp_is_home(row):
                t, ht, at = row.get('team'), row.get('home_team'), row.get('away_team')
                if pd.isna(t) or (pd.isna(ht) and pd.isna(at)): return np.nan
                if t == ht: return 1
                if t == at: return 0
                return np.nan
            props['is_home'] = props.apply(comp_is_home, axis=1)

            props['over_odds'] = pd.to_numeric(props['over_odds'], errors='coerce').fillna(-110).astype(int)
            props['under_odds'] = pd.to_numeric(props.get('under_odds', -110), errors='coerce').fillna(-110).astype(int)

else:
    # CSV fallback view (useful for testing)
    week = st.selectbox("NFL Week (CSV fallback)", list(range(1,19)), index=0)
    players_proj = project_players_from_csv(week)
    markets_path = DATA_DIR/"sample_markets.csv"
    if not markets_path.exists():
        st.error("CSV fallback requires data/sample_markets.csv")
        st.stop()
    markets = pd.read_csv(markets_path)
    for c in ["over_odds","under_odds","line","week"]:
        markets[c] = pd.to_numeric(markets[c], errors='coerce')
    markets['over_odds'] = markets['over_odds'].fillna(-110).astype(int)
    markets['under_odds'] = markets['under_odds'].fillna(-110).astype(int)
    markets['line'] = markets['line'].fillna(0.0).astype(float)
    markets['week'] = markets['week'].fillna(week).astype(int)
    props = markets.merge(players_proj, on=["player","team","week"], how="left", suffixes=("_mkt","_proj"))
    if 'game' not in props.columns: props['game'] = props.get('game_mkt', props.get('game_proj', np.nan))
    if 'role' not in props.columns: props['role'] = props.get('role_mkt', props.get('role_proj', 'N/A'))
    props['is_home'] = np.nan

# No props? stop cleanly
if 'props' not in locals() or props is None or props.empty:
    st.stop()

# --------------------------- Probability engine ----------------------
def load_reg_model_safe(market_label: str):
    mk = LABEL_TO_MK.get(market_label)
    return load_reg_model(mk) if mk else (None, [], None)

def compute_p_over_row(rr: pd.Series, model_mode: str) -> Tuple[float, float]:
    market = rr.get("market", "")
    line = float(rr.get("line", np.nan))
    if not np.isfinite(line): return (np.nan, np.nan)
    # Regression if available
    if model_mode.startswith("Regression"):
        model, feats, sigma = load_reg_model_safe(market)
        if model is not None:
            feat_row = enrich_for_reg_row(rr)
            try:
                p, mu = predict_p_over_from_reg(model, feats, sigma, feat_row, line)
                return (p, mu)
            except Exception:
                pass
    # Heuristic fallback
    if market == "Receptions":
        p = receptions_prob_over(rr.get('targets_mean',0.0), rr.get('catch_rate',0.6), line, 1.0)
        mu = rr.get('receptions_mu', np.nan)
    elif market == "Rec Yds":
        ypr = max(7.5, float(rr.get('receptions_mu',0.0))/max(rr.get('receptions_mu',1e-6),1e-6) * 9.0)
        p = recyards_prob_over(rr.get('receptions_mu',0.0), ypr, 8.0, line, 1.0); mu = rr.get('rec_yards_mu', np.nan)
    elif market == "Rush Yds":
        p = rushyards_prob_over(rr.get('carries_mean',0.0), rr.get('yards_per_carry',4.2), 1.1, line, 1.0); mu = rr.get('rush_yards_mu', np.nan)
    elif market == "Pass Yds":
        mu = 250.0 * float(rr.get('pass_rate',0.55)); p = float(1 - norm.cdf(line, loc=mu, scale=45.0))
    else:
        p, mu = np.nan, np.nan
    return (p, mu)

p_over_vals, mu_vals = [], []
for _, r in props.iterrows():
    p, mu = compute_p_over_row(r, model_mode)
    p_over_vals.append(p); mu_vals.append(mu)
props['p_over'] = p_over_vals
props['model_mu'] = mu_vals

# Optional bias to unders
props = apply_bias(props, prefer_unders)

# Pricing & EV
p_clean = pd.to_numeric(props['p_over'], errors='coerce')
props['fair_over'] = p_clean.apply(fair_price)
props['p_under'] = 1 - p_clean
props['edge_ev'] = props.apply(
    lambda rr: ev_from_prob(float(rr['p_over']) if np.isfinite(rr['p_over']) else 0.0, int(rr['over_odds'])),
    axis=1
)

# Confidence scaffold
props['conf_data_fresh'] = 0.8
props['conf_role_stab'] = pd.to_numeric(props.get('snap_share', 0.6), errors='coerce').fillna(0.6).clip(0.4,0.9)
props['conf_sample'] = pd.to_numeric(props.get('sample_games', 6), errors='coerce').fillna(6.0).rdiv(10.0).clip(0.3,1.0)
line_disp = props.groupby(['player','market'])['line'].transform(lambda s: (s.max()-s.min()))
props['conf_line_disp'] = (line_disp.clip(0.0,0.5) / 0.5).fillna(0.0)
props['confidence'] = [
    confidence_score(a,b,c,d)
    for a,b,c,d in zip(
        props['conf_data_fresh'], props['conf_role_stab'], props['conf_line_disp'], props['conf_sample']
    )
]
props['edge_score'] = props['edge_ev'] * props['confidence']

# --------------------------- Probability-first picks ------------------
st.subheader("Best Picks (by hit probability)")
# Best-side fields
props['under_odds'] = pd.to_numeric(props.get('under_odds', -110), errors='coerce').fillna(-110).astype(int)
props['side_best'] = np.where(props['p_over'] >= 0.5, 'Over', 'Under')
props['p_best']    = np.where(props['side_best']=='Over', props['p_over'], 1 - props['p_over'])
props['odds_best'] = np.where(props['side_best']=='Over', props['over_odds'], props['under_odds'])
props['fair_best'] = props['p_best'].apply(fair_price)
props['ev_best']   = props.apply(lambda rr: ev_from_prob(float(rr['p_best']) if np.isfinite(rr['p_best']) else 0.0,
                                                         int(rr['odds_best'])), axis=1)
props['prob_score'] = props['p_best'] * props['confidence']

f1, f2, f3 = st.columns([1,1,1])
min_prob = f1.slider("Min hit probability", 0.50, 0.80, 0.58, 0.01)
max_hold = f2.selectbox("Max juice allowed", ["No limit","-250","-200","-175","-150","-130"], index=2)
sort_mode = f3.selectbox("Sort by", ["Hit probability", "Probability × Confidence", "EV (best side)"], index=0)

view = props[props['p_best'] >= float(min_prob)].copy()
if max_hold != "No limit":
    cap = int(max_hold)
    view = view[(view['odds_best'] >= cap) | (view['odds_best'] > 0)]

if sort_mode == "Hit probability":
    view = view.sort_values(['p_best','confidence'], ascending=False)
elif sort_mode == "Probability × Confidence":
    view = view.sort_values(['prob_score','p_best'], ascending=False)
else:
    view = view.sort_values(['ev_best','p_best'], ascending=False)

best_cols = ["game","player","team","role","market","side_best","line","odds_best",
             "p_best","fair_best","confidence","model_mu"]
show_cols = [c for c in best_cols if c in view.columns]
st.dataframe(view[show_cols].round(3), use_container_width=True)

# --------------------------- Explanation panel -----------------------
with st.expander("Prop details & explanation"):
    if view.empty:
        st.info("No rows after filters. Lower the min probability or relax the juice cap.")
    else:
        idx = st.selectbox(
            "Select a pick",
            list(view.index),
            format_func=lambda i: f"{view.loc[i].get('game','')} — {view.loc[i,'player']} — {view.loc[i,'side_best']} {view.loc[i,'market']} {view.loc[i,'line']} ({view.loc[i].get('book','')})"
        )
        row = view.loc[idx]
        market = row['market']
        # Rough distribution
        if market == 'Receptions':
            mean = float(row.get('model_mu', row.get('receptions_mu', 4.0)))
            sd = max(0.8, np.sqrt(max(row.get('targets_mean',4.0),1.0)*row.get('catch_rate',0.62)*(1-row.get('catch_rate',0.62))))
        elif market == 'Rec Yds':
            mean = float(row.get('model_mu', row.get('rec_yards_mu', 48.0))); sd = 20.0
        elif market == 'Rush Yds':
            mean = float(row.get('model_mu', row.get('rush_yards_mu', 45.0))); sd = 15.0
        else:
            mean = float(row.get('model_mu', 250.0 * float(row.get('pass_rate',0.55)))); sd = 45.0

        x = np.linspace(max(0, mean-3*sd), mean+3*sd, 121)
        pdf = (1/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/sd)**2)
        ch = alt.Chart(pd.DataFrame({"x":x,"pdf":pdf})).mark_line().encode(x="x", y="pdf")
        st.altair_chart(ch, use_container_width=True)

        sigma_txt = "—"
        mk = LABEL_TO_MK.get(market)
        if mk:
            _, _, sig = load_reg_model(mk)
            if sig is not None:
                sigma_txt = f"{sig:.2f}"

        is_home = row.get('is_home', np.nan)
        loc_txt = "home" if is_home == 1 else ("away" if is_home == 0 else "—")
        spread = row.get('team_spread', np.nan)
        total = row.get('total_line', np.nan)
        snap = row.get('snap_share', np.nan)
        sample = row.get('sample_games', np.nan)

        st.markdown(f"""
**Why this pick**

- **Pick:** **{row['player']} — {row['side_best']} {row['market']} {row['line']}** at **{row.get('book','')} {int(row['odds_best'])}**
- **Hit probability:** **{row['p_best']:.1%}**  |  **Fair price:** {int(row['fair_best']) if np.isfinite(row['fair_best']) else '—'}
- **Model:** {('Regression' if model_mode.startswith('Regression') else 'Heuristic')} | **μ (mean):** {mean:.1f} | **σ:** {sigma_txt}
- **Context:** {loc_txt}, spread {'' if pd.isna(spread) else round(spread,1)}, total {'' if pd.isna(total) else round(total,1)}
- **Role/Sample:** snap {'' if pd.isna(snap) else f'{snap:.0%}'} | sample {'' if pd.isna(sample) else int(sample)} games
        """.strip())

        stake = kelly_fraction(float(row['p_best']) if np.isfinite(row['p_best']) else 0.0, int(row['odds_best']), float(max_kelly))
        stake = min(float(stake), float(exposure_cap))
        st.info(f"Suggested stake (capped): {stake:.2f} units")

        if st.button("Log this pick"):
            entry = {
                "ts": datetime.now().isoformat(timespec='seconds'),
                "game": row.get('game','N/A'),
                "player": row['player'],
                "team": row.get('team',''),
                "market": f"{row['side_best']} {row['market']}",
                "line": row['line'],
                "odds": int(row['odds_best']),
                "book": row.get('book',''),
                "model_prob": float(row['p_best']) if np.isfinite(row['p_best']) else np.nan,
                "fair_price": int(row['fair_best']) if np.isfinite(row['fair_best']) else np.nan,
                "confidence": float(row['confidence']),
                "stake_units": float(stake),
            }
            try:
                df_new = pd.DataFrame([entry])
                if PICKS_LOG.exists():
                    df_out = pd.concat([pd.read_csv(PICKS_LOG), df_new], ignore_index=True)
                else:
                    df_out = df_new
                df_out.to_csv(PICKS_LOG, index=False)
                st.success("Logged to data/picks_log.csv")
            except Exception as e:
                st.error(f"Could not write picks_log.csv: {e}")

# --------------------------- Line-Move Radar --------------------------
st.subheader("Line-Move Radar")
if not props.empty and all(c in props.columns for c in ['line','book','market','player']):
    radar = props.groupby(['player','market']).agg(
        line_min=('line','min'),
        line_max=('line','max'),
        books=('book','nunique')
    ).reset_index()
    radar['dispersion'] = radar['line_max'] - radar['line_min']
    st.dataframe(radar.sort_values('dispersion', ascending=False).head(20), use_container_width=True)
else:
    st.info("No market data loaded.")

# --------------------------- Live odds helpers ------------------------
@st.cache_data(ttl=45)
def fetch_player_props_for_event(event_id: str, markets: List[str], bookmaker_keys: List[str] | None) -> pd.DataFrame:
    params = {"regions": REGIONS, "oddsFormat": ODDS_FMT, "markets": ",".join(markets)}
    if bookmaker_keys: params["bookmakers"] = ",".join(bookmaker_keys)
    js = _odds_get(f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds", params)
    out_rows = []
    for bm in js.get("bookmakers", []):
        bkey = bm.get("key","")
        for m in bm.get("markets", []):
            mkey = m.get("key",""); if mkey not in markets: continue
            for o in m.get("outcomes", []):
                side = str(o.get("name","")).strip()   # Over/Under
                player = o.get("description") or o.get("participant") or o.get("player") or ""
                point = o.get("point"); price = o.get("price")
                if not player or side not in {"Over","Under"}:
                    if side and side not in {"Over","Under"}:
                        player, side = side, o.get("description","")
                if side not in {"Over","Under"}: continue
                out_rows.append({
                    "event_id": js.get("id"),
                    "home_team": js.get("home_team"),
                    "away_team": js.get("away_team"),
                    "game": f"{js.get('away_team')} @ {js.get('home_team')}",
                    "market_key": mkey,
                    "market": MARKET_LABEL.get(mkey, mkey),
                    "player": player,
                    "line": float(point) if point is not None else np.nan,
                    "book_key": bkey,
                    "book": BOOK_LABEL.get(bkey, bkey),
                    "side": side,
                    "price": int(price) if price is not None else np.nan,
                })
    df = pd.DataFrame(out_rows)
    if df.empty: return df
    over = df[df["side"]=="Over"][["player","market","line","book","book_key","price","game","home_team","away_team"]].rename(columns={"price":"over_odds"})
    under= df[df["side"]=="Under"][["player","market","line","book","book_key","price","game","home_team","away_team"]].rename(columns={"price":"under_odds"})
    merged = pd.merge(over, under, on=["player","market","line","book","book_key","game","home_team","away_team"], how="outer")
    return merged

@st.cache_data(ttl=45)
def fetch_event_mainlines(event_id: str, bookmaker_keys: List[str] | None) -> Tuple[float, Dict[str,float]]:
    params = {"regions": REGIONS, "oddsFormat": ODDS_FMT, "markets": "totals,spreads"}
    if bookmaker_keys: params["bookmakers"] = ",".join(bookmaker_keys)
    js = _odds_get(f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds", params)
    totals = []; team_spreads = {}
    for bm in js.get("bookmakers", []):
        for m in bm.get("markets", []):
            key = m.get("key")
            if key == "totals":
                for o in m.get("outcomes", []):
                    if "point" in o and o.get("name") in {"Over","Under"}:
                        totals.append(float(o["point"]))
            elif key == "spreads":
                for o in m.get("outcomes", []):
                    tname = o.get("name"); pt = o.get("point")
                    if tname and pt is not None:
                        team_spreads.setdefault(tname, []).append(float(pt))
    median_total = float(np.median(totals)) if totals else np.nan
    spread_map = {t: float(np.median(vals)) for t,vals in team_spreads.items()}
    return median_total, spread_map
