# etl_build_history.py
# Build per-market training tables from nfl_data_py weekly logs + schedules + scoring lines
from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

OUT_DIR = Path("data"); OUT_DIR.mkdir(parents=True, exist_ok=True)

START_SEASON = 2018
END_SEASON   = 2024
SEASONS = list(range(START_SEASON, END_SEASON+1))

# Markets we’ll train: stat column mapping in weekly data
MARKETS = {
    "player_receptions":     {"y": "receptions",        "aux": ["targets"]},
    "player_reception_yds":  {"y": "receiving_yards",   "aux": ["receptions","targets"]},
    "player_rush_yds":       {"y": "rushing_yards",     "aux": ["carries"]},
    "player_pass_yds":       {"y": "passing_yards",     "aux": ["attempts","completions"]},
}

def safe_col(df, name, default=np.nan):
    if name in df.columns: return df[name]
    return pd.Series(default, index=df.index)

def build_base():
    # Weekly player stats (regular + postseason included; that’s fine)
    wk = nfl.import_weekly_data(SEASONS)                                         # :contentReference[oaicite:2]{index=2}
    # Schedules + lines (spread/total are game-level; join via game_id)
    sch = nfl.import_schedules(SEASONS)                                          # :contentReference[oaicite:3]{index=3}
    lines = nfl.import_sc_lines(SEASONS)                                         # :contentReference[oaicite:4]{index=4}

    # Keep main columns
    keep = ["player","player_id","position","team","opponent_team","season","week",
            "game_id","receptions","targets","receiving_yards",
            "rushing_yards","carries","passing_yards","attempts","completions","fantasy_points_ppr"]
    wk = wk[[c for c in keep if c in wk.columns]].copy()

    # Merge schedule + lines
    g = sch[["game_id","home_team","away_team","gameday"]].copy()
    l = lines[["game_id","spread_line","total_line"]].drop_duplicates("game_id")
    base = wk.merge(g, on="game_id", how="left").merge(l, on="game_id", how="left")

    # Home/away & team-centric spread
    base["is_home"] = (base["team"] == base["home_team"]).astype(int)
    # spread_line is from home team POV; convert to player's team POV
    base["team_spread"] = np.where(base["is_home"]==1, base["spread_line"], -safe_col(base,"spread_line"))
    base["total_line"]  = safe_col(base, "total_line")

    # Days rest (approx): previous game date per player
    base["gameday"] = pd.to_datetime(base["gameday"])
    base.sort_values(["player","season","week"], inplace=True)
    base["prev_gameday"] = base.groupby("player")["gameday"].shift(1)
    base["days_rest"] = (base["gameday"] - base["prev_gameday"]).dt.days.fillna(10)

    return base

def add_rolling_features(df, stat_cols):
    # Make player-level rolling means (shifted to avoid leakage)
    df = df.copy()
    df.sort_values(["player","season","week"], inplace=True)
    for col in stat_cols:
        if col not in df.columns: 
            df[col] = np.nan
        for w in [3,5,8]:
            df[f"{col}_l{w}"] = (
                df.groupby("player")[col].shift(1).rolling(w, min_periods=1).mean()
            )
    return df

def add_head_to_head(df, stat_col):
    # Career average of stat vs THAT opponent (up to previous meeting)
    df = df.copy()
    df["h2h_key"] = df["player"] + "@" + df["opponent_team"]
    df.sort_values(["player","season","week"], inplace=True)
    grp = df.groupby("h2h_key")[stat_col].expanding().mean().reset_index(level=0, drop=True)
    df["h2h_mean_alltime"] = grp.groupby(df["h2h_key"]).shift(1)  # shift to pregame value
    return df

def add_opp_allowed(df, stat_col):
    # Team-allowed stat: sum of opponent players’ stat vs that defense, rolling by week
    tmp = df.groupby(["season","week","opponent_team"], as_index=False)[stat_col].sum()
    tmp = tmp.rename(columns={"opponent_team":"def_team", stat_col:f"{stat_col}_allowed"})
    tmp.sort_values(["def_team","season","week"], inplace=True)
    for w in [3,5,8]:
        tmp[f"{stat_col}_allowed_l{w}"] = (
            tmp.groupby("def_team")[f"{stat_col}_allowed"].shift(1).rolling(w, min_periods=1).mean()
        )
    # Merge back: the defense we’re facing is opponent_team
    df = df.merge(tmp[["season","week","def_team",
                       f"{stat_col}_allowed_l3",f"{stat_col}_allowed_l5",f"{stat_col}_allowed_l8"]],
                  left_on=["season","week","opponent_team"],
                  right_on=["season","week","def_team"], how="left")
    return df.drop(columns=["def_team"])

def build_market_table(base, market_key, ycol, aux_cols):
    df = base.copy()
    # Rolling on target stat + aux volume columns
    stat_cols = [ycol] + [c for c in aux_cols if c in df.columns]
    df = add_rolling_features(df, stat_cols)
    df = add_head_to_head(df, ycol)
    df = add_opp_allowed(df, ycol)

    # Minimal feature set (model can impute if NaN)
    feat_cols = [
        "is_home","team_spread","total_line","days_rest",
        f"{ycol}_l3", f"{ycol}_l5", f"{ycol}_l8",
        f"{ycol}_allowed_l3", f"{ycol}_allowed_l5", f"{ycol}_allowed_l8",
        "h2h_mean_alltime",
    ]
    # Add auxiliaries’ l5 if present
    for aux in aux_cols:
        if f"{aux}_l5" in df.columns:
            feat_cols.append(f"{aux}_l5")

    keep = ["player","player_id","position","team","opponent_team","season","week","game_id", ycol] + feat_cols
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={ycol:"y"})
    df["market_key"] = market_key
    return df, feat_cols

def main():
    base = build_base()
    for mkt, m in MARKETS.items():
        tbl, feats = build_market_table(base, mkt, m["y"], m["aux"])
        # Save per-market training frame + feature list (metadata)
        out = OUT_DIR / f"train_{mkt}.parquet"
        tbl.to_parquet(out, index=False)
        with open(OUT_DIR / f"train_{mkt}.feats.txt","w") as f:
            f.write("\n".join(feats))
        print(f"[OK] {mkt}: {len(tbl):,} rows, saved -> {out}")

if __name__ == "__main__":
    main()
