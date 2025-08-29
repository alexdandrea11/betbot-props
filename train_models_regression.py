# train_models_regression.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from joblib import dump as joblib_dump

DATA = Path("data")
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)

MARKETS = [
    "player_receptions",
    "player_reception_yds",
    "player_rush_yds",
    "player_pass_yds",
]

def load_train(mkt):
    df = pd.read_parquet(DATA / f"train_{mkt}.parquet")
    feats = (DATA / f"train_{mkt}.feats.txt").read_text().strip().splitlines()
    # Drop rows without target
    df = df.dropna(subset=["y"]).copy()
    return df, feats

def season_split(df, test_season):
    tr = df[df["season"] < test_season].copy()
    te = df[df["season"] == test_season].copy()
    return tr, te

def train_market(mkt):
    df, feats = load_train(mkt)
    seasons = sorted(df["season"].dropna().unique())
    if len(df) < 2000 or len(seasons) < 3:
        print(f"[{mkt}] Not enough data yet ({len(df)} rows, {len(seasons)} seasons)"); 
        return

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400))
    ])

    # Time-aware CV by season (last season is test in each fold)
    rmses = []
    for s in seasons[2:]:  # start after first 2 seasons to give train size
        tr, te = season_split(df, s)
        if tr.empty or te.empty: 
            continue
        model.fit(tr[feats], tr["y"])
        pred = model.predict(te[feats])
        rmse = mean_squared_error(te["y"], pred, squared=False)
        rmses.append(rmse)

    sigma = float(np.median(rmses)) if rmses else float(df["y"].std())
    # Fit final on ALL
    model.fit(df[feats], df["y"])

    joblib_dump(model, MODELS / f"props_reg_{mkt}.joblib")
    meta = {"features": feats, "sigma": sigma, "rows": int(len(df)), "seasons": seasons}
    (MODELS / f"props_reg_{mkt}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{mkt}] saved model + meta (σ≈{sigma:.2f})")

def main():
    for m in MARKETS:
        try:
            train_market(m)
        except FileNotFoundError:
            print(f"[skip] training frame missing for {m}. Run etl_build_history.py first.")
    print("Done.")

if __name__ == "__main__":
    main()
