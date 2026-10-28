"""Season-level crop-yield baselines on the Gold candidate.

The question this answers comes from my label write-up: if the yield label had stayed on
the post-harvest December rows, the model would only ever see winter readings. Do
growing-season (June-September) features actually do better, and does either beat
simple baselines when there are only 24 seasons to learn from?

One row = one (region, crop, year) season. Two feature sets are built the same way:
  growing       windows starting June to September of the season year
  post_harvest  windows starting in December of the season year (where the label used to sit)

Models, simplest first:
  region_crop_mean    average yield of that region-crop in the training years
  previous_season     last year's yield for the same region-crop
  linear_trend        straight-line trend per region-crop
  trend_plus_ridge    trend, then a ridge model on the season features explains the
                      relative gap to trend (alpha fixed at 1.0, not tuned)
  random_forest       shallow forest on season features + crop, region and year dummies

Two ways of splitting:
  leave_one_year_out  hold one year out, train on the other five (what the project used)
  expanding_window    train only on years before the test year, test 2022 to 2025

Usage:
    python src/season_yield_baselines.py --data-dir data --out-dir results
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data import load_gold

SERIES = ["region", "crop_type"]
KEY = SERIES + ["year"]
FEATURE_WINDOWS = {"growing": [6, 7, 8, 9], "post_harvest": [12]}
FEATURES = ["ndvi_avg", "ndvi_peak", "savi_avg", "tmax_avg", "precip_total", "radiation_total"]
SEED = 42


def season_table(gold: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """One row per season: the annual yield plus features from the chosen windows."""
    w = gold[gold["window_start"].dt.month.isin(months) & (gold["window_start"].dt.year == gold["year"])]
    feats = w.groupby(KEY).agg(
        ndvi_avg=("ndvi_mean", "mean"),
        ndvi_peak=("ndvi_mean", "max"),
        savi_avg=("savi_mean", "mean"),
        tmax_avg=("t2m_max", "mean"),
        precip_total=("tp_sum", "sum"),
        radiation_total=("ssrd_sum", "sum"),
        windows=("window_id", "nunique"),
    )
    target = gold.groupby(KEY)["gov_yield_tonnes_per_ha"].first().rename("yield_t_ha")
    return feats.join(target).reset_index()


# ---- models: each takes (train, test) season tables and returns predictions for test ----

def trend_fit(train: pd.DataFrame) -> dict:
    """Straight-line trend per state-crop, as (slope, intercept).

    Closed-form least squares rather than np.polyfit: polyfit goes through LAPACK, whose last
    digits differ between maths libraries, and with corn and soybean sharing identical features
    those last digits are enough to flip tied tree splits. This version gives the same bits
    everywhere.
    """
    fits = {}
    for key, g in train.groupby(SERIES):
        x, y = g["year"].to_numpy(dtype=float), g["yield_t_ha"].to_numpy(dtype=float)
        if len(np.unique(x)) >= 2:
            xm, ym = x.mean(), y.mean()
            slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
            fits[key] = (slope, ym - slope * xm)
        else:
            fits[key] = (0.0, y.mean())
    return fits


def trend_predict(fits: dict, df: pd.DataFrame) -> np.ndarray:
    return np.array([fits[(r, c)][0] * y + fits[(r, c)][1] for r, c, y in df[KEY].itertuples(index=False)])


def region_crop_mean(train, test, **_):
    means = train.groupby(SERIES)["yield_t_ha"].mean()
    return test.set_index(SERIES).index.map(means).to_numpy()


def previous_season(train, test, **_):
    lookup = train.set_index(KEY)["yield_t_ha"]
    means = train.groupby(SERIES)["yield_t_ha"].mean()
    out = []
    for r, c, y in test[KEY].itertuples(index=False):
        out.append(lookup.get((r, c, y - 1), means[(r, c)]))  # 2020 has no prior season
    return np.array(out)


def linear_trend(train, test, **_):
    return trend_predict(trend_fit(train), test)


def trend_plus_ridge(train, test, features, alpha=1.0):
    fits = trend_fit(train)
    trend_train, trend_test = trend_predict(fits, train), trend_predict(fits, test)
    rel_gap = train["yield_t_ha"].to_numpy() / trend_train - 1
    scaler = StandardScaler().fit(train[features])
    model = Ridge(alpha=alpha).fit(scaler.transform(train[features]), rel_gap)
    return trend_test * (1 + model.predict(scaler.transform(test[features])))


def random_forest(train, test, features):
    def design(df):
        x = df[features].copy()
        x["is_corn"] = (df["crop_type"] == "corn").astype(int)
        x["is_iowa"] = (df["region"] == "US-Iowa").astype(int)
        x["year"] = df["year"]
        return x

    model = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2, random_state=SEED)
    model.fit(design(train), train["yield_t_ha"])
    return model.predict(design(test))


MODELS = [
    ("region_crop_mean", region_crop_mean, False),
    ("previous_season", previous_season, False),
    ("linear_trend", linear_trend, False),
    ("trend_plus_ridge", trend_plus_ridge, True),
    ("random_forest", random_forest, True),
]


def splits(years: list[int], scheme: str):
    if scheme == "leave_one_year_out":
        for y in years:
            yield [t for t in years if t != y], y
    elif scheme == "expanding_window":
        for y in years[2:]:
            yield [t for t in years if t < y], y
    else:
        raise ValueError(scheme)


def run(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    years = sorted(tables["growing"]["year"].unique())
    for scheme in ("leave_one_year_out", "expanding_window"):
        for train_years, test_year in splits(years, scheme):
            for name, fn, uses_features in MODELS:
                feature_sets = list(tables) if uses_features else ["none"]
                for fs in feature_sets:
                    table = tables["growing" if fs == "none" else fs]
                    train = table[table["year"].isin(train_years)]
                    test = table[table["year"] == test_year]
                    pred = fn(train, test, features=FEATURES)
                    for (r, c, y, actual), p in zip(test[KEY + ["yield_t_ha"]].itertuples(index=False), pred):
                        rows.append({"scheme": scheme, "model": name, "features": fs, "test_year": y,
                                     "region": r, "crop_type": c, "actual": actual, "predicted": float(p)})
    return pd.DataFrame(rows)


def summarise(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.assign(err=pred["predicted"] - pred["actual"])

    def stats(g):
        e = g["err"]
        return pd.Series({
            "mae": e.abs().mean(),
            "rmse": np.sqrt((e ** 2).mean()),
            "mape_pct": (e.abs() / g["actual"]).mean() * 100,
            "mae_corn": e[g["crop_type"] == "corn"].abs().mean(),
            "mae_soybean": e[g["crop_type"] == "soybean"].abs().mean(),
            "n_predictions": len(g),
        })

    out = pred.groupby(["scheme", "model", "features"]).apply(stats, include_groups=False).reset_index()
    out["n_predictions"] = out["n_predictions"].astype(int)
    out = out.sort_values(["scheme", "mae"]).reset_index(drop=True)
    return out.round({"mae": 3, "rmse": 3, "mape_pct": 2, "mae_corn": 3, "mae_soybean": 3})


def signal_check(tables: dict[str, pd.DataFrame]) -> dict:
    """Correlation between each feature and the relative gap to a full-sample trend."""
    out = {}
    for fs, t in tables.items():
        fits = trend_fit(t)
        gap = t["yield_t_ha"] / trend_predict(fits, t) - 1
        out[fs] = {f: round(float(np.corrcoef(t[f], gap)[0, 1]), 2) for f in FEATURES}
    return out


def alpha_sensitivity(tables: dict[str, pd.DataFrame], alphas=(0.3, 1.0, 3.0, 10.0)) -> dict:
    """MAE of trend_plus_ridge across alphas, to check the main result isn't an artefact of alpha=1."""
    years = sorted(tables["growing"]["year"].unique())
    out = {}
    for scheme in ("leave_one_year_out", "expanding_window"):
        for fs, t in tables.items():
            for alpha in alphas:
                errs = []
                for train_years, test_year in splits(years, scheme):
                    train, test = t[t["year"].isin(train_years)], t[t["year"] == test_year]
                    pred = trend_plus_ridge(train, test, FEATURES, alpha=alpha)
                    errs += list(np.abs(pred - test["yield_t_ha"].to_numpy()))
                out.setdefault(scheme, {}).setdefault(fs, {})[str(alpha)] = round(float(np.mean(errs)), 3)
    return out


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    labels = {"none": "", "growing": " (Jun-Sep)", "post_harvest": " (Dec)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, scheme in zip(axes, ("leave_one_year_out", "expanding_window")):
        m = metrics[metrics["scheme"] == scheme].sort_values("mae", ascending=False)
        names = [f"{a}{labels[b]}" for a, b in zip(m["model"], m["features"])]
        colours = ["#8a8f98" if f == "none" else "#2f6b3a" if f == "growing" else "#b5442c" for f in m["features"]]
        ax.barh(names, m["mae"], color=colours)
        for i, v in enumerate(m["mae"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_title(scheme.replace("_", " "), loc="left", fontsize=10)
        ax.set_xlabel("MAE, t/ha (lower is better)")
        ax.set_xlim(0, metrics["mae"].max() * 1.15)
    fig.suptitle("Season-level yield error by model and feature window", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_series(tables: dict[str, pd.DataFrame], path: Path) -> None:
    t = tables["growing"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, crop in zip(axes, ("corn", "soybean")):
        for region, colour in (("US-Illinois", "#2f6b3a"), ("US-Iowa", "#c08a2d")):
            s = t[(t["crop_type"] == crop) & (t["region"] == region)].sort_values("year")
            ax.plot(s["year"], s["yield_t_ha"], marker="o", color=colour, label=region)
        ax.set_title(f"{crop} yield, t/ha", loc="left", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.data_dir)
    tables = {name: season_table(gold, months) for name, months in FEATURE_WINDOWS.items()}
    pred = run(tables)
    metrics = summarise(pred)

    pred.to_csv(args.out_dir / "season_experiment_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "season_experiment_metrics.csv", index=False)
    info = {
        "seasons": int(len(tables["growing"])),
        "windows_per_season": {k: sorted(v["windows"].unique().tolist()) for k, v in tables.items()},
        "feature_vs_gap_to_trend_correlation": signal_check(tables),
        "trend_plus_ridge_mae_by_alpha": alpha_sensitivity(tables),
    }
    (args.out_dir / "season_experiment_info.json").write_text(json.dumps(info, indent=2))
    plot_metrics(metrics, args.out_dir / "season_experiment_mae.png")
    plot_series(tables, args.out_dir / "season_yields.png")

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(metrics.to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
