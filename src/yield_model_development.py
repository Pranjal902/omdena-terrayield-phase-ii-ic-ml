"""Crop-yield model development on the Gold candidate.

Builds on season_yield_baselines.py. Three things change:

  1. Target. Models predict each season's relative gap to its own state-crop trend
     (detrended yield) instead of the raw yield level. The raw level is mostly "which crop
     is it" plus a slow upward trend, which simple baselines already capture.
  2. Features. Agronomic season features (src/features.py) instead of plain averages.
  3. Tuning. Hyperparameters are picked by an inner leave-one-year-out loop on the training
     years only (nested cross-validation), so the test year never influences the choice.
     When two settings tie, the simpler one wins.

On top of the model ladder it runs:
  - the same nested protocol on the plain June-September averages from the first experiment,
    so the two feature sets are compared on equal terms
  - in-season forecasts with cut-offs on 1 July, 1 August, 1 September and 1 October
  - interpretation of the ridge model (coefficients, sign stability, drop-column importance)
  - error analysis by season
  - a sensitivity run without the Iowa soybean 2022 value the project had flagged

Usage:
    python src/yield_model_development.py --data-dir data --out-dir results
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_gold
from features import FEATURES, KEY, describe_cutoffs, season_features
from season_yield_baselines import FEATURES as PLAIN_FEATURES
from season_yield_baselines import (linear_trend, previous_season, region_crop_mean, season_table, splits,
                                    summarise, trend_fit, trend_predict)

SEED = 42
SCHEMES = ("leave_one_year_out", "expanding_window")
CUTOFFS = {7: "1 Jul", 8: "1 Aug", 9: "1 Sep", 10: "1 Oct"}
FLAGGED_SEASON = ("US-Iowa", "soybean", 2022)

# Each grid is ordered from simplest to most flexible, so ties go to the simpler setting.
MODELS = {
    "ridge": (
        lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"])),
        [{"alpha": a} for a in (100, 30, 10, 3, 1, 0.3)],
    ),
    "elastic_net": (
        lambda p: make_pipeline(StandardScaler(), ElasticNet(alpha=p["alpha"], l1_ratio=0.5, max_iter=100_000)),
        [{"alpha": a} for a in (0.1, 0.03, 0.01, 0.003, 0.001)],
    ),
    "random_forest": (
        lambda p: RandomForestRegressor(n_estimators=300, max_depth=p["max_depth"],
                                        min_samples_leaf=p["min_samples_leaf"], random_state=SEED),
        [{"max_depth": d, "min_samples_leaf": leaf} for d in (2, 3) for leaf in (4, 2)],
    ),
    "gradient_boosting": (
        lambda p: GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=0.05,
                                            max_depth=p["max_depth"], subsample=0.8, random_state=SEED),
        [{"n_estimators": n, "max_depth": d} for d in (1, 2) for n in (50, 150)],
    ),
}


# ---- fitting -------------------------------------------------------------------------

def fit_predict_gap(name, params, train, test, features):
    """Fit on the relative gap to trend, return yield predictions for the test rows."""
    make, _ = MODELS[name]
    fits = trend_fit(train)
    trend_train, trend_test = trend_predict(fits, train), trend_predict(fits, test)
    gap = train["yield_t_ha"].to_numpy() / trend_train - 1
    model = make(params).fit(train[features], gap)
    return trend_test * (1 + model.predict(test[features]))


def fit_predict_raw(name, params, train, test, features):
    """Same model on the raw yield level, with crop, state and year as extra inputs."""
    make, _ = MODELS[name]

    def design(df):
        x = df[features].copy()
        x["is_corn"] = (df["crop_type"] == "corn").astype(int)
        x["is_iowa"] = (df["region"] == "US-Iowa").astype(int)
        x["year"] = df["year"]
        return x

    model = make(params).fit(design(train), train["yield_t_ha"])
    return model.predict(design(test))


def tune(name, train, features, fit_predict=fit_predict_gap):
    """Inner leave-one-year-out on the training years only."""
    _, grid = MODELS[name]
    years = sorted(train["year"].unique())
    best, best_mae = grid[0], np.inf
    for params in grid:
        errs = []
        for y in years:
            tr, te = train[train["year"] != y], train[train["year"] == y]
            errs += list(np.abs(fit_predict(name, params, tr, te, features) - te["yield_t_ha"].to_numpy()))
        mae = float(np.mean(errs))
        if mae < best_mae - 1e-9:
            best, best_mae = params, mae
    return best


def run_ladder(table, features=FEATURES, schemes=SCHEMES, include_raw=True, models=tuple(MODELS)):
    rows, chosen = [], []
    years = sorted(table["year"].unique())
    baselines = {"region_crop_mean": region_crop_mean, "previous_season": previous_season,
                 "linear_trend": linear_trend}
    learners = [(m, "detrended", fit_predict_gap) for m in models]
    if include_raw:
        learners.append(("random_forest", "raw_level", fit_predict_raw))
    for scheme in schemes:
        for train_years, test_year in splits(years, scheme):
            train, test = table[table["year"].isin(train_years)], table[table["year"] == test_year]
            preds = {(name, "none"): fn(train, test) for name, fn in baselines.items()}
            for name, target, fp in learners:
                params = tune(name, train, features, fp)
                preds[(name, target)] = fp(name, params, train, test, features)
                chosen.append({"scheme": scheme, "test_year": test_year, "model": name, "target": target, **params})
            for (name, target), pred in preds.items():
                for (r, c, y, actual), p in zip(test[KEY + ["yield_t_ha"]].itertuples(index=False), pred):
                    rows.append({"scheme": scheme, "model": name, "features": target, "test_year": y,
                                 "region": r, "crop_type": c, "actual": actual, "predicted": float(p)})
    return pd.DataFrame(rows), pd.DataFrame(chosen)


# ---- feature sets on equal terms -------------------------------------------------------

def compare_feature_sets(gold):
    """Nested ridge and gradient boosting on both feature sets, same splits and tuning."""
    sets = {"plain Jun-Sep averages": (season_table(gold, [6, 7, 8, 9]), PLAIN_FEATURES),
            "agronomic season features": (season_features(gold), FEATURES)}
    rows = []
    for label, (table, feats) in sets.items():
        pred, _ = run_ladder(table, features=feats, include_raw=False, models=("ridge", "gradient_boosting"))
        m = summarise(pred)
        for _, r in m[m["features"] == "detrended"].iterrows():
            rows.append({"feature_set": label, "scheme": r["scheme"], "model": r["model"],
                         "mae": r["mae"], "rmse": r["rmse"]})
    return pd.DataFrame(rows)


# ---- in-season forecasts --------------------------------------------------------------

def run_in_season(gold):
    rows = []
    for month, label in CUTOFFS.items():
        table = season_features(gold, month)
        pred, _ = run_ladder(table, include_raw=False, models=("ridge",))
        m = summarise(pred)
        for _, r in m.iterrows():
            rows.append({"cutoff": label, "scheme": r["scheme"], "model": r["model"], "mae": r["mae"],
                         "rmse": r["rmse"]})
    return pd.DataFrame(rows)


# ---- interpretation -------------------------------------------------------------------

def ridge_interpretation(table):
    """Standardised coefficients on the full data, their sign stability across LOYO folds,
    and how much leave-one-year-out MAE rises when each feature is dropped."""
    years = sorted(table["year"].unique())
    alpha = tune("ridge", table, FEATURES)["alpha"]

    def coefs(df):
        fits = trend_fit(df)
        gap = df["yield_t_ha"].to_numpy() / trend_predict(fits, df) - 1
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(df[FEATURES], gap)
        return pipe[-1].coef_ * 100  # percent of trend yield per standard deviation

    full = coefs(table)
    folds = np.array([coefs(table[table["year"] != y]) for y in years])
    same_sign = (np.sign(folds) == np.sign(full)).mean(axis=0)

    def loyo_mae(features):
        errs = []
        for train_years, test_year in splits(years, "leave_one_year_out"):
            tr, te = table[table["year"].isin(train_years)], table[table["year"] == test_year]
            params = tune("ridge", tr, features)
            errs += list(np.abs(fit_predict_gap("ridge", params, tr, te, features) - te["yield_t_ha"].to_numpy()))
        return float(np.mean(errs))

    base = loyo_mae(FEATURES)
    drop = {f: loyo_mae([x for x in FEATURES if x != f]) - base for f in FEATURES}
    out = pd.DataFrame({"feature": FEATURES, "coef_pct_per_sd": full.round(3),
                        "same_sign_share_of_folds": same_sign.round(2),
                        "mae_increase_when_dropped": [round(drop[f], 4) for f in FEATURES]})
    return out.sort_values("mae_increase_when_dropped", ascending=False).reset_index(drop=True), alpha, base


# ---- charts ---------------------------------------------------------------------------

def _label(model, target):
    return {"none": model, "detrended": f"{model} (detrended)", "raw_level": f"{model} (raw yield)"}[target]


def plot_ladder(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    colour = {"none": "#8a8f98", "detrended": "#2f6b3a", "raw_level": "#b5442c"}
    for ax, scheme in zip(axes, SCHEMES):
        m = metrics[metrics["scheme"] == scheme].sort_values("mae", ascending=False)
        ax.barh([_label(a, b) for a, b in zip(m["model"], m["features"])], m["mae"],
                color=[colour[f] for f in m["features"]])
        for i, v in enumerate(m["mae"]):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(scheme.replace("_", " "), loc="left", fontsize=10)
        ax.set_xlabel("MAE, t/ha (lower is better)")
        ax.set_xlim(0, metrics["mae"].max() * 1.18)
    fig.suptitle("Model ladder: grey baselines, green detrended models, red raw-yield model",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_in_season(in_season, metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    order = list(CUTOFFS.values())
    for ax, scheme in zip(axes, SCHEMES):
        s = in_season[(in_season["scheme"] == scheme) & (in_season["model"] == "ridge")].set_index("cutoff").loc[order]
        ax.plot(order, s["mae"], marker="o", color="#2f6b3a", label="ridge, features up to the cut-off")
        for base, style in (("linear_trend", "--"), ("previous_season", ":")):
            v = metrics[(metrics["scheme"] == scheme) & (metrics["model"] == base)]["mae"].iloc[0]
            ax.axhline(v, color="#8a8f98", ls=style, label=f"{base} ({v:.3f})")
        ax.set_title(scheme.replace("_", " "), loc="left", fontsize=10)
        ax.set_xlabel("forecast made on")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("MAE, t/ha")
    fig.suptitle("How early does the season tell you anything?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_series(pred, path):
    p = pred[pred["scheme"] == "leave_one_year_out"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for ax, ((region, crop), g) in zip(axes.ravel(), p.groupby(["region", "crop_type"])):
        actual = g[g["model"] == "linear_trend"].sort_values("test_year")
        ax.plot(actual["test_year"], actual["actual"], "o-", color="#222222", label="reported yield")
        ax.plot(actual["test_year"], actual["predicted"], "--", color="#8a8f98", label="trend")
        ridge = g[(g["model"] == "ridge") & (g["features"] == "detrended")].sort_values("test_year")
        ax.plot(ridge["test_year"], ridge["predicted"], "s-", color="#2f6b3a", ms=4, label="trend + ridge")
        ax.set_title(f"{region} {crop}, t/ha", loc="left", fontsize=10)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Leave-one-year-out predictions by season", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- main -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.data_dir)
    table = season_features(gold)

    pred, chosen = run_ladder(table)
    metrics = summarise(pred)
    in_season = run_in_season(gold)
    feature_sets = compare_feature_sets(gold)
    importance, alpha_full, ridge_loyo_mae = ridge_interpretation(table)

    flagged = (table["region"] == FLAGGED_SEASON[0]) & (table["crop_type"] == FLAGGED_SEASON[1]) & \
              (table["year"] == FLAGGED_SEASON[2])
    sens_pred, _ = run_ladder(table[~flagged], schemes=("leave_one_year_out",), include_raw=False)
    sensitivity = summarise(sens_pred)

    loyo = pred[(pred["scheme"] == "leave_one_year_out")]
    errors = (loyo[loyo["model"].isin(["linear_trend", "ridge"]) & loyo["features"].isin(["none", "detrended"])]
              .assign(abs_error=lambda d: (d["predicted"] - d["actual"]).abs())
              .pivot_table(index=["region", "crop_type", "test_year"], columns="model", values="abs_error")
              .round(3).reset_index())
    errors["ridge_minus_trend"] = (errors["ridge"] - errors["linear_trend"]).round(3)

    pred.to_csv(args.out_dir / "model_dev_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "model_dev_metrics.csv", index=False)
    in_season.to_csv(args.out_dir / "model_dev_in_season.csv", index=False)
    feature_sets.to_csv(args.out_dir / "model_dev_feature_sets.csv", index=False)
    importance.to_csv(args.out_dir / "model_dev_ridge_importance.csv", index=False)
    errors.to_csv(args.out_dir / "model_dev_errors_by_season.csv", index=False)
    info = {
        "seasons": int(len(table)),
        "features": FEATURES,
        "windows_per_season_by_cutoff": {CUTOFFS[k]: v for k, v in describe_cutoffs(gold).items()},
        "chosen_hyperparameters_count": {
            f"{m}|{t}": chosen[(chosen["model"] == m) & (chosen["target"] == t)]
            .drop(columns=["scheme", "test_year", "model", "target"]).dropna(axis=1)
            .astype(str).agg(", ".join, axis=1).value_counts().to_dict()
            for m, t in chosen[["model", "target"]].drop_duplicates().itertuples(index=False)
        },
        "ridge_alpha_on_full_data": alpha_full,
        "ridge_loyo_mae_all_features": round(ridge_loyo_mae, 3),
        "sensitivity_without_flagged_iowa_soybean_2022": sensitivity.to_dict(orient="records"),
    }
    (args.out_dir / "model_dev_info.json").write_text(json.dumps(info, indent=2))

    plot_ladder(metrics, args.out_dir / "model_dev_ladder.png")
    plot_in_season(in_season, metrics, args.out_dir / "model_dev_in_season.png")
    plot_series(pred, args.out_dir / "model_dev_predictions_by_season.png")

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(metrics.to_string(index=False))
        print("\nin-season:\n", in_season.to_string(index=False))
        print("\nfeature sets:\n", feature_sets.to_string(index=False))
        print("\nridge interpretation:\n", importance.to_string(index=False))
        print("\nerrors by season (LOYO):\n", errors.to_string(index=False))
        print("\nsensitivity (no flagged Iowa soybean 2022):\n", sensitivity.to_string(index=False))
    print(json.dumps({k: v for k, v in info.items() if k != "sensitivity_without_flagged_iowa_soybean_2022"},
                     indent=2))


if __name__ == "__main__":
    # ElasticNet at its smallest alpha doesn't always fully converge on 24 rows. The inner
    # tuning loop picks larger alphas, so the warning is noise here.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
