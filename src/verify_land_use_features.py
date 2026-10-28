"""Rerun the verification log behind docs/land-use-feature-manifest.md.

The land-use feature table works at one row per (region, window). Before writing the
manifest I wanted every number in it recomputed from the data rather than copied from
another document. This script does that, check by check (V-01 to V-08), and adds two
extra checks (rainfall aggregation and the 2022 yield values).

Usage:
    python src/verify_land_use_features.py --data-dir data --out-dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_gold, load_master, satellite_columns

INDICES = ["ndvi_mean", "savi_mean"]


def region_window_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the crop rows to one row per region-window (safe only if V-01 holds)."""
    cols = ["region_id", "window_id", "window_start", "window_end", "valid_pixel_count",
            "ndvi_mean", "savi_mean", "band_b03_mean", "band_b08_mean", "band_b11_mean"]
    rw = df[cols].drop_duplicates(["region_id", "window_id"]).sort_values(["region_id", "window_start"])
    return rw.reset_index(drop=True)


def v01_crop_identity(df: pd.DataFrame) -> dict:
    nunique = df.groupby(["region_id", "window_id"])[satellite_columns(df)].nunique(dropna=False)
    return {"groups": int(len(nunique)), "differing_cells": int((nunique > 1).sum().sum())}


def v02_yoy_coverage(rw: pd.DataFrame, tolerance_days: int = 10) -> dict:
    """For each window, look for the window that started ~365 days earlier (+/- tolerance)."""
    unmatched = []
    for region, g in rw.groupby("region_id"):
        target = g[["window_id", "window_start"]].assign(target=g["window_start"] - pd.Timedelta(days=365))
        prior = g[["window_start"]].rename(columns={"window_start": "prior_start"})
        m = pd.merge_asof(target.sort_values("target"), prior.sort_values("prior_start"),
                          left_on="target", right_on="prior_start", direction="nearest",
                          tolerance=pd.Timedelta(days=tolerance_days))
        miss = m[m["prior_start"].isna()]
        unmatched += [(region, d.year) for d in miss["window_start"]]
    years = pd.Series([y for _, y in unmatched]).value_counts().to_dict()
    return {"rows": int(len(rw)), "unmatched": len(unmatched),
            "unmatched_share": round(len(unmatched) / len(rw), 3),
            "unmatched_by_year": {int(k): int(v) for k, v in years.items()}}


def v03_window_backbone(rw: pd.DataFrame) -> dict:
    per_region = rw.groupby("region_id")["window_id"].nunique().to_dict()
    per_year = rw.groupby([rw["region_id"], rw["window_start"].dt.year])["window_id"].nunique()
    first_region = per_year.index.get_level_values(0)[0]
    return {"windows_per_region": {k: int(v) for k, v in per_region.items()},
            "windows_per_year": {int(y): int(n) for (r, y), n in per_year.items() if r == first_region},
            "same_for_both_regions": bool(per_year.unstack(0).nunique(axis=1).eq(1).all())}


def v04_delta_coverage(rw: pd.DataFrame) -> dict:
    delta = rw.groupby("region_id")["ndvi_mean"].diff()
    return {"rows": int(len(rw)), "nan_after_first_difference": int(delta.isna().sum())}


def v05_cloud_cover(df: pd.DataFrame) -> dict:
    values = df["cloud_cover_pct"].unique().tolist()
    return {"distinct_values": values[:5], "rows_equal_minus_one": int((df["cloud_cover_pct"] == -1).sum()),
            "rows": int(len(df))}


def v06_index_ranges(df: pd.DataFrame) -> dict:
    return {
        "evi_min_below_minus_one": int((df["evi_min"] < -1).sum()),
        "evi_max_above_one": int((df["evi_max"] > 1).sum()),
        "evi_min_lowest_value": float(df["evi_min"].min()),
        "ndvi_min_exactly_minus_one": int((df["ndvi_min"] == -1).sum()),
        "ndvi_min_below_minus_one": int((df["ndvi_min"] < -1).sum()),
        "rows": int(len(df)),
    }


def normalised_difference(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    denom = a + b
    out = (a - b) / denom.where(denom.abs() > 1e-6)
    return out, denom


def v07_ndbi_ndwi(rw: pd.DataFrame) -> dict:
    ndbi, d_ndbi = normalised_difference(rw["band_b11_mean"], rw["band_b08_mean"])
    ndwi, d_ndwi = normalised_difference(rw["band_b03_mean"], rw["band_b08_mean"])
    return {
        "ndbi_range": [round(float(ndbi.min()), 4), round(float(ndbi.max()), 4)],
        "ndwi_range": [round(float(ndwi.min()), 4), round(float(ndwi.max()), 4)],
        "ndbi_min_abs_denominator": round(float(d_ndbi.abs().min()), 4),
        "ndwi_min_abs_denominator": round(float(d_ndwi.abs().min()), 4),
        "nan_values": int(ndbi.isna().sum() + ndwi.isna().sum()),
        "outside_minus_one_to_one": int(((ndbi.abs() > 1) | (ndwi.abs() > 1)).sum()),
    }


def v08_snapshot_identity(master: pd.DataFrame, gold: pd.DataFrame) -> dict:
    return {
        "master_dataset_columns": int(master.attrs["raw_shape"][1]),
        "gold_candidate_columns": int(gold.attrs["raw_shape"][1]),
        "master_cloud_cover_range": [float(master["cloud_cover_pct"].min()), float(master["cloud_cover_pct"].max())],
        "gold_cloud_cover_range": [round(float(gold["cloud_cover_pct"].min()), 1),
                                   round(float(gold["cloud_cover_pct"].max()), 1)],
    }


def extra_rainfall_aggregation(df: pd.DataFrame) -> dict:
    return {"aggregation_method_values": df["aggregation_method"].value_counts().to_dict()}


def extra_yield_disagreement(master: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Compare the annual yield each snapshot holds for the same region, crop and year."""
    m = (master.dropna(subset=["yield_tonnes_per_ha"])
         .groupby(["region_id", "crop_type", "season_year"])["yield_tonnes_per_ha"].first())
    g = gold.groupby(["region", "crop_type", "year"])["gov_yield_tonnes_per_ha"].first()
    g.index.names = m.index.names
    both = pd.concat({"master": m, "gold": g}, axis=1).dropna()
    both["diff"] = (both["gold"] - both["master"]).round(3)
    off = both[both["diff"].abs() > 0.01]
    return {
        "seasons_compared": int(len(both)),
        "seasons_that_differ": [
            {"region": r, "crop": c, "year": int(y), "master": round(row["master"], 3),
             "gold": round(row["gold"], 3)} for (r, c, y), row in off.iterrows()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(args.data_dir)
    gold = load_gold(args.data_dir)
    rw = region_window_table(master)

    report = {
        "V-01 satellite values identical across crops": v01_crop_identity(master),
        "V-02 year-over-year match within +/-10 days": v02_yoy_coverage(rw),
        "V-03 window backbone": v03_window_backbone(rw),
        "V-04 window-over-window delta coverage": v04_delta_coverage(rw),
        "V-05 cloud_cover_pct placeholder": v05_cloud_cover(master),
        "V-06 EVI and NDVI minimum ranges": v06_index_ranges(master),
        "V-07 NDBI and NDWI": v07_ndbi_ndwi(rw),
        "V-08 which 580-row snapshot": v08_snapshot_identity(master, gold),
        "extra: weather aggregation method": extra_rainfall_aggregation(master),
        "extra: yield values across snapshots": extra_yield_disagreement(master, gold),
    }
    (args.out_dir / "land_use_feature_checks.json").write_text(json.dumps(report, indent=2, default=str))

    for name, result in report.items():
        print(f"{name}: {json.dumps(result, default=str)}")
    print(f"wrote {args.out_dir / 'land_use_feature_checks.json'}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
