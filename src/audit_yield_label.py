"""Reproduce the yield-label checks behind docs/yield-and-land-use-labels.md.

Runs on master_dataset.csv and answers five questions:
  1. How many labelled rows are there, and how many independent seasons do they hold?
  2. Which windows carry the label, and do any of them fall in the growing season?
  3. Does production = yield x area hold?
  4. Do corn and soybean rows actually differ in their satellite and weather values?
  5. Is there anything that could serve as a land-use label?

Usage:
    python src/audit_yield_label.py --data-dir data --out-dir results
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import MASTER_FILE, MASTER_SHA256, WEATHER_COLS, load_master, satellite_columns, sha256

GROWING_MONTHS = [6, 7, 8, 9]  # June to September
SEASON_KEY = ["region_id", "crop_type", "season_year"]


def label_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["yield_tonnes_per_ha"].notna()].copy()


def check_label_grain(lab: pd.DataFrame) -> dict:
    copies = lab.groupby(SEASON_KEY).size()
    return {
        "labelled_rows": int(len(lab)),
        "independent_seasons": int(copies.size),
        "rows_per_season": sorted(copies.unique().tolist()),
        "fill_tags": lab["temporal_fill_method_government"].value_counts().to_dict(),
        "report_types": lab["report_type"].value_counts().to_dict(),
    }


def check_label_windows(lab: pd.DataFrame) -> dict:
    months = lab["window_start"].dt.month
    by_season = lab.groupby("season_year")["window_start"].agg(["min", "max"])
    return {
        "window_start_months": {int(k): int(v) for k, v in months.value_counts().sort_index().items()},
        "labelled_rows_in_jun_sep": int(months.isin(GROWING_MONTHS).sum()),
        "label_window_span_by_season": {
            int(y): [r["min"].strftime("%Y-%m-%d"), r["max"].strftime("%Y-%m-%d")] for y, r in by_season.iterrows()
        },
    }


def check_production_identity(lab: pd.DataFrame) -> dict:
    lab = lab.dropna(subset=["production_tonnes", "area_harvested_ha"])
    ratio = lab["production_tonnes"] / (lab["yield_tonnes_per_ha"] * lab["area_harvested_ha"])
    by_crop = ratio.groupby(lab["crop_type"]).mean().round(2).to_dict()
    by_crop_year = ratio.groupby([lab["crop_type"], lab["season_year"]]).mean().round(2)
    return {
        "rows_checked": int(len(lab)),
        "mean_ratio_by_crop": by_crop,
        "ratio_range_by_crop": {
            crop: [float(s.min()), float(s.max())] for crop, s in by_crop_year.groupby(level=0)
        },
    }


def check_crop_duplication(df: pd.DataFrame) -> dict:
    cols = satellite_columns(df) + list(WEATHER_COLS)
    nunique = df.groupby(["region_id", "window_id"])[cols].nunique(dropna=False)
    return {
        "region_window_pairs": int(len(nunique)),
        "columns_compared": len(cols),
        "cells_that_differ_between_crops": int((nunique > 1).sum().sum()),
    }


def check_land_use_label(df: pd.DataFrame) -> dict:
    keys = ("land_use", "landuse", "land_cover", "landcover", "cdl", "scl", "crop_area_share", "lulc")
    hits = [c for c in df.columns if any(k in c.lower() for k in keys)]
    return {"candidate_columns": hits, "band_scl_present": "band_scl" in df.columns}


def plot_label_windows(df: pd.DataFrame, lab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, region in zip(axes, sorted(df["region_id"].unique())):
        sat = df[(df["region_id"] == region) & (df["crop_type"] == "corn")].sort_values("window_start")
        ax.plot(sat["window_start"], sat["ndvi_mean"], color="#2f6b3a", lw=1.4, label="NDVI (regional mean)")
        for year in range(sat["window_start"].dt.year.min(), sat["window_start"].dt.year.max() + 1):
            if pd.Timestamp(year, 6, 1) > sat["window_start"].max():
                continue
            ax.axvspan(pd.Timestamp(year, 6, 1), pd.Timestamp(year, 9, 30), color="#c9e4c5", alpha=0.5, lw=0)
        rows = lab[(lab["region_id"] == region) & (lab["crop_type"] == "corn")]
        marks = sat[sat["window_start"].isin(rows["window_start"])]
        ax.scatter(marks["window_start"], marks["ndvi_mean"], color="#b5442c", zorder=3, s=22,
                   label="rows carrying the annual yield label")
        ax.set_title(region, loc="left", fontsize=10)
        ax.set_ylabel("NDVI")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8, frameon=False)
    axes[1].set_xlabel("Shaded: June to September growing season")
    fig.suptitle("Where the yield label sat in the interim table", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_master(args.data_dir)
    lab = label_rows(df)
    digest = sha256(args.data_dir / MASTER_FILE)

    report = {
        "file": MASTER_FILE,
        "shape": list(df.attrs["raw_shape"]),
        "sha256": digest,
        "sha256_matches_pin": digest == MASTER_SHA256,
        "label_grain": check_label_grain(lab),
        "label_windows": check_label_windows(lab),
        "production_identity": check_production_identity(lab),
        "crop_duplication": check_crop_duplication(df),
        "land_use_label": check_land_use_label(df),
    }
    (args.out_dir / "label_audit.json").write_text(json.dumps(report, indent=2))
    plot_label_windows(df, lab, args.out_dir / "label_windows_vs_ndvi.png")

    g, w = report["label_grain"], report["label_windows"]
    p, d = report["production_identity"], report["crop_duplication"]
    rows, cols = df.attrs["raw_shape"]
    print(f"{MASTER_FILE}: {rows} rows x {cols} cols, hash matches pin: {report['sha256_matches_pin']}")
    print(f"labelled rows: {g['labelled_rows']} -> {g['independent_seasons']} independent seasons "
          f"({g['rows_per_season']} rows each, tags {g['fill_tags']})")
    print(f"label window start months: {w['window_start_months']}; rows in Jun-Sep: {w['labelled_rows_in_jun_sep']}")
    print(f"production / (yield x area), mean by crop: {p['mean_ratio_by_crop']}")
    print(f"corn vs soybean: {d['cells_that_differ_between_crops']} differing cells across "
          f"{d['region_window_pairs']} region-window pairs ({d['columns_compared']} columns)")
    print(f"land-use label candidates: {report['land_use_label']['candidate_columns']}, "
          f"band_scl present: {report['land_use_label']['band_scl_present']}")
    print(f"wrote {args.out_dir / 'label_audit.json'} and {args.out_dir / 'label_windows_vs_ndvi.png'}")


if __name__ == "__main__":
    main()
