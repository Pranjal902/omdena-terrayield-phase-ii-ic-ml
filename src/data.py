"""Loading helpers shared by the scripts in this folder.

Two snapshots of the project data are used:
  master_dataset.csv            580 rows x 154 cols, the interim table from Sprint 1
  Gold_Dataset_Candidate_v1.csv 580 rows x 81 cols, the Gold candidate handed to ML in Sprint 3
Neither file is committed here. See data/README.md.
"""
import hashlib
from pathlib import Path

import pandas as pd

MASTER_FILE = "master_dataset.csv"
GOLD_FILE = "Gold_Dataset_Candidate_v1.csv"
MASTER_SHA256 = "5b3e8a1c956e834c2670777fccff9372773299b2105973bc5117f46a2ffc03f7"

# Column families as they appear in master_dataset.csv
SATELLITE_PREFIXES = ("band_", "ndvi_", "evi_", "savi_")
SATELLITE_EXTRA = ("valid_pixel_count", "cloud_cover_pct")
WEATHER_COLS = ("humidity_pct", "wind_speed_kmh", "temperature_min_c", "temperature_max_c", "rainfall_mm")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def satellite_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SATELLITE_PREFIXES) or c in SATELLITE_EXTRA]


def load_master(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / MASTER_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for where it comes from.")
    df = pd.read_csv(path)
    df.attrs["raw_shape"] = df.shape
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])
    # the annual label's crop year, taken from the report's own reporting period
    season_year = pd.to_datetime(df["reporting_period_start"]).dt.year.astype("Int64").rename("season_year")
    out = pd.concat([df, season_year], axis=1)
    out.attrs["raw_shape"] = df.attrs["raw_shape"]
    return out


def load_gold(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / GOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for where it comes from.")
    df = pd.read_csv(path)
    df.attrs["raw_shape"] = df.shape
    # dates in this file come in two formats (01-01-2020 and 1/17/2020)
    df["window_start"] = pd.to_datetime(df["window_start"], format="mixed")
    df["window_end"] = pd.to_datetime(df["window_end"], format="mixed")
    return df
