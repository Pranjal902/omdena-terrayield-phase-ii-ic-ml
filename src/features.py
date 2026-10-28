"""Season-level crop features from the 16-day window series, with an optional forecast cut-off.

Each season (region, crop, year) is summarised from the windows that start in April or
later and end before the cut-off date. With no cut-off the season runs to 30 September.
Only windows that are complete by the cut-off are used, so a feature never sees data
from after the date a forecast would be made.

Features:
  ndvi_peak        highest regional NDVI so far
  ndvi_peak_doy    day of year of that peak (window midpoint)
  ndvi_integral    sum of NDVI over the windows, a proxy for accumulated green biomass
  greenup_rate     largest window-to-window NDVI rise
  gdd              growing degree days above 10 C, from the window mean temperature
  hot_windows      windows whose maximum temperature reached 34 C or more
  tmax_avg         average of the window maximum temperatures
  precip_total     total precipitation, mm
  radiation_total  total solar radiation
"""
import pandas as pd

KEY = ["region", "crop_type", "year"]
FEATURES = ["ndvi_peak", "ndvi_peak_doy", "ndvi_integral", "greenup_rate", "gdd",
            "hot_windows", "tmax_avg", "precip_total", "radiation_total"]
SEASON_START_MONTH = 4
GDD_BASE_C = 10.0
HOT_TMAX_C = 34.0


def _season_summary(w: pd.DataFrame) -> pd.Series:
    w = w.sort_values("window_start")
    days = (w["window_end"] - w["window_start"]).dt.days + 1
    mid = w["window_start"] + (w["window_end"] - w["window_start"]) / 2
    peak = w["ndvi_mean"].idxmax()
    rise = w["ndvi_mean"].diff().max()
    return pd.Series({
        "ndvi_peak": w["ndvi_mean"].max(),
        "ndvi_peak_doy": mid.loc[peak].dayofyear,
        "ndvi_integral": w["ndvi_mean"].sum(),
        "greenup_rate": rise if pd.notna(rise) else 0.0,
        "gdd": float(((w["t2m_mean"] - GDD_BASE_C).clip(lower=0) * days).sum()),
        "hot_windows": float((w["t2m_max"] >= HOT_TMAX_C).sum()),
        "tmax_avg": w["t2m_max"].mean(),
        "precip_total": w["tp_sum"].sum(),
        "radiation_total": w["ssrd_sum"].sum(),
        "windows": len(w),
    })


def season_features(gold: pd.DataFrame, cutoff_month: int = 10) -> pd.DataFrame:
    """One row per season. cutoff_month=8 means 'what was known on 1 August'."""
    in_year = gold["window_start"].dt.year == gold["year"]
    started = gold["window_start"].dt.month >= SEASON_START_MONTH
    cutoff = pd.to_datetime(gold["year"].astype(str) + f"-{cutoff_month:02d}-01")
    complete = gold["window_end"] < cutoff
    windows = gold[in_year & started & complete]
    feats = windows.groupby(KEY).apply(_season_summary, include_groups=False)
    target = gold.groupby(KEY)["gov_yield_tonnes_per_ha"].first().rename("yield_t_ha")
    out = feats.join(target).reset_index()
    out["windows"] = out["windows"].astype(int)
    return out[KEY + FEATURES + ["windows", "yield_t_ha"]]


def describe_cutoffs(gold: pd.DataFrame, months=(7, 8, 9, 10)) -> dict:
    """How many windows each cut-off leaves per season (min, max)."""
    out = {}
    for m in months:
        n = season_features(gold, m)["windows"]
        out[m] = [int(n.min()), int(n.max())]
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from data import load_gold

    g = load_gold(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
    print(season_features(g).round(3).to_string())
    print(describe_cutoffs(g))
