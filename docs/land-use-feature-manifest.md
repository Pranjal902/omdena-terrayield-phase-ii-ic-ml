# Land-use feature manifest

*Written during Sprint 3, cleaned up for this repo. The verification log below is reproduced by `src/verify_land_use_features.py`; the full column register is in [`artifacts/land_use_feature_register.csv`](../artifacts/land_use_feature_register.csv).*

## Why it has two layers

Late in Sprint 3 the land-use task had been redefined around raw RGB before/after image pairs, and nobody on the team had access to them. What did exist was a tabular feature table built by a teammate's pipeline: one row per (region, window), 290 rows, derived from `master_dataset.csv`.

So I wrote the manifest in two layers and kept them clearly apart:

- **Layer A** documents the feature table that existed. Real, inspectable, reproducible.
- **Layer B** specifies what the RGB pair manifest would need to contain once access arrived. It is a schema with zero rows, because no imagery, label source or spatial unit had been supplied.

I didn't want either one mistaken for the other, or a schema mistaken for a dataset.

## Layer A: the feature table that existed

Grain: one row per (region, window). Only US-Iowa and US-Illinois existed, 145 sixteen-day windows each.

| Family | Columns | What it is | Main caveat |
|---|---|---|---|
| Keys | `region_id`, `window_id`, `window_start` / `window_end`, `dataset_release_id` | join and ordering keys, run provenance | no sub-state geometry anywhere, so no pixel or parcel unit is reachable |
| Quality | `valid_pixel_count` | per-window count of usable pixels | the only quality signal left, since `cloud_cover_pct` is dead (V-05) |
| Indices | `ndvi_mean`, `savi_mean`, `ndbi_mean`, `ndwi_mean` | regional means; NDBI and NDWI derived from bands B11/B08 and B03/B08 | identical for corn and soybean (V-01), so no crop-specific signal |
| Change | `*_delta` | difference from the previous window | NaN for each region's first window (2 of 290) |
| Rolling | `*_rolling_mean`, `*_rolling_std`, `*_rolling_z` | trailing window (default 4) | early rows partial; divide-by-zero handling for z not documented |
| Seasonal | `*_yoy_delta` | difference from the window ~365 days earlier (±10 days) | NaN for all of 2020 (46 of 290 rows) because there is no prior year (V-02) |
| Spatial | `cross_region_*_delta` | this region minus the other one | with two regions it's a mirrored raw difference; weak as a spatial feature |
| Anomaly | `change_candidate_score`, `change_candidate_flag` | YoY delta in units of the region's own YoY spread; flag at z = 2.0 | **not a land-use label**. Proxy only, cannot be validated, threshold not tuned on anything |
| Deferred | `ndre_mean`, `ndvi_b8a_mean` | red-edge NDVI and a narrow-NIR NDVI cross-check | scoped but held back so the reviewed column set wasn't changed mid-sprint |

### What was left out, and why

| Input | Decision | Reason |
|---|---|---|
| `evi_*` | excluded (defect) | unclamped denominator: `evi_min` below -1 in 574 of 580 rows, down to about -17 million |
| `cloud_cover_pct` | excluded (dead field) | -1 in all 580 rows; a placeholder, never a measurement |
| `rainfall_mm`, `humidity_pct` | excluded (aggregation bug) | aggregated with `max` where a total (rainfall) and a mean (humidity) are needed. Fixing it quietly downstream would hide the upstream bug |
| temperature and wind | open | in scope according to the feature scope, but the pipeline shipped no weather columns. I recorded the gap rather than guessing which side was right |
| geometry, neighbour features | blocked | no geometry populated and no raster retained |
| per-pixel texture, object shape | blocked | needs raw raster access |
| land-use class or change label | blocked | none exists in any source column |
| share-regression targets | deferred | task type (binary change vs share regression) still undecided |
| post-harvest measurements | excluded | leakage, same rule as the crop-yield track |

### How the pipeline handled bad inputs

The pipeline was non-silent: it wrote NaN or failed hard instead of imputing quietly, and it aborted if corn and soybean rows ever disagreed on satellite values. But it carried no flag columns, so a downstream user couldn't tell a "no prior year" NaN from a "near-zero denominator" NaN. The flag vocabulary existed in a separate quality policy; the two hadn't been joined up yet. I recorded the gap here because the manifest is where a consumer would look for it.

## Layer B: RGB pair schema (specification only, 0 rows)

| Field | Type | Rule before handoff | Status |
|---|---|---|---|
| `pair_id` | string | unique; 1:1 with the label record | proposed |
| `image_before_uri`, `image_after_uri` | URI | resolve against the approved store; never embed credentials; not identical | waiting on imagery |
| `date_before`, `date_after` | date | observation dates, `date_before < date_after` enforced | minimum gap not set |
| `available_at_before/after` | datetime | release timestamps kept separate from observation dates | proposed |
| `spatial_unit_id` | string | consistent with the declared label unit | blocked: unit undecided |
| `geographic_group` | string | split key; overlapping patches and shared scenes share a group; no group in two splits | proposed |
| `geometry_ref`, CRS, resolution | mixed | both images share footprint and CRS | waiting on metadata |
| `label` | {0, 1, null} | null is never silently turned into 0; every null has a reason code | blocked: no label source |
| `class_before`, `class_after` | string | mapped through a frozen, versioned crosswalk | blocked: no taxonomy |
| `quality_flags`, `reason_codes` | list | members of an agreed vocabulary | vocabularies not merged |
| `eligibility_status` | enum | ineligible pairs never enter the test set | proposed |
| `valid_coverage_fraction` | float | meets an agreed minimum | threshold not set |
| `dataset_version`, `pipeline_commit`, `created_at` | string / datetime | resolve to one immutable release and commit | proposed |

## Verification log

Every figure above was recomputed from `master_dataset.csv` rather than copied from another document.

| ID | Check | Result | What it means |
|---|---|---|---|
| V-01 | Satellite fields identical across corn and soybean for each (region, window) | 290 groups, 0 differing cells | collapsing 580 crop rows to 290 region-window rows loses nothing, and the features carry no crop signal |
| V-02 | Year-over-year match within ±10 days | 46 of 290 unmatched (15.9%), all in 2020 | not a tolerance problem; the first year has no predecessor |
| V-03 | Window backbone | 145 windows per region; 23 a year for 2020-24, 22 in 2025, 8 in 2026 | gap-free and symmetric; 2026 is partial and shouldn't be treated as a season |
| V-04 | Window-over-window delta coverage | 2 of 290 NaN | as designed, one per region |
| V-05 | `cloud_cover_pct` | -1 in 580 of 580 rows | cannot be a cloud filter |
| V-06 | EVI and NDVI minimum ranges | `evi_min` < -1 in 574/580; `evi_max` > 1 in 580/580; `ndvi_min` = -1 in 462/580, never below | EVI exclusion justified. Another review had attributed 574/580 to `ndvi_min`; on this file that's the EVI count, so the figure needed correcting before being quoted again |
| V-07 | NDBI and NDWI | NDBI -0.80 to 0.24, NDWI -0.82 to 0.01, all inside [-1, 1]; smallest denominators 0.45 and 0.27 | both indices are sound, but the divide-by-zero guard never fires on real data. It's an untested code path that needs a synthetic unit test |
| V-08 | Which "580-row dataset" | `master_dataset.csv`: 154 columns, cloud cover -1. Gold candidate: 81 columns, cloud cover 14 to 100 | two different files, not two readings of one. Figures from one shouldn't be quoted against the other |

## Decisions the manifest was waiting on

1. Spatial unit for the image task (pixel, patch, polygon or admin unit). This blocked the whole RGB manifest.
2. Label source: the imagery's own labels, CDL, or documented manual annotation.
3. Task type: binary change vs share regression.
4. Canonical time key: `window_id` or date.
5. One merged vocabulary for quality flags and exclusion codes.
6. Every unset threshold (coverage, alignment tolerance, minimum time gap), frozen on development data, not on test results.
7. Whether weather features belong in the land-use set.
8. Which 580-row file is the reference for land-use work.

None of these were mine to close. The manifest recorded what each one blocked, so the decision makers could see the cost of leaving it open.
