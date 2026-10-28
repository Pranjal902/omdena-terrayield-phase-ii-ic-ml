# Workshops

I prepared and ran two sessions for the TerraYield Analytics Phase II team. Both came out of the same observation: most of the risk in this project sat in how the data was structured and how results would be judged, not in which model we picked.

## 1. From canonical data to a benchmark model

[`ml-problem-framing.pdf`](ml-problem-framing.pdf), 20 slides

This was for the whole ML team, after the label work in Sprint 1. It walks from what the data team handed over to an ML specification we could agree on:

- what was actually usable in the interim table (580 rows, but 24 independent yield labels)
- what was missing: a forecast cut-off field, a crop mask year, any land-use label, crop-level attribution
- task shapes (classification, regression, detection, segmentation, forecasting) and which two we were solving
- features versus ground truth, and three worked examples from our own data: crop stress vs harvest, rotation vs conversion, and a mixed cloudy pixel with no clean label
- CRS, raster vs vector, and why spatial autocorrelation makes random splits optimistic
- six ways a benchmark can look better than it is
- seven open decisions blocking the benchmarks, none of them modelling questions

## 2. Crop yield benchmarking: features, baselines and evaluation

[`crop-yield-benchmarking.pdf`](crop-yield-benchmarking.pdf), 19 slides

Run just before Sprint 3 modelling started, so everyone building crop-yield models would use the same rules:

- target vs label vs feature, and the contract for a yield label (unit, season, source)
- what Sentinel-2 can and can't tell you about harvested yield
- season-level feature engineering (peak NDVI, timing, green-up and senescence slopes)
- preprocessing as part of the method: fit imputers and scalers on training data only
- leakage, and random vs temporal vs spatial splits
- the baseline ladder: mean, linear, tree ensembles, all on the same split and metrics
- MAE as the primary metric, RMSE and R² as supporting ones

The model comparison numbers in this deck are illustrative teaching examples, not project results.

## GIS primer

[`gis-primer.md`](gis-primer.md) is a one-page note I wrote early in the project for teammates who hadn't worked with satellite data.
