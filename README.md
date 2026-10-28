# TerraYield Analytics Phase II: AI for Land Use and Crop Yield Prediction — ML Work

My ML work from **[TerraYield Analytics Phase II: AI for Land Use and Crop Yield Prediction](https://www.omdena.com/projects/terrayield-2-ai-analytics-for-land-use-and-crop-yield-prediction)**, an 8-week Omdena AI Innovation Challenge (July to September 2026), where I worked as an ML Engineer on the 10-person ML team.

## The problem

Satellite imagery, weather, grain prices and USDA yield reports are all public, but they come from different sources, in different formats and at different time steps. The challenge was to join them into one time-series dataset for Iowa and Illinois, then use it to predict corn and soybean yield and to detect land-use change.

On the ML side, the hardest question came before any model: what does one training row mean when the label arrives once a year, the features arrive every 16 days, and only 24 real yield observations exist?

## My role

I worked on how the labels for both tracks should be built, on the land-use feature set, and on getting the team aligned before modelling started. I also built the crop-yield models myself, to test my own label proposals against real results.

- **Label strategy**: where the yield and land-use labels come from, which row each one lands on, and when it's safe to use
- **Two workshops**: one on framing the ML problem, one on crop-yield benchmarking
- **Land-use feature manifest**: a column-by-column register of the land-use feature table, with every figure recomputed from the data
- **Crop-yield modelling**: a growing-season experiment, then a full model-development pass with nested tuning and in-season forecasts

## Where this fit into the larger project

The project ran four teams in parallel.

| Team | Focus |
|---|---|
| Team 1 — OP | Operations, documentation, DevOps, QA |
| Team 2 — DE | Data governance, ingestion, Gold dataset |
| Team 3 — ML | Geospatial science, benchmarking, modelling *(mine)* |
| Team 4 — FE | Dashboard and dataset explorer |

The ML team had two tracks: crop yield and land use. Data access problems (Earth Engine and S3 credentials, a delayed Gold dataset, imagery that never arrived) cost most of the first half of the project, so the team put crop yield first. Land use was handed over as a documented future track.

The team delivered a Gold dataset candidate, a crop-yield benchmark and a working dashboard MVP (Next.js on a FastAPI backend). The dashboard ran locally; we didn't have the budget or infrastructure to host it on a public link.

## What I found

![Where the yield label sat](results/label_windows_vs_ndvi.png)

*The annual yield label (red) sat on post-harvest winter rows and never touched the June to September growing season (green).*

1. **The yield label was on the wrong rows.** All 72 labelled rows started in December or January, after harvest. A model reading features and label off the same row would have learned from bare fields. I proposed joining each yield to a growing-season aggregate instead ([label write-up](docs/yield-and-land-use-labels.md)).
2. **72 labelled rows were really 24 observations.** Each annual yield was copied onto three windows. Scoring per row would have tripled the apparent sample size, so the season became the unit for training and evaluation.
3. **Corn and soybean had identical inputs.** None of the 290 region-window pairs differed in any satellite or weather column. The features were regional, so cross-validation folds had to be grouped and nothing could be claimed as crop-specific.
4. **Land use had no label.** None of the 154 columns could serve as one. I proposed the USDA Cropland Data Layer as the source, a short class list, and crop-share regression instead of classification.
5. **I re-checked the data instead of trusting the documents.** While building the land-use feature manifest I recomputed every figure myself. That confirmed defects already on record (cloud cover a -1 placeholder on all 580 rows, EVI out of range on 574, rainfall aggregated with `max`) and turned up a few more worth raising:
   - a figure that had drifted between documents: 574 was quoted for `ndvi_min`, whose real count was 462
   - divide-by-zero guards in two index calculations that never ran on real data
   - two different files both being called "the 580-row dataset"
   - weather features that were in scope for land use but missing from the pipeline output

   ([feature manifest](docs/land-use-feature-manifest.md))
6. **Growing-season features beat the winter rows.** I tested my own proposal: June to September features beat the old December rows in three of four comparisons ([experiment](docs/crop-yield-season-experiment.md)).
7. **Detrending mattered more than model choice.** In the model-development pass, the same random forest dropped from 0.407 to 0.315 t/ha MAE just by predicting the gap to trend instead of raw yield ([model development](docs/crop-yield-model-development.md)):
   - with nested tuning, the best retrospective model beat the trend by about 4% (0.314 vs 0.327)
   - in the forecast-style split, in-season forecasts only pulled clearly ahead of the trend from 1 September
   - for a forecast, last year's yield (0.261) still beat every model, which matches what the team found

## What's in this repo

| Folder | What it is |
|---|---|
| [`src/`](src) | `audit_yield_label.py` (findings 1-4), `verify_land_use_features.py` (finding 5), `season_yield_baselines.py` (finding 6), `features.py` and `yield_model_development.py` (finding 7) |
| [`notebooks/`](notebooks) | Both crop-yield notebooks, run end to end with outputs |
| [`docs/`](docs) | The label strategy, feature manifest and both crop-yield write-ups, plus [`JOURNEY.md`](docs/JOURNEY.md) (how the work moved) and [`CHALLENGES.md`](docs/CHALLENGES.md) (what got in the way) |
| [`results/`](results) | Every JSON, CSV and chart the scripts write |
| [`artifacts/`](artifacts) | The land-use feature register, all 37 columns |
| [`workshops/`](workshops) | Both workshop decks as PDFs, plus the GIS primer I wrote for the team |
| [`data/`](data) | Which files the scripts expect, and why they aren't in git |

Outputs from my own runs are already in `results/`, so you can read everything without the data.

## Setup

Python 3.11 or newer, from the repo root. No API keys are needed.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The project data isn't redistributed here. Put the two CSVs in `data/` (details in [`data/README.md`](data/README.md)).

## Run (from repo root)

```bash
python src/audit_yield_label.py           # label checks          -> results/label_audit.json
python src/verify_land_use_features.py    # manifest checks       -> results/land_use_feature_checks.json
python src/season_yield_baselines.py      # growing-season test   -> results/season_experiment_*
python src/yield_model_development.py     # model development     -> results/model_dev_*  (about 2 minutes)
```

Each script accepts `--data-dir` and `--out-dir`. The notebooks read from `data/`, or from the folder in `TERRAYIELD_DATA_DIR`.

## What I'd do next

- **County-level yields.** Iowa and Illinois have about 200 counties between them. That would turn 24 seasons into more than a thousand, make spatial splits meaningful, and give deeper models something to learn from.
- **A crop mask.** It would let the satellite features differ between corn and soybean.
- **The Cropland Data Layer.** It would give land use a real label, so that track can start.

## Skills this work draws on

- Turning an annual yield label and 16-day satellite windows into a sound training row: season-level grain, label-availability rules, grouped folds for duplicated crop rows
- Evaluation that holds up on 24 seasons: leave-one-year-out and expanding-window splits, nested cross-validation, baselines before models
- Crop-yield modelling: detrended targets, Sentinel-2 phenology and weather features, in-season forecasts with cut-off dates, error analysis
- Data-quality verification: recomputing documented figures from source data and tracing discrepancies between dataset versions
- Explaining ML decisions to a mixed team through workshops and written specs
