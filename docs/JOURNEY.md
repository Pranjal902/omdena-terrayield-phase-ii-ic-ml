# How the work unfolded

What I worked on, what I checked, and why each step led to the next one.

## Starting with no data

The plan for TerraYield Analytics Phase II was to join Sentinel-2 imagery, weather, grain prices and USDA reports into one time-series table, then build two benchmarks on it: crop-yield regression and land-use change.

For the first few weeks the ML team had almost nothing to work with. The data team was waiting on Google Earth Engine and AWS S3 credentials. I used the time to read up on remote sensing: how Sentinel-2 bands are stored, why reflectance needs cloud masking, what NDVI, SAVI and NDWI respond to, and why latitude and longitude in degrees can't be used to measure area.

## Sprint 1: what does one training row mean?

When the interim joined table (`master_dataset.csv`) became available, I wrote a one-page [primer](../workshops/gis-primer.md) for anyone on the team who hadn't worked with GIS. It's built around one idea: once the bands are numbers in a table, the rest is feature engineering.

A teammate and I had shortlisted the candidate targets, and yield in tonnes per hectare was the obvious one. I then took on the question underneath it: how each label gets built, which row it lands on, and when it's safe to use. I started with the 72 rows that carried a yield value, and the questions led into each other:

- **How many independent labels were there?** 24. Each annual value was copied onto three window rows, so scoring per row would have tripled the apparent sample size. The season had to be the unit.
- **Which windows carried the label?** Every one started in December or January. The label sat on post-harvest winter rows, and the June to September windows that grew the crop had no label at all. Any model reading features and label off the same row would learn from bare fields. I proposed storing the label separately and joining it to a growing-season aggregate.
- **Did corn and soybean rows actually differ?** Not in any satellite or weather column, across all 290 region-window pairs. So the features were regional, cross-validation had to be grouped, and we had to say upfront that nothing in the inputs was crop-specific.
- **Could `production_tonnes` back up yield?** No. Production / (yield x area) came out at about 5x for corn and 12x for soybean, and it drifted by year, so it wasn't a unit slip. I kept it out.
- **What was the land-use label?** There wasn't one. I proposed the USDA Cropland Data Layer as the source, a short class list, and continuous crop-share regression rather than classification, because the dominant crop in a state almost never changes from one year to the next.

That became the [label write-up](yield-and-land-use-labels.md).

## Getting the team on the same page

Writing that document showed me two things. People on the team were using "target", "label" and "feature" in slightly different ways, and most of the evaluation risks came from how the data was structured, not from model choice. So I put together a workshop that walked from the data team's handoff to an agreed ML specification: problem framing, the canonical row, where labels come from and where they break, spatial leakage, and the open decisions that were blocking both benchmarks.

Just before Sprint 3 modelling started, I ran a second session on crop-yield benchmarking: one row per field-season, features only if they exist before the forecast cut-off, baselines before boosting, temporal and spatial splits, MAE as the primary metric. Both decks are in [workshops/](../workshops/).

Around the same time, the data team asked whether ML could start Sprint 3 on sample data while the full dataset was still blocked. I agreed. It meant we could define our processes in parallel instead of waiting. I also wrote up the ML-relevant points from that sync and shared them with the rest of the ML team.

## Sprint 3: land use without its imagery

Sprint 3 didn't go to plan for land use. The official Gold release was marked not ready at the start of the sprint, so the ML team worked from a candidate version with known caveats. Midway through, the land-use input was redefined as RGB before/after image pairs, which nobody on the team could access, and a few days later the storage credentials were revoked. Crop yield had a usable dataset, so it became the priority.

What land use did have was a tabular feature table built by a teammate's pipeline, and I wrote its [feature manifest](land-use-feature-manifest.md). I split it into two layers so nobody would mistake a schema for a dataset: one documenting the table that existed, and one specifying the image-pair manifest for when the imagery arrived, with zero rows filled.

I recomputed every number in it instead of copying figures from other documents. Some of it confirmed problems already on record: `cloud_cover_pct` was -1 on all 580 rows, EVI was out of range in 574 of them (as low as about -17 million), and rainfall was aggregated with `max` instead of a sum. It also turned up a few things worth raising:

- One review had attributed the 574 figure to `ndvi_min`. Its real count was 462, so a number had drifted between documents.
- The divide-by-zero guards in two index calculations never fired on real data, so they were untested code paths.
- Two different files were both being called "the 580-row dataset". They had different columns and disagreed on cloud cover.
- Weather features were in scope for land use, but the pipeline shipped none. I logged it as an open question instead of guessing which side was right.

None of these were mine to fix upstream. Recording them in one place, with the method used for each, meant they could be checked and fixed instead of being rediscovered later.

## Testing my own proposal

One question from Sprint 1 was still open: does tying the label to the growing season change what a model learns? The Gold dataset attached each yield to every window of its year, so I could finally compare June to September features against the December rows where the label used to sit.

I ran a [small season-level experiment](crop-yield-season-experiment.md) with baselines first and two ways of splitting. Growing-season features clearly beat the winter rows when a trend did the heavy lifting, but for a forecast-style split nothing beat last year's yield. While comparing the two data snapshots I also found three 2022 yields that disagree between them.

## Building the crop-yield models

That experiment used one fixed setting. To see how much a well-built model could add, I did a full [model-development pass](crop-yield-model-development.md) on the crop-yield side.

I started by reading how the team's benchmark was set up. Its models predicted the raw yield level, which is mostly "corn or soybean" plus a slow upward trend, so the simple baselines had already captured most of it. I switched the target to each season's gap from its own trend. I built nine agronomic features (peak NDVI and its timing, green-up rate, growing degree days, heat, rain, radiation) from windows completed before a cut-off date, and tuned every model in an inner loop that never saw the test year.

What came out of it:

- Detrending did more than any choice of model. The same random forest went from 0.407 to 0.315 t/ha.
- Tuning inside the training years pulled the models back towards the trend. It chose heavy regularisation in most folds, and the best model beat the trend by about 4%.
- In-season forecasts only pulled clearly ahead of the trend from 1 September, once the canopy peak was in the data.
- The biggest miss was Iowa corn in 2020, the year of the August derecho. Sudden damage like that doesn't show in a statewide average.
- The same tuning corrected one of my own numbers: the first experiment's 0.265 became 0.294. A small change, but I now check for it first.

## What I'd carry into the next project

- Look at where the label physically sits before thinking about models.
- Count independent observations, not rows.
- Remove the trend before asking a model to explain the rest.
- Tune inside the training data only, even when it makes the result look worse.
- Recompute a number before repeating it.
- Start with a baseline that is hard to beat. Here that was last year's yield.
