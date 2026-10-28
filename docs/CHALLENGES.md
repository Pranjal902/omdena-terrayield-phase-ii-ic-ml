# Challenges and how I handled them

**Weeks without data.** Credentials for Earth Engine and S3 took weeks to arrive, so the ML team had nothing to model for much of the first half. I used the time to learn the remote-sensing side and wrote a GIS primer for the team. When the data team proposed that ML start Sprint 3 on sample data, I agreed, so we could settle our processes while the full dataset was still blocked.

**A label in the wrong place.** The yield label sat on post-harvest winter windows. The fix belonged in how the table is built, not in any model: store the label as its own record and join it to growing-season features. I wrote it up with the evidence, so it could be checked rather than taken on trust.

**No land-use label at all.** There was nothing to supervise against. I proposed a source (the Cropland Data Layer), a short class list and a target framing that would hold up with very few changes per year. I also made sure the one anomaly score in the feature table was labelled as a proxy, so nobody would treat it as ground truth.

**The task changed mid-sprint.** When land use was redefined around image pairs nobody could access, I wrote the manifest in two layers: what existed, and a schema ready for when the imagery arrived. That way the work didn't stall and didn't pretend to be more than it was.

**One name, two files.** Different documents described "the 580-row dataset" and meant different files. I recomputed every figure on the file the pipeline actually read, and flagged the question of which snapshot was the reference as an open decision.

**Only 24 real observations.** Small samples make almost any model look good on R², and they make chance correlations look real. I scored per season, grouped the folds and put baselines first. In the modelling work I went further:

- compared two split strategies
- detrended the target so models weren't rewarded for simply knowing the crop
- tuned every hyperparameter inside the training years only

That last step lowered one of my own earlier results. I'd rather find that myself than have someone else find it.

**Features that couldn't tell the crops apart.** Satellite and weather values were identical for corn and soybean in every region and window. I couldn't fix that without a crop mask, so I made sure it was stated upfront in every write-up instead of being found after a model was trained.
