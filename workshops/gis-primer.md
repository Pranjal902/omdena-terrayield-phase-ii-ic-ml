# GIS in one page

*A short note I wrote early in the project for teammates who hadn't worked with satellite data before.*

It's called remote sensing because the data is collected by sensing something from a distance, without touching it. Remote means far away; sensing means detecting or measuring.

For our purposes, GIS data ends up much like a normal tabular dataset. The only difference is the starting point: we begin with an image, turn it into bands, turn the bands into spectral features, and from there it's ordinary feature engineering.

```
GeoTIFF (.tif) from cloud storage
        |
        v
Load the raster (rasterio / GDAL)
    with rasterio.open("image.tif") as src:
        blue  = src.read(2)   # B02
        green = src.read(3)   # B03
        red   = src.read(4)   # B04
        nir   = src.read(8)   # B08
        |
        v
Preprocess
    cloud masking, clip to the area of interest,
    reproject the CRS, resample, handle NoData
        |
        v
Spectral bands
    B2 blue, B3 green, B4 red, B8 NIR, B11 SWIR1, B12 SWIR2
        |
        v
Spectral features
    NDVI, EVI, SAVI, NDWI and other indices
        |
        v
Training samples
    overlay field boundaries or labels, sample pixel or polygon values,
    attach the target (crop type, yield, land use)
        |
        v
Tabular dataset
    | B2 | B3 | B4 | B8 | NDVI | EVI | label |
        |
        v
Model
    XGBoost, random forest, LightGBM, CatBoost
    (a CNN or ViT if you work on image patches instead of tabular features)
```
