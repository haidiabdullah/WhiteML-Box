# Example data for WhiteML-Box

This folder contains the example data used for the WhiteML-Box SoftwareX demonstration.

## Files

### `Spectra_Nitrogen.xlsx`

Leaf-level spectral dataset with:

- one worksheet: `Sheet1`,
- 66 data samples,
- 1948 numeric wavelength predictor columns from 400 to 2347 nm,
- one response column: `Nitrogen`.

This file can be loaded directly in the **Data** tab. Select `Nitrogen` as the target variable and the numeric wavelength columns as predictors.

### `WhiteML-Box.tif`

Example 12-band GeoTIFF raster stack for the mapping demonstration.

Raster summary:

- driver: GeoTIFF,
- size: 1230 columns × 1050 rows,
- bands: 12,
- coordinate reference system: EPSG:32633,
- pixel size: 10 m,
- data type: unsigned 16-bit integer,
- compression: lossless DEFLATE compression, values unchanged from the uploaded raster.

The raster is intended as a demonstration input for the raster-prediction workflow. The GeoTIFF band order must match the trained model predictor order. For the Sentinel-2-resampled example, the expected band-centre order is:

`443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190 nm`

## Critical warning

Do not treat the example map as an independently validated ecological product. It is provided to demonstrate software functionality. For scientific use, prepare raster data with matching band order, reflectance scaling, preprocessing assumptions, and independent validation data.
