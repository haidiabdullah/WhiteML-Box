# WhiteML-Box v0.7.2 Mapping Workflow

The `5 Mapping` tab applies the latest successfully trained regression model to a multiband GeoTIFF and exports a single-band prediction map.

## Required raster format

The input raster must be one multiband GeoTIFF. The required band stack depends on whether the model was trained with sensor band-centre conversion.

### Case 1 — sensor conversion was not used

The raster bands must match the final model X predictors exactly:

1. Same number of bands as the final model X columns after noisy-band deletion.
2. Same band order as the final model X columns.
3. Same units/scale as the reflectance or predictor columns used for training.
4. If the table used reflectance scaled 0–1, the raster bands must also be 0–1. If the table used percent reflectance, the raster must also use percent reflectance.

### Case 2 — sensor conversion was used

The raster can follow either of these two valid inputs:

1. **Target-sensor bands directly** — for example, a 12-band Sentinel-2 MSI GeoTIFF in the same band order as the trained target-sensor model.
2. **Original source X predictors after noisy-band deletion** — for example, the original hyperspectral band stack used before conversion. WhiteML-Box then interpolates each raster pixel internally to the selected target sensor band centres before prediction.

This matters. If you trained by converting hyperspectral spectra to Sentinel-2 MSI, you can now map either with the matching Sentinel-2-style band stack directly or with the original source bands for internal conversion.

If the trained model used spectral indices or continuous-wavelet features, the raster still needs only the required target-sensor/model bands or the original source bands. WhiteML-Box rebuilds the index/wavelet predictors internally before prediction.

## Workflow

1. Load Excel/CSV data.
2. Select Y and X columns.
3. Optional: enable sensor band-centre conversion and select the target sensor.
4. Apply preprocessing, spectral indices, wavelet regions, noisy-band deletion and outlier decisions as needed.
5. Run the selected model.
6. Open `5 Mapping`.
7. Load a multiband GeoTIFF.
8. Choose an export folder and output filename.
9. Click `Generate prediction map`.
10. Inspect the generated map preview.

## Common error: no mapping-ready trained model

If you see `No mapping-ready trained model is available`, the raster is not the problem. It means the app does not currently hold a fitted model for mapping. Go back to `3 Model` and click `RUN selected model`, then wait until the log says `Mapping-ready model stored`. Do not change the Y column, X columns, sensor conversion, preprocessing settings, outlier decision, model type, validation settings, or hyperparameters after training unless you plan to rerun the model.

## Outputs

The mapping step writes:

- A single-band GeoTIFF prediction map.
- A JSON sidecar file ending in `_mapping_config.json` containing the model, target, sensor conversion settings, transform, excluded ranges, index/wavelet settings and required predictor-band order.

## Warning

WhiteML-Box does not infer wavelength names from raster bands. A band-order mismatch produces an invalid map even if the app can run without errors. Prepare the raster stack carefully before prediction.
