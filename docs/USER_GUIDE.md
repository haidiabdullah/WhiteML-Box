# WhiteML-Box User Guide

This guide describes the practical workflow for using WhiteML-Box for spectral vegetation-trait modelling and mapping.

## 1. Prepare the input table

Use an Excel or CSV file where rows are samples and columns are variables. One column must contain the measured response trait, such as `Nitrogen`, and numeric wavelength or reflectance columns must contain the spectral predictors.

For wavelength-dependent operations, numeric predictor column names should represent wavelengths, for example `400`, `401`, `402`, and so on. If column names are not true wavelengths, provide a separate original band-centre file before using sensor conversion.

## 2. Load data

Open the **Data** tab, load the Excel/CSV file, select the worksheet, choose the response variable, and select the spectral predictor columns.

## 3. Optional sensor band-centre conversion

Use this when you want to harmonise spectra to another sensor configuration, such as Sentinel-2 MSI. Select the target sensor and preview the converted band centres before modelling.

## 4. Spectral preprocessing

Start with raw reflectance as a baseline. Then test preprocessing options such as SNV, MSC, absorbance, Savitzky-Golay smoothing, derivatives, continuum removal, spectral indices, or continuous wavelet features. Only keep a more complex option if it improves cross-validated performance and is scientifically defensible.

## 5. Noisy-band inspection

Use the spectrum plot to identify and mark noisy wavelength ranges. Do not remove bands casually; document any removed wavelength regions.

## 6. Outlier screening

WhiteML-Box can flag outliers using response-variable robust z-score, PCA Hotelling T², Isolation Forest, or combined screening. Flagged points should be reviewed. Do not automatically remove samples unless there is a defensible measurement or sampling reason.

## 7. Model fitting and validation

PLSR is recommended as the first baseline for high-dimensional spectral data. The software also supports GPR, RF, Extra Trees, SVR, Gradient Boosting, XGBoost, and KNN. Use cross-validation rather than relying on training accuracy.

For small datasets, prefer K-fold, repeated K-fold, or leave-one-out. For stronger claims, use nested or grouped validation when the sampling design contains repeated trees, plots, dates, or sites.

## 8. Interpret outputs

Review measured-versus-predicted plots, residuals, R², adjusted R², RMSE, MAE, bias, RPD, RPIQ, and feature-importance outputs. For PLSR, inspect the component scan and VIP scores.

## 9. Raster prediction

Open the **Mapping** tab only after a model has been successfully trained. Load a multiband GeoTIFF whose band order and scaling match the trained model input. Export the predicted trait map and inspect the preview.

## 10. Export and reporting

Export metrics, plots, predictions, feature importance, run configuration, and mapping outputs. Report preprocessing, validation design, model settings, and raster-band assumptions clearly in any paper or report.
