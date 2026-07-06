# WhiteML-Box

**WhiteML-Box** is an open-source Windows desktop application for spectral machine-learning-based vegetation trait modelling and mapping. It supports tabular hyperspectral, multispectral, proximal-sensing, and laboratory reflectance datasets and can apply trained regression models to multiband GeoTIFF raster stacks.

Current version: **0.7.2**  
License: **MIT**  
Main application file: `mlbox_app.py`

## Main capabilities

- Import `.xlsx`, `.xls`, and `.csv` spectral datasets.
- Select a response variable, such as foliar nitrogen, chlorophyll, or another vegetation trait.
- Select numeric wavelength/reflectance columns as predictors.
- Inspect spectra and interactively remove noisy wavelength ranges.
- Apply spectral preprocessing and feature generation:
  - raw reflectance,
  - standard normal variate correction,
  - multiplicative scatter correction,
  - apparent absorbance / `log(1/R)`,
  - Savitzky-Golay smoothing,
  - first derivative,
  - second derivative,
  - continuum removal,
  - all-pair spectral indices,
  - continuous wavelet features.
- Perform outlier screening using robust response z-score, PCA Hotelling T², Isolation Forest, or combined screening.
- Fit regression models:
  - PLSR,
  - Gaussian Process Regression,
  - Random Forest,
  - Extra Trees,
  - Support Vector Regression,
  - Gradient Boosting,
  - XGBoost,
  - k-nearest neighbours.
- Validate models using K-fold CV, repeated K-fold CV, leave-one-out CV, or train/test split.
- Export metrics, predictions, residual plots, feature importance/VIP scores, model configuration, and mapping outputs.
- Apply trained models to multiband GeoTIFF files and export single-band vegetation-trait prediction maps.

## Repository structure

```text
WhiteML-Box/
├── mlbox_app.py                         # Main GUI application
├── requirements.txt                     # Python dependencies
├── install_and_run.bat                  # Windows first-time setup and launch script
├── run_mlbox.bat                        # Windows launcher after installation
├── build_windows_exe.bat                # Optional PyInstaller build script
├── WhiteML-Box.iss                      # Optional Inno Setup installer script
├── white_ml_box_logo.png                # GUI logo
├── data/
│   ├── Spectra_Nitrogen.xlsx            # Example spectral dataset
│   ├── WhiteML-Box.tif                  # Example 12-band GeoTIFF raster stack
│   └── README.md                        # Dataset description and cautions
├── docs/
│   ├── USER_GUIDE.md                    # Practical user guide
│   ├── MAPPING_WORKFLOW.md              # Raster mapping workflow
│   └── SOFTWAREX_METADATA.md            # SoftwareX metadata summary
├── scripts/
│   └── check_repository.py              # Lightweight repository self-check
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
└── REPOSITORY_CHECKLIST.md
```

## Install and run on Windows

1. Install **Python 3.10 or newer** from <https://www.python.org/>.
2. During installation, tick **Add Python to PATH**.
3. Download or clone this repository.
4. Double-click `install_and_run.bat`.
5. After the first setup, launch the software with `run_mlbox.bat`.

## Run from command line

```bat
python -m venv .venv
.venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
python mlbox_app.py
```

## Example data

The `data/` folder contains the demonstration files used for the SoftwareX example:

- `Spectra_Nitrogen.xlsx`: leaf-level spectral reflectance data with wavelength columns from 400 to 2347 nm and a `Nitrogen` response column.
- `WhiteML-Box.tif`: a 12-band GeoTIFF raster stack for the raster-based mapping demonstration.

See `data/README.md` before using the example raster. Band order and reflectance scaling must match the trained model input; otherwise the exported prediction map is scientifically invalid even if the software runs.

## Recommended modelling workflow

1. Load the Excel or CSV spectral dataset.
2. Select the response variable, for example `Nitrogen`.
3. Select numeric wavelength/reflectance columns as predictors.
4. Optional: enable sensor band-centre conversion and select the target sensor.
5. Inspect spectra and remove obvious noisy wavelength ranges.
6. Run outlier screening, but do not blindly delete flagged samples.
7. Start with raw reflectance and PLSR as a baseline.
8. Compare preprocessing options only using cross-validated metrics.
9. Export metrics, predictions, plots, and feature-importance results.
10. For mapping, load a multiband GeoTIFF whose bands match the model configuration exactly.
11. Export the predicted single-band GeoTIFF and inspect the preview.

## Important scientific warning

WhiteML-Box will run models, but it will not rescue weak experimental design. Spectral datasets often have many more predictors than samples. For small datasets, a single train/test split is unstable. Prefer K-fold, repeated K-fold, leave-one-out, nested cross-validation, or grouped validation where the sampling design supports it. Do not overinterpret high model performance unless the validation design is defensible.

Spectral indices and wavelet features can generate a very large number of predictors. That can improve models, but it can also produce overfitted trash when sample size is small. Always compare complex transformations against raw reflectance and PLSR baselines.

## Mapping warning

The mapping tab does not infer raster band names. The GeoTIFF band count, band order, spectral meaning, and reflectance scale must match the trained model input. If sensor conversion was used, follow the rules in `docs/MAPPING_WORKFLOW.md`. If the raster stack is wrong, the map is wrong.

## Citation

If you use WhiteML-Box, cite the SoftwareX article associated with this repository and the archived GitHub release. A `CITATION.cff` file is provided for GitHub citation metadata.

## Contact

Haidi Abdullah  
Faculty of Geo-Information Science and Earth Observation, University of Twente  
Email: h.j.abdullah-1@utwente.nl
