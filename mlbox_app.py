"""
WhiteML-Box: desktop machine-learning tool for spectral/reflectance data.

Input assumption:
- Excel/CSV table where one column is the response variable Y (for example Mn).
- Numeric wavelength/reflectance columns are X, preferably named by wavelength values
  such as 2435.327, 2442.188, ...

Run:
    python mlbox_app.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm
from matplotlib.widgets import SpanSelector

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d
except Exception:  # pragma: no cover
    savgol_filter = None
    interp1d = None

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IsolationForest = None

try:
    from xgboost import XGBRegressor

    HAVE_XGBOOST = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    HAVE_XGBOOST = False


APP_NAME = "WhiteML-Box"
VERSION = "0.7.2"

INFO_TEXT = {
    "app": "Reset analysis clears band ranges, outlier masks, model outputs and plots while keeping the loaded file. Reset all clears the loaded file too. Use the visible action buttons to move from outlier screening to modelling and to run the selected model.",
    "load": "Load an Excel or CSV table. Use one numeric column as Y, for example Mn, and numeric wavelength/reflectance columns as X.",
    "sheet": "Choose the Excel worksheet to read. Changing the sheet resets downstream preprocessing, outlier and model results.",
    "target": "This is the response variable Y to be predicted, such as Mn, N, K, LMA, SLA, EWT or another measured trait.",
    "xcols": "These are predictor variables X. For spectral work, select the reflectance bands/wavelength columns. Numeric wavelength names enable wavelength-based deletion.",
    "accept": "Confirms the current Y and X selection and moves to preprocessing. This does not train a model yet.",
    "preprocess": "Spectral preprocessing transforms X before modelling. Use Raw as a baseline, then test derivatives, SNV, MSC, continuum removal, spectral indices, or continuous-wavelet features only if cross-validation improves.",
    "transform": "Raw keeps reflectance unchanged. SNV/MSC reduce scatter effects. Derivatives emphasize spectral shape. Continuum removal normalizes absorption features. Spectral indices generate two-band index combinations. Continuous-wavelet features isolate absorption-shape information inside selected wavelength regions.",
    "scale": "Standardizes X variables before modelling. Keep this on for PLSR, GPR, SVR and KNN. Tree models do not require it.",
    "sg": "Savitzky-Golay settings control smoothing and derivatives. Window length must be odd and should not be too large for your number of bands.",
    "bands": "Use the spectrum plot to mark noisy wavelength ranges for deletion before outlier screening and modelling. Typical noisy bands include low-SNR or water absorption regions. For wavelet features, select separate CWT regions below; do not confuse CWT regions with deleted noisy bands.",
    "outliers": "Screens abnormal samples using robust Y statistics, PCA Hotelling T² on X, Isolation Forest on X, or a combined method. After screening, explicitly choose Keep flagged outliers or Remove flagged outliers before moving to modelling. Do not remove outliers blindly; inspect the plot first.",
    "model": "Choose the regression algorithm. The Run selected model button is duplicated in the always-visible action bar so it is not hidden by the model settings panel. PLSR is the baseline for spectral data. Complex models can overfit small sample sets, so compare them under identical validation.",
    "validation": "Choose how predictions are evaluated. K-fold CV is usually the safest default. Leave-one-out can be noisy. Train/test split is weak for small datasets.",
    "plsr": "PLSR component selection scans component counts and chooses the one with minimum cross-validated RMSE. This avoids arbitrary component choices.",
    "trees": "Tree and boosting settings. More trees stabilize Random Forest and Extra Trees but increase runtime. XGBoost depth and learning rate control overfitting risk.",
    "other_models": "SVR and KNN can work for spectroscopy after scaling, but they are sensitive to tuning and sample size.",
    "results": "Shows validation metrics, measured-vs-predicted plots, residual plots, feature importance, PLSR VIP and exports.",
    "metrics": "R², RMSE, nRMSE, bias, RPD/RPIQ and residual uncertainty are computed from validation predictions, not from fitted training predictions.",
    "mapping": "Apply the final trained model to a multiband GeoTIFF. If sensor band-center conversion was not used, the raster band count and order must match the model X predictors after noisy-band deletion. If sensor conversion was used, the raster may either match the trained target-sensor bands directly or match the original source bands for internal conversion. Spectral indices and wavelet features are then rebuilt internally. The output is a single-band prediction GeoTIFF with the same grid, CRS and transform as the input raster.",
    "indices": "Spectral indices create machine-learning predictors from every selected pair of bands. The implemented formulas are normalized difference (Bi-Bj)/(Bi+Bj), simple ratio Bi/Bj, and difference Bi-Bj. Use the pair limit only to protect memory on very high-dimensional data; 0 means use every possible pair.",
    "wavelet": "Continuous-wavelet features use Mexican-hat/Ricker wavelet coefficients across selected wavelength regions and scales. Select regions that correspond to absorption features or known spectral windows. If no CWT region is selected, the full spectrum is used.",
    "sensor_resampling": "Optional step before preprocessing/model fitting. Upload the original sensor band centers if your X column names are not numeric wavelengths, then select a target sensor. The app interpolates the original reflectance curves to the target band centers and uses the converted bands for all next steps. If disabled, the original uploaded reflectance bands are used unchanged.",
    "workflow": "Open a brief guide explaining the workflow and what each step does.",
}


def resource_path(name: str) -> Path:
    """Return a resource path that works in source mode and PyInstaller one-file builds."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def _nm_range(start: float, stop: float, step: float) -> List[float]:
    vals = np.arange(float(start), float(stop) + 0.5 * float(step), float(step), dtype=float)
    return [float(round(v, 4)) for v in vals]


SENSOR_BAND_CENTERS_NM: Dict[str, List[float]] = {
    # Reflective bands only; thermal/panchromatic channels are intentionally excluded.
    "Landsat 5 TM": [485, 560, 660, 830, 1650, 2215],
    "Landsat 8 OLI": [443, 482, 561, 655, 865, 1373, 1609, 2201],
    "Landsat 9 OLI-2": [443, 482, 561, 655, 865, 1373, 1609, 2201],
    "Sentinel-2 MSI": [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190],
    # Hyperspectral mission presets use nominal centre grids. Users can still export and replace
    # these with exact product-specific band-centre files when that level of traceability matters.
    "PRISMA nominal hyperspectral": _nm_range(400, 2500, 10),
    "DESIS nominal hyperspectral": [float(round(v, 4)) for v in np.linspace(400, 1000, 235)],
    "EnMAP nominal hyperspectral": sorted(set(_nm_range(420, 997.5, 6.5) + _nm_range(1000, 2450, 10))),
    "CHIME nominal hyperspectral": _nm_range(400, 2500, 10),
}


WORKFLOW_GUIDE = """WhiteML-Box workflow guide

Developed by Haidi Abdullah - University of Twente

1. Data
Load an Excel/CSV table, select the measured target variable Y, and select the spectral or environmental predictor columns X. Numeric wavelength column names are best because the app can use them directly as wavelength centres.

2. Optional sensor band-centre conversion
Use this only when you want to simulate or harmonise your spectra to another sensor. Upload the original sensor band centres if your X column names are not true wavelengths. Then enable conversion and select Landsat 5, Landsat 8/9, Sentinel-2, PRISMA, DESIS, EnMAP, or CHIME. The app interpolates each sample spectrum to the selected sensor centres and uses those converted bands in all later steps. Leave this disabled when you want to use the original uploaded reflectance bands.

3. Preprocessing
Raw is the baseline. SNV and MSC reduce scatter effects. Log(1/R) changes reflectance to apparent absorbance. Savitzky-Golay smoothing and derivatives reduce noise and highlight spectral shape. Continuum removal normalises absorption features. Spectral indices create band-pair features. Continuous-wavelet features extract absorption-shape information from selected wavelength regions.

4. Noisy band removal
Use the spectrum plot to mark wavelength ranges that are clearly noisy or outside the reliable sensor range. Do not delete bands just to improve R²; that is overfitting disguised as cleaning.

5. Outlier screening
Run the outlier tools, inspect the plot, then explicitly choose Keep or Remove. Removing samples blindly is bad science. Remove only defensible measurement errors or samples outside the modelling domain.

6. Modelling and validation
Run PLSR first as the baseline. For PLSR, the app scans components and selects the lowest cross-validated RMSE. Then compare GPR, Random Forest, Extra Trees, SVR, Gradient Boosting, KNN, or XGBoost under the same validation design. Trust cross-validated R²/RMSE, not training fit.

7. Results
Inspect measured-vs-predicted, residuals, feature importance/VIP, and PLSR component scan. Export the metrics, predictions, plots, selected configuration, and feature tables.

8. Mapping
After training, load a multiband GeoTIFF. If sensor conversion was not used, raster bands must match the model predictor bands. If conversion was used, the raster can either already contain the selected target-sensor bands in the trained model order, or contain the original source bands after noisy-band deletion so the app can convert pixels internally to the selected target sensor before prediction.
"""


class ToolTip:
    """Small hover tooltip for Tkinter widgets."""

    def __init__(self, widget, text: str, delay: int = 450, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(
                self._tip,
                text=self.text,
                justify="left",
                relief="solid",
                borderwidth=1,
                padding=(8, 5),
                wraplength=self.wraplength,
            )
            label.pack()
        except Exception:
            self._tip = None

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# ---------------------------- data utilities ---------------------------- #


def safe_float(value) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def numeric_column_names(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            out.append(str(c))
    return out


def wavelength_values(columns: Sequence[str]) -> Tuple[np.ndarray, bool]:
    vals = [safe_float(c) for c in columns]
    if all(v is not None for v in vals):
        return np.array(vals, dtype=float), True
    return np.arange(len(columns), dtype=float), False


def sort_by_wavelength(columns: Sequence[str]) -> List[str]:
    vals = [safe_float(c) for c in columns]
    if all(v is not None for v in vals):
        return [c for _, c in sorted(zip(vals, columns), key=lambda t: t[0])]
    return list(columns)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return np.zeros_like(x, dtype=float)
        return (x - np.nanmean(x)) / std
    return 0.6745 * (x - med) / mad


# ---------------------------- spectral transforms ---------------------------- #


class SpectralTransformer(BaseEstimator, TransformerMixin):
    """Spectral preprocessing and feature engineering used inside sklearn pipelines."""

    def __init__(
        self,
        method: str = "Raw",
        wavelengths: Optional[Sequence[float]] = None,
        sg_window: int = 11,
        sg_poly: int = 2,
        index_formulas: str = "NDI,Ratio,Difference",
        index_max_pairs: int = 5000,
        cwt_regions: Optional[Sequence[Tuple[float, float]]] = None,
        cwt_min_scale: int = 2,
        cwt_max_scale: int = 16,
        cwt_num_scales: int = 8,
    ):
        # Store constructor parameters without mutation so sklearn.clone can safely copy pipelines.
        self.method = method
        self.wavelengths = wavelengths
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.index_formulas = index_formulas
        self.index_max_pairs = index_max_pairs
        self.cwt_regions = cwt_regions
        self.cwt_min_scale = cwt_min_scale
        self.cwt_max_scale = cwt_max_scale
        self.cwt_num_scales = cwt_num_scales
        self._msc_ref = None
        self._feature_names_out: List[str] = []
        self._index_pairs: List[Tuple[int, int]] = []
        self._cwt_plan: List[Tuple[str, np.ndarray]] = []
        self._cwt_scales: np.ndarray = np.array([], dtype=float)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = int(X.shape[1])
        if self.method == "MSC":
            self._msc_ref = np.nanmean(X, axis=0)
        self._index_pairs = self._select_index_pairs(self.n_features_in_)
        self._cwt_plan = self._make_cwt_plan(self.n_features_in_)
        self._cwt_scales = self._make_cwt_scales()
        self._feature_names_out = self._make_feature_names([str(i) for i in range(self.n_features_in_)])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        method = self.method
        if method == "Raw":
            return X
        if method == "SNV":
            return self._snv(X)
        if method == "MSC":
            return self._msc(X)
        if method == "Log(1/R)":
            return np.log(1.0 / np.clip(X, 1e-12, None))
        if method == "Savitzky-Golay smoothing":
            return self._savgol(X, deriv=0)
        if method == "First derivative":
            return self._savgol(X, deriv=1)
        if method == "Second derivative":
            return self._savgol(X, deriv=2)
        if method == "Continuum removal":
            return self._continuum_removal(X)
        if method.startswith("Spectral indices"):
            return self._spectral_indices(X)
        if method.startswith("Continuous wavelet"):
            return self._continuous_wavelet_features(X)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [str(i) for i in range(getattr(self, "n_features_in_", 0))]
        return np.asarray(self._make_feature_names([str(f) for f in input_features]), dtype=object)

    def estimate_output_features(self, n_features: int, input_features: Optional[Sequence[str]] = None) -> int:
        names = list(input_features) if input_features is not None else [str(i) for i in range(n_features)]
        return len(self._make_feature_names(names))

    @staticmethod
    def _snv(X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd

    def _msc(self, X: np.ndarray) -> np.ndarray:
        ref = self._msc_ref
        if ref is None or len(ref) != X.shape[1]:
            ref = np.nanmean(X, axis=0)
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            try:
                slope, intercept = np.polyfit(ref, row, 1)
                if abs(slope) < 1e-12:
                    out[i] = row
                else:
                    out[i] = (row - intercept) / slope
            except Exception:
                out[i] = row
        return out

    def _valid_window(self, n_features: int) -> int:
        if savgol_filter is None or n_features < 5:
            return max(3, n_features if n_features % 2 == 1 else n_features - 1)
        w = max(3, int(self.sg_window))
        if w % 2 == 0:
            w += 1
        if w >= n_features:
            w = n_features - 1 if n_features % 2 == 0 else n_features
        if w < 3:
            w = 3
        return int(w)

    def _savgol(self, X: np.ndarray, deriv: int) -> np.ndarray:
        n_features = X.shape[1]
        if savgol_filter is None or n_features < 5:
            # Fallback: numerical derivative without smoothing.
            if deriv == 0:
                return X
            wl = np.asarray(self.wavelengths, dtype=float) if self.wavelengths is not None and len(self.wavelengths) == n_features else None
            if wl is None:
                g = np.gradient(X, axis=1)
            else:
                g = np.gradient(X, wl, axis=1)
            if deriv == 1:
                return g
            return np.gradient(g, axis=1)
        w = self._valid_window(n_features)
        poly = min(max(1, int(self.sg_poly)), w - 1)
        if self.wavelengths is not None and len(self.wavelengths) == n_features:
            wl_arr = np.asarray(self.wavelengths, dtype=float)
            diffs = np.diff(wl_arr)
            delta = float(np.nanmedian(np.abs(diffs))) if len(diffs) else 1.0
            if not np.isfinite(delta) or delta == 0:
                delta = 1.0
        else:
            delta = 1.0
        return savgol_filter(X, window_length=w, polyorder=poly, deriv=deriv, delta=delta, axis=1, mode="interp")

    def _continuum_removal(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        wl = np.asarray(self.wavelengths, dtype=float) if self.wavelengths is not None and len(self.wavelengths) == n_features else np.arange(n_features)
        if interp1d is None or n_features < 4:
            return X
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            out[i] = _continuum_remove_row(wl, row)
        return out

    def _formula_list(self) -> List[str]:
        raw = str(self.index_formulas or "NDI,Ratio,Difference")
        aliases = {
            "normalized difference": "NDI",
            "normalised difference": "NDI",
            "nd": "NDI",
            "ndi": "NDI",
            "ratio": "Ratio",
            "sr": "Ratio",
            "simple ratio": "Ratio",
            "difference": "Difference",
            "diff": "Difference",
            "dsi": "Difference",
        }
        out: List[str] = []
        for item in raw.replace(";", ",").split(","):
            key = item.strip().lower()
            if not key:
                continue
            val = aliases.get(key, item.strip())
            if val in {"NDI", "Ratio", "Difference"} and val not in out:
                out.append(val)
        return out or ["NDI", "Ratio", "Difference"]

    def _select_index_pairs(self, n_features: int) -> List[Tuple[int, int]]:
        if n_features < 2:
            return []
        total_pairs = n_features * (n_features - 1) // 2
        limit = int(self.index_max_pairs)
        if limit <= 0 or limit >= total_pairs:
            step = 1
            target = total_pairs
        else:
            step = int(math.ceil(total_pairs / max(1, limit)))
            target = limit
        pairs: List[Tuple[int, int]] = []
        k = 0
        for i in range(n_features - 1):
            for j in range(i + 1, n_features):
                if k % step == 0:
                    pairs.append((i, j))
                    if len(pairs) >= target:
                        return pairs
                k += 1
        return pairs

    def _spectral_indices(self, X: np.ndarray) -> np.ndarray:
        pairs = self._index_pairs or self._select_index_pairs(X.shape[1])
        if not pairs:
            return X
        pi = np.asarray([p[0] for p in pairs], dtype=int)
        pj = np.asarray([p[1] for p in pairs], dtype=int)
        a = X[:, pi]
        b = X[:, pj]
        pieces = []
        for formula in self._formula_list():
            if formula == "NDI":
                pieces.append(_safe_divide(a - b, a + b))
            elif formula == "Ratio":
                pieces.append(_safe_divide(a, b))
            elif formula == "Difference":
                pieces.append(a - b)
        return np.hstack(pieces) if pieces else X

    def _make_cwt_scales(self) -> np.ndarray:
        lo = max(1, int(self.cwt_min_scale))
        hi = max(lo, int(self.cwt_max_scale))
        n = max(1, int(self.cwt_num_scales))
        vals = np.linspace(lo, hi, n)
        vals = np.unique(np.maximum(1, np.rint(vals).astype(int))).astype(float)
        return vals

    def _make_cwt_plan(self, n_features: int) -> List[Tuple[str, np.ndarray]]:
        wl = np.asarray(self.wavelengths, dtype=float) if self.wavelengths is not None and len(self.wavelengths) == n_features else np.arange(n_features, dtype=float)
        regions = list(self.cwt_regions or [])
        plan: List[Tuple[str, np.ndarray]] = []
        if not regions:
            plan.append(("all", np.arange(n_features, dtype=int)))
            return plan
        for ridx, pair in enumerate(regions, start=1):
            try:
                a, b = float(pair[0]), float(pair[1])
            except Exception:
                continue
            lo, hi = min(a, b), max(a, b)
            idx = np.where((wl >= lo) & (wl <= hi))[0]
            if len(idx) >= 2:
                plan.append((f"R{ridx}", idx.astype(int)))
        if not plan:
            plan.append(("all", np.arange(n_features, dtype=int)))
        return plan

    def _continuous_wavelet_features(self, X: np.ndarray) -> np.ndarray:
        plan = self._cwt_plan or self._make_cwt_plan(X.shape[1])
        scales = self._cwt_scales if len(self._cwt_scales) else self._make_cwt_scales()
        pieces: List[np.ndarray] = []
        for _label, idx in plan:
            seg = X[:, idx]
            if seg.shape[1] < 2:
                continue
            for scale in scales:
                pieces.append(_ricker_cwt_matrix(seg, float(scale)))
        if not pieces:
            return X
        return np.hstack(pieces)

    def _make_feature_names(self, input_features: Sequence[str]) -> List[str]:
        method = self.method
        n_features = len(input_features)
        if not method.startswith("Spectral indices") and not method.startswith("Continuous wavelet"):
            return list(input_features)
        if method.startswith("Spectral indices"):
            pairs = self._index_pairs or self._select_index_pairs(n_features)
            names: List[str] = []
            for formula in self._formula_list():
                for i, j in pairs:
                    bi, bj = str(input_features[i]), str(input_features[j])
                    if formula == "NDI":
                        names.append(f"NDI_{bi}_{bj}")
                    elif formula == "Ratio":
                        names.append(f"Ratio_{bi}_{bj}")
                    elif formula == "Difference":
                        names.append(f"Diff_{bi}_{bj}")
            return names or list(input_features)
        plan = self._cwt_plan or self._make_cwt_plan(n_features)
        scales = self._cwt_scales if len(self._cwt_scales) else self._make_cwt_scales()
        names = []
        for label, idx in plan:
            for scale in scales:
                for i in idx:
                    names.append(f"CWT_{label}_s{int(scale)}_{input_features[int(i)]}")
        return names or list(input_features)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=np.abs(denominator) > 1e-12)
    out[~np.isfinite(out)] = 0.0
    return out


def _ricker_kernel(scale: float) -> np.ndarray:
    scale = max(1.0, float(scale))
    radius = max(3, int(math.ceil(4.0 * scale)))
    x = np.arange(-radius, radius + 1, dtype=float)
    xs = x / scale
    kernel = (1.0 - xs ** 2) * np.exp(-0.5 * xs ** 2)
    kernel = kernel - np.nanmean(kernel)
    norm = math.sqrt(float(np.sum(kernel ** 2)))
    if norm > 0:
        kernel = kernel / norm
    return kernel


def _ricker_cwt_matrix(X: np.ndarray, scale: float) -> np.ndarray:
    kernel = _ricker_kernel(scale)
    out = np.empty_like(X, dtype=float)
    n = X.shape[1]
    for i in range(X.shape[0]):
        conv = np.convolve(X[i], kernel, mode="same")
        if len(conv) != n:
            start = max(0, (len(conv) - n) // 2)
            conv = conv[start:start + n]
            if len(conv) < n:
                conv = np.pad(conv, (0, n - len(conv)), mode="edge")
        out[i] = conv
    out[~np.isfinite(out)] = 0.0
    return out


def _continuum_remove_row(wl: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Simple upper-envelope continuum removal.

    This is intentionally conservative. It is not a full hyperspectral convex hull
    package, but it is stable for spectroscopy-style reflectance curves.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(row, dtype=float)
    mask = np.isfinite(wl) & np.isfinite(y)
    if mask.sum() < 4:
        return y.copy()
    x = wl[mask]
    yy = y[mask]
    order = np.argsort(x)
    x = x[order]
    yy = yy[order]

    # Build a monotone upper hull in x-y space.
    points = list(zip(x, yy))
    upper: List[Tuple[float, float]] = []
    for p in points:
        while len(upper) >= 2:
            (x1, y1), (x2, y2) = upper[-2], upper[-1]
            x3, y3 = p
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    if len(upper) < 2:
        return y.copy()
    hx = np.array([p[0] for p in upper], dtype=float)
    hy = np.array([p[1] for p in upper], dtype=float)
    f = interp1d(hx, hy, bounds_error=False, fill_value="extrapolate")
    continuum = f(wl)
    continuum = np.where(np.abs(continuum) < 1e-12, np.nan, continuum)
    cr = y / continuum
    cr[~np.isfinite(cr)] = y[~np.isfinite(cr)]
    return cr


# ---------------------------- model and metrics ---------------------------- #


@dataclass
class RunResult:
    model_name: str
    cv_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_index: np.ndarray
    metrics: Dict[str, float]
    feature_table: Optional[pd.DataFrame]
    plsr_scan: Optional[pd.DataFrame]
    fitted_model: object
    notes: List[str]


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_effective: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    n = len(y_true)
    score_r2 = float(r2_score(y_true, y_pred)) if n >= 2 else float("nan")
    if n > p_effective + 1 and np.isfinite(score_r2):
        adj = 1.0 - (1.0 - score_r2) * (n - 1.0) / (n - p_effective - 1.0)
    else:
        adj = float("nan")
    y_range = float(np.nanmax(y_true) - np.nanmin(y_true)) if n else float("nan")
    y_mean = float(np.nanmean(np.abs(y_true))) if n else float("nan")
    q75, q25 = np.nanpercentile(y_true, [75, 25]) if n else (float("nan"), float("nan"))
    iqr = float(q75 - q25)
    e_rmse = rmse(y_true, y_pred)
    sd = float(np.nanstd(residual, ddof=1)) if n > 1 else float("nan")
    return {
        "n": float(n),
        "R2": score_r2,
        "Adjusted_R2": float(adj),
        "RMSE": e_rmse,
        "nRMSE_range_%": float(100.0 * e_rmse / y_range) if y_range and np.isfinite(y_range) else float("nan"),
        "nRMSE_mean_%": float(100.0 * e_rmse / y_mean) if y_mean and np.isfinite(y_mean) else float("nan"),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias_mean_pred_minus_obs": float(np.nanmean(residual)),
        "Residual_SD": sd,
        "Uncertainty_95pct_residual": float(1.96 * sd) if np.isfinite(sd) else float("nan"),
        "RPD": float(np.nanstd(y_true, ddof=1) / e_rmse) if e_rmse > 0 and n > 1 else float("nan"),
        "RPIQ": float(iqr / e_rmse) if e_rmse > 0 else float("nan"),
    }


def pls_vip(pls: PLSRegression, feature_names: Sequence[str]) -> pd.DataFrame:
    # VIP implementation for sklearn PLSRegression.
    # Based on W, T and Q matrices.
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    if total_s == 0:
        vip = np.zeros((p,))
    else:
        vip = np.sqrt(p * (w ** 2 @ s).ravel() / total_s)
    coef = np.asarray(pls.coef_).ravel()
    if len(coef) != len(feature_names):
        coef = np.resize(coef, len(feature_names))
    return pd.DataFrame({"feature": list(feature_names), "VIP": vip, "coefficient": coef})


def make_cv(method: str, k: int, random_state: int, repeats: int = 3):
    if method == "Leave-One-Out":
        return LeaveOneOut(), "Leave-One-Out"
    if method == "Repeated KFold":
        return RepeatedKFold(n_splits=max(2, int(k)), n_repeats=max(2, int(repeats)), random_state=random_state), f"Repeated KFold ({k} x {repeats})"
    return KFold(n_splits=max(2, int(k)), shuffle=True, random_state=random_state), f"KFold ({k})"


def build_regressor(model_name: str, params: Dict[str, float], n_features: int):
    random_state = int(params.get("random_state", 42))
    if model_name == "PLSR":
        n_components = int(params.get("n_components", min(2, n_features)))
        n_components = max(1, min(n_components, n_features))
        return PLSRegression(n_components=n_components), n_components
    if model_name == "GPR":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2))
        return GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, random_state=random_state), n_features
    if model_name == "XGBoost":
        if not HAVE_XGBOOST:
            raise RuntimeError("XGBoost is not installed. Install it with: pip install xgboost")
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ), n_features
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
            n_jobs=-1,
        ), n_features
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
            n_jobs=-1,
        ), n_features
    if model_name == "SVR":
        return SVR(C=float(params.get("svr_c", 10.0)), epsilon=float(params.get("svr_epsilon", 0.1)), kernel="rbf"), n_features
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=random_state, n_estimators=int(params.get("n_estimators", 300))), n_features
    if model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=int(params.get("knn_neighbors", 5)), weights="distance"), n_features
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(
    model_name: str,
    params: Dict[str, float],
    transform_method: str,
    wavelengths: np.ndarray,
    sg_window: int,
    sg_poly: int,
    scale_x: bool,
    n_features: int,
    index_max_pairs: int = 5000,
    cwt_regions: Optional[Sequence[Tuple[float, float]]] = None,
    cwt_min_scale: int = 2,
    cwt_max_scale: int = 16,
    cwt_num_scales: int = 8,
) -> Tuple[Pipeline, int]:
    spectral = SpectralTransformer(
        transform_method,
        wavelengths,
        sg_window,
        sg_poly,
        index_max_pairs=index_max_pairs,
        cwt_regions=cwt_regions,
        cwt_min_scale=cwt_min_scale,
        cwt_max_scale=cwt_max_scale,
        cwt_num_scales=cwt_num_scales,
    )
    n_model_features = max(1, spectral.estimate_output_features(n_features))
    reg, p_eff = build_regressor(model_name, params, n_model_features)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    steps.append(("spectral", spectral))
    # PLSR, GPR and SVR need scaling by default. Tree models do not need it but it is harmless.
    if scale_x or model_name in {"PLSR", "GPR", "SVR", "KNN"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", reg))
    return Pipeline(steps), p_eff


def scan_plsr_components(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    transform_method: str,
    wavelengths: np.ndarray,
    sg_window: int,
    sg_poly: int,
    scale_x: bool,
    cv,
    max_components: int,
    index_max_pairs: int = 5000,
    cwt_regions: Optional[Sequence[Tuple[float, float]]] = None,
    cwt_min_scale: int = 2,
    cwt_max_scale: int = 16,
    cwt_num_scales: int = 8,
) -> pd.DataFrame:
    spectral = SpectralTransformer(
        transform_method,
        wavelengths,
        sg_window,
        sg_poly,
        index_max_pairs=index_max_pairs,
        cwt_regions=cwt_regions,
        cwt_min_scale=cwt_min_scale,
        cwt_max_scale=cwt_max_scale,
        cwt_num_scales=cwt_num_scales,
    )
    n_model_features = max(1, spectral.estimate_output_features(X.shape[1], feature_names))
    max_allowed = max(1, min(int(max_components), n_model_features, X.shape[0] - 1))
    rows = []
    for n_comp in range(1, max_allowed + 1):
        pipe, _ = make_pipeline(
            "PLSR",
            {"n_components": n_comp, "random_state": 42},
            transform_method,
            wavelengths,
            sg_window,
            sg_poly,
            scale_x,
            X.shape[1],
            index_max_pairs=index_max_pairs,
            cwt_regions=cwt_regions,
            cwt_min_scale=cwt_min_scale,
            cwt_max_scale=cwt_max_scale,
            cwt_num_scales=cwt_num_scales,
        )
        pred = cross_val_predict(pipe, X, y, cv=cv)
        rows.append({"n_components": n_comp, "RMSE_CV": rmse(y, pred), "R2_CV": r2_score(y, pred)})
    return pd.DataFrame(rows)


# ---------------------------- GUI app ---------------------------- #


class MLBoxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {VERSION}")
        self.geometry("1450x900")
        self.minsize(1050, 700)

        self.file_path: Optional[Path] = None
        self.excel_file: Optional[pd.ExcelFile] = None
        self.df: Optional[pd.DataFrame] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.excluded_ranges: List[Tuple[float, float]] = []
        self.last_result: Optional[RunResult] = None
        self.last_clean_data: Optional[Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]] = None
        # Dedicated mapping state. This is deliberately separate from the Results tab
        # so a trained mapping-ready model is not lost by harmless plot/result refreshes.
        self.mapping_model = None
        self.mapping_xcols: List[str] = []
        self.mapping_wavelengths: List[float] = []
        self.mapping_source_xcols: List[str] = []
        self.mapping_source_wavelengths: List[float] = []
        self.mapping_target_sensor: str = ""
        self.mapping_sensor_conversion_enabled: bool = False
        self.mapping_model_info: Dict[str, object] = {}
        self.raster_path: Optional[Path] = None
        self.last_map_path: Optional[Path] = None
        self.spectrum_span_selector = None
        self.cwt_span_selector = None
        self.cwt_regions: List[Tuple[float, float]] = []
        self.source_band_centers: List[float] = []
        self.source_band_center_map: Dict[str, float] = {}
        self.logo_img = None
        self._suspend_state_callbacks = False

        self._build_vars()
        self._build_ui()
        self._attach_state_traces()
        self.after(150, self._maximize_if_possible)

    def _build_vars(self):
        self.sheet_var = tk.StringVar(value="")
        self.target_var = tk.StringVar(value="")
        self.transform_var = tk.StringVar(value="Raw")
        self.scale_var = tk.BooleanVar(value=True)
        self.sensor_resample_enabled_var = tk.BooleanVar(value=False)
        self.target_sensor_var = tk.StringVar(value="Sentinel-2 MSI")
        self.source_band_file_var = tk.StringVar(value="")
        self.sg_window_var = tk.IntVar(value=11)
        self.sg_poly_var = tk.IntVar(value=2)
        self.index_max_pairs_var = tk.IntVar(value=5000)
        self.cwt_start_var = tk.StringVar(value="")
        self.cwt_end_var = tk.StringVar(value="")
        self.cwt_min_scale_var = tk.IntVar(value=2)
        self.cwt_max_scale_var = tk.IntVar(value=16)
        self.cwt_num_scales_var = tk.IntVar(value=8)
        self.remove_outliers_var = tk.BooleanVar(value=False)
        self.outlier_method_var = tk.StringVar(value="PCA Hotelling T2 on X")
        self.outlier_threshold_var = tk.DoubleVar(value=3.5)
        self.outlier_contam_var = tk.DoubleVar(value=0.05)
        self.pca_conf_var = tk.DoubleVar(value=0.975)
        self.model_var = tk.StringVar(value="PLSR")
        self.cv_method_var = tk.StringVar(value="KFold")
        self.kfold_var = tk.IntVar(value=5)
        self.repeats_var = tk.IntVar(value=3)
        self.test_size_var = tk.DoubleVar(value=0.25)
        self.random_state_var = tk.IntVar(value=42)
        self.pls_comp_var = tk.IntVar(value=5)
        self.pls_max_comp_var = tk.IntVar(value=20)
        self.n_trees_var = tk.IntVar(value=500)
        self.max_depth_var = tk.IntVar(value=3)
        self.learning_rate_var = tk.DoubleVar(value=0.05)
        self.min_leaf_var = tk.IntVar(value=1)
        self.svr_c_var = tk.DoubleVar(value=10.0)
        self.svr_eps_var = tk.DoubleVar(value=0.1)
        self.knn_k_var = tk.IntVar(value=5)
        self.band_start_var = tk.StringVar(value="")
        self.band_end_var = tk.StringVar(value="")
        self.result_plot_var = tk.StringVar(value="Measured vs predicted")
        self.raster_path_var = tk.StringVar(value="")
        self.map_folder_var = tk.StringVar(value="")
        self.map_output_name_var = tk.StringVar(value="white_mlbox_prediction_map.tif")
        self.map_nodata_var = tk.DoubleVar(value=-9999.0)
        self.map_scale_mode_var = tk.StringVar(value="Auto: detect raster 0-10000")
        self.map_preview_render_var = tk.StringVar(value="Stretch: robust percentiles")
        self.map_preview_cmap_var = tk.StringVar(value="viridis")
        self.map_preview_min_pct_var = tk.DoubleVar(value=2.0)
        self.map_preview_max_pct_var = tk.DoubleVar(value=98.0)
        self.map_preview_classes_var = tk.IntVar(value=7)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=4)
        self._load_logo_for_header()
        if self.logo_img is not None:
            ttk.Label(top, image=self.logo_img).pack(side=tk.LEFT, padx=(0, 8))
        title_block = ttk.Frame(top)
        title_block.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(title_block, text=f"{APP_NAME}", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(title_block, text="Developed by Haidi Abdullah - University of Twente", font=("Segoe UI", 9)).pack(anchor="w")
        self._info_button(top, "WhiteML-Box workflow", INFO_TEXT["app"]).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(top, text="Spectral regression workflow: Excel → sensor conversion → outliers/bands → model → validation outputs").pack(side=tk.LEFT, padx=16)
        ttk.Button(top, text="Workflow guide", command=self.show_workflow_guide).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Reset analysis", command=self.reset_analysis_state).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Reset all", command=self.reset_all_workflow).pack(side=tk.RIGHT, padx=4)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_pre = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_mapping = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="1 Data")
        self.notebook.add(self.tab_pre, text="2 Preprocess")
        self.notebook.add(self.tab_model, text="3 Model")
        self.notebook.add(self.tab_results, text="4 Results")
        self.notebook.add(self.tab_mapping, text="5 Mapping")
        self.notebook.add(self.tab_help, text="6 Help / workflow")
        self._build_data_tab()
        self._build_pre_tab()
        self._build_model_tab()
        self._build_results_tab()
        self._build_mapping_tab()
        self._build_help_tab()

    def _load_logo_for_header(self):
        try:
            logo_path = resource_path("white_ml_box_logo.png")
            if logo_path.exists():
                img = tk.PhotoImage(file=str(logo_path))
                # Original logo is large; keep the header compact.
                factor = max(1, int(max(img.width(), img.height()) / 80))
                self.logo_img = img.subsample(factor, factor)
                try:
                    self.iconphoto(False, self.logo_img)
                except Exception:
                    pass
        except Exception:
            self.logo_img = None

    def show_workflow_guide(self):
        try:
            self.notebook.select(self.tab_help)
        except Exception:
            messagebox.showinfo("WhiteML-Box workflow guide", WORKFLOW_GUIDE)

    def _build_help_tab(self):
        frame = self.tab_help
        outer = ttk.Frame(frame)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(outer, text="WhiteML-Box workflow and option guide", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))
        txt = tk.Text(outer, wrap="word")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        txt.insert("1.0", WORKFLOW_GUIDE)
        txt.configure(state="disabled")

    # ---------------------------- UI helpers and reset logic ---------------------------- #

    def _info_button(self, parent, title: str, text: str):
        btn = ttk.Button(parent, text="?", width=3, command=lambda: messagebox.showinfo(title, text))
        ToolTip(btn, text)
        return btn

    def _maximize_if_possible(self):
        """Start large on Windows while keeping the layout usable on smaller screens."""
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass

    def _make_scrollable_side_panel(self, parent, width: int = 360):
        """Create a fixed-width vertical control panel with scrolling.

        The app has many spectroscopy options. Without this panel, lower buttons can
        disappear on laptop screens or when Windows display scaling is high.
        """
        holder = ttk.Frame(parent)
        holder.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        canvas = tk.Canvas(holder, width=width, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        def _update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        def _resize_inner(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                delta = getattr(event, "delta", 0)
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", _update_scrollregion)
        canvas.bind("<Configure>", _resize_inner)
        inner.bind("<Enter>", _bind_wheel)
        inner.bind("<Leave>", _unbind_wheel)
        return inner

    def _pack_label_info(self, parent, text: str, info: str = "", font=None, pady=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, anchor="w", pady=pady if pady is not None else 0)
        ttk.Label(row, text=text, font=font).pack(side=tk.LEFT, anchor="w")
        if info:
            self._info_button(row, text, info).pack(side=tk.LEFT, padx=(5, 0))
        return row

    def _button_with_info(self, parent, text: str, command, info_title: str, info_text: str):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text=text, command=command).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._info_button(row, info_title, info_text).pack(side=tk.LEFT, padx=(5, 0))
        return row

    def _attach_state_traces(self):
        data_vars = [self.target_var]
        pre_vars = [
            self.transform_var,
            self.sensor_resample_enabled_var,
            self.target_sensor_var,
            self.sg_window_var,
            self.sg_poly_var,
            self.index_max_pairs_var,
            self.cwt_min_scale_var,
            self.cwt_max_scale_var,
            self.cwt_num_scales_var,
            self.outlier_method_var,
            self.outlier_threshold_var,
            self.outlier_contam_var,
            self.pca_conf_var,
        ]
        model_vars = [
            self.scale_var,
            self.remove_outliers_var,
            self.model_var,
            self.cv_method_var,
            self.kfold_var,
            self.repeats_var,
            self.test_size_var,
            self.random_state_var,
            self.pls_comp_var,
            self.pls_max_comp_var,
            self.n_trees_var,
            self.max_depth_var,
            self.learning_rate_var,
            self.min_leaf_var,
            self.svr_c_var,
            self.svr_eps_var,
            self.knn_k_var,
        ]
        for v in data_vars:
            v.trace_add("write", lambda *_: self.invalidate_from_data_selection())
        for v in pre_vars:
            v.trace_add("write", lambda *_: self.invalidate_from_preprocessing())
        for v in model_vars:
            v.trace_add("write", lambda *_: self.invalidate_results_only())

    def invalidate_from_data_selection(self):
        if getattr(self, "_suspend_state_callbacks", False):
            return
        self.excluded_ranges = []
        self.cwt_regions = []
        self.source_band_centers = []
        self.source_band_center_map = {}
        self.source_band_file_var.set("")
        if hasattr(self, "band_center_status"):
            self.update_band_center_status()
        self.outlier_mask = None
        self.last_result = None
        self.last_clean_data = None
        self._clear_mapping_model_state()
        if hasattr(self, "range_list"):
            self.refresh_range_list()
        if hasattr(self, "cwt_region_list"):
            self.refresh_cwt_region_list()
        if hasattr(self, "outlier_status"):
            self.outlier_status.configure(text="Data selection changed. Re-run outlier analysis if needed.")
        self.last_map_path = None
        self._clear_results_display()
        self._clear_mapping_display()

    def invalidate_from_preprocessing(self):
        if getattr(self, "_suspend_state_callbacks", False):
            return
        self.outlier_mask = None
        self.last_result = None
        self.last_clean_data = None
        self._clear_mapping_model_state()
        if hasattr(self, "outlier_status"):
            self.outlier_status.configure(text="Preprocessing/settings changed. Re-run outlier analysis if needed.")
        self.last_map_path = None
        self._clear_results_display()
        self._clear_mapping_display()

    def invalidate_results_only(self):
        if getattr(self, "_suspend_state_callbacks", False):
            return
        self.last_result = None
        self.last_clean_data = None
        self._clear_mapping_model_state()
        self.last_map_path = None
        self._clear_results_display()
        self._clear_mapping_display()


    def _clear_mapping_model_state(self):
        """Remove the currently stored model that is safe to use for raster mapping."""
        self.mapping_model = None
        self.mapping_xcols = []
        self.mapping_wavelengths = []
        self.mapping_source_xcols = []
        self.mapping_source_wavelengths = []
        self.mapping_target_sensor = ""
        self.mapping_sensor_conversion_enabled = False
        self.mapping_model_info = {}

    def _clear_results_display(self):
        if hasattr(self, "results_text"):
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", "No current result. Run a model to generate metrics and plots.\n")
        if hasattr(self, "results_fig"):
            self.results_fig.clear()
            self.results_ax = self.results_fig.add_subplot(111)
            self.results_ax.text(0.1, 0.5, "No current result.")
            self.results_ax.axis("off")
            self.results_canvas.draw_idle()

    def _clear_mapping_display(self):
        if hasattr(self, "mapping_status"):
            self.mapping_status.delete("1.0", tk.END)
            self.mapping_status.insert(
                "1.0",
                "No prediction map yet. Train a model first, then load a multiband GeoTIFF with matching predictor bands.\n",
            )
        if hasattr(self, "map_fig"):
            self.map_fig.clear()
            ax = self.map_fig.add_subplot(111)
            ax.text(0.08, 0.5, "No generated map.")
            ax.axis("off")
            self.map_canvas.draw_idle()

    def reset_analysis_state(self):
        self._suspend_state_callbacks = True
        try:
            self._reset_preprocessing_values()
            self._reset_model_values()
            self.outlier_mask = None
            self.excluded_ranges = []
            self.cwt_regions = []
            self.last_result = None
            self.last_clean_data = None
            self._clear_mapping_model_state()
            if hasattr(self, "range_list"):
                self.refresh_range_list()
            if hasattr(self, "cwt_region_list"):
                self.refresh_cwt_region_list()
            if hasattr(self, "outlier_status"):
                self.outlier_status.configure(text="No outlier analysis run.")
            self._clear_results_display()
            self._clear_mapping_display()
            if hasattr(self, "model_log"):
                self.model_log.delete("1.0", tk.END)
                self.model_log.insert(tk.END, "Analysis reset. Data are still loaded; start again from preprocessing/model selection.\n")
        finally:
            self._suspend_state_callbacks = False
        if self.df is not None and hasattr(self, "pre_canvas"):
            self.plot_spectra()

    def reset_all_workflow(self):
        self._suspend_state_callbacks = True
        try:
            self.file_path = None
            self.excel_file = None
            self.df = None
            self.raster_path = None
            self.last_map_path = None
            self.raster_path_var.set("")
            self.sheet_var.set("")
            self.target_var.set("")
            if hasattr(self, "sheet_combo"):
                self.sheet_combo["values"] = []
            if hasattr(self, "target_combo"):
                self.target_combo["values"] = []
            if hasattr(self, "x_listbox"):
                self.x_listbox.delete(0, tk.END)
            if hasattr(self, "preview_tree"):
                self.preview_tree.delete(*self.preview_tree.get_children())
                self.preview_tree["columns"] = []
            if hasattr(self, "data_info"):
                self.data_info.delete("1.0", tk.END)
                self.data_info.insert("1.0", "Workflow reset. Load a new Excel/CSV file.\n")
            self._reset_preprocessing_values()
            self._reset_model_values()
            self.outlier_mask = None
            self.excluded_ranges = []
            self.cwt_regions = []
            self.last_result = None
            self.last_clean_data = None
            self._clear_mapping_model_state()
            if hasattr(self, "range_list"):
                self.refresh_range_list()
            if hasattr(self, "cwt_region_list"):
                self.refresh_cwt_region_list()
            if hasattr(self, "outlier_status"):
                self.outlier_status.configure(text="No outlier analysis run.")
            if hasattr(self, "pre_fig"):
                self.pre_fig.clear()
                self.pre_ax = self.pre_fig.add_subplot(111)
                self.pre_ax.text(0.1, 0.5, "Load data first.")
                self.pre_ax.axis("off")
                self.pre_canvas.draw_idle()
            self._clear_results_display()
            self._clear_mapping_display()
            if hasattr(self, "model_log"):
                self.model_log.delete("1.0", tk.END)
                self.model_log.insert(tk.END, "Workflow reset. Load data to begin.\n")
            self.notebook.select(self.tab_data)
        finally:
            self._suspend_state_callbacks = False

    def reset_preprocessing_state(self):
        self._suspend_state_callbacks = True
        try:
            self._reset_preprocessing_values()
            self.outlier_mask = None
            self.excluded_ranges = []
            self.cwt_regions = []
            self.last_result = None
            self.last_clean_data = None
            self._clear_mapping_model_state()
            if hasattr(self, "range_list"):
                self.refresh_range_list()
            if hasattr(self, "cwt_region_list"):
                self.refresh_cwt_region_list()
            if hasattr(self, "outlier_status"):
                self.outlier_status.configure(text="Preprocessing reset. No outlier analysis run.")
            self.last_map_path = None
            self._clear_results_display()
            self._clear_mapping_display()
        finally:
            self._suspend_state_callbacks = False
        if self.df is not None:
            self.plot_spectra()

    def reset_outlier_analysis(self):
        self.outlier_mask = None
        self.remove_outliers_var.set(False)
        if hasattr(self, "outlier_status"):
            self.outlier_status.configure(text="Outlier analysis reset. Re-run screening if needed.")
        self.last_result = None
        self.last_clean_data = None
        self._clear_mapping_model_state()
        self.last_map_path = None
        self._clear_results_display()
        self._clear_mapping_display()
        if self.df is not None:
            self.plot_spectra()

    def reset_model_settings(self):
        self._suspend_state_callbacks = True
        try:
            self._reset_model_values()
            self.last_result = None
            self.last_clean_data = None
            self._clear_mapping_model_state()
            self.last_map_path = None
            self._clear_results_display()
            self._clear_mapping_display()
            if hasattr(self, "model_log"):
                self.model_log.insert(tk.END, "Model settings reset.\n")
                self.model_log.see(tk.END)
        finally:
            self._suspend_state_callbacks = False

    def continue_to_model_step(self):
        """Move to model tab after the user has accepted preprocessing/outlier decisions."""
        if self.df is None:
            messagebox.showerror("No data", "Load and accept a data table first.")
            return
        try:
            # Validate current settings before moving forward. This catches empty X selection
            # or destructive band deletion early instead of failing later in modelling.
            self.get_current_xy(apply_outlier_removal=True, apply_band_removal=True)
            self.notebook.select(self.tab_model)
            if hasattr(self, "model_log"):
                status = "REMOVE flagged outliers" if self.remove_outliers_var.get() else "KEEP flagged outliers"
                self.model_log.insert(tk.END, f"Preprocessing accepted. Outlier decision: {status}. Ready to run a model.\n")
                self.model_log.see(tk.END)
        except Exception as e:
            messagebox.showerror("Cannot continue", str(e))

    def keep_outliers_and_continue(self):
        self.remove_outliers_var.set(False)
        if hasattr(self, "outlier_status"):
            if self.outlier_mask is None:
                self.outlier_status.configure(text="Decision saved: keep all samples. No outlier screening has been applied.")
            else:
                self.outlier_status.configure(text=f"Decision saved: keep all samples. {int(self.outlier_mask.sum())} flagged sample(s) will remain in modelling.")
        self.continue_to_model_step()

    def remove_outliers_and_continue(self):
        if self.outlier_mask is None:
            ok = messagebox.askyesno(
                "No outlier analysis",
                "No outlier analysis has been run yet. Continue without removing outliers?",
            )
            if not ok:
                return
            self.remove_outliers_var.set(False)
        else:
            self.remove_outliers_var.set(True)
            if hasattr(self, "outlier_status"):
                self.outlier_status.configure(text=f"Decision saved: remove {int(self.outlier_mask.sum())} flagged sample(s) before modelling.")
        self.continue_to_model_step()

    def reset_results_state(self):
        self.last_result = None
        self.last_clean_data = None
        self._clear_mapping_model_state()
        self.last_map_path = None
        self._clear_results_display()
        self._clear_mapping_display()
        if hasattr(self, "model_log"):
            self.model_log.insert(tk.END, "Results cleared.\n")
            self.model_log.see(tk.END)

    def _reset_preprocessing_values(self):
        self.transform_var.set("Raw")
        self.scale_var.set(True)
        self.sensor_resample_enabled_var.set(False)
        self.target_sensor_var.set("Sentinel-2 MSI")
        self.source_band_file_var.set("")
        self.source_band_centers = []
        self.source_band_center_map = {}
        if hasattr(self, "band_center_status"):
            self.update_band_center_status()
        self.sg_window_var.set(11)
        self.sg_poly_var.set(2)
        self.index_max_pairs_var.set(5000)
        self.cwt_start_var.set("")
        self.cwt_end_var.set("")
        self.cwt_min_scale_var.set(2)
        self.cwt_max_scale_var.set(16)
        self.cwt_num_scales_var.set(8)
        self.cwt_regions = []
        if hasattr(self, "cwt_region_list"):
            self.refresh_cwt_region_list()
        self.remove_outliers_var.set(False)
        self.outlier_method_var.set("PCA Hotelling T2 on X")
        self.outlier_threshold_var.set(3.5)
        self.outlier_contam_var.set(0.05)
        self.pca_conf_var.set(0.975)
        self.band_start_var.set("")
        self.band_end_var.set("")

    def _reset_model_values(self):
        self.model_var.set("PLSR")
        self.cv_method_var.set("KFold")
        self.kfold_var.set(5)
        self.repeats_var.set(3)
        self.test_size_var.set(0.25)
        self.random_state_var.set(42)
        self.pls_comp_var.set(5)
        self.pls_max_comp_var.set(20)
        self.n_trees_var.set(500)
        self.max_depth_var.set(3)
        self.learning_rate_var.set(0.05)
        self.min_leaf_var.set(1)
        self.svr_c_var.set(10.0)
        self.svr_eps_var.set(0.1)
        self.knn_k_var.set(5)

    # ---------------------------- data tab ---------------------------- #

    def _build_data_tab(self):
        frame = self.tab_data
        left = self._make_scrollable_side_panel(frame, width=360)
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._button_with_info(left, "Load Excel/CSV", self.load_file, "Load Excel/CSV", INFO_TEXT["load"])
        self._pack_label_info(left, "Sheet", INFO_TEXT["sheet"], pady=(12, 2))
        self.sheet_combo = ttk.Combobox(left, textvariable=self.sheet_var, state="readonly", width=30)
        self.sheet_combo.pack(fill=tk.X)
        self.sheet_combo.bind("<<ComboboxSelected>>", lambda e: self.read_selected_sheet())

        self._pack_label_info(left, "Y target column", INFO_TEXT["target"], pady=(12, 2))
        self.target_combo = ttk.Combobox(left, textvariable=self.target_var, state="readonly", width=30)
        self.target_combo.pack(fill=tk.X)
        self.target_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_x_columns())

        self._pack_label_info(left, "X reflectance / predictor columns", INFO_TEXT["xcols"], pady=(12, 2))
        self.x_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, width=34, height=20, exportselection=False)
        self.x_listbox.pack(fill=tk.BOTH, expand=True)
        self.x_listbox.bind("<<ListboxSelect>>", lambda e: self.invalidate_from_data_selection())
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Select all", command=self.select_all_x).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns, text="Clear", command=self.clear_x).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self._button_with_info(left, "Accept data selection", self.accept_data_selection, "Accept data selection", INFO_TEXT["accept"])
        ttk.Button(left, text="Reset data step", command=self.reset_all_workflow).pack(fill=tk.X, pady=(8, 4))

        self.data_info = tk.Text(right, height=5, wrap="word")
        self.data_info.pack(fill=tk.X, pady=(0, 8))
        self.preview_tree = ttk.Treeview(right, show="headings")
        yscroll = ttk.Scrollbar(right, orient="vertical", command=self.preview_tree.yview)
        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.LEFT, fill=tk.Y)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Load data",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.file_path = Path(path)
        try:
            if self.file_path.suffix.lower() in {".xlsx", ".xls"}:
                self.excel_file = pd.ExcelFile(self.file_path)
                self.sheet_combo["values"] = self.excel_file.sheet_names
                self.sheet_var.set(self.excel_file.sheet_names[0])
                self.read_selected_sheet()
            else:
                self.excel_file = None
                self.sheet_combo["values"] = ["CSV"]
                self.sheet_var.set("CSV")
                self.df = pd.read_csv(self.file_path)
                self.on_dataframe_loaded()
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load file:\n{e}")

    def read_selected_sheet(self):
        if self.file_path is None:
            return
        try:
            if self.excel_file is not None:
                self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_var.get())
            else:
                self.df = pd.read_csv(self.file_path)
            self.on_dataframe_loaded()
        except Exception as e:
            messagebox.showerror("Sheet error", f"Could not read sheet:\n{e}")

    def on_dataframe_loaded(self):
        assert self.df is not None
        self.df.columns = [str(c).strip() for c in self.df.columns]
        cols = list(self.df.columns)
        self.target_combo["values"] = cols
        default_target = ""
        for c in cols:
            if str(c).strip().lower() in {"mn", "y", "target", "response"}:
                default_target = c
                break
        if not default_target and cols:
            default_target = cols[-1]
        self.target_var.set(default_target)
        self.refresh_x_columns()
        self.update_preview()
        self.write_data_info()

    def refresh_x_columns(self):
        self.x_listbox.delete(0, tk.END)
        if self.df is None:
            return
        target = self.target_var.get()
        cols = numeric_column_names(self.df)
        xcols = [c for c in cols if c != target]
        xcols = sort_by_wavelength(xcols)
        for c in xcols:
            self.x_listbox.insert(tk.END, c)
        self.select_all_x()

    def select_all_x(self):
        self.x_listbox.select_set(0, tk.END)
        self.invalidate_from_data_selection()

    def clear_x(self):
        self.x_listbox.select_clear(0, tk.END)
        self.invalidate_from_data_selection()

    def selected_x_columns(self) -> List[str]:
        return [self.x_listbox.get(i) for i in self.x_listbox.curselection()]

    def accept_data_selection(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=False, apply_band_removal=False)
            msg = f"Accepted {X.shape[0]} samples and {X.shape[1]} X columns for target '{self.target_var.get()}'."
            if len(xcols) > 0:
                msg += f"\nFirst X: {xcols[0]}    Last X: {xcols[-1]}"
            self.data_info.insert(tk.END, "\n" + msg + "\n")
            self.notebook.select(self.tab_pre)
            self.plot_spectra()
        except Exception as e:
            messagebox.showerror("Data selection error", str(e))

    def update_preview(self):
        if self.df is None:
            return
        self.preview_tree.delete(*self.preview_tree.get_children())
        dfp = self.df.head(100).copy()
        # Keep wide tables usable.
        cols = list(dfp.columns[:80])
        self.preview_tree["columns"] = cols
        for c in cols:
            self.preview_tree.heading(c, text=c)
            self.preview_tree.column(c, width=95, stretch=False)
        for _, row in dfp[cols].iterrows():
            vals = ["" if pd.isna(v) else str(v)[:24] for v in row.values]
            self.preview_tree.insert("", tk.END, values=vals)

    def write_data_info(self):
        if self.df is None:
            return
        self.data_info.delete("1.0", tk.END)
        numeric_cols = numeric_column_names(self.df)
        missing = int(self.df.isna().sum().sum())
        text = [
            f"File: {self.file_path}",
            f"Rows: {len(self.df)}    Columns: {self.df.shape[1]}    Numeric columns: {len(numeric_cols)}    Missing cells: {missing}",
            "Default rule: Y is the selected target column; X is selected numeric columns excluding Y.",
            "For reflectance data, wavelength column names should be numeric. Non-numeric names still work, but wavelength-range deletion becomes index-based.",
        ]
        self.data_info.insert("1.0", "\n".join(text))

    # ---------------------------- preprocess tab ---------------------------- #

    def _build_pre_tab(self):
        frame = self.tab_pre
        left = self._make_scrollable_side_panel(frame, width=390)
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._pack_label_info(left, "Spectral preprocessing", INFO_TEXT["preprocess"], font=("Segoe UI", 10, "bold"))
        transform_options = [
            "Raw",
            "SNV",
            "MSC",
            "Log(1/R)",
            "Savitzky-Golay smoothing",
            "First derivative",
            "Second derivative",
            "Continuum removal",
            "Spectral indices (all pairs)",
            "Continuous wavelet removal/features",
        ]
        transform_row = ttk.Frame(left)
        transform_row.pack(fill=tk.X, pady=4)
        ttk.Combobox(transform_row, textvariable=self.transform_var, values=transform_options, state="readonly", width=28).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._info_button(transform_row, "Spectral transform", INFO_TEXT["transform"]).pack(side=tk.LEFT, padx=(5, 0))
        scale_row = ttk.Frame(left)
        scale_row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(scale_row, text="Scale X before modelling", variable=self.scale_var).pack(side=tk.LEFT, anchor="w")
        self._info_button(scale_row, "Scale X", INFO_TEXT["scale"]).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(left, text="Reset preprocessing step", command=self.reset_preprocessing_state).pack(fill=tk.X, pady=(4, 8))

        sg = ttk.LabelFrame(left, text="Savitzky-Golay settings")
        sg.pack(fill=tk.X, pady=8)
        self._info_button(sg, "Savitzky-Golay settings", INFO_TEXT["sg"]).grid(row=0, column=2, rowspan=2, sticky="ne", padx=4, pady=2)
        ttk.Label(sg, text="Window length").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(sg, textvariable=self.sg_window_var, from_=3, to=101, increment=2, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(sg, text="Polynomial order").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(sg, textvariable=self.sg_poly_var, from_=1, to=5, width=8).grid(row=1, column=1, padx=4)

        indices = ttk.LabelFrame(left, text="Spectral indices")
        indices.pack(fill=tk.X, pady=8)
        self._info_button(indices, "Spectral indices", INFO_TEXT["indices"]).grid(row=0, column=2, rowspan=2, sticky="ne", padx=4, pady=2)
        ttk.Label(indices, text="Max band pairs").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(indices, textvariable=self.index_max_pairs_var, width=10).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Label(indices, text="0 = all possible pairs", foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=0)
        ttk.Button(indices, text="Use spectral indices for modelling", command=self.activate_spectral_indices).grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(5, 2))
        ttk.Button(indices, text="Preview transformed features", command=self.plot_spectra).grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ttk.Button(indices, text="Export calculated indices/features CSV", command=self.export_transformed_features).grid(row=4, column=0, columnspan=3, sticky="ew", padx=4, pady=(2, 5))
        ttk.Label(indices, text="The model uses indices only when this preprocessing is active.", foreground="gray", wraplength=310).grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))
        indices.columnconfigure(1, weight=1)

        wave = ttk.LabelFrame(left, text="Continuous wavelet regions")
        wave.pack(fill=tk.X, pady=8)
        self._info_button(wave, "Continuous wavelet removal/features", INFO_TEXT["wavelet"]).grid(row=0, column=3, rowspan=4, sticky="ne", padx=4, pady=2)
        ttk.Label(wave, text="Region").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(wave, textvariable=self.cwt_start_var, width=8).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Entry(wave, textvariable=self.cwt_end_var, width=8).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(wave, text="Add", command=self.add_cwt_region).grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        ttk.Button(wave, text="Plot/select", command=self.plot_cwt_region_selector).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(wave, text="Clear", command=self.clear_cwt_regions).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        ttk.Label(wave, text="Scales min/max/count").grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        scale_row = ttk.Frame(wave)
        scale_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ttk.Entry(scale_row, textvariable=self.cwt_min_scale_var, width=6).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(scale_row, textvariable=self.cwt_max_scale_var, width=6).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Entry(scale_row, textvariable=self.cwt_num_scales_var, width=6).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cwt_region_list = tk.Listbox(wave, height=4)
        self.cwt_region_list.grid(row=4, column=0, columnspan=4, sticky="ew", padx=4, pady=3)
        ttk.Button(wave, text="Remove selected CWT region", command=self.remove_selected_cwt_region).grid(row=5, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        wave.columnconfigure(1, weight=1)
        wave.columnconfigure(2, weight=1)

        bands = ttk.LabelFrame(left, text="Noisy spectral band removal")
        bands.pack(fill=tk.X, pady=8)
        self._button_with_info(bands, "Plot spectra / select ranges", self.plot_spectra, "Noisy spectral band removal", INFO_TEXT["bands"])
        row = ttk.Frame(bands)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.band_start_var, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.band_end_var, width=10).pack(side=tk.LEFT)
        ttk.Button(row, text="Add", command=self.add_manual_band_range).pack(side=tk.LEFT, padx=4)
        ttk.Button(bands, text="Clear ranges", command=self.clear_band_ranges).pack(fill=tk.X, pady=3)
        self.range_list = tk.Listbox(bands, height=5)
        self.range_list.pack(fill=tk.X, pady=3)
        ttk.Button(bands, text="Remove selected range", command=self.remove_selected_band_range).pack(fill=tk.X)

        out = ttk.LabelFrame(left, text="Outlier screening")
        out.pack(fill=tk.X, pady=8)
        out_header = ttk.Frame(out)
        out_header.pack(fill=tk.X)
        ttk.Label(out_header, text="Method").pack(side=tk.LEFT)
        self._info_button(out_header, "Outlier screening", INFO_TEXT["outliers"]).pack(side=tk.LEFT, padx=(5, 0))
        methods = ["None", "Robust z-score on Y", "PCA Hotelling T2 on X", "Isolation Forest on X", "Combined robust Y + PCA X"]
        ttk.Combobox(out, textvariable=self.outlier_method_var, values=methods, state="readonly", width=28).pack(fill=tk.X, pady=3)
        ttk.Label(out, text="Robust z threshold").pack(anchor="w")
        ttk.Entry(out, textvariable=self.outlier_threshold_var).pack(fill=tk.X)
        ttk.Label(out, text="PCA confidence (0.95-0.999)").pack(anchor="w")
        ttk.Entry(out, textvariable=self.pca_conf_var).pack(fill=tk.X)
        ttk.Label(out, text="Isolation contamination").pack(anchor="w")
        ttk.Entry(out, textvariable=self.outlier_contam_var).pack(fill=tk.X)
        ttk.Button(out, text="Run outlier analysis", command=self.run_outlier_analysis).pack(fill=tk.X, pady=5)
        ttk.Button(out, text="Reset outlier analysis", command=self.reset_outlier_analysis).pack(fill=tk.X, pady=2)
        ttk.Checkbutton(out, text="Remove flagged outliers before modelling", variable=self.remove_outliers_var).pack(anchor="w")
        decision_row = ttk.Frame(out)
        decision_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(decision_row, text="Keep", command=lambda: self.remove_outliers_var.set(False)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(decision_row, text="Remove", command=lambda: self.remove_outliers_var.set(True)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(out, text="Continue to model step", command=self.continue_to_model_step).pack(fill=tk.X, pady=3)
        self.outlier_status = ttk.Label(out, text="No outlier analysis run.", wraplength=260)
        self.outlier_status.pack(anchor="w", pady=4)

        sensor = ttk.LabelFrame(right, text="Optional sensor band-centre conversion")
        sensor.pack(fill=tk.X, pady=(0, 8))
        sensor_top = ttk.Frame(sensor)
        sensor_top.pack(fill=tk.X, padx=6, pady=(5, 2))
        ttk.Checkbutton(
            sensor_top,
            text="Convert original spectra to selected sensor band centers before modelling",
            variable=self.sensor_resample_enabled_var,
        ).pack(side=tk.LEFT, anchor="w")
        self._info_button(sensor_top, "Sensor band-centre conversion", INFO_TEXT["sensor_resampling"]).pack(side=tk.LEFT, padx=(5, 0))
        sensor_mid = ttk.Frame(sensor)
        sensor_mid.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(sensor_mid, text="Upload original band centers", command=self.load_source_band_centers).pack(side=tk.LEFT)
        ttk.Button(sensor_mid, text="Clear centers", command=self.clear_source_band_centers).pack(side=tk.LEFT, padx=5)
        self.band_center_status = ttk.Label(sensor_mid, text="Using numeric X column names as source centers.", wraplength=360)
        self.band_center_status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        sensor_bottom = ttk.Frame(sensor)
        sensor_bottom.pack(fill=tk.X, padx=6, pady=(2, 6))
        ttk.Label(sensor_bottom, text="Target sensor").pack(side=tk.LEFT)
        ttk.Combobox(
            sensor_bottom,
            textvariable=self.target_sensor_var,
            values=list(SENSOR_BAND_CENTERS_NM.keys()),
            state="readonly",
            width=28,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(sensor_bottom, text="Preview conversion", command=self.preview_sensor_conversion).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            sensor_bottom,
            text="Disabled = use original uploaded bands unchanged.",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=8)

        decision = ttk.LabelFrame(right, text="Outlier decision and next step")
        decision.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            decision,
            text="After plotting spectra and/or running outlier screening, choose exactly how the model should use the data.",
            wraplength=760,
        ).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(decision, text="Keep flagged outliers → Model", command=self.keep_outliers_and_continue).pack(side=tk.RIGHT, padx=4, pady=6)
        ttk.Button(decision, text="Remove flagged outliers → Model", command=self.remove_outliers_and_continue).pack(side=tk.RIGHT, padx=4, pady=6)
        self._info_button(decision, "Outlier decision", INFO_TEXT["outliers"]).pack(side=tk.RIGHT, padx=4, pady=6)

        self.pre_fig = Figure(figsize=(8.5, 6.5), dpi=100)
        self.pre_ax = self.pre_fig.add_subplot(111)
        self.pre_canvas = FigureCanvasTkAgg(self.pre_fig, master=right)
        self.pre_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.pre_canvas, right)
        toolbar.update()

    def current_spectral_transformer(self, wl: np.ndarray) -> SpectralTransformer:
        return SpectralTransformer(
            self.transform_var.get(),
            wl,
            int(self.sg_window_var.get()),
            int(self.sg_poly_var.get()),
            index_max_pairs=int(self.index_max_pairs_var.get()),
            cwt_regions=list(self.cwt_regions),
            cwt_min_scale=int(self.cwt_min_scale_var.get()),
            cwt_max_scale=int(self.cwt_max_scale_var.get()),
            cwt_num_scales=int(self.cwt_num_scales_var.get()),
        )

    def refresh_cwt_region_list(self):
        if not hasattr(self, "cwt_region_list"):
            return
        self.cwt_region_list.delete(0, tk.END)
        for a, b in self.cwt_regions:
            self.cwt_region_list.insert(tk.END, f"{min(a,b):.6g} – {max(a,b):.6g}")

    def add_cwt_region(self):
        try:
            a = float(self.cwt_start_var.get())
            b = float(self.cwt_end_var.get())
            self.cwt_regions.append((min(a, b), max(a, b)))
            self.refresh_cwt_region_list()
            self.invalidate_from_preprocessing()
            self.plot_cwt_region_selector()
        except Exception:
            messagebox.showerror("CWT region", "Enter numeric start and end values for the wavelet region.")

    def clear_cwt_regions(self):
        self.cwt_regions = []
        self.refresh_cwt_region_list()
        self.invalidate_from_preprocessing()
        if self.df is not None:
            self.plot_cwt_region_selector()

    def remove_selected_cwt_region(self):
        selected = list(self.cwt_region_list.curselection()) if hasattr(self, "cwt_region_list") else []
        if not selected:
            return
        for i in reversed(selected):
            self.cwt_regions.pop(i)
        self.refresh_cwt_region_list()
        self.invalidate_from_preprocessing()
        if self.df is not None:
            self.plot_cwt_region_selector()

    def plot_cwt_region_selector(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=False, apply_band_removal=False)
            Xi = SimpleImputer(strategy="median").fit_transform(X)
            mean = np.nanmean(Xi, axis=0)
            std = np.nanstd(Xi, axis=0)
            self.pre_fig.clear()
            ax = self.pre_fig.add_subplot(111)
            ax.plot(wl, mean, label="Mean raw spectrum")
            ax.fill_between(wl, mean - std, mean + std, alpha=0.2, label="±1 SD")
            for a, b in self.cwt_regions:
                ax.axvspan(min(a, b), max(a, b), alpha=0.25, label="CWT region" if a == self.cwt_regions[0][0] and b == self.cwt_regions[0][1] else None)
            ax.set_xlabel("Wavelength / X column order")
            ax.set_ylabel("Raw reflectance")
            ax.set_title("Draw on the plot to add continuous-wavelet feature regions")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            self.pre_canvas.draw_idle()
            self.cwt_span_selector = SpanSelector(
                ax,
                self.on_cwt_span,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.2),
                interactive=True,
                drag_from_anywhere=True,
            )
        except Exception as e:
            messagebox.showerror("CWT region plot error", str(e))

    def on_cwt_span(self, xmin, xmax):
        if xmin is None or xmax is None or abs(xmax - xmin) < 1e-12:
            return
        self.cwt_regions.append((float(min(xmin, xmax)), float(max(xmin, xmax))))
        self.refresh_cwt_region_list()
        self.invalidate_from_preprocessing()
        self.plot_cwt_region_selector()

    # ---------------------------- sensor band-centre conversion ---------------------------- #

    def update_band_center_status(self):
        if not hasattr(self, "band_center_status"):
            return
        if self.source_band_centers:
            name = Path(self.source_band_file_var.get()).name if self.source_band_file_var.get() else "uploaded file"
            self.band_center_status.configure(text=f"Loaded {len(self.source_band_centers)} source band center(s) from {name}.")
        else:
            self.band_center_status.configure(text="Using numeric X column names as source centers. Upload centers if column names are not true wavelengths.")

    def load_source_band_centers(self):
        path = filedialog.askopenfilename(
            title="Load original sensor band centers",
            filetypes=[
                ("Band center tables", "*.csv *.txt *.xlsx *.xls"),
                ("CSV / text", "*.csv *.txt"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            centers = self._read_band_center_file(Path(path))
            if len(centers) < 2:
                raise RuntimeError("The file must contain at least two numeric band-center values.")
            selected = self.selected_x_columns() if hasattr(self, "x_listbox") else []
            self.source_band_centers = [float(v) for v in centers]
            self.source_band_file_var.set(str(path))
            self.source_band_center_map = {}
            if selected and len(selected) == len(self.source_band_centers):
                self.source_band_center_map = {str(c): float(v) for c, v in zip(selected, self.source_band_centers)}
            elif selected:
                messagebox.showwarning(
                    "Band-center count mismatch",
                    f"Loaded {len(self.source_band_centers)} band centers, but {len(selected)} X columns are selected.\n\n"
                    "The app will use these centers only when the count matches the active X columns. Otherwise it will fall back to numeric X column names.",
                )
            self.update_band_center_status()
            self.invalidate_from_preprocessing()
            try:
                self.preview_sensor_conversion()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Band-center load error", str(e))

    def clear_source_band_centers(self):
        self.source_band_centers = []
        self.source_band_center_map = {}
        self.source_band_file_var.set("")
        self.update_band_center_status()
        self.invalidate_from_preprocessing()
        if self.df is not None:
            try:
                self.plot_spectra()
            except Exception:
                pass

    def _read_band_center_file(self, path: Path) -> List[float]:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            table = pd.read_excel(path)
        else:
            try:
                table = pd.read_csv(path)
            except Exception:
                table = pd.read_csv(path, header=None, sep=None, engine="python")
        if table.empty:
            raise RuntimeError("Band-center file is empty.")
        candidates = [
            "band_center", "band centre", "band_center_nm", "center", "centre", "center_nm",
            "wavelength", "wavelength_nm", "lambda", "nm",
        ]
        lower_map = {str(c).strip().lower(): c for c in table.columns}
        chosen = None
        for c in candidates:
            if c in lower_map:
                chosen = lower_map[c]
                break
        if chosen is None:
            numeric_counts = []
            for c in table.columns:
                vals = pd.to_numeric(table[c], errors="coerce")
                numeric_counts.append((int(vals.notna().sum()), c))
            numeric_counts.sort(reverse=True, key=lambda t: t[0])
            if not numeric_counts or numeric_counts[0][0] == 0:
                raise RuntimeError("No numeric band-center column found. Use one column containing wavelengths in nm.")
            chosen = numeric_counts[0][1]
        vals = pd.to_numeric(table[chosen], errors="coerce").dropna().astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        return [float(v) for v in vals]

    def source_wavelengths_for_columns(self, xcols: Sequence[str]) -> np.ndarray:
        xcols = [str(c) for c in xcols]
        if self.source_band_center_map and all(c in self.source_band_center_map for c in xcols):
            wl = np.array([self.source_band_center_map[c] for c in xcols], dtype=float)
            if len(wl) == len(xcols) and np.all(np.isfinite(wl)):
                return wl
        if self.source_band_centers and len(self.source_band_centers) == len(xcols):
            wl = np.asarray(self.source_band_centers, dtype=float)
            if np.all(np.isfinite(wl)):
                return wl
        wl, is_wl = wavelength_values(xcols)
        if not is_wl and self.sensor_resample_enabled_var.get():
            raise RuntimeError(
                "The selected X column names are not numeric wavelengths. Upload the original sensor band centers first, "
                "or rename the X columns to wavelength values in nm."
            )
        return wl

    def target_sensor_wavelengths(self) -> np.ndarray:
        name = self.target_sensor_var.get()
        vals = SENSOR_BAND_CENTERS_NM.get(name)
        if not vals:
            raise RuntimeError(f"Unknown target sensor preset: {name}")
        return np.asarray(vals, dtype=float)

    def _unique_source_grid(self, source_wl: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        source_wl = np.asarray(source_wl, dtype=float)
        order = np.argsort(source_wl)
        wl_sorted = source_wl[order]
        X_sorted = np.asarray(X, dtype=float)[:, order]
        unique_wl, inverse = np.unique(wl_sorted, return_inverse=True)
        if len(unique_wl) == len(wl_sorted):
            return unique_wl, X_sorted
        Xu = np.empty((X_sorted.shape[0], len(unique_wl)), dtype=float)
        for i in range(len(unique_wl)):
            cols = np.where(inverse == i)[0]
            Xu[:, i] = np.nanmean(X_sorted[:, cols], axis=1)
        return unique_wl, Xu

    def resample_matrix_to_sensor(self, X: np.ndarray, source_wl: Sequence[float], target_wl: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        source = np.asarray(source_wl, dtype=float)
        target = np.asarray(target_wl, dtype=float)
        if len(source) != X.shape[1]:
            raise RuntimeError(f"Source band-center count ({len(source)}) does not match X columns ({X.shape[1]}).")
        if len(source) < 2:
            raise RuntimeError("At least two source band centers are required for interpolation.")
        source, Xs = self._unique_source_grid(source, X)
        if len(source) < 2:
            raise RuntimeError("At least two unique source band centers are required for interpolation.")
        lo, hi = float(np.nanmin(source)), float(np.nanmax(source))
        keep = np.isfinite(target) & (target >= lo) & (target <= hi)
        target_keep = target[keep]
        if len(target_keep) < 2:
            raise RuntimeError(
                f"Fewer than two target bands from {self.target_sensor_var.get()} fall inside the source wavelength range "
                f"({lo:.3g}–{hi:.3g} nm). Choose another target sensor or use wider source spectra."
            )
        out = np.empty((Xs.shape[0], len(target_keep)), dtype=float)
        for r in range(Xs.shape[0]):
            row = Xs[r]
            valid = np.isfinite(row) & np.isfinite(source)
            if valid.sum() < 2:
                out[r, :] = np.nan
            else:
                out[r, :] = np.interp(target_keep, source[valid], row[valid])
        return out, target_keep

    def apply_sensor_conversion_if_needed(self, X: np.ndarray, xcols: List[str], wl: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        if not self.sensor_resample_enabled_var.get():
            return X, xcols, wl
        target_wl = self.target_sensor_wavelengths()
        Xr, target_keep = self.resample_matrix_to_sensor(X, wl, target_wl)
        prefix = self.target_sensor_var.get().replace(" ", "_").replace("/", "_")
        new_cols = [f"{prefix}_{v:g}nm" for v in target_keep]
        return Xr, new_cols, target_keep

    def preview_sensor_conversion(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy_base(apply_outlier_removal=False, apply_band_removal=True)
            Xi = SimpleImputer(strategy="median").fit_transform(X)
            mean_raw = np.nanmean(Xi, axis=0)
            self.pre_fig.clear()
            ax = self.pre_fig.add_subplot(111)
            ax.plot(wl, mean_raw, label=f"Original mean spectrum ({len(wl)} bands)")
            title = "Original source band centres"
            if self.sensor_resample_enabled_var.get():
                Xr, target_keep = self.resample_matrix_to_sensor(Xi, wl, self.target_sensor_wavelengths())
                ax.scatter(target_keep, np.nanmean(Xr, axis=0), label=f"Converted to {self.target_sensor_var.get()} ({len(target_keep)} bands)")
                ax.plot(target_keep, np.nanmean(Xr, axis=0), alpha=0.8)
                title = f"Sensor conversion preview: source → {self.target_sensor_var.get()}"
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            self.pre_canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Sensor conversion preview error", str(e))

    def get_current_xy_base(self, apply_outlier_removal: bool = True, apply_band_removal: bool = True):
        """Return X/Y using original source bands, after optional noisy-band and outlier removal."""
        if self.df is None:
            raise RuntimeError("Load an Excel/CSV file first.")
        target = self.target_var.get()
        xcols = self.selected_x_columns()
        if not target:
            raise RuntimeError("Select a Y target column.")
        if not xcols:
            raise RuntimeError("Select at least one X column.")
        cols = [target] + xcols
        sub = self.df[cols].copy()
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        # Remove samples with missing Y only. X missing is imputed in the model pipeline.
        valid_y = sub[target].notna().to_numpy()
        sub = sub.loc[valid_y].copy()
        row_index = self.df.index[valid_y].to_numpy()
        X = sub[xcols].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)
        xcols_current = list(xcols)
        wl = self.source_wavelengths_for_columns(xcols_current)

        if apply_band_removal and self.excluded_ranges:
            keep = np.ones(len(xcols_current), dtype=bool)
            for a, b in self.excluded_ranges:
                lo, hi = min(a, b), max(a, b)
                keep &= ~((wl >= lo) & (wl <= hi))
            if keep.sum() < 2:
                raise RuntimeError("Band deletion leaves fewer than 2 X columns. Clear or shrink the deleted ranges.")
            X = X[:, keep]
            xcols_current = [c for c, k in zip(xcols_current, keep) if k]
            wl = wl[keep]

        if apply_outlier_removal and self.remove_outliers_var.get() and self.outlier_mask is not None:
            # outlier_mask corresponds to valid-Y rows before outlier removal.
            mask = ~self.outlier_mask
            if len(mask) == len(y):
                X = X[mask]
                y = y[mask]
                row_index = row_index[mask]
        return X, y, xcols_current, wl, row_index

    def get_current_xy(self, apply_outlier_removal: bool = True, apply_band_removal: bool = True):
        X, y, xcols_current, wl, row_index = self.get_current_xy_base(apply_outlier_removal, apply_band_removal)
        X, xcols_current, wl = self.apply_sensor_conversion_if_needed(X, xcols_current, wl)
        return X, y, xcols_current, wl, row_index

    def transformed_matrix_for_plots(self, X: np.ndarray, wl: np.ndarray):
        Xi = SimpleImputer(strategy="median").fit_transform(X)
        transformer = self.current_spectral_transformer(wl)
        transformer.fit(Xi)
        return transformer.transform(Xi), list(transformer.get_feature_names_out([str(v) for v in wl]))

    def activate_spectral_indices(self):
        """Make the spectral-index transform explicit from the indices panel."""
        self.transform_var.set("Spectral indices (all pairs)")
        self.invalidate_from_preprocessing()
        try:
            self.plot_spectra()
        except Exception:
            pass

    def export_transformed_features(self):
        """Export the currently selected transformed predictor matrix for inspection/reuse."""
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=True, apply_band_removal=True)
            Xp, feature_names = self.transformed_matrix_for_plots(X, wl)
            out = pd.DataFrame(Xp, columns=[str(v) for v in feature_names])
            out.insert(0, "source_row", row_index)
            target = self.target_var.get() or "Y"
            out.insert(1, target, y)
            default_name = f"white_ml_box_transformed_features_{self.transform_var.get().replace('/', '_').replace(' ', '_')}.csv"
            path = filedialog.asksaveasfilename(
                title="Save transformed features",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            out.to_csv(path, index=False)
            self.log(f"Exported transformed features: {out.shape[0]} samples × {out.shape[1]-2} predictors -> {path}")
            messagebox.showinfo("Export complete", f"Saved {out.shape[0]} samples and {out.shape[1]-2} transformed predictors.\n\n{path}")
        except Exception as e:
            messagebox.showerror("Feature export error", str(e))

    def plot_spectra(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=False, apply_band_removal=False)
            Xp, feature_names = self.transformed_matrix_for_plots(X, wl)
            mean = np.nanmean(Xp, axis=0)
            std = np.nanstd(Xp, axis=0)
            self.pre_fig.clear()
            self.pre_ax = self.pre_fig.add_subplot(111)
            if Xp.shape[1] == len(wl):
                x_axis = wl
                xlabel = "Wavelength / X column order"
                title = "Draw left-to-right on the plot to mark noisy wavelength ranges for deletion"
                allow_band_span = True
            else:
                x_axis = np.arange(Xp.shape[1], dtype=float)
                xlabel = "Transformed feature index"
                title = f"Preview of {self.transform_var.get()} output ({Xp.shape[1]} predictors). Band deletion is disabled on this preview."
                allow_band_span = False
            self.pre_ax.plot(x_axis, mean, label="Mean transformed spectrum/features")
            self.pre_ax.fill_between(x_axis, mean - std, mean + std, alpha=0.2, label="±1 SD")
            if allow_band_span:
                for a, b in self.excluded_ranges:
                    self.pre_ax.axvspan(min(a, b), max(a, b), alpha=0.25)
            self.pre_ax.set_xlabel(xlabel)
            self.pre_ax.set_ylabel(self.transform_var.get())
            self.pre_ax.set_title(title)
            self.pre_ax.legend(loc="best")
            self.pre_ax.grid(True, alpha=0.25)
            self.pre_canvas.draw_idle()
            if allow_band_span:
                self.spectrum_span_selector = SpanSelector(
                    self.pre_ax,
                    self.on_band_span,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.2),
                    interactive=True,
                    drag_from_anywhere=True,
                )
            else:
                self.spectrum_span_selector = None
        except Exception as e:
            messagebox.showerror("Spectrum plot error", str(e))

    def on_band_span(self, xmin, xmax):
        if xmin is None or xmax is None or abs(xmax - xmin) < 1e-12:
            return
        self.excluded_ranges.append((float(min(xmin, xmax)), float(max(xmin, xmax))))
        self.refresh_range_list()
        self.invalidate_from_preprocessing()
        self.plot_spectra()

    def refresh_range_list(self):
        self.range_list.delete(0, tk.END)
        for a, b in self.excluded_ranges:
            self.range_list.insert(tk.END, f"{min(a,b):.6g} – {max(a,b):.6g}")

    def add_manual_band_range(self):
        try:
            a = float(self.band_start_var.get())
            b = float(self.band_end_var.get())
            self.excluded_ranges.append((min(a, b), max(a, b)))
            self.refresh_range_list()
            self.invalidate_from_preprocessing()
            self.plot_spectra()
        except Exception:
            messagebox.showerror("Band range", "Enter numeric start and end values.")

    def clear_band_ranges(self):
        self.excluded_ranges = []
        self.refresh_range_list()
        self.invalidate_from_preprocessing()
        self.plot_spectra()

    def remove_selected_band_range(self):
        selected = list(self.range_list.curselection())
        if not selected:
            return
        for i in reversed(selected):
            self.excluded_ranges.pop(i)
        self.refresh_range_list()
        self.invalidate_from_preprocessing()
        self.plot_spectra()

    def run_outlier_analysis(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=False, apply_band_removal=True)
            Xp, _feature_names = self.transformed_matrix_for_plots(X, wl)
            Xp = SimpleImputer(strategy="median").fit_transform(Xp)
            method = self.outlier_method_var.get()
            mask = np.zeros(len(y), dtype=bool)
            score = np.zeros(len(y), dtype=float)
            title = "Outlier analysis"
            if method == "None":
                mask[:] = False
                title = "No outlier method selected"
            elif method == "Robust z-score on Y":
                score = np.abs(robust_zscore(y))
                mask = score > float(self.outlier_threshold_var.get())
                title = "Y outliers: robust z-score"
            elif method == "PCA Hotelling T2 on X":
                mask, score = self._pca_hotelling_mask(Xp)
                title = "X spectral outliers: PCA Hotelling T²"
            elif method == "Isolation Forest on X":
                if IsolationForest is None:
                    raise RuntimeError("IsolationForest is unavailable in this scikit-learn installation.")
                contamination = min(max(float(self.outlier_contam_var.get()), 0.001), 0.49)
                iso = IsolationForest(contamination=contamination, random_state=int(self.random_state_var.get()))
                pred = iso.fit_predict(StandardScaler().fit_transform(Xp))
                mask = pred == -1
                score = -iso.score_samples(Xp)
                title = "X spectral outliers: Isolation Forest"
            elif method == "Combined robust Y + PCA X":
                yscore = np.abs(robust_zscore(y))
                ymask = yscore > float(self.outlier_threshold_var.get())
                xmask, xscore = self._pca_hotelling_mask(Xp)
                mask = ymask | xmask
                score = np.maximum(yscore / max(float(self.outlier_threshold_var.get()), 1e-9), xscore / np.nanmax(xscore))
                title = "Combined Y robust z-score + X PCA Hotelling T²"
            else:
                raise RuntimeError(f"Unknown outlier method: {method}")
            self.outlier_mask = mask
            self.outlier_status.configure(text=f"Flagged {int(mask.sum())} outlier(s) out of {len(mask)} samples. Remove-before-modelling is {'ON' if self.remove_outliers_var.get() else 'OFF'}.")
            self.plot_outliers(Xp, y, mask, score, title)
        except Exception as e:
            messagebox.showerror("Outlier analysis error", str(e))

    def _pca_hotelling_mask(self, Xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_comp = min(5, Xp.shape[1], Xp.shape[0] - 1)
        if n_comp < 1:
            return np.zeros(Xp.shape[0], dtype=bool), np.zeros(Xp.shape[0])
        Z = StandardScaler().fit_transform(Xp)
        pca = PCA(n_components=n_comp, random_state=int(self.random_state_var.get()))
        scores = pca.fit_transform(Z)
        var = np.var(scores, axis=0, ddof=1)
        var[var == 0] = 1.0
        t2 = np.sum((scores ** 2) / var, axis=1)
        if chi2 is None:
            cutoff = np.nanpercentile(t2, float(self.pca_conf_var.get()) * 100.0)
        else:
            cutoff = chi2.ppf(float(self.pca_conf_var.get()), df=n_comp)
        return t2 > cutoff, t2

    def plot_outliers(self, Xp: np.ndarray, y: np.ndarray, mask: np.ndarray, score: np.ndarray, title: str):
        self.pre_fig.clear()
        ax = self.pre_fig.add_subplot(111)
        if Xp.shape[0] > 2 and Xp.shape[1] > 1:
            scores = PCA(n_components=2, random_state=int(self.random_state_var.get())).fit_transform(StandardScaler().fit_transform(Xp))
            ax.scatter(scores[~mask, 0], scores[~mask, 1], label="Kept samples")
            if mask.any():
                ax.scatter(scores[mask, 0], scores[mask, 1], marker="x", s=80, label="Flagged outliers")
            ax.set_xlabel("PCA score 1")
            ax.set_ylabel("PCA score 2")
        else:
            idx = np.arange(len(y))
            ax.scatter(idx[~mask], y[~mask], label="Kept samples")
            if mask.any():
                ax.scatter(idx[mask], y[mask], marker="x", s=80, label="Flagged outliers")
            ax.set_xlabel("Sample index")
            ax.set_ylabel(self.target_var.get())
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.pre_canvas.draw_idle()

    # ---------------------------- model tab ---------------------------- #

    def _build_model_tab(self):
        frame = self.tab_model
        left = self._make_scrollable_side_panel(frame, width=360)
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._pack_label_info(left, "Model", INFO_TEXT["model"], font=("Segoe UI", 10, "bold"))
        models = ["PLSR", "GPR", "XGBoost", "Random Forest", "Extra Trees", "SVR", "Gradient Boosting", "KNN"]
        model_row = ttk.Frame(left)
        model_row.pack(fill=tk.X, pady=4)
        ttk.Combobox(model_row, textvariable=self.model_var, values=models, state="readonly", width=28).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._info_button(model_row, "Model selection", INFO_TEXT["model"]).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(left, text="Reset model settings", command=self.reset_model_settings).pack(fill=tk.X, pady=(4, 8))
        if not HAVE_XGBOOST:
            ttk.Label(left, text="XGBoost is optional. Install xgboost if that model is selected.", wraplength=280).pack(anchor="w")

        cvbox = ttk.LabelFrame(left, text="Validation")
        cvbox.pack(fill=tk.X, pady=8)
        self._info_button(cvbox, "Validation", INFO_TEXT["validation"]).grid(row=0, column=2, sticky="ne", padx=4, pady=3)
        ttk.Combobox(
            cvbox,
            textvariable=self.cv_method_var,
            values=["KFold", "Repeated KFold", "Leave-One-Out", "Train/Test split"],
            state="readonly",
            width=25,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=3)
        cvbox.columnconfigure(1, weight=1)
        self._grid_labeled(cvbox, "K folds", self.kfold_var, 1, "Number of folds for K-fold or repeated K-fold cross-validation. Keep k lower than sample size.")
        self._grid_labeled(cvbox, "Repeats", self.repeats_var, 2, "Number of repetitions for Repeated KFold. More repeats stabilize the estimate but take longer.")
        self._grid_labeled(cvbox, "Test size", self.test_size_var, 3, "Fraction of samples reserved for the test set when Train/Test split is selected.")
        self._grid_labeled(cvbox, "Random seed", self.random_state_var, 4, "Controls reproducible splits and stochastic models.")

        pls = ttk.LabelFrame(left, text="PLSR")
        pls.pack(fill=tk.X, pady=8)
        self._info_button(pls, "PLSR", INFO_TEXT["plsr"]).grid(row=0, column=2, rowspan=2, sticky="ne", padx=4, pady=2)
        self._grid_labeled(pls, "Selected components", self.pls_comp_var, 0, "Number of latent variables used by PLSR. The scan can set this automatically using lowest CV RMSE.")
        self._grid_labeled(pls, "Max components scan", self.pls_max_comp_var, 1, "Upper limit for the component scan. It is automatically capped by sample and feature count.")
        ttk.Button(pls, text="Scan PLSR components", command=self.scan_pls_components_button).grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)

        tree = ttk.LabelFrame(left, text="Tree / boosting")
        tree.pack(fill=tk.X, pady=8)
        self._info_button(tree, "Tree / boosting", INFO_TEXT["trees"]).grid(row=0, column=2, rowspan=4, sticky="ne", padx=4, pady=2)
        self._grid_labeled(tree, "Number of trees", self.n_trees_var, 0, "Number of trees/estimators for Random Forest, Extra Trees, XGBoost and Gradient Boosting.")
        self._grid_labeled(tree, "Min leaf", self.min_leaf_var, 1, "Minimum samples in a terminal leaf for Random Forest and Extra Trees. Higher values reduce overfitting.")
        self._grid_labeled(tree, "XGB max depth", self.max_depth_var, 2, "Maximum tree depth for XGBoost. Small values are safer for small spectral datasets.")
        self._grid_labeled(tree, "XGB learning rate", self.learning_rate_var, 3, "XGBoost shrinkage parameter. Lower values usually need more trees.")

        other = ttk.LabelFrame(left, text="Other models")
        other.pack(fill=tk.X, pady=8)
        self._info_button(other, "Other models", INFO_TEXT["other_models"]).grid(row=0, column=2, rowspan=3, sticky="ne", padx=4, pady=2)
        self._grid_labeled(other, "SVR C", self.svr_c_var, 0, "SVR regularization strength. Larger values fit training data more aggressively.")
        self._grid_labeled(other, "SVR epsilon", self.svr_eps_var, 1, "Insensitive loss margin for SVR. Larger epsilon produces smoother fits.")
        self._grid_labeled(other, "KNN k", self.knn_k_var, 2, "Number of neighbours for KNN regression. Larger k smooths predictions.")

        ttk.Button(left, text="Run selected model", command=self.run_model_threaded).pack(fill=tk.X, pady=(12, 4))
        ttk.Button(left, text="Clear results", command=self.reset_results_state).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export last results", command=self.export_results).pack(fill=tk.X, pady=4)

        action = ttk.LabelFrame(right, text="Visible model actions")
        action.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action, text="RUN selected model", command=self.run_model_threaded).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(action, text="Scan PLSR components", command=self.scan_pls_components_button).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(action, text="Clear results", command=self.reset_results_state).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(action, text="Export last results", command=self.export_results).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(action, text="Open Mapping tab", command=lambda: self.notebook.select(self.tab_mapping)).pack(side=tk.LEFT, padx=4, pady=6)
        self._info_button(action, "Run model", "This action bar stays visible. It runs the model selected in the left settings panel using the current preprocessing, band-deletion, outlier decision and validation settings.").pack(side=tk.LEFT, padx=4, pady=6)

        self.model_log = tk.Text(right, height=18, wrap="word")
        self.model_log.pack(fill=tk.BOTH, expand=True)
        self.model_log.insert(tk.END, "Ready. Load data, select bands/outliers, choose the outlier decision, then press RUN selected model.\n")

    def _grid_labeled(self, parent, label, variable, row, info: str = ""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        if info:
            self._info_button(parent, label, info).grid(row=row, column=2, sticky="e", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

    def log(self, text: str):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda text=text: self.log(text))
            return
        self.model_log.insert(tk.END, text + "\n")
        self.model_log.see(tk.END)
        self.update_idletasks()

    def params_from_gui(self) -> Dict[str, float]:
        return {
            "random_state": int(self.random_state_var.get()),
            "n_components": int(self.pls_comp_var.get()),
            "n_estimators": int(self.n_trees_var.get()),
            "min_samples_leaf": int(self.min_leaf_var.get()),
            "max_depth": int(self.max_depth_var.get()),
            "learning_rate": float(self.learning_rate_var.get()),
            "svr_c": float(self.svr_c_var.get()),
            "svr_epsilon": float(self.svr_eps_var.get()),
            "knn_neighbors": int(self.knn_k_var.get()),
        }

    def current_cv(self, n_samples: int):
        method = self.cv_method_var.get()
        if method == "Train/Test split":
            return None, "Train/Test split"
        k = min(max(2, int(self.kfold_var.get())), n_samples)
        return make_cv(method, k, int(self.random_state_var.get()), int(self.repeats_var.get()))

    def scan_pls_components_button(self):
        try:
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=True, apply_band_removal=True)
            cv, cv_name = self.current_cv(len(y))
            if cv is None:
                cv, cv_name = make_cv("KFold", min(max(2, int(self.kfold_var.get())), len(y)), int(self.random_state_var.get()))
            self.log(f"Scanning PLSR components with {cv_name}...")
            scan = scan_plsr_components(
                X,
                y,
                xcols,
                self.transform_var.get(),
                wl,
                int(self.sg_window_var.get()),
                int(self.sg_poly_var.get()),
                bool(self.scale_var.get()),
                cv,
                int(self.pls_max_comp_var.get()),
                index_max_pairs=int(self.index_max_pairs_var.get()),
                cwt_regions=list(self.cwt_regions),
                cwt_min_scale=int(self.cwt_min_scale_var.get()),
                cwt_max_scale=int(self.cwt_max_scale_var.get()),
                cwt_num_scales=int(self.cwt_num_scales_var.get()),
            )
            best = scan.loc[scan["RMSE_CV"].idxmin()]
            self.pls_comp_var.set(int(best["n_components"]))
            self.last_result = RunResult("PLSR scan", cv_name, y, np.full_like(y, np.nan), row_index, {}, None, scan, None, [])
            self.show_plsr_scan(scan)
            self.notebook.select(self.tab_results)
            self.log(f"Best PLSR component count by RMSE: {int(best['n_components'])}; RMSE={best['RMSE_CV']:.5g}")
        except Exception as e:
            messagebox.showerror("PLSR scan error", str(e))
            self.log(traceback.format_exc())

    def run_model_threaded(self):
        t = threading.Thread(target=self.run_model, daemon=True)
        t.start()

    def run_model(self):
        try:
            self.log("Preparing data...")
            X_source, y_source, xcols_source, wl_source, row_index_source = self.get_current_xy_base(apply_outlier_removal=True, apply_band_removal=True)
            X, y, xcols, wl, row_index = self.get_current_xy(apply_outlier_removal=True, apply_band_removal=True)
            if len(y) < 4:
                raise RuntimeError("Too few samples after cleaning. Need at least 4 samples.")
            self.last_clean_data = (X, y, xcols, wl, row_index)
            model_name = self.model_var.get()
            params = self.params_from_gui()
            cv_method = self.cv_method_var.get()

            self.log(f"Running {model_name} on {X.shape[0]} samples and {X.shape[1]} predictors...")
            if self.sensor_resample_enabled_var.get():
                self.log(f"Sensor conversion: original source bands ({X_source.shape[1]}) → {self.target_sensor_var.get()} ({X.shape[1]} retained target bands).")
            else:
                self.log("Sensor conversion: disabled; using original uploaded reflectance bands.")
            self.log(f"Preprocessing: {self.transform_var.get()}; excluded band ranges: {self.excluded_ranges or 'none'}; outliers removed: {self.remove_outliers_var.get()}")

            plsr_scan_df = None
            if model_name == "PLSR":
                cv_for_scan, scan_name = self.current_cv(len(y))
                if cv_for_scan is None:
                    cv_for_scan, scan_name = make_cv("KFold", min(int(self.kfold_var.get()), len(y)), int(self.random_state_var.get()))
                plsr_scan_df = scan_plsr_components(
                    X,
                    y,
                    xcols,
                    self.transform_var.get(),
                    wl,
                    int(self.sg_window_var.get()),
                    int(self.sg_poly_var.get()),
                    bool(self.scale_var.get()),
                    cv_for_scan,
                    int(self.pls_max_comp_var.get()),
                    index_max_pairs=int(self.index_max_pairs_var.get()),
                    cwt_regions=list(self.cwt_regions),
                    cwt_min_scale=int(self.cwt_min_scale_var.get()),
                    cwt_max_scale=int(self.cwt_max_scale_var.get()),
                    cwt_num_scales=int(self.cwt_num_scales_var.get()),
                )
                best = plsr_scan_df.loc[plsr_scan_df["RMSE_CV"].idxmin()]
                self.pls_comp_var.set(int(best["n_components"]))
                params["n_components"] = int(best["n_components"])
                self.log(f"PLSR auto-selected {params['n_components']} components by minimum RMSE.")

            pipe, p_eff = make_pipeline(
                model_name,
                params,
                self.transform_var.get(),
                wl,
                int(self.sg_window_var.get()),
                int(self.sg_poly_var.get()),
                bool(self.scale_var.get()),
                X.shape[1],
                index_max_pairs=int(self.index_max_pairs_var.get()),
                cwt_regions=list(self.cwt_regions),
                cwt_min_scale=int(self.cwt_min_scale_var.get()),
                cwt_max_scale=int(self.cwt_max_scale_var.get()),
                cwt_num_scales=int(self.cwt_num_scales_var.get()),
            )

            self.log(f"Model input after preprocessing/feature engineering: {p_eff} predictor(s).")

            if cv_method == "Train/Test split":
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X,
                    y,
                    row_index,
                    test_size=float(self.test_size_var.get()),
                    random_state=int(self.random_state_var.get()),
                )
                pipe.fit(X_train, y_train)
                y_pred = np.asarray(pipe.predict(X_test)).ravel()
                y_eval = y_test
                sample_eval = idx_test
                cv_name = f"Train/Test split ({1-float(self.test_size_var.get()):.0%}/{float(self.test_size_var.get()):.0%})"
                # Fit a final model on all cleaned data for feature outputs.
                final_model = clone(pipe).fit(X, y)
            else:
                cv, cv_name = self.current_cv(len(y))
                y_pred = np.asarray(cross_val_predict(pipe, X, y, cv=cv)).ravel()
                y_eval = y
                sample_eval = row_index
                final_model = clone(pipe).fit(X, y)

            metrics = regression_metrics(y_eval, y_pred, p_eff)
            notes: List[str] = []
            if not np.isfinite(metrics["Adjusted_R2"]):
                notes.append("Adjusted R² is undefined because effective predictor count is too high for the number of validation samples. This is common in spectral data; do not force it.")
            feature_table = self.model_feature_table(final_model, model_name, xcols, X, y)
            result = RunResult(model_name, cv_name, y_eval, y_pred, sample_eval, metrics, feature_table, plsr_scan_df, final_model, notes)
            self.last_result = result
            self.mapping_model = final_model
            self.mapping_xcols = list(xcols)
            self.mapping_wavelengths = [float(v) if np.isfinite(v) else None for v in np.asarray(wl, dtype=float)]
            self.mapping_source_xcols = list(xcols_source)
            self.mapping_source_wavelengths = [float(v) if np.isfinite(v) else None for v in np.asarray(wl_source, dtype=float)]
            self.mapping_target_sensor = self.target_sensor_var.get()
            self.mapping_sensor_conversion_enabled = bool(self.sensor_resample_enabled_var.get())
            self.mapping_model_info = {
                "model": model_name,
                "validation": cv_name,
                "target": self.target_var.get(),
                "transform": self.transform_var.get(),
                "scale_x": bool(self.scale_var.get()),
                "excluded_ranges": list(self.excluded_ranges),
                "sensor_conversion_enabled": bool(self.sensor_resample_enabled_var.get()),
                "target_sensor": self.target_sensor_var.get(),
                "source_band_center_file": self.source_band_file_var.get(),
                "source_predictor_columns_after_band_removal": list(xcols_source),
                "source_wavelengths_after_band_removal": [float(v) if np.isfinite(v) else None for v in np.asarray(wl_source, dtype=float)],
                "target_predictor_columns_after_conversion": list(xcols),
                "target_wavelengths_after_conversion": [float(v) if np.isfinite(v) else None for v in np.asarray(wl, dtype=float)],
                "index_max_pairs": int(self.index_max_pairs_var.get()),
                "cwt_regions": list(self.cwt_regions),
                "cwt_min_scale": int(self.cwt_min_scale_var.get()),
                "cwt_max_scale": int(self.cwt_max_scale_var.get()),
                "cwt_num_scales": int(self.cwt_num_scales_var.get()),
                "n_source_predictors_after_band_removal": int(X_source.shape[1]),
                "n_predictors_used_for_model_before_transform": int(X.shape[1]),
                "n_model_predictors_after_transform": int(p_eff),
                "training_input_min": float(np.nanmin(X)),
                "training_input_max": float(np.nanmax(X)),
                "training_input_abs_p95": float(np.nanpercentile(np.abs(X[np.isfinite(X)]), 95)) if np.any(np.isfinite(X)) else None,
            }
            self.log("Done. Mapping-ready model stored. You can now use the Mapping tab with a matching multiband GeoTIFF.")
            self.after(0, lambda: self.show_results(result))
        except Exception as e:
            err = str(e)
            self.log(traceback.format_exc())
            self.after(0, lambda err=err: messagebox.showerror("Model error", err))

    def model_feature_table(self, fitted_pipeline: Pipeline, model_name: str, xcols: List[str], X: np.ndarray, y: np.ndarray) -> Optional[pd.DataFrame]:
        model = fitted_pipeline.named_steps.get("model")
        if model is None:
            return None
        transformed_names = list(xcols)
        spectral = fitted_pipeline.named_steps.get("spectral")
        if spectral is not None and hasattr(spectral, "get_feature_names_out"):
            try:
                transformed_names = list(spectral.get_feature_names_out(xcols))
            except Exception:
                transformed_names = list(xcols)
        if model_name == "PLSR":
            return pls_vip(model, transformed_names)
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            names = transformed_names
            if len(names) != len(imp):
                names = [f"feature_{i+1}" for i in range(len(imp))]
            return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        if model_name == "GPR":
            # Permutation importance is calculated on the original input bands because the pipeline transforms internally.
            try:
                n_repeats = 10 if len(y) < 250 else 5
                pi = permutation_importance(fitted_pipeline, X, y, n_repeats=n_repeats, random_state=int(self.random_state_var.get()), n_jobs=-1)
                return pd.DataFrame({"feature": xcols, "permutation_importance_mean": pi.importances_mean, "permutation_importance_sd": pi.importances_std}).sort_values("permutation_importance_mean", ascending=False)
            except Exception:
                return None
        return None

    # ---------------------------- results tab ---------------------------- #

    def _build_results_tab(self):
        frame = self.tab_results
        top = ttk.Frame(frame)
        top.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(top, text="Plot").pack(side=tk.LEFT)
        self._info_button(top, "Results and metrics", INFO_TEXT["results"] + "\n\n" + INFO_TEXT["metrics"]).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Combobox(
            top,
            textvariable=self.result_plot_var,
            values=["Measured vs predicted", "Residuals", "Feature importance / VIP", "PLSR components RMSE"],
            state="readonly",
            width=30,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Refresh plot", command=self.refresh_result_plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear results", command=self.reset_results_state).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export last results", command=self.export_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open Mapping tab", command=lambda: self.notebook.select(self.tab_mapping)).pack(side=tk.LEFT, padx=4)

        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self.results_text = tk.Text(left, wrap="word")
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.results_fig = Figure(figsize=(8, 6), dpi=100)
        self.results_ax = self.results_fig.add_subplot(111)
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, master=right)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.results_canvas, right)
        toolbar.update()

    def show_results(self, result: RunResult):
        self.notebook.select(self.tab_results)
        lines = [
            f"Model: {result.model_name}",
            f"Validation: {result.cv_name}",
            f"Target: {self.target_var.get()}",
            f"Sensor conversion: {'ON → ' + self.mapping_target_sensor if self.mapping_sensor_conversion_enabled else 'OFF'}",
            "",
            "Metrics:",
        ]
        for k, v in result.metrics.items():
            if k == "n":
                lines.append(f"  {k}: {int(v)}")
            else:
                lines.append(f"  {k}: {v:.6g}" if np.isfinite(v) else f"  {k}: undefined")
        if result.notes:
            lines.append("\nNotes:")
            lines.extend(["  - " + n for n in result.notes])
        if result.feature_table is not None:
            lines.append("\nTop feature rows:")
            lines.append(result.feature_table.head(12).to_string(index=False))
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "\n".join(lines))
        self.result_plot_var.set("Measured vs predicted")
        self.refresh_result_plot()

    def show_plsr_scan(self, scan: pd.DataFrame):
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "PLSR component scan\n\n" + scan.to_string(index=False))
        self.result_plot_var.set("PLSR components RMSE")
        self.refresh_result_plot()

    def refresh_result_plot(self):
        if self.last_result is None:
            return
        result = self.last_result
        choice = self.result_plot_var.get()
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)
        if choice == "Measured vs predicted":
            if np.all(~np.isfinite(result.y_pred)):
                ax.text(0.1, 0.5, "No prediction vector for this result.")
            else:
                ax.scatter(result.y_true, result.y_pred)
                lo = float(np.nanmin([np.nanmin(result.y_true), np.nanmin(result.y_pred)]))
                hi = float(np.nanmax([np.nanmax(result.y_true), np.nanmax(result.y_pred)]))
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_xlabel("Measured")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{result.model_name}: measured vs predicted")
                txt = f"R²={result.metrics.get('R2', np.nan):.3f}\nRMSE={result.metrics.get('RMSE', np.nan):.3g}\nBias={result.metrics.get('Bias_mean_pred_minus_obs', np.nan):.3g}"
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top")
        elif choice == "Residuals":
            if np.all(~np.isfinite(result.y_pred)):
                ax.text(0.1, 0.5, "No prediction vector for this result.")
            else:
                residual = result.y_pred - result.y_true
                ax.scatter(result.y_pred, residual)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residual: predicted - measured")
                ax.set_title("Residual plot")
        elif choice == "Feature importance / VIP":
            ft = result.feature_table
            if ft is None or ft.empty:
                ax.text(0.1, 0.5, "No feature table available for this model.")
            else:
                if "VIP" in ft.columns:
                    plot_df = ft.sort_values("VIP", ascending=False).head(25).iloc[::-1]
                    ax.barh(plot_df["feature"].astype(str), plot_df["VIP"])
                    ax.axvline(1.0, linestyle="--")
                    ax.set_xlabel("VIP score")
                    ax.set_title("PLSR variable importance in projection (VIP)")
                elif "importance" in ft.columns:
                    plot_df = ft.sort_values("importance", ascending=False).head(25).iloc[::-1]
                    ax.barh(plot_df["feature"].astype(str), plot_df["importance"])
                    ax.set_xlabel("Importance")
                    ax.set_title("Tree-based feature importance")
                elif "permutation_importance_mean" in ft.columns:
                    plot_df = ft.sort_values("permutation_importance_mean", ascending=False).head(25).iloc[::-1]
                    ax.barh(plot_df["feature"].astype(str), plot_df["permutation_importance_mean"])
                    ax.set_xlabel("Permutation importance")
                    ax.set_title("Permutation importance")
                else:
                    ax.text(0.1, 0.5, "Unknown feature table format.")
        elif choice == "PLSR components RMSE":
            scan = result.plsr_scan
            if scan is None or scan.empty:
                ax.text(0.1, 0.5, "No PLSR scan has been run.")
            else:
                ax.plot(scan["n_components"], scan["RMSE_CV"], marker="o")
                best = scan.loc[scan["RMSE_CV"].idxmin()]
                ax.axvline(best["n_components"], linestyle="--")
                ax.set_xlabel("Number of PLS components")
                ax.set_ylabel("Cross-validated RMSE")
                ax.set_title(f"PLSR component selection: best = {int(best['n_components'])}")
        ax.grid(True, alpha=0.25)
        self.results_fig.tight_layout()
        self.results_canvas.draw_idle()

    def export_results(self):
        if self.last_result is None:
            messagebox.showinfo("Export", "No results to export yet.")
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        folder_path = Path(folder)
        result = self.last_result
        try:
            preds = pd.DataFrame({"sample_index": result.sample_index, "measured": result.y_true, "predicted": result.y_pred})
            preds["residual_pred_minus_obs"] = preds["predicted"] - preds["measured"]
            preds.to_csv(folder_path / "mlbox_predictions.csv", index=False)
            pd.DataFrame([result.metrics]).to_csv(folder_path / "mlbox_metrics.csv", index=False)
            config = {
                "app": APP_NAME,
                "version": VERSION,
                "model": result.model_name,
                "validation": result.cv_name,
                "target": self.target_var.get(),
                "transform": self.transform_var.get(),
                "scale_x": bool(self.scale_var.get()),
                "excluded_ranges": self.excluded_ranges,
                "sensor_conversion_enabled": bool(self.sensor_resample_enabled_var.get()),
                "target_sensor": self.target_sensor_var.get(),
                "source_band_center_file": self.source_band_file_var.get(),
                "index_max_pairs": int(self.index_max_pairs_var.get()),
                "cwt_regions": list(self.cwt_regions),
                "cwt_min_scale": int(self.cwt_min_scale_var.get()),
                "cwt_max_scale": int(self.cwt_max_scale_var.get()),
                "cwt_num_scales": int(self.cwt_num_scales_var.get()),
                "outlier_method": self.outlier_method_var.get(),
                "remove_outliers": bool(self.remove_outliers_var.get()),
                "notes": result.notes,
            }
            (folder_path / "mlbox_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            if result.feature_table is not None:
                result.feature_table.to_csv(folder_path / "mlbox_feature_outputs.csv", index=False)
            if result.plsr_scan is not None:
                result.plsr_scan.to_csv(folder_path / "mlbox_plsr_component_scan.csv", index=False)
            self.results_fig.savefig(folder_path / "mlbox_current_plot.png", dpi=200, bbox_inches="tight")
            messagebox.showinfo("Export complete", f"Results exported to:\n{folder_path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


    # ---------------------------- mapping tab ---------------------------- #

    def _build_mapping_tab(self):
        frame = self.tab_mapping
        left = self._make_scrollable_side_panel(frame, width=380)
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._pack_label_info(left, "Prediction mapping", INFO_TEXT["mapping"], font=("Segoe UI", 10, "bold"))
        ttk.Label(
            left,
            text=(
                "Use this only after a model has been trained. The GeoTIFF must be multiband. "
                "Without sensor conversion, raster bands must match the model X bands. "
                "With sensor conversion, raster bands must match the original source bands after noisy-band deletion; "
                "the app converts pixels internally. Spectral indices and wavelet features are rebuilt internally."
            ),
            wraplength=310,
        ).pack(fill=tk.X, pady=(4, 10))

        self._button_with_info(left, "Load multiband GeoTIFF", self.load_raster_file, "Load raster", INFO_TEXT["mapping"])
        ttk.Label(left, text="Input raster").pack(anchor="w", pady=(8, 2))
        raster_entry = ttk.Entry(left, textvariable=self.raster_path_var, width=42, state="readonly")
        raster_entry.pack(fill=tk.X)

        ttk.Label(left, text="Export folder").pack(anchor="w", pady=(12, 2))
        folder_row = ttk.Frame(left)
        folder_row.pack(fill=tk.X)
        ttk.Entry(folder_row, textvariable=self.map_folder_var, width=28).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="Browse", command=self.choose_map_export_folder).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(left, text="Output file name").pack(anchor="w", pady=(12, 2))
        ttk.Entry(left, textvariable=self.map_output_name_var).pack(fill=tk.X)

        nodata_row = ttk.Frame(left)
        nodata_row.pack(fill=tk.X, pady=(12, 4))
        ttk.Label(nodata_row, text="Output NoData").pack(side=tk.LEFT)
        ttk.Entry(nodata_row, textvariable=self.map_nodata_var, width=12).pack(side=tk.RIGHT)

        ttk.Label(left, text="Raster reflectance scaling").pack(anchor="w", pady=(12, 2))
        ttk.Combobox(
            left,
            textvariable=self.map_scale_mode_var,
            values=[
                "Auto: detect raster 0-10000",
                "No scaling",
                "Divide raster by 10000",
                "Multiply raster by 10000",
            ],
            state="readonly",
            width=34,
        ).pack(fill=tk.X)

        preview_box = ttk.LabelFrame(left, text="Map preview colours")
        preview_box.pack(fill=tk.X, pady=(14, 6))
        ttk.Label(
            preview_box,
            text="These controls affect only the preview. The exported GeoTIFF values are not changed.",
            wraplength=300,
        ).pack(fill=tk.X, padx=6, pady=(5, 5))

        ttk.Label(preview_box, text="Display approach").pack(anchor="w", padx=6, pady=(2, 2))
        ttk.Combobox(
            preview_box,
            textvariable=self.map_preview_render_var,
            values=[
                "Stretch: robust percentiles",
                "Stretch: full min-max",
                "Classify: equal interval",
                "Classify: quantile",
            ],
            state="readonly",
            width=30,
        ).pack(fill=tk.X, padx=6)

        ttk.Label(preview_box, text="Colour ramp").pack(anchor="w", padx=6, pady=(8, 2))
        ttk.Combobox(
            preview_box,
            textvariable=self.map_preview_cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "YlGn", "YlGnBu", "RdYlGn", "Spectral", "terrain", "jet"],
            state="readonly",
            width=30,
        ).pack(fill=tk.X, padx=6)

        pct_row = ttk.Frame(preview_box)
        pct_row.pack(fill=tk.X, padx=6, pady=(8, 2))
        ttk.Label(pct_row, text="Robust range %").pack(side=tk.LEFT)
        ttk.Entry(pct_row, textvariable=self.map_preview_min_pct_var, width=6).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Label(pct_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(pct_row, textvariable=self.map_preview_max_pct_var, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(pct_row, text="Classes").pack(side=tk.LEFT)
        ttk.Entry(pct_row, textvariable=self.map_preview_classes_var, width=5).pack(side=tk.RIGHT)

        ttk.Button(preview_box, text="Refresh preview display", command=self.preview_last_map).pack(fill=tk.X, padx=6, pady=(8, 6))

        ttk.Button(left, text="Generate prediction map", command=self.generate_prediction_map_threaded).pack(fill=tk.X, pady=(14, 4))
        ttk.Button(left, text="Preview generated map", command=self.preview_last_map).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Reset mapping step", command=self.reset_mapping_state).pack(fill=tk.X, pady=4)

        ttk.Label(left, text="Mapping status").pack(anchor="w", pady=(12, 2))
        self.mapping_status = tk.Text(left, width=42, height=16, wrap="word")
        self.mapping_status.pack(fill=tk.BOTH, expand=True)
        self.mapping_status.insert(
            tk.END,
            "No prediction map yet. Train a model first, then load a multiband GeoTIFF with matching predictor bands.\n",
        )

        self.map_fig = Figure(figsize=(8.5, 6.5), dpi=100)
        ax = self.map_fig.add_subplot(111)
        ax.text(0.08, 0.5, "No generated map.")
        ax.axis("off")
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=right)
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.map_canvas, right)
        toolbar.update()

    def mapping_log(self, text: str):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda text=text: self.mapping_log(text))
            return
        if hasattr(self, "mapping_status"):
            self.mapping_status.insert(tk.END, text + "\n")
            self.mapping_status.see(tk.END)
            self.update_idletasks()

    def load_raster_file(self):
        if rasterio is None:
            messagebox.showerror(
                "Raster support missing",
                "Raster mapping needs rasterio. Install it with:\n\npython -m pip install rasterio",
            )
            return
        path = filedialog.askopenfilename(
            title="Load multiband GeoTIFF",
            filetypes=[("GeoTIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        self.raster_path = Path(path)
        self.raster_path_var.set(str(self.raster_path))
        self.last_map_path = None
        try:
            with rasterio.open(self.raster_path) as src:
                desc = [
                    f"Loaded raster: {self.raster_path.name}",
                    f"Size: {src.width} columns x {src.height} rows",
                    f"Bands: {src.count}",
                    f"CRS: {src.crs}",
                    f"NoData: {src.nodata}",
                ]
            self.mapping_log("\n".join(desc))
        except Exception as e:
            messagebox.showerror("Raster load error", str(e))

    def choose_map_export_folder(self):
        folder = filedialog.askdirectory(title="Choose map export folder")
        if folder:
            self.map_folder_var.set(folder)

    def reset_mapping_state(self):
        self.raster_path = None
        self.last_map_path = None
        self.raster_path_var.set("")
        self.map_folder_var.set("")
        self.map_output_name_var.set("white_mlbox_prediction_map.tif")
        self.map_scale_mode_var.set("Auto: detect raster 0-10000")
        self.map_preview_render_var.set("Stretch: robust percentiles")
        self.map_preview_cmap_var.set("viridis")
        self.map_preview_min_pct_var.set(2.0)
        self.map_preview_max_pct_var.set(98.0)
        self.map_preview_classes_var.set(7)
        self._clear_mapping_display()

    def generate_prediction_map_threaded(self):
        t = threading.Thread(target=self.generate_prediction_map, daemon=True)
        t.start()

    def _estimate_raster_input_abs_p95(self, src, indexes, expected_input_features: int, nodatavals) -> Optional[float]:
        """Estimate raster predictor scale from a downsampled read of the selected bands."""
        try:
            max_dim = 256
            scale = max(src.width / max_dim, src.height / max_dim, 1.0)
            out_w = max(1, int(src.width / scale))
            out_h = max(1, int(src.height / scale))
            sample = src.read(indexes=indexes, out_shape=(expected_input_features, out_h, out_w)).astype("float64")
            valid = np.ones(sample.shape[1:], dtype=bool)
            for b in range(expected_input_features):
                band = sample[b]
                valid &= np.isfinite(band)
                nd = nodatavals[b] if b < len(nodatavals) else src.nodata
                if nd is not None:
                    try:
                        nd_float = float(nd)
                        if np.isfinite(nd_float):
                            valid &= band != nd_float
                    except Exception:
                        pass
            vals = sample[:, valid]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            return float(np.nanpercentile(np.abs(vals), 95))
        except Exception:
            return None

    def _resolve_raster_scale_multiplier(self, raster_abs_p95: Optional[float]) -> float:
        """Return the multiplier applied to raster pixels before model prediction."""
        mode = self.map_scale_mode_var.get()
        if mode == "No scaling":
            return 1.0
        if mode == "Divide raster by 10000":
            return 0.0001
        if mode == "Multiply raster by 10000":
            return 10000.0

        # Auto mode: compare raster predictor scale with the trained model input scale.
        train_abs_p95 = self.mapping_model_info.get("training_input_abs_p95")
        try:
            train_abs_p95 = float(train_abs_p95)
        except Exception:
            train_abs_p95 = None

        if train_abs_p95 is not None and raster_abs_p95 is not None:
            # Typical reflectance training data are 0-1, while many Sentinel/Landsat exports
            # store reflectance as scaled integers around 0-10000.
            if train_abs_p95 <= 2.0 and raster_abs_p95 > 10.0:
                return 0.0001
            # Less common opposite case: model was trained on scaled integers, raster is 0-1.
            if train_abs_p95 > 10.0 and raster_abs_p95 <= 2.0:
                return 10000.0
        return 1.0

    def generate_prediction_map(self):
        try:
            if rasterio is None:
                raise RuntimeError("Raster mapping needs rasterio. Install it with: python -m pip install rasterio")
            if self.mapping_model is None or not self.mapping_xcols:
                raise RuntimeError(
                    "No mapping-ready trained model is available.\n\n"
                    "Go to tab 3 Model and press RUN selected model. "
                    "Running outlier analysis, scanning PLSR components, loading a raster, or changing model settings does not create a mapping-ready model."
                )
            if self.raster_path is None or not self.raster_path.exists():
                raise RuntimeError("Load a multiband GeoTIFF first.")

            folder_text = self.map_folder_var.get().strip()
            if not folder_text:
                raise RuntimeError("Choose an export folder before generating the map.")
            out_folder = Path(folder_text)
            out_folder.mkdir(parents=True, exist_ok=True)
            out_name = self.map_output_name_var.get().strip() or "mlbox_prediction_map.tif"
            if not out_name.lower().endswith((".tif", ".tiff")):
                out_name += ".tif"
            out_path = out_folder / out_name
            output_nodata = float(self.map_nodata_var.get())

            xcols = list(self.mapping_xcols)
            model = self.mapping_model
            sensor_convert = bool(self.mapping_sensor_conversion_enabled)
            input_raster_band_mode = "model_predictor_bands"
            do_sensor_resample = False
            input_xcols = xcols
            input_wl = np.asarray(self.mapping_wavelengths, dtype=float)
            target_wl = input_wl

            if sensor_convert:
                source_xcols = list(self.mapping_source_xcols)
                source_wl = np.asarray(self.mapping_source_wavelengths, dtype=float)
                target_wl = np.asarray(self.mapping_wavelengths, dtype=float)
                source_features = len(source_xcols)
                target_features = len(xcols)
                self.mapping_log(
                    f"Preparing prediction map for {self.mapping_target_sensor} model bands ({target_features}). "
                    f"Accepted raster inputs: {target_features} target-sensor band(s) directly, or {source_features} original source band(s) for internal conversion."
                )
            else:
                source_xcols = xcols
                source_wl = input_wl
                source_features = len(xcols)
                target_features = len(xcols)
                self.mapping_log(f"Preparing prediction map with {target_features} model predictor bands...")

            with rasterio.open(self.raster_path) as src:
                if sensor_convert:
                    if src.count == target_features:
                        expected_input_features = target_features
                        input_xcols = xcols
                        input_wl = target_wl
                        input_raster_band_mode = "target_sensor_bands_direct"
                        do_sensor_resample = False
                        self.mapping_log(
                            f"Raster has {src.count} band(s), matching the trained {self.mapping_target_sensor} model bands. "
                            "Using raster bands directly without another sensor conversion."
                        )
                    elif src.count == source_features:
                        expected_input_features = source_features
                        input_xcols = source_xcols
                        input_wl = source_wl
                        input_raster_band_mode = "source_bands_resampled_to_target_sensor"
                        do_sensor_resample = True
                        self.mapping_log(
                            f"Raster has {src.count} original source band(s). Converting pixels internally to "
                            f"{self.mapping_target_sensor} model bands ({target_features})."
                        )
                    else:
                        msg = (
                            "Raster band count does not match the trained target-sensor workflow.\n\n"
                            f"The model was trained after converting {source_features} source band(s) to "
                            f"{target_features} {self.mapping_target_sensor} band(s), but the raster has {src.count} band(s).\n\n"
                            f"Use either a {target_features}-band GeoTIFF in the same target-sensor band order as the trained model, "
                            f"or a {source_features}-band GeoTIFF matching the original source X columns after noisy-band deletion."
                        )
                        raise RuntimeError(msg)
                else:
                    expected_input_features = target_features
                    if src.count != expected_input_features:
                        msg = (
                            "Raster band count does not match the trained model.\n\n"
                            f"Model expects {expected_input_features} predictor band(s), but the raster has {src.count} band(s).\n"
                            "Use a GeoTIFF whose bands are in the exact same order as the X columns used for training, after noisy-band deletion."
                        )
                        raise RuntimeError(msg)

                profile = src.profile.copy()
                profile.update(count=1, dtype="float32", nodata=output_nodata, compress="lzw")
                indexes = list(range(1, expected_input_features + 1))
                nodatavals = src.nodatavals or tuple([src.nodata] * src.count)
                raster_abs_p95 = self._estimate_raster_input_abs_p95(src, indexes, expected_input_features, nodatavals)
                raster_scale_multiplier = self._resolve_raster_scale_multiplier(raster_abs_p95)
                train_abs_p95 = self.mapping_model_info.get("training_input_abs_p95")
                self.mapping_log(
                    "Reflectance scale check: "
                    f"training input abs-P95={train_abs_p95}, raster sample abs-P95={raster_abs_p95}, "
                    f"multiplier applied to raster={raster_scale_multiplier:g}."
                )
                if raster_scale_multiplier == 0.0001:
                    self.mapping_log("Raster values look like scaled reflectance (0-10000). Dividing raster predictors by 10000 before prediction.")
                elif raster_scale_multiplier == 10000.0:
                    self.mapping_log("Raster values look like 0-1 reflectance while the model was trained on scaled values. Multiplying raster predictors by 10000 before prediction.")

                with rasterio.open(out_path, "w", **profile) as dst:
                    total_windows = 0
                    predicted_pixels = 0
                    for _, window in src.block_windows(1):
                        total_windows += 1
                        data = src.read(indexes=indexes, window=window).astype("float64")
                        valid = np.ones(data.shape[1:], dtype=bool)
                        for b in range(expected_input_features):
                            band = data[b]
                            valid &= np.isfinite(band)
                            nd = nodatavals[b] if b < len(nodatavals) else src.nodata
                            if nd is not None:
                                try:
                                    nd_float = float(nd)
                                    if np.isfinite(nd_float):
                                        valid &= band != nd_float
                                    else:
                                        valid &= ~np.isnan(band)
                                except Exception:
                                    pass
                        out = np.full(data.shape[1:], output_nodata, dtype="float32")
                        if np.any(valid):
                            pixel_x = data[:, valid].T
                            if raster_scale_multiplier != 1.0:
                                pixel_x = pixel_x * raster_scale_multiplier
                            if do_sensor_resample:
                                pixel_x, _target_used = self.resample_matrix_to_sensor(pixel_x, input_wl, target_wl)
                            pred = np.asarray(model.predict(pixel_x), dtype="float64").ravel()
                            pred[~np.isfinite(pred)] = output_nodata
                            out[valid] = pred.astype("float32")
                            predicted_pixels += int(valid.sum())
                        dst.write(out, 1, window=window)

            sidecar = out_path.with_name(out_path.stem + "_mapping_config.json")
            config = {
                "app": APP_NAME,
                "version": VERSION,
                "input_raster": str(self.raster_path),
                "output_raster": str(out_path),
                "target": self.mapping_model_info.get("target", self.target_var.get()),
                "model": self.mapping_model_info.get("model", "unknown"),
                "validation": self.mapping_model_info.get("validation", "unknown"),
                "transform": self.mapping_model_info.get("transform", self.transform_var.get()),
                "scale_x": self.mapping_model_info.get("scale_x", bool(self.scale_var.get())),
                "excluded_ranges": self.mapping_model_info.get("excluded_ranges", list(self.excluded_ranges)),
                "sensor_conversion_enabled": self.mapping_model_info.get("sensor_conversion_enabled", bool(self.mapping_sensor_conversion_enabled)),
                "target_sensor": self.mapping_model_info.get("target_sensor", self.mapping_target_sensor),
                "source_band_center_file": self.mapping_model_info.get("source_band_center_file", ""),
                "input_raster_band_mode": input_raster_band_mode,
                "raster_reflectance_scaling_mode": self.map_scale_mode_var.get(),
                "raster_scale_multiplier_applied": raster_scale_multiplier,
                "input_raster_abs_p95_sample": raster_abs_p95,
                "training_input_abs_p95": self.mapping_model_info.get("training_input_abs_p95"),
                "training_input_min": self.mapping_model_info.get("training_input_min"),
                "training_input_max": self.mapping_model_info.get("training_input_max"),
                "source_predictor_columns_in_required_raster_order": list(self.mapping_source_xcols),
                "source_predictor_wavelengths_in_required_raster_order": list(self.mapping_source_wavelengths),
                "index_max_pairs": self.mapping_model_info.get("index_max_pairs", int(self.index_max_pairs_var.get())),
                "cwt_regions": self.mapping_model_info.get("cwt_regions", list(self.cwt_regions)),
                "cwt_min_scale": self.mapping_model_info.get("cwt_min_scale", int(self.cwt_min_scale_var.get())),
                "cwt_max_scale": self.mapping_model_info.get("cwt_max_scale", int(self.cwt_max_scale_var.get())),
                "cwt_num_scales": self.mapping_model_info.get("cwt_num_scales", int(self.cwt_num_scales_var.get())),
                "n_model_predictors_after_transform": self.mapping_model_info.get("n_model_predictors_after_transform"),
                "model_predictor_columns_after_sensor_conversion": list(xcols),
                "model_predictor_wavelengths_after_sensor_conversion": list(self.mapping_wavelengths),
                "output_nodata": output_nodata,
            }
            sidecar.write_text(json.dumps(config, indent=2), encoding="utf-8")

            self.last_map_path = out_path
            self.mapping_log(f"Map created: {out_path}")
            self.mapping_log(f"Traceability file: {sidecar}")
            self.mapping_log(f"Predicted pixels: {predicted_pixels:,}")
            self.after(0, lambda path=out_path: self.plot_map_preview(path))
            self.after(0, lambda: messagebox.showinfo("Mapping complete", f"Prediction map exported to:\n{out_path}"))
        except Exception as e:
            err = str(e)
            self.mapping_log(traceback.format_exc())
            self.after(0, lambda err=err: messagebox.showerror("Mapping error", err))

    def preview_last_map(self):
        if self.last_map_path is None or not self.last_map_path.exists():
            messagebox.showinfo("Preview map", "No generated map to preview yet.")
            return
        self.plot_map_preview(self.last_map_path)

    def _get_preview_percentiles(self) -> Tuple[float, float]:
        """Return safe lower/upper preview percentiles. Used only for visual display."""
        try:
            low = float(self.map_preview_min_pct_var.get())
        except Exception:
            low = 2.0
        try:
            high = float(self.map_preview_max_pct_var.get())
        except Exception:
            high = 98.0
        low = max(0.0, min(49.9, low))
        high = min(100.0, max(50.1, high))
        if low >= high:
            low, high = 2.0, 98.0
        return low, high

    def _get_preview_class_count(self) -> int:
        try:
            n_classes = int(self.map_preview_classes_var.get())
        except Exception:
            n_classes = 7
        return max(2, min(20, n_classes))

    @staticmethod
    def _preview_valid_values(arr) -> np.ndarray:
        values = np.asarray(np.ma.masked_invalid(arr).compressed(), dtype="float64")
        values = values[np.isfinite(values)]
        return values

    @staticmethod
    def _expand_degenerate_range(vmin: float, vmax: float) -> Tuple[float, float]:
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if vmin == vmax:
            pad = abs(vmin) * 0.01 if vmin != 0 else 1.0
            return vmin - pad, vmax + pad
        return float(vmin), float(vmax)

    @staticmethod
    def _unique_boundaries(boundaries: np.ndarray) -> np.ndarray:
        boundaries = np.asarray(boundaries, dtype="float64")
        boundaries = boundaries[np.isfinite(boundaries)]
        boundaries = np.unique(boundaries)
        return boundaries

    def plot_map_preview(self, path: Path):
        if rasterio is None:
            messagebox.showerror("Raster support missing", "Raster preview needs rasterio.")
            return
        try:
            with rasterio.open(path) as src:
                max_dim = 900
                scale = max(src.width / max_dim, src.height / max_dim, 1.0)
                out_w = max(1, int(src.width / scale))
                out_h = max(1, int(src.height / scale))
                arr = src.read(1, out_shape=(out_h, out_w), masked=True)
                nodata = src.nodata
                if nodata is not None:
                    arr = np.ma.masked_where(np.asarray(arr) == nodata, arr)
                arr = np.ma.masked_invalid(arr)

            values = self._preview_valid_values(arr)
            if values.size == 0:
                raise RuntimeError("The map preview contains no valid finite pixels after masking NoData.")

            actual_min = float(np.nanmin(values))
            actual_max = float(np.nanmax(values))
            render_mode = self.map_preview_render_var.get() if hasattr(self, "map_preview_render_var") else "Stretch: robust percentiles"
            cmap_name = self.map_preview_cmap_var.get() if hasattr(self, "map_preview_cmap_var") else "viridis"
            low_pct, high_pct = self._get_preview_percentiles()

            if render_mode == "Stretch: full min-max":
                display_min, display_max = self._expand_degenerate_range(actual_min, actual_max)
                norm = None
                imshow_kwargs = {"cmap": cmap_name, "vmin": display_min, "vmax": display_max}
                preview_note = "full min-max stretch"
            else:
                display_min, display_max = np.nanpercentile(values, [low_pct, high_pct])
                display_min, display_max = self._expand_degenerate_range(float(display_min), float(display_max))
                preview_note = f"robust {low_pct:g}-{high_pct:g}% display range"
                if render_mode == "Classify: equal interval":
                    n_classes = self._get_preview_class_count()
                    boundaries = self._unique_boundaries(np.linspace(display_min, display_max, n_classes + 1))
                    if boundaries.size < 3:
                        boundaries = np.asarray(self._expand_degenerate_range(display_min, display_max))
                    norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
                    imshow_kwargs = {"cmap": cmap_name, "norm": norm}
                    preview_note = f"equal-interval classes, {low_pct:g}-{high_pct:g}% range"
                elif render_mode == "Classify: quantile":
                    n_classes = self._get_preview_class_count()
                    q = np.linspace(low_pct, high_pct, n_classes + 1)
                    boundaries = self._unique_boundaries(np.nanpercentile(values, q))
                    if boundaries.size < 3:
                        boundaries = self._unique_boundaries(np.linspace(display_min, display_max, n_classes + 1))
                    if boundaries.size < 3:
                        boundaries = np.asarray(self._expand_degenerate_range(display_min, display_max))
                    norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
                    imshow_kwargs = {"cmap": cmap_name, "norm": norm}
                    preview_note = f"quantile classes, {low_pct:g}-{high_pct:g}% range"
                else:
                    norm = None
                    imshow_kwargs = {"cmap": cmap_name, "vmin": display_min, "vmax": display_max}

            self.map_fig.clear()
            ax = self.map_fig.add_subplot(111)
            im = ax.imshow(arr, **imshow_kwargs)
            ax.set_title(
                f"Predicted map preview: {path.name}\n"
                f"Actual range {actual_min:.4g} to {actual_max:.4g}; display: {display_min:.4g} to {display_max:.4g}"
            )
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            cbar = self.map_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Predicted {self.target_var.get()}")
            self.map_fig.tight_layout()
            self.map_canvas.draw_idle()
            self.notebook.select(self.tab_mapping)
            self.mapping_log(
                f"Preview display: actual valid range {actual_min:.6g} to {actual_max:.6g}; "
                f"shown as {display_min:.6g} to {display_max:.6g} using {preview_note} and {cmap_name}."
            )
        except Exception as e:
            messagebox.showerror("Preview error", str(e))


def main():
    app = MLBoxApp()
    app.mainloop()


if __name__ == "__main__":
    main()
