# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.2",
#     "pyarrow>=17",
#     "numpy>=1.26",
#     "scipy>=1.12",
#     "scikit-learn>=1.4",
#     "xgboost>=2.0",
#     "matplotlib>=3.8",
#     "joblib>=1.3",
# ]
# ///
"""
Train a sliding-window oscillation classifier from temporal annotations.

Reads annotations from oscillation_annotations.json (produced by the Dash annotator)
and trains an XGBoost classifier on windowed features.  The trained model can then be
applied to time series of *any* length.

Usage:
    uv run train_oscillation_classifier.py <parquet_file> [--annotations oscillation_annotations.json]

The script:
  1. Loads annotations (per-UID oscillation time regions)
  2. Slides a window across each annotated trace
  3. Extracts features per window (FFT, ACF, spectral entropy, time-domain)
  4. Labels each window: oscillating if it overlaps an annotated region
  5. Trains XGBoost with cross-validation
  6. Exports the model + scaler for inference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 20  # timesteps per window
WINDOW_STEP = 5  # stride
MIN_WINDOW_POINTS = 20  # minimum actual data points in a window
OVERLAP_THRESHOLD = (
    0.5  # fraction of window that must overlap annotation to be "oscillating"
)

FREQ_MIN = 1.0 / 30
FREQ_MAX = 1.0 / 4
ACF_MIN_LAG = 3
ACF_PEAK_MIN_HEIGHT = 0.1
DETREND_METHOD = "rolling_median"
ROLLING_WINDOW = 21
SMOOTHING_WINDOW = 2

SEED = 42
MODEL_FILE = Path("oscillation_model.joblib")
RESULTS_FILE = Path("oscillation_classification_results.parquet")

COLUMNS_NEEDED = [
    "particle",
    "fov",
    "cnr_median",
    "cnr",
    "timestep",
    "stim_exposure",
    "cell_line",
    "optocheck",
]


# ---------------------------------------------------------------------------
# Signal processing helpers (from notebook)
# ---------------------------------------------------------------------------


def detrend_signal(y, method="rolling_median", window=21):
    if method == "rolling_median":
        baseline = (
            pd.Series(y)
            .rolling(window=max(3, window), center=True, min_periods=1)
            .median()
            .values
        )
    elif method == "rolling_mean":
        baseline = (
            pd.Series(y)
            .rolling(window=max(3, window), center=True, min_periods=1)
            .mean()
            .values
        )
    elif method == "median":
        baseline = np.full_like(y, np.nanmedian(y), dtype=float)
    else:
        baseline = np.full_like(y, np.nanmean(y), dtype=float)
    return y - baseline


def smooth_signal(y, window=2):
    if window and window > 1:
        return (
            pd.Series(y)
            .rolling(window=int(window), center=True, min_periods=1)
            .mean()
            .values
        )
    return y


def compute_fft(y, dt=1.0):
    n = len(y)
    w = np.hanning(n) if n > 1 else np.ones(n)
    fft_vals = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft_vals) ** 2
    return freqs, fft_vals, power


def compute_acf(y, max_lag=None):
    y = y - np.mean(y)
    n = len(y)
    if max_lag is None:
        max_lag = n // 2
    acf = np.correlate(y, y, mode="full")
    acf = acf[n - 1 :]
    acf = acf[: max_lag + 1]
    if acf[0] != 0:
        acf = acf / acf[0]
    return acf


# ---------------------------------------------------------------------------
# Feature extraction (per window)
# ---------------------------------------------------------------------------


def extract_window_features(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Extract features from a single window of (timestep, cnr_median) data.

    Returns dict of scalar features, or None if too few points.
    """
    if len(y) < MIN_WINDOW_POINTS:
        return None

    y = np.nan_to_num(y, nan=0.0)
    dt = float(np.median(np.diff(x))) if len(x) > 1 else 1.0

    y_det = detrend_signal(y, DETREND_METHOD, ROLLING_WINDOW)
    y_proc = smooth_signal(y_det, SMOOTHING_WINDOW)
    n = len(y_proc)

    feat: dict[str, float] = {}

    # --- FFT features ---
    freqs, fft_vals, power = compute_fft(y_proc, dt)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

    if np.any(mask):
        power_band = power[mask]
        peak_rel = int(np.argmax(power_band))
        peak_abs = np.where(mask)[0][peak_rel]

        feat["fft_peak_freq"] = float(freqs[peak_abs])
        feat["fft_peak_power"] = float(power_band[peak_rel])
        feat["fft_median_power"] = float(np.median(power_band))
        feat["fft_prom_ratio"] = feat["fft_peak_power"] / max(
            feat["fft_median_power"], 1e-15
        )
        feat["fft_amplitude"] = float(2.0 * np.abs(fft_vals[peak_abs]) / n)

        p_norm = power_band / power_band.sum()
        p_pos = p_norm[p_norm > 0]
        max_ent = np.log2(len(power_band)) if len(power_band) > 1 else 1.0
        feat["spectral_entropy"] = (
            float(-np.sum(p_pos * np.log2(p_pos)) / max_ent) if max_ent > 0 else 1.0
        )
    else:
        feat.update(
            {
                "fft_peak_freq": 0.0,
                "fft_peak_power": 0.0,
                "fft_median_power": 0.0,
                "fft_prom_ratio": 0.0,
                "fft_amplitude": 0.0,
                "spectral_entropy": 1.0,
            }
        )

    # --- ACF features ---
    acf = compute_acf(y_proc)
    acf_search = acf[ACF_MIN_LAG:] if len(acf) > ACF_MIN_LAG else np.array([])
    if len(acf_search) > 2:
        peaks, _ = sp_signal.find_peaks(acf_search, height=ACF_PEAK_MIN_HEIGHT)
        peaks = peaks + ACF_MIN_LAG
    else:
        peaks = np.array([], dtype=int)

    if len(peaks) > 0:
        feat["acf_first_peak_height"] = float(acf[peaks[0]])
        feat["acf_first_peak_lag"] = int(peaks[0])
        feat["acf_n_peaks"] = len(peaks)
        if len(peaks) >= 2:
            gaps = np.diff(peaks).astype(float)
            feat["acf_regularity_cv"] = float(np.std(gaps) / max(np.mean(gaps), 1.0))
        else:
            feat["acf_regularity_cv"] = 1.0
        feat["acf_mean_peak_height"] = float(np.mean(acf[peaks]))
    else:
        feat.update(
            {
                "acf_first_peak_height": 0.0,
                "acf_first_peak_lag": 0,
                "acf_n_peaks": 0,
                "acf_regularity_cv": 1.0,
                "acf_mean_peak_height": 0.0,
            }
        )

    # --- Time-domain features ---
    feat["zero_crossing_rate"] = float(np.sum(np.diff(np.sign(y_proc)) != 0) / n)
    feat["signal_std"] = float(np.std(y_proc))
    feat["signal_range"] = float(np.ptp(y_proc))
    feat["signal_mean_abs"] = float(np.mean(np.abs(y_proc)))

    return feat


# ---------------------------------------------------------------------------
# Sliding window + labeling
# ---------------------------------------------------------------------------


def window_overlaps_regions(
    w_start: float, w_end: float, regions: list[list[float]]
) -> float:
    """Fraction of the window [w_start, w_end] that overlaps with annotated regions."""
    w_len = w_end - w_start
    if w_len <= 0:
        return 0.0
    overlap = 0.0
    for r_start, r_end in regions:
        ov_start = max(w_start, r_start)
        ov_end = min(w_end, r_end)
        if ov_end > ov_start:
            overlap += ov_end - ov_start
    return overlap / w_len


def build_windowed_dataset(
    df: pd.DataFrame,
    annotations: dict,
) -> pd.DataFrame:
    """Slide windows across all annotated UIDs, extract features, label each window."""
    rows = []

    annotated_uids = [uid for uid in annotations if uid in df["uid"].unique()]
    print(f"Processing {len(annotated_uids)} annotated UIDs...")

    for uid in annotated_uids:
        ann = annotations[uid]
        regions = ann.get("regions", [])
        is_no_osc = ann.get("no_osc", False)

        sub = df[df["uid"] == uid].sort_values("timestep")
        x = sub["timestep"].values.astype(float)
        y = sub["cnr_median"].values.astype(float)
        meta_row = sub.iloc[0]

        t_min, t_max = x.min(), x.max()

        # Slide window
        w_start = t_min
        while w_start + WINDOW_SIZE <= t_max + WINDOW_STEP:
            w_end = w_start + WINDOW_SIZE
            mask = (x >= w_start) & (x < w_end)
            x_win = x[mask]
            y_win = y[mask]

            if len(x_win) < MIN_WINDOW_POINTS:
                w_start += WINDOW_STEP
                continue

            feat = extract_window_features(x_win, y_win)
            if feat is None:
                w_start += WINDOW_STEP
                continue

            # Label
            if is_no_osc:
                label = 0
            else:
                overlap_frac = window_overlaps_regions(w_start, w_end, regions)
                label = 1 if overlap_frac >= OVERLAP_THRESHOLD else 0

            feat["uid"] = uid
            feat["window_start"] = w_start
            feat["window_end"] = w_end
            feat["label"] = label
            feat["overlap_frac"] = overlap_frac if not is_no_osc else 0.0

            # Metadata
            feat["cell_line"] = str(meta_row.get("cell_line", ""))
            feat["stim_exposure"] = (
                float(meta_row["stim_exposure"])
                if "stim_exposure" in meta_row
                and pd.notna(meta_row.get("stim_exposure"))
                else None
            )

            rows.append(feat)
            w_start += WINDOW_STEP

    result = pd.DataFrame(rows)
    print(
        f"Generated {len(result)} windows "
        f"({result['label'].sum()} oscillating, {(result['label'] == 0).sum()} not)"
    )
    return result


# ---------------------------------------------------------------------------
# Inference: apply to new data
# ---------------------------------------------------------------------------


def predict_trace(
    x: np.ndarray,
    y: np.ndarray,
    clf: XGBClassifier,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> list[dict]:
    """Slide window across a single trace and return per-window predictions."""
    results = []
    t_min, t_max = x.min(), x.max()

    w_start = t_min
    while w_start + WINDOW_SIZE <= t_max + WINDOW_STEP:
        w_end = w_start + WINDOW_SIZE
        mask = (x >= w_start) & (x < w_end)
        x_win, y_win = x[mask], y[mask]

        if len(x_win) < MIN_WINDOW_POINTS:
            w_start += WINDOW_STEP
            continue

        feat = extract_window_features(x_win, y_win)
        if feat is None:
            w_start += WINDOW_STEP
            continue

        feat_vec = np.array([[feat.get(c, 0.0) for c in feature_cols]])
        feat_scaled = scaler.transform(feat_vec)
        prob = clf.predict_proba(feat_scaled)[0, 1]

        results.append(
            {
                "window_start": w_start,
                "window_end": w_end,
                "osc_probability": float(prob),
                "predicted_osc": int(prob >= 0.5),
            }
        )
        w_start += WINDOW_STEP

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train oscillation classifier")
    parser.add_argument("parquet", help="Path to parquet file")
    parser.add_argument(
        "--annotations", type=str, default="oscillation_annotations.json"
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=WINDOW_STEP)
    args = parser.parse_args()

    # Override module-level constants via args
    import train_oscillation_classifier as _self

    _self.WINDOW_SIZE = args.window_size
    _self.WINDOW_STEP = args.window_step

    # --- Load data ---
    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(args.parquet).names)
    cols = [c for c in COLUMNS_NEEDED if c in schema_cols]
    df = pd.read_parquet(args.parquet, columns=cols)
    if "cnr_median" not in df.columns and "cnr" in df.columns:
        df.rename(columns={"cnr": "cnr_median"}, inplace=True)
    df["uid"] = df["fov"].astype(str) + "_" + df["particle"].astype(str)
    df.sort_values(["uid", "timestep"], inplace=True)
    print(f"Loaded {len(df)} rows, {df['uid'].nunique()} UIDs")

    # --- Load annotations ---
    ann_path = Path(args.annotations)
    if not ann_path.exists():
        print(f"ERROR: annotations file not found: {ann_path}")
        print("Run the annotator first: uv run oscillation_annotator.py <parquet>")
        return
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    n_with_regions = sum(1 for a in annotations.values() if a.get("regions"))
    n_no_osc = sum(1 for a in annotations.values() if a.get("no_osc"))
    print(
        f"Annotations: {len(annotations)} total | {n_with_regions} with regions | {n_no_osc} no-osc"
    )

    if len(annotations) < 20:
        print("Need at least ~20 annotations. Run the annotator first.")
        return

    # --- Build windowed dataset ---
    windows_df = build_windowed_dataset(df, annotations)
    if len(windows_df) == 0:
        print("No windows generated. Check your annotations.")
        return

    # --- Feature columns ---
    FEATURE_COLS = [
        c
        for c in windows_df.columns
        if c
        not in (
            "uid",
            "window_start",
            "window_end",
            "label",
            "overlap_frac",
            "cell_line",
            "stim_exposure",
        )
    ]
    print(f"\nUsing {len(FEATURE_COLS)} features: {FEATURE_COLS}")

    X = windows_df[FEATURE_COLS].fillna(0).values
    y = windows_df["label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\nTraining set: {len(y)} windows | {n_pos} oscillating | {n_neg} not")

    if n_pos == 0 or n_neg == 0:
        print("ERROR: need both positive and negative examples.")
        return

    # --- Train ---
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=SEED,
        eval_metric="logloss",
    )

    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        print(
            f"Cross-validation accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
        )

        cv_auc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
        print(f"Cross-validation AUC:      {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    clf.fit(X_scaled, y)

    # Classification report
    y_pred = clf.predict(X_scaled)
    print("\nClassification report (training data):")
    print(
        classification_report(
            y, y_pred, target_names=["not oscillating", "oscillating"]
        )
    )

    # Feature importance
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=True
    )
    fig, ax = plt.subplots(figsize=(7, max(4, len(FEATURE_COLS) * 0.3)))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Feature Importance (XGBoost)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("oscillation_feature_importance.png", dpi=150)
    print("Saved feature importance plot: oscillation_feature_importance.png")

    # --- Save model ---
    model_data = {
        "clf": clf,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "config": {
            "freq_min": FREQ_MIN,
            "freq_max": FREQ_MAX,
            "acf_min_lag": ACF_MIN_LAG,
            "acf_peak_min_height": ACF_PEAK_MIN_HEIGHT,
            "detrend_method": DETREND_METHOD,
            "rolling_window": ROLLING_WINDOW,
            "smoothing_window": SMOOTHING_WINDOW,
            "min_window_points": MIN_WINDOW_POINTS,
            "overlap_threshold": OVERLAP_THRESHOLD,
        },
    }
    joblib.dump(model_data, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

    # --- Predict all UIDs ---
    print("\nPredicting all UIDs...")
    all_results = []
    for uid, grp in df.groupby("uid"):
        grp = grp.sort_values("timestep")
        x = grp["timestep"].values.astype(float)
        y_sig = grp["cnr_median"].values.astype(float)
        preds = predict_trace(x, y_sig, clf, scaler, FEATURE_COLS)
        for p in preds:
            p["uid"] = uid
            row0 = grp.iloc[0]
            p["cell_line"] = str(row0.get("cell_line", ""))
            p["stim_exposure"] = (
                float(row0["stim_exposure"])
                if "stim_exposure" in row0 and pd.notna(row0.get("stim_exposure"))
                else None
            )
        all_results.extend(preds)

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(RESULTS_FILE, index=False)
    print(f"Saved predictions ({len(results_df)} windows) to {RESULTS_FILE}")

    # --- Summary ---
    uid_osc = results_df.groupby("uid")["predicted_osc"].max()
    n_osc_uids = int(uid_osc.sum())
    print(
        f"\nSummary: {n_osc_uids}/{len(uid_osc)} UIDs "
        f"({100*n_osc_uids/len(uid_osc):.1f}%) have at least one oscillating window"
    )

    # Per-condition breakdown
    uid_summary = (
        results_df.groupby("uid")
        .agg(
            max_prob=("osc_probability", "max"),
            mean_prob=("osc_probability", "mean"),
            n_osc_windows=("predicted_osc", "sum"),
            n_windows=("predicted_osc", "count"),
            cell_line=("cell_line", "first"),
            stim_exposure=("stim_exposure", "first"),
        )
        .reset_index()
    )
    uid_summary["has_osc"] = uid_summary["n_osc_windows"] > 0

    if "stim_exposure" in uid_summary.columns:
        cond_summary = (
            uid_summary.groupby("stim_exposure")
            .agg(
                n_cells=("uid", "count"),
                n_osc=("has_osc", "sum"),
            )
            .reset_index()
        )
        cond_summary["pct_osc"] = 100 * cond_summary["n_osc"] / cond_summary["n_cells"]
        print("\nPer-condition breakdown:")
        print(cond_summary.to_string(index=False))

    plt.show()


if __name__ == "__main__":
    main()
