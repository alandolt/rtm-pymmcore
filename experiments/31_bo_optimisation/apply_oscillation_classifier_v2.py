# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.2",
#     "pyarrow>=17",
#     "numpy>=1.26",
#     "scipy>=1.12",
#     "scikit-learn>=1.4",
#     "xgboost>=2.0",
#     "joblib>=1.3",
# ]
# ///
"""
Apply a trained oscillation classifier to a new dataset.

Loads the model exported by train_oscillation_classifier.py and runs
sliding-window inference on every UID in the input parquet file.

Usage:
    uv run apply_oscillation_classifier.py <parquet_file> [--model oscillation_model.joblib] [--output results.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Signal processing helpers (must match training script)
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
# Feature extraction (per window) — identical to training script
# ---------------------------------------------------------------------------


def extract_window_features(x, y, cfg):
    if len(y) < cfg["min_window_points"]:
        return None

    y = np.nan_to_num(y, nan=0.0)
    dt = float(np.median(np.diff(x))) if len(x) > 1 else 1.0

    y_det = detrend_signal(y, cfg["detrend_method"], cfg["rolling_window"])
    y_proc = smooth_signal(y_det, cfg["smoothing_window"])
    n = len(y_proc)

    feat = {}

    # --- FFT features ---
    freqs, fft_vals, power = compute_fft(y_proc, dt)
    mask = (freqs >= cfg["freq_min"]) & (freqs <= cfg["freq_max"])

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
    acf_search = (
        acf[cfg["acf_min_lag"] :] if len(acf) > cfg["acf_min_lag"] else np.array([])
    )
    if len(acf_search) > 2:
        peaks, _ = sp_signal.find_peaks(acf_search, height=cfg["acf_peak_min_height"])
        peaks = peaks + cfg["acf_min_lag"]
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
# Inference
# ---------------------------------------------------------------------------


def predict_trace(x, y, clf, scaler, feature_cols, cfg):
    window_size = cfg["window_size"]
    window_step = cfg["window_step"]
    t_min, t_max = x.min(), x.max()

    # Collect all window features first, then batch-predict
    windows = []
    feat_rows = []

    w_start = t_min
    while w_start + window_size <= t_max + window_step:
        w_end = w_start + window_size
        mask = (x >= w_start) & (x < w_end)
        x_win, y_win = x[mask], y[mask]

        if len(x_win) >= cfg["min_window_points"]:
            feat = extract_window_features(x_win, y_win, cfg)
            if feat is not None:
                windows.append((w_start, w_end))
                feat_rows.append([feat.get(c, 0.0) for c in feature_cols])

        w_start += window_step

    if not feat_rows:
        return []

    feat_matrix = scaler.transform(np.array(feat_rows))
    probs = clf.predict_proba(feat_matrix)[:, 1]

    return [
        {
            "window_start": ws,
            "window_end": we,
            "osc_probability": float(p),
            "predicted_osc": int(p >= 0.5),
        }
        for (ws, we), p in zip(windows, probs)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained oscillation classifier to a new parquet dataset",
    )
    parser.add_argument("parquet", help="Path to input parquet file")
    parser.add_argument(
        "--model",
        type=str,
        default="oscillation_model.joblib",
        help="Path to trained model file (default: oscillation_model.joblib)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (default: <input>_oscillation_results.parquet)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for oscillation (default: 0.5)",
    )
    args = parser.parse_args()

    # --- Load model ---
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        print("Train a model first: uv run train_oscillation_classifier.py <parquet>")
        return

    model_data = joblib.load(model_path)
    clf = model_data["clf"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]

    cfg = model_data["config"]
    cfg["window_size"] = model_data["window_size"]
    cfg["window_step"] = model_data["window_step"]

    print(f"Loaded model from {model_path}")
    print(f"  Window: {cfg['window_size']} steps, stride {cfg['window_step']}")
    print(f"  Features: {feature_cols}")

    # --- Load data ---
    import pyarrow.parquet as pq

    columns_needed = [
        "particle",
        "fov",
        "cnr_median",
        "cnr",
        "timestep",
        "stim_exposure",
        "cell_line",
        "optocheck",
    ]
    schema_cols = set(pq.read_schema(args.parquet).names)
    cols = [c for c in columns_needed if c in schema_cols]
    df = pd.read_parquet(args.parquet, columns=cols)
    if "cnr_median" not in df.columns and "cnr" in df.columns:
        df.rename(columns={"cnr": "cnr_median"}, inplace=True)
    df["uid"] = df["fov"].astype(str) + "_" + df["particle"].astype(str)
    df.sort_values(["uid", "timestep"], inplace=True)
    print(f"Loaded {len(df)} rows, {df['uid'].nunique()} UIDs")

    # --- Predict ---
    threshold = args.threshold
    all_results = []
    meta_cols = [
        c
        for c in ["cell_line", "stim_exposure", "fov", "particle", "optocheck"]
        if c in df.columns
    ]
    groups = df.groupby("uid", sort=False)
    n_uids = len(groups)
    for i, (uid, grp) in enumerate(groups):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing UID {i + 1}/{n_uids}...")
        grp = grp.sort_values("timestep")
        x = grp["timestep"].values.astype(float)
        y = grp["cnr_median"].values.astype(float)
        preds = predict_trace(x, y, clf, scaler, feature_cols, cfg)
        row0 = grp.iloc[0]
        meta = {col: row0[col] for col in meta_cols}
        meta["uid"] = uid
        if not preds:
            all_results.append(
                {
                    **meta,
                    "window_start": np.nan,
                    "window_end": np.nan,
                    "osc_probability": 0.0,
                    "predicted_osc": 0,
                }
            )
            continue
        for p in preds:
            p["predicted_osc"] = int(p["osc_probability"] >= threshold)
            p.update(meta)
        all_results.extend(preds)

    results_df = pd.DataFrame(all_results)

    # --- Output ---
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.parquet).with_name(
            Path(args.parquet).stem + "_oscillation_results.parquet"
        )
    results_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(results_df)} window predictions to {out_path}")

    # --- Summary ---
    uid_osc = results_df.groupby("uid")["predicted_osc"].max()
    n_osc = int(uid_osc.sum())
    n_total = len(uid_osc)
    print(
        f"\nSummary: {n_osc}/{n_total} UIDs ({100 * n_osc / n_total:.1f}%) "
        f"have at least one oscillating window (threshold={threshold})"
    )

    if "stim_exposure" in results_df.columns:
        uid_summary = (
            results_df.groupby("uid")
            .agg(
                max_prob=("osc_probability", "max"),
                n_osc_windows=("predicted_osc", "sum"),
                n_windows=("window_start", lambda x: x.notna().sum()),
                stim_exposure=("stim_exposure", "first"),
            )
            .reset_index()
        )
        uid_summary["has_osc"] = uid_summary["n_osc_windows"] > 0
        cond = (
            uid_summary.groupby("stim_exposure")
            .agg(
                n_cells=("uid", "count"),
                n_osc=("has_osc", "sum"),
            )
            .reset_index()
        )
        cond["pct_osc"] = 100 * cond["n_osc"] / cond["n_cells"]
        print("\nPer-condition breakdown:")
        print(cond.to_string(index=False))


if __name__ == "__main__":
    main()
