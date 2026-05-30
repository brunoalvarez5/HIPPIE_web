"""Per-unit scalar metrics for the post-cluster boxplot section.

Spike times are expected in seconds. The web app stores them in milliseconds in
``nc.spike_times_train`` and divides by 1000 before calling here.

Temporal waveform features (halfwidth, trough-to-peak, slopes, recovery time,
mean |dV|) require a sampling rate to be expressed in physical units (ms,
units/ms). When ``sampling_rate_hz`` is None these columns are emitted as NaN
rather than as sample-count quantities, so that downstream plots and CSVs are
not silently sampling-rate dependent.

References used in the per-unit statistics:
  - Holt, Softky, Koch, Douglas (1996), J. Neurophys. — CV2 of ISIs.
  - Shinomoto et al. (2009), PLOS Comput. Biol. — Local Variation (LV).
  - Hill, Mehta, Kleinfeld (2011), J. Neurosci. — refractory violation rate.
  - Petersen, Buzsáki et al. (2021), eLife — CellExplorer per-unit metrics.
  - Jia, Siegle et al. (2019), J. Neurophys. — neuropixels waveform features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Waveform scalars
# ---------------------------------------------------------------------------

def compute_waveform_scalars(
    waveform: np.ndarray,
    sampling_rate_hz: "float | None" = None,
) -> np.ndarray:
    """8 waveform-shape scalars. Amplitudes are in waveform units; halfwidth,
    trough-to-peak, and slopes are in ms / (units/ms) when ``sampling_rate_hz``
    is given, otherwise NaN. Peak is anchored to the post-trough segment so a
    pre-spike shoulder cannot be mistaken for the repolarization peak.
    """
    w = np.asarray(waveform, dtype=np.float64)
    T = len(w)
    out = np.full(8, np.nan, dtype=np.float32)
    if T < 4:
        return out

    trough_idx = int(np.argmin(w))
    trough_val = float(w[trough_idx])
    if trough_idx < T - 1:
        post = w[trough_idx + 1:]
        peak_idx = trough_idx + 1 + int(np.argmax(post))
    else:
        peak_idx = trough_idx
    peak_val = float(w[peak_idx])
    amp = peak_val - trough_val
    pt_ratio = peak_val / abs(trough_val) if trough_val < 0 else np.nan

    if sampling_rate_hz and sampling_rate_hz > 0 and trough_val < 0:
        ms_per_sample = 1000.0 / float(sampling_rate_hz)
        half = 0.5 * trough_val
        left = trough_idx
        while left > 0 and w[left - 1] <= half:
            left -= 1
        right = trough_idx
        while right < T - 1 and w[right + 1] <= half:
            right += 1
        halfwidth_ms = (right - left) * ms_per_sample
        if peak_idx > trough_idx:
            t2p_ms = (peak_idx - trough_idx) * ms_per_sample
            repol = (peak_val - trough_val) / ((peak_idx - trough_idx) * ms_per_sample)
        else:
            t2p_ms = np.nan
            repol = np.nan
        if peak_idx < T - 1:
            recovery = (w[-1] - peak_val) / ((T - 1 - peak_idx) * ms_per_sample)
        else:
            recovery = np.nan
    else:
        halfwidth_ms = t2p_ms = repol = recovery = np.nan

    out[:] = [peak_val, trough_val, amp, halfwidth_ms, pt_ratio,
              t2p_ms, repol, recovery]
    return out


def compute_waveform_scalars_ext(
    waveform: np.ndarray,
    sampling_rate_hz: "float | None" = None,
) -> np.ndarray:
    """9 extended waveform descriptors. Integrals are sample-area normalized
    by ``|trough_val| * T`` (dimensionless, independent of fs and window).
    """
    w = np.asarray(waveform, dtype=np.float64)
    T = len(w)
    out = np.full(9, np.nan, dtype=np.float32)
    if T < 4:
        return out
    trough_idx = int(np.argmin(w))
    trough_val = float(w[trough_idx])
    if trough_val >= 0:
        return out
    abs_trough = abs(trough_val)
    if trough_idx < T - 1:
        post = w[trough_idx + 1:]
        peak_idx = trough_idx + 1 + int(np.argmax(post))
        peak_val = float(post.max())
    else:
        peak_idx = trough_idx
        peak_val = trough_val
    ahp_ratio = peak_val / abs_trough

    if sampling_rate_hz and sampling_rate_hz > 0:
        ms_per_sample = 1000.0 / float(sampling_rate_hz)
        fwhm_trough_ms = float((w <= 0.5 * trough_val).sum()) * ms_per_sample
        fwhm_peak_ms = (
            float((w >= 0.5 * peak_val).sum()) * ms_per_sample
            if peak_val > 0 else np.nan
        )
        mean_abs_deriv = float(np.abs(np.diff(w)).mean()) / ms_per_sample
        if peak_idx > trough_idx:
            target = trough_val + 0.75 * (peak_val - trough_val)
            crossings = np.where(w[trough_idx:] >= target)[0]
            recovery_75_ms = float(crossings[0]) * ms_per_sample if len(crossings) else np.nan
        else:
            recovery_75_ms = np.nan
    else:
        fwhm_trough_ms = fwhm_peak_ms = mean_abs_deriv = recovery_75_ms = np.nan

    pre_sum = float(np.abs(w[: trough_idx + 1]).sum())
    post_sum = float(np.abs(w[trough_idx + 1:]).sum())
    asym = (post_sum - pre_sum) / (pre_sum + post_sum + 1e-12)
    integral_pre_frac = float(w[: trough_idx + 1].sum()) / (abs_trough * T)
    integral_post_frac = float(w[trough_idx + 1:].sum()) / (abs_trough * T)

    if peak_idx > trough_idx + 2:
        seg = w[trough_idx:peak_idx + 1]
        if seg.std() > 0:
            mono = float(np.corrcoef(seg, np.arange(len(seg)))[0, 1])
        else:
            mono = np.nan
    else:
        mono = np.nan

    out[:] = [ahp_ratio, fwhm_trough_ms, fwhm_peak_ms, asym,
              integral_pre_frac, integral_post_frac, recovery_75_ms,
              mean_abs_deriv, mono]
    return out


# ---------------------------------------------------------------------------
# Spike-time scalars
# ---------------------------------------------------------------------------

def compute_spike_scalars(
    spike_times_sec: np.ndarray,
    refractory_period_ms: float = 2.0,
) -> np.ndarray:
    """8 ISI/firing scalars. ``isi_violation_rate`` is the Hill 2011 form:
    fraction of spikes whose preceding ISI is shorter than the refractory
    period (default 2 ms).
    """
    st = np.asarray(spike_times_sec, dtype=np.float64)
    out = np.full(8, np.nan, dtype=np.float32)
    if len(st) < 3:
        return out
    st = np.sort(st)
    duration = float(st[-1] - st[0])
    if duration <= 0:
        return out
    isis = np.diff(st)
    mean_isi = float(isis.mean())
    if mean_isi <= 0:
        return out
    mean_rate = float(len(st) / duration)
    log_mean_isi = float(np.log(mean_isi))
    cv_isi = float(isis.std() / mean_isi)
    pair_sum = isis[:-1] + isis[1:]
    if len(pair_sum):
        cv2 = float((2 * np.abs(np.diff(isis)) / (pair_sum + 1e-12)).mean())
        lv = float((3 * ((isis[:-1] - isis[1:]) / (pair_sum + 1e-12)) ** 2).mean())
    else:
        cv2 = np.nan
        lv = np.nan
    burst_idx = float((isis < 0.005).mean())
    if duration >= 2.0:
        edges = np.arange(st[0], st[-1] + 1.0, 1.0)
        counts, _ = np.histogram(st, bins=edges)
        fano = float(counts.var() / counts.mean()) if counts.mean() > 0 else np.nan
    else:
        fano = np.nan
    t_ref = refractory_period_ms / 1000.0
    isi_violation_rate = float((isis < t_ref).sum() / len(st))
    out[:] = [mean_rate, log_mean_isi, cv_isi, cv2, lv,
              burst_idx, fano, isi_violation_rate]
    return out


def compute_spike_scalars_ext(spike_times_sec: np.ndarray) -> np.ndarray:
    """9 extended ISI scalars: median, log-ISI skew/kurt, and 6 ISI fraction
    bins that span 0–∞ ms without gaps (<2, 2–10, 10–50, 50–500, 500–1000, ≥1000 ms).
    """
    st = np.asarray(spike_times_sec, dtype=np.float64)
    out = np.full(9, np.nan, dtype=np.float32)
    if len(st) < 4:
        return out
    st = np.sort(st)
    isis = np.diff(st)
    if len(isis) < 2:
        return out
    isi_median = float(np.median(isis))
    log_isi = np.log(isis + 1e-9)
    mu = float(log_isi.mean())
    sd = float(log_isi.std())
    if sd > 0:
        skew = float(((log_isi - mu) ** 3).mean() / (sd ** 3))
        kurt = float(((log_isi - mu) ** 4).mean() / (sd ** 4) - 3.0)
    else:
        skew = np.nan
        kurt = np.nan
    n = len(isis)
    frac_lt_2 = float((isis < 0.002).sum() / n)
    frac_2_10 = float(((isis >= 0.002) & (isis < 0.010)).sum() / n)
    frac_10_50 = float(((isis >= 0.010) & (isis < 0.050)).sum() / n)
    frac_50_500 = float(((isis >= 0.050) & (isis < 0.500)).sum() / n)
    frac_500_1000 = float(((isis >= 0.500) & (isis < 1.000)).sum() / n)
    frac_gt_1 = float((isis >= 1.0).sum() / n)
    out[:] = [isi_median, skew, kurt, frac_lt_2, frac_2_10, frac_10_50,
              frac_50_500, frac_500_1000, frac_gt_1]
    return out


def compute_acg_scalars(spike_times_sec: np.ndarray) -> np.ndarray:
    """10 ACG-derived scalars from an inline 0–50 ms / 1 ms autocorrelogram.

    Decay times are measured from the post-zero ACG peak onward (not from lag 0),
    so a unit whose ACG is low at short lag and peaks later is summarised by the
    actual decay constant rather than by the rising side.
    """
    st = np.asarray(spike_times_sec, dtype=np.float64)
    out = np.full(10, np.nan, dtype=np.float32)
    if len(st) < 4:
        return out
    st = np.sort(st)

    half = 0.050
    bin_w = 0.001
    n_one = int(round(half / bin_w))
    counts = np.zeros(n_one, dtype=np.int64)
    j_max = 0
    for i in range(len(st)):
        if j_max <= i:
            j_max = i + 1
        while j_max < len(st) and st[j_max] - st[i] <= half:
            j_max += 1
        if j_max > i + 1:
            lags = st[i + 1: j_max] - st[i]
            bin_idx = np.minimum((lags / bin_w).astype(np.int64), n_one - 1)
            np.add.at(counts, bin_idx, 1)

    total = counts.sum()
    if total == 0:
        return out
    pdf = counts.astype(np.float64) / total

    early = float(pdf[1:3].sum())
    mid = float(pdf[3:10].sum())
    burst_idx_acg = early / mid if mid > 0 else np.nan
    acg_low_lag_frac = float(pdf[:2].sum())

    pdf_positive = pdf[1:]
    peak = float(pdf_positive.max()) if pdf_positive.size else 0.0
    if peak > 0:
        peak_pos = int(np.argmax(pdf_positive))
        max_lag_ms = float(peak_pos + 1)
        post = pdf_positive[peak_pos:]
        below_50 = np.where(post < 0.5 * peak)[0]
        below_25 = np.where(post < 0.25 * peak)[0]
        decay_50 = float(below_50[0]) if len(below_50) else np.nan
        decay_25 = float(below_25[0]) if len(below_25) else np.nan
    else:
        max_lag_ms = decay_50 = decay_25 = np.nan

    lags_axis = np.arange(n_one, dtype=np.float64)
    mean_l = float((lags_axis * pdf).sum())
    var_l = float(((lags_axis - mean_l) ** 2 * pdf).sum())
    skew_r = (
        float(((lags_axis - mean_l) ** 3 * pdf).sum() / (var_l ** 1.5))
        if var_l > 0 else np.nan
    )

    early_frac = float(pdf[:5].sum())
    mid_frac = float(pdf[5:20].sum())
    late_frac = float(pdf[20:50].sum())
    ratio_3_30 = (
        float(pdf[3] / pdf[30])
        if n_one > 30 and pdf[30] > 0 else np.nan
    )

    out[:] = [burst_idx_acg, acg_low_lag_frac, max_lag_ms, decay_50, decay_25,
              skew_r, early_frac, mid_frac, late_frac, ratio_3_30]
    return out


# ---------------------------------------------------------------------------
# Column names — order matches the function outputs above
# ---------------------------------------------------------------------------

WF_NAMES_BASIC = ["peak_amp", "trough_amp", "peak_minus_trough",
                  "halfwidth_ms", "peak_trough_ratio", "trough_to_peak_ms",
                  "repol_slope_per_ms", "recovery_slope_per_ms"]
WF_NAMES_EXT = ["ahp_ratio", "fwhm_trough_ms", "fwhm_peak_ms",
                "asymmetry_pre_post", "integral_pre_trough_frac",
                "integral_post_trough_frac", "recovery_75_ms",
                "mean_abs_deriv_per_ms", "monotonicity_post"]
ISI_NAMES_BASIC = ["mean_rate", "log_mean_isi", "cv_isi", "cv2_isi", "lv_isi",
                   "burst_index_5ms", "fano_factor_1s", "isi_violation_rate"]
ISI_NAMES_EXT = ["isi_median", "log_isi_skew", "log_isi_kurt",
                 "frac_isi_lt_2ms", "frac_isi_2_10ms", "frac_isi_10_50ms",
                 "frac_isi_50_500ms", "frac_isi_500_1000ms", "frac_isi_gt_1s"]
ACG_NAMES = ["burst_idx_acg", "acg_low_lag_frac", "acg_max_lag_ms",
             "acg_decay_50_ms", "acg_decay_25_ms", "acg_skew_right",
             "early_pair_frac", "mid_pair_frac", "late_pair_frac",
             "ratio_3_to_30"]

WF_NAMES = WF_NAMES_BASIC + WF_NAMES_EXT
SPIKE_NAMES = ISI_NAMES_BASIC + ISI_NAMES_EXT + ACG_NAMES
ALL_NAMES = WF_NAMES + SPIKE_NAMES

# Defaults shown by the boxplot multiselect. One representative per
# physiological axis to keep collinearity low:
#   amplitude, halfwidth, trough-to-peak, AHP, firing rate, ISI regularity,
#   local variation, refractory QC, second-scale variability, ACG peak lag.
CURATED_METRICS = [
    "peak_minus_trough",
    "halfwidth_ms",
    "trough_to_peak_ms",
    "ahp_ratio",
    "mean_rate",
    "cv2_isi",
    "lv_isi",
    "isi_violation_rate",
    "fano_factor_1s",
    "acg_max_lag_ms",
]


def compute_per_unit_table(
    waveforms_raw: np.ndarray,
    spike_times_sec: "list[np.ndarray] | None",
    sampling_rate_hz: "float | None" = None,
) -> pd.DataFrame:
    """One row per unit, columns = WF_NAMES + SPIKE_NAMES (44 cols).

    Spike-time / ACG columns are NaN when ``spike_times_sec`` is None.
    Sampling-rate-dependent waveform columns (halfwidth, trough-to-peak,
    slopes, recovery time, mean |dV|, FWHMs) are NaN when
    ``sampling_rate_hz`` is None or non-positive.
    """
    n_units = len(waveforms_raw)
    if n_units:
        wf_block = np.stack([
            np.concatenate([
                compute_waveform_scalars(w, sampling_rate_hz=sampling_rate_hz),
                compute_waveform_scalars_ext(w, sampling_rate_hz=sampling_rate_hz),
            ])
            for w in waveforms_raw
        ])
    else:
        wf_block = np.zeros((0, len(WF_NAMES)), dtype=np.float32)

    if spike_times_sec is None:
        spike_block = np.full((n_units, len(SPIKE_NAMES)), np.nan, dtype=np.float32)
    else:
        if len(spike_times_sec) != n_units:
            raise ValueError(
                f"spike_times_sec length {len(spike_times_sec)} "
                f"does not match waveforms_raw length {n_units}"
            )
        if n_units:
            spike_block = np.stack([
                np.concatenate([
                    compute_spike_scalars(st),
                    compute_spike_scalars_ext(st),
                    compute_acg_scalars(st),
                ])
                for st in spike_times_sec
            ])
        else:
            spike_block = np.zeros((0, len(SPIKE_NAMES)), dtype=np.float32)

    return pd.DataFrame(
        np.concatenate([wf_block, spike_block], axis=1),
        columns=ALL_NAMES,
    )
