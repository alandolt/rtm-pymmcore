# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.10",
#     "matplotlib==3.11.0",
#     "numpy==2.5.0",
#     "polars==1.41.2",
#     "scipy==1.18.0",
# ]
# ///
import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy.stats import qmc

    BUDGET_MS = 4000
    FIRST_FRAME_STIM = 10
    LAST_FRAME_STIM = 70

    def pulse_positions(pulse_interval, first=FIRST_FRAME_STIM, last=LAST_FRAME_STIM):
        """Frame indices where pulses fire: range(first, last, pulse_interval)."""
        return np.arange(first, last, int(pulse_interval))

    def shape_normalized_time(n):
        """n pulse slots -> positions in [0, 1] (independent of n)."""
        if n <= 1:
            return np.array([0.5])
        return np.linspace(0.0, 1.0, n)

    def shape_basis_poly(t, K=3):
        """Legendre P1..PK on [0, 1] (constant P0 dropped): tilt, curvature, cubic."""
        x = 2 * t - 1
        P = [np.ones_like(x), x]
        for k in range(2, K + 1):
            P.append(((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k)
        return np.column_stack([P[k] for k in range(1, K + 1)])

    def shape_exposures(coeffs, n_pulses, budget=BUDGET_MS):
        """3-coef polynomial -> per-pulse exposures, softmax-normalised to budget."""
        if n_pulses <= 0:
            return np.array([])
        t = shape_normalized_time(n_pulses)
        logits = shape_basis_poly(t, 3) @ np.asarray(coeffs, dtype=float)
        w = np.exp(logits - logits.max())
        return budget * w / w.sum()

    def compose_pattern(
        segments, total_budget=BUDGET_MS, first=FIRST_FRAME_STIM, last=LAST_FRAME_STIM
    ):
        """Build a composed pattern from multiple segments.

        Each segment is a dict with keys: coeffs (3,), pi (int), budget_frac (float).
        budget_frac values are normalised so they sum to 1.
        The frame window [first, last] is split into len(segments) equal parts.
        Returns combined (frames, expo) arrays.
        """
        n_seg = len(segments)
        window = last - first
        seg_len = window / n_seg
        fracs = np.array([s["budget_frac"] for s in segments], dtype=float)
        fracs /= fracs.sum()

        all_frames = []
        all_expo = []
        for i, seg in enumerate(segments):
            seg_first = first + int(round(i * seg_len))
            seg_last = first + int(round((i + 1) * seg_len))
            frames = pulse_positions(seg["pi"], first=seg_first, last=seg_last)
            expo = shape_exposures(
                seg["coeffs"], len(frames), budget=total_budget * fracs[i]
            )
            all_frames.append(frames)
            all_expo.append(expo)
        return np.concatenate(all_frames), np.concatenate(all_expo)

    def superpose_patterns(
        layers, total_budget=BUDGET_MS, first=FIRST_FRAME_STIM, last=LAST_FRAME_STIM
    ):
        """Overlay multiple full-window patterns, summing exposures at shared frames.

        Each layer is a dict with keys: coeffs (3,), pi (int).
        Every layer gets budget = total_budget / len(layers).
        Pulses that land on the same frame have their exposures added.
        Returns sorted (frames, expo) arrays.
        """
        per_layer_budget = total_budget / len(layers)
        combined = {}
        for layer in layers:
            fr = pulse_positions(layer["pi"], first=first, last=last)
            ex = shape_exposures(layer["coeffs"], len(fr), budget=per_layer_budget)
            for f, e in zip(fr, ex):
                combined[f] = combined.get(f, 0.0) + e
        frames = np.array(sorted(combined.keys()))
        expo = np.array([combined[f] for f in frames])
        return frames, expo

    def patterns_to_csv(pattern_list, path):
        """Export patterns to CSV in long format: columns=[uid, time, value].

        Time runs 0..90 (integer minutes). Value is exposure (ms) at that
        minute, 0 outside [FIRST_FRAME_STIM, LAST_FRAME_STIM] and at
        unstimulated frames.
        """
        all_times = np.arange(0, 91)
        rows_uid = []
        rows_time = []
        rows_value = []
        for i, p in enumerate(pattern_list):
            lookup = dict(zip(p["frames"].astype(int), p["expo"]))
            for t in all_times:
                rows_uid.append(i)
                rows_time.append(int(t))
                rows_value.append(float(lookup.get(t, 0.0)))
        df = pl.DataFrame({"uid": rows_uid, "time": rows_time, "value": rows_value})
        df.write_csv(path)
        return path

    return (
        BUDGET_MS,
        FIRST_FRAME_STIM,
        LAST_FRAME_STIM,
        compose_pattern,
        mo,
        np,
        patterns_to_csv,
        plt,
        pulse_positions,
        qmc,
        shape_exposures,
        superpose_patterns,
    )


@app.cell
def _(mo):
    c1 = mo.ui.slider(-4, 4, 0.1, label="C1", value=0)
    c2 = mo.ui.slider(-4, 4, 0.1, label="C2", value=0)
    c3 = mo.ui.slider(-4, 4, 0.1, label="C3", value=0)
    pi = mo.ui.slider(1, 20, 1, label="pi", value=5)
    mo.vstack([c1, c2, c3, pi])
    return c1, c2, c3, pi


@app.cell
def _(
    FIRST_FRAME_STIM,
    LAST_FRAME_STIM,
    c1,
    c2,
    c3,
    np,
    pi,
    plt,
    pulse_positions,
    shape_exposures,
):
    coeffs = np.array([c1.value, c2.value, c3.value])
    pi_val = int(round(pi.value))

    frames = pulse_positions(pi_val)
    expo = shape_exposures(coeffs, len(frames))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="orange", alpha=0.06)
    if len(frames):
        ax.vlines(frames, 0, expo, color="orange", lw=3)
        ax.scatter(frames, expo, color="orange", s=26, zorder=5)
    ax.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
    ax.set_ylim(0, 1400)
    ax.set_xlabel("frame  (minutes)")
    ax.set_ylabel("exposure per pulse (ms)")
    _peak = expo.max() if len(expo) else 0.0
    ax.set_title(
        f"pi = {pi_val} min   →   {len(frames)} pulses,  "
        f"Σ = {expo.sum():.0f} ms,  peak = {_peak:.0f} ms"
    )
    ax.grid(alpha=0.3)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Inspect entire dataset

    For ranges given below, render all the curves.
    """
    )
    return


@app.cell
def _(np, pulse_positions, qmc, shape_exposures):
    N_PATTERNS = 88

    # Per-dial sampling ranges (lo, hi). pi is rounded to whole minutes.
    C1_RANGE = (-3.0, 3.0)  # tilt / ramp
    C2_RANGE = (-3.0, 3.0)  # curvature
    C3_RANGE = (-3.0, 3.0)  # cubic / asymmetry
    PI_RANGE = (1, 8)  # pulse interval (min)

    # Latin Hypercube fills the 4-D box evenly on every axis AND in the
    # interior -> best shape diversity for an arbitrary N. Seeded = reproducible.
    _bounds = np.array([C1_RANGE, C2_RANGE, C3_RANGE, PI_RANGE], dtype=float)
    _sampler = qmc.LatinHypercube(d=4, seed=0, optimization="random-cd")
    _unit = _sampler.random(n=N_PATTERNS)
    _scaled = qmc.scale(_unit, _bounds[:, 0], _bounds[:, 1])

    patterns = []
    for _c1, _c2, _c3, _pif in _scaled:
        _pi = int(round(_pif))
        _coeffs = np.array([_c1, _c2, _c3])
        _frames = pulse_positions(_pi)
        _expo = shape_exposures(_coeffs, len(_frames))
        patterns.append(
            {
                "c1": _c1,
                "c2": _c2,
                "c3": _c3,
                "pi": _pi,
                "frames": _frames,
                "expo": _expo,
            }
        )
    return (patterns,)


@app.cell
def _(FIRST_FRAME_STIM, LAST_FRAME_STIM, np, patterns, plt):
    _ncol = 11
    _nrow = int(np.ceil(len(patterns) / _ncol))
    _fig, _axes = plt.subplots(
        _nrow, _ncol, figsize=(_ncol * 1.5, _nrow * 1.1), sharex=True, sharey=True
    )
    for _i, _ax in enumerate(_axes.flat):
        if _i >= len(patterns):
            _ax.axis("off")
            continue
        _p = patterns[_i]
        _ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="orange", alpha=0.05)
        if len(_p["frames"]):
            _ax.vlines(_p["frames"], 0, _p["expo"], color="orange", lw=1.3)
        _ax.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
        _ax.set_ylim(0, 1400)
        _ax.set_title(
            f"[{_p['c1']:+.1f},{_p['c2']:+.1f},{_p['c3']:+.1f}] pi{_p['pi']}",
            fontsize=5.5,
            pad=1,
        )
        _ax.tick_params(labelsize=4)
    _fig.suptitle(
        f"{len(patterns)} stimulation patterns (Latin Hypercube)", fontsize=10
    )
    _fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    export_lhs_btn = mo.ui.run_button(label="Export LHS patterns to CSV")
    export_lhs_btn
    return (export_lhs_btn,)


@app.cell
def _(export_lhs_btn, mo, patterns, patterns_to_csv):
    mo.stop(not export_lhs_btn.value)
    _path = patterns_to_csv(patterns, "lhs_patterns.csv")
    mo.md(f"Exported {len(patterns)} patterns to `{_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## New idea: composition
    In the prev, there are multiple iterations of the same pattern, slightly shifted / scaled / diluted. This might not be a bad thing, but it makes the sample diversity smaller.

    So, what if for some fraction of our dataset, instead of generating a single stim with a budget of 4s, we generate 2 with budgets of 2, or some other combination?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive composition explorer

    Two segments split the stimulation window in half. Each gets its own shape coefficients and pulse interval; the budget slider controls how the 4 s total is divided between them.
    """
    )
    return


@app.cell
def _(mo):
    seg1_c1 = mo.ui.slider(-3, 3, 0.1, label="Seg 1 C1", value=0)
    seg1_c2 = mo.ui.slider(-3, 3, 0.1, label="Seg 1 C2", value=0)
    seg1_c3 = mo.ui.slider(-3, 3, 0.1, label="Seg 1 C3", value=0)
    seg1_pi = mo.ui.slider(1, 8, 1, label="Seg 1 pi", value=3)
    seg2_c1 = mo.ui.slider(-3, 3, 0.1, label="Seg 2 C1", value=0)
    seg2_c2 = mo.ui.slider(-3, 3, 0.1, label="Seg 2 C2", value=0)
    seg2_c3 = mo.ui.slider(-3, 3, 0.1, label="Seg 2 C3", value=0)
    seg2_pi = mo.ui.slider(1, 8, 1, label="Seg 2 pi", value=5)
    budget_split = mo.ui.slider(
        0.1, 0.9, 0.05, label="Seg 1 budget fraction", value=0.5
    )
    mo.hstack(
        [
            mo.vstack([seg1_c1, seg1_c2, seg1_c3, seg1_pi]),
            mo.vstack([seg2_c1, seg2_c2, seg2_c3, seg2_pi]),
            mo.vstack([budget_split]),
        ]
    )
    return (
        budget_split,
        seg1_c1,
        seg1_c2,
        seg1_c3,
        seg1_pi,
        seg2_c1,
        seg2_c2,
        seg2_c3,
        seg2_pi,
    )


@app.cell
def _(
    BUDGET_MS,
    FIRST_FRAME_STIM,
    LAST_FRAME_STIM,
    budget_split,
    compose_pattern,
    np,
    plt,
    seg1_c1,
    seg1_c2,
    seg1_c3,
    seg1_pi,
    seg2_c1,
    seg2_c2,
    seg2_c3,
    seg2_pi,
):
    _segments = [
        {
            "coeffs": np.array([seg1_c1.value, seg1_c2.value, seg1_c3.value]),
            "pi": int(round(seg1_pi.value)),
            "budget_frac": budget_split.value,
        },
        {
            "coeffs": np.array([seg2_c1.value, seg2_c2.value, seg2_c3.value]),
            "pi": int(round(seg2_pi.value)),
            "budget_frac": 1 - budget_split.value,
        },
    ]
    comp_frames, comp_expo = compose_pattern(_segments, total_budget=BUDGET_MS)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _mid = (FIRST_FRAME_STIM + LAST_FRAME_STIM) / 2
    _ax.axvspan(FIRST_FRAME_STIM, _mid, color="steelblue", alpha=0.06)
    _ax.axvspan(_mid, LAST_FRAME_STIM, color="coral", alpha=0.06)
    _ax.axvline(_mid, color="grey", ls="--", lw=0.8, alpha=0.5)
    if len(comp_frames):
        _colors = ["steelblue" if f < _mid else "coral" for f in comp_frames]
        _ax.vlines(comp_frames, 0, comp_expo, colors=_colors, lw=3)
        _ax.scatter(comp_frames, comp_expo, c=_colors, s=26, zorder=5)
    _ax.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
    _ax.set_ylim(0, 1400)
    _ax.set_xlabel("frame (minutes)")
    _ax.set_ylabel("exposure per pulse (ms)")
    _peak = comp_expo.max() if len(comp_expo) else 0.0
    _ax.set_title(
        f"Composed: {len(comp_frames)} pulses, "
        f"budget split {budget_split.value:.0%} / {1-budget_split.value:.0%}, "
        f"Σ = {comp_expo.sum():.0f} ms, peak = {_peak:.0f} ms"
    )
    _ax.grid(alpha=0.3)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Mixed dataset: single + composed patterns

    A fraction of the dataset uses composed (2-segment) patterns; the rest are single-polynomial patterns as before. Both are sampled via Latin Hypercube for even coverage.
    """
    )
    return


@app.cell
def _(
    BUDGET_MS,
    compose_pattern,
    np,
    pulse_positions,
    qmc,
    shape_exposures,
    superpose_patterns,
):
    N_SINGLE = 44
    N_COMPOSED = 22
    N_SUPERPOSED = 22

    _bounds_single = np.array([[-3, 3], [-3, 3], [-3, 3], [1, 8]], dtype=float)
    _sampler_s = qmc.LatinHypercube(d=4, seed=42, optimization="random-cd")
    _unit_s = _sampler_s.random(n=N_SINGLE)
    _scaled_s = qmc.scale(_unit_s, _bounds_single[:, 0], _bounds_single[:, 1])

    mixed_patterns = []
    for _c1, _c2, _c3, _pif in _scaled_s:
        _pi = int(round(_pif))
        _coeffs = np.array([_c1, _c2, _c3])
        _frames = pulse_positions(_pi)
        _expo = shape_exposures(_coeffs, len(_frames))
        mixed_patterns.append(
            {
                "kind": "single",
                "frames": _frames,
                "expo": _expo,
                "label": f"[{_c1:+.1f},{_c2:+.1f},{_c3:+.1f}] pi{_pi}",
            }
        )

    _bounds_comp = np.array(
        [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [1, 8],  # seg 1
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [1, 8],  # seg 2
            [0.2, 0.8],  # budget fraction for seg 1
        ],
        dtype=float,
    )
    _sampler_c = qmc.LatinHypercube(d=9, seed=7, optimization="random-cd")
    _unit_c = _sampler_c.random(n=N_COMPOSED)
    _scaled_c = qmc.scale(_unit_c, _bounds_comp[:, 0], _bounds_comp[:, 1])

    for _row in _scaled_c:
        _segs = [
            {
                "coeffs": np.array(_row[0:3]),
                "pi": int(round(_row[3])),
                "budget_frac": _row[8],
            },
            {
                "coeffs": np.array(_row[4:7]),
                "pi": int(round(_row[7])),
                "budget_frac": 1 - _row[8],
            },
        ]
        _frames, _expo = compose_pattern(_segs, total_budget=BUDGET_MS)
        mixed_patterns.append(
            {
                "kind": "composed",
                "frames": _frames,
                "expo": _expo,
                "label": f"comp {_row[8]:.0%}",
            }
        )

    _bounds_sup = np.array(
        [
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [1, 8],  # layer A
            [-3, 3],
            [-3, 3],
            [-3, 3],
            [1, 8],  # layer B
        ],
        dtype=float,
    )
    _sampler_sup = qmc.LatinHypercube(d=8, seed=13, optimization="random-cd")
    _unit_sup = _sampler_sup.random(n=N_SUPERPOSED)
    _scaled_sup = qmc.scale(_unit_sup, _bounds_sup[:, 0], _bounds_sup[:, 1])

    for _row in _scaled_sup:
        _layers = [
            {"coeffs": np.array(_row[0:3]), "pi": int(round(_row[3]))},
            {"coeffs": np.array(_row[4:7]), "pi": int(round(_row[7]))},
        ]
        _frames, _expo = superpose_patterns(_layers, total_budget=BUDGET_MS)
        mixed_patterns.append(
            {
                "kind": "superposed",
                "frames": _frames,
                "expo": _expo,
                "label": f"sup pi{_layers[0]['pi']}+{_layers[1]['pi']}",
            }
        )
    return (mixed_patterns,)


@app.cell
def _(FIRST_FRAME_STIM, LAST_FRAME_STIM, mixed_patterns, np, plt):
    _ncol = 11
    _nrow = int(np.ceil(len(mixed_patterns) / _ncol))
    _fig, _axes = plt.subplots(
        _nrow, _ncol, figsize=(_ncol * 1.5, _nrow * 1.1), sharex=True, sharey=True
    )
    _mid = (FIRST_FRAME_STIM + LAST_FRAME_STIM) / 2
    for _i, _ax in enumerate(_axes.flat):
        if _i >= len(mixed_patterns):
            _ax.axis("off")
            continue
        _p = mixed_patterns[_i]
        if _p["kind"] == "single":
            _ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="orange", alpha=0.05)
            _c = "orange"
        elif _p["kind"] == "superposed":
            _ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="green", alpha=0.05)
            _c = "green"
        else:
            _ax.axvspan(FIRST_FRAME_STIM, _mid, color="steelblue", alpha=0.05)
            _ax.axvspan(_mid, LAST_FRAME_STIM, color="coral", alpha=0.05)
            _c = None
        if len(_p["frames"]):
            if _c:
                _ax.vlines(_p["frames"], 0, _p["expo"], color=_c, lw=1.3)
            else:
                _colors = ["steelblue" if f < _mid else "coral" for f in _p["frames"]]
                _ax.vlines(_p["frames"], 0, _p["expo"], colors=_colors, lw=1.3)
        _ax.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
        _ax.set_ylim(0, 1400)
        _ax.set_title(_p["label"], fontsize=5.5, pad=1)
        _ax.tick_params(labelsize=4)
    _n_s = sum(1 for p in mixed_patterns if p["kind"] == "single")
    _n_c = sum(1 for p in mixed_patterns if p["kind"] == "composed")
    _n_sup = sum(1 for p in mixed_patterns if p["kind"] == "superposed")
    _fig.suptitle(
        f"{len(mixed_patterns)} patterns "
        f"({_n_s} single + {_n_c} composed + {_n_sup} superposed)",
        fontsize=10,
    )
    _fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    export_mixed_btn = mo.ui.run_button(label="Export uniform mixed patterns to CSV")
    export_mixed_btn
    return (export_mixed_btn,)


@app.cell
def _(export_mixed_btn, mixed_patterns, mo, patterns_to_csv):
    mo.stop(not export_mixed_btn.value)
    _path = patterns_to_csv(mixed_patterns, "mixed_uniform_patterns.csv")
    mo.md(f"Exported {len(mixed_patterns)} patterns to `{_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Normal-sampled mixed dataset

    Same structure (single + composed + superposed) but C coefficients drawn from **N(0, 1.5)** clipped to [-3, 3]. This concentrates most patterns near flat/mild shapes while still allowing extreme ones in the tails. Pulse interval stays uniform in [1, 8].
    """
    )
    return


@app.cell
def _(
    BUDGET_MS,
    compose_pattern,
    np,
    pulse_positions,
    shape_exposures,
    superpose_patterns,
):
    _C_SIGMA = 1.5
    _C_CLIP = 3.0
    _PI_LO, _PI_HI = 1, 8
    _N_SINGLE_N = 44
    _N_COMPOSED_N = 22
    _N_SUPERPOSED_N = 22

    _rng = np.random.default_rng(seed=99)

    def _sample_coeffs(rng, n, n_c=3):
        c = rng.normal(0, _C_SIGMA, size=(n, n_c))
        return np.clip(c, -_C_CLIP, _C_CLIP)

    def _sample_pi(rng, n):
        return rng.integers(_PI_LO, _PI_HI + 1, size=n)

    normal_patterns = []

    _cs = _sample_coeffs(_rng, _N_SINGLE_N)
    _pis = _sample_pi(_rng, _N_SINGLE_N)
    for _coeffs, _pi in zip(_cs, _pis):
        _frames = pulse_positions(int(_pi))
        _expo = shape_exposures(_coeffs, len(_frames))
        normal_patterns.append(
            {
                "kind": "single",
                "frames": _frames,
                "expo": _expo,
                "label": f"[{_coeffs[0]:+.1f},{_coeffs[1]:+.1f},{_coeffs[2]:+.1f}] pi{_pi}",
            }
        )

    _cs1 = _sample_coeffs(_rng, _N_COMPOSED_N)
    _cs2 = _sample_coeffs(_rng, _N_COMPOSED_N)
    _pis1 = _sample_pi(_rng, _N_COMPOSED_N)
    _pis2 = _sample_pi(_rng, _N_COMPOSED_N)
    _bfracs = _rng.uniform(0.2, 0.8, size=_N_COMPOSED_N)
    for _c1, _c2, _p1, _p2, _bf in zip(_cs1, _cs2, _pis1, _pis2, _bfracs):
        _segs = [
            {"coeffs": _c1, "pi": int(_p1), "budget_frac": _bf},
            {"coeffs": _c2, "pi": int(_p2), "budget_frac": 1 - _bf},
        ]
        _frames, _expo = compose_pattern(_segs, total_budget=BUDGET_MS)
        normal_patterns.append(
            {
                "kind": "composed",
                "frames": _frames,
                "expo": _expo,
                "label": f"comp {_bf:.0%}",
            }
        )

    _cs_a = _sample_coeffs(_rng, _N_SUPERPOSED_N)
    _cs_b = _sample_coeffs(_rng, _N_SUPERPOSED_N)
    _pis_a = _sample_pi(_rng, _N_SUPERPOSED_N)
    _pis_b = _sample_pi(_rng, _N_SUPERPOSED_N)
    for _ca, _cb, _pa, _pb in zip(_cs_a, _cs_b, _pis_a, _pis_b):
        _layers = [
            {"coeffs": _ca, "pi": int(_pa)},
            {"coeffs": _cb, "pi": int(_pb)},
        ]
        _frames, _expo = superpose_patterns(_layers, total_budget=BUDGET_MS)
        normal_patterns.append(
            {
                "kind": "superposed",
                "frames": _frames,
                "expo": _expo,
                "label": f"sup pi{_pa}+{_pb}",
            }
        )
    return (normal_patterns,)


@app.cell
def _(FIRST_FRAME_STIM, LAST_FRAME_STIM, normal_patterns, np, plt):
    _ncol = 11
    _nrow = int(np.ceil(len(normal_patterns) / _ncol))
    _fig, _axes = plt.subplots(
        _nrow, _ncol, figsize=(_ncol * 1.5, _nrow * 1.1), sharex=True, sharey=True
    )
    _mid = (FIRST_FRAME_STIM + LAST_FRAME_STIM) / 2
    for _i, _ax in enumerate(_axes.flat):
        if _i >= len(normal_patterns):
            _ax.axis("off")
            continue
        _p = normal_patterns[_i]
        if _p["kind"] == "single":
            _ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="orange", alpha=0.05)
            _c = "orange"
        elif _p["kind"] == "superposed":
            _ax.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="green", alpha=0.05)
            _c = "green"
        else:
            _ax.axvspan(FIRST_FRAME_STIM, _mid, color="steelblue", alpha=0.05)
            _ax.axvspan(_mid, LAST_FRAME_STIM, color="coral", alpha=0.05)
            _c = None
        if len(_p["frames"]):
            if _c:
                _ax.vlines(_p["frames"], 0, _p["expo"], color=_c, lw=1.3)
            else:
                _colors = ["steelblue" if f < _mid else "coral" for f in _p["frames"]]
                _ax.vlines(_p["frames"], 0, _p["expo"], colors=_colors, lw=1.3)
        _ax.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
        _ax.set_ylim(0, 1400)
        _ax.set_title(_p["label"], fontsize=5.5, pad=1)
        _ax.tick_params(labelsize=4)
    _n_s = sum(1 for p in normal_patterns if p["kind"] == "single")
    _n_c = sum(1 for p in normal_patterns if p["kind"] == "composed")
    _n_sup = sum(1 for p in normal_patterns if p["kind"] == "superposed")
    _fig.suptitle(
        f"{len(normal_patterns)} patterns, normal-sampled "
        f"({_n_s} single + {_n_c} composed + {_n_sup} superposed)",
        fontsize=10,
    )
    _fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    export_normal_btn = mo.ui.run_button(
        label="Export normal-sampled patterns to CSV (Warning: floats; not sure if i should have quantized it somehow)"
    )
    export_normal_btn
    return (export_normal_btn,)


@app.cell
def _(export_normal_btn, mo, normal_patterns, patterns_to_csv):
    mo.stop(not export_normal_btn.value)
    _path = patterns_to_csv(normal_patterns, "mixed_normal_patterns.csv")
    mo.md(f"Exported {len(normal_patterns)} patterns to `{_path}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Superposition composition

    Instead of splitting the time window, generate 2 independent patterns that each span the **full** window with half the budget (2 s each). Exposures at the same frame are summed. This can produce shapes a single polynomial cannot: e.g. two different frequencies overlaid.
    """
    )
    return


@app.cell
def _(mo):
    sup_a_c1 = mo.ui.slider(-3, 3, 0.1, label="Layer A C1", value=1.0)
    sup_a_c2 = mo.ui.slider(-3, 3, 0.1, label="Layer A C2", value=0)
    sup_a_c3 = mo.ui.slider(-3, 3, 0.1, label="Layer A C3", value=0)
    sup_a_pi = mo.ui.slider(1, 8, 1, label="Layer A pi", value=2)
    sup_b_c1 = mo.ui.slider(-3, 3, 0.1, label="Layer B C1", value=-1.0)
    sup_b_c2 = mo.ui.slider(-3, 3, 0.1, label="Layer B C2", value=0)
    sup_b_c3 = mo.ui.slider(-3, 3, 0.1, label="Layer B C3", value=0)
    sup_b_pi = mo.ui.slider(1, 8, 1, label="Layer B pi", value=5)
    mo.hstack(
        [
            mo.vstack([sup_a_c1, sup_a_c2, sup_a_c3, sup_a_pi]),
            mo.vstack([sup_b_c1, sup_b_c2, sup_b_c3, sup_b_pi]),
        ]
    )
    return (
        sup_a_c1,
        sup_a_c2,
        sup_a_c3,
        sup_a_pi,
        sup_b_c1,
        sup_b_c2,
        sup_b_c3,
        sup_b_pi,
    )


@app.cell
def _(
    BUDGET_MS,
    FIRST_FRAME_STIM,
    LAST_FRAME_STIM,
    np,
    plt,
    pulse_positions,
    shape_exposures,
    sup_a_c1,
    sup_a_c2,
    sup_a_c3,
    sup_a_pi,
    sup_b_c1,
    sup_b_c2,
    sup_b_c3,
    sup_b_pi,
    superpose_patterns,
):
    _layers = [
        {
            "coeffs": np.array([sup_a_c1.value, sup_a_c2.value, sup_a_c3.value]),
            "pi": int(round(sup_a_pi.value)),
        },
        {
            "coeffs": np.array([sup_b_c1.value, sup_b_c2.value, sup_b_c3.value]),
            "pi": int(round(sup_b_pi.value)),
        },
    ]
    sup_frames, sup_expo = superpose_patterns(_layers, total_budget=BUDGET_MS)

    _half = BUDGET_MS / 2
    _fr_a = pulse_positions(int(round(sup_a_pi.value)))
    _ex_a = shape_exposures(
        np.array([sup_a_c1.value, sup_a_c2.value, sup_a_c3.value]),
        len(_fr_a),
        budget=_half,
    )
    _fr_b = pulse_positions(int(round(sup_b_pi.value)))
    _ex_b = shape_exposures(
        np.array([sup_b_c1.value, sup_b_c2.value, sup_b_c3.value]),
        len(_fr_b),
        budget=_half,
    )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    _ax1.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="steelblue", alpha=0.04)
    if len(_fr_a):
        _ax1.vlines(_fr_a, 0, _ex_a, color="steelblue", lw=2, alpha=0.6, label="A")
    _ax1.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="coral", alpha=0.04)
    if len(_fr_b):
        _ax1.vlines(_fr_b, 0, _ex_b, color="coral", lw=2, alpha=0.6, label="B")
    _ax1.legend(fontsize=8)
    _ax1.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
    _ax1.set_ylim(0, 1400)
    _ax1.set_xlabel("frame (minutes)")
    _ax1.set_ylabel("exposure per pulse (ms)")
    _ax1.set_title("Individual layers (2 s each)")
    _ax1.grid(alpha=0.3)

    _ax2.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color="green", alpha=0.04)
    if len(sup_frames):
        _ax2.vlines(sup_frames, 0, sup_expo, color="green", lw=3)
        _ax2.scatter(sup_frames, sup_expo, color="green", s=26, zorder=5)
    _ax2.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
    _ax2.set_xlabel("frame (minutes)")
    _peak = sup_expo.max() if len(sup_expo) else 0.0
    _ax2.set_title(
        f"Superposed: {len(sup_frames)} pulses, "
        f"Σ = {sup_expo.sum():.0f} ms, peak = {_peak:.0f} ms"
    )
    _ax2.grid(alpha=0.3)
    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
