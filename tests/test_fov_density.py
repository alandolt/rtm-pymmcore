"""Tests for the hardware-agnostic density scoring in faro.agents.fov_density."""

import numpy as np
import pytest

from faro.agents.fov_density import (
    FovDensityScorer,
    build_stage_montage,
    cells_in_window,
    find_fov_windows,
    label_centroids_um,
    nearest_neighbor_distances,
)


def test_nearest_neighbor_distances_basic():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    nn = nearest_neighbor_distances(xy)
    # point 0 -> point1 (3), point1 -> point0 (3), point2 -> point1 (4)
    assert np.allclose(nn, [3.0, 3.0, 4.0])


def test_nearest_neighbor_single_point_is_inf():
    assert np.isinf(nearest_neighbor_distances(np.array([[1.0, 2.0]]))[0])


def test_cells_in_window():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [10.0, 10.0]])
    mask = cells_in_window(xy, center=(0, 0), fov_w_um=40, fov_h_um=40)
    assert mask.tolist() == [True, False, True]


def test_label_centroids_um_centering_and_units():
    # 100x100 label image, one object at pixel (row=75, col=25).
    lab = np.zeros((100, 100), dtype=int)
    lab[74:77, 24:27] = 1  # centroid ~ (75, 25)
    pts = label_centroids_um(lab, pixel_size_um=2.0, origin_xy=(1000.0, 2000.0))
    # col 25 -> x: (25 - 50) * 2 = -50 + origin_x; row 75 -> y: (75-50)*2 = 50 + origin_y
    assert pts.shape == (1, 2)
    assert pytest.approx(pts[0, 0], abs=1.0) == 1000.0 - 50.0
    assert pytest.approx(pts[0, 1], abs=1.0) == 2000.0 + 50.0


def test_label_centroids_empty():
    assert label_centroids_um(np.zeros((10, 10), dtype=int)).shape == (0, 2)


@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
def test_montage_and_centroids_agree(flip_x, flip_y):
    # A bright dot at a known pixel must end up at the SAME stage coordinate in
    # both the montage and label_centroids_um — i.e. the two transforms are
    # consistent for any flip setting. This is exactly the check the validation
    # notebook does: where the agent thinks a cell is must be where it's drawn.
    px = 0.65
    H = W = 100
    tile_center = (1000.0, -2000.0)
    img = np.zeros((H, W), dtype=float)
    img[70, 20] = 1000.0  # one bright pixel at (row=70, col=20)
    lab = np.zeros((H, W), dtype=int)
    lab[70, 20] = 1

    # where the finder places this cell in stage coordinates:
    cell_xy = label_centroids_um(
        lab, pixel_size_um=px, origin_xy=tile_center, flip_x=flip_x, flip_y=flip_y
    )[0]

    montage, (x_min, x_max, y_min, y_max) = build_stage_montage(
        np.array([tile_center]), [img], px, flip_x=flip_x, flip_y=flip_y
    )
    # locate the bright pixel in the montage and convert its index to stage µm
    r, c = np.unravel_index(int(np.argmax(montage)), montage.shape)
    mont_x = x_min + (c + 0.5) * px
    mont_y = y_min + (r + 0.5) * px
    assert abs(mont_x - cell_xy[0]) <= px
    assert abs(mont_y - cell_xy[1]) <= px


def test_scorer_rejects_clumped_field():
    # 30 cells packed within a few µm of each other -> heavily clumped.
    rng = np.random.default_rng(1)
    xy = rng.normal(loc=[0, 0], scale=3.0, size=(30, 2))
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=10,
        clump_distance_um=20.0,
        max_clumped_fraction=0.3,
    )
    m = scorer.metrics(xy)
    ok, reason = scorer.passes(m)
    assert not ok and reason == "too_clumped"


def test_scorer_accepts_spread_field():
    # Evenly spaced grid of cells, ~40 µm apart -> not clumped.
    gx, gy = np.meshgrid(np.arange(6) * 40.0, np.arange(6) * 40.0)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=10,
        clump_distance_um=20.0,
        max_clumped_fraction=0.3,
    )
    ok, reason = scorer.passes(scorer.metrics(xy))
    assert ok, reason


def test_scorer_count_bounds():
    xy = np.zeros((5, 2)) + np.arange(5)[:, None] * 50.0
    scorer = FovDensityScorer(fov_w_um=700, fov_h_um=700, min_cells=10)
    ok, reason = scorer.passes(scorer.metrics(xy))
    assert not ok and reason == "below_min_cells"


def test_find_fov_windows_recenters_onto_cluster():
    # A tight-ish but spread cluster sitting far from the origin. A window
    # seeded anywhere on it should mean-shift onto the cluster centroid.
    rng = np.random.default_rng(2)
    cluster = rng.normal(loc=[5000, -3000], scale=100, size=(40, 2))
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=15,
        clump_distance_um=10.0,
        max_clumped_fraction=0.5,
    )
    _, df_sel = find_fov_windows(cluster, scorer, n_select=1, recenter=True)
    assert len(df_sel) == 1
    # selected centre lands near the true cluster centroid
    assert abs(df_sel.iloc[0]["x"] - 5000) < 200
    assert abs(df_sel.iloc[0]["y"] - (-3000)) < 200


def test_find_fov_windows_non_overlapping():
    # Two separated clusters; selected windows must be far apart.
    rng = np.random.default_rng(3)
    a = rng.normal(loc=[0, 0], scale=120, size=(40, 2))
    b = rng.normal(loc=[3000, 0], scale=120, size=(40, 2))
    xy = np.vstack([a, b])
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=15,
        clump_distance_um=10.0,
        max_clumped_fraction=0.5,
    )
    _, df_sel = find_fov_windows(
        xy, scorer, n_select=2, recenter=True, min_separation_um=700
    )
    assert len(df_sel) == 2
    pts = df_sel[["x", "y"]].to_numpy()
    assert np.linalg.norm(pts[0] - pts[1]) >= 700


def test_find_fov_windows_empty_cloud():
    scorer = FovDensityScorer(fov_w_um=700, fov_h_um=700)
    df_all, df_sel = find_fov_windows(np.empty((0, 2)), scorer, n_select=3)
    assert len(df_all) == 0 and len(df_sel) == 0


def _no_rectangles_overlap(df_sel, fov_w, fov_h):
    """True if no two selected FOV rectangles overlap (exact axis-aligned test)."""
    pts = df_sel[["x", "y"]].to_numpy()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = abs(pts[i, 0] - pts[j, 0])
            dy = abs(pts[i, 1] - pts[j, 1])
            if dx < fov_w and dy < fov_h:
                return False
    return True


def test_selected_fovs_never_overlap():
    # Dense blob of cells: naive selection would stack windows on the blob.
    rng = np.random.default_rng(7)
    xy = rng.normal(loc=[1000, 1000], scale=300, size=(300, 2))
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=10,
        clump_distance_um=8.0,
        max_clumped_fraction=0.6,
    )
    _, df_sel = find_fov_windows(xy, scorer, n_select=3, fill_invalid=True)
    assert _no_rectangles_overlap(df_sel, 700, 700)


def test_clustered_well_fills_without_duplicates():
    # Reproduce the C2 failure: one good cluster + sparse cells scattered far
    # away. With strict fill we must still get 3 *distinct, non-overlapping*
    # FOVs rather than 3 copies of the cluster centre.
    rng = np.random.default_rng(11)
    cluster = rng.normal(loc=[0, 0], scale=120, size=(60, 2))
    scattered = rng.uniform(-4000, 4000, size=(40, 2))
    xy = np.vstack([cluster, scattered])
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=40,  # only the cluster passes
        clump_distance_um=10.0,
        max_clumped_fraction=0.5,
    )
    _, df_sel = find_fov_windows(xy, scorer, n_select=3, fill_invalid=True)
    assert len(df_sel) == 3
    assert _no_rectangles_overlap(df_sel, 700, 700)
    # the three centres must be genuinely distinct (no near-duplicates)
    pts = df_sel[["x", "y"]].to_numpy()
    dists = [np.linalg.norm(pts[i] - pts[j]) for i in range(3) for j in range(i + 1, 3)]
    assert min(dists) >= 700  # at least one FOV side apart


def test_fill_prefers_dense_nearby_over_empty_far():
    # One passing cluster, a decent sub-threshold patch ~1 FOV away, and a few
    # lonely cells far out. The fill must take the dense nearby patch (best of
    # the bad), NOT the far empty corner that a spread objective would grab.
    rng = np.random.default_rng(19)
    cluster = rng.normal(loc=[0, 0], scale=120, size=(60, 2))  # valid
    nearby = rng.normal(loc=[800, 0], scale=120, size=(25, 2))  # dense, sub-threshold
    far = rng.uniform([5000, -300], [5300, 300], size=(3, 2))  # near-empty, far
    xy = np.vstack([cluster, nearby, far])
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=40,
        clump_distance_um=10.0,
        max_clumped_fraction=0.6,
    )
    _, df_sel = find_fov_windows(xy, scorer, n_select=2, fill_invalid=True)
    assert len(df_sel) == 2
    assert _no_rectangles_overlap(df_sel, 700, 700)
    fill = df_sel.iloc[1]  # second pick is the fill
    assert not bool(fill["valid"])
    # the fill landed on the dense nearby patch, not the far corner
    assert fill["n_cells"] >= 15
    assert abs(fill["x"] - 800) < 400 and abs(fill["x"]) < 4000


def test_fill_invalid_false_may_return_fewer():
    # Only one cluster passes; without fill we get exactly that one FOV.
    rng = np.random.default_rng(13)
    cluster = rng.normal(loc=[0, 0], scale=120, size=(60, 2))
    scattered = rng.uniform(-4000, 4000, size=(5, 2))
    xy = np.vstack([cluster, scattered])
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=40,
        clump_distance_um=10.0,
        max_clumped_fraction=0.5,
    )
    _, df_sel = find_fov_windows(xy, scorer, n_select=3, fill_invalid=False)
    assert len(df_sel) == 1 and bool(df_sel.iloc[0]["valid"])


def test_extra_min_separation_enforced():
    # Three well-separated clusters; demand >= 1500 µm extra spacing.
    rng = np.random.default_rng(17)
    xy = np.vstack(
        [
            rng.normal(loc=c, scale=80, size=(40, 2))
            for c in ([0, 0], [2000, 0], [0, 2000])
        ]
    )
    scorer = FovDensityScorer(
        fov_w_um=700,
        fov_h_um=700,
        min_cells=15,
        clump_distance_um=8.0,
        max_clumped_fraction=0.6,
    )
    _, df_sel = find_fov_windows(xy, scorer, n_select=3, min_separation_um=1500)
    pts = df_sel[["x", "y"]].to_numpy()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.linalg.norm(pts[i] - pts[j]) >= 1500


# ---------------------------------------------------------------------------
# ERK-activity (per-cell feature) gating
# ---------------------------------------------------------------------------


def test_label_centroids_um_return_labels():
    lab = np.zeros((50, 50), dtype=int)
    lab[10:13, 10:13] = 2
    lab[30:33, 30:33] = 5
    xy, labels = label_centroids_um(lab, pixel_size_um=1.0, return_labels=True)
    assert xy.shape == (2, 2)
    assert labels.tolist() == [2, 5]  # label-sorted order, aligned to rows


def test_find_fov_windows_feature_condition_gate():
    """A window is rejected when its cells fail the FOVCondition (e.g. ERK)."""
    from faro.agents.fov_finder import FOVCondition

    rng = np.random.default_rng(0)
    ready = rng.normal([0.0, 0.0], 30.0, size=(40, 2))  # low CNR
    active = rng.normal([5000.0, 5000.0], 30.0, size=(40, 2))  # high CNR
    xy = np.vstack([ready, active])
    feat = {"cnr": np.concatenate([np.full(40, 0.5), np.full(40, 2.0)])}

    cond = FOVCondition("cnr", "below", 1.0, min_fraction=0.7)
    scorer = FovDensityScorer(
        fov_w_um=400,
        fov_h_um=400,
        min_cells=10,
        max_cells=200,
        max_clumped_fraction=1.0,
        fov_conditions=[cond],
    )
    df_all, df_sel = find_fov_windows(
        xy, scorer, n_select=2, feat=feat, fill_invalid=False
    )

    near_active = df_all[(abs(df_all.x - 5000) < 200) & (abs(df_all.y - 5000) < 200)]
    near_ready = df_all[(df_all.x.abs() < 200) & (df_all.y.abs() < 200)]
    assert near_ready["valid"].any()
    assert not near_active["valid"].any()
    assert (near_active["reason"] == "condition_failed").all()
    # the satisfied-fraction column is recorded for inspection
    assert "cond_cnr_below_frac" in df_all.columns
    # only the ready cluster is selected (valid-only, no fill)
    assert (df_sel["x"].abs() < 300).all()


def test_find_fov_windows_no_conditions_unchanged():
    """Without conditions the feat path is inert (backward compatible)."""
    xy = np.random.default_rng(1).normal([0.0, 0.0], 100.0, size=(50, 2))
    scorer = FovDensityScorer(
        fov_w_um=300, fov_h_um=300, min_cells=5, max_clumped_fraction=1.0
    )
    df_all, df_sel = find_fov_windows(xy, scorer, n_select=2)
    # No feature-gating columns are produced when fov_conditions is empty.
    assert "conditions_pass" not in df_all.columns
    assert not any(c.startswith("cond_") for c in df_all.columns)
    assert df_all["valid"].any() and len(df_sel) >= 1
