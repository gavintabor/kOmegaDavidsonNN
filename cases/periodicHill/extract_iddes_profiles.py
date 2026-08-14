"""
One-time provenance script (task #13): extracts spanwise+time-averaged
U/UPrime2Mean profiles at Davidson's x/H stations from the SA-IDDES
tutorial leg (`transient/`), writing them into
`transient/postProcessing/sample/<time>/xbyh{X}_UMean_UPrime2Mean.xy` in
the same format `postprocess_hill.py`'s `load_station()` already expects
from every other case -- so the main script needs no changes to pick this
up.

Why this exists instead of just using the case's own in-solve `sample`
function object: that object's field list (`columnAverage:columnAverage
(UMean)`, `...(UPrime2Mean)`) never resolved at run time -- some
execute/write-ordering mismatch between `columnAverage` (executeControl
writeTime) and `sample` (executeControl onEnd) meant `sample` ran before
`columnAverage`'s registered fields existed for it to find. Confirmed
byte-identical to the stock OpenFOAM v2606 tutorial's own controlDict, so
this is a tutorial-side quirk, not something this port introduced.

*** Data-loss note (2026-08-14): a first attempt at working around this
via `postProcess -dict` (running fieldAverage1+columnAverage+sample
together in one standalone invocation) turned out to force-write
fieldAverage1's output despite `writeControl none` in that dict --
silently OVERWRITING the correctly-accumulated UPrime2Mean at the run's
true final time (1509.9999999974898) with a near-zero degenerate
single-instant recomputation (visible as ~1e-19 magnitude values where
~1e-4 was expected). That file is not recoverable (fieldAverage doesn't
persist the intermediate accumulator needed to reconstruct it). This
script therefore deliberately uses the *previous* purgeWrite-kept time,
1509.9499999975023 (0.05s / ~1 part in 190 less averaging than the true
final time -- statistically negligible against the ~9.5s/37.7-through-flow
window), which was never touched by that mistake (verified: file mtimes
predate it, values are real). U/UMean/p/pMean at the true final time are
still fine -- only that one UPrime2Mean file was affected, and only at
that one time. Left as-is (not worth another risky write attempt to
"fix" cosmetically); this comment is the record of it, matching this
project's practice elsewhere (see README) of documenting rather than
hiding mistakes found along the way.

Second wrinkle: this mesh's streamwise "columns" are NOT at constant x
for all y (cell-centre x varies smoothly with y along what's nominally
one i-index, presumably from grading/smoothing near the wavy hill
surface propagating some curvature into the interior) -- so a fixed-x
"column" can't be looked up directly; it needs actual scattered-data
interpolation onto a true vertical line, matching what DNS/RANS-leg
comparisons assume. Also, per-(x,y) spanwise averaging is done directly
in Python here (grouping the ~40-80 z-copies at each identical (x,y) by
rounding), not via OpenFOAM's `columnAverage` -- this script only ever
*reads* existing fields (U, UMean, UPrime2Mean, Cx, Cy, all already
correct on disk) and writes its own separate output files, so there's no
repeat risk of the mistake above.

Run once, from this directory, after the target time's fields have been
converted to ascii and cell-centres written -- see README "periodic hill
sub-cases" for the exact commands used:

    python3 extract_iddes_profiles.py
"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

CASE_DIR = Path("transient")
TIME = "1509.9499999975023"  # see data-loss note above -- NOT the run's
                              # nominal final time, deliberately
N_PROC = 16
STATIONS = [0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
H = 0.028       # m, hill height -- matches postprocess_hill.py's default
WINDOW = 0.006  # m, +/- x window around each station used for the local
                # triangulation (~5x the ~0.00126m mean streamwise cell
                # spacing -- wide enough for good interpolation coverage,
                # narrow enough that curvature within the window is mild)
N_Y = 200       # points in the output y grid, evenly spaced over the
                # full domain height; points outside the local data's
                # convex hull (e.g. below the wavy wall at that exact x)
                # come back NaN and are dropped below


def _read_scalar(fp):
    text = Path(fp).read_text()
    lower = text.lower()
    idx = lower.find("nonuniform")
    start = text.find("(", idx)
    end = text.find(")", start)
    return np.array([float(v) for v in text[start + 1:end].split()])


def _read_tuple_field(fp):
    """volVectorField or volSymmTensorField internalField -> (N, width) array."""
    text = Path(fp).read_text()
    lower = text.lower()
    if "nonuniform" not in lower:
        return np.zeros((0, 0))
    start = text.find("(", lower.find("nonuniform"))
    depth, i = 0, start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    tuples = re.findall(r"\(([^)]+)\)", text[start + 1:i])
    return np.array([[float(v) for v in t.split()] for t in tuples])


def _spanwise_average(x, y, UMean, UPrime2Mean):
    """Collapse the z direction: group cells sharing the same (x, y)
    footprint (identical up to float rounding -- confirmed by direct
    inspection that z-copies at fixed (x,y) match to >9 decimal places)
    and average their fields. Returns reduced (x, y, UMean, UPrime2Mean)
    arrays, one row per unique footprint."""
    xy_round = np.stack([np.round(x, 8), np.round(y, 8)], axis=1)
    _, inverse, counts = np.unique(
        xy_round, axis=0, return_inverse=True, return_counts=True)
    n_groups = counts.size
    x_avg = np.zeros(n_groups)
    y_avg = np.zeros(n_groups)
    um_avg = np.zeros((n_groups, UMean.shape[1]))
    up_avg = np.zeros((n_groups, UPrime2Mean.shape[1]))
    np.add.at(x_avg, inverse, x)
    np.add.at(y_avg, inverse, y)
    np.add.at(um_avg, inverse, UMean)
    np.add.at(up_avg, inverse, UPrime2Mean)
    x_avg /= counts
    y_avg /= counts
    um_avg /= counts[:, None]
    up_avg /= counts[:, None]
    return x_avg, y_avg, um_avg, up_avg


def main():
    xs, ys, ums, ups = [], [], [], []
    for p in range(N_PROC):
        td = CASE_DIR / f"processor{p}" / TIME
        if not td.exists():
            print(f"  WARNING: missing {td}, skipping")
            continue
        x = _read_scalar(td / "Cx")
        y = _read_scalar(td / "Cy")
        um = _read_tuple_field(td / "UMean")
        up = _read_tuple_field(td / "UPrime2Mean")
        n = min(len(x), len(y), len(um), len(up))
        if n == 0:
            continue
        xs.append(x[:n]); ys.append(y[:n]); ums.append(um[:n]); ups.append(up[:n])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    UMean = np.concatenate(ums, axis=0)
    UPrime2Mean = np.concatenate(ups, axis=0)
    print(f"Loaded {len(x)} cells across {N_PROC} processors "
          f"(x in [{x.min():.4f}, {x.max():.4f}], "
          f"y in [{y.min():.4f}, {y.max():.4f}])")

    x, y, UMean, UPrime2Mean = _spanwise_average(x, y, UMean, UPrime2Mean)
    print(f"After spanwise averaging: {len(x)} unique (x,y) footprints")
    # sanity check: UPrime2Mean should be a real turbulent-fluctuation
    # magnitude (~1e-5 to 1e-3 for this flow), not near-zero/garbage
    print(f"  UPrime2Mean_xy range: [{UPrime2Mean[:,1].min():.3e}, "
          f"{UPrime2Mean[:,1].max():.3e}]  (sanity check -- should NOT be ~1e-18)")

    y_grid = np.linspace(y.min(), y.max(), N_Y)
    out_dir = CASE_DIR / "postProcessing" / "sample" / TIME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full-domain spanwise-averaged scatter (all 32,000 (x,y) footprints --
    # topologically the same 200x160 grid as the 2D RANS legs, since they
    # share this mesh's x-y footprint by construction -- see README's "Why
    # 1-cell-thick" note). Used by postprocess_hill.py's field_contours.png
    # to add the SA-IDDES leg's velocity field alongside kOmega/
    # kOmegaDavidsonNN (no k/omega available from this leg -- SA-IDDES
    # doesn't solve either).
    field_fp = out_dir / "field_UMean.xy"
    with open(field_fp, "w") as f:
        f.write("# Spanwise-averaged UMean over the full (x,y) domain "
                "(one row per unique footprint)\n")
        f.write("# x y UMean_x UMean_y UMean_z\n")
        for i in range(len(x)):
            f.write(f"{x[i]:.8e} {y[i]:.8e} "
                    f"{UMean[i,0]:.8e} {UMean[i,1]:.8e} {UMean[i,2]:.8e}\n")
    print(f"  wrote {field_fp} ({len(x)} rows)")

    for station in STATIONS:
        x_target = station * H
        mask = np.abs(x - x_target) < WINDOW
        n_pts = mask.sum()
        if n_pts < 50:
            print(f"  WARNING: only {n_pts} points near x/H={station}, skipping")
            continue

        pts = np.column_stack([x[mask], y[mask]])
        tri = Delaunay(pts)
        xi = np.column_stack([np.full(N_Y, x_target), y_grid])

        cols = []
        for comp in range(3):
            interp = LinearNDInterpolator(tri, UMean[mask, comp])
            cols.append(interp(xi))
        for comp in range(6):
            interp = LinearNDInterpolator(tri, UPrime2Mean[mask, comp])
            cols.append(interp(xi))
        cols = np.column_stack(cols)  # (N_Y, 9)

        valid = ~np.any(np.isnan(cols), axis=1)
        y_out = y_grid[valid]
        cols = cols[valid]

        tag = f"{station:g}"
        out_fp = out_dir / f"xbyh{tag}_UMean_UPrime2Mean.xy"
        with open(out_fp, "w") as f:
            f.write(f"# x/H={station}  x_target={x_target:.6f} m  "
                    f"({n_pts} local (already spanwise-averaged) points "
                    f"triangulated, {len(y_out)}/{N_Y} y-grid points inside hull)\n")
            f.write("# y  UMean_x UMean_y UMean_z  "
                    "UPrime2Mean_xx UPrime2Mean_xy UPrime2Mean_xz "
                    "UPrime2Mean_yy UPrime2Mean_yz UPrime2Mean_zz\n")
            for i in range(len(y_out)):
                row = [y_out[i], *cols[i]]
                f.write(" ".join(f"{v:.8e}" for v in row) + "\n")
        print(f"  wrote {out_fp} ({len(y_out)} rows)")


if __name__ == "__main__":
    main()
