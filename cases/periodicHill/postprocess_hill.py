"""
postprocess_hill.py
====================
Post-processing script for the OpenFOAM periodic hill case.
Produces figures equivalent to Davidson (2026) Figs 14-16:
  Fig 14  - streamwise velocity U profiles at 6 x-stations
  Fig 15  - Reynolds shear stress u'v' profiles at 6 x-stations
  Fig 16  - turbulent kinetic energy k profiles at 6 x-stations
plus a fourth series (this port's addition to Davidson's own two-model
comparison): the pristine OpenFOAM tutorial's SA-IDDES result.

Case parameters (OpenFOAM tutorial / Davidson 2026):
  H   = 0.028 m  (hill height, reference length)
  Ub  = 1.0  m/s (bulk velocity at hill crest)
  Re  = 10565 (nu = 2.643e-6 m^2/s; Davidson quotes Re=10,600)
  Domain: 9H x 3.035H (x,y); z is 1-cell-thick (empty) for the RANS cases,
          4.5H periodic for the tutorial's 3D SA-IDDES mesh.
  z_mid = 0.063 m (mid-span of the tutorial's original 4.5H-wide mesh;
          the 1-cell-thick RANS meshes were extruded to the same z-width
          purely so this single z coordinate samples all cases identically)

Hill shape: 6-piece piecewise cubic (OpenFOAM blockMeshDict / Almeida)
  x argument runs 0-54 in internal blockMesh units.
  Normalised so hill peak y = H.

Usage
-----
    python postprocess_hill.py [OPTIONS]

    --cases      Space-separated list of case directories
                 default: transient_kOmega_2D transient_kOmegaDavidsonNN_2D
                          transient
    --dns        Path to DNS data directory (default: ./DNS)
    --H          Hill height in metres (default: 0.028)
    --Ub         Bulk velocity (default: 1.0)
    --z          z coordinate for sample lines (default: 0.063)
    --out        Output directory for figures (default: ./figures)
    --sample-subdir  Subdirectory for sample data
                     (default: postProcessing/sample)
    --print-sampledict   Print the sets dict used by these cases and exit

DNS data (Froehlich, Mellen, Rodi et al. 2005, JFM 526:19-66 -- the
DNS Davidson (2026) Sec 5.3 / ref [15] actually compares against)
----------------------------------------------------------------
Files: DNS_1xh.dat ... DNS_8xh.dat  (x/H = 1-8)
       DNS_x005h.dat                (x/H = 0.05)
       DNS_x05h.dat                 (x/H = 0.5, not currently plotted --
                                      0.5 isn't one of Davidson's 6 stations)
       DNS_cf.dat                   (skin friction, not used by this script)
The x<1h files' naming strips the decimal point, so digit COUNT
distinguishes them: "0.05" -> "005" (3 digits), "0.5" -> "05" (2 digits).
Confirmed by u'v' shape against Davidson Fig 15(a) -- see _DNS_FILENAMES's
comment for the full story of how this (and an earlier, opposite-direction
version of the same digit-count confusion, baked into this script's very
first commit as "DNS_x005h.dat is x/H=0.005, NOT 0.05") got sorted out.

Column semantics: of the 10 columns, 5 are independently verified -- 0
(y/H, confirmed by range matching the known 3.035H domain height), 1
(U/Ub, confirmed by shape match to Davidson's own Fig 14), 7 (u'v'/Ub^2,
confirmed by checking its sign flips exactly where dU/dy does, as
physically required -- column 3, the original unverified guess, never
changes sign and is ~10x too large, so is NOT u'v'), and 4/5/6 (v'v'-
and w'w'-like: clean single-hump profiles vanishing at both wall and
freestream, and 0.5*(4+5+6) is k/Ub^2 -- see _DNS_COLS's comment for the
full derivation, including why column 8, k's original unverified guess,
is a near-exact constant multiple of the real thing rather than the real
thing itself). The remaining columns (2, 3, 8, 9) are unused by this
script and their semantics are unconfirmed guesses at best; don't trust a
comment claiming otherwise without re-deriving it the way 0/1/4/5/6/7
were.
y/H in column 0 is measured from the domain bottom (absolute), NOT from
the local hill surface -- do not add y_wall when plotting.

Sample file format
-------------------
Each case's `sample` function object (system/controlDict, inline) writes
ONE combined raw file per station per set of *found* fields, named
    xbyh<station>_<field1>_<field2>....xy
where the field tokens are whichever of the requested fields were actually
present in the registry at sample time (order is registry-dependent, not
necessarily the order listed in the dict -- confirmed empirically, not
assumed). Columns are [coordinate, field1's components..., field2's
components..., ...] in that same token order, 1 column per scalar, 3 per
vector, 6 per symmTensor (OpenFOAM order xx,xy,xz,yy,yz,zz). This script
parses the filename to recover the column layout rather than assuming a
fixed one, since the token order isn't guaranteed.

Reynolds shear stress / TKE: modelled + resolved
--------------------------------------------------
The two RANS legs (transient_kOmega_2D, transient_kOmegaDavidsonNN_2D) are
run unsteady (URANS, matching Davidson's own method for the hill flow --
"unsteady simulations are carried out marching to steady-state
conditions"). Their *total* turbulent shear stress / TKE is the modelled
closure contribution (RMean / kMean) plus whatever large-scale unsteadiness
the run resolves (UPrime2Mean) -- using only one or the other would
silently drop part of the real total. This script adds them together for
those two cases.

The tutorial's `transient` (SA-IDDES) leg is left as a pristine,
unmodified copy of the stock tutorial (see README "Not in the git repo" /
periodicHill sub-cases), so it has no registered modelled-stress field --
its shear stress / TKE panels are resolved-only (UPrime2Mean based), which
is the conventional way LES/hybrid results are reported and is a small
effect away from the wall, but is a real asymmetry against the two RANS
legs' modelled+resolved total worth keeping in mind.

Note on the RMean actually used for the first run of the two RANS legs:
`fieldAverage`'s field-validity check runs at construction time, before
`turbulenceFields`'s first execute() has registered its output -- and that
output is registered as "turbulenceProperties:R", not plain "R" besides.
Both controlDicts are now fixed (see transient_kOmega_2D/system/
controlDict's turbulenceFields1 comment) so a future rerun computes a true
running average, but the first run's RMean/turbulenceProperties:RMean was
never actually written. Its `postProcessing/sample/<t>/xbyh<x>_RMean.xy`
files were instead reconstructed post-hoc, by rerunning `turbulenceFields`
on the 5 instantaneous fields `purgeWrite` happened to leave on disk
(t=10.898-11.098s, within the averaging window) and arithmetic-averaging
those 5 snapshots externally -- an approximation of the true running
average, not the real thing. load_station() below merges that
supplementary file's RMean in with the main sample file's fields
transparently (a station can be covered by more than one .xy file).
"""

import argparse
import re
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# X stations -- Davidson (2026) Fig 14-16's own 6 stations. A superset of 10
# (0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, matching the OpenFOAM tutorial's own
# validation stations) is sampled by every case's `sample` function object;
# only Davidson's 6 are plotted by default so these figures line up with
# his directly.
# ---------------------------------------------------------------------------

X_STATIONS = [0.05, 1.0, 3.0, 4.0, 5.0, 7.0]

# ---------------------------------------------------------------------------
# DNS file mapping (Froehlich et al. 2005 -- Davidson (2026) ref [15])
#
# The x<0.05h/0.5h files' naming strips the decimal point, so digit COUNT
# distinguishes them: "0.05" -> "005" (3 digits) -> DNS_x005h.dat, "0.5" ->
# "05" (2 digits) -> DNS_x05h.dat. The original (pre-2026-08-11, never
# actually plotted/checked) mapping had these backwards -- confirmed by
# shape: DNS_x005h.dat's u'v' profile is a smooth single trough matching
# Davidson (2026) Fig 15(a)'s x=0.05 panel; DNS_x05h.dat's has a sharp
# near-wall spike (y/H=1.01, u'v'=-0.026) plus a separate, shallower second
# hump (y/H~1.75, u'v'~-0.010) -- a different, double-featured shape,
# consistent with it actually being the x/H=0.5 station (not currently
# plotted -- 0.5 isn't one of Davidson's six stations).
# ---------------------------------------------------------------------------

_DNS_FILENAMES = {
    0.05: "DNS_x005h.dat",
    0.5: "DNS_x05h.dat",
    1.0: "DNS_1xh.dat",
    2.0: "DNS_2xh.dat",
    3.0: "DNS_3xh.dat",
    4.0: "DNS_4xh.dat",
    5.0: "DNS_5xh.dat",
    6.0: "DNS_6xh.dat",
    7.0: "DNS_7xh.dat",
    8.0: "DNS_8xh.dat",
}

# Column indices (0-based) in DNS files. The original (pre-2026-08-11,
# never actually run/checked) guess had "uv": 3 -- wrong: column 3 is
# always positive, magnitude ~0.13-0.22 (too large, wrong sign convention
# for a shear stress). Column 7 changes sign exactly where dU/dy (inferred
# from column 1) changes sign near the domain top -- the physically
# required behaviour for u'v' -- and its magnitude (~0.01-0.04) matches
# Davidson (2026) Fig 15's axis ranges. Column 0 vs 9 also disproved the
# original "9: y/H repeat" claim (they're not close), so nothing about
# this file's column layout inherited unverified assumptions -- only
# columns 0 (y/H) and 1 (U/Ub) are independently confirmed (range/domain
# match for the former, near-exact match to Davidson's own Fig 14 shape
# for the latter); 8 (k) is a plausible-but-unverified holdover.
_DNS_COLS = {
    "U":  1,   # U/Ub
    "uv": 7,   # u'v'/Ub^2  (negative in shear layer)
    # k/Ub^2 = 0.5*(columns 4,5,6), NOT column 8. Column 8 has the right
    # shape (vanishes at wall and freestream, peaks mid-shear-layer) but is
    # a near-exact constant multiple (3.5797, std 0.02% across every x/H
    # station checked) of 0.5*(col4+col5+col6) -- too precise/consistent to
    # be noise, and far too clean to be "a different but related quantity";
    # almost certainly the same k under a different (unidentified) velocity
    # normalisation. 0.5*(col4+col5+col6) is the one that lands in
    # Davidson (2026) Fig 16's actual axis range (peaks ~0.05-0.08 there,
    # matching his ~0.10-0.15 axis cap); col8 alone would be 2-3x off his
    # chart. col4/5/6 individually presumed v'v'/w'w'-like components
    # (clean single-hump profiles vanishing at both wall and freestream);
    # column 3, the original "k" search's other candidate, does neither
    # (peaks *at* the wall, stays large in the freestream) so is excluded.
    "k": (4, 5, 6),
}

# ---------------------------------------------------------------------------
# Hill geometry
# 6-piece piecewise cubic from OpenFOAM blockMeshDict (Almeida hill).
# Internal x coordinate runs 0-54 over 9H physical length.
# Coefficients: a + b*x + c*x^2 + d*x^3, x in blockMesh units.
# y/H = y_bm / 28  (segment 1 peak a-coeff = 28 = H in BM units).
# ---------------------------------------------------------------------------

_HILL_COEFF = [
    (( 0,  9), [ 28.0,               0.0,                 6.775070969851e-3, -2.124527775800e-3]),
    (( 9, 14), [ 25.07355893131,      0.9754803562315,    -1.016116352781e-1,  1.889794677828e-3]),
    ((14, 20), [  2.579601052357e1,   8.206693007457e-1,  -9.055370274339e-2,  1.626510569859e-3]),
    ((20, 30), [  4.046435022819e1,  -1.379581654948,      1.945884504128e-2, -2.070318932190e-4]),
    ((30, 40), [  1.792461334664e1,   8.743920332081e-1,  -5.567361123058e-2,  6.277731764683e-4]),
    ((40, 54), [  5.639011190988e1,  -2.010520359035,      1.644919857549e-2,  2.674976141766e-5]),
]


def hill_y_lower(x_phys, H=0.028):
    """
    Return y coordinate of lower hill wall at physical x [m].
    Returns y [m], clamped to [0, H].

    KNOWN BUG (found 2026-08-11, not fixed): this decays ~monotonically
    over the full 9H domain instead of the real hill's shape -- crest at
    x/H=0, down to the flat channel floor (y=0) by x/H~2, flat until
    x/H~7, back up to the crest by x/H~9. Confirmed against actual mesh
    cell-centre data: true wall height at x/H=1.65 is ~0.01H; this
    function says ~0.95H there. Likely a transcription error in
    _HILL_COEFF against the real blockMeshDict codeStream, or a missing
    mirroring step -- not re-derived yet. Was never caught earlier because
    every existing caller only evaluates it at Davidson's 6-10 discrete
    x/H stations (0.05, 0.5, 1, 2, ..., 8), and none of the profile plots
    actually depend on its output -- OpenFOAM's own `sample` sets clip to
    real mesh faces regardless of what this function returns; only
    plot_nn_coefficients (currently dormant, no data) and
    print_sample_dict (a documentation helper) call it. The contour plots'
    _hill_masked_triangulation() deliberately avoids this function
    entirely, deriving the wall height empirically from the real mesh
    instead -- see its docstring.
    """
    x_bm = (x_phys % (9.0 * H)) * (54.0 / (9.0 * H))
    y_bm = 0.0
    for (x0, x1), (a, b, c, d) in _HILL_COEFF:
        if x0 <= x_bm <= x1:
            t = x_bm
            y_bm = a + b*t + c*t**2 + d*t**3
            if x0 == 0:
                y_bm = min(28.0, y_bm)
            if x1 == 54:
                y_bm = max(0.0, y_bm)
            break
    return float(np.clip(y_bm / 28.0 * H, 0.0, H))


# ---------------------------------------------------------------------------
# Plotting styles
# ---------------------------------------------------------------------------

CASE_STYLE = {
    "transient_kOmegaDavidsonNN_2D": dict(
        color="#1f77b4", lw=1.8, ls="-", label=r"$k$-$\omega$-PINN-NN"),
    "transient_kOmega_2D": dict(
        color="#ff7f0e", lw=1.8, ls="--", label=r"standard $k$-$\omega$"),
    "transient": dict(
        color="#2ca02c", lw=1.6, ls=":", label="OpenFOAM tutorial (SA-IDDES)"),
}
DNS_STYLE = dict(color="black", lw=1.4, ls="-.",
                 label="DNS (Froehlich et al. 2005)")

NN_COEFF_COLORS = {
    "sigmakNN":  "#1f77b4",
    "CkNN":      "#d62728",
    "Comega2NN": "#2ca02c",
}
NN_COEFF_LABELS = {
    "sigmakNN":  r"$\sigma_{k,NN}$",
    "CkNN":      r"$C_{k,NN}$",
    "Comega2NN": r"$C_{\omega 2,NN}$",
}
NN_STANDARD = {
    "sigmakNN":  2.0,
    "CkNN":      1.0,
    "Comega2NN": 3.0 / 40.0,  # 0.075
}

# ---------------------------------------------------------------------------
# Field-type registry, for parsing the combined raw sample files
# ---------------------------------------------------------------------------

_VECTOR_FIELDS = {"U", "UMean"}
_SYMMTENSOR_FIELDS = {"R", "RMean", "UPrime2Mean"}
# Anything not listed above is treated as scalar (k, kMean, p, pMean, omega,
# nut, sigmakNN, CkNN, Comega2NN, ...).

_WRAPPED_RE = re.compile(r"^.*\(([^()]+)\)$")  # e.g. columnAverage:columnAverage(UMean)


def _base_name(token):
    """Strip a functionObject wrapper like 'columnAverage:columnAverage(UMean)'
    down to the underlying field name 'UMean', or a registry-namespace
    prefix like 'turbulenceProperties:RMean' (what turbulenceFields/
    fieldAverage actually register its output as -- not plain 'R'/'RMean',
    a real bug this port hit once, see transient_kOmega_2D/system/
    controlDict's turbulenceFields1 comment) down to 'RMean'."""
    m = _WRAPPED_RE.match(token)
    if m:
        return m.group(1)
    if ":" in token:
        return token.rsplit(":", 1)[-1]
    return token


def _field_width(token):
    base = _base_name(token)
    if base in _VECTOR_FIELDS:
        return 3
    if base in _SYMMTENSOR_FIELDS:
        return 6
    return 1


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def find_latest_time(case_dir, sample_subdir):
    base = Path(case_dir) / sample_subdir
    if not base.exists():
        return None
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and _is_float(d.name)],
        key=lambda d: float(d.name))
    return dirs[-1] if dirs else None


def read_sample_file(filepath):
    """Read whitespace-delimited file, skip # comment lines."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data.append([float(v) for v in line.split()])
            except ValueError:
                continue
    return np.array(data) if data else None


def _station_tag(x_station):
    """Station tag as used in the `sample` set names, e.g. xbyh0.05, xbyh1."""
    return f"{x_station:g}"


def load_station(time_dir, x_station):
    """
    Load and parse the raw sample file(s) for one x/H station. Normally a
    case's single `sample` function object writes ONE combined file per
    station; but a station can also be covered by more than one file (e.g.
    a supplementary post-hoc file merged in alongside the normal one -- see
    the RMean note in the module docstring), so every matching file is
    parsed and merged. Returns dict {base_field_name: (N,width) array} plus
    'y': (N,) array, or None if no matching file is found. Field blocks are
    recovered from each filename itself (see module docstring), not
    assumed from any fixed/requested order.
    """
    time_dir = Path(time_dir)
    tag = _station_tag(x_station)
    matches = sorted(time_dir.glob(f"xbyh{tag}_*.xy"))
    if not matches:
        return None

    out = {}
    for fp in matches:
        d = read_sample_file(fp)
        if d is None:
            continue
        out.setdefault("y", d[:, 0])
        tokens = fp.stem[len(f"xbyh{tag}_"):].split("_")
        col = 1
        for tok in tokens:
            width = _field_width(tok)
            if col + width > d.shape[1]:
                break  # malformed/truncated -- stop rather than misread columns
            out[_base_name(tok)] = d[:, col:col + width]
            col += width
    return out if "y" in out else None


def load_dns(dns_dir, field, x_station):
    """
    Load DNS data (Froehlich et al. 2005) for a given field and x/H station.
    Returns two-column array [y/H, field_value], or None if not available.
    y/H is absolute from domain bottom -- do NOT add y_wall when plotting.
    _DNS_COLS[field] is either a single column index, or (for "k") a tuple
    of columns to combine as 0.5*sum(...) -- see its comment.
    """
    dns_dir = Path(dns_dir)
    if not dns_dir.exists():
        return None
    fname = _DNS_FILENAMES.get(float(x_station))
    if fname is None:
        return None   # No DNS file for this station (e.g. x/H=0.05)
    fp = dns_dir / fname
    if not fp.exists():
        print(f"  Note: DNS file not found: {fp}")
        return None
    col = _DNS_COLS.get(field)
    if col is None:
        return None
    cols = (col,) if isinstance(col, int) else col
    d = read_sample_file(fp)
    if d is None or d.shape[1] <= max(cols):
        print(f"  Note: DNS file {fname} has only {d.shape[1] if d is not None else 0} columns")
        return None
    value = 0.5 * sum(d[:, c] for c in cols) if len(cols) > 1 else d[:, cols[0]]
    return np.column_stack([d[:, 0], value])


# ---------------------------------------------------------------------------
# Derived quantities: modelled + resolved, generic over whatever a case
# actually sampled (mean-only, mean+modelled, or raw instantaneous).
# ---------------------------------------------------------------------------

def mean_Ux(parsed):
    """Streamwise velocity: prefer the time mean, fall back to raw U."""
    if "UMean" in parsed:
        return parsed["UMean"][:, 0]
    if "U" in parsed:
        return parsed["U"][:, 0]
    return None


def total_uv(parsed):
    """
    Total Reynolds shear stress u'v': resolved (UPrime2Mean_xy) +
    modelled (RMean_xy or R_xy), whichever are present. None if neither.
    symmTensor column order is xx,xy,xz,yy,yz,zz -- xy is index 1.
    """
    total = None
    if "UPrime2Mean" in parsed:
        total = parsed["UPrime2Mean"][:, 1].copy()
    for modelled_name in ("RMean", "R"):
        if modelled_name in parsed:
            contrib = parsed[modelled_name][:, 1]
            total = contrib.copy() if total is None else total + contrib
            break
    return total


def total_k(parsed):
    """
    Total turbulent kinetic energy: resolved (0.5*trace(UPrime2Mean)) +
    modelled (kMean or k), whichever are present. None if neither.
    """
    total = None
    if "UPrime2Mean" in parsed:
        t = parsed["UPrime2Mean"]
        total = 0.5 * (t[:, 0] + t[:, 3] + t[:, 5])
    for modelled_name in ("kMean", "k"):
        if modelled_name in parsed:
            contrib = parsed[modelled_name][:, 0]
            total = contrib.copy() if total is None else total + contrib
            break
    return total


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+",
                   default=["transient_kOmega_2D",
                            "transient_kOmegaDavidsonNN_2D",
                            "transient"])
    p.add_argument("--dns",  default="DNS")
    p.add_argument("--H",    type=float, default=0.028,
                   help="Hill height [m] (default: 0.028)")
    p.add_argument("--Ub",   type=float, default=1.0,
                   help="Bulk velocity [m/s] (default: 1.0)")
    p.add_argument("--z",    type=float, default=0.063,
                   help="z coordinate for sample lines (default: 0.063)")
    p.add_argument("--out",  default="figures")
    p.add_argument("--sample-subdir", default="postProcessing/sample",
                   dest="sample_subdir")
    p.add_argument("--print-sampledict", action="store_true",
                   dest="print_sampledict")
    return p.parse_args()


# ---------------------------------------------------------------------------
# sampleDict printer -- documents the convention already baked into every
# case's system/controlDict `sample` function object (see e.g.
# transient_kOmegaDavidsonNN_2D/system/controlDict), for reference when
# setting up a new case in the same style.
# ---------------------------------------------------------------------------

_ALL_STATIONS = [0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]


def print_sample_dict(H, z_mid):
    print(f"""
// Inline `sample` entry under controlDict's functions{{}} (see e.g.
// transient_kOmegaDavidsonNN_2D/system/controlDict for the full version,
// which also averages a modelled Reynolds-stress field via a preceding
// turbulenceFields function object).
// H = {H} m,  z = {z_mid} m

sample
{{
    type            sets;
    libs            (sampling);
    interpolationScheme cellPoint;
    setFormat       raw;
    executeControl  onEnd;
    writeControl    onEnd;

    fields ( U UMean UPrime2Mean );

    sets
    {{""")
    for x in _ALL_STATIONS:
        x_phys = x * H
        tag = f"{x:g}"
        print(f"""        xbyh{tag}
        {{
            type    face;
            axis    y;
            start   ({x_phys:.6f} 0 {z_mid});
            end     ({x_phys:.6f} 1 {z_mid});
        }}""")
    print("    }\n}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _legend(ax):
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=8, loc="best")


def _save(fig, out_dir, name):
    p = Path(out_dir) / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _case_style(case):
    return CASE_STYLE.get(
        case, dict(color="grey", lw=1.5, ls=":", label=case))


# ---------------------------------------------------------------------------
# Fig 14: Streamwise velocity
# ---------------------------------------------------------------------------

def plot_velocity(cases_data, dns_dir, H, Ub, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        dns = load_dns(dns_dir, "U", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        for case, td in cases_data.items():
            parsed = load_station(td, xs)
            if parsed is None:
                print(f"  WARNING: no sample file for {case} at x/H={xs}")
                continue
            u = mean_Ux(parsed)
            if u is None:
                print(f"  WARNING: U not found for {case} at x/H={xs}")
                continue
            ax.plot(u / Ub, parsed["y"] / H, **_case_style(case))

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel(r"$U_x/U_b$", fontsize=9)
        ax.set_ylabel(r"$y/H$", fontsize=9)
        ax.set_ylim([0, 3.05])
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.grid(True, lw=0.4, alpha=0.5)

    _legend(axes.flat[0])
    fig.suptitle(r"Periodic hill — Streamwise velocity $U_x/U_b$,  $Re=10565$",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "hill_velocity.png")


# ---------------------------------------------------------------------------
# Fig 15: Reynolds shear stress
# ---------------------------------------------------------------------------

def plot_shear_stress(cases_data, dns_dir, H, Ub, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        dns = load_dns(dns_dir, "uv", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        for case, td in cases_data.items():
            parsed = load_station(td, xs)
            if parsed is None:
                continue
            uv = total_uv(parsed)
            if uv is None:
                print(f"  WARNING: no shear-stress data for {case} at x/H={xs}")
                continue
            ax.plot(uv / Ub**2, parsed["y"] / H, **_case_style(case))

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel(r"$\overline{u'v'}/U_b^2$", fontsize=9)
        ax.set_ylabel(r"$y/H$", fontsize=9)
        ax.set_ylim([0, 3.05])
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.grid(True, lw=0.4, alpha=0.5)
        # 4-decimal tick labels (e.g. x/H=0.05's narrow range) crowd the
        # narrower panel width of a 3x2 grid at the default tick count.
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    _legend(axes.flat[0])
    fig.suptitle(r"Periodic hill — Reynolds shear stress $\overline{u'v'}/U_b^2$"
                 "\n(RANS legs: modelled + resolved; SA-IDDES: resolved only)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "hill_shear_stress.png")


# ---------------------------------------------------------------------------
# Fig 16: Turbulent kinetic energy
#
# DNS k/Ub^2 = 0.5*(columns 4,5,6), not column 8 (the original, unverified
# guess) -- see _DNS_COLS's comment. Fixed 2026-08-11; previously produced
# DNS k peaks 2-3x above Davidson (2026) Fig 16's own DNS curve while the
# RANS-side curves plotted here already matched his published curves fine,
# which was the tell that only the DNS side was wrong.
# ---------------------------------------------------------------------------

def plot_tke(cases_data, dns_dir, H, Ub, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        dns = load_dns(dns_dir, "k", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        for case, td in cases_data.items():
            parsed = load_station(td, xs)
            if parsed is None:
                continue
            k = total_k(parsed)
            if k is None:
                print(f"  WARNING: no TKE data for {case} at x/H={xs}")
                continue
            ax.plot(k / Ub**2, parsed["y"] / H, **_case_style(case))

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel(r"$k/U_b^2$", fontsize=9)
        ax.set_ylabel(r"$y/H$", fontsize=9)
        ax.set_ylim([0, 3.05])
        ax.set_xlim(left=0)
        ax.grid(True, lw=0.4, alpha=0.5)

    _legend(axes.flat[0])
    fig.suptitle(r"Periodic hill — Turbulent kinetic energy $k/U_b^2$"
                 "\n(RANS legs: modelled + resolved; SA-IDDES: resolved only)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "hill_tke.png")


# ---------------------------------------------------------------------------
# NN coefficient fields (sigma_k,NN / C_k,NN / C_omega2,NN)
# Not sampled by transient_kOmegaDavidsonNN_2D's `sample` function object
# (only U/UMean/UPrime2Mean/RMean/kMean are), so this is a no-op by default
# for the main 4-way comparison -- kept as a hook in case a future run adds
# these three scalars to that case's sample fields. The steady diagnostic
# case (steadyState_kOmegaDavidsonNN_2D) already samples them via its own
# hillsSample/crestProbe function objects, at different stations/format.
# ---------------------------------------------------------------------------

def plot_nn_coefficients(cases_data, H, out_dir):
    nn_case = "transient_kOmegaDavidsonNN_2D"
    if nn_case not in cases_data:
        return
    td = cases_data[nn_case]
    probe = load_station(td, X_STATIONS[0])
    if probe is None or not any(f in probe for f in NN_COEFF_COLORS):
        print(f"  Skipping NN-coefficient plot -- {nn_case}'s sample data "
              f"doesn't include sigmakNN/CkNN/Comega2NN (see docstring).")
        return

    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    for idx, xs in enumerate(X_STATIONS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        y_wall_m = hill_y_lower(xs * H, H)
        parsed = load_station(td, xs)
        any_data = False

        for field, color in NN_COEFF_COLORS.items():
            if parsed is None or field not in parsed:
                continue
            any_data = True
            y_from_wall = (parsed["y"] - y_wall_m) / H
            mask = y_from_wall >= 0
            ax.plot(parsed[field][mask, 0], y_from_wall[mask],
                    color=color, lw=1.6, label=NN_COEFF_LABELS[field])
            ax.axvline(NN_STANDARD[field], color=color,
                       lw=0.8, ls="--", alpha=0.5)

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel("Coefficient value", fontsize=9)
        ax.set_ylabel(r"$(y - y_{wall})/H$", fontsize=9)
        ax.set_ylim([0, 0.15])
        ax.grid(True, lw=0.4, alpha=0.5)
        if idx == 0 and any_data:
            ax.legend(fontsize=8)

    fig.suptitle(
        r"Periodic hill — NN coefficients $\sigma_{k,NN}$, $C_{k,NN}$, "
        r"$C_{\omega2,NN}$ (kOmegaDavidsonNN, near-wall)",
        fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "hill_nn_coefficients.png")


# ---------------------------------------------------------------------------
# Field contour plots: U, k, omega over the whole (x,y) domain.
#
# Same technique as ../pitzDaily's postProcessing/plotPitzDaily.py (raw
# OpenFOAM field parsing + writeCellCentres + masked tricontourf) rather
# than a VTK/PyFoam dependency, for consistency with that script and
# because it's self-contained. Kept separate from that file rather than
# imported -- each case here owns its own plotting script.
# ---------------------------------------------------------------------------

def _find_snapshot_times(case_dir, n=5):
    """The last n instantaneous full-field time directories purgeWrite left
    on disk, as their exact on-disk names -- used to time-average fields
    fieldAverage was never told to average (nut, and kOmegaDavidsonNN's own
    sigmakNN/CkNN/Comega2NN). Discovered fresh per case/run rather than a
    fixed list: adaptive timestepping means exact time values are chaotic
    and won't reproduce bit-for-bit between reruns (confirmed the hard way
    -- a hardcoded list from an earlier run silently matched nothing after
    a rerun, since none of its literal timestamps existed on disk anymore,
    and every field relying on it quietly went missing)."""
    case_dir = Path(case_dir)
    dirs = [d.name for d in case_dir.iterdir()
            if d.is_dir() and _is_float(d.name) and d.name != "0"]
    return sorted(dirs, key=float)[-n:]


def _find_latest_written_time(case_dir):
    """Latest solved-field time directory (as its exact on-disk name,
    unlike find_latest_time() above which targets postProcessing/sample --
    the two don't necessarily coincide: `sample`'s onEnd executes at the
    solver's true final timestep, which usually falls between two regular
    writeInterval checkpoints)."""
    case_dir = Path(case_dir)
    dirs = [d for d in case_dir.iterdir()
            if d.is_dir() and _is_float(d.name) and d.name not in ("0",)]
    return max(dirs, key=lambda d: float(d.name)).name if dirs else None


def _read_of_field(filepath):
    """Parse an OpenFOAM volScalarField's internalField into a flat array."""
    text = Path(filepath).read_text()
    lower = text.lower()
    if "nonuniform" in lower:
        idx = lower.find("nonuniform")
        start = text.find("(", idx)
        end = text.find(")", start)
        return np.array([float(v) for v in text[start + 1:end].split()])
    if "uniform" in lower:
        idx = lower.find("uniform")
        val = float(text[idx:].split()[1].rstrip(";"))
        return np.array([val])
    return np.array([])


def _read_of_vector_field(filepath, component):
    """Parse an OpenFOAM volVectorField's internalField, one component."""
    text = Path(filepath).read_text()
    lower = text.lower()
    if "nonuniform" not in lower:
        return np.array([])
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
    return np.array([float(t.split()[component]) for t in tuples])


def _prepare_ascii_fields(case_dir, times):
    """These cases write binary fields (system/controlDict's writeFormat)
    -- fine for the solver, but our parsers here are text-based (matching
    ../pitzDaily's plotPitzDaily.py, to stay dependency-free). Temporarily
    flip writeFormat to ascii for one window in which we (a) convert the
    given (already-solved) time directories with foamFormatConvert and (b)
    generate cell centres (writeCellCentres, at the latest of `times`) so
    that new C file comes out ascii too -- then restore the original
    controlDict. So the case's own configuration is untouched; only the
    on-disk serialisation of those already-final time directories and the
    new C file change (same data, different encoding). No-op if already
    ascii or if the latest time already has a C file."""
    case_dir = Path(case_dir)
    latest = max(times, key=float)
    if (case_dir / latest / "C").exists():
        return
    cd_path = case_dir / "system" / "controlDict"
    original = cd_path.read_text()
    m = re.search(r"writeFormat\s+(\w+);", original)
    if m is None or m.group(1) != "binary":
        subprocess.run(
            ["postProcess", "-func", "writeCellCentres", "-time", latest],
            cwd=case_dir, capture_output=True, text=True)
        return  # already ascii (or unusual dict layout) -- no flip needed
    try:
        cd_path.write_text(re.sub(
            r"(writeFormat\s+)binary;", r"\1ascii;", original))
        subprocess.run(
            ["foamFormatConvert", "-time", ",".join(times)],
            cwd=case_dir, capture_output=True, text=True)
        subprocess.run(
            ["postProcess", "-func", "writeCellCentres", "-time", latest],
            cwd=case_dir, capture_output=True, text=True)
    finally:
        cd_path.write_text(original)  # always restore, even if a step failed


def _cell_centres(case_dir, time_name):
    """(x, y) cell-centre coordinates at time_name. Assumes
    _prepare_ascii_fields() has already been called for this case."""
    case_dir = Path(case_dir)
    c_file = case_dir / time_name / "C"
    if not c_file.exists():
        print(f"  WARNING: could not generate cell centres in {case_dir}")
        return None, None
    return _read_of_vector_field(c_file, 0), _read_of_vector_field(c_file, 1)


def _time_average_field(case_dir, field, times):
    """Elementwise mean of a scalar field across a fixed set of saved
    snapshots (same static mesh throughout, so no interpolation needed) --
    used for omega (which fieldAverage was never set up to average) and
    for nut/sigmakNN/CkNN/Comega2NN (used by the coefficient-ratio contour
    plots). Skips any time that doesn't have this field at all -- e.g.
    sigmakNN/CkNN/Comega2NN only exist for kOmegaDavidsonNN, not kOmega."""
    case_dir = Path(case_dir)
    arrays = [_read_of_field(case_dir / t / field)
              for t in times if (case_dir / t / field).exists()]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _hill_masked_triangulation(x, y, H):
    """Delaunay-triangulate cell-centre data for tricontourf, masked so it
    doesn't bridge through the solid hill -- analogous to pitzDaily's
    masked_triangulation() for its step corner.

    Deliberately does NOT use hill_y_lower(): checking it here for the
    first time against the real mesh (previously only ever evaluated at
    Davidson's 6-10 discrete stations, where OpenFOAM's own sample-line
    clipping did the real work, not this function) found it's wrong almost
    everywhere -- it decays ~monotonically over the full 9H instead of
    flattening to the channel floor by x/H~2 and rising back to the crest
    by x/H~9 (confirmed against actual cell-centre data: true wall height
    at x/H=1.65 is ~0.01H, hill_y_lower() there says ~0.95H). Pre-existing,
    not introduced by this plot; doesn't affect the profile plots (their
    y-values come straight from OpenFOAM's own sample sets, which clip to
    real mesh faces regardless of this function). Not fixed here -- see
    hill_y_lower()'s own comment and the "hill_y_lower is wrong" README
    note for the underlying issue. This function instead derives the wall
    height empirically from the real mesh: minimum cell-centre y in each
    of 200 x-bins, linearly interpolated -- self-consistent with whatever
    case's mesh is actually being plotted, no dependency on re-deriving
    the analytic hill profile correctly."""
    triang = mtri.Triangulation(x, y)
    xc = x[triang.triangles].mean(axis=1)
    yc = y[triang.triangles].mean(axis=1)

    n_bins = 200
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    wall_by_bin = np.full(n_bins, y.max())
    for b in range(n_bins):
        in_bin = bin_idx == b
        if np.any(in_bin):
            wall_by_bin[b] = y[in_bin].min()
    # A handful of narrow bins miss the bottom-most row of cells outright
    # (fewer cells captured than their neighbours -- the structured mesh's
    # rows aren't perfectly aligned to arbitrary bin edges), giving a
    # falsely-elevated wall estimate right at those x -- visible as thin
    # spurious spikes poking up from the flat channel floor. A small
    # rolling minimum over neighbouring bins fixes it without needing
    # finer/coarser binning to dodge the alignment.
    pad = 2
    padded = np.concatenate([wall_by_bin[:pad], wall_by_bin, wall_by_bin[-pad:]])
    wall_by_bin = np.array([padded[i:i + 2 * pad + 1].min()
                             for i in range(n_bins)])
    y_wall = np.interp(xc, centres, wall_by_bin)
    hill_mask = yc < y_wall - 1e-4

    # Plain Delaunay triangulation doesn't know the mesh's actual topology
    # -- along the curved, boundary-layer-graded wall it connects scattered
    # cell-centre points into long thin "fan" triangles radiating from the
    # crest corner instead of following the true curve (visible as
    # diagonal comb-like streaks cutting across the recirculation zone,
    # not the smooth curved wall the mesh actually has). Real mesh-
    # conforming triangles have edges comparable to local cell spacing
    # (median longest-edge here is ~1.3mm); the fan artifacts are wildly
    # longer (up to the full domain length, 0.25m). 5x the median cleanly
    # removes them -- checked visually against a zoomed render of the
    # crest region, not just picked analytically: 10x still left one
    # visible residual sliver right at the corner (a moderately-long,
    # non-extreme tail between the ~2.8mm 99.5th and ~85mm 99.9th
    # percentiles that a coarser cutoff missed), 5x removed it cleanly
    # with no new gaps appearing elsewhere.
    pts = np.column_stack([x, y])
    tris = triang.triangles
    edge_lengths = np.stack([
        np.linalg.norm(pts[tris[:, 0]] - pts[tris[:, 1]], axis=1),
        np.linalg.norm(pts[tris[:, 1]] - pts[tris[:, 2]], axis=1),
        np.linalg.norm(pts[tris[:, 2]] - pts[tris[:, 0]], axis=1),
    ], axis=1)
    max_edge = edge_lengths.max(axis=1)
    sliver_mask = max_edge > 5 * np.median(max_edge)

    triang.set_mask(hill_mask | sliver_mask)
    return triang


def _load_field_case(case_dir, H):
    """Cell centres + UMean/kMean/omegaMean (true running averages, already
    written as full fields by fieldAverage) + 5-snapshot averages of nut
    and (kOmegaDavidsonNN only) sigmakNN/CkNN/Comega2NN, for one case.
    Returns None if the case hasn't been run."""
    case_dir = Path(case_dir)
    time_name = _find_latest_written_time(case_dir)
    if time_name is None:
        return None
    snapshot_times = _find_snapshot_times(case_dir)
    times = sorted(set(snapshot_times) | {time_name}, key=float)
    _prepare_ascii_fields(case_dir, times)
    x, y = _cell_centres(case_dir, time_name)
    if x is None:
        return None
    td = case_dir / time_name
    Ux = _read_of_vector_field(td / "UMean", 0)
    k = _read_of_field(td / "kMean")
    # omegaMean only exists on a rerun since the omega/turbulenceFields fix
    # (see transient_kOmega_2D/system/controlDict's fieldAverage1 comment);
    # earlier runs predate it, so fall back to the same 5-snapshot post-hoc
    # average nut/the NN coefficients still use below.
    if (td / "omegaMean").exists():
        omega = _read_of_field(td / "omegaMean")
        omega_source = "true running average"
    else:
        omega = _time_average_field(case_dir, "omega", snapshot_times)
        omega_source = "5-snapshot approximation"
    nut = _time_average_field(case_dir, "nut", snapshot_times)
    sigmakNN = _time_average_field(case_dir, "sigmakNN", snapshot_times)
    CkNN = _time_average_field(case_dir, "CkNN", snapshot_times)
    Comega2NN = _time_average_field(case_dir, "Comega2NN", snapshot_times)
    return dict(x=x, y=y, Ux=Ux, k=k, omega=omega, omega_source=omega_source,
               nut=nut, sigmakNN=sigmakNN, CkNN=CkNN, Comega2NN=Comega2NN,
               time=time_name)


def _load_iddes_field_case(case_dir, sample_subdir):
    """Spanwise-averaged Ux over the full domain for the SA-IDDES
    `transient` leg, from the flat scatter file
    `extract_iddes_profiles.py` writes (see that script's docstring for
    why this doesn't go through `_load_field_case`'s normal
    single-time-directory-under-case_dir machinery -- this case is
    decomposed across 16 processors with no reconstructed serial mesh).
    Returns the same dict shape as `_load_field_case` (k/omega left as
    None -- SA-IDDES solves neither) or None if not yet extracted.

    Resamples onto a clean regular grid first (via griddata) rather than
    handing the raw scatter straight to `_hill_masked_triangulation`:
    unlike the RANS legs' mesh (genuinely vertical i-lines), this
    tutorial mesh's cell-centre x varies smoothly with y along a nominal
    column (see extract_iddes_profiles.py's docstring), so near the
    domain edges some points end up nearly collinear in x across a wide
    y range -- Delaunay triangulating that raw scatter directly produces
    long, thin sliver triangles there that render as a fine vertical
    striping/ringing artefact in tricontourf (worst near the outlet,
    x~0.21-0.25). A regular grid has no such degenerate triangles."""
    case_dir = Path(case_dir)
    matches = sorted((case_dir / sample_subdir).glob("*/field_UMean.xy"))
    if not matches:
        return None
    fp = matches[-1]  # latest time, by directory mtime-independent sort
    d = read_sample_file(fp)
    if d is None:
        return None
    x_raw, y_raw, ux_raw = d[:, 0], d[:, 1], d[:, 2]

    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    nx, ny = 200, 160  # matches the RANS legs' own mesh resolution
    xg = np.linspace(x_raw.min(), x_raw.max(), nx)
    yg = np.linspace(y_raw.min(), y_raw.max(), ny)
    Xg, Yg = np.meshgrid(xg, yg)
    Ux_grid = griddata((x_raw, y_raw), ux_raw, (Xg, Yg), method="linear")
    valid0 = ~np.isnan(Ux_grid)

    # griddata's own internal Delaunay triangulation of the raw scatter
    # still inherits the sliver-triangle conditioning problem described
    # above -- it just relocates the resulting interpolation noise onto
    # the (otherwise clean) regular output grid instead of matplotlib's
    # triangulation. A light NaN-aware Gaussian smoothing pass removes
    # that residual noise; justified since this is a turbulence-time-
    # averaged mean field with no reason to have real structure at the
    # single-grid-cell scale the noise appears at. NaN-safe via the
    # standard "smooth both the filled data and the validity mask, then
    # divide" trick, sigma=1 grid cell.
    filled = np.where(valid0, Ux_grid, 0.0)
    weight = gaussian_filter(valid0.astype(float), sigma=1.0)
    smoothed = gaussian_filter(filled, sigma=1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ux_grid = np.where(weight > 0.3, smoothed / weight, np.nan)
    valid = ~np.isnan(Ux_grid)

    return dict(x=Xg[valid], y=Yg[valid], Ux=Ux_grid[valid], k=None,
               omega=None, omega_source=None,
               nut=None, sigmakNN=None, CkNN=None, Comega2NN=None,
               time=fp.parent.name)


def plot_field_contours(cases_data, H, out_dir, sample_subdir="postProcessing/sample"):
    """U, k, omega contour plots over the whole domain, one combined figure,
    3 rows (fields) x N columns (kOmega, kOmegaDavidsonNN, and -- U row
    only -- the SA-IDDES tutorial leg, which solves neither k nor omega so
    has no columns in those rows). Each row shares one color scale/colorbar
    across whichever columns it has data for, so left-right comparison
    reads directly as a physical difference, not a scale difference."""
    ordered_cases = [c for c in cases_data if c in CASE_STYLE]  # kOmega, kOmegaDavidsonNN, transient, in --cases order
    field_cases = {}
    for case in ordered_cases:
        c = (_load_iddes_field_case(case, sample_subdir) if case == "transient"
             else _load_field_case(case, H))
        if c is not None:
            field_cases[case] = c

    if not field_cases:
        print("  Skipping field contour plots -- no case data found.")
        return

    specs = [
        ("Ux", "RdBu_r", r"$U_x$ (m/s)", "Streamwise velocity", False),
        ("k", "hot_r", r"$k$ (m$^2$/s$^2$)", "Turbulent kinetic energy", False),
        ("omega", "viridis", r"$\omega$ (1/s)",
         "Specific dissipation rate", True),
    ]
    rows = [s for s in specs
            if any(d.get(s[0]) is not None for d in field_cases.values())]
    if not rows:
        print("  Skipping field contour plots -- no field data found.")
        return

    fig, axes = plt.subplots(len(rows), len(field_cases),
                             figsize=(6 * len(field_cases), 3.4 * len(rows)),
                             squeeze=False, constrained_layout=True)

    for row, (key, cmap, cbar_label, title, log_scale) in enumerate(rows):
        cases_with_field = {c: d for c, d in field_cases.items()
                            if d.get(key) is not None}
        vals_all = np.concatenate([d[key] for d in cases_with_field.values()])

        if log_scale:
            vmin = max(np.percentile(vals_all, 1), 1.0)
            vmax = vals_all.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
            decade_lo, decade_hi = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
            ticks = 10.0 ** np.arange(decade_lo, decade_hi + 1)
            kwargs = dict(levels=levels, norm=norm, extend="both")
        else:
            vmin, vmax = min(0, vals_all.min()), np.percentile(vals_all, 99)
            ticks = None
            kwargs = dict(levels=100, vmin=vmin, vmax=vmax)

        row_axes = axes[row]
        cf = None
        for col, case in enumerate(field_cases):
            ax = row_axes[col]
            if case not in cases_with_field:
                ax.axis("off")
                continue
            d = cases_with_field[case]
            triang = _hill_masked_triangulation(d["x"], d["y"], H)
            cf = ax.tricontourf(triang, d[key], cmap=cmap, **kwargs)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")
            ax.set_title(f"{title} — {_case_style(case)['label']}", fontsize=10)

        if cf is not None:
            fig.colorbar(cf, ax=row_axes, label=cbar_label, ticks=ticks,
                        shrink=0.9)

    omega_sources = {d["omega_source"] for d in field_cases.values()
                     if d.get("omega") is not None}
    omega_note = (omega_sources.pop() if len(omega_sources) == 1
                 else "mixed sources, see per-case data" if omega_sources
                 else "n/a")
    fig.suptitle("Periodic hill — U, k, ω field comparison\n"
                f"(U, k: true UMean/kMean running averages; ω: {omega_note})",
                fontsize=13)
    _save(fig, out_dir, "field_contours.png")


# Molecular kinematic viscosity, cases/periodicHill/*/constant/transportProperties
_NU = 2.643e-6  # m^2/s


def plot_coefficient_ratio_contours(cases_data, out_dir):
    """Davidson (2026) Fig 18 equivalent: sigma_k,NN, C_k,NN, C_omega2,NN
    and the total-viscosity ratio, as 2x2 field contours. Unlike his
    absolute-value grayscale panels (a)-(c) (colorbar maxima 1.21, 0.41,
    0.043), these are plotted as a ratio to the standard-kOmega baseline
    value each coefficient replaces (2.0, 1.0, 0.075 -- NN_STANDARD, same
    convention already used for the line-profile version of this comparison
    in pitzDaily/channelFlow5200/flatPlate), on the same 0-2 RdBu_r scale
    centred at 1.0 as pitzDaily's nut_ratio.png -- so all four panels here
    share one reading: blue/red is below/above the standard-kOmega value,
    white is no NN effect. Panel (d) already was a ratio in Davidson's own
    figure (nu_tot,NN / nu_tot,kOmega); kept as such here.

    Needs both transient_kOmega_2D (nut only) and transient_kOmegaDavidsonNN_2D
    (nut + the three NN coefficients) -- both meshes come from the same
    blockMeshDict/topoSetDict, generated independently but deterministically,
    so their cell ordering matches exactly and the ratio can be taken
    array-for-array with no interpolation, same assumption plotPitzDaily.py's
    field_ratio() makes."""
    nn_case_name = next((c for c in cases_data if c == "transient_kOmegaDavidsonNN_2D"), None)
    base_case_name = next((c for c in cases_data if c == "transient_kOmega_2D"), None)
    if nn_case_name is None or base_case_name is None:
        print("  Skipping coefficient-ratio contours -- need both "
              "transient_kOmega_2D and transient_kOmegaDavidsonNN_2D.")
        return

    nn = _load_field_case(nn_case_name, 0.028)
    base = _load_field_case(base_case_name, 0.028)
    if nn is None or base is None or nn["nut"] is None or base["nut"] is None:
        print("  Skipping coefficient-ratio contours -- missing nut data.")
        return
    if nn["nut"].shape != base["nut"].shape:
        print(f"  Skipping coefficient-ratio contours -- cell count mismatch "
              f"({nn['nut'].shape} vs {base['nut'].shape}).")
        return

    panels = [("sigmakNN", r"$\sigma_{k,NN}$ / standard"),
              ("CkNN", r"$C_{k,NN}$ / standard"),
              ("Comega2NN", r"$C_{\omega 2,NN}$ / standard")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    levels = np.linspace(0, 2, 81)
    triang = _hill_masked_triangulation(nn["x"], nn["y"], 0.028)

    cf = None
    for ax, (key, label) in zip(axes.flat[:3], panels):
        if nn[key] is None:
            ax.axis("off")
            continue
        ratio = nn[key] / NN_STANDARD[key]
        cf = ax.tricontourf(triang, ratio, levels=levels, cmap="RdBu_r",
                            norm=norm, extend="both")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)

    ax = axes.flat[3]
    nu_tot_ratio = (_NU + nn["nut"]) / (_NU + base["nut"])
    cf = ax.tricontourf(triang, nu_tot_ratio, levels=levels, cmap="RdBu_r",
                        norm=norm, extend="both")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(r"$\nu_{tot,NN}/\nu_{tot,k-\omega}$", fontsize=11)

    if cf is not None:
        fig.colorbar(cf, ax=axes, label="ratio", ticks=np.linspace(0, 2, 9),
                    shrink=0.9)

    fig.suptitle("Periodic hill — NN coefficients and total-viscosity ratio\n"
                "(cf. Davidson (2026) Fig. 18; coefficients shown relative "
                "to their standard-kOmega baseline, not absolute)",
                fontsize=12)
    _save(fig, out_dir, "coefficient_ratio_contours.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.print_sampledict:
        print_sample_dict(args.H, args.z)
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_data = {}
    for case in args.cases:
        if not Path(case).exists():
            print(f"WARNING: '{case}' not found, skipping.")
            continue
        td = find_latest_time(case, args.sample_subdir)
        if td is None:
            print(f"WARNING: No sample data in {case}/{args.sample_subdir}")
            print(f"  Run {case}'s Allrun to completion first.")
            continue
        print(f"Case '{case}': {td}")
        cases_data[case] = td

    if not cases_data:
        print("\nNo case data found. Suggested `sample` dict:")
        print_sample_dict(args.H, args.z)
        sys.exit(1)

    if not Path(args.dns).exists():
        print(f"Note: DNS directory '{args.dns}' not found — RANS/LES only.\n")

    dns_dir = Path(args.dns)
    print()
    plot_velocity(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_shear_stress(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_tke(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_nn_coefficients(cases_data, args.H, out_dir)
    plot_field_contours(cases_data, args.H, out_dir, args.sample_subdir)
    plot_coefficient_ratio_contours(cases_data, out_dir)
    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
