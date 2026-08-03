"""
postprocess_hill.py
====================
Post-processing script for the OpenFOAM periodic hill case.
Produces figures equivalent to Davidson (2026) Figs 14-17:
  Fig 14  - streamwise velocity U profiles at 6 x-stations
  Fig 15  - Reynolds shear stress u'v' profiles at 6 x-stations
  Fig 16  - turbulent kinetic energy k profiles at 6 x-stations
  Fig 17  - NN coefficient fields sigma_k,NN / C_k,NN / C_omega2,NN

Case parameters (OpenFOAM tutorial / Davidson 2026):
  H   = 0.028 m  (hill height, reference length)
  Ub  = 1.0  m/s (bulk velocity at hill crest)
  Re  = 10565
  nu  = 2.65e-6 m^2/s
  Domain: 9H x 3.035H x 0.001m (1 cell thick, 2D)
  z_mid = 0.0005 m  (sample line z coordinate)

Hill shape: 6-piece piecewise cubic (OpenFOAM blockMeshDict / Almeida)
  x argument runs 0-54 in internal blockMesh units.
  Normalised so hill peak y = H.

Usage
-----
    python postprocess_hill.py [OPTIONS]

    --cases      Space-separated list of case directories
                 default: kOmega kOmegaDavidsonNN
    --dns        Path to DNS data directory (default: ./DNS)
    --H          Hill height in metres (default: 0.028)
    --Ub         Bulk velocity (default: 1.0)
    --z          z coordinate for sample lines (default: 0.0005)
    --out        Output directory for figures (default: ./figures)
    --sample-subdir  Subdirectory for sample data
                     (default: postProcessing/sampleLines)
    --print-sampledict   Print a suggested sampleDict and exit

DNS data (Davidson 2026 zip archive, hill-2D-periodic directory)
----------------------------------------------------------------
Files: DNS_1xh.dat ... DNS_8xh.dat  (x/H = 1-8)
       DNS_x005h.dat                (x/H = 0.005, near hill crest — NOT 0.05)
       DNS_cf.dat                   (skin friction)
Note: no DNS file available at x/H = 0.05; that panel plots RANS only.

Each file has 10 columns:
  0: y/H   1: U/Ub   2: V/Ub   3: u'v'/Ub^2
  4: u'u'/Ub^2   5: v'v'/Ub^2   6: w'w'/Ub^2
  7: (unused)    8: k/Ub^2      9: y/H (repeat)
y/H in column 0 is measured from the domain bottom (absolute), NOT from
the local hill surface — do not add y_wall when plotting.

Reynolds shear stress note
--------------------------
Steady simpleFoam does not produce uPrime2Mean. The script reads the
modelled Reynolds stress tensor field R (symmetric tensor, 6 components:
xx yy zz xy xz yz). Add R to the sampleDict fields list and ensure it
is written by the solver (add 'R' to writeFields in turbulenceProperties,
or use a writeDerivedFields function object).
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# X stations (Davidson Fig 14-17 / Froehlich DNS stations)
# ---------------------------------------------------------------------------

X_STATIONS = [0.05, 1.0, 3.0, 4.0, 5.0, 7.0]

# ---------------------------------------------------------------------------
# DNS file mapping
# Davidson zip archive naming: DNS_Nxh.dat for x/H=N, DNS_x005h.dat for
# x/H=0.005 (near hill crest, NOT x/H=0.05).
# No DNS data available at x/H=0.05 — that panel plots RANS only.
# ---------------------------------------------------------------------------

_DNS_FILENAMES = {
    0.05: "DNS_x05h.dat",
    1.0: "DNS_1xh.dat",
    2.0: "DNS_2xh.dat",
    3.0: "DNS_3xh.dat",
    4.0: "DNS_4xh.dat",
    5.0: "DNS_5xh.dat",
    6.0: "DNS_6xh.dat",
    7.0: "DNS_7xh.dat",
    8.0: "DNS_8xh.dat",
}

# Column indices (0-based) in DNS files
_DNS_COLS = {
    "U":  1,   # U/Ub
    "uv": 3,   # u'v'/Ub^2  (negative in shear layer)
    "k":  8,   # k/Ub^2
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
    "kOmegaDavidsonNN": dict(color="#1f77b4", lw=1.8, ls="-",
                             label=r"$k$-$\omega$-PINN-NN"),
    "kOmega":           dict(color="#ff7f0e", lw=1.8, ls="--",
                             label=r"standard $k$-$\omega$"),
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


def _station_tags(x_station):
    """Return filename fragment variants for a given x/H value."""
    s1 = f"{x_station:g}".replace(".", "p")
    s2 = f"{x_station:.2f}".replace(".", "p")
    s3 = f"{x_station:g}"
    return list(dict.fromkeys([s1, s2, s3]))


def find_sample_file(time_dir, field, x_station):
    """Search for OpenFOAM sample file for given field and x/H station."""
    time_dir = Path(time_dir)
    for tag in _station_tags(x_station):
        for pat in [
            f"{field}_x{tag}*",
            f"x{tag}*{field}*",
            f"*x{tag}*{field}*",
            f"{field}*{tag}*",
        ]:
            matches = sorted(time_dir.glob(pat))
            if matches:
                return matches[0]
    return None


def load_dns(dns_dir, field, x_station):
    """
    Load Davidson DNS data for a given field and x/H station.
    Returns two-column array [y/H, field_value], or None if not available.
    y/H is absolute from domain bottom — do NOT add y_wall when plotting.
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
    d = read_sample_file(fp)
    if d is None or d.shape[1] <= col:
        print(f"  Note: DNS file {fname} has only {d.shape[1] if d is not None else 0} columns")
        return None
    return np.column_stack([d[:, 0], d[:, col]])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+",
                   default=["kOmega", "kOmegaDavidsonNN"])
    p.add_argument("--dns",  default="DNS")
    p.add_argument("--H",    type=float, default=0.028,
                   help="Hill height [m] (default: 0.028)")
    p.add_argument("--Ub",   type=float, default=1.0,
                   help="Bulk velocity [m/s] (default: 1.0)")
    p.add_argument("--z",    type=float, default=0.0005,
                   help="z coordinate for sample lines (default: 0.0005)")
    p.add_argument("--out",  default="figures")
    p.add_argument("--sample-subdir", default="postProcessing/sampleLines",
                   dest="sample_subdir")
    p.add_argument("--print-sampledict", action="store_true",
                   dest="print_sampledict")
    return p.parse_args()


# ---------------------------------------------------------------------------
# sampleDict generator
# ---------------------------------------------------------------------------

def print_sample_dict(H, z_mid):
    print(f"""
// system/sampleLines
// Run: postProcess -func sampleLines -latestTime
// H = {H} m,  z_mid = {z_mid} m
// Note: start y is set just above the hill surface at each x station.

type            sets;
libs            (sampling);
interpolationScheme cellPoint;
setFormat       raw;

fields ( U k R sigmakNN CkNN Comega2NN );

sets
(""")
    for x in X_STATIONS:
        x_phys = x * H
        y_lo   = hill_y_lower(x_phys, H) + 1e-6  # small offset above wall
        y_hi   = 3.035 * H
        tag    = f"{x:g}".replace(".", "p")
        print(f"""    x{tag}
    {{
        type    uniform;
        axis    y;
        start   ({x_phys:.6f} {y_lo:.6f} {z_mid});
        end     ({x_phys:.6f} {y_hi:.6f} {z_mid});
        nPoints 200;
    }}
""")
    print(");")


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


# ---------------------------------------------------------------------------
# Fig 14: Streamwise velocity
# ---------------------------------------------------------------------------

def plot_velocity(cases_data, dns_dir, H, Ub, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        # DNS: y/H is absolute from domain bottom — plot directly, no offset
        dns = load_dns(dns_dir, "U", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        # RANS: sample file y is physical [m], divide by H
        for case, td in cases_data.items():
            sty = CASE_STYLE.get(case,
                                 dict(color="grey", lw=1.5, ls=":", label=case))
            fp = find_sample_file(td, "U", xs)
            if fp is None:
                print(f"  WARNING: U not found for {case} at x/H={xs}")
                continue
            d = read_sample_file(fp)
            if d is None:
                continue
            # columns: y  Ux  Uy  Uz
            ax.plot(d[:, 1] / Ub, d[:, 0] / H, **sty)

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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        # DNS: col 3 = u'v'/Ub^2 (negative in shear layer), y/H absolute
        dns = load_dns(dns_dir, "uv", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        # RANS: R field, columns: y Rxx Ryy Rzz Rxy Rxz Ryz
        # Rxy (col 4) = u'v', already negative in shear layer
        for case, td in cases_data.items():
            sty = CASE_STYLE.get(case,
                                 dict(color="grey", lw=1.5, ls=":", label=case))
            fp = find_sample_file(td, "R", xs)
            if fp is None:
                print(f"  WARNING: R not found for {case} at x/H={xs} "
                      f"(add R to sampleDict fields and writeFields)")
                continue
            d = read_sample_file(fp)
            if d is None:
                continue
            if d.shape[1] < 5:
                print(f"  WARNING: R has only {d.shape[1]} cols at x/H={xs}")
                continue
            ax.plot(d[:, 4] / Ub**2, d[:, 0] / H, **sty)

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel(r"$\overline{u'v'}/U_b^2$", fontsize=9)
        ax.set_ylabel(r"$y/H$", fontsize=9)
        ax.set_ylim([0, 3.05])
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.grid(True, lw=0.4, alpha=0.5)

    _legend(axes.flat[0])
    fig.suptitle(r"Periodic hill — Reynolds shear stress $\overline{u'v'}/U_b^2$",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "hill_shear_stress.png")


# ---------------------------------------------------------------------------
# Fig 16: Turbulent kinetic energy
# ---------------------------------------------------------------------------

def plot_tke(cases_data, dns_dir, H, Ub, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, xs in enumerate(X_STATIONS):
        ax = axes.flat[idx]

        # DNS: col 8 = k/Ub^2, y/H absolute
        dns = load_dns(dns_dir, "k", xs)
        if dns is not None:
            ax.plot(dns[:, 1], dns[:, 0], **DNS_STYLE)

        # RANS: k field, columns: y  k
        for case, td in cases_data.items():
            sty = CASE_STYLE.get(case,
                                 dict(color="grey", lw=1.5, ls=":", label=case))
            fp = find_sample_file(td, "k", xs)
            if fp is None:
                print(f"  WARNING: k not found for {case} at x/H={xs}")
                continue
            d = read_sample_file(fp)
            if d is None:
                continue
            ax.plot(d[:, 1] / Ub**2, d[:, 0] / H, **sty)

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel(r"$k/U_b^2$", fontsize=9)
        ax.set_ylabel(r"$y/H$", fontsize=9)
        ax.set_ylim([0, 3.05])
        ax.set_xlim(left=0)
        ax.grid(True, lw=0.4, alpha=0.5)

    _legend(axes.flat[0])
    fig.suptitle(r"Periodic hill — Turbulent kinetic energy $k/U_b^2$",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "hill_tke.png")


# ---------------------------------------------------------------------------
# Fig 17: NN coefficient fields (zoomed near-wall, 6 panels)
# ---------------------------------------------------------------------------

def plot_nn_coefficients(cases_data, H, out_dir):
    if "kOmegaDavidsonNN" not in cases_data:
        print("  Skipping Fig 17 — kOmegaDavidsonNN not in cases")
        return

    td = cases_data["kOmegaDavidsonNN"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for idx, xs in enumerate(X_STATIONS):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        y_wall_m = hill_y_lower(xs * H, H)
        any_data = False

        for field, color in NN_COEFF_COLORS.items():
            fp = find_sample_file(td, field, xs)
            if fp is None:
                print(f"  WARNING: {field} not found at x/H={xs}")
                continue
            d = read_sample_file(fp)
            if d is None:
                continue
            any_data = True
            # y from local wall surface, normalised by H
            y_from_wall = (d[:, 0] - y_wall_m) / H
            mask = y_from_wall >= 0  # exclude any points below hill surface
            ax.plot(d[mask, 1], y_from_wall[mask],
                    color=color, lw=1.6, label=NN_COEFF_LABELS[field])
            # Standard value reference line
            ax.axvline(NN_STANDARD[field], color=color,
                       lw=0.8, ls="--", alpha=0.5)

        ax.set_title(f"$x/H = {xs}$", fontsize=10)
        ax.set_xlabel("Coefficient value", fontsize=9)
        ax.set_ylabel(r"$(y - y_{wall})/H$", fontsize=9)
        ax.set_ylim([0, 0.15])  # zoomed near-wall, matching Davidson Fig 17
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.print_sampledict:
        print_sample_dict(args.H, args.z)
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate sample data for each case
    cases_data = {}
    for case in args.cases:
        if not Path(case).exists():
            print(f"WARNING: '{case}' not found, skipping.")
            continue
        td = find_latest_time(case, args.sample_subdir)
        if td is None:
            print(f"WARNING: No sample data in {case}/{args.sample_subdir}")
            print(f"  Run inside {case}/: postProcess -func sampleLines -latestTime")
            continue
        print(f"Case '{case}': {td}")
        cases_data[case] = td

    if not cases_data:
        print("\nNo case data found. Suggested sampleDict:")
        print_sample_dict(args.H, args.z)
        sys.exit(1)

    if not Path(args.dns).exists():
        print(f"Note: DNS directory '{args.dns}' not found — RANS only.\n")

    dns_dir = Path(args.dns)
    print()
    plot_velocity(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_shear_stress(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_tke(cases_data, dns_dir, args.H, args.Ub, out_dir)
    plot_nn_coefficients(cases_data, args.H, out_dir)
    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
