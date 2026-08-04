#!/usr/bin/env python3
"""
plotPitzDaily.py

Compare OpenFOAM pitzDaily results between kOmegaDavidsonNN and standard kOmega.
Produces plots of:
    1. Reattachment length (Ux sign change in near-wall cells)
    2. U field 2D overview
    3. k field 2D overview
    4. NN coefficient fields (if available)
    5. U and k profiles at fixed x/H stations (sampling lines), run
       automatically via `postProcess -dict system/sampleDict`

Usage:
    python3 plotPitzDaily.py case1_dir [case2_dir] ...

    e.g.:
    python3 plotPitzDaily.py ../pitzDaily ../pitzDailyStandardKW

Reattachment length is computed directly from the U field — no postProcess
function objects required. The reattachment point is where Ux changes sign
in the near-wall cells on the bottom wall (y < Y_WALL_THRESHOLD).

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, LogNorm
import os
import sys
import re
import subprocess

plt.rcParams.update({'font.size': 14})

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
STYLES = [
    {'color': 'b', 'linestyle': '-',  'linewidth': 2},
    {'color': 'r', 'linestyle': '-',  'linewidth': 2},
    {'color': 'g', 'linestyle': '--', 'linewidth': 2},
]

# Experimental reattachment length: Pitz & Daily (1983) report x/H=7.0 at
# Re_H=22,000. This case's Re_H=25,400 (U=10 m/s, H=0.0254m, nu=1e-5) is
# somewhat higher -- the OpenFOAM tutorial geometry matches their step and
# expansion ratio but was never tuned to match their flow conditions exactly.
REATTACHMENT_EXP = 7.0    # x/H, Pitz & Daily (1983), Re_H=22,000
# Step height H: the mesh floor drops from y=0 (inlet duct, blockMeshDict
# vertices 0/3) to y=-0.0254 m (expanded duct, vertices 2/5) at x=0 -- i.e.
# H=25.4mm, not 12.7mm. Using the wrong (half) value here previously made
# every x/H figure (reattachment length, profile station positions) read
# almost exactly double its true value.
STEP_HEIGHT      = 0.0254  # m (step height H for pitzDaily tutorial)

# Downstream of the step the floor drops from y=0 (inlet channel) to
# y=-0.0254 m (expanded channel) -- "near-wall" must be measured relative to
# THIS floor, not y=0. WALL_BAND selects a thin band of cells just above it.
LOWER_WALL_Y = -0.0254     # m, downstream lower-wall y-coordinate
WALL_BAND    = 0.004       # m (~0.3H)

# Minimum x to start looking for reattachment: skips the small corner-vortex
# complex immediately behind the step (a real secondary eddy with its own
# sign changes in Ux, extending to x~0.031 m here) so the search finds the
# primary recirculation zone's reattachment, not a corner-eddy artefact.
X_REATTACH_MIN   = 0.032   # m (~2.5H downstream of step)

# Inlet velocity (0/U fixedValue), used to normalise profile plots
UINF = 10.0  # m/s

# NN coefficient baseline values -- matches the reference lines used in
# plotChannelFlow.py / plotFlatPlate.py's NN coefficient plots, so "fraction
# of baseline" here means the same thing project-wide. sigmakNN/CkNN are the
# model's own dimensionedScalar initial values (kOmegaDavidsonNN.C); the
# Comega2NN reference is the standard Wilcox kOmega beta1, used everywhere
# else in this repo as the "no NN correction" comparison point (the model's
# own numerical initial value is 0.075 = 3/40, close but not the same).
NN_BASELINE = {'sigmakNN': 2.0, 'CkNN': 1.0, 'Comega2NN': 0.072}

# x/H locations for profile plots -- spans the recirculation zone and
# straddles the actual reattachment point (~7.3H, see compute_reattachment)
PROFILE_X = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # x/H


# ----------------------------------------------------------------
# Utility: find latest time directory
# ----------------------------------------------------------------
def find_latest_time(case_dir):
    dirs = []
    for d in os.listdir(case_dir):
        try:
            dirs.append(float(d))
        except ValueError:
            pass
    if not dirs:
        raise RuntimeError(f"No time directories found in {case_dir}")
    latest = max(dirs)
    return str(int(latest)) if latest == int(latest) else str(latest)


# ----------------------------------------------------------------
# Read OpenFOAM scalar field
# ----------------------------------------------------------------
def read_of_field(filepath):
    with open(filepath) as f:
        text = f.read()
    if 'nonuniform' in text.lower():
        idx         = text.lower().find('nonuniform')
        block_start = text.find('(', idx)
        block_end   = text.find(')', block_start)
        numbers     = text[block_start+1:block_end].split()
        return np.array([float(x) for x in numbers])
    elif 'uniform' in text.lower():
        idx = text.lower().find('uniform')
        val = float(text[idx:].split()[1].rstrip(';'))
        return np.array([val])
    return np.array([])


# ----------------------------------------------------------------
# Read OpenFOAM vector field (one component)
# ----------------------------------------------------------------
def read_of_vector_field(filepath, component=0):
    with open(filepath) as f:
        text = f.read()
    values = []
    if 'nonuniform' in text.lower():
        idx         = text.lower().find('nonuniform')
        block_start = text.find('(', idx)
        depth = 0
        i = block_start
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    block_end = i
                    break
            i += 1
        block  = text[block_start+1:block_end]
        tuples = re.findall(r'\(([^)]+)\)', block)
        for t in tuples:
            vals = t.split()
            values.append(float(vals[component]))
    return np.array(values)


# ----------------------------------------------------------------
# Read cell-centre coordinates
# ----------------------------------------------------------------
def read_cell_centres(case_dir, component=0):
    """Read cell centre coordinates. component: 0=x, 1=y, 2=z"""
    cc_file = os.path.join(case_dir, '0', 'C')
    if not os.path.exists(cc_file):
        os.system(
            f'cd {case_dir} && postProcess -func writeCellCentres '
            f'-time 0 > /dev/null 2>&1')
    if os.path.exists(cc_file):
        return read_of_vector_field(cc_file, component=component)
    print(f"  Warning: could not read cell centres from {case_dir}")
    return None


# ----------------------------------------------------------------
# Build a triangulation for contour plots, masked so it doesn't bridge
# through the solid step corner
# ----------------------------------------------------------------
def masked_triangulation(x, y):
    """
    Delaunay-triangulate the (scattered, cell-centre) field data for use
    with tricontourf. The domain is L-shaped -- the step corner (x<0, y<0,
    upstream of the step and below the inlet-duct floor) is solid, but a
    plain Delaunay triangulation is convex and will happily draw triangles
    that cut straight across that corner, connecting inlet-duct cells to
    downstream-duct cells through solid geometry. Any such triangle has its
    centroid in the excluded region (no real cells are ever there), so it's
    identified and masked out by centroid location alone.
    """
    triang = mtri.Triangulation(x, y)
    xc = x[triang.triangles].mean(axis=1)
    yc = y[triang.triangles].mean(axis=1)
    triang.set_mask((xc < 0) & (yc < 0))
    return triang


def find_nn_and_baseline(cases):
    """The first NN case (has sigmak_of) and first non-NN case, or (None, None)."""
    nn_case   = next((c for c in cases if c['sigmak_of'] is not None), None)
    base_case = next((c for c in cases if c['sigmak_of'] is None), None)
    return nn_case, base_case


def field_ratio(nn_case, base_case, key):
    """nn_case[key] / base_case[key], or None if either is missing/mismatched."""
    if nn_case is None or base_case is None:
        return None
    a, b = nn_case.get(key), base_case.get(key)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        print(f"  Warning: {key} ratio skipped -- {nn_case['label']} and "
              f"{base_case['label']} have different cell counts "
              f"({a.shape} vs {b.shape})")
        return None
    return a / b


def plot_ratio_panel(ax, nn_case, ratio, field_label):
    """Draw a NN/baseline ratio contour on ax, sharing the nut_ratio convention:
    RdBu_r, centred at 1.0 (no NN effect), fixed 0-2 range."""
    triang = masked_triangulation(nn_case['x'], nn_case['y'])
    norm   = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    cf = ax.tricontourf(triang, ratio, levels=np.linspace(0, 2, 81),
                         cmap='RdBu_r', norm=norm, extend='both')
    plt.colorbar(cf, ax=ax, label=field_label, ticks=np.linspace(0, 2, 9))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')


# ----------------------------------------------------------------
# Compute reattachment length directly from U field
# ----------------------------------------------------------------
def compute_reattachment(x_all, y_all, Ux_all):
    """
    Find reattachment length from the Ux field.
    Strategy: for each x-column of near-wall cells, take the cell with
    minimum y (closest to wall) and find where its Ux changes sign.
    Returns (x_reattachment_metres, x_reattachment_over_H) or (None, None).
    """
    if x_all is None or y_all is None or Ux_all is None:
        return None, None

    # Select cells in a thin band just above the downstream lower wall
    mask = (y_all < LOWER_WALL_Y + WALL_BAND) & (x_all > X_REATTACH_MIN)
    if not np.any(mask):
        print(f"  Warning: no near-wall cells found with y < {LOWER_WALL_Y + WALL_BAND} m and x > {X_REATTACH_MIN} m")
        print(f"  Min y for x > {X_REATTACH_MIN}: {y_all[x_all > X_REATTACH_MIN].min():.6f} m")
        return None, None

    x_wall  = x_all[mask]
    y_wall  = y_all[mask]
    Ux_wall = Ux_all[mask]
    print(f"  Near-wall cells: {len(x_wall)}, x=[{x_wall.min():.4f},{x_wall.max():.4f}] m, "
          f"y=[{y_wall.min():.5f},{y_wall.max():.5f}] m")
    print(f"  Ux range: [{Ux_wall.min():.4f}, {Ux_wall.max():.4f}] m/s")

    # Bin cells by x and take the minimum-y cell in each bin
    # This gives us a 1D profile of the wall-nearest Ux vs x
    n_bins   = 200
    x_min, x_max = x_wall.min(), x_wall.max()
    bin_edges    = np.linspace(x_min, x_max, n_bins + 1)
    x_profile    = []
    Ux_profile   = []

    for i in range(n_bins):
        in_bin = (x_wall >= bin_edges[i]) & (x_wall < bin_edges[i+1])
        if np.any(in_bin):
            # Take the cell closest to the wall (minimum y) in this bin
            idx_min_y = np.argmin(y_wall[in_bin])
            x_profile.append(x_wall[in_bin][idx_min_y])
            Ux_profile.append(Ux_wall[in_bin][idx_min_y])

    x_profile  = np.array(x_profile)
    Ux_profile = np.array(Ux_profile)

    if len(x_profile) < 2:
        print(f"  Warning: insufficient binned profile points")
        return None, None

    # Find negative-to-positive sign changes in the wall-nearest profile
    signs        = np.sign(Ux_profile)
    sign_changes = np.where(np.diff(signs) > 0)[0]

    if len(sign_changes) == 0:
        print(f"  Warning: no reattachment found in wall-nearest profile")
        print(f"  Ux profile range: [{Ux_profile.min():.4f}, {Ux_profile.max():.4f}]")
        return None, None

    # Take the first sign change (primary reattachment)
    i   = sign_changes[0]
    dx  = x_profile[i+1] - x_profile[i]
    dUx = Ux_profile[i+1] - Ux_profile[i]
    x_r = x_profile[i] - Ux_profile[i] * dx / dUx if abs(dUx) > 1e-10 else x_profile[i]
    print(f"  Reattachment at x = {x_r:.4f} m (x/H = {x_r/STEP_HEIGHT:.2f})")
    return x_r, x_r / STEP_HEIGHT


# ----------------------------------------------------------------
# Read case data
# ----------------------------------------------------------------
def read_case(case_dir):
    print(f"\nReading case: {case_dir}")
    latest   = find_latest_time(case_dir)
    time_dir = os.path.join(case_dir, latest)
    label    = os.path.basename(os.path.abspath(case_dir))

    # Read main fields
    U_file     = os.path.join(time_dir, 'U')
    k_file     = os.path.join(time_dir, 'k')
    nut_file   = os.path.join(time_dir, 'nut')
    omega_file = os.path.join(time_dir, 'omega')

    Ux    = read_of_vector_field(U_file, component=0) if os.path.exists(U_file)     else None
    Uy    = read_of_vector_field(U_file, component=1) if os.path.exists(U_file)     else None
    k     = read_of_field(k_file)                     if os.path.exists(k_file)     else None
    nut   = read_of_field(nut_file)                   if os.path.exists(nut_file)   else None
    omega = read_of_field(omega_file)                 if os.path.exists(omega_file) else None

    # Read NN fields if available
    sk_file  = os.path.join(time_dir, 'sigmakNN')
    ck_file  = os.path.join(time_dir, 'CkNN')
    co2_file = os.path.join(time_dir, 'Comega2NN')
    sigmak_of  = read_of_field(sk_file)  if os.path.exists(sk_file)  else None
    ck_of      = read_of_field(ck_file)  if os.path.exists(ck_file)  else None
    comega2_of = read_of_field(co2_file) if os.path.exists(co2_file) else None

    # Read cell centres
    x = read_cell_centres(case_dir, component=0)
    y = read_cell_centres(case_dir, component=1)

    # Compute reattachment length directly from U field
    x_r, xH = compute_reattachment(x, y, Ux)
    if x_r is not None:
        print(f"  Reattachment length: x = {x_r:.4f} m  (x/H = {xH:.2f})")
    else:
        print(f"  Reattachment length: could not determine")

    # Sample U and k profiles at fixed x/H stations
    x_positions_m = [xh * STEP_HEIGHT for xh in PROFILE_X]
    write_sample_dict(case_dir, x_positions_m)
    run_sample(case_dir)
    profiles = read_profiles(case_dir, latest, len(PROFILE_X))

    return dict(
        label=label, latest=latest, time_dir=time_dir,
        Ux=Ux, Uy=Uy, k=k, nut=nut, omega=omega, x=x, y=y, profiles=profiles,
        sigmak_of=sigmak_of, ck_of=ck_of, comega2_of=comega2_of,
        x_reattachment=x_r, x_reattachment_H=xH
    )


# ----------------------------------------------------------------
# Generate sampleDict for profile extraction
# ----------------------------------------------------------------
def write_sample_dict(case_dir, x_positions_m):
    """
    Write a sampleDict to extract U and k profiles at specified x locations.
    OpenFOAM v2606's postProcess -dict expects a functionObject-list-style
    file: one or more NAMED top-level entries (like controlDict's `functions`
    block, but without the wrapping `functions{}`), not a bare {type sets;...}
    at file scope -- the older bare form is silently ignored (no error, no
    output) by the modern postProcess -dict reader.
    """
    sample_dict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      sampleDict;
}

functions
{

profiles
{
    type            sets;
    libs            (sampling);
    interpolationScheme cellPoint;
    setFormat       raw;

    fields (U k);

    sets
    (
"""
    for i, x in enumerate(x_positions_m):
        sample_dict += f"""
        profile_x{i}
        {{
            type        uniform;
            axis        y;
            start       ({x:.6f} {LOWER_WALL_Y:.6f} 0.0);
            end         ({x:.6f} {-LOWER_WALL_Y:.6f} 0.0);
            nPoints     200;
        }}
"""
    sample_dict += "    );\n}\n\n}\n"

    outpath = os.path.join(case_dir, 'system', 'sampleDict')
    with open(outpath, 'w') as f:
        f.write(sample_dict)
    print(f"  Written: {outpath}")


# ----------------------------------------------------------------
# Run the sample function object and read the resulting profiles
# ----------------------------------------------------------------
def run_sample(case_dir):
    result = subprocess.run(
        ['postProcess', '-dict', 'system/sampleDict', '-fields', '(U k)',
         '-latestTime'],
        cwd=case_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: postProcess sampling failed in {case_dir}")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])


def read_profiles(case_dir, latest, n_stations):
    """
    Read the profile_xN_k_U.xy files written by run_sample().
    Columns: y, k, Ux, Uy, Uz.
    Returns a list of dicts (one per station) or None if unavailable.
    """
    sample_dir = os.path.join(case_dir, 'postProcessing', 'profiles', latest)
    profiles = []
    for i in range(n_stations):
        fpath = os.path.join(sample_dir, f'profile_x{i}_k_U.xy')
        if not os.path.exists(fpath):
            profiles.append(None)
            continue
        data = np.loadtxt(fpath)
        profiles.append(dict(
            y=data[:, 0], k=data[:, 1], Ux=data[:, 2], Uy=data[:, 3]))
    return profiles


# ----------------------------------------------------------------
# Plot U and k profiles at fixed x/H sampling stations
# ----------------------------------------------------------------
def plot_profiles(cases):
    cases_with_profiles = [c for c in cases if c.get('profiles') is not None]
    if not cases_with_profiles:
        return

    colors = {c['label']: STYLES[i % len(STYLES)]['color']
              for i, c in enumerate(cases_with_profiles)}

    n = len(PROFILE_X)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5), sharey=True)
    for j, xH in enumerate(PROFILE_X):
        ax = axes[j]
        for c in cases_with_profiles:
            prof = c['profiles'][j]
            if prof is None:
                continue
            ax.plot(prof['Ux'] / UINF, prof['y'] / STEP_HEIGHT,
                    color=colors[c['label']], label=c['label'])
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'x/H={xH:g}')
        ax.set_xlabel(r'$U_x/U_{ref}$')
        if j == 0:
            ax.set_ylabel(r'$y/H$')
    axes[-1].legend(fontsize=9)
    plt.suptitle('Velocity profiles at fixed x/H stations')
    plt.tight_layout()
    plt.savefig('U_profiles.png', dpi=150)
    print("Saved: U_profiles.png")

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5), sharey=True)
    for j, xH in enumerate(PROFILE_X):
        ax = axes[j]
        for c in cases_with_profiles:
            prof = c['profiles'][j]
            if prof is None:
                continue
            ax.plot(prof['k'] / UINF**2, prof['y'] / STEP_HEIGHT,
                    color=colors[c['label']], label=c['label'])
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'x/H={xH:g}')
        ax.set_xlabel(r'$k/U_{ref}^2$')
        if j == 0:
            ax.set_ylabel(r'$y/H$')
    axes[-1].legend(fontsize=9)
    plt.suptitle('Turbulent kinetic energy profiles at fixed x/H stations')
    plt.tight_layout()
    plt.savefig('k_profiles.png', dpi=150)
    print("Saved: k_profiles.png")


# ----------------------------------------------------------------
# Plot NN coefficient fields
# ----------------------------------------------------------------
def plot_nn_coefficients(cases):
    nn_cases = [c for c in cases if c['sigmak_of'] is not None]
    if not nn_cases:
        return

    # Fields shown as a fraction of their baseline (pre-NN-correction) value,
    # so all three share one interpretable scale: 1.0 = standard model,
    # above/below = where the NN pushes the coefficient up/down.
    fields = [
        ('sigmak_of',  'sigmakNN',  r'$\sigma_{k,NN}$ / baseline'),
        ('ck_of',      'CkNN',      r'$C_{k,NN}$ / baseline'),
        ('comega2_of', 'Comega2NN', r'$C_{\omega2,NN}$ / baseline'),
    ]

    for c in nn_cases:
        if c['x'] is None:
            continue

        present = [(key, base_key, title) for key, base_key, title in fields
                   if c[key] is not None]
        fig, axes = plt.subplots(len(present), 1, figsize=(14, 3.2*len(present)))
        if len(present) == 1:
            axes = [axes]

        triang = masked_triangulation(c['x'], c['y'])
        norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.2)
        levels = np.linspace(0.4, 1.2, 81)
        for ax, (key, base_key, title) in zip(axes, present):
            frac = c[key] / NN_BASELINE[base_key]
            cf = ax.tricontourf(triang, frac, levels=levels, cmap='RdBu_r',
                                 norm=norm, extend='both')
            plt.colorbar(cf, ax=ax, label='fraction of baseline',
                         ticks=np.linspace(0.4, 1.2, 9))
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
            ax.set_title(title)

        plt.suptitle(f'NN coefficient fields — {c["label"]}')
        plt.tight_layout()
        fname = f'NN_coefficients_{c["label"]}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")


# ----------------------------------------------------------------
# Plot the ratio of turbulent viscosity, NN model over baseline kOmega
# ----------------------------------------------------------------
def plot_nut_ratio(cases):
    nn_case, base_case = find_nn_and_baseline(
        [c for c in cases if c['nut'] is not None])
    ratio = field_ratio(nn_case, base_case, 'nut')
    if ratio is None:
        return

    fig, ax = plt.subplots(figsize=(14, 3.2))
    plot_ratio_panel(ax, nn_case, ratio, r'$\nu_{t,NN} / \nu_{t,kOmega}$')
    ax.set_title(f"Turbulent viscosity ratio — {nn_case['label']} / {base_case['label']}")

    plt.tight_layout()
    plt.savefig('nut_ratio.png', dpi=150)
    print("Saved: nut_ratio.png")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['.']

    cases = [read_case(d) for d in case_dirs]

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'Case':<40} {'Reattachment x/H':>18} {'Iterations':>12}")
    print('-' * 72)
    for c in cases:
        xH = f"{c['x_reattachment_H']:.2f}" if c['x_reattachment_H'] is not None else "N/A"
        print(f"{c['label']:<40} {xH:>18} {'see log':>12}")
    print(f"{'Experiment (Pitz & Daily 1983, Re_H=22000)':<40} {REATTACHMENT_EXP:>18.1f} {'—':>12}")

    # ----------------------------------------------------------------
    # Plot 1: k field contour plot (2D overview)
    # ----------------------------------------------------------------
    cases_with_coords = [c for c in cases if c['x'] is not None
                         and c['k'] is not None]
    if cases_with_coords:
        nn_case, base_case = find_nn_and_baseline(cases_with_coords)
        k_ratio = field_ratio(nn_case, base_case, 'k')
        nrows = len(cases_with_coords) + (1 if k_ratio is not None else 0)

        fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.2*nrows))
        if nrows == 1:
            axes = [axes]

        for ax, c in zip(axes, cases_with_coords):
            triang = masked_triangulation(c['x'], c['y'])
            cf = ax.tricontourf(triang, c['k'], levels=100, cmap='hot_r',
                                 vmin=0, vmax=np.percentile(c['k'], 99))
            plt.colorbar(cf, ax=ax, label=r'$k$ (m²/s²)')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
            ax.set_title(f"k field — {c['label']}")
            # Mark reattachment
            if c['x_reattachment'] is not None:
                ax.axvline(c['x_reattachment'], color='w',
                           linestyle='--', linewidth=1,
                           label=f"Reattachment x={c['x_reattachment']:.3f}m")
                ax.legend(fontsize=10)

        if k_ratio is not None:
            ax = axes[-1]
            plot_ratio_panel(ax, nn_case, k_ratio,
                              f"$k_{{NN}}/k_{{standard}}$")
            ax.set_title(f"k ratio — {nn_case['label']} / {base_case['label']}")

        plt.tight_layout()
        plt.savefig('k_field_comparison.png', dpi=150)
        print("\nSaved: k_field_comparison.png")

    # ----------------------------------------------------------------
    # Plot 2: Ux field contour plot
    # ----------------------------------------------------------------
    cases_with_U = [c for c in cases if c['x'] is not None
                    and c['Ux'] is not None]
    if cases_with_U:
        fig, axes = plt.subplots(len(cases_with_U), 1,
                                 figsize=(14, 3.2*len(cases_with_U)))
        if len(cases_with_U) == 1:
            axes = [axes]

        for ax, c in zip(axes, cases_with_U):
            triang = masked_triangulation(c['x'], c['y'])
            cf = ax.tricontourf(triang, c['Ux'], levels=100, cmap='RdBu_r',
                                 vmin=-2, vmax=np.percentile(c['Ux'], 99))
            plt.colorbar(cf, ax=ax, label=r'$U_x$ (m/s)')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
            ax.set_title(f"Velocity field — {c['label']}")
            if c['x_reattachment'] is not None:
                ax.axvline(c['x_reattachment'], color='k',
                           linestyle='--', linewidth=1,
                           label=f"Reattachment x={c['x_reattachment']:.3f}m")
                ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig('U_field_comparison.png', dpi=150)
        print("Saved: U_field_comparison.png")

    # ----------------------------------------------------------------
    # Plot 2b: omega field contour plot
    # ----------------------------------------------------------------
    cases_with_omega = [c for c in cases if c['x'] is not None
                        and c['omega'] is not None]
    if cases_with_omega:
        nn_case, base_case = find_nn_and_baseline(cases_with_omega)
        omega_ratio = field_ratio(nn_case, base_case, 'omega')
        nrows = len(cases_with_omega) + (1 if omega_ratio is not None else 0)

        fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.2*nrows))
        if nrows == 1:
            axes = [axes]

        # Log scale: omega ~ 1/y^2 at the wall spans several orders of
        # magnitude more than the bulk flow -- a linear scale saturates
        # everywhere except a thin near-wall sliver and hides the bulk
        # structure the comparison is actually about.
        omega_min = min(np.percentile(c['omega'], 1) for c in cases_with_omega)
        omega_max = max(c['omega'].max() for c in cases_with_omega)
        omega_min = max(omega_min, 1.0)
        log_norm   = LogNorm(vmin=omega_min, vmax=omega_max)
        log_levels = np.logspace(np.log10(omega_min), np.log10(omega_max), 100)
        decade_lo, decade_hi = int(np.floor(np.log10(omega_min))), int(np.ceil(np.log10(omega_max)))
        tick_locs  = 10.0 ** np.arange(decade_lo, decade_hi + 1)

        for ax, c in zip(axes, cases_with_omega):
            triang = masked_triangulation(c['x'], c['y'])
            cf = ax.tricontourf(triang, c['omega'], levels=log_levels,
                                 cmap='viridis', norm=log_norm, extend='both')
            plt.colorbar(cf, ax=ax, label=r'$\omega$ (1/s)', ticks=tick_locs)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
            ax.set_title(f"Specific dissipation rate — {c['label']}")
            if c['x_reattachment'] is not None:
                ax.axvline(c['x_reattachment'], color='w',
                           linestyle='--', linewidth=1,
                           label=f"Reattachment x={c['x_reattachment']:.3f}m")
                ax.legend(fontsize=10)

        if omega_ratio is not None:
            ax = axes[-1]
            plot_ratio_panel(ax, nn_case, omega_ratio,
                              r"$\omega_{NN}/\omega_{standard}$")
            ax.set_title(f"omega ratio — {nn_case['label']} / {base_case['label']}")

        plt.tight_layout()
        plt.savefig('omega_field_comparison.png', dpi=150)
        print("Saved: omega_field_comparison.png")

    # ----------------------------------------------------------------
    # Plot 3: NN coefficient fields
    # ----------------------------------------------------------------
    plot_nn_coefficients(cases)

    # ----------------------------------------------------------------
    # Plot 4: turbulent viscosity ratio, NN model over baseline kOmega
    # ----------------------------------------------------------------
    plot_nut_ratio(cases)

    # ----------------------------------------------------------------
    # Plot 5: U and k profiles at fixed x/H sampling stations
    # ----------------------------------------------------------------
    plot_profiles(cases)

    plt.show()


if __name__ == '__main__':
    main()
