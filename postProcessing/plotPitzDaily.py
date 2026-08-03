#!/usr/bin/env python3
"""
plotPitzDaily.py

Compare OpenFOAM pitzDaily results between kOmegaDavidsonNN and standard kOmega.
Produces plots of:
    1. Reattachment length (Ux sign change in near-wall cells)
    2. U field 2D overview
    3. k field 2D overview
    4. NN coefficient fields (if available)

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
import os
import sys
import re

plt.rcParams.update({'font.size': 14})

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
STYLES = [
    {'color': 'b', 'linestyle': '-',  'linewidth': 2},
    {'color': 'r', 'linestyle': '-',  'linewidth': 2},
    {'color': 'g', 'linestyle': '--', 'linewidth': 2},
]

# Experimental reattachment length (Pitz & Daily 1983)
REATTACHMENT_EXP = 8.0    # x/H
STEP_HEIGHT      = 0.0127  # m (step height H for pitzDaily tutorial)

# Near-wall threshold: cells with y < this are treated as wall-adjacent
# pitzDaily has a tall recirculation zone — needs a large threshold
Y_WALL_THRESHOLD = 0.025   # m (~2H)

# Minimum x to start looking for reattachment (avoid step corner artefacts)
X_REATTACH_MIN   = 0.02    # m (~1.5H downstream of step)

# x/H locations for profile plots
PROFILE_X = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # x/H


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

    # Select cells downstream of the step within Y_WALL_THRESHOLD
    # y > -0.002 excludes step face cells which have negative y
    mask = (y_all < Y_WALL_THRESHOLD) & (y_all > -0.002) & (x_all > X_REATTACH_MIN)
    if not np.any(mask):
        print(f"  Warning: no near-wall cells found with y < {Y_WALL_THRESHOLD} m and x > {X_REATTACH_MIN} m")
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
    U_file = os.path.join(time_dir, 'U')
    k_file = os.path.join(time_dir, 'k')

    Ux = read_of_vector_field(U_file, component=0) if os.path.exists(U_file) else None
    Uy = read_of_vector_field(U_file, component=1) if os.path.exists(U_file) else None
    k  = read_of_field(k_file)                     if os.path.exists(k_file) else None

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

    return dict(
        label=label, latest=latest, time_dir=time_dir,
        Ux=Ux, Uy=Uy, k=k, x=x, y=y,
        sigmak_of=sigmak_of, ck_of=ck_of, comega2_of=comega2_of,
        x_reattachment=x_r, x_reattachment_H=xH
    )


# ----------------------------------------------------------------
# Generate sampleDict for profile extraction
# ----------------------------------------------------------------
def write_sample_dict(case_dir, x_positions_m):
    """
    Write a sampleDict to extract U and k profiles at specified x locations.
    """
    lines_str = '\n'.join([
        f'        ({x:.6f} -0.0254 0.0)\n'
        f'        ({x:.6f}  0.0508 0.0)'
        for x in x_positions_m
    ])

    sample_dict = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      sampleDict;
}}

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
        start       ({x:.6f} -0.0254 0.0);
        end         ({x:.6f}  0.0508 0.0);
        nPoints     100;
    }}
"""
    sample_dict += ");\n"

    outpath = os.path.join(case_dir, 'system', 'sampleDict')
    with open(outpath, 'w') as f:
        f.write(sample_dict)
    print(f"  Written: {outpath}")
    print(f"  Run: postProcess -func sample -latestTime")


# ----------------------------------------------------------------
# Plot NN coefficient fields
# ----------------------------------------------------------------
def plot_nn_coefficients(cases):
    nn_cases = [c for c in cases if c['sigmak_of'] is not None]
    if not nn_cases:
        return

    for c in nn_cases:
        if c['x'] is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        sc = axes[0].scatter(c['x'], c['y'], c=c['sigmak_of'],
                             s=1, cmap='RdBu_r', vmin=0, vmax=2)
        plt.colorbar(sc, ax=axes[0])
        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('y (m)')
        axes[0].set_title(r'$\sigma_{k,NN}$')

        if c['ck_of'] is not None:
            sc = axes[1].scatter(c['x'], c['y'], c=c['ck_of'],
                                 s=1, cmap='RdBu_r', vmin=0, vmax=1)
            plt.colorbar(sc, ax=axes[1])
            axes[1].set_xlabel('x (m)')
            axes[1].set_ylabel('y (m)')
            axes[1].set_title(r'$C_{k,NN}$')

        if c['comega2_of'] is not None:
            sc = axes[2].scatter(c['x'], c['y'], c=c['comega2_of'],
                                 s=1, cmap='RdBu_r', vmin=0, vmax=0.075)
            plt.colorbar(sc, ax=axes[2])
            axes[2].set_xlabel('x (m)')
            axes[2].set_ylabel('y (m)')
            axes[2].set_title(r'$C_{\omega2,NN}$')

        plt.suptitle(f'NN coefficient fields — {c["label"]}')
        plt.tight_layout()
        fname = f'NN_coefficients_{c["label"]}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")


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
    print(f"{'Experiment (Pitz & Daily 1983)':<40} {'~8.0':>18} {'—':>12}")

    # ----------------------------------------------------------------
    # Plot 1: k field scatter plot (2D overview)
    # ----------------------------------------------------------------
    cases_with_coords = [c for c in cases if c['x'] is not None
                         and c['k'] is not None]
    if cases_with_coords:
        fig, axes = plt.subplots(len(cases_with_coords), 1,
                                 figsize=(12, 4*len(cases_with_coords)))
        if len(cases_with_coords) == 1:
            axes = [axes]

        for ax, c in zip(axes, cases_with_coords):
            sc = ax.scatter(c['x'], c['y'], c=c['k'],
                            s=1, cmap='hot_r',
                            vmin=0, vmax=np.percentile(c['k'], 99))
            plt.colorbar(sc, ax=ax, label=r'$k$ (m²/s²)')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f"k field — {c['label']}")
            # Mark reattachment
            if c['x_reattachment'] is not None:
                ax.axvline(c['x_reattachment'], color='w',
                           linestyle='--', linewidth=1,
                           label=f"Reattachment x={c['x_reattachment']:.3f}m")
                ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig('k_field_comparison.png', dpi=150)
        print("\nSaved: k_field_comparison.png")

    # ----------------------------------------------------------------
    # Plot 2: Ux field scatter plot
    # ----------------------------------------------------------------
    cases_with_U = [c for c in cases if c['x'] is not None
                    and c['Ux'] is not None]
    if cases_with_U:
        fig, axes = plt.subplots(len(cases_with_U), 1,
                                 figsize=(12, 4*len(cases_with_U)))
        if len(cases_with_U) == 1:
            axes = [axes]

        for ax, c in zip(axes, cases_with_U):
            sc = ax.scatter(c['x'], c['y'], c=c['Ux'],
                            s=1, cmap='RdBu_r',
                            vmin=-2, vmax=np.percentile(c['Ux'], 99))
            plt.colorbar(sc, ax=ax, label=r'$U_x$ (m/s)')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
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
    # Plot 3: NN coefficient fields
    # ----------------------------------------------------------------
    plot_nn_coefficients(cases)

    # ----------------------------------------------------------------
    # Write sampleDict for profile extraction
    # ----------------------------------------------------------------
    x_positions_m = [x * STEP_HEIGHT for x in PROFILE_X]
    print(f"\nWriting sampleDict for profile extraction...")
    for case_dir in case_dirs:
        write_sample_dict(case_dir, x_positions_m)

    print("\nTo extract velocity and k profiles at x/H locations, run:")
    print("  postProcess -func sample -latestTime")
    print("in each case directory, then rerun this script with --profiles flag")
    print("(profile plotting requires sample output)")

    plt.show()


if __name__ == '__main__':
    main()
