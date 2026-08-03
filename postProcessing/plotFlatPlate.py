#!/usr/bin/env python3
"""
plotFlatPlate.py

Compare OpenFOAM flat-plate boundary layer results between
kOmegaDavidsonNN and standard kOmega.

Produces:
    1. Skin friction Cf vs Re_theta
    2. U+ vs y+ profile at selected x locations
    3. k+ vs y+ profile at selected x locations
    4. NN coefficient fields along the plate (if available)

Usage:
    python3 plotFlatPlate.py case1_dir [case2_dir] ...

    e.g.:
    python3 plotFlatPlate.py ../flatPlateDavidsonNN ../flatPlate

Requires:
    - wallShearStress field in latest time directory
      (run: simpleFoam -postProcess -func wallShearStress -latestTime)
    - Cell centres in 0/C
      (run: simpleFoam -postProcess -func writeCellCentres -time 0)

Physical parameters (adjust if case differs):
    NU   = 1.5e-5   kinematic viscosity (air)
    UINF = 1.0      freestream velocity

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
NU   = 1.5e-5   # kinematic viscosity m^2/s
UINF = 1.0      # freestream velocity m/s
RHO  = 1.0      # density kg/m^3 (incompressible, normalised)

# Plot styles per case
STYLES = [
    {'color': 'b', 'linestyle': '-',  'linewidth': 2},
    {'color': 'r', 'linestyle': '-',  'linewidth': 2},
    {'color': 'g', 'linestyle': '--', 'linewidth': 2},
]

# x locations for profile extraction (fraction of domain length)
# Profiles extracted at these x positions
PROFILE_X_FRAC = [0.3, 0.5, 0.7, 0.9]


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
# Read cell centres
# ----------------------------------------------------------------
def read_cell_centres(case_dir, component=0):
    cc_file = os.path.join(case_dir, '0', 'C')
    if not os.path.exists(cc_file):
        os.system(
            f'cd {case_dir} && simpleFoam -postProcess '
            f'-func writeCellCentres -time 0 > /dev/null 2>&1')
    if os.path.exists(cc_file):
        return read_of_vector_field(cc_file, component=component)
    print(f"  Warning: could not read cell centres from {case_dir}")
    return None


# ----------------------------------------------------------------
# Read wall patch face centres from postProcess output
# ----------------------------------------------------------------
def read_wall_face_centres_x(case_dir, patch_name='bottom'):
    """
    Read x coordinates of face centres on a wall patch.
    These are written by writeCellCentres to 0/Cf (face centres).
    Falls back to reconstructing from mesh if not available.
    """
    # writeCellCentres writes face centres to 0/Cf
    cf_file = os.path.join(case_dir, '0', 'Cf')
    if not os.path.exists(cf_file):
        os.system(
            f'cd {case_dir} && simpleFoam -postProcess '
            f'-func writeCellCentres -time 0 > /dev/null 2>&1')

    # Try reading face centres file
    if os.path.exists(cf_file):
        # Cf is a surfaceVectorField — we need to find the bottom patch section
        with open(cf_file) as f:
            text = f.read()
        # Find the bottom patch boundary field
        patch_idx = text.find(f'{patch_name}')
        if patch_idx > 0:
            # Find the value block after the patch name
            val_idx = text.find('value', patch_idx)
            if val_idx > 0:
                block_start = text.find('(', val_idx)
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
                x_vals = []
                for t in tuples:
                    vals = t.split()
                    x_vals.append(float(vals[0]))
                if x_vals:
                    return np.array(x_vals)

    # Fallback: reconstruct from cell centres
    # The bottom wall face centres in x are the same as cell centres in x
    # for a structured mesh with uniform x spacing
    x_all = read_cell_centres(case_dir, component=0)
    y_all = read_cell_centres(case_dir, component=1)
    if x_all is not None and y_all is not None:
        y_min    = y_all.min()
        bot_mask = y_all < (y_min + 1e-4)
        x_bot    = x_all[bot_mask]
        return np.sort(x_bot)
    return None


# ----------------------------------------------------------------
# Compute skin friction from wallShearStress field
# ----------------------------------------------------------------
def compute_cf(case_dir):
    """
    Read wallShearStress from latest time directory and compute
    Cf = |tau_w| / (0.5 * rho * Uinf^2) along the bottom wall.

    wallShearStress is a boundary patch field with N_faces values
    (not N_cells), so we read its x-coordinates from the face centres.

    Returns (x_wall, Re_theta, Cf) arrays, or (None, None, None).

    Requires: simpleFoam -postProcess -func wallShearStress -latestTime
    """
    latest   = find_latest_time(case_dir)
    wss_file = os.path.join(case_dir, latest, 'wallShearStress')

    if not os.path.exists(wss_file):
        print(f"  Warning: wallShearStress not found in {case_dir}/{latest}")
        print(f"  Run: simpleFoam -postProcess -func wallShearStress -latestTime")
        return None, None, None

    # wallShearStress is a volVectorField but only boundary values matter.
    # The bottom patch has 150 face values (one per face).
    # We need to extract just the bottom patch section.
    with open(wss_file) as f:
        text = f.read()

    # Find bottom patch boundary field
    patch_idx = text.find('bottom')
    if patch_idx < 0:
        print(f"  Warning: 'bottom' patch not found in wallShearStress")
        return None, None, None

    val_idx = text.find('value', patch_idx)
    if val_idx < 0:
        return None, None, None

    block_start = text.find('(', val_idx)
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
    tau_x_wall = np.array([float(t.split()[0]) for t in tuples])

    print(f"  wallShearStress bottom patch: {len(tau_x_wall)} faces")

    # Get x coordinates of wall faces
    # For a structured mesh, wall face x = cell centre x for bottom row
    x_all = read_cell_centres(case_dir, component=0)
    y_all = read_cell_centres(case_dir, component=1)

    if x_all is None:
        return None, None, None

    y_min    = y_all.min()
    bot_mask = y_all < (y_min + 2e-4)
    x_bot    = x_all[bot_mask]
    print(f"  Bottom wall cells: {bot_mask.sum()}, x range: [{x_bot.min():.4f}, {x_bot.max():.4f}] m")

    # Sort both by x
    order_x      = np.argsort(x_bot)
    x_bot        = x_bot[order_x]

    # tau_x_wall should have same length as bottom wall cells
    if len(tau_x_wall) != len(x_bot):
        print(f"  Note: tau_x size {len(tau_x_wall)}, x_bot size {len(x_bot)}")
        # Take every nth cell to match face count if needed
        step = len(x_bot) // len(tau_x_wall)
        if step > 1:
            x_bot = x_bot[::step][:len(tau_x_wall)]
        else:
            n = min(len(tau_x_wall), len(x_bot))
            x_bot      = x_bot[:n]
            tau_x_wall = tau_x_wall[:n]
    tau_x_wall = tau_x_wall

    x_wall   = x_bot
    tau_wall = np.abs(tau_x_wall)

    # Skin friction coefficient
    q_inf = 0.5 * RHO * UINF**2
    Cf    = tau_wall / q_inf

    # Friction velocity
    u_tau = np.sqrt(tau_wall / RHO)

    # Estimate Re_theta from x position
    # For turbulent flat plate: Re_theta ~ 0.036 * Re_x^0.8 (approximate)
    Re_x     = UINF * x_wall / NU
    Re_theta = 0.036 * Re_x**0.8

    print(f"  Cf range: [{Cf.min():.6f}, {Cf.max():.6f}]")
    print(f"  Re_theta range: [{Re_theta.min():.0f}, {Re_theta.max():.0f}]")

    return x_wall, Re_theta, Cf


# ----------------------------------------------------------------
# Spalding law of the wall (for U+ profile reference)
# ----------------------------------------------------------------
def spalding_uplus(yplus_range):
    """Compute U+ from Spalding's law for reference."""
    kappa = 0.41
    B     = 5.0
    Uplus = np.linspace(0, 30, 1000)
    yplus = (Uplus
             + np.exp(-kappa*B) * (
                 np.exp(kappa*Uplus) - 1
                 - kappa*Uplus
                 - 0.5*(kappa*Uplus)**2
                 - (1/6)*(kappa*Uplus)**3))
    # Interpolate to requested yplus range
    return np.interp(yplus_range, yplus, Uplus)


# ----------------------------------------------------------------
# Experimental Cf correlation (Schlichting)
# ----------------------------------------------------------------
def cf_schlichting(Re_theta):
    """
    Skin friction correlation for turbulent flat plate.
    Schlichting: Cf = 0.0594 * Re_x^(-0.2)
    Expressed in terms of Re_theta using Re_x ~ (Re_theta/0.036)^(1/0.8)
    """
    # White (2006) correlation: Cf = 0.455 / (log10(Re_x))^2.58
    # Using Re_x from Re_theta: Re_x = (Re_theta / 0.036)^(1/0.8)
    Re_x = (Re_theta / 0.036)**(1.0/0.8)
    Cf   = 0.455 / (np.log10(Re_x))**2.58
    return Cf


# ----------------------------------------------------------------
# Extract U and k profiles at given x locations
# ----------------------------------------------------------------
def extract_profiles(case_dir, x_locations):
    """
    Extract wall-normal profiles of U+ and k+ at specified x locations.
    Returns dict of {x_loc: (yplus, Uplus, kplus)} for each x location.
    """
    latest   = find_latest_time(case_dir)
    U_file   = os.path.join(case_dir, latest, 'U')
    k_file   = os.path.join(case_dir, latest, 'k')
    nut_file = os.path.join(case_dir, latest, 'nut')
    wss_file = os.path.join(case_dir, latest, 'wallShearStress')

    if not all(os.path.exists(f) for f in [U_file, k_file, wss_file]):
        print(f"  Warning: required fields not found for profile extraction")
        return {}

    x_all   = read_cell_centres(case_dir, component=0)
    y_all   = read_cell_centres(case_dir, component=1)
    Ux_all  = read_of_vector_field(U_file, component=0)
    k_all   = read_of_field(k_file)

    if x_all is None:
        return {}

    # Get wall shear stress from bottom patch (patch field, not volume field)
    # Re-use the same patch extraction logic as compute_cf
    with open(wss_file) as f:
        wss_text = f.read()
    patch_idx = wss_text.find('bottom')
    tau_wall_all = np.array([])
    x_wall_all   = np.array([])
    if patch_idx >= 0:
        val_idx = wss_text.find('value', patch_idx)
        if val_idx >= 0:
            block_start = wss_text.find('(', val_idx)
            depth = 0
            i = block_start
            while i < len(wss_text):
                if wss_text[i] == '(':
                    depth += 1
                elif wss_text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        block_end = i
                        break
                i += 1
            block  = wss_text[block_start+1:block_end]
            tuples = re.findall(r'\(([^)]+)\)', block)
            tau_x_patch = np.array([float(t.split()[0]) for t in tuples])
            # Match with bottom wall cell centres (sorted by x)
            y_min    = y_all.min()
            bot_mask = y_all < (y_min + 1e-4)
            x_bot    = x_all[bot_mask]
            order    = np.argsort(x_bot)
            x_wall_all   = x_bot[order]
            n = min(len(tau_x_patch), len(x_wall_all))
            tau_wall_all = np.abs(tau_x_patch[:n])
            x_wall_all   = x_wall_all[:n]

    profiles = {}
    x_total  = x_all.max() - x_all.min()

    for x_frac in x_locations:
        x_target = x_all.min() + x_frac * x_total

        # Find column of cells near this x location
        dx      = x_total / 150  # approximate cell width
        col_mask = np.abs(x_all - x_target) < dx

        if not np.any(col_mask):
            continue

        y_col  = y_all[col_mask]
        Ux_col = Ux_all[col_mask]
        k_col  = k_all[col_mask]

        # Sort by y
        order  = np.argsort(y_col)
        y_col  = y_col[order]
        Ux_col = Ux_col[order]
        k_col  = k_col[order]

        # Get local wall shear stress (nearest x_wall)
        idx_wall = np.argmin(np.abs(x_wall_all - x_target))
        tau_w    = tau_wall_all[idx_wall]
        u_tau    = np.sqrt(tau_w / RHO)

        if u_tau < 1e-10:
            continue

        yplus  = y_col * u_tau / NU
        Uplus  = Ux_col / u_tau
        kplus  = k_col / u_tau**2

        x_actual = x_all[col_mask].mean()
        Re_x     = UINF * x_actual / NU
        Re_theta = 0.036 * Re_x**0.8

        profiles[x_frac] = {
            'yplus': yplus, 'Uplus': Uplus, 'kplus': kplus,
            'x': x_actual, 'Re_theta': Re_theta, 'u_tau': u_tau
        }

    return profiles


# ----------------------------------------------------------------
# Read NN coefficient fields along the plate
# ----------------------------------------------------------------
def read_nn_fields_along_plate(case_dir):
    """
    Return mean NN coefficient values along the plate (averaged over y).
    """
    latest   = find_latest_time(case_dir)
    sk_file  = os.path.join(case_dir, latest, 'sigmakNN')
    ck_file  = os.path.join(case_dir, latest, 'CkNN')
    co2_file = os.path.join(case_dir, latest, 'Comega2NN')

    if not os.path.exists(sk_file):
        return None

    x_all      = read_cell_centres(case_dir, component=0)
    sigmak_all = read_of_field(sk_file)
    ck_all     = read_of_field(ck_file)     if os.path.exists(ck_file)  else None
    comega_all = read_of_field(co2_file)    if os.path.exists(co2_file) else None

    if x_all is None:
        return None

    # Bin by x and take mean
    n_bins   = 100
    x_bins   = np.linspace(x_all.min(), x_all.max(), n_bins+1)
    x_mid    = 0.5*(x_bins[:-1] + x_bins[1:])
    sk_mean  = np.zeros(n_bins)
    ck_mean  = np.zeros(n_bins)
    co2_mean = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (x_all >= x_bins[i]) & (x_all < x_bins[i+1])
        if np.any(mask):
            sk_mean[i]  = sigmak_all[mask].mean()
            if ck_all  is not None: ck_mean[i]  = ck_all[mask].mean()
            if comega_all is not None: co2_mean[i] = comega_all[mask].mean()

    return {
        'x': x_mid,
        'sigmak': sk_mean,
        'ck': ck_mean if ck_all is not None else None,
        'comega2': co2_mean if comega_all is not None else None
    }


# ----------------------------------------------------------------
# Read case
# ----------------------------------------------------------------
def read_case(case_dir):
    print(f"\nReading case: {case_dir}")
    label          = os.path.basename(os.path.abspath(case_dir))
    x_wall, Re_theta, Cf = compute_cf(case_dir)
    profiles       = extract_profiles(case_dir, PROFILE_X_FRAC)
    nn_fields      = read_nn_fields_along_plate(case_dir)

    return dict(
        label=label,
        x_wall=x_wall, Re_theta=Re_theta, Cf=Cf,
        profiles=profiles,
        nn_fields=nn_fields
    )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['.']
    cases     = [read_case(d) for d in case_dirs]

    # ----------------------------------------------------------------
    # Plot 1: Cf vs Re_theta
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    # Experimental correlation
    Re_ref = np.linspace(2000, 10000, 200)
    Cf_ref = cf_schlichting(Re_ref)
    ax.plot(Re_ref, Cf_ref, 'k--', linewidth=2,
            label='Schlichting correlation')

    # OpenFOAM results
    for i, c in enumerate(cases):
        if c['Re_theta'] is not None:
            style = STYLES[i % len(STYLES)]
            ax.plot(c['Re_theta'], c['Cf'],
                    label=c['label'], **style)

    # Davidson reference lines (approximate from his Figure 12a)
    ax.axhline(0.0032, color='gray', linestyle=':', linewidth=1,
               label='Davidson kOmega (~14% high)')

    ax.set_xlabel(r'$Re_\theta$')
    ax.set_ylabel(r'$C_f$')
    ax.set_title('Skin friction coefficient — flat-plate boundary layer')
    ax.legend()
    ax.set_xlim([2000, 9000])
    ax.set_ylim([0.002, 0.005])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Cf_vs_Retheta.png', dpi=150)
    print("\nSaved: Cf_vs_Retheta.png")

    # ----------------------------------------------------------------
    # Plot 2: U+ profiles at selected x locations
    # ----------------------------------------------------------------
    cases_with_profiles = [c for c in cases if c['profiles']]
    if cases_with_profiles:
        n_locs = len(PROFILE_X_FRAC)
        fig, axes = plt.subplots(1, n_locs, figsize=(4*n_locs, 6),
                                 sharey=True)
        if n_locs == 1:
            axes = [axes]

        for ax, x_frac in zip(axes, PROFILE_X_FRAC):
            # Law of wall reference
            yp_ref = np.logspace(-0.5, 3.5, 200)
            Up_ref = spalding_uplus(yp_ref)
            ax.semilogx(yp_ref, Up_ref, 'k--', linewidth=1,
                        label='Spalding')

            for i, c in enumerate(cases_with_profiles):
                if x_frac in c['profiles']:
                    p = c['profiles'][x_frac]
                    style = STYLES[i % len(STYLES)]
                    ax.semilogx(p['yplus'], p['Uplus'],
                                label=c['label'], **style)
                    ax.set_title(
                        rf'$x/L={x_frac:.1f}$, '
                        rf'$Re_\theta={p["Re_theta"]:.0f}$',
                        fontsize=12)

            ax.set_xlabel(r'$y^+$')
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlim([0.1, 3000])

        axes[0].set_ylabel(r'$U^+$')
        axes[0].legend(fontsize=10)
        plt.suptitle(r'$U^+$ profiles — flat-plate boundary layer')
        plt.tight_layout()
        plt.savefig('Uplus_profiles.png', dpi=150)
        print("Saved: Uplus_profiles.png")

    # ----------------------------------------------------------------
    # Plot 3: k+ profiles at selected x locations
    # ----------------------------------------------------------------
    if cases_with_profiles:
        fig, axes = plt.subplots(1, n_locs, figsize=(4*n_locs, 6),
                                 sharey=True)
        if n_locs == 1:
            axes = [axes]

        for ax, x_frac in zip(axes, PROFILE_X_FRAC):
            for i, c in enumerate(cases_with_profiles):
                if x_frac in c['profiles']:
                    p = c['profiles'][x_frac]
                    style = STYLES[i % len(STYLES)]
                    ax.plot(p['yplus'], p['kplus'],
                            label=c['label'], **style)
                    ax.set_title(
                        rf'$x/L={x_frac:.1f}$, '
                        rf'$Re_\theta={p["Re_theta"]:.0f}$',
                        fontsize=12)
            ax.set_xlabel(r'$y^+$')
            ax.set_xlim([0, 200])
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r'$k^+$')
        axes[0].legend(fontsize=10)
        plt.suptitle(r'$k^+$ profiles — flat-plate boundary layer')
        plt.tight_layout()
        plt.savefig('kplus_profiles.png', dpi=150)
        print("Saved: kplus_profiles.png")

    # ----------------------------------------------------------------
    # Plot 4: NN coefficient fields along the plate
    # ----------------------------------------------------------------
    nn_cases = [c for c in cases if c['nn_fields'] is not None]
    if nn_cases:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for c in nn_cases:
            nn = c['nn_fields']
            Re_x_nn = UINF * nn['x'] / NU

            axes[0].plot(Re_x_nn, nn['sigmak'], label=c['label'],
                         **STYLES[cases.index(c) % len(STYLES)])
            axes[0].axhline(2.0, color='k', linestyle='--',
                            linewidth=1, label='Standard kOmega')
            axes[0].set_xlabel(r'$Re_x$')
            axes[0].set_ylabel(r'$\sigma_{k,NN}$')
            axes[0].set_title(r'$\sigma_{k,NN}$')
            axes[0].grid(True, alpha=0.3)

            if nn['ck'] is not None:
                axes[1].plot(Re_x_nn, nn['ck'], label=c['label'],
                             **STYLES[cases.index(c) % len(STYLES)])
                axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1)
                axes[1].set_xlabel(r'$Re_x$')
                axes[1].set_ylabel(r'$C_{k,NN}$')
                axes[1].set_title(r'$C_{k,NN}$')
                axes[1].grid(True, alpha=0.3)

            if nn['comega2'] is not None:
                axes[2].plot(Re_x_nn, nn['comega2'], label=c['label'],
                             **STYLES[cases.index(c) % len(STYLES)])
                axes[2].axhline(0.072, color='k', linestyle='--',
                                linewidth=1)
                axes[2].set_xlabel(r'$Re_x$')
                axes[2].set_ylabel(r'$C_{\omega2,NN}$')
                axes[2].set_title(r'$C_{\omega2,NN}$')
                axes[2].grid(True, alpha=0.3)

        axes[0].legend(fontsize=10)
        plt.suptitle('NN coefficient fields — flat-plate boundary layer')
        plt.tight_layout()
        plt.savefig('NN_coefficients_flatplate.png', dpi=150)
        print("Saved: NN_coefficients_flatplate.png")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'Case':<40} {'Cf_min':>10} {'Cf_max':>10} "
          f"{'Re_theta_max':>14}")
    print('-' * 76)
    for c in cases:
        if c['Cf'] is not None:
            print(f"{c['label']:<40} {c['Cf'].min():>10.6f} "
                  f"{c['Cf'].max():>10.6f} {c['Re_theta'].max():>14.0f}")
    Re_ref2 = np.array([2550, 8000])
    Cf_ref2 = cf_schlichting(Re_ref2)
    print(f"{'Schlichting correlation':<40} {Cf_ref2[0]:>10.6f} "
          f"{Cf_ref2[1]:>10.6f} {'8000':>14}")


if __name__ == '__main__':
    main()
