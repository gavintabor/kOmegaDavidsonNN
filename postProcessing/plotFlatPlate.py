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
    NU   = 3.57e-5  kinematic viscosity, matches Davidson (2026) Sec. 5.2's
                    precursor RANS setup (literature/pythons-rans-code-RANS-open/
                    boundary-layer-.../exec-pyCALC-RANS.py)
    UINF = 1.0441   freestream velocity at the inlet (last row of Davidson's
                    saved inlet profile -- see generate_inlet_profile.py)

Re_theta is computed from the actual CFD velocity profile via the momentum
thickness, theta(x) = integral of (U/Uinf)(1-U/Uinf) dy, Re_theta = Uinf*theta/nu
-- NOT from a leading-edge power-law correlation. That correlation (removed)
assumed Re_theta=0 at the inlet, which is wrong here: the inlet condition is
Davidson's precursor turbulent boundary layer at Re_theta=2550, not a sharp
leading edge.

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
NU   = 3.57e-5   # kinematic viscosity m^2/s
UINF = 1.0441    # freestream velocity m/s (Davidson's inlet profile, y->max)
RHO  = 1.0      # density kg/m^3 (incompressible, normalised)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DNS_KPLUS_FILE = os.path.join(
    SCRIPT_DIR, 'DNS_data', 'Sillero_Jimenez_Moser_2013_Retheta5500.prof')
DNS_RE_THETA = 5500  # this is the DNS station Davidson's Fig. 12 cites ([13]
                      # = Sillero, Jimenez, Moser 2013); closest one saved
                      # locally to his Reθ=4500 profile, not an exact match

# Plot styles per case
STYLES = [
    {'color': 'b', 'linestyle': '-',  'linewidth': 2},
    {'color': 'r', 'linestyle': '-',  'linewidth': 2},
    {'color': 'g', 'linestyle': '--', 'linewidth': 2},
]

# Re_theta targets for profile extraction -- the x-station for each is
# found from the actual computed Re_theta(x), not picked as a fraction of
# domain length (which would give whatever Re_theta happens to fall out,
# not a chosen physical condition). Includes 4500 (Davidson's Fig. 12
# comparison station) and 5500 (matches the DNS profile available locally).
PROFILE_RE_THETA = [3000, 4000, 4500, 5500]


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
# Select exactly one streamwise column of cells nearest x_target.
#
# blockMesh's graded mesh introduces ~1e-7 floating-point jitter in x
# between mesh rows that are nominally the same column (traced to how it
# evaluates the transfinite mapping for a graded block) -- so a tolerance
# tight enough to reject the neighbouring column (< ~1e-6) fragments a
# single column into two, while a tolerance loose enough to survive that
# jitter (comparable to dx) occasionally merges two adjacent columns.
# Either way, the merged/fragmented result has near-duplicate y values,
# which is harmless for interpolation/trapz but makes np.gradient divide
# by ~0 -- this is what silently dropped the kOmega curve from one Re_theta
# panel in the shear-stress plot (NaNs from a merged column, not caught
# because compute_re_theta's trapz-based use of the same bug tolerated it).
#
# Fix: snap to the closest actual x value in the data first, then use a
# tolerance far above the jitter but far below the column spacing.
# ----------------------------------------------------------------
def column_mask(x_all, x_target, dx):
    candidates = x_all[np.abs(x_all - x_target) < 0.5 * dx]
    if len(candidates) == 0:
        return np.zeros_like(x_all, dtype=bool)
    nearest_x = candidates[np.argmin(np.abs(candidates - x_target))]
    return np.abs(x_all - nearest_x) < 0.1 * dx


# ----------------------------------------------------------------
# Momentum-thickness Re_theta at given x stations, from the actual
# CFD velocity field (not a leading-edge correlation).
# ----------------------------------------------------------------
def compute_re_theta(x_targets, x_all, y_all, Ux_all, dx):
    """
    theta(x) = integral over y of (U/Ue)*(1 - U/Ue) dy
    Re_theta(x) = Ue * theta(x) / nu

    Uses the LOCAL edge velocity Ue(x) (U at the top of the domain in that
    column), not the fixed global UINF: this domain's height is finite
    (20*delta_in), so as the boundary layer grows the core flow mildly
    accelerates (blockage/displacement effect -- confirmed by checking U at
    the domain top, which rises measurably from inlet to outlet). Using a
    fixed reference velocity there is the standard textbook definition for
    an unconfined boundary layer, but here it makes theta(x) -- and hence
    Re_theta(x) -- spuriously DECREASE downstream instead of growing, since
    the accelerated outer flow contributes a growing negative term to the
    integral. The local edge velocity is the physically correct reference.

    x_targets : wall x-stations to evaluate at
    x_all, y_all, Ux_all : cell-centre coordinates and streamwise velocity
                           for every cell in the domain
    dx : approximate streamwise cell width, used to select the column of
         cells nearest each x station
    """
    re_theta = np.full(len(x_targets), np.nan)
    for i, xt in enumerate(x_targets):
        mask = column_mask(x_all, xt, dx)
        if not np.any(mask):
            continue
        y_col = y_all[mask]
        U_col = Ux_all[mask]
        order = np.argsort(y_col)
        y_col, U_col = y_col[order], U_col[order]
        Ue = U_col[-1]  # value at the top of the domain, this column
        theta = np.trapz((U_col / Ue) * (1.0 - U_col / Ue), y_col)
        re_theta[i] = Ue * theta / NU
    return re_theta


def find_x_for_re_theta(x_all, y_all, Ux_all, targets):
    """
    Invert Re_theta(x) to find the x-station where each target Re_theta is
    reached, by evaluating it at all 150 streamwise stations and
    interpolating. Re_theta(x) grows monotonically along this plate, so
    this is well posed.
    """
    x_total = x_all.max() - x_all.min()
    dx      = x_total / 150
    x_stations = np.linspace(x_all.min() + 0.5*dx, x_all.max() - 0.5*dx, 150)
    re_theta_stations = compute_re_theta(x_stations, x_all, y_all, Ux_all, dx)
    return np.interp(targets, re_theta_stations, x_stations)


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

    # Re_theta from the actual velocity field's momentum thickness (not a
    # leading-edge correlation -- the inlet here starts at Re_theta=2550,
    # not 0, so that correlation would be wrong on top of being approximate).
    latest  = find_latest_time(case_dir)
    Ux_full = read_of_vector_field(os.path.join(case_dir, latest, 'U'), component=0)
    x_total = x_all.max() - x_all.min()
    dx      = x_total / 150
    Re_theta = compute_re_theta(x_wall, x_all, y_all, Ux_full, dx)

    print(f"  Cf range: [{Cf.min():.6f}, {Cf.max():.6f}]")
    print(f"  Re_theta range: [{Re_theta.min():.0f}, {Re_theta.max():.0f}]")

    return x_wall, Re_theta, Cf



# ----------------------------------------------------------------
# DNS reference k+ profile (Sillero, Jimenez & Moser 2013 -- the dataset
# Davidson (2026) cites as ref. [13] for Fig. 12), for overlay on the k+
# plot. Columns: y/d99, y+, urms+, vrms+, wrms+, ... (all in wall units,
# so k+ = 0.5*(urms+^2 + vrms+^2 + wrms+^2)).
# ----------------------------------------------------------------
def read_dns_profile():
    """Returns (yplus, Uplus, kplus, uvplus), or (None, None, None, None)
    if unavailable."""
    if not os.path.exists(DNS_KPLUS_FILE):
        return None, None, None, None
    d = np.loadtxt(DNS_KPLUS_FILE, comments='%')
    yplus  = d[:, 1]
    urms, vrms, wrms = d[:, 2], d[:, 3], d[:, 4]
    uvplus = d[:, 5]  # already in wall units
    Uplus  = d[:, 8]  # umed, already in wall units
    kplus  = 0.5 * (urms**2 + vrms**2 + wrms**2)
    return yplus, Uplus, kplus, uvplus


# ----------------------------------------------------------------
# Skin friction correlation, matching Davidson (2026) Fig. 12(a)
# ----------------------------------------------------------------
def cf_davidson(Re_theta):
    """Cf(Re_theta) correlation as plotted in Fig. 12(a) -- NOT the
    Schlichting/White correlation used here previously, which was a
    different curve on different (log, wide-range) axes entirely."""
    return 2.0 * (1.0/0.384 * np.log(Re_theta) + 4.127)**-2


# ----------------------------------------------------------------
# Extract U and k profiles at given x locations
# ----------------------------------------------------------------
def extract_profiles(case_dir, re_theta_targets):
    """
    Extract wall-normal profiles of U+ and k+ at the x-stations where the
    given Re_theta targets are reached (found via find_x_for_re_theta).
    Returns dict of {target_re_theta: {...}} for each target.
    """
    latest   = find_latest_time(case_dir)
    U_file   = os.path.join(case_dir, latest, 'U')
    k_file   = os.path.join(case_dir, latest, 'k')
    nut_file = os.path.join(case_dir, latest, 'nut')
    wss_file = os.path.join(case_dir, latest, 'wallShearStress')
    sk_file  = os.path.join(case_dir, latest, 'sigmakNN')
    ck_file  = os.path.join(case_dir, latest, 'CkNN')
    co2_file = os.path.join(case_dir, latest, 'Comega2NN')

    if not all(os.path.exists(f) for f in [U_file, k_file, wss_file]):
        print(f"  Warning: required fields not found for profile extraction")
        return {}

    x_all      = read_cell_centres(case_dir, component=0)
    y_all      = read_cell_centres(case_dir, component=1)
    Ux_all     = read_of_vector_field(U_file, component=0)
    k_all      = read_of_field(k_file)
    nut_all    = read_of_field(nut_file) if os.path.exists(nut_file) else None
    sigmak_all = read_of_field(sk_file)  if os.path.exists(sk_file)  else None
    ck_all     = read_of_field(ck_file)  if os.path.exists(ck_file)  else None
    comega_all = read_of_field(co2_file) if os.path.exists(co2_file) else None

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
    dx       = x_total / 150  # approximate cell width

    x_targets = find_x_for_re_theta(x_all, y_all, Ux_all, re_theta_targets)

    for target_re, x_target in zip(re_theta_targets, x_targets):
        col_mask = column_mask(x_all, x_target, dx)

        if not np.any(col_mask):
            continue

        y_col     = y_all[col_mask]
        Ux_col    = Ux_all[col_mask]
        k_col     = k_all[col_mask]
        nut_col   = nut_all[col_mask]    if nut_all    is not None else None
        sigmak_col = sigmak_all[col_mask] if sigmak_all is not None else None
        ck_col     = ck_all[col_mask]     if ck_all     is not None else None
        comega_col = comega_all[col_mask] if comega_all is not None else None

        # Sort by y
        order  = np.argsort(y_col)
        y_col  = y_col[order]
        Ux_col = Ux_col[order]
        k_col  = k_col[order]
        if nut_col is not None:
            nut_col = nut_col[order]
        if sigmak_col is not None:
            sigmak_col = sigmak_col[order]
        if ck_col is not None:
            ck_col = ck_col[order]
        if comega_col is not None:
            comega_col = comega_col[order]

        # Get local wall shear stress (nearest x_wall)
        idx_wall = np.argmin(np.abs(x_wall_all - x_target))
        tau_w    = tau_wall_all[idx_wall]
        u_tau    = np.sqrt(tau_w / RHO)

        if u_tau < 1e-10:
            continue

        yplus  = y_col * u_tau / NU
        Uplus  = Ux_col / u_tau
        kplus  = k_col / u_tau**2

        # Modelled Reynolds shear stress (Boussinesq): -u'v' = nut*dU/dy
        uvplus = None
        if nut_col is not None:
            dUdy   = np.gradient(Ux_col, y_col)
            uvplus = -nut_col * dUdy / u_tau**2

        x_actual = x_all[col_mask].mean()
        Ue       = Ux_col[-1]  # local edge velocity -- see compute_re_theta
        theta    = np.trapz((Ux_col / Ue) * (1.0 - Ux_col / Ue), y_col)
        Re_theta = Ue * theta / NU

        # Boundary-layer thickness delta_99: y where U first reaches 99% of
        # the local edge velocity -- needed to plot vs y/delta like
        # Davidson's Fig. 12(e), rather than vs a raw dimensional y.
        delta99   = np.interp(0.99, Ux_col / Ue, y_col)
        y_over_delta = y_col / delta99

        profiles[target_re] = {
            'yplus': yplus, 'Uplus': Uplus, 'kplus': kplus, 'uvplus': uvplus,
            'x': x_actual, 'Re_theta': Re_theta, 'u_tau': u_tau,
            'delta99': delta99, 'y_over_delta': y_over_delta,
            'sigmak': sigmak_col, 'ck': ck_col, 'comega2': comega_col,
        }

    return profiles


# ----------------------------------------------------------------
# Read case
# ----------------------------------------------------------------
def read_case(case_dir):
    print(f"\nReading case: {case_dir}")
    label          = os.path.basename(os.path.abspath(case_dir))
    x_wall, Re_theta, Cf = compute_cf(case_dir)
    profiles       = extract_profiles(case_dir, PROFILE_RE_THETA)

    return dict(
        label=label,
        x_wall=x_wall, Re_theta=Re_theta, Cf=Cf,
        profiles=profiles,
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

    # Reference correlation + +-6% band, matching Fig. 12(a) exactly
    Re_ref = np.linspace(3000, 5000, 200)
    Cf_ref = cf_davidson(Re_ref)
    ax.plot(Re_ref, Cf_ref*1.06, 'k--', linewidth=1, label=r'$\pm 6\%$')
    ax.plot(Re_ref, Cf_ref*0.94, 'k--', linewidth=1)
    ax.plot(Re_ref, Cf_ref, 'b.', markersize=3,
            label=r'$C_f = 2(\frac{1}{0.384}\ln(Re_\theta) + 4.127)^{-2}$')

    # OpenFOAM results
    for i, c in enumerate(cases):
        if c['Re_theta'] is not None:
            style = STYLES[i % len(STYLES)]
            ax.plot(c['Re_theta'], c['Cf'],
                    label=c['label'], **style)

    ax.set_xlabel(r'$Re_\theta$')
    ax.set_ylabel(r'$C_f$')
    ax.set_title('Skin friction coefficient — flat-plate boundary layer')
    ax.legend(fontsize=9)
    ax.set_xlim([3000, 5000])
    ax.set_ylim([2.8e-3, 3.6e-3])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Cf_vs_Retheta.png', dpi=150)
    print("\nSaved: Cf_vs_Retheta.png")

    # ----------------------------------------------------------------
    # Plot 2: U+ profiles at selected x locations
    # ----------------------------------------------------------------
    cases_with_profiles = [c for c in cases if c['profiles']]
    if cases_with_profiles:
        n_locs = len(PROFILE_RE_THETA)
        yplus_dns, Uplus_dns, _, _ = read_dns_profile()

        fig, axes = plt.subplots(1, n_locs, figsize=(4*n_locs, 6),
                                 sharey=True)
        if n_locs == 1:
            axes = [axes]

        for ax, target_re in zip(axes, PROFILE_RE_THETA):
            if yplus_dns is not None:
                ax.semilogx(yplus_dns, Uplus_dns, 'k--', linewidth=1.5,
                            label=rf'DNS ($Re_\theta={DNS_RE_THETA}$)')

            for i, c in enumerate(cases_with_profiles):
                if target_re in c['profiles']:
                    p = c['profiles'][target_re]
                    style = STYLES[i % len(STYLES)]
                    ax.semilogx(p['yplus'], p['Uplus'],
                                label=c['label'], **style)
                    ax.set_title(rf'$Re_\theta={p["Re_theta"]:.0f}$',
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
        yplus_dns, _, kplus_dns, _ = read_dns_profile()

        fig, axes = plt.subplots(1, n_locs, figsize=(4*n_locs, 6),
                                 sharey=True)
        if n_locs == 1:
            axes = [axes]

        for ax, target_re in zip(axes, PROFILE_RE_THETA):
            if yplus_dns is not None:
                ax.plot(kplus_dns, yplus_dns, 'k--', linewidth=1.5,
                        label=rf'DNS ($Re_\theta={DNS_RE_THETA}$)')
            for i, c in enumerate(cases_with_profiles):
                if target_re in c['profiles']:
                    p = c['profiles'][target_re]
                    style = STYLES[i % len(STYLES)]
                    # Axes swapped to match Davidson (2026) Fig. 12(c):
                    # k+ on x, y+ (linear) on y.
                    ax.plot(p['kplus'], p['yplus'],
                            label=c['label'], **style)
                    ax.set_title(rf'$Re_\theta={p["Re_theta"]:.0f}$',
                                 fontsize=12)
            ax.set_xlabel(r'$k^+$')
            ax.set_ylim([0, 1600])
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r'$y^+$')
        axes[0].legend(fontsize=10)
        plt.suptitle(r'$k^+$ profiles — flat-plate boundary layer')
        plt.tight_layout()
        plt.savefig('kplus_profiles.png', dpi=150)
        print("Saved: kplus_profiles.png")

    # ----------------------------------------------------------------
    # Plot 4: turbulent shear stress profiles at selected x locations
    # (matches Davidson (2026) Fig. 12(d): u'v'+ on x, y+ (linear) on y)
    # ----------------------------------------------------------------
    if cases_with_profiles:
        yplus_dns, _, _, uvplus_dns = read_dns_profile()

        fig, axes = plt.subplots(1, n_locs, figsize=(4*n_locs, 6),
                                 sharey=True)
        if n_locs == 1:
            axes = [axes]

        for ax, target_re in zip(axes, PROFILE_RE_THETA):
            if yplus_dns is not None:
                ax.plot(uvplus_dns, yplus_dns, 'k--', linewidth=1.5,
                        label=rf'DNS ($Re_\theta={DNS_RE_THETA}$)')
            for i, c in enumerate(cases_with_profiles):
                if target_re in c['profiles']:
                    p = c['profiles'][target_re]
                    if p['uvplus'] is None:
                        continue
                    style = STYLES[i % len(STYLES)]
                    ax.plot(p['uvplus'], p['yplus'],
                            label=c['label'], **style)
                    ax.set_title(rf'$Re_\theta={p["Re_theta"]:.0f}$',
                                 fontsize=12)
            ax.set_xlabel(r"$\overline{u'v'}^+$")
            ax.set_xlim([-1.1, 0.1])
            ax.set_ylim([0, 1600])
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r'$y^+$')
        axes[0].legend(fontsize=10)
        plt.suptitle(r"Turbulent shear stress — flat-plate boundary layer")
        plt.tight_layout()
        plt.savefig('uv_profiles.png', dpi=150)
        print("Saved: uv_profiles.png")

    # ----------------------------------------------------------------
    # Plot 5: NN coefficient fields at selected Re_theta stations
    #
    # Matches Davidson (2026) Fig. 12(e): sigma_k,NN and C_k,NN on the left
    # axis (both O(1)) vs y/delta_99, C_omega2,NN on its own right axis
    # (standard value 0.072, an order of magnitude smaller) -- same
    # dual-axis treatment as the channelFlow5200 NN_coefficients plot.
    # Previously this plotted the y-averaged coefficients vs streamwise
    # Re_x, i.e. it collapsed the wall-normal structure entirely instead
    # of showing it, which is the opposite of what Fig. 12(e) shows.
    # ----------------------------------------------------------------
    nn_cases = [c for c in cases_with_profiles
                if any(p['sigmak'] is not None for p in c['profiles'].values())]
    if nn_cases:
        fig, axes = plt.subplots(1, n_locs, figsize=(4.5*n_locs, 6))
        if n_locs == 1:
            axes = [axes]

        for ax, target_re in zip(axes, PROFILE_RE_THETA):
            ax2 = ax.twinx()
            lines = []
            for c in nn_cases:
                if target_re not in c['profiles']:
                    continue
                p = c['profiles'][target_re]
                if p['sigmak'] is None:
                    continue
                l, = ax.plot(p['y_over_delta'], p['sigmak'], color='tab:blue',
                             lw=2, label=r'$\sigma_{k,NN}$')
                ax.axhline(2.0, color='tab:blue', ls=':', lw=1)
                lines.append(l)
                if p['ck'] is not None:
                    l, = ax.plot(p['y_over_delta'], p['ck'], color='tab:red',
                                 lw=2, label=r'$C_{k,NN}$')
                    ax.axhline(1.0, color='tab:red', ls=':', lw=1)
                    lines.append(l)
                if p['comega2'] is not None:
                    l, = ax2.plot(p['y_over_delta'], p['comega2'],
                                  color='tab:green', lw=2,
                                  label=r'$C_{\omega2,NN}$')
                    ax2.axhline(0.072, color='tab:green', ls=':', lw=1)
                    lines.append(l)
                ax.set_title(rf'$Re_\theta={p["Re_theta"]:.0f}$', fontsize=12)

            ax.set_xlabel(r'$y/\delta_{99}$')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 2.0])
            ax2.set_ylim([0, 0.09])
            ax.grid(True, alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel(r'$\sigma_{k,NN}$, $C_{k,NN}$')
                ax.legend(lines, [l.get_label() for l in lines],
                          fontsize=9, loc='upper left')
            if ax is axes[-1]:
                ax2.set_ylabel(r'$C_{\omega2,NN}$')

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
    Re_ref2 = np.array([3000, 5000])
    Cf_ref2 = cf_davidson(Re_ref2)
    print(f"{'Davidson correlation':<40} {Cf_ref2[0]:>10.6f} "
          f"{Cf_ref2[1]:>10.6f} {'5000':>14}")


if __name__ == '__main__':
    main()
