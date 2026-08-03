#!/usr/bin/env python3
"""
plotChannelFlow.py

Compare OpenFOAM channel flow results against Lee & Moser (2015) DNS data
at Re_tau = 5200. Supports multiple cases on the same plot.

Usage:
    python3 plotChannelFlow.py case1_dir [case2_dir] [case3_dir] ...

    Each case_dir should be an OpenFOAM case directory containing time
    directories and optionally the DNS data files.

    The script reads the latest time directory from each case.
    u_tau is estimated from the pressure gradient written by meanVelocityForce
    (reads from the solver log if present, otherwise falls back to 1.0).

Examples:
    python3 plotChannelFlow.py .
    python3 plotChannelFlow.py ../channelFlow5200_NN ../channelFlow5200_kOmega

DNS data files (place in working directory or first case directory):
    LM_Channel_5200_mean_prof.dat
    LM_Channel_5200_vel_fluc_prof.dat

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
NU    = 1.0 / 5200.0   # kinematic viscosity
DELTA = 1.0             # half-channel width

DNS_MEAN  = 'LM_Channel_5200_mean_prof.dat'
DNS_FLUCT = 'LM_Channel_5200_vel_fluc_prof.dat'

# Plot styles for each case (cycles if more cases than entries)
STYLES = [
    {'color': 'b',      'linestyle': '-',  'linewidth': 2},
    {'color': 'r',      'linestyle': '-',  'linewidth': 2},
    {'color': 'g',      'linestyle': '-',  'linewidth': 2},
    {'color': 'orange', 'linestyle': '--', 'linewidth': 2},
    {'color': 'purple', 'linestyle': '--', 'linewidth': 2},
]


# ----------------------------------------------------------------
# Utility: find latest time directory in a case
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
# Utility: estimate u_tau from solver log
# ----------------------------------------------------------------
def estimate_u_tau(case_dir):
    """
    Try to read the pressure gradient from the meanVelocityForce output
    in the solver log. Falls back to 1.0 if not found.
    """
    for logname in ['log.simpleFoam', 'log', 'log.txt']:
        logpath = os.path.join(case_dir, logname)
        if os.path.exists(logpath):
            with open(logpath) as f:
                text = f.read()
            matches = re.findall(
                r'[Pp]ressure\s+[Gg]radient\s*=\s*([\d.eE+\-]+)', text)
            if matches:
                grad  = float(matches[-1])
                u_tau = np.sqrt(abs(grad) * DELTA)
                print(f"  [{case_dir}] u_tau from log: sqrt({grad:.4f}) = {u_tau:.4f}")
                return u_tau
    print(f"  [{case_dir}] u_tau not found in log, using 1.0")
    return 1.0


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
# Read cell-centre y coordinates
# ----------------------------------------------------------------
def read_cell_centres_y(case_dir):
    cc_file = os.path.join(case_dir, '0', 'C')
    if not os.path.exists(cc_file):
        os.system(
            f'cd {case_dir} && postProcess -func writeCellCentres '
            f'-time 0 > /dev/null 2>&1')
    if os.path.exists(cc_file):
        return read_of_vector_field(cc_file, component=1)

    # Fallback: reconstruct from blockMesh parameters
    # 70 cells, y=0 to 1, grading factor 1.1 per cell
    print(f"  Warning: reconstructing cell centres from mesh parameters")
    n   = 70
    r   = 1.1
    h0  = (r - 1.0) / (r**n - 1.0)
    heights = h0 * r**np.arange(n)
    y_faces = np.concatenate([[0.0], np.cumsum(heights)])
    return 0.5*(y_faces[:-1] + y_faces[1:])


# ----------------------------------------------------------------
# Read DNS data (searches working dir then each case dir)
# ----------------------------------------------------------------
def read_dns(search_dirs):
    for d in search_dirs:
        mean_file  = os.path.join(d, DNS_MEAN)
        fluct_file = os.path.join(d, DNS_FLUCT)
        if os.path.exists(mean_file) and os.path.exists(fluct_file):
            dns_mean  = np.loadtxt(mean_file,  comments='%')
            dns_fluct = np.loadtxt(fluct_file, comments='%')
            yplus_dns = dns_mean[:, 1]
            Uplus_dns = dns_mean[:, 2]
            kplus_dns = dns_fluct[:, 8]
            print(f"  DNS data read from: {d}")
            return yplus_dns, Uplus_dns, kplus_dns
    raise FileNotFoundError(
        f"DNS data files not found in any of: {search_dirs}\n"
        f"Please place {DNS_MEAN} and {DNS_FLUCT} in the working directory "
        f"or one of the case directories.")


# ----------------------------------------------------------------
# Read one OpenFOAM case
# ----------------------------------------------------------------
def read_case(case_dir):
    print(f"\nReading case: {case_dir}")
    u_tau    = estimate_u_tau(case_dir)
    latest   = find_latest_time(case_dir)
    time_dir = os.path.join(case_dir, latest)
    print(f"  Time directory: {latest}")

    y_of     = read_cell_centres_y(case_dir)
    yplus_of = y_of * u_tau / NU

    U_file   = os.path.join(time_dir, 'U')
    Ux_of    = read_of_vector_field(U_file, component=0)
    Uplus_of = Ux_of / u_tau

    k_file   = os.path.join(time_dir, 'k')
    k_of     = read_of_field(k_file)
    kplus_of = k_of / u_tau**2

    nut_file = os.path.join(time_dir, 'nut')
    nut_of   = read_of_field(nut_file) if os.path.exists(nut_file) else None

    sk_file    = os.path.join(time_dir, 'sigmakNN')
    ck_file    = os.path.join(time_dir, 'CkNN')
    co2_file   = os.path.join(time_dir, 'Comega2NN')
    sigmak_of  = read_of_field(sk_file)  if os.path.exists(sk_file)  else None
    ck_of      = read_of_field(ck_file)  if os.path.exists(ck_file)  else None
    comega2_of = read_of_field(co2_file) if os.path.exists(co2_file) else None

    label = os.path.basename(os.path.abspath(case_dir))

    faces  = np.concatenate([[0.0], 0.5*(y_of[:-1]+y_of[1:]), [DELTA]])
    cell_h = np.diff(faces)
    U_bulk = np.dot(Ux_of, cell_h) / DELTA

    print(f"  u_tau={u_tau:.4f}, Re_tau={u_tau*DELTA/NU:.0f}")
    print(f"  k_max+ = {kplus_of.max():.4f},  k_centre+ = {kplus_of[-1]:.4f}")
    print(f"  U_bulk = {U_bulk:.4f}  (volume-weighted)")

    return dict(
        y_of=y_of, yplus_of=yplus_of, Uplus_of=Uplus_of,
        kplus_of=kplus_of, nut_of=nut_of, Ux_of=Ux_of, k_of=k_of,
        sigmak_of=sigmak_of, ck_of=ck_of, comega2_of=comega2_of,
        u_tau=u_tau, label=label, U_bulk=U_bulk,
    )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['.']

    print("\nReading DNS data...")
    yplus_dns, Uplus_dns, kplus_dns = read_dns(['.'] + case_dirs)

    cases = [read_case(d) for d in case_dirs]

    # ----------------------------------------------------------------
    # Plot 1: U+ vs y+ (log scale)
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(yplus_dns, Uplus_dns, 'k--', linewidth=2,
                label='DNS (Lee & Moser 2015)')
    kappa, B = 0.41, 5.2
    yp_ref   = np.array([30, 6000])
    ax.semilogx(yp_ref, np.log(yp_ref)/kappa + B, 'g:',
                linewidth=1, label='Log law')
    for i, c in enumerate(cases):
        ax.semilogx(c['yplus_of'], c['Uplus_of'],
                    label=c['label'], **STYLES[i % len(STYLES)])
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$U^+$')
    ax.set_title(r'Mean velocity profile, $Re_\tau = 5200$')
    ax.legend()
    ax.set_xlim([0.1, 6000])
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Uplus_profile.png', dpi=150)
    print("\nSaved: Uplus_profile.png")

    # ----------------------------------------------------------------
    # Plot 2: k+ vs y+ (full channel)
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(yplus_dns, kplus_dns, 'k--', linewidth=2,
            label='DNS (Lee & Moser 2015)')
    for i, c in enumerate(cases):
        ax.plot(c['yplus_of'], c['kplus_of'],
                label=c['label'], **STYLES[i % len(STYLES)])
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$k^+$')
    ax.set_title(r'Turbulent kinetic energy, $Re_\tau = 5200$')
    ax.legend()
    ax.set_xlim([0, 5200])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kplus_profile.png', dpi=150)
    print("Saved: kplus_profile.png")

    # ----------------------------------------------------------------
    # Plot 3: k+ vs y+ (near-wall zoom)
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(yplus_dns, kplus_dns, 'k--', linewidth=2,
            label='DNS (Lee & Moser 2015)')
    for i, c in enumerate(cases):
        ax.plot(c['yplus_of'], c['kplus_of'],
                label=c['label'], **STYLES[i % len(STYLES)])
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$k^+$')
    ax.set_title(r'Turbulent kinetic energy (near wall), $Re_\tau = 5200$')
    ax.set_xlim([0, 200])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kplus_nearwall.png', dpi=150)
    print("Saved: kplus_nearwall.png")

    # ----------------------------------------------------------------
    # Plot 4: nut vs y
    # ----------------------------------------------------------------
    cases_with_nut = [c for c in cases if c['nut_of'] is not None]
    if cases_with_nut:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, c in enumerate(cases_with_nut):
            ax.plot(c['y_of'], c['nut_of'],
                    label=c['label'], **STYLES[i % len(STYLES)])
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$\nu_t$')
        ax.set_title(r'Turbulent viscosity, $Re_\tau = 5200$')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('nut_profile.png', dpi=150)
        print("Saved: nut_profile.png")

    # ----------------------------------------------------------------
    # Plot 5: NN coefficient fields (one plot per NN case)
    # ----------------------------------------------------------------
    for c in [c for c in cases if c['sigmak_of'] is not None]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].plot(c['y_of'], c['sigmak_of'], 'b-', linewidth=2)
        axes[0].axhline(2.0, color='k', linestyle='--', linewidth=1,
                        label='Standard kOmega')
        axes[0].set_xlabel(r'$y$')
        axes[0].set_ylabel(r'$\sigma_{k,NN}$')
        axes[0].set_title(r'$\sigma_{k,NN}$')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        if c['ck_of'] is not None:
            axes[1].plot(c['y_of'], c['ck_of'], 'r-', linewidth=2)
            axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1,
                            label='Standard kOmega')
            axes[1].set_xlabel(r'$y$')
            axes[1].set_ylabel(r'$C_{k,NN}$')
            axes[1].set_title(r'$C_{k,NN}$')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

        if c['comega2_of'] is not None:
            axes[2].plot(c['y_of'], c['comega2_of'], 'g-', linewidth=2)
            axes[2].axhline(0.072, color='k', linestyle='--', linewidth=1,
                            label='Standard kOmega')
            axes[2].set_xlabel(r'$y$')
            axes[2].set_ylabel(r'$C_{\omega2,NN}$')
            axes[2].set_title(r'$C_{\omega2,NN}$')
            axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(
            rf'NN coefficient fields — {c["label"]}, $Re_\tau = 5200$')
        plt.tight_layout()
        fname = f'NN_coefficients_{c["label"]}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'Case':<35} {'u_tau':>8} {'Re_tau':>8} "
          f"{'k_max+':>8} {'k_ctr+':>8} {'U_bulk':>8}")
    print('-' * 75)
    for c in cases:
        re_tau = c['u_tau'] * DELTA / NU
        print(f"{c['label']:<35} {c['u_tau']:>8.4f} {re_tau:>8.0f} "
              f"{c['kplus_of'].max():>8.4f} {c['kplus_of'][-1]:>8.4f} "
              f"{c['U_bulk']:>8.4f}")
    print(f"{'DNS (Lee & Moser 2015)':<35} {'—':>8} {'5186':>8} "
          f"{kplus_dns.max():>8.4f} {kplus_dns[-1]:>8.4f} {'—':>8}")

    plt.show()


if __name__ == '__main__':
    main()
