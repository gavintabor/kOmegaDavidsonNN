#!/usr/bin/env python3
"""
plotChannelFlow.py

Compare OpenFOAM channel flow results against Lee & Moser (2015) DNS data
at Re_tau = 5200.

Usage (from cases/channelFlow5200/):
    python3 ../../postProcessing/plotChannelFlow.py [caseDir1 caseDir2 ...]

    caseDirN : subdirectory containing OpenFOAM results (default: '.')
               The latest numeric time directory is used from each case.

DNS data is read from the DNS_data/ subdirectory next to this script.

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
NU    = 1.0 / 5200.0
# u_tau = 1.0 exactly by construction: the case is driven by a fixed body
# force of 1.0 N/m^3 in x over a half-channel of height DELTA=1 m, so
# momentum balance gives tau_w = f*DELTA = 1.0 Pa, u_tau = sqrt(tau_w) = 1.0.
U_TAU = 1.0
DELTA = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DNS_DIR    = os.path.join(SCRIPT_DIR, 'DNS_data')
DNS_MEAN   = os.path.join(DNS_DIR, 'LM_Channel_5200_mean_prof.dat')
DNS_FLUCT  = os.path.join(DNS_DIR, 'LM_Channel_5200_vel_fluc_prof.dat')

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def find_latest_time(case_dir):
    dirs = []
    for d in os.listdir(case_dir):
        try:
            dirs.append(float(d))
        except ValueError:
            pass
    if not dirs:
        raise RuntimeError(f"No numeric time directories found in {case_dir}")
    latest = max(dirs)
    name = str(int(latest)) if latest == int(latest) else str(latest)
    return os.path.join(case_dir, name)


def read_of_scalar(filepath):
    with open(filepath) as f:
        text = f.read()
    if 'nonuniform' in text.lower():
        idx = text.lower().find('nonuniform')
        block_start = text.find('(', idx)
        block_end   = text.find(')', block_start)
        return np.array([float(x) for x in text[block_start+1:block_end].split()])
    elif 'uniform' in text.lower():
        idx = text.lower().find('uniform')
        val = float(text[idx:].split()[1].rstrip(';'))
        return np.array([val])
    return np.array([])


def read_of_vector_component(filepath, component=0):
    with open(filepath) as f:
        text = f.read()
    if 'nonuniform' not in text.lower():
        return np.array([])
    idx = text.lower().find('nonuniform')
    block_start = text.find('(', idx)
    depth, i = 0, block_start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                block_end = i
                break
        i += 1
    block = text[block_start+1:block_end]
    tuples = re.findall(r'\(([^)]+)\)', block)
    return np.array([float(t.split()[component]) for t in tuples])


def read_cell_centres_y(case_dir):
    cc_file = os.path.join(case_dir, '0', 'C')
    if not os.path.exists(cc_file):
        os.system(f'cd {case_dir} && postProcess -func writeCellCentres -time 0 > /dev/null 2>&1')
    if os.path.exists(cc_file):
        return read_of_vector_component(cc_file, component=1)
    # fallback: reconstruct from blockMesh grading (70 cells, r=1.1, y=[0,1])
    print(f"  Warning: reconstructing cell centres from mesh parameters for {case_dir}")
    n, r = 70, 1.1
    h0 = (r - 1.0) / (r**n - 1.0)
    heights = h0 * r**np.arange(n)
    y_faces = np.concatenate([[0.0], np.cumsum(heights)])
    return 0.5*(y_faces[:-1] + y_faces[1:])


def y_weighted_mean(y, field, y_max=DELTA):
    """Cell-height-weighted mean over a graded 1-D mesh (y=0 wall to y=y_max)."""
    faces = np.concatenate([[0.0], 0.5*(y[:-1] + y[1:]), [y_max]])
    weights = np.diff(faces)
    return np.sum(field*weights) / y_max


def read_dns():
    if not os.path.exists(DNS_MEAN):
        raise FileNotFoundError(f"DNS mean profile not found: {DNS_MEAN}")
    if not os.path.exists(DNS_FLUCT):
        raise FileNotFoundError(f"DNS fluctuation profile not found: {DNS_FLUCT}")
    dns_mean  = np.loadtxt(DNS_MEAN,  comments='%')
    dns_fluct = np.loadtxt(DNS_FLUCT, comments='%')
    # mean: y/delta, y+, U+, ...
    # fluct: y/delta, y+, uu, vv, ww, uv, uw, vw, k
    return dns_mean[:, 1], dns_mean[:, 2], dns_fluct[:, 8]


def read_case(case_dir):
    time_dir = find_latest_time(case_dir)
    print(f"  Reading {case_dir} from {os.path.basename(time_dir)}/")
    y = read_cell_centres_y(case_dir)
    U_file = os.path.join(time_dir, 'U')
    if not os.path.exists(U_file):
        raise FileNotFoundError(f"U not found: {U_file}")
    Ux = read_of_vector_component(U_file, 0)
    k_file = os.path.join(time_dir, 'k')
    if not os.path.exists(k_file):
        raise FileNotFoundError(f"k not found: {k_file}")
    k = read_of_scalar(k_file)

    nn_fields = {}
    for name in ('sigmakNN', 'CkNN', 'Comega2NN'):
        path = os.path.join(time_dir, name)
        if os.path.exists(path):
            nn_fields[name] = read_of_scalar(path)

    return y, Ux, k, nn_fields


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['.']

    print("Reading DNS data...")
    yplus_dns, Uplus_dns, kplus_dns = read_dns()

    results = []
    for cd in case_dirs:
        print(f"Reading case: {cd}")
        y, Ux, k, nn = read_case(cd)
        results.append((os.path.basename(os.path.normpath(cd)), y, Ux, k, nn))

    # ----------------------------------------------------------------
    # Plot 1: U+ vs y+
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(yplus_dns, Uplus_dns, 'k--', lw=2, label='DNS (Lee & Moser 2015)')
    yp_ref = np.array([30, 5200])
    ax.semilogx(yp_ref, np.log(yp_ref)/0.41 + 5.2, 'g:', lw=1, label='Log law')
    for i, (label, y, Ux, k, _) in enumerate(results):
        yplus = y * U_TAU / NU
        ax.semilogx(yplus, Ux/U_TAU, color=COLORS[i % len(COLORS)], lw=2, label=label)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$U^+$')
    ax.set_title(r'Mean velocity profile, $Re_\tau = 5200$')
    ax.legend()
    ax.set_xlim([0.1, 6000])
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Uplus_profile.png', dpi=150)
    print("Saved: Uplus_profile.png")

    # ----------------------------------------------------------------
    # Plot 2: k+ vs y+
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(yplus_dns, kplus_dns, 'k--', lw=2, label='DNS (Lee & Moser 2015)')
    for i, (label, y, Ux, k, _) in enumerate(results):
        yplus = y * U_TAU / NU
        ax.plot(yplus, k/U_TAU**2, color=COLORS[i % len(COLORS)], lw=2, label=label)
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
    # Plot 3: k+ near wall
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(yplus_dns, kplus_dns, 'k--', lw=2, label='DNS (Lee & Moser 2015)')
    for i, (label, y, Ux, k, _) in enumerate(results):
        yplus = y * U_TAU / NU
        ax.plot(yplus, k/U_TAU**2, color=COLORS[i % len(COLORS)], lw=2, label=label)
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
    # Plot 4: NN coefficient fields (first kOmegaDavidsonNN case only)
    #
    # Matches Davidson (2026) Fig. 8(d): sigma_k,NN and C_k,NN share the
    # left axis (both O(1)), C_omega2,NN gets its own right axis since its
    # standard value (0.072) is ~an order of magnitude smaller.
    # ----------------------------------------------------------------
    for label, y, Ux, k, nn in results:
        if nn:
            fig, ax1 = plt.subplots(figsize=(8, 6))
            ax2 = ax1.twinx()

            lines = []
            if 'sigmakNN' in nn:
                l, = ax1.plot(y, nn['sigmakNN'], color='tab:blue', lw=2,
                               label=r'$\sigma_{k,NN}$')
                ax1.axhline(2.0, color='tab:blue', ls=':', lw=1)
                lines.append(l)
            if 'CkNN' in nn:
                l, = ax1.plot(y, nn['CkNN'], color='tab:red', lw=2,
                               label=r'$C_{k,NN}$')
                ax1.axhline(1.0, color='tab:red', ls=':', lw=1)
                lines.append(l)
            if 'Comega2NN' in nn:
                l, = ax2.plot(y, nn['Comega2NN'], color='tab:green', lw=2,
                               label=r'$C_{\omega2,NN}$')
                ax2.axhline(0.072, color='tab:green', ls=':', lw=1)
                lines.append(l)

            ax1.set_xlabel(r'$y$')
            ax1.set_ylabel(r'$\sigma_{k,NN}$, $C_{k,NN}$')
            ax2.set_ylabel(r'$C_{\omega2,NN}$')
            ax1.set_title(rf'NN coefficient fields — {label}, $Re_\tau = 5200$')
            ax1.grid(True, alpha=0.3)
            ax1.legend(lines, [l.get_label() for l in lines], loc='best')

            plt.tight_layout()
            plt.savefig(f'NN_coefficients_{label}.png', dpi=150)
            print(f"Saved: NN_coefficients_{label}.png")
            break  # only first NN case

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\nSummary (U_tau = {U_TAU}):")
    print(f"  {'Case':<25} {'y+_min':>8} {'y+_max':>8} {'U_bulk':>8} {'k+_max':>8}")
    for label, y, Ux, k, _ in results:
        yplus = y * U_TAU / NU
        print(f"  {label:<25} {yplus.min():>8.3f} {yplus.max():>8.1f} {y_weighted_mean(y, Ux):>8.4f} {(k/U_TAU**2).max():>8.4f}")
    print(f"  {'DNS':.<25} {'':>8} {'':>8} {'':>8} {kplus_dns.max():>8.4f}")


if __name__ == '__main__':
    main()
