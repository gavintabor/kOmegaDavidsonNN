#!/usr/bin/env python3
"""
plotSigmaK_NN_vs_pySR_domain.py

Compare the sigma_k NN and the pySR expression (Davidson 2026, Eq. 22)
directly as functions of the (x0, x1) input pair, swept over the entire
trained domain -- independent of any CFD run. This removes every
CFD-specific confound (u_tau estimation, EWMA smoothing, which particular
trajectory through input space a given simulation happens to sample) and
answers a narrower question than plotSigmaK_pySR_vs_NN.py: how well does
Eq. 22 actually approximate the NN, everywhere it could be evaluated?

NN weights are parsed directly out of src/kOmegaDavidsonNN.C (not retyped),
so this can't itself introduce a weight-transcription error while checking
for one.

Usage:
    python3 plotSigmaK_NN_vs_pySR_domain.py [case_dir]

    case_dir : optional OpenFOAM case (e.g. cases/channelFlow5200/kOmegaDavidsonNN)
               whose (x0, x1) trajectory is overlaid on the domain heatmap for
               context. If omitted, only the domain-wide comparison is plotted.

Requires: numpy, matplotlib
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, '..', 'src', 'kOmegaDavidsonNN.C')

NU = 1.0 / 5200.0

VOY_MIN, VOY_MAX = 1.883e-04, 3.679e-01
UV_MIN,  UV_MAX  = 9.149e-02, 9.950e-01
SIGMAK_MIN, SIGMAK_MAX = 2.088e-03, 1.847e+00
SMALL = 1e-15  # matches OpenFOAM's SMALL, used as a divide-by-zero guard


# ----------------------------------------------------------------
# Parse NN weights straight out of the C++ source
# ----------------------------------------------------------------
def parse_array(text, name, shape):
    m = re.search(rf'static const scalar {re.escape(name)}\s*(\[[^=]*)?=\s*\{{(.*?)\}};',
                   text, re.S)
    if not m:
        raise RuntimeError(f"couldn't find weight array '{name}' in {SRC}")
    nums = [float(x) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?', m.group(2))]
    arr = np.array(nums)
    return arr.reshape(shape) if shape else arr


def parse_scalar(text, name):
    m = re.search(rf'static const scalar {re.escape(name)}\s*=\s*(-?\d+\.\d+(?:e-?\d+)?)\s*;', text)
    if not m:
        raise RuntimeError(f"couldn't find scalar '{name}' in {SRC}")
    return float(m.group(1))


def load_weights():
    text = open(SRC).read()
    return dict(
        W1=parse_array(text, 'sk_W1', (10, 2)),
        b1=parse_array(text, 'sk_b1', (10,)),
        W2=parse_array(text, 'sk_W2', (10, 10)),
        b2=parse_array(text, 'sk_b2', (10,)),
        W3=parse_array(text, 'sk_W3', (10,)),
        b3=parse_scalar(text, 'sk_b3'),
    )


def nn_sigma_k(voy, uv, w):
    """Forward pass, matching computeNNCoefficients(): clip (voy,uv) to the
    training range, normalise to [0,1], 2-10-10-1 ReLU network, clip output."""
    xv = np.clip(voy, VOY_MIN, VOY_MAX)
    xu = np.clip(uv,  UV_MIN,  UV_MAX)
    x0 = (xv - VOY_MIN) / (VOY_MAX - VOY_MIN)
    x1 = (xu - UV_MIN)  / (UV_MAX  - UV_MIN)

    h1 = np.maximum(np.einsum('...,i->...i', x0, w['W1'][:, 0])
                     + np.einsum('...,i->...i', x1, w['W1'][:, 1])
                     + w['b1'], 0.0)
    h2 = np.maximum(np.einsum('...i,ji->...j', h1, w['W2']) + w['b2'], 0.0)
    out = np.einsum('...i,i->...', h2, w['W3']) + w['b3']
    return np.clip(out, SIGMAK_MIN, SIGMAK_MAX)


def pysr_sigma_k(voy, uv):
    """Eq. 22, raw clipped (not normalised) inputs, with the SMALL guard on
    the near-singular denominator (see kOmegaDavidsonNN.C computeNNCoefficients())."""
    xv = np.clip(voy, VOY_MIN, VOY_MAX)
    xu = np.clip(uv,  UV_MIN,  UV_MAX)
    denom_inner = np.maximum(xv*xu*xu - 0.362, SMALL)
    out = (
        0.469*xv
        + (0.574 + 1.0/(49.3 + 1.0/denom_inner))
        / (xv + 0.246 + 0.0516*xu/np.maximum(xv, SMALL))
    )
    return np.clip(out, SIGMAK_MIN, SIGMAK_MAX)


# ----------------------------------------------------------------
# Optional: read a case's actual (voy, uv) trajectory for context
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
    text = open(filepath).read()
    if 'nonuniform' in text.lower():
        idx = text.lower().find('nonuniform')
        bs = text.find('(', idx)
        be = text.find(')', bs)
        return np.array([float(x) for x in text[bs+1:be].split()])
    if 'uniform' in text.lower():
        idx = text.lower().find('uniform')
        return np.array([float(text[idx:].split()[1].rstrip(';'))])
    return np.array([])


def read_of_vector_component(filepath, component=0):
    text = open(filepath).read()
    if 'nonuniform' not in text.lower():
        return np.array([])
    idx = text.lower().find('nonuniform')
    bs = text.find('(', idx)
    depth, i = 0, bs
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                be = i
                break
        i += 1
    tuples = re.findall(r'\(([^)]+)\)', text[bs+1:be])
    return np.array([float(t.split()[component]) for t in tuples])


def read_trajectory(case_dir):
    """(voy, uv) along a case's wall-normal profile, same reconstruction as
    plotSigmaK_pySR_vs_NN.py: single wall-derived u_tau, held constant."""
    time_dir = find_latest_time(case_dir)
    y = read_of_vector_component(os.path.join(case_dir, '0', 'C'), 1)
    Ux = read_of_vector_component(os.path.join(time_dir, 'U'), 0)
    nut = read_of_scalar(os.path.join(time_dir, 'nut'))
    dUdy = np.gradient(Ux, y)
    tauTot = (NU + nut) * np.abs(dUdy)
    uTau = np.sqrt(NU * abs(Ux[0]) / y[0])
    voy = nut / np.maximum(y * uTau, SMALL)
    uv = tauTot / np.maximum(uTau**2, SMALL)
    return voy, uv


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dir = sys.argv[1] if len(sys.argv) > 1 else None

    w = load_weights()
    print("Parsed sigma_k NN weights from", os.path.normpath(SRC))

    n = 300
    voy_grid = np.linspace(VOY_MIN, VOY_MAX, n)
    uv_grid  = np.linspace(UV_MIN,  UV_MAX,  n)
    VOY, UV = np.meshgrid(voy_grid, uv_grid)

    NN = nn_sigma_k(VOY, UV, w)
    PYSR = pysr_sigma_k(VOY, UV)
    DIFF = NN - PYSR

    print(f"\nOver the full (voy, uv) domain ({n}x{n} grid):")
    print(f"  max|NN-pySR| = {np.max(np.abs(DIFF)):.4f}")
    print(f"  RMS|NN-pySR| = {np.sqrt(np.mean(DIFF**2)):.4f}")
    iworst = np.unravel_index(np.argmax(np.abs(DIFF)), DIFF.shape)
    print(f"  worst point: voy={VOY[iworst]:.4f}, uv={UV[iworst]:.4f}, "
          f"NN={NN[iworst]:.4f}, pySR={PYSR[iworst]:.4f}")

    voy_traj = uv_traj = None
    if case_dir:
        print(f"\nReading trajectory from {case_dir}...")
        voy_traj, uv_traj = read_trajectory(case_dir)
        print(f"  voy range: [{voy_traj.min():.4f}, {voy_traj.max():.4f}]")
        print(f"  uv range:  [{uv_traj.min():.4f}, {uv_traj.max():.4f}]")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    im = ax.pcolormesh(voy_grid, uv_grid, np.abs(DIFF), shading='auto',
                        cmap='viridis', vmax=1.0)
    plt.colorbar(im, ax=ax, label=r'$|\sigma_{k,NN}-\sigma_{k,pySR}|$')
    if voy_traj is not None:
        label = os.path.basename(os.path.normpath(case_dir)) + ' trajectory'
        ax.plot(voy_traj, uv_traj, 'r.-', lw=1.5, ms=3, label=label)
        ax.legend(loc='upper right')
    ax.set_xlabel(r'$x_0$ (voy, raw)')
    ax.set_ylabel(r'$x_1$ (uv, raw)')
    ax.set_title('|NN - pySR| over full trained domain')

    ax = axes[1]
    for uv0, col in zip([0.15, 0.35, 0.6, 0.9],
                         ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        j = np.argmin(np.abs(uv_grid - uv0))
        ax.plot(voy_grid, NN[j, :], color=col, lw=2, label=f'NN, x1={uv_grid[j]:.2f}')
        ax.plot(voy_grid, PYSR[j, :], color=col, lw=2, ls='--')
    ax.set_xlabel(r'$x_0$ (voy)')
    ax.set_ylabel(r'$\sigma_k$')
    ax.set_title('NN (solid) vs pySR (dashed) at fixed $x_1$ slices')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outfile = 'sigmak_NN_vs_pySR_domain.png'
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
