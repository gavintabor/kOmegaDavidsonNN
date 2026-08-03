#!/usr/bin/env python3
"""
plotSigmaK_pySR_vs_NN.py

Recreate Davidson (2026) Figure 19: sigma_k predicted by the pySR symbolic
expression (Eq. 22) compared against sigma_k predicted by the neural network.

Davidson's Figure 19 compares sigma_k,pySR against sigma_k,PINN (the PINN
target the NN was trained to reproduce). This OpenFOAM implementation does
not carry a PINN field, so this script compares against sigma_k,NN instead
(the 'sigmakNN' field actually written by the solver) -- the two are trained
to be close, so the comparison is equivalent in spirit.

Method: run the case ONCE with the library default (usePySR false, i.e. the
real NN is active). Read the converged y, U, k, nut fields plus the
'sigmakNN' field the solver wrote, then independently evaluate the pySR
expression (Eq. 22) in Python on the same x0(y), x1(y) input features the
solver computes internally (see computeNNCoefficients() in
src/kOmegaDavidsonNN.C) -- reproducing both curves on one profile, exactly
as Davidson's Figure 19 does.

Usage (from cases/channelFlow5200/):
    python3 ../../postProcessing/plotSigmaK_pySR_vs_NN.py [caseDir]

    caseDir : sub-case directory containing OpenFOAM results
              (default: 'kOmegaDavidsonNN'). Must have been run with
              usePySR false so that 'sigmakNN' holds the actual NN output.
              The latest numeric time directory is used.

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
DELTA = 1.0

# Clipping bounds from src/kOmegaDavidsonNN.C (training data ranges)
VOY_MIN, VOY_MAX = 1.883e-04, 3.679e-01
UV_MIN,  UV_MAX  = 9.149e-02, 9.950e-01
SIGMAK_MIN, SIGMAK_MAX = 2.088e-03, 1.847e+00


# ----------------------------------------------------------------
# Helpers (same field-reading approach as plotChannelFlow.py)
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


SMALL = 1e-15  # matches OpenFOAM's SMALL, used as a divide-by-zero guard below


def sigma_pySR(x0, x1):
    """Davidson (2026) Eq. 22, evaluated on (already clipped) x0, x1.

    x0*x1**2 - 0.362 comes within ~1e-3 of zero for x0, x1 near the top of
    their clipped ranges, so this needs the same max(..., SMALL) guard the
    C++ implementation uses (kOmegaDavidsonNN.C, computeNNCoefficients());
    without it, np.abs(x0*x1**2 - 0.362) close to zero produces spurious
    sharp spikes/dips near the pole that aren't in the actual model.
    """
    denom_inner = np.maximum(x0*x1**2 - 0.362, SMALL)
    return (
        0.469*x0
        + (0.574 + 1.0/(49.3 + 1.0/denom_inner))
        / (x0 + 0.246 + 0.0516*x1/np.maximum(x0, SMALL))
    )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    case_dir = sys.argv[1] if len(sys.argv) > 1 else 'kOmegaDavidsonNN'

    time_dir = find_latest_time(case_dir)
    print(f"Reading {case_dir} from {os.path.basename(time_dir)}/")

    y   = read_cell_centres_y(case_dir)
    Ux  = read_of_vector_component(os.path.join(time_dir, 'U'), 0)
    k   = read_of_scalar(os.path.join(time_dir, 'k'))
    nut = read_of_scalar(os.path.join(time_dir, 'nut'))

    sk_file = os.path.join(time_dir, 'sigmakNN')
    if not os.path.exists(sk_file):
        raise FileNotFoundError(
            f"sigmakNN not found in {time_dir} -- is this a kOmegaDavidsonNN case?")
    sigmak_NN = read_of_scalar(sk_file)

    # ------------------------------------------------------------
    # Reconstruct the NN/pySR input features exactly as
    # computeNNCoefficients() does in src/kOmegaDavidsonNN.C.
    # Fully-developed channel flow: sqrt(2)*|symm(grad U)| = |dU/dy|.
    #
    # u_tau is a single domain-wide value computed once from the
    # wall-adjacent cell (viscous-sublayer formula) and held constant,
    # matching the C++ implementation -- NOT a local Cmu^0.25*sqrt(k)
    # estimate (see README "Known limitations" for why that differs).
    # ------------------------------------------------------------
    dUdy   = np.gradient(Ux, y)
    uTau   = np.sqrt(NU * abs(Ux[0]) / y[0])
    tauTot = (NU + nut) * np.abs(dUdy)

    voy = nut / np.maximum(y*uTau, 1e-30)
    uv  = tauTot / np.maximum(uTau**2, 1e-30)

    x0 = np.clip(voy, VOY_MIN, VOY_MAX)
    x1 = np.clip(uv,  UV_MIN,  UV_MAX)

    sigmak_pySR = np.clip(sigma_pySR(x0, x1), SIGMAK_MIN, SIGMAK_MAX)

    diff = sigmak_NN - sigmak_pySR
    print(f"  max |NN - pySR| = {np.max(np.abs(diff)):.4f}")
    print(f"  RMS |NN - pySR| = {np.sqrt(np.mean(diff**2)):.4f}")

    # ------------------------------------------------------------
    # Plot (mirrors Davidson Figure 19: sigma_k vs y, y in [0, 1])
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(y, sigmak_pySR, 'b-',  lw=2, label='pySR (Eq. 22)')
    ax.plot(y, sigmak_NN,   'r--', lw=2, label='NN (sigmakNN)')
    ax.set_xlabel(r'$y/\delta$')
    ax.set_ylabel(r'$\sigma_k$')
    ax.set_title(r'$\sigma_k$: pySR vs. NN, $Re_\tau = 5200$')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sigmak_pySR_vs_NN.png', dpi=150)
    print("Saved: sigmak_pySR_vs_NN.png")


if __name__ == '__main__':
    main()
