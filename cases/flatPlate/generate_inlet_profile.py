#!/usr/bin/env python3
"""
generate_inlet_profile.py

One-time provenance script: (re)generates the inlet nonuniform-list
boundary values in kOmega/0/{U,k,omega} and kOmegaDavidsonNN/0/{U,k,omega}
from Davidson's actual precursor turbulent boundary-layer profile.

Why this exists: Davidson (2026), Sec. 5.2, states the flat-plate case's
inlet condition is NOT a uniform freestream growing from a sharp leading
edge (which is what this case previously had) -- it is a fully turbulent
boundary layer at Re_theta = 2550 taken from a precursor 2D RANS
simulation, on a domain 92*delta_in x 20*delta_in with a 150x90 grid. This
script reproduces that inlet condition using Davidson's own saved profile
and grid data (from literature/pythons-rans-code-RANS-open/), so that the
"y" case here actually matches the Reynolds-number range (Re_theta ~
2550-8000) needed to compare against his Fig. 12, instead of the
Re_theta_max=243 obtained from a leading-edge-developing boundary layer
over a 0.92 m plate.

Not part of Allrun: this only needs to be run once (or again if the mesh
resolution changes). It requires literature/ (Davidson's reference code,
not distributed with this repo), but its *output* -- the field files it
writes -- are committed and are all Allrun needs going forward.

Usage:
    python3 generate_inlet_profile.py
"""

import os
import re
import subprocess
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
REF_DIR = os.path.join(
    REPO_ROOT, 'literature', 'pythons-rans-code-RANS-open',
    'boundary-layer-k-omega-ni-150-nj-100-yfac-yplus-0.8-ymax-4.5-'
    'ML-NN-from-channel-NN-vist-over-y-uv_tot-2nd-submission')

PROFILE_FILE = os.path.join(
    REF_DIR, 'y_u_k_om-boundary-layer-RANS-kom-omega-create-inlet-data.txt')
Y2D_FILE = os.path.join(REF_DIR, 'y2d.dat')
X2D_FILE = os.path.join(REF_DIR, 'x2d.dat')

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_davidson_grid():
    """Davidson's actual (x,y) vertex grid: ni=150 (uniform), nj=90 (graded)."""
    datax = np.loadtxt(X2D_FILE)
    ni = int(datax[-1])
    datay = np.loadtxt(Y2D_FILE)
    nj = int(datay[-1])
    y2d = datay[:-1].reshape(ni + 1, nj + 1)
    x2d = datax[:-1].reshape(ni + 1, nj + 1)
    return ni, nj, x2d[:, 0], y2d[0, :]


def read_vec_component(filepath, component):
    text = open(filepath).read()
    idx = text.lower().find('nonuniform')
    bs = text.find('(', idx)
    depth = 0
    i = bs
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                be = i
                break
        i += 1
    block = text[bs+1:be]
    tuples = re.findall(r'\(([^)]+)\)', block)
    return np.array([float(t.split()[component]) for t in tuples])


def load_mesh_ycentres():
    """
    Our actual generated mesh's wall-normal cell-centre y-positions (one
    inlet-face column), read via blockMesh + writeCellCentres in the kOmega
    sub-case (both sub-cases share an identical blockMeshDict).

    This does NOT reproduce Davidson's y2d.dat cell centres exactly: his
    y2d.dat growth in the near-wall geometric region isn't a pure single-
    ratio geometric series (only the total length/ratio/cell-count over that
    segment match what we derive from it), so OpenFOAM's simpleGrading
    diverges from his actual point distribution by up to ~7.5% in y near the
    wall, even though both use the same fractions/ratio/cell-count. Since
    the profile is steep there, interpolating onto Davidson's positions and
    writing the result into our mesh by cell INDEX (as an earlier version of
    this script did) silently mismatches the assigned value to the actual
    cell location -- up to 46% wrong in k at the wall-adjacent cell. Reading
    our own mesh's real coordinates and interpolating onto those directly
    is correct regardless of any grading-algorithm mismatch.

    Deliberately does NOT assert the cell count against Davidson's own nj
    (from y2d.dat) -- blockMeshDict's own (150 x NJ x 1) cell count is the
    sole source of truth for the mesh we actually solve on. This lets
    blockMeshDict's y-resolution be refined independently of Davidson's
    reference grid (see the near-wall mesh-refinement experiment in
    [[project_flatplate_cf_investigation]]) without this script erroring out.
    """
    case_dir = os.path.join(CASE_DIR, 'kOmega')
    subprocess.run(['blockMesh', '-case', case_dir],
                    check=True, capture_output=True)
    subprocess.run(['postProcess', '-func', 'writeCellCentres',
                     '-case', case_dir, '-time', '0'],
                    check=True, capture_output=True)
    cc_file = os.path.join(case_dir, '0', 'C')
    x = read_vec_component(cc_file, 0)
    y = read_vec_component(cc_file, 1)
    mask = np.isclose(x, x.min(), atol=1e-6)
    yc = np.sort(y[mask])
    return yc


def main():
    ni, nj, xv, yv = load_davidson_grid()
    Lx, Ly = xv[-1], yv[-1]
    print(f"Grid: ni={ni}, nj={nj}, Lx={Lx:.6f}, Ly={Ly:.6f}")

    # Multi-grading in y: geometric growth up to where dy stops growing,
    # then uniform. Reproduced here (from the same y2d.dat) so blockMesh
    # regenerates (very nearly, see load_mesh_ycentres) this same grid.
    dy = np.diff(yv)
    ratios = dy[1:] / dy[:-1]
    stop = int(np.argmax(ratios < 1.0001)) + 1
    n1, n2 = stop + 1, nj - (stop + 1)
    L1, L2 = yv[stop + 1], Ly - yv[stop + 1]
    r1 = dy[stop] / dy[0]
    print(f"y-grading: segment 1: n={n1}, L={L1:.6f}, r={r1:.4f}  "
          f"| segment 2 (uniform): n={n2}, L={L2:.6f}")
    print(f"  -> blockMeshDict fractions: "
          f"({L1/Ly:.6f} {n1/nj:.6f} {r1:.4f}) ({L2/Ly:.6f} {n2/nj:.6f} 1)")

    # Interpolate onto OUR mesh's actual generated cell centres, not
    # Davidson's y2d.dat-derived ones -- see load_mesh_ycentres docstring.
    # nj here is only used above for the grading-fraction printout; the
    # mesh's real cell count (set independently in blockMeshDict) is
    # whatever load_mesh_ycentres actually finds.
    yc = load_mesh_ycentres()
    if len(yc) != nj:
        print(f"Note: mesh has {len(yc)} wall-normal cells, "
              f"not Davidson's {nj} -- using the mesh's own count "
              f"(see load_mesh_ycentres docstring).")

    # Inlet profile (Re_theta=2550), given at its own (finer, BL-only) grid.
    prof = np.loadtxt(PROFILE_FILE)
    yp, Up, kp, omp = prof[:, 0], prof[:, 1], prof[:, 2], prof[:, 3]
    Uinf, kinf, ominf = Up[-1], kp[-1], omp[-1]
    print(f"Profile: Re_theta check, Uinf={Uinf:.4f}, kinf={kinf:.3e}, "
          f"ominf={ominf:.4f}")

    # Interpolate onto our mesh's inlet face y-centres; hold flat at the
    # freestream value above the profile's max y (standard precursor-BL
    # inflow practice -- the profile itself has already flattened out by
    # its last few rows, so this is a smooth extension, not a discontinuity).
    U_in = np.interp(yc, yp, Up, left=Up[0], right=Uinf)
    k_in = np.interp(yc, yp, kp, left=kp[0], right=kinf)
    om_in = np.interp(yc, yp, omp, left=omp[0], right=ominf)

    print(f"Interpolated onto {len(yc)} mesh cell-centres "
          f"(y={yc[0]:.6f} .. {yc[-1]:.6f})")

    write_fields(U_in, k_in, om_in, Uinf, kinf)


def foam_header(cls, obj):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           | www.openfoam.com
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def write_scalar_list(vals):
    body = "\n".join(f"{v:.6e}" for v in vals)
    return f"nonuniform List<scalar>\n{len(vals)}\n(\n{body}\n)\n;"


def write_vector_list(u_vals):
    body = "\n".join(f"({v:.6e} 0 0)" for v in u_vals)
    return f"nonuniform List<vector>\n{len(u_vals)}\n(\n{body}\n)\n;"


def write_fields(U_in, k_in, om_in, Uinf, kinf):
    U_field = f"""{foam_header('volVectorField', 'U')}
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({Uinf:.4f} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           {write_vector_list(U_in)}
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    bottom
    {{
        type            noSlip;
    }}

    top
    {{
        type            slip;
    }}

    front
    {{
        type            empty;
    }}

    back
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""

    k_field = f"""{foam_header('volScalarField', 'k')}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {kinf:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           {write_scalar_list(k_in)}
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    bottom
    {{
        // Dirichlet k=0, matching Davidson's own solver exactly
        // (k_bc_south=0, k_bc_south_type='d' in exec-pyCALC-RANS.py) --
        // NOT kqRWallFunction (a zero-gradient/high-Re treatment that
        // never pulls k to zero). On this low-Re, y+<1 wall-resolving
        // mesh, kqRWallFunction only "worked" for standard kOmega because
        // omega's near-wall blow-up keeps the destruction term
        // (Cmu*omega*k) strong regardless of wall BC; kOmegaDavidsonNN
        // weakens that same term via CkNN~0.01-0.05 near the wall, so
        // nothing was left pulling k to its correct near-zero wall value.
        // See project_flatplate_cf_investigation memory for the full
        // root-cause chain (~12% Cf over-prediction, this BC fix closes
        // part but not all of the gap).
        type            fixedValue;
        value           uniform 0;
    }}

    top
    {{
        type            zeroGradient;
    }}

    front
    {{
        type            empty;
    }}

    back
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""

    omega_field = f"""{foam_header('volScalarField', 'omega')}
dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 100;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           {write_scalar_list(om_in)}
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    bottom
    {{
        type            omegaWallFunction;
        value           uniform 100;
    }}

    top
    {{
        type            zeroGradient;
    }}

    front
    {{
        type            empty;
    }}

    back
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""

    for case in ('kOmega', 'kOmegaDavidsonNN'):
        for name, content in (('U', U_field), ('k', k_field), ('omega', omega_field)):
            path = os.path.join(CASE_DIR, case, '0', name)
            with open(path, 'w') as f:
                f.write(content)
            print(f"Wrote {path}")


if __name__ == '__main__':
    main()
