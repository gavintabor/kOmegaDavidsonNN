# kOmegaDavidsonNN

OpenFOAM v2606 implementation of the k-ω-PINN-NN turbulence model of:

> Lars Davidson (2026), "Using physics informed neural network (PINN) and
> neural network (NN) to improve a k-omega turbulence model",
> Journal of Turbulence, DOI: 10.1080/14685248.2026.2665148

## Description

The model replaces three constant coefficients in the Wilcox k-ω model
with spatially varying fields computed by three small neural networks
(2→10→10→1, ReLU activations). The networks take two local dimensionless
input features and produce improved predictions of turbulent kinetic
energy k without degrading the velocity profile.

All neural network weights are hardcoded as C++ static arrays — no
runtime ML library dependency is required.

## Prerequisites

- OpenFOAM v2606 (ESI/com)
- Python 3 with numpy and matplotlib (for post-processing)

## Build

```bash
cd src
wmake libso
```

The library is installed to `$FOAM_USER_LIBBIN`.

Note for porting to other OpenFOAM versions: the two custom omega wall BCs
(`kOmegaDavidsonNNWallFunction`, `kOmegaDavidsonNNOmegaBC`) deliberately omit
the bare "construct as copy" constructor and no-arg `clone()`, matching the
constructor pattern OpenFOAM v2606 uses for its own wall functions (e.g.
`kqRWallFunctionFvPatchField`) — their base classes delete that constructor.
Older OpenFOAM versions may still provide it; newer ones may require the same
omission here.

## Quick start

```bash
./Allrun
```

Builds the library and runs all verification and application cases.

## Cases

Each case directory contains sub-cases for the turbulence models being compared.
Most cases provide a `kOmega` baseline and a `kOmegaDavidsonNN` variant.

| Case | Description | Reference |
|---|---|---|
| channelFlow5200 | Fully-developed channel flow Re_tau=5200 | Davidson Fig. 8 |
| flatPlate | Flat-plate boundary layer, developing from Re_theta=2550 | Davidson Fig. 12 |
| pitzDaily | Backward-facing step (OpenFOAM tutorial) | — |
| periodicHill | Periodic hill Re=10565 | Davidson Fig. 14 |

### channelFlow5200 sub-cases

Fully-developed channel flow at Re_τ = 5200 (ν = 1/5200 m²/s, δ = 1 m).
Both sub-cases are driven by a **fixed body force** of 1.0 N/m³ in x
(`vectorSemiImplicitSource`), which fixes u_τ = 1.0 m/s by construction
(τ_w = f·δ = 1.0 Pa, u_τ = √τ_w = 1.0 m/s).  Both use SIMPLEC.

| Sub-case | Model | omega wall BC | U_bulk | k⁺ peak | k⁺ centre |
|---|---|---|---|---|---|
| `kOmega` | Wilcox k-ω | omegaWallFunction | 24.13 | 3.11 | 0.47 |
| `kOmegaDavidsonNN` | kOmegaDavidsonNN | kOmegaDavidsonNNOmegaBC | 24.33 | 5.75 | 0.67 |
| DNS (Lee & Moser 2015) | — | — | ~24.1 | 5.87 | 0.87 |

`kOmegaDavidsonNN` runs with the actual neural network (`usePySR false`, the
library default — see "Optional PySR expression for σ_k" below). The NN
correction closely tracks DNS throughout the channel — U⁺ and k⁺ both match
Davidson (2026) Fig. 8 closely, with U_bulk staying near both `kOmega` and
DNS (24.33 vs 24.13/~24.1) and k⁺ peak rising from 3.11 to 5.75 (DNS 5.87).
See "Known limitations" below for the friction-velocity fix this relies on.
Post-processing: `python3 plotChannelFlow.py kOmega kOmegaDavidsonNN` from
the `channelFlow5200/` directory. `NN_coefficients_kOmegaDavidsonNN.png`
plots σ_k,NN and C_k,NN on a shared left axis and C_ω2,NN on its own right
axis (both vs y), matching Davidson (2026) Fig. 8(d)'s combined-axes layout
rather than three separately-scaled panels.

### flatPlate sub-cases

Flat-plate zero-pressure-gradient turbulent boundary layer, matching Davidson
(2026) Sec. 5.2's actual setup rather than a simpler leading-edge-developing
approximation: a 150×90 grid over a 75.19×18.02 m domain (uniform streamwise,
geometric-then-uniform wall-normal grading, matching his stated 92δ_in×20δ_in
size), ν = 3.57×10⁻⁵ m²/s, and — most importantly — an **inlet condition that
is already a fully turbulent boundary layer at Re_θ=2550** (U, k, ω profiles
from Davidson's own precursor RANS run,
`literature/pythons-rans-code-RANS-open/boundary-layer-.../`), not a uniform
freestream growing from a sharp leading edge. `generate_inlet_profile.py`
regenerates the `0/{U,k,omega}` inlet boundary values and the blockMesh
grading fractions from that reference data; it's a one-time provenance
script, not part of `Allrun` (its inputs live in `literature/`, which isn't
distributed with this repo — see "Not in the git repo" below).

Re_θ(x) is computed from the actual velocity field's momentum thickness
(using the *local* edge velocity at each station, not a fixed freestream
reference — this domain's finite height causes mild core-flow acceleration
downstream that a fixed reference would misread as Re_θ *decreasing*), not
a leading-edge correlation. `plotFlatPlate.py` extracts profiles at fixed
Re_θ targets (3000, 4000, 4500, 5500 — the third matches Davidson's Fig. 12
comparison station, the fourth matches the DNS reference data below) rather
than arbitrary fractions of the plate length, and overlays DNS (Sillero,
Jiménez & Moser 2014 — the dataset Davidson's Fig. 12 actually cites, closest
available station locally at Re_θ=5500) on the U⁺, k⁺, and u'v'⁺ plots. The
skin-friction plot uses Davidson's own correlation and ±6% band
(`Cf = 2(1/0.384·ln(Re_θ)+4.127)⁻²`) on his exact linear Re_θ=3000–5000 /
Cf=2.8–3.6×10⁻³ axes, and the NN-coefficient plot shows σ_k,NN/C_k,NN (left
axis) and C_ω2,NN (right axis) against y/δ₉₉ at each station — all five
now structurally match Davidson's Fig. 12(a)-(e) rather than the differently
axed/binned plots this port previously produced.

**Known open finding**: around Re_θ≈4600–4900, `kOmegaDavidsonNN` shows a
sharp (near-discontinuous) transition in near-wall k, ν_t, and the NN
coefficients — traced to the σ_k,NN/C_k,NN/C_ω2,NN ↔ k feedback becoming
locally numerically stiff at the flat-plate default EWMA window
(`ewmaM=500`, Davidson Eq. 18) once the boundary layer develops far enough
downstream (further than Davidson's own 92δ_in domain likely reaches). Using
his channel-flow window instead (`ewmaM=3000`) smooths this into a gradual
transition over a much wider Re_θ range instead — same eventual state, no
longer stiff — at the cost of a much slower-responding coefficient
everywhere else on the plate, and a correspondingly much longer run to
convergence (roughly 100,000 iterations here, vs ~15,000 at `ewmaM=500`).
This case now uses `ewmaM=3000`. See "Known limitations" below.

### periodicHill sub-cases

The periodic hill case (Re=10565, ν=2.650×10⁻⁶ m²/s, U_b=1 m/s, H=28 mm) provides
four sub-cases for a two-stage steady → transient workflow:

| Sub-case | Solver | Model | Purpose |
|---|---|---|---|
| `steadyState` | simpleFoam | Spalart-Allmaras | Tutorial baseline initialisation |
| `steadyState_kOmegaDavidsonNN` | simpleFoam | kOmegaDavidsonNN | Steady-state initialisation for transient |
| `transient` | pimpleFoam | SA-IDDES | Tutorial LES reference |
| `transient_kOmegaDavidsonNN` | pimpleFoam | kOmegaDavidsonNN | NN-corrected RANS transient |

The two steady-state cases use second-order linearUpwind momentum convection and one
non-orthogonal corrector.  The `transient_kOmegaDavidsonNN` case links its processor
directories from `steadyState_kOmegaDavidsonNN` and runs for 10 through-flow times (2.52 s),
with time-averaging of U, p, k, R, sigmakNN, CkNN and Comega2NN.

## Usage in your own cases

Add to `system/controlDict`:

```
libs ("libkOmegaDavidsonNN.so");
```

Set in `constant/turbulenceProperties`:

```
simulationType  RAS;
RAS
{
    RASModel    kOmegaDavidsonNN;
    ewmaM       3000;       // EWMA window; default 500 converges too fast
    turbulence  yes;
    printCoeffs on;
}
```

The `ewmaM` parameter controls how quickly the NN outputs are smoothed into the
coefficient fields via exponential weighted moving averaging
(`a = exp(-1/ewmaM)`). The default of 500 produces noticeably different
steady-state velocity profiles from 3000; always set it explicitly.

## Optional PySR expression for σ_k

The model coefficient `usePySR` (default `false`) replaces the σ_k neural network
with the symbolic regression expression from Davidson (2026) Eq. 22:

    sigma_k = 0.469 x0 + (0.574 + 1/(49.3 + 1/(x0 x1² − 0.362)))
                         / (x0 + 0.246 + 0.0516 x1/x0)

where x0 = ν_t/(y u_τ) and x1 = τ_tot/u_τ², both clipped to training range.
The NN forward passes for C_k and C_ω2 are unaffected.

Enable in `constant/turbulenceProperties`:

```
RAS
{
    RASModel        kOmegaDavidsonNN;
    ewmaM           3000;
    turbulence      yes;
    printCoeffs     on;
    usePySR         true;
}
```

### Comparing σ_k,NN against the pySR expression

`postProcessing/plotSigmaK_pySR_vs_NN.py` recreates Davidson (2026) Figure 19,
comparing σ_k predicted by the neural network against Eq. 22 (pySR).
`cases/channelFlow5200/Allrun` now runs this automatically after
`plotChannelFlow.py`, so `sigmak_pySR_vs_NN.png` is regenerated on every full
run of the case. To run it standalone against an existing result (the case
must have been run with `usePySR false`, the default), from
`cases/channelFlow5200/`:

```bash
python3 ../../postProcessing/plotSigmaK_pySR_vs_NN.py kOmegaDavidsonNN
```

It reads the converged `y`, `U`, `k`, `nut` and `sigmakNN` fields from the
case's latest time directory, reconstructs the same x0, x1 input features
`computeNNCoefficients()` computes internally, evaluates Eq. 22 on them in
Python (including the `max(..., SMALL)` guard on Eq. 22's near-singular term
that the C++ implementation uses — omitting it produces spurious spikes right
next to the pole), and plots both curves against the NN's actual `sigmakNN`
output — saving `sigmak_pySR_vs_NN.png`.

The two curves broadly track each other but are not expected to match
tightly everywhere: Davidson's Eq. 22 was fit by symbolic regression to the
*PINN* target, not to this smaller deployment NN, so pySR ≈ PINN by
construction while NN ≈ PINN only as well as the small network's own
training/generalisation error allows. Where the two diverge along a
particular profile depends on which part of (x0, x1) input space that
profile's u_τ-normalised trajectory happens to pass through — see below.

#### σ_k,NN vs. pySR over the full input domain

`postProcessing/plotSigmaK_NN_vs_pySR_domain.py` evaluates both the NN
(weights parsed directly from `src/kOmegaDavidsonNN.C`, so this can't itself
introduce a transcription error while checking for one) and Eq. 22 over the
*entire* trained (x0, x1) rectangle — independent of any CFD run, so it has
no u_τ, EWMA, or trajectory-dependent confounds:

```bash
python3 ../../postProcessing/plotSigmaK_NN_vs_pySR_domain.py [case_dir]
```

`case_dir` is optional; if given, that case's actual (x0, x1) trajectory is
overlaid on the domain heatmap for context. For `channelFlow5200`, the
trajectory hugs the domain boundary (running along x1≈1 near the wall, then
peeling diagonally toward the far corner) rather than passing through the
interior; a local σ_k,NN vs. σ_k,pySR mismatch near y/δ≈0.02–0.05 lines up
exactly with where that boundary-hugging path crosses a region where the two
functions genuinely disagree (confirmed on the synthetic grid, RMS
difference ≈0.24 over the full domain — comparable to what's seen along the
real profile). This is an intrinsic pySR-fit-quality limitation, not a bug
in this port.

## Omega wall boundary conditions

Two omega wall BCs are provided in the library:

- **`kOmegaDavidsonNNWallFunction`** — derives from `omegaWallFunctionFvPatchScalarField`.
  Applies Davidson (2026) Eq. 7 (`omega_w = 6ν/(Comega2NN·y²)`) when `Comega2NN` is in
  the registry, otherwise falls back to the standard wall function. Use this for
  low-Re meshes with a wall function approach.
- **`kOmegaDavidsonNNOmegaBC`** — derives from `fixedValueFvPatchScalarField`.
  A pure Dirichlet BC that sets `omega_w = 6ν/(Comega2NN·y²)` directly on each wall
  face (falls back to `6ν/(beta1·y²)` with beta1=0.072 if the model is not active).
  Requires a sufficiently fine near-wall mesh (y⁺ ~ 1).

Both BCs clamp `Comega2NN` to a minimum of 0.01 before use in the denominator,
preventing division by near-zero values on the first iteration before the NN fields
have fully initialised. Physical values of Cω2,NN are always above 0.02.

## Solver recommendations

For body-force-driven periodic flows (channel, periodic hill) use **SIMPLEC**
(`consistent yes` in the `SIMPLE` block) with pressure relaxation set to 1.0.
Standard SIMPLE with `p` under-relaxation of 0.3–0.5 significantly underpredicts
bulk velocity in these configurations, even after 15 000 iterations.
Recommended relaxation factors:

```
relaxationFactors
{
    fields    { p  1.0; }
    equations { U  0.9;  k  0.5;  omega  0.5; }
}
```

## Known limitations

- The model was trained on 2D channel flow data. Performance in strongly
  3D or highly separated flows has not been validated.
- The friction velocity used to normalise the NN's own input features
  (`x0 = ν_t/(y·u_τ)`, `x1 = τ_tot/u_τ²`, Davidson Eq. 17) is computed
  **once from the wall-adjacent cells** (viscous-sublayer formula,
  area-averaged over all wall patches) and held constant across the whole
  domain — matching Davidson's reference Python implementation, where
  `u_τ` is computed once per wall-normal column and broadcast unchanged
  along it (see `literature/pythons-rans-code-RANS-open/channel-10000-half-channel-NN-PINN-.../modify_case.py`).
  An earlier version of this port instead evaluated `Cmu^0.25·√k` locally
  at every cell, which couples the NN's inputs to the very k field the
  model is boosting and was found to distort ν_t substantially outside the
  immediate near-wall region (up to ~75% off from the standard model at
  the channel centreline) — degrading both the U⁺ and k⁺ match to DNS.
  This single-domain-average approach is correct for a homogeneous channel
  (`channelFlow5200`) and a reasonable approximation for a slowly-growing
  boundary layer (`flatPlate`), but does **not** yet handle cases with
  multiple or separated walls (`pitzDaily`, `periodicHill`) — those still
  use the same domain-wide average, which is a coarser approximation there.
- **`flatPlate`'s σ_k,NN/C_k,NN/C_ω2,NN ↔ k feedback is numerically stiff at
  Davidson's stated flat-plate EWMA window.** With `ewmaM=500` (his default
  for non-channel cases), near-wall k, ν_t, and the NN coefficients undergo a
  near-discontinuous ~3× collapse over a handful of cells around
  Re_θ≈4600–4900 — confirmed to persist in the fully-converged solution, not
  a transient artifact. Davidson's own note that this class of instability
  ("slow oscillations... related to the strong elliptic character") doesn't
  occur in flat-plate flow is about *temporal* oscillation at a point; this
  is a distinct *spatial* stiffness that only appears once the boundary
  layer develops far enough downstream — plausibly further than his own
  92δ_in domain reaches, which may be why it isn't reported in the paper.
  Using his channel-flow window (`ewmaM=3000`) instead resolves it into a
  smooth transition over a much wider Re_θ range (same eventual state, not
  stiff), at the cost of a much more sluggish coefficient response
  everywhere else and a proportionally much longer run to convergence. This
  case now uses `ewmaM=3000`; see the "flatPlate sub-cases" section above.

## Citation

If you use this model please cite:

Davidson, L. (2026). Using physics informed neural network (PINN) and
neural network (NN) to improve a k-omega turbulence model.
Journal of Turbulence. DOI: 10.1080/14685248.2026.2665148

## Literature

The `literature/` directory holds the source papers and Davidson's original
reference implementation this library was ported from:

| File / directory | Content |
|---|---|
| `Davidson.pdf` | The journal paper itself — full derivation of σ_k,PINN, C_k,PINN, C_ω2,PINN (Eqs. 1–22) and the σ_k,NN / C_k,NN / C_ω2,NN neural networks implemented here. |
| `DavidsonJoTPages/` | The same paper split one page per PDF file (`page_0001.pdf` … `page_0023.pdf`). |
| `Using-Physical-Informed-Neural-Network-PINN-to-Improve-a-k-omega-Turbulence-Model.pdf` | Earlier ERCOFTAC ETMM conference precursor. PINN only (no NN generalisation): σ_k, C_k and C_ω2 are fixed functions of y/δ, which only works for non-recirculating channel/boundary-layer flow. The journal paper replaces these with the NN models (functions of local flow variables) used in this library, enabling recirculating flows like periodicHill. |
| `py-calc-rans.pdf` | Manual for **pyCALC-RANS**, Davidson's 2D Python RANS solver used to develop and validate the model before porting to OpenFOAM. Covers the numerics (SIMPLEC, MUSCL/hybrid convection schemes), the k-ω/k-ε/EARSM models, and how the NN model is called from the solver — useful background if the OpenFOAM behaviour ever needs cross-checking against the original Python implementation. |
| `OpenFOAM Documentation - Periodic hill.pdf` | Standard OpenFOAM tutorial reference for the periodicHill case geometry and mesh. |
| `pythons-rans-code-RANS-open/` | Davidson's original Python source and training data (from his website) that produced the coefficients hardcoded into `src/kOmegaDavidsonNN.C`. Its own `README` maps scripts to outputs; the relevant ones live in `PINN-NN/`: `vist-diffusion-pinn-5200-half-channel-*.py` (solves for ν_t,PINN, Eq. 10), `compute-c_k-and-c_omega_2-from-balance-of-k-and-omega-eqns.py` (C_k,PINN / C_ω2,PINN, Eqs. 15–16), `neural-k-omega-{c_k,c_omega_2,prand_k}-vist-over-y-and-uv_tot.py` (training scripts for the three NNs), and `prand_k-symbolic.py` (the pySR expression for σ_k, Eq. 22 — exposed here as the `usePySR` option). The `channel-*-NN-PINN-*` and `boundary-layer-*-NN-*` sub-directories are the reference pyCALC-RANS runs validated against DNS before the OpenFOAM port. Note: this directory also contains an unrelated, earlier EARSM+NN model (`NN/`, `channel-10000-earsm-NN/`, `Using-Neural-Network-for-Improving-an-Explicit-Algebraic-Stress-Model-in-2D-Flow.pdf`) from a separate Davidson paper — not used by this library.

**Not in the git repo**: `literature/` is excluded via `.gitignore` — it's Davidson's own paper and reference code, not ours to redistribute, so it stays local-only. The `.gitignore` also excludes everything regenerable by `Allrun` (solver logs, solved time-step directories, case `postProcessing/` output, compiled build artifacts) so clones stay small; run `./Allrun` after cloning to reproduce them.
