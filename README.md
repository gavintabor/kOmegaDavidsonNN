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
the `channelFlow5200/` directory. `NN_coefficients_channelFlow5200_kOmegaDavidsonNN.png`
plots σ_k,NN and C_k,NN on a shared left axis and C_ω2,NN on its own right
axis (both vs y), matching Davidson (2026) Fig. 8(d)'s combined-axes layout
rather than three separately-scaled panels. (Prefixed with the case name so
it doesn't collide with `pitzDaily`'s own NN-coefficient plot when figures
from multiple cases are collected together — both cases' sub-case is
literally named `kOmegaDavidsonNN`.)

### flatPlate sub-cases

Flat-plate zero-pressure-gradient turbulent boundary layer, matching Davidson
(2026) Sec. 5.2's actual setup rather than a simpler leading-edge-developing
approximation: a 150×222 grid over a 75.19×18.02 m domain (uniform streamwise,
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

**Fixed bug**: the derived grading fractions match Davidson's total
length/expansion-ratio/cell-count for the near-wall geometric region
exactly, but OpenFOAM's `simpleGrading` (a single-ratio geometric series)
doesn't reproduce his actual internal point distribution there — his
mesh-generation code apparently isn't a pure fixed-ratio series either, so
our regenerated mesh's cell-centre y-positions drift up to ~7.5% from his in
that region (converging to an exact match beyond y≈5, i.e. the outer
uniform 2/3 of the domain). An earlier version of this script interpolated
Davidson's profile onto *his* y2d.dat-derived positions and wrote the
result into our mesh by cell index, silently assuming the two meshes'
cell-index-to-y-position mapping matched. Where the profile is steep near
the wall, that 7.5% position error translated into inlet values up to 46%
wrong (in k, at the wall-adjacent cell; 7.6% in U). Fixed by having the
script read our own mesh's actual generated cell centres (`blockMesh` +
`writeCellCentres`) and interpolating onto those directly — correct
regardless of any grading-algorithm mismatch. The effect on results is
small but real (see updated numbers below).

Rerunning after that fix exposed a second, unrelated latent bug: `kOmega`'s
Re_θ growth came out fractionally slower (max ≈5496 vs. the previous
≈5564), landing its last profile-extraction target (Re_θ=5500) right on the
domain's very last streamwise station — where `column_mask`'s tolerance
check (`< 0.5*dx`) failed by ~3e-9 due to ordinary floating-point rounding,
silently dropping that station (`compute_re_theta` returned NaN there,
poisoning `np.interp` for any nearby target). This is why `kOmega` was
missing from the last panel of `Uplus_profiles.png`/`kplus_profiles.png`/
`uv_profiles.png` after the inlet-profile fix — not a new problem, just a
previously-dormant edge case now landed on. Fixed by adding a small
relative tolerance to the boundary check.

Re_θ(x) is computed from the actual velocity field's momentum thickness
(using the *local* edge velocity at each station, not a fixed freestream
reference — this domain's finite height causes mild core-flow acceleration
downstream that a fixed reference would misread as Re_θ *decreasing*), not
a leading-edge correlation. `plotFlatPlate.py` extracts profiles at fixed
Re_θ targets (3000, 4000, 4500, 5500 — the third matches Davidson's Fig. 12
comparison station, the fourth matches the DNS reference data below) rather
than arbitrary fractions of the plate length, and overlays DNS (Sillero,
Jiménez & Moser 2013 — the dataset Davidson's Fig. 12 actually cites, closest
available station locally at Re_θ=5500) on the U⁺, k⁺, and u'v'⁺ plots. The
skin-friction plot uses Davidson's own correlation and ±6% band
(`Cf = 2(1/0.384·ln(Re_θ)+4.127)⁻²`) on his exact linear Re_θ=3000–5000 /
Cf=2.8–3.6×10⁻³ axes, and the NN-coefficient plot shows σ_k,NN/C_k,NN (left
axis) and C_ω2,NN (right axis) against y/δ₉₉ at each station — all five
now structurally match Davidson's Fig. 12(a)-(e) rather than the differently
axed/binned plots this port previously produced.

**Resolved finding (was "Known open finding")**: an earlier version of this
section reported a sharp (near-discontinuous) transition in near-wall k,
ν_t, and the NN coefficients around Re_θ≈4750–4900 — a ~3× collapse in k
over about 4 cells — and, after ruling out the EWMA averaging window
(Davidson Eq. 18) as the cause, concluded it was "a genuine converged
steady-state property of the coupled model". That conclusion was wrong: the
real cause was `uTauGlobal`, the single domain-wide-averaged friction
velocity then used to normalise the NN's (x0,x1) inputs (see "Known
limitations" below) — a fair approximation near the mesh's mean Reynolds
number but progressively worse for stations far from it, worst at the
highest-Re_θ station, exactly where the transition appeared. Rerunning
after switching to per-column u_τ (matching Davidson's own method) removes
it completely: `Cf_vs_Retheta.png` is now smooth and monotonic across the
whole Re_θ=3000–5000 range, and the previously-anomalous last panel of
`kplus_profiles.png`/`Uplus_profiles.png`/`uv_profiles.png` now follows the
same trend as the other three stations. Confirmed by diffing freshly
regenerated plots against the previously-committed ones. This case keeps
the default `ewmaM=500` (Davidson's stated default), since 3000 reaches the
identical answer for ~6× the iterations.

**Fixed: ~12% Cf over-prediction, root-caused to two compounding issues.**
After the u_τ fix above, `kOmegaDavidsonNN`'s Cf still ran ~9-12% above
`kOmega`/the correlation throughout, unlike Davidson's own Fig. 12(a) (where
his NN model tracks kOmega/correlation closely — confirmed by loading his
own saved `cf_vs_re_mom.png` from the exact reference run this port's NN
weights come from). The near-wall U⁺-vs-y⁺ profiles looking almost
identical between models is *not* evidence against a real difference: both
axes are non-dimensionalised by each model's own u_τ, so the viscous
sublayer always plots as U⁺≈y⁺ regardless of the actual dimensional wall
shear. Pulling raw (non-normalised) fields showed `kOmegaDavidsonNN`'s
near-wall k running two to three orders of magnitude above `kOmega`'s at
the same physical point.

Every production/destruction/diffusion term in `src/kOmegaDavidsonNN.C` was
checked algebraically against Davidson's own solver
(`literature/.../exec-pyCALC-RANS.py`) and matches exactly. The NN's own
output (`CkNN`≈0.01-0.05 in the viscous sublayer) also matches his saved
run closely — the trained network itself isn't at fault. Two real,
independent issues were found instead:

1. **Wall BC mismatch**: this model's k destruction term is weakened
   20-70× by `CkNN` near the wall (vs. standard `kOmega`, which is immune
   because ω's near-wall blow-up keeps `Cmu·ω·k` large regardless). Davidson's
   solver compensates with an explicit Dirichlet `k=0` wall condition
   (`k_bc_south=0`, `k_bc_south_type='d'`); this port used stock
   `kqRWallFunction` (a zero-gradient/high-Re treatment that never pulls k
   to zero) for k on **every** case. Switched the bottom patch to
   `fixedValue; value uniform 0;` in `flatPlate/{kOmega,kOmegaDavidsonNN}/0/k`
   (and in `generate_inlet_profile.py`'s template, so regeneration doesn't
   revert it). This closed part of the gap (~11.7%→~9.0%) but not all of it.
2. **Under-resolved first cell**: adding more near-wall cells while keeping
   the first-cell height fixed did nothing (ruled out generic
   under-resolution of the sharp near-wall NN-coefficient transition).
   Shrinking the *first cell itself* 5× (150×90 → 150×222, first cell
   y⁺≈0.1 instead of ≈0.6, same total near-wall segment length) closed
   almost all of the remaining gap (~9.0%→~0.6%) — and shifted `kOmega`'s
   own Cf too, revealing that the original mesh (which faithfully matched
   Davidson's documented y⁺≈0.8 design) wasn't actually fully grid-converged
   in OpenFOAM's discretization for *either* model. This mesh now
   deliberately exceeds Davidson's own resolution rather than literally
   reproducing his 150×90 grid.

`Cf_vs_Retheta.png` now shows `kOmega` and `kOmegaDavidsonNN` nearly
overlapping and tracking close to the correlation, matching the shape of
Davidson's Fig. 12(a). Only one refinement step was tested (not confirmed
grid-independent with further refinement) — see
`project_flatplate_cf_investigation` session notes if revisiting this.

### pitzDaily sub-cases

Standard OpenFOAM tutorial backward-facing-step geometry (step height
H=25.4mm, expansion ratio 2:1, inlet U=10 m/s, ν=1×10⁻⁵ m²/s, Re_H=25,400),
not one of Davidson's own test cases — used here as a generic separated-flow
sanity check. `Allrun` runs `blockMesh` then `simpleFoam` for both `kOmega`
and `kOmegaDavidsonNN`, then `postProcessing/plotPitzDaily.py` computes the
reattachment length directly from the converged U field and samples U/k
profiles at fixed x/H stations via `postProcess -dict system/sampleDict`
(the standalone `sample` utility used by older tutorials no longer exists in
v2606).

| Sub-case | Reattachment x/H |
|---|---|
| `kOmega` | 7.32 |
| `kOmegaDavidsonNN` | 7.28 |
| Experiment (Pitz & Daily 1983, Re_H=22,000) | 7.0 |

Rerun with the per-column u_τ fix (see "Known limitations" below) — the
NN's own friction-velocity normalisation now varies along the wall instead
of using one domain-wide average. Reattachment length barely moved
(7.31→7.28); `kOmega` is unaffected (it doesn't use the NN's u_τ
normalisation at all). This case's separate wall-BC/mesh-resolution issue
found for `flatPlate` (see that section) has *not* been applied here yet —
pitzDaily still uses `kqRWallFunction` for k.

Good agreement, with a small residual gap plausibly explained by the
Reynolds number mismatch (this case's Re_H=25,400 vs. the experiment's
22,000 — the OpenFOAM tutorial geometry was never tuned to match Pitz &
Daily's flow conditions exactly, just its step/expansion geometry).

Both models are essentially indistinguishable in the mean velocity field —
the U profiles at x/H = 1, 2, 4, 6, 8, 10 (`U_profiles.png`) overlay almost
exactly. `kOmegaDavidsonNN` does measurably raise turbulent kinetic energy
through the shear layer at every station from x/H=2 onward relative to
`kOmega` (`k_profiles.png`), without changing the mean flow — a real,
repeatable difference, not noise.

The 2D field plots (`k_field_comparison.png`, `U_field_comparison.png`,
`omega_field_comparison.png`, `NN_coefficients_pitzDaily_kOmegaDavidsonNN.png`,
`nut_ratio.png`) use `tricontourf` on a Delaunay triangulation of the
cell-centre data, masked to drop triangles whose centroid falls in the solid
step corner (x<0, y<0) — an unmasked triangulation bridges straight across
that corner and blends inlet-duct values into downstream-duct ones through
solid geometry. All panels use equal-aspect axes so the domain isn't
visually stretched. `omega_field_comparison.png` uses a log color scale,
since ω∝1/y² at the wall spans several orders of magnitude more than the
bulk flow and saturates a linear one. `k_field_comparison.png` and
`omega_field_comparison.png` each end with an extra panel giving the
`kOmegaDavidsonNN`/`kOmega` ratio directly (as does the standalone
`nut_ratio.png` for turbulent viscosity), and `NN_coefficients_*.png` plots
σ_k,NN/C_k,NN/C_ω2,NN as a fraction of their baseline (pre-NN-correction)
value — 2.0/1.0/0.072 respectively, matching the reference lines already
used in `plotChannelFlow.py`/`plotFlatPlate.py` — on a shared 0.4–1.2
colorbar (all three stay within a fairly narrow band below baseline
throughout the domain). Together these show `kOmegaDavidsonNN` raising both
k and ν_t through the shear layer and downstream of reattachment relative to
`kOmega`, while suppressing all three NN coefficients somewhat below their
baseline values almost everywhere.

`streamlines_pitzDaily_{kOmega,kOmegaDavidsonNN}.png` visualize the flow
topology directly: (Ux,Uy) is interpolated from the masked triangulation
onto a regular grid with matplotlib's `LinearTriInterpolator` (which returns
a masked array outside the fluid domain, so streamlines correctly stop at
the true boundary rather than crossing the solid step corner), plotted over
a filled |U| contour. A broad, automatically-seeded `streamplot` pass
captures the through-flow and the large primary recirculation zone; a
second pass with explicit `start_points` concentrated in a small patch
right behind the step (x=0.5–15mm, hugging the new lower wall) resolves the
much smaller counter-rotating secondary corner vortex there, which
automatic density-based seeding alone misses. Both models show the same
topology: through-flow, one primary recirculation vortex closing out at the
reattachment point, and one small secondary corner vortex — no qualitative
difference in flow structure between `kOmega` and `kOmegaDavidsonNN`, only
in the k/ν_t/ω magnitudes discussed above.

No digitized Pitz & Daily (1983) velocity/turbulence profile dataset could
be found publicly for comparison. The ERCOFTAC Classic Collection's
similarly-named backward-facing-step case (case030) is actually a different
experiment — Driver & Seegmiller (1985), with a different expansion ratio
(1.125 vs. this case's 2.0) and a compressible free-stream (M=0.128) — so it
was not used. The single reattachment-length figure above (7.0, Pitz &
Daily 1983) is the only literature reference value available for this exact
geometry.

**Known-bug note**: an earlier version of `plotPitzDaily.py` used
`STEP_HEIGHT=0.0127` (half the actual 0.0254m mesh step, misread from the
blockMeshDict), which silently doubled every x/H figure it reported —
reattachment length came out as ~14.6 instead of ~7.3, and the sampling
lines' upper end (`0.0508`) actually extended past the real top wall at
`0.0254`. Both are fixed; the numbers in the table above are correct.

### periodicHill sub-cases

The periodic hill case (Re=10565, ν=2.643×10⁻⁶ m²/s, U_b=1 m/s, H=28 mm) is
Section 5.3 of Davidson (2026) — the only recirculating-flow test case, and
the one this port's setup diverges from most, deliberately. The baseline
(`steadyState`, Spalart-Allmaras → `transient`, SA-IDDES) is a pristine,
unmodified copy of `$WM_PROJECT_DIR/tutorials/incompressible/pimpleFoam/LES/periodicHill`
— OpenFOAM's own verification case for this exact geometry — used as-is for
provenance rather than a from-scratch reproduction of Davidson's own
(different) numerics.

| Sub-case | Solver | Model | Mesh | Purpose |
|---|---|---|---|---|
| `steadyState` | simpleFoam | Spalart-Allmaras | 3D (200×160×80, periodic front/back) | Tutorial baseline initialisation |
| `transient` | pimpleFoam | SA-IDDES | 3D, inherited from `steadyState` | Tutorial LES reference |
| `steadyState_kOmegaDavidsonNN` | simpleFoam | kOmegaDavidsonNN | 3D, same as baseline | Diagnostic only — see below |
| `steadyState_kOmegaDavidsonNN_2D` | simpleFoam | kOmegaDavidsonNN | 1-cell-thick (200×160×1, `empty` front/back) | Diagnostic only — see below |
| `transient_kOmegaDavidsonNN_2D` | pimpleFoam | kOmegaDavidsonNN | Same 1-cell-thick mesh, own fresh start | NN-corrected transient result |

**Why 1-cell-thick, not Davidson's 2-cell mesh.** Davidson's own hill mesh
is 2 cells thick in z with slip walls, not truly 2D — necessary because his
solver (`pyCALC-LES`) has no equivalent of OpenFOAM's `empty` patch type, so
a genuinely 1-cell direction gives it a degenerate self-referencing stencil.
OpenFOAM's `empty` patch removes the direction from the discretisation
entirely, so `steadyState_kOmegaDavidsonNN_2D`/`transient_kOmegaDavidsonNN_2D`
use a real 1-cell mesh (`front`/`back` type `empty`) — a closer match to
his 2D *intent* than his own workaround, and ~80× fewer cells. Only the
`kOmegaDavidsonNN` sub-cases move to this mesh; the baseline keeps the full
3D periodic mesh, since SA-IDDES is scale-resolving and genuinely needs it.

**Why `transient_kOmegaDavidsonNN_2D` starts fresh at t=0, not restarted
from a steady solution.** The obvious approach — steady SIMPLE init, then
restart transient from it, mirroring the baseline's own SA→SA-IDDES
workflow — was tried first and doesn't work for this model: the steady
`kOmegaDavidsonNN` solve converges (by its own residual metric) to a state
with a severe, spatially broad `k`/`ω` blow-up in a band along the wall
just behind the hill crest (`k` up to ~10⁶, confirmed by direct field
instrumentation to be spatially real, not a single-cell artifact). This
is a pseudo-time SIMPLE artifact, not a numerics bug in this port:
restarting `transient_kOmegaDavidsonNN_2D` from it diverges to a floating
point exception *instantly* under real time-accurate stepping, regardless
of how small a timestep is used (tried down to 1e-72 s). Davidson's own
paper explains why no steady fixed point should be expected there at all
— Section 5.3 never runs a steady RANS stage for periodic hill (his own
`pyCALC-RANS` "did not succeed" adjusting the driving pressure-gradient
coefficient for this flow); he goes straight to unsteady `pyCALC-LES` from
a fresh start. The same section separately notes DNS shows a large-scale
flapping motion near the crest that steady RANS cannot represent — plausibly
*why* no steady solution exists there: the physical attractor at that
location isn't a fixed point.

`transient_kOmegaDavidsonNN_2D` therefore starts from the same small-uniform
initial condition as the steady case (only the mesh is shared, via an
`Allrun`-time symlink, not the steady case's fields) and runs `pimpleFoam`
directly for ~44 through-flows (t=0–11.1 s, matching Davidson's own
40,000-step run), with adaptive timestepping (`maxCo 0.5`). This converges
cleanly to a fully bounded result (`k` max ≈0.1, vs ≈10⁶ restarting from
steady), with `U`/`p` time-averaged over the last ~20% of the run and
profiles sampled at Davidson's Fig. 14 x/H stations for DNS comparison.

`steadyState_kOmegaDavidsonNN`/`steadyState_kOmegaDavidsonNN_2D` are kept
as diagnostic cases even though they're no longer part of the transient
run's initialisation path: their saturated NN coefficients (pinned at
their clip bounds along most of the wall) independently corroborate a
finding in Davidson's own paper — Section 5.3 reports σ_k,NN/C_k,NN/C_ω2,NN
constant across 88% of his domain for the same reason (NN inputs clipped
to the training range), and shows that substituting those constant values
everywhere reproduces the full model's predictions almost exactly. So the
saturation itself is expected model behaviour on this flow, not a bug —
useful to keep visible rather than discard.

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
(`a = exp(-1/ewmaM)`). This mainly affects how many iterations are needed to
*reach* a converged steady state, not necessarily what that steady state is
— confirmed for `flatPlate`, where 500 and 3000 converge (at vastly
different iteration counts) to an identical result; see "Known
limitations". It may still matter for cases with a genuinely oscillatory
(non-fixed-point) steady state, so set it explicitly and check convergence
rather than assuming either value is "correct" by default.

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

`cases/NNCheck/plotSigmaK_NN_vs_pySR_domain.py` (also in `postProcessing/`)
evaluates both the NN (weights parsed directly from
`src/kOmegaDavidsonNN.C`, so this can't itself introduce a transcription
error while checking for one) and Eq. 22 over the *entire* trained (x0, x1)
rectangle — independent of any CFD run, so it has no u_τ, EWMA, or
trajectory-dependent confounds. `cases/NNCheck/` is the canonical place to
run this and `evaluate_NN_from_DNS.py` (which reproduces Davidson Fig. 5
directly from DNS, also CFD-independent) — both are pure NN-weight checks,
kept together separately from the CFD case directories:

```bash
cd cases/NNCheck
python3 plotSigmaK_NN_vs_pySR_domain.py [case_dir]
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
  **per wall-normal column** — the viscous-sublayer formula
  (`u* = √(ν·|U|/y)`) evaluated at each wall face, then broadcast to every
  cell whose nearest wall face is that one (constant along y, varying along
  the wall), via OpenFOAM's `wallDistAddressing` nearest-wall-face
  transport. This matches Davidson's reference Python implementation
  exactly (see `literature/pythons-rans-code-RANS-open/channel-10000-half-channel-NN-PINN-.../modify_case.py`,
  where `u_τ` is computed once per wall-normal column and broadcast
  unchanged along it) and applies uniformly to every case using this
  model. An earlier version of this port instead evaluated `Cmu^0.25·√k`
  locally at every cell, which couples the NN's inputs to the very k field
  the model is boosting and was found to distort ν_t substantially outside
  the immediate near-wall region — that approach is not used. A version
  after that used a single domain-wide average instead of per-column
  (exact for `channelFlow5200`'s homogeneous wall, a reasonable
  approximation for `flatPlate`'s slowly-growing boundary layer, but too
  coarse for `periodicHill`'s multi-regime wall); per-column normalisation
  replaces it everywhere, confirmed regression-clean on `channelFlow5200`
  (unchanged U_bulk/k⁺ vs previously documented values). For `periodicHill`
  specifically, switching to per-column did **not** resolve the NN
  coefficient saturation there — confirmed via direct field
  instrumentation that u_τ genuinely varies ~20× along the wall under the
  new method, yet the coefficients still saturate at their clip bounds
  almost everywhere. That turned out to be expected model behaviour on
  this flow rather than a normalisation artifact — see the "periodicHill
  sub-cases" section above. `pitzDaily` (also a multi/separated-wall case)
  inherits the same per-column change and has been re-validated against it
  — reattachment length barely moved (7.31→7.28), see "pitzDaily sub-cases"
  above.
- **`flatPlate`'s previously-reported Re_θ≈4750–4900 near-discontinuous
  transition was a u_τ normalisation artifact, not a genuine model
  property — now fixed.** An earlier investigation ruled out the EWMA
  averaging window as the cause and concluded the ~3× near-wall k collapse
  there was "a genuine converged steady-state property of the coupled
  model". That conclusion didn't hold up: the actual cause was the
  domain-wide-averaged `uTauGlobal` described above, which is least
  accurate at the highest-Re_θ (farthest-from-mean) station — exactly where
  the transition appeared. Switching to per-column u_τ removes it entirely;
  see the "flatPlate sub-cases" section above for the confirmed before/after
  comparison. This case keeps `ewmaM=500` (Davidson's stated default), since
  3000 reaches the identical answer for ~6× the iterations.

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
