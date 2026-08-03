#!/usr/bin/env python3
"""
evaluate_NN_from_DNS.py

Reproduce Davidson (2026) Figure 5: evaluate the three trained neural networks
(sigma_k,NN  C_k,NN  C_omega2,NN) directly from Lee & Moser (2015) DNS data at
Re_tau = 5200, bypassing the CFD solver entirely.

This is a pure verification of the hardcoded NN weights — if the weights were
transcribed correctly from Davidson's PyTorch models, the profiles here should
match his Figure 5 to plotting accuracy.

Inputs required (same directory as this script, or pass paths as arguments):
    LM_Channel_5200_mean_prof.dat
    LM_Channel_5200_vel_fluc_prof.dat

Usage:
    python3 evaluate_NN_from_DNS.py
    python3 evaluate_NN_from_DNS.py /path/to/dns/data/

Reference:
    Davidson L. (2026), Journal of Turbulence,
    DOI: 10.1080/14685248.2026.2665148
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams.update({'font.size': 13})

# -----------------------------------------------------------------------
# DNS file locations
# -----------------------------------------------------------------------
DNS_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
MEAN_FILE  = os.path.join(DNS_DIR, 'LM_Channel_5200_mean_prof.dat')
FLUCT_FILE = os.path.join(DNS_DIR, 'LM_Channel_5200_vel_fluc_prof.dat')

# -----------------------------------------------------------------------
# DNS parameters (from file header)
# -----------------------------------------------------------------------
U_TAU_DNS = 4.14872e-02   # friction velocity in DNS units
NU_DNS    = 8.00000e-06   # kinematic viscosity in DNS units
RE_TAU    = 5185.897      # actual Re_tau of the DNS
DELTA     = 1.0           # channel half-width

# -----------------------------------------------------------------------
# MinMax scaler bounds (from Davidson's min-max.txt, same for all 3 models)
# -----------------------------------------------------------------------
VOVY_MIN = 1.883151195742736211e-04   # nut/(y*uTau) minimum
VOVY_MAX = 3.679415222987406642e-01   # nut/(y*uTau) maximum
UV_MIN   = 9.149365160399970665e-02   # tauTot/uTau^2 minimum
UV_MAX   = 9.949999999999999956e-01   # tauTot/uTau^2 maximum

# Output clipping bounds [min, max] per model: [sigma_k, C_k, C_omega2]
OUT_MIN = np.array([2.088044421529970922e-03,
                    4.182055545665602065e-04,
                    1.971251556352628771e-04])
OUT_MAX = np.array([1.846785570367704743e+00,
                    1.000000000000000000e+00,
                    7.499999999999999722e-02])

# -----------------------------------------------------------------------
# Hardcoded NN weights — transcribed from Davidson's PyTorch .pth files
# Architecture: 2 -> 10 (ReLU) -> 10 (ReLU) -> 1 (linear)
# Model 0: sigma_k,NN
# Model 1: C_k,NN
# Model 2: C_omega2,NN
# -----------------------------------------------------------------------

# --- Model 0: sigma_k,NN (prand_k) ---
W0_0 = np.array([
    [ 0.84573334,  0.71675086],
    [ 0.30598933, -0.15374757],
    [-0.01062202, -0.42763130],
    [-0.38246710, -0.92450360],
    [ 1.24818490, -0.12163923],
    [ 0.07547101, -0.46526850],
    [ 0.75766486,  0.43806720],
    [-0.27057737, -0.31967464],
    [-0.44039032,  0.84182750],
    [ 0.24243979, -0.67738307],
])
b0_0 = np.array([-1.47228320, -0.52008015, -0.53476745,  0.47544998,  0.11527929,
                 -0.40769680, -0.32854357,  0.09974384,  0.16558872, -0.10354772])

W1_0 = np.array([
    [-0.02149744,  0.23287492, -0.13618506, -0.19264500, -1.14958580,
      0.13889815, -0.55999213, -0.17327847,  0.52973930, -0.15481530],
    [-0.02863370, -0.12169755,  0.14927661,  0.07517920, -0.24786106,
      0.09324402, -0.43199304, -0.13053326, -0.09313165, -0.11620283],
    [ 0.13326661,  0.02152391, -0.12624392,  0.10522312,  0.03179962,
      0.06272271, -0.16517013, -0.27851660, -0.30229142,  0.25073700],
    [ 0.51622456, -0.22447969,  0.19519451,  0.16535406, -0.11379611,
     -0.00536794,  0.37536030,  0.04931431, -0.12518048,  0.14165910],
    [-0.27521715,  0.18363190, -0.23444520, -0.09755673,  0.02493668,
     -0.26737870, -0.02570092, -0.04516621,  0.05903452, -0.26542327],
    [ 0.27670395, -0.00626766,  0.29013163,  0.20745046,  0.18861842,
     -0.19051180,  0.21369375, -0.29337433, -0.21886474, -0.20482357],
    [-1.36297140, -0.00798734,  0.20846705, -0.79892176,  0.28744307,
      0.02012251, -0.28814160,  0.30818254,  0.28893405,  0.12214890],
    [ 0.05828657, -0.21776566,  0.17297788, -0.24987470, -0.24160889,
     -0.17431074,  0.18842086, -0.10950086,  0.14537056, -0.04841214],
    [-0.26397598,  0.24057534, -0.10995997, -0.18727595, -0.47234324,
     -0.31139007, -0.22392434,  0.08821702,  0.05741542, -0.19032577],
    [ 0.32245833, -0.24188577, -0.15992098, -0.24725310,  0.16982083,
     -0.17523241,  0.00941061,  0.15298535,  0.21800682, -0.30430284],
])
b1_0 = np.array([ 0.19627425,  0.67638320, -0.21749318,  0.49432520, -0.12401722,
                  0.38866934,  0.20923899, -0.16567044,  0.20975457,  0.03469171])
W2_0 = np.array([-1.32515490,  0.74092640,  0.07024844,  0.63121665,  0.23236890,
                  0.41286683, -1.37455430,  0.00994973, -0.43276906,  0.14725070])
b2_0 = 0.91513103

# --- Model 1: C_k,NN (c_k) ---
W0_1 = np.array([
    [-0.67063520, -0.11565896],
    [ 0.08191139, -0.29265523],
    [ 0.71940750, -0.45442116],
    [ 0.10355792,  0.15975173],
    [ 0.33451240, -0.45836246],
    [-0.14580584,  0.66664460],
    [-0.19294487,  1.00301580],
    [-0.04875571,  0.43166134],
    [-0.18230611, -0.67036533],
    [-0.13399394, -0.51753290],
])
b0_1 = np.array([-0.02854716, -0.06567212,  0.01560317, -0.69735104,  0.54883635,
                  0.52213330, -0.35360262, -0.43210363,  0.01414929, -0.68314680])

W1_1 = np.array([
    [ 0.12527200,  0.25092533,  0.27659700,  0.18672176,  0.23821600,
      0.08103832, -0.69043570, -0.04154851,  0.22215566, -0.15291502],
    [ 0.03483474, -0.31559518, -0.28940590,  0.26887560, -0.20037095,
     -0.19373661, -0.04507084,  0.07235176,  0.19697790,  0.03715689],
    [-0.16980796, -0.00178221, -0.18045530,  0.30071650, -0.53796750,
      0.34389973,  0.60373574, -0.17898515,  0.22570576,  0.25042510],
    [ 0.18793225,  0.06275942, -0.25753555,  0.31257105, -0.25562736,
     -0.16699792, -0.27974987,  0.31149948, -0.14857933, -0.25764546],
    [-0.19426894,  0.14390619, -0.06763518,  0.10293505, -0.08274508,
     -0.15527982,  0.30341870,  0.15530504, -0.25604692,  0.16808674],
    [-0.30942252,  0.12802982, -0.07983677,  0.01279171,  0.26017380,
      0.09602885,  0.22321604, -0.17011765,  0.06142471, -0.15742862],
    [ 0.20460443, -0.05032362, -0.01530452,  0.14976811, -0.07467391,
      0.07019374, -0.22705245,  0.17492770,  0.18005736,  0.30938244],
    [-0.22235526, -0.31228143, -0.21157862, -0.21810286,  0.03713537,
     -0.21250850,  0.06637079, -0.29066867,  0.01055705, -0.17920250],
    [-0.22521487, -0.31225184, -0.23799416, -0.03030545, -0.17002623,
     -0.17663479,  0.07101712,  0.24684360,  0.02512064, -0.08649548],
    [-0.01022392, -0.01680956,  0.19381377,  0.05422423,  0.06047041,
     -0.14414650, -0.11931799,  0.16921456,  0.08405398,  0.23965436],
])
b1_1 = np.array([ 0.15603745,  0.22847530,  0.06215555, -0.12819026,  0.21622367,
                  0.00643995,  0.28422590, -0.23632964, -0.11477045, -0.27203998])
W2_1 = np.array([ 0.68675494,  0.10051088, -0.79175390,  0.28616210, -0.19273394,
                 -0.12251709,  0.20660718, -0.20291503, -0.25311300, -0.20967540])
b2_1 = 0.69326560

# --- Model 2: C_omega2,NN (c_omega_2) ---
W0_2 = np.array([
    [ 0.64225286,  0.64217930],
    [-0.26242650, -0.54147400],
    [-0.26130334,  0.48005596],
    [ 0.07950017, -0.16250071],
    [ 0.68885887,  0.51953160],
    [ 0.63955630, -0.34741870],
    [-0.57070200, -0.26000914],
    [-0.33751840, -0.38430062],
    [ 0.55936890, -0.46433762],
    [ 0.39201390, -0.60101070],
])
b0_2 = np.array([ 0.13212705, -0.12339620,  0.45035556, -0.49396905, -0.27922280,
                  0.49126515,  0.13989820,  0.31832808,  0.22840098, -0.48710874])

W1_2 = np.array([
    [ 4.34355922e-02, -1.73254088e-01, -1.15541860e-01,  1.02654614e-01,
     -2.41500750e-01, -7.52514824e-02,  7.65006104e-03,  7.73532838e-02,
      2.93647975e-01, -1.63473755e-01],
    [ 5.43159200e-03,  2.41725370e-01,  2.65700459e-01, -3.01275216e-02,
      1.44886062e-01,  1.77900225e-01,  8.41812864e-02,  1.30336508e-01,
      1.18385606e-01, -2.55000800e-01],
    [-2.83855736e-01, -6.41989335e-02, -1.00109100e-01, -2.33556256e-01,
      2.40719080e-01, -4.45787385e-02, -3.00688982e-01, -9.53193232e-02,
      1.64467007e-01, -3.08365017e-01],
    [ 1.33214116e-01, -1.77128658e-01, -3.21721464e-01, -1.33528158e-01,
     -1.81705326e-01,  9.25299674e-02, -8.50628763e-02, -2.16491818e-01,
      2.85836488e-01,  4.52365167e-02],
    [ 1.23714708e-01, -1.45128593e-01,  1.83202460e-01, -1.64559513e-01,
     -1.44198015e-01, -9.90626514e-02,  2.89266855e-01,  3.44305150e-02,
     -1.55026555e-01, -2.50073615e-02],
    [ 1.04679324e-01, -4.36686873e-02, -1.44178659e-01,  1.65821880e-01,
      2.53947258e-01,  2.19149530e-01,  1.20860368e-01,  6.40927255e-02,
      2.36142203e-02,  7.78831616e-02],
    [-1.76065728e-01, -1.95544913e-01,  1.20514013e-01, -2.71857083e-01,
      4.78894673e-02, -1.05473241e-02, -3.14956635e-01, -1.41534135e-01,
      2.74416715e-01, -3.05148542e-01],
    [ 1.78533003e-01,  2.82409340e-01,  1.32515714e-01,  1.52094260e-01,
     -5.06751388e-02,  2.22569361e-01, -2.99837589e-01,  5.98013029e-02,
      1.92218766e-01, -1.77610323e-01],
    [ 1.03784963e-01,  1.89790726e-01, -1.16089739e-01, -4.07170281e-02,
      2.34020650e-01, -2.58162975e-01, -2.00638533e-01, -1.52147532e-01,
     -1.10467978e-01,  1.20202474e-01],
    [-3.45364541e-01, -2.59202063e-01,  1.92471713e-01, -2.20736653e-01,
      2.86736995e-01, -3.10422108e-02, -1.95518523e-01, -1.84867502e-04,
     -1.78102300e-01, -1.85488552e-01],
])
b1_2 = np.array([ 0.31513286, -0.08149902, -0.26853368,  0.05588951, -0.14562656,
                 -0.16596714,  0.01635724, -0.03219016, -0.23034962,  0.11533230])
W2_2 = np.array([ 0.35092634,  0.00715044,  0.22688176, -0.23939902, -0.15560350,
                  0.01669093, -0.16002087,  0.10286507, -0.22823867, -0.15349457])
b2_2 = -0.04704970

# Collect into lists indexed by model number
W0 = [W0_0, W0_1, W0_2]
b0 = [b0_0, b0_1, b0_2]
W1 = [W1_0, W1_1, W1_2]
b1 = [b1_0, b1_1, b1_2]
W2 = [W2_0, W2_1, W2_2]
b2 = [b2_0, b2_1, b2_2]
LABELS = [r'$\sigma_{k,\mathrm{NN}}$',
          r'$C_{k,\mathrm{NN}}$',
          r'$C_{\omega2,\mathrm{NN}}$']
STANDARD = [2.0, 1.0, 3.0/40.0]   # standard k-omega values

# -----------------------------------------------------------------------
# NN forward pass (vectorised over all y points)
# -----------------------------------------------------------------------
def nn_forward(x0s, x1s, m):
    """
    Evaluate model m at scaled inputs x0s, x1s (both shape [N]).
    Returns raw output [N] before clipping.
    """
    X = np.stack([x0s, x1s], axis=1)      # [N, 2]
    h = np.maximum(X @ W0[m].T + b0[m], 0.0)   # [N, 10] ReLU
    h = np.maximum(h @ W1[m].T + b1[m], 0.0)   # [N, 10] ReLU
    out = h @ W2[m] + b2[m]               # [N]
    return out

# -----------------------------------------------------------------------
# Load DNS data
# -----------------------------------------------------------------------
def load_dns():
    if not os.path.exists(MEAN_FILE):
        raise FileNotFoundError(f"DNS mean file not found: {MEAN_FILE}")
    if not os.path.exists(FLUCT_FILE):
        raise FileNotFoundError(f"DNS fluctuation file not found: {FLUCT_FILE}")

    mean  = np.loadtxt(MEAN_FILE,  comments='%')
    fluct = np.loadtxt(FLUCT_FILE, comments='%')

    # Mean file columns: y/delta, y+, U+, dU/dy (inner units), W, P
    y_delta = mean[:, 0]
    yplus   = mean[:, 1]
    Uplus   = mean[:, 2]
    dUdy_p  = mean[:, 3]   # dU+/dy+ (inner units)

    # Fluct file columns: y/delta, y+, uu, vv, ww, uv, uw, vw, k
    uv_plus  = fluct[:, 5]  # u'v'/u_tau^2  (negative in channel)
    kplus    = fluct[:, 8]  # k/u_tau^2

    return y_delta, yplus, Uplus, dUdy_p, uv_plus, kplus

# -----------------------------------------------------------------------
# Compute NN inputs from DNS data
#
# Davidson's definitions (Eq. 17):
#   x0 = nut / (y * uTau)         -- "vist_over_y"
#   x1 = tauTot / uTau^2          -- "uv_tot"
#
# From DNS (all quantities in inner units, i.e. normalised by uTau and nu):
#   y_phys   = y+ * nu / uTau
#   nut_DNS  = -u'v' / (dU/dy)   (Boussinesq, inner units: nut+ = -uv+/dUdy+)
#   tauTot   = (1 + nut+) * dUdy+  = total shear stress / uTau^2
#              (nu*dU/dy + nut*dU/dy, all non-dimensionalised)
#
# So in inner units:
#   x0 = nut+ / y+    (since nut/(y*uTau) = (nut/uTau^2)/(y/uTau) = nut+/y+
#                      ... wait: nut/(y*uTau) = nut*uTau/(y*uTau^2)... let's
#                      be careful with the non-dimensionalisation)
#
# Davidson's code uses:
#   vist_over_y = nut_t / (y_c * u_tau)   [dimensional]
#   uv_tot      = tau_tot / u_tau^2        [dimensional -> u_tau^2 / u_tau^2 = 1]
#
# In DNS inner units (nut+ = nut*uTau/nu, y+ = y*uTau/nu):
#   nut/(y*uTau) = nut+ * nu / (y+ * nu) = nut+ / y+     [dimensionless, same]
#   tauTot/uTau^2 = (nu+nut)*dU/dy / uTau^2
#                 = (1 + nut+) * dUdy+                    [dimensionless]
# -----------------------------------------------------------------------
def compute_nn_inputs(yplus, dUdy_p, uv_plus):
    """
    Compute raw (unscaled) NN input features from DNS inner-unit profiles.
    Returns x0_raw, x1_raw arrays of shape [N].
    """
    # nut+ = -u'v'+ / (dU+/dy+)
    # Use a small floor on dUdy+ to avoid division by zero at centreline
    dUdy_safe = np.where(np.abs(dUdy_p) > 1e-10, dUdy_p, 1e-10)
    nut_plus  = -uv_plus / dUdy_safe   # should be positive (u'v' is negative)
    nut_plus  = np.maximum(nut_plus, 0.0)

    # x0 = nut/(y*uTau) = nut+ / y+
    yplus_safe = np.where(yplus > 1e-10, yplus, 1e-10)
    x0_raw = nut_plus / yplus_safe

    # x1 = tauTot/uTau^2 = (1 + nut+) * dUdy+
    x1_raw = (1.0 + nut_plus) * dUdy_p
    x1_raw = np.maximum(x1_raw, 0.0)   # total stress is positive

    return x0_raw, x1_raw, nut_plus

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("Loading DNS data...")
    y_delta, yplus, Uplus, dUdy_p, uv_plus, kplus = load_dns()
    print(f"  Loaded {len(yplus)} DNS points, Re_tau = {yplus.max():.1f}")

    # Exclude the y/delta = 0 wall point (nut = 0, causes x0 = 0/0)
    # and the centreline point if present
    mask = (yplus > 0.05) & (y_delta < 0.999)
    y_delta = y_delta[mask]
    yplus   = yplus[mask]
    Uplus   = Uplus[mask]
    dUdy_p  = dUdy_p[mask]
    uv_plus = uv_plus[mask]
    kplus   = kplus[mask]

    print("Computing NN input features from DNS...")
    x0_raw, x1_raw, nut_plus = compute_nn_inputs(yplus, dUdy_p, uv_plus)

    # Clip to training range
    x0_clipped = np.clip(x0_raw, VOVY_MIN, VOVY_MAX)
    x1_clipped = np.clip(x1_raw, UV_MIN,   UV_MAX)

    # MinMax scale to [0, 1]
    x0s = (x0_clipped - VOVY_MIN) / (VOVY_MAX - VOVY_MIN)
    x1s = (x1_clipped - UV_MIN)   / (UV_MAX   - UV_MIN)

    # Fraction of points within training range
    in_range_x0 = np.mean((x0_raw >= VOVY_MIN) & (x0_raw <= VOVY_MAX)) * 100
    in_range_x1 = np.mean((x1_raw >= UV_MIN)   & (x1_raw <= UV_MAX))   * 100
    print(f"  x0 (nut/(y*uTau)) in training range: {in_range_x0:.1f}%")
    print(f"  x1 (tauTot/uTau2) in training range: {in_range_x1:.1f}%")

    # Evaluate all three NNs
    print("Evaluating NNs...")
    coeffs = []
    for m in range(3):
        raw   = nn_forward(x0s, x1s, m)
        clipped = np.clip(raw, OUT_MIN[m], OUT_MAX[m])
        coeffs.append(clipped)
        print(f"  Model {m} ({LABELS[m]}): "
              f"min={clipped.min():.4f}  max={clipped.max():.4f}  "
              f"mean={clipped.mean():.4f}")

    # -----------------------------------------------------------------------
    # Plot — reproduce Davidson Figure 5
    # Match his axes: y/delta on x-axis, coefficient on y-axis
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ['steelblue', 'tomato', 'seagreen']

    for i, ax in enumerate(axes):
        ax.plot(y_delta, coeffs[i], color=colors[i], linewidth=2,
                label='Present (OpenFOAM NN weights)')
        ax.axhline(STANDARD[i], color='k', linestyle='--', linewidth=1.2,
                   label=f'Standard k-ω = {STANDARD[i]:.4g}')
        ax.set_xlabel(r'$y/\delta$')
        ax.set_ylabel(LABELS[i])
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    axes[0].set_title(r'(a) Turbulent Prandtl number $\sigma_{k,\mathrm{NN}}$')
    axes[1].set_title(r'(b) Dissipation modifier $C_{k,\mathrm{NN}}$')
    axes[2].set_title(r'(c) Destruction modifier $C_{\omega2,\mathrm{NN}}$')

    fig.suptitle(
        r'NN coefficients evaluated on DNS data (Re$_\tau$ = 5200, Lee \& Moser 2015)'
        '\nReproducing Davidson (2026) Fig. 5 — pure NN evaluation, no CFD solver',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('NN_coefficients_from_DNS.png', dpi=150)
    print("\nSaved: NN_coefficients_from_DNS.png")

    # -----------------------------------------------------------------------
    # Also plot the raw NN inputs to show the feature distribution
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    axes2[0].semilogy(y_delta, x0_raw, 'b-', linewidth=1.5, label='Raw')
    axes2[0].axhline(VOVY_MIN, color='r', linestyle=':', linewidth=1,
                     label=f'Clip min = {VOVY_MIN:.2e}')
    axes2[0].axhline(VOVY_MAX, color='r', linestyle='--', linewidth=1,
                     label=f'Clip max = {VOVY_MAX:.3f}')
    axes2[0].set_xlabel(r'$y/\delta$')
    axes2[0].set_ylabel(r'$\nu_t / (y \cdot u_\tau)$')
    axes2[0].set_title('NN input $x_0$')
    axes2[0].legend(fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(y_delta, x1_raw, 'r-', linewidth=1.5, label='Raw')
    axes2[1].axhline(UV_MIN, color='b', linestyle=':', linewidth=1,
                     label=f'Clip min = {UV_MIN:.3f}')
    axes2[1].axhline(UV_MAX, color='b', linestyle='--', linewidth=1,
                     label=f'Clip max = {UV_MAX:.3f}')
    axes2[1].set_xlabel(r'$y/\delta$')
    axes2[1].set_ylabel(r'$\tau_{\mathrm{tot}} / u_\tau^2$')
    axes2[1].set_title('NN input $x_1$')
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    fig2.suptitle('NN input features computed from DNS (Re$_\\tau$ = 5200)',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig('NN_inputs_from_DNS.png', dpi=150)
    print("Saved: NN_inputs_from_DNS.png")

    plt.show()


if __name__ == '__main__':
    main()
