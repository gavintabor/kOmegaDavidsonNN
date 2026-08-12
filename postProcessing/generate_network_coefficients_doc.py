"""
Extracts the hardcoded NN weights/biases/clipping bounds directly from
src/kOmegaDavidsonNN.C via regex parsing (not manual retyping -- avoids
transcription error the same way this repo's other field-parsing tools
do) and formats them as markdown tables in NETWORK_COEFFICIENTS.md.

Run from anywhere; paths are resolved relative to this script's location
(postProcessing/), not the working directory:

    python3 postProcessing/generate_network_coefficients_doc.py

Regenerate this after any change to the hardcoded weights in
src/kOmegaDavidsonNN.C (e.g. retraining) rather than hand-editing
NETWORK_COEFFICIENTS.md.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "kOmegaDavidsonNN.C"
OUT = REPO_ROOT / "NETWORK_COEFFICIENTS.md"

text = SRC.read_text()

# Isolate the region between the clipping-bounds comment and the cell loop
# comment, so we don't accidentally match unrelated "static const scalar"
# declarations elsewhere in the file (e.g. Cmu_, beta_ constructor defaults).
start = text.index("// Clipping bounds (training data ranges)")
end = text.index("// Cell loop")
region = text[start:end]

_NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_scalar(name):
    m = re.search(rf"static const scalar {name}\s*=\s*({_NUM})\s*;", region)
    return float(m.group(1))


def parse_vector(name, n):
    m = re.search(rf"static const scalar {name}\[{n}\]\s*=\s*\{{(.*?)\}};", region, re.S)
    vals = [float(v) for v in re.findall(_NUM, m.group(1))]
    assert len(vals) == n, f"{name}: expected {n}, got {len(vals)}"
    return vals


def parse_matrix(name, rows, cols):
    m = re.search(rf"static const scalar {name}\[{rows}\]\[{cols}\]\s*=\s*\{{(.*?)\}};", region, re.S)
    body = m.group(1)
    # Each row is its own {...} group
    row_texts = re.findall(r"\{([^{}]*)\}", body)
    assert len(row_texts) == rows, f"{name}: expected {rows} rows, got {len(row_texts)}"
    matrix = []
    for rt in row_texts:
        vals = [float(v) for v in re.findall(_NUM, rt)]
        assert len(vals) == cols, f"{name} row: expected {cols}, got {len(vals)}"
        matrix.append(vals)
    return matrix


# ---------------------------------------------------------------------------
# Clipping bounds
# ---------------------------------------------------------------------------
clip = dict(
    voyMin=parse_scalar("voyMin"), voyMax=parse_scalar("voyMax"),
    uvMin=parse_scalar("uvMin"), uvMax=parse_scalar("uvMax"),
    sigmakMin=parse_scalar("sigmakMin"), sigmakMax=parse_scalar("sigmakMax"),
    CkMin=parse_scalar("CkMin"), CkMax=parse_scalar("CkMax"),
    Comega2Min=parse_scalar("Comega2Min"), Comega2Max=parse_scalar("Comega2Max"),
)

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
NETWORKS = [
    ("sk", "sigmakNN", "prand_k", "σ<sub>k,NN</sub>"),
    ("ck", "CkNN", "c_k", "C<sub>k,NN</sub>"),
    ("co", "Comega2NN", "c_omega_2", "C<sub>ω2,NN</sub>"),
]

nets = {}
for prefix, field, davidson_name, tex in NETWORKS:
    nets[prefix] = dict(
        field=field, davidson_name=davidson_name, tex=tex,
        W1=parse_matrix(f"{prefix}_W1", 10, 2),
        b1=parse_vector(f"{prefix}_b1", 10),
        W2=parse_matrix(f"{prefix}_W2", 10, 10),
        b2=parse_vector(f"{prefix}_b2", 10),
        W3=parse_vector(f"{prefix}_W3", 10),
        b3=parse_scalar(f"{prefix}_b3"),
    )

n_params = 10*2 + 10 + 10*10 + 10 + 10 + 1
print(f"Parsed {len(nets)} networks, {n_params} parameters each, "
      f"{n_params*len(nets)} total. Clipping bounds: {len(clip)} values.")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def fmt(v):
    return f"{v:.8g}"


def vector_table(vals, header="i"):
    lines = ["| " + header + " | " + " | ".join(str(i) for i in range(len(vals))) + " |",
             "|" + "---|" * (len(vals) + 1)]
    lines.append("| value | " + " | ".join(fmt(v) for v in vals) + " |")
    return "\n".join(lines)


def matrix_table(mat, row_label="i", col_label="j"):
    ncols = len(mat[0])
    header = f"| {row_label}\\{col_label} | " + " | ".join(str(j) for j in range(ncols)) + " |"
    sep = "|" + "---|" * (ncols + 1)
    rows = []
    for i, row in enumerate(mat):
        rows.append(f"| **{i}** | " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


out = []
out.append("# Network coefficients: weights, biases, and clipping ranges\n")
out.append(
"""This document is the full, authoritative reference for the three hardcoded
neural networks in `src/kOmegaDavidsonNN.C` — σ<sub>k,NN</sub>, C<sub>k,NN</sub>, and
C<sub>ω2,NN</sub>, the coefficient fields that replace three constants in the
Wilcox k-ω model (see the main [README](README.md) for the physics). All
values below are extracted directly from the C++ source (not retyped by
hand) by `postProcessing/generate_network_coefficients_doc.py`, so they're
guaranteed to match what's actually compiled into the library. If the
weights are ever retrained/re-extracted, rerun that script to regenerate
this file rather than hand-editing it.

**Architecture**: all three networks are identical in shape — 2 inputs,
two hidden layers of 10 neurons each with ReLU activations, one linear
output (2→10→10→1), matching Davidson (2026)'s own PyTorch model. Each
has 141 weights + 10 biases (layer 1) + 100 weights + 10 biases (layer 2)
+ 10 weights + 1 bias (output) = 151 parameters; 453 across all three.

**Inputs** (both networks take the same two local dimensionless features):

- x₀ = ν_t/(y·u_τ), clipped to [{voyMin}, {voyMax}] before use
- x₁ = τ_tot/u_τ², clipped to [{uvMin}, {uvMax}] before use

then linearly rescaled to [0, 1] over that clipped range before being fed
to the network (`x0`/`x1` in the code).

**Forward pass** (per network; the source names each array `<prefix>_W1`,
`<prefix>_b1`, `<prefix>_W2`, `<prefix>_b2`, `<prefix>_W3`, `<prefix>_b3`,
with `<prefix>` one of `sk`/`ck`/`co` for σ<sub>k,NN</sub>/C<sub>k,NN</sub>/C<sub>ω2,NN</sub>
respectively — same names used in the per-network tables below):

```
h1[i] = ReLU( W1[i]·(x0,x1) + b1[i] )        for i in 0..9
h2[i] = ReLU( W2[i]·h1      + b2[i] )        for i in 0..9
out   =        W3  ·h2      + b3
```

The raw output is then clipped to that network's own training range (below)
before being blended into the field via the EWMA smoothing described in
the README.

**Clipping ranges**

| Quantity | Min | Max |
|---|---|---|
| x₀ = ν_t/(y·u_τ) (input) | {voyMin} | {voyMax} |
| x₁ = τ_tot/u_τ² (input) | {uvMin} | {uvMax} |
| σ<sub>k,NN</sub> (output) | {sigmakMin} | {sigmakMax} |
| C<sub>k,NN</sub> (output) | {CkMin} | {CkMax} |
| C<sub>ω2,NN</sub> (output) | {Comega2Min} | {Comega2Max} |
""".format(**{k: fmt(v) for k, v in clip.items()}))

for prefix, net in nets.items():
    out.append(f"\n## {net['tex']} network (`{net['field']}`, Davidson's `{net['davidson_name']}`)\n")

    out.append(f"\n### Layer 1 — `{prefix}_W1` (10×2) and `{prefix}_b1` (10)\n")
    out.append("\nWeights (row *i* = neuron *i*, columns = inputs x₀, x₁):\n")
    header = f"| neuron | x₀ weight | x₁ weight | bias |\n|---|---|---|---|\n"
    rows = "\n".join(
        f"| {i} | {fmt(net['W1'][i][0])} | {fmt(net['W1'][i][1])} | {fmt(net['b1'][i])} |"
        for i in range(10)
    )
    out.append(header + rows + "\n")

    out.append(f"\n### Layer 2 — `{prefix}_W2` (10×10) and `{prefix}_b2` (10)\n")
    out.append("\nWeights (row *i* = neuron *i* in layer 2, columns *j* = input from layer-1 neuron *j*):\n")
    out.append(matrix_table(net["W2"]) + "\n")
    out.append("\nBiases:\n")
    out.append(vector_table(net["b2"], header="neuron") + "\n")

    out.append(f"\n### Output layer — `{prefix}_W3` (10) and `{prefix}_b3` (scalar)\n")
    out.append("\nWeights (one per layer-2 neuron) and the single output bias:\n")
    out.append(vector_table(net["W3"], header="neuron") + "\n")
    out.append(f"\n`{prefix}_b3` = {fmt(net['b3'])}\n")

doc = "\n".join(out)
doc = re.sub(r"\n{3,}", "\n\n", doc)  # collapse the section-join blank runs
OUT.write_text(doc)
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
