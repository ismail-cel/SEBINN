# CLAUDE.md

## Project
SEBINN — **Singularity Enriched Boundary Integral Neural Networks** for elliptic boundary value problems with corner singularities.

This repository implements a boundary-integral analogue of the SEPINN idea from:

> Hu, Jin, Zhou (2024), *Singularity Enriched Neural Networks for Boundary Integral Equations: A Dimension-Reduced Approach to Corner Singularities in Elliptic Problems*, SIAM J. Sci. Comput.

The target use case is the Laplace equation on polygonal domains with boundary singularities induced by corners.

---

## Mathematical objective

We solve the Dirichlet problem

\[
-\Delta u = 0 \quad \text{in } \Omega, \qquad u = g \quad \text{on } \partial\Omega,
\]

using a **single-layer potential** representation

\[
u(x) = \int_{\partial\Omega} G(x,y)\,\sigma(y)\,ds(y),
\]

where:

- \(G(x,y)\) is the fundamental solution of the Laplace operator,
- \(\sigma\) is the unknown boundary density,
- \(ds(y)\) is boundary arc-length measure.

The core idea is to split the density into a smooth part and a known singular part:

\[
\sigma = \sigma_w + \gamma\,\sigma_s.
\]

Here:

- \(\sigma_w \in H^{1/2}(\partial\Omega)\) is the smooth remainder,
- \(\sigma_s\) is the analytically known corner singular component,
- \(\gamma\) is a scalar stress-intensity-type coefficient learned as a trainable parameter.

For a corner of interior angle \(\omega\), the singular density is

\[
\sigma_s = -\frac{\pi}{\omega}\, r^{\pi/\omega - 1},
\]

with \(r\) the distance to the corner in the local corner coordinate system.

This method mirrors the **SEPINN** philosophy, but acts on the **boundary density** rather than the domain solution.

---

## Core model assumptions

Unless explicitly changed in a task, assume:

- PDE: Laplace equation
- Geometry: polygonal domains
- Boundary condition: Dirichlet
- Representation: single-layer potential
- Learning target: smooth density remainder \(\sigma_w\)
- Singular correction: explicit enrichment through \(\gamma \sigma_s\)
- Framework: Python + PyTorch
- Writing style for theory and paper text: precise mathematical notation suitable for **SIAM J. Sci. Comput.**

---

## Canonical formulas

Treat the following as project-level canonical objects:

1. **Density splitting theorem / ansatz**
   \[
   \sigma = \sigma_w + \gamma \sigma_s
   \]

2. **Explicit singular density formula**
   \[
   \sigma_s = -\frac{\pi}{\omega}\, r^{\pi/\omega - 1}
   \]

3. **Enriched boundary integral equation**
   The boundary integral equation obtained after substituting
   \(\sigma = \sigma_w + \gamma \sigma_s\)
   into the boundary representation and enforcing the boundary condition.

4. **\(\gamma\)-extraction formula**
   The derived formula used to identify or compute the singular amplitude coefficient.

If there is any apparent inconsistency between code and these formulas, preserve the mathematics and flag the implementation.

---

## What Claude should do in this repository

When working in this repo:

1. **Read the mathematical structure first.**
   Always preserve the density split and the role of the singular enrichment.

2. **Distinguish clearly between**
   - analytic singular structure,
   - neural approximation of the smooth remainder,
   - numerical quadrature / discretization,
   - training loss construction,
   - post-processing / reconstruction of \(u\).

3. **Use precise notation in all theory-facing outputs.**
   Prefer standard symbols such as
   \(\Omega\), \(\partial\Omega\), \(u\), \(g\), \(\sigma\), \(\sigma_w\), \(\sigma_s\), \(\gamma\), \(G(x,y)\), \(r\), \(\omega\).

4. **Keep implementation aligned with analysis.**
   Any code for quadrature, singular handling, or \(\gamma\)-learning must be explicitly justified against the mathematical derivation.

5. **Prefer modular scientific code.**
   Separate:
   - geometry / boundary parametrization,
   - kernel evaluation,
   - singular-term construction,
   - neural model definition,
   - loss assembly,
   - experiment scripts,
   - plotting / reconstruction.

---

## Repository expectations

Preferred implementation language: **Python/PyTorch**.

Suggested module responsibilities:

- `boundary/`  
  Boundary parametrization, normals, arc-length factors, corner metadata, polygon handling.

- `quadrature/`  
  Boundary quadrature, treatment of logarithmic kernel behavior, near-singular and singular integration logic.

- `singular/` or equivalent  
  Construction of \(\sigma_s\), corner-local coordinates, angle-dependent exponents, multi-corner enrichment logic.

- `models/`  
  Neural approximation for \(\sigma_w\), trainable scalar \(\gamma\), possible multi-parameter extensions.

- `training/`  
  Loss definitions, collocation or quadrature-based residuals, optimization loops, evaluation metrics.

- `reconstruction/`  
  Evaluation of \(u(x)\) in the interior or on the boundary from the learned density.

- `experiments/`  
  Reproducible numerical studies, especially corner-dominated benchmark geometries.

- `theory/`  
  Stored derivations, formulas, theorem statements, and notes synchronized with the implementation.

- `paper/`  
  Draft sections, figures, tables, and SIAM-style writing materials.

---

## Non-negotiable coding rules

- Do **not** silently change the mathematical definition of \(\sigma_s\), \(\gamma\), or the enriched ansatz.
- Do **not** replace precise math notation with vague prose in theory files.
- Do **not** merge singular and smooth components in code without preserving interpretability.
- Any new numerical trick must be documented with a brief mathematical rationale.
- Any new experiment must state:
  - geometry,
  - boundary data \(g\),
  - corner angle(s) \(\omega\),
  - loss definition,
  - training setup,
  - evaluation metric.

---

## Testing expectations

Every substantial implementation change should be accompanied by checks such as:

- correctness of the singular exponent \(\pi/\omega - 1\),
- consistency of local corner coordinates and distance \(r\),
- stability of quadrature near \(x=y\),
- correct assembly of the enriched density
  \[
  \sigma = \sigma_w + \gamma \sigma_s,
  \]
- sanity checks on learned \(\gamma\),
- reconstruction accuracy for benchmark problems with known corner behavior.

Prefer small, testable changes over broad refactors.

---

## Writing and paper guidance

Target journal: **SIAM Journal on Scientific Computing**.

When drafting text for the paper or notes:

- use formal mathematical language,
- define all symbols,
- state assumptions clearly,
- separate theorem / derivation / numerics cleanly,
- avoid overstating claims not supported by analysis or experiments.

Use notation consistently across:
- theory notes,
- code comments,
- experiments,
- manuscript sections.

---

## Project knowledge already assumed

The following are considered established references for this project:

- the full **SEPINN** paper,
- all derived formulas for:
  - \(\sigma_s\),
  - the enriched BIE,
  - \(\gamma\) extraction.

If asked to derive, verify, or implement something, use these as the primary mathematical reference points.

---

## Default startup behavior for Claude

At the beginning of a coding session:

1. Read this `CLAUDE.md`.
2. Inspect the repository structure.
3. Identify which part is currently implemented and which part is missing.
4. Summarize the next 3 highest-leverage tasks.
5. Before coding, explain the mathematical intent of the selected task.
6. Then implement in small, verifiable steps.

If the task touches singular quadrature, corner asymptotics, or \(\gamma\)-identification, prioritize mathematical correctness over speed.

---

## Default instruction style

Always:
- use precise mathematical notation,
- keep theory and implementation synchronized,
- write for a research-grade scientific computing workflow,
- assume Python/PyTorch unless explicitly told otherwise.
## Reference implementation

The file `reference/bem_pinn_nystrom_comparison.m` is the primary reference implementation for the baseline boundary-integral workflow.

Claude must use this MATLAB file as the structural template for the Python/PyTorch implementation.

When working on the codebase:
- read `reference/bem_pinn_nystrom_comparison.m` first before proposing architecture changes,
- preserve its numerical workflow unless there is a clear mathematical reason to change it,
- map its logic into Python modules rather than rewriting from scratch,
- explicitly identify which MATLAB block corresponds to each new Python file or function.

Treat the MATLAB code as the baseline for:
- boundary discretization,
- quadrature,
- kernel evaluation,
- residual construction,
- optimization / training workflow,
- interior reconstruction.

SE-BINN modification:
The MATLAB file represents the non-enriched baseline, where the density is modeled as a single smooth neural network output.

In this repository, extend that baseline to the enriched density
\[
\sigma = \sigma_w + \gamma \sigma_s,
\]
where
\[
\sigma_s = -\frac{\pi}{\omega} r^{\pi/\omega - 1}.
\]

Therefore:
- reuse the MATLAB quadrature and operator-assembly ideas as much as possible,
- keep the same boundary-integral backbone,
- modify only the density model and residual evaluation as needed for singular enrichment,
- do not discard the MATLAB structure unless explicitly instructed.

Whenever implementing a new module, first explain:
1. which part of the MATLAB file it corresponds to,
2. what is reused directly,
3. what is changed for SE-BINN,
4. how the Python version will be validated against the MATLAB baseline.
## Default geometry

Unless I explicitly say otherwise, always use the Koch geometry as the default geometry for this project.

In particular:
- treat the Koch boundary as the canonical benchmark geometry,
- do not switch to circles, squares, or generic polygons by default,
- when creating tests, demos, experiments, or examples, start from the Koch case first,
- if a new module needs geometry input, assume Koch unless another geometry is explicitly requested,
- when porting MATLAB code, preserve the Koch-based workflow and data structures as closely as possible.

If a task is ambiguous, choose Koch automatically.
## Reference MATLAB baseline

The file `reference/bem_pinn_nystrom_comparison.m` is the baseline implementation.

Claude must:
- read this file before making major implementation decisions,
- preserve its numerical workflow where possible,
- use it as the structural template for the Python/PyTorch version,
- keep Koch as the primary geometry unless explicitly told otherwise.