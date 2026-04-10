# Mellin Analysis of the Boundary Density Singularity for the Single-Layer Potential

## 1. Setting

We solve the Dirichlet problem for the Laplacian on a polygonal domain Ω:

$$-\Delta u = 0 \quad \text{in } \Omega, \qquad u = g \quad \text{on } \partial\Omega,$$

using the **single-layer potential** representation:

$$u(x) = \int_{\partial\Omega} G(x,y)\,\sigma(y)\,ds(y), \qquad G(x,y) = -\frac{1}{2\pi}\ln|x-y|.$$

Enforcing the boundary condition gives the **first-kind BIE**:

$$(V\sigma)(x) = g(x), \qquad x \in \partial\Omega, \tag{1}$$

where $V$ is the single-layer boundary integral operator defined by

$$(V\sigma)(x) = \int_{\partial\Omega} G(x,y)\,\sigma(y)\,ds(y).$$

We seek the singular structure of the density $\sigma = V^{-1}g$ near a corner
of interior angle $\omega$.

---

## 2. Key reference

Von Petersdorff & Stephan (2014), *Decomposition in regular and singular
parts (in domains with corners) and stability under perturbation of the
geometry*, Applicable Analysis **93**(10), 2158–2173.

This paper analyzes the BIE $V\psi = (1+K)g$ for the **normal derivative**
$\psi = \partial u/\partial n$. Our BIE $V\sigma = g$ has the **same operator** $V$ on
the left-hand side but a **different (simpler) right-hand side**. The Mellin
analysis of $V^{-1}$ applies identically: the singularity structure of the
density is determined by the poles of $\hat{V}^0(\lambda)^{-1}$, regardless of
whether we solve $V\sigma = g$ or $V\psi = (1+K)g$.

---

## 3. Mellin symbol of $V$ on a wedge

Near a corner with interior angle $\omega$, the boundary $\partial\Omega$ consists of
two straight edges $\Gamma^+$ and $\Gamma^-$ meeting at the vertex. A function $f$
on $\partial\Omega$ near the vertex is identified with a pair $(f^-, f^+)$ of
functions on $\mathbb{R}^+$.

Following §3 of von Petersdorff–Stephan, the single-layer operator is
decomposed as $V^* = V^0 + \ell^0$ where $\ell^0$ is a rank-1 logarithmic term.
The Mellin symbol of $V^0$ is the 2×2 matrix (their eq. after (3.3)):

$$\hat{V}^0(\lambda) = \frac{1}{\lambda \sinh(\pi\lambda)}
\begin{pmatrix}
\cosh(\pi\lambda) & \cosh((\pi-\omega)\lambda) \\
\cosh((\pi-\omega)\lambda) & \cosh(\pi\lambda)
\end{pmatrix}.$$

Its inverse (eq. (3.23) in the paper) is:

$$\hat{V}^0(\lambda)^{-1} = \frac{\sinh(\pi\lambda)}{\sinh(\omega\lambda)\sinh((2\pi-\omega)\lambda)}\, A_\alpha(\lambda),$$

where

$$A_\alpha(\lambda) = \lambda
\begin{pmatrix}
\cosh(\pi\lambda) & -\cosh((\pi-\omega)\lambda) \\
-\cosh((\pi-\omega)\lambda) & \cosh(\pi\lambda)
\end{pmatrix}$$

is entire (holomorphic everywhere).

---

## 4. Poles of $\hat{V}^0(\lambda)^{-1}$ and the singular exponents

The poles of $\hat{V}^0(\lambda)^{-1}$ come from the zeros of the denominator:

$$\sinh(\omega\lambda)\,\sinh((2\pi-\omega)\lambda) = 0.$$

These zeros occur at:

- $\lambda = \frac{k\pi}{\omega}\,i$, $\quad k \in \mathbb{Z}$ $\quad$ (from $\sinh(\omega\lambda)=0$),
- $\lambda = \frac{k\pi}{2\pi-\omega}\,i$, $\quad k \in \mathbb{Z}$ $\quad$ (from $\sinh((2\pi-\omega)\lambda)=0$).

The second family corresponds to exterior-problem singularities and has
zero residue for the interior problem (noted at the end of the proof of
Theorem 3.2 in the paper). We therefore only retain the first family.

By Lemma A.2 (inverse Mellin transform), a simple pole of
$\hat{\sigma}(\lambda)$ at $\lambda = \beta i$ in the strip
$0 < \mathrm{Im}\,\lambda < s$ corresponds to a singularity term
$c \cdot r^\beta$ in $\sigma(r)$. The Mellin convention in the paper uses
$\hat{\psi}^1_\alpha(\lambda - i)$, so the pole at $\lambda = \alpha i$
(with $\alpha = \pi/\omega$) produces an exponent:

$$\boxed{\text{leading exponent of } \sigma: \quad \alpha - 1 = \frac{\pi}{\omega} - 1.}$$

For the Koch(1) reentrant corners ($\omega = 4\pi/3$):

$$\alpha - 1 = \frac{3}{4} - 1 = -\frac{1}{4}.$$

**This confirms the exponent used in the SE-BINN code.**

---

## 5. The angular structure: same coefficient on both edges

The residue computation in eq. (3.25) of the paper gives:

$$\mathrm{Res}_{\lambda=\alpha i}\, \hat{\psi}^1_\alpha(\lambda-i) = -i\, a_0(\alpha) \begin{pmatrix} 1 \\ 1 \end{pmatrix}.$$

The vector $(1, 1)^T$ means the singular term takes the **same scalar
coefficient** $a_0(\alpha)$ on **both edges** $\Gamma^-$ and $\Gamma^+$
meeting at the corner.

In our notation, near the corner vertex:

$$\sigma(y) = a_0 \cdot r^{\pi/\omega - 1} + \sigma_{\mathrm{reg}}(y), \tag{2}$$

where $a_0$ is **the same constant on both boundary edges** adjacent to the
corner. There is **no angular dependence** (no $\sin(\pi\theta/\omega)$ or
similar factor) — the singular density is a **purely radial** function of the
distance $r = |y - v|$ from the vertex.

This is a fundamental difference from the **domain solution** singularity
$u \sim c \cdot r^{\pi/\omega} \sin(\pi\theta/\omega)$, which has explicit
angular dependence through $\sin(\pi\theta/\omega)$.

**The SE-BINN formula $\sigma_s = C \cdot r^{\pi/\omega-1}$ (radial only) is
therefore correct in its functional form.**

---

## 6. The prefactor and the role of $\gamma$

The coefficient $a_0$ in eq. (2) depends globally on the boundary data $g$
and on the geometry. It is **not** a universal constant. From eq. (3.25)–(3.26),
$a_0(\alpha)$ involves:

- the Mellin transform $\hat{g}^*(\alpha i)$ of the (localised) boundary data,
- the localization remainder $\hat{h}^*_\alpha(\alpha i)$,
- the matrix $A_\alpha(\alpha i)$ and trigonometric functions of $\omega$.

In the SE-BINN ansatz

$$\sigma = \sigma_w + \gamma \cdot \sigma_s, \qquad \sigma_s = -\frac{\pi}{\omega}\, r^{\pi/\omega - 1},$$

the trainable parameter $\gamma$ absorbs $a_0$ up to the chosen prefactor:

$$\gamma = \frac{a_0}{-\pi/\omega} = -\frac{\omega}{\pi}\, a_0.$$

The specific choice of prefactor $-\pi/\omega$ is a normalisation convention;
any nonzero prefactor would work, with $\gamma$ adjusting accordingly. The
choice $-\pi/\omega$ is natural because it makes $\sigma_s$ equal to the
$r$-derivative of the domain singular function:

$$\frac{d}{dr}\bigl[r^{\pi/\omega}\bigr] = \frac{\pi}{\omega}\, r^{\pi/\omega - 1},$$

with a sign convention.

**The prefactor does not affect correctness, only the numerical scale of
the learned $\gamma$.** The SE-BINN code's choice is acceptable.

---

## 7. Nonzero $\gamma^*$ for smooth boundary data

From the Mellin analysis, $a_0$ is **generically nonzero** even when $g$ is
a smooth (polynomial) function. This is because:

1. The operator $V^{-1}$ introduces singularities through its Mellin
   symbol, whose poles at $\lambda = k\pi i/\omega$ persist regardless of the
   smoothness of $g$.

2. For $g(x,y) = x^2 - y^2$, the localised boundary data $g^1_\alpha$
   near each reentrant corner has a nonzero Mellin transform at
   $\lambda = \alpha i = (3/4)i$. This feeds into $a_0$ via eq. (3.25).

3. The BEM density computed numerically shows $r^{-1/4}$ spikes at every
   reentrant corner, confirming $a_0 \ne 0$.

**Therefore $\gamma^* \ne 0$ for the benchmark $g = x^2 - y^2$ on Koch(1).**
The enrichment should be capturing a real singularity.

---

## 8. Why per-corner $\gamma_c$ matters

The coefficient $a_0$ at each reentrant corner depends on the **global**
boundary data $g$. For $g(x,y) = x^2 - y^2$:

- This function has 2-fold symmetry ($g(x,y) = g(-x,-y)$), not 6-fold.
- The 6 reentrant corners of Koch(1) are related by 60° rotations, but
  $g$ is only invariant under 180° rotation.
- Therefore the 6 coefficients $a_0^{(c)}$ split into 3 pairs of equal
  values, with different magnitudes across pairs.

A single shared $\gamma$ must compromise across corners with different
$a_0^{(c)}$ values, reducing the effectiveness of the enrichment.

**Using `per_corner_gamma=True` (6 separate $\gamma_c$ values) should
significantly improve the enrichment effect for $g = x^2 - y^2$.**

---

## 9. Summary of conclusions

| Aspect | Status | Detail |
|---|---|---|
| Exponent $\pi/\omega - 1$ | **Correct** | Confirmed by poles of $\hat{V}^0(\lambda)^{-1}$ at $\lambda = k\pi i/\omega$ |
| No angular dependence | **Correct** | Residue vector is $(1,1)^T$: same coefficient on both edges |
| Prefactor $-\pi/\omega$ | **Acceptable** | Absorbed by learnable $\gamma$; any nonzero prefactor works |
| $\gamma^* = 0$ for smooth $g$? | **No** — $\gamma^* \ne 0$ | $V^{-1}$ creates singularities even from smooth data |
| Single shared $\gamma$ | **Suboptimal** | Different corners have different $a_0^{(c)}$; use per-corner $\gamma_c$ |

---

## 10. Revised diagnosis of the enrichment failure

Given that:
- the σ\_s formula is mathematically correct (exponent, no angular factor),
- γ\* ≠ 0 for the benchmark problem,
- the enrichment is properly wired into the loss,

the most likely reasons the enrichment has little visible effect are:

1. **Single shared γ across 6 corners with non-uniform $a_0^{(c)}$.**
   The optimal shared γ is a weighted average that doesn't perfectly
   match any individual corner, reducing marginal improvement.

2. **The network σ\_w partially absorbs the mild singularity.**
   The exponent −1/4 is a weak singularity (r^{−1/4} ∈ L^2). A 4×80
   tanh network on the 1D boundary can approximate it reasonably well.
   The enrichment helps at the margin but doesn't produce a dramatic
   error reduction.

3. **γ\_init = 0 creates optimisation inertia.**
   The network starts fitting the full density (including the singular
   part) from iteration 1. By the time γ has significant gradient,
   σ\_w has already partially compensated, creating a flat loss direction
   for γ.

**Recommended immediate actions:**
1. Set `per_corner_gamma=True` (6 independent γ\_c).
2. Use `gamma_init` matching the estimated scale of $a_0$ (e.g., from
   a rough BEM-based estimate), or at least a nonzero value like 1.0.
3. Monitor γ\_c values during training alongside loss.
4. Compare SE-BINN vs plain BINN (γ fixed at 0) on the same benchmark to
   measure the exact improvement from enrichment.

---

## References

- T. von Petersdorff, E.P. Stephan, *Decomposition in regular and singular
  parts (in domains with corners) and stability under perturbation of the
  geometry*, Applicable Analysis **93**(10), 2158–2173, 2014.
- M. Costabel, E. Stephan, *Boundary integral equations for mixed boundary
  value problems in polygonal domains and Galerkin approximation*, Banach
  Center Publications **15**, 175–251, 1985.
- Hu, Jin, Zhou, *Singularity enriched PINNs*, SIAM J. Sci. Comput., 2024.
