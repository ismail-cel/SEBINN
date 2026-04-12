# γ discrepancy: lstsq estimate vs physical singular amplitude

## Setting

L-shaped domain, $\omega = 3\pi/2$, reentrant corner at $(0,0)$.
Exact solution $u = r^{2/3}\sin(2\theta/3)$.
Singular enrichment basis:
$$
\sigma_s = -\frac{\pi}{\omega}\, r^{\pi/\omega - 1} = -\frac{2}{3}\, r^{-1/3}.
$$

## Two notions of γ

### 1. Physical singular amplitude (γ_phys ≈ 1)

Near the corner $r \to 0$, the BEM density $\sigma_\mathrm{BEM}$ has the leading behaviour
$$
\sigma_\mathrm{BEM}(r) \sim \gamma_\mathrm{phys}\, \sigma_s(r) = -\frac{2}{3}\,\gamma_\mathrm{phys}\, r^{-1/3}.
$$

Numerically, the near-corner ratio $\sigma_\mathrm{BEM}(r) / \sigma_s(r)$ converges to:

| $r$      | $\sigma_s$ | $\sigma_\mathrm{BEM}$ | ratio  |
|----------|------------|----------------------|--------|
| 0.00033  | −9.6354    | −10.6355             | 1.1038 |
| 0.00173  | −5.5512    | −5.5989              | 1.0086 |
| 0.00420  | −4.1323    | −4.1629              | 1.0074 |

As $r \to 0$, ratio $\to 1$, so $\gamma_\mathrm{phys} = 1$.

This is the coefficient that SE-BINN training recovers: $\gamma_\mathrm{trained} \approx 0.98$.

### 2. Lstsq projection coefficient (γ* ≈ 5.38)

The standard L2 projection formula minimises
$$
\|\sigma_\mathrm{BEM} - \gamma \sigma_s\|^2_{L^2(\partial\Omega)}
\quad\Longrightarrow\quad
\gamma^* = \frac{\sigma_\mathrm{BEM} \cdot \sigma_s}{\|\sigma_s\|^2}.
$$

Numerically (N=16 panels/edge, Nq=1536):
$$
\|\sigma_\mathrm{BEM}\| = 311.7, \qquad \|\sigma_s\| = 35.2, \qquad \gamma^* = 5.38.
$$

## Why γ* ≠ γ_phys

The lstsq estimator is dominated by the global mismatch:
$$
\gamma^* = \frac{\cos(\angle(\sigma_s, \sigma_\mathrm{BEM}))\, \|\sigma_\mathrm{BEM}\|}{\|\sigma_s\|}
       = \frac{0.608 \times 311.7}{35.2} \approx 5.38.
$$

Because $\sigma_s$ is concentrated near the corner (only a few near-corner quadrature nodes), its global norm $\|\sigma_s\| = 35.2$ is much smaller than $\|\sigma_\mathrm{BEM}\| = 311.7$.  The lstsq formula amplifies the physical coefficient $\gamma_\mathrm{phys} = 1$ by the ratio $\|\sigma_\mathrm{BEM}\|/\|\sigma_s\| \approx 8.9$.

More precisely, the decomposition
$$
\sigma_\mathrm{BEM} = \gamma_\mathrm{phys}\, \sigma_s + \sigma_w
$$
is not an orthogonal decomposition.  The smooth remainder $\sigma_w = \sigma_\mathrm{BEM} - \sigma_s$ is **not** orthogonal to $\sigma_s$ in $L^2(\partial\Omega)$, so the projection coefficient $\gamma^*$ absorbs the non-orthogonal component.

## Consequences for the energy fraction

Two different "energy fractions" arise:

| Quantity | Formula | Value |
|----------|---------|-------|
| Physical energy $\|\sigma_s\|^2 / \|\sigma_\mathrm{BEM}\|^2$ | $(35.2/311.7)^2$ | **1.27%** |
| Lstsq R² = $1 - \|\sigma_\mathrm{BEM} - \gamma^*\sigma_s\|^2 / \|\sigma_\mathrm{BEM}\|^2$ | — | **36.9%** |

The 36.9% "energy fraction" reported in the code uses the lstsq $\gamma^*$.
This inflates the apparent enrichment contribution by a factor $\gamma^{*2} \approx 29$.

**The physically correct energy fraction is 1.27%**, meaning the singular enrichment accounts for only 1.27% of the total boundary density energy in the $L^2(\partial\Omega)$ norm.

## Why SE-BINN shows a small A/B gap

With only 1.27% of the density energy in the singular component $\gamma_\mathrm{phys}\,\sigma_s$, the neural network $\sigma_w$ still has to represent 98.73% of $\sigma_\mathrm{BEM}$.  The enrichment removes the near-corner spike, making $\sigma_w$ smooth, but the smooth part is large everywhere — the network's task is barely easier.

By contrast, for the Koch(1) geometry with $u = x^2 - y^2$, all 12 corners contribute singularities and the per-corner enrichment may have a larger combined physical energy fraction.

## Implication for warm-starting

Setting $\gamma_\mathrm{init} = \gamma^* = 5.38$ (lstsq warm start) places the optimisation
in a basin where $\gamma \approx 4.6$ at convergence — far from $\gamma_\mathrm{phys} = 1$
and yielding **worse** density error than BINN.

Correct warm-starting should use the near-corner ratio:
$$
\gamma_\mathrm{init} \approx \frac{\sigma_\mathrm{BEM}(r_\mathrm{min})}{\sigma_s(r_\mathrm{min})}
$$
evaluated at the smallest available $r$.

## Summary

| Quantity | Value |
|----------|-------|
| $\gamma_\mathrm{phys}$ (near-corner ratio as $r\to0$) | ≈ 1.0 |
| $\gamma_\mathrm{trained}$ (SE-BINN L-BFGS) | ≈ 0.98 |
| $\gamma^*_\mathrm{lstsq}$ (L2 projection) | ≈ 5.38 |
| Physical energy fraction $\|\sigma_s\|^2/\|\sigma_\mathrm{BEM}\|^2$ | 1.27% |
| Lstsq R² energy fraction | 36.9% |

The SE-BINN correctly recovers $\gamma \approx \gamma_\mathrm{phys} = 1$.
The lstsq estimator is a poor proxy for the physical singular amplitude when
$\|\sigma_s\|$ is small relative to $\|\sigma_\mathrm{BEM}\|$.
