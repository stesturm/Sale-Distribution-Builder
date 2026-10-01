# Sale-Distribution-Builder
An implementation of the projection of distributions specified via distribution builder for the problem of timing an asset sale. Details can be found in the paper "Optimal Selling of Defaultable Assets using the Distribution Builder" by S. Jin and S. Sturm, https://arxiv.org/abs/2608.21716. Created with the help of ChatGPT 5.6(Sol) based on an earlier implementation of the distribution builder by Benjamin Rajotte.

# Distribution Builder — optimal selling projection

## Purpose

The application provides a distribution-builder interface for choosing a distribution of the price at which a defaultable asset is sold.

The user draws a target law μ. The program tests its scaled mean and, when appropriate, computes an attainable or optimal attainable law ν using the extended-f-divergence framework of Jin and Sturm, [*Optimal Selling of Defaultable Assets using the Distribution Builder*](\url{https://arxiv.org/abs/2608.21716).

The numerical result is a projected **distribution**. The program does not construct or simulate the corresponding Azéma–Yor selling rule.

## Market model

The first implementation uses shifted arithmetic Brownian motion

$$
R_t=r_0+bt+\sigma B_t,
$$

absorbed when it reaches zero. There is no fixed calendar horizon. The paper's formal admissibility condition permits sale at ruin, so a mass at zero is possible.

The scale function is normalized by $S(r_0) = 0$:

$$
S(x)=
 \begin{bmatrix}
x-r_0, & b=0,\\
\frac{\sigma^2}{2b}
\left[1-\exp\left(-\dfrac{2b(x-r_0)}{\sigma^2}\right)\right],&b\ne0.
\end{bmatrix} 
$$

For a law μ, define its scaled mean

$$
m_\mu=\int S(x)\,\mu(dx).
$$

For this process:

- if $b \leq 0$, $\mu$ is attainable before or at ruin exactly when $m_\mu \leq 0$;
- if $b > 0$, $\mu$ is attainable exactly when $m_\mu = 0$;
- the projection problems below use the boundary $m_\nu = 0$.

For the paper's example $R_t = 2 + B_t$ for a Brownian motion $B$, $S(x) = x − 2$, so the scaled mean is simply $\int x \mu(dx) - 2$.

## Which projection is solved?

### Positive target scaled mean: B′

If $m_\mu > 0$, the target is neither attainable nor super-attainable. The program finds the closest law ν satisfying

$$
\int S(x)\,\nu(dx)=0.
$$

There is no stochastic-dominance constraint. For divergences with finite recession constant, the projected law may introduce mass at the default point zero.

### Zero target scaled mean

If $m_\mu = 0$ within numerical tolerance, the target is already attainable and first order stochastic dominance-maximal under the scaled-mean constraint. It is returned unchanged.

### Negative target scaled mean: compact CM′

If $m_\mu < 0$, the program finds the closest zero-scaled-mean law that first-order stochastically dominates the target:

$$
\int S(x)\,\nu(dx)=0,
\qquad F_\nu(x)\le F_\mu(x)\quad\text{for every grid point }x.
$$

The unbounded problem need not have an optimizer: improving mass can escape to infinity. The application therefore solves the paper's compact problem CM′. The visible upper sale-price range is the explicit support cap M; it is displayed in the result and can be enlarged with **Extend upper**.

## Finite-dimensional program

The completed builder contains N equally likely states. Repeated sale-price columns are coalesced into support points $x_i$ with target masses $p_i$.

The optimizer chooses candidate masses $q_i$ and minimizes

$$
\sum_{i:p_i>0}p_i f\Bigl(\frac{q_i}{p_i}\Bigr)
+f'(\infty)\sum_{i:p_i=0}q_i
$$

subject to

$$
q_i\ge0,\qquad \sum_iq_i=1,
\qquad \sum_iS(x_i)q_i=0.
$$

CM′ additionally imposes the cumulative linear inequalities

$$
\sum_{i\le k}q_i\le\sum_{i\le k}p_i.
$$

For superlinear divergences, candidate mass is fixed to zero wherever the target has no mass. For finite-recession divergences, B′ may add the endpoint 0 and CM′ may add the disclosed endpoint M.

## Available distances

### Kullback–Leibler

Uses the classical relative entropy

$$
D_{KL}(\nu\|\mu)=\sum_iq_i\log\Bigl(\frac{q_i}{p_i}\Bigr).
$$

Its recession constant is infinite, so ν must be absolutely continuous with respect to μ: no new sale-price atom can be introduced.

### Rényi, configurable order α

When Rényi is selected, an order field appears. Choose $0.1 ≤ \alpha ≤ 5$. At $\alpha = 1$ the program calculates the exact KL limit; orders other than 1 within 0.01 of 1 are rejected for numerical stability (select 1 instead). The solver reports

$$
D_\alpha(\nu\|\mu)=\frac{1}{\alpha-1}\log\left(\sum_{i:p_i>0}p_i\Bigl(\frac{q_i}{p_i}\Bigr)^\alpha\right).
$$

For $\alpha > 1$ it minimizes the convex power sum inside the logarithm; it is superlinear, so **new sale-price outcomes are forbidden**. At $\alpha = 2$ this is the original order-two implementation. For $0 < \alpha < 1$ it minimizes the convex *negative* power sum. Its recession constant is zero, so the extended divergence **can add** default mass in B′ or upper-cap mass in CM′. At $\alpha = 1$ the KL limit is solved directly, rather than dividing by $\slpha - 1$.

### Squared Hellinger

Uses

$$
\frac12\sum_i(\sqrt{q_i}-\sqrt{p_i})^2.
$$

The recession constant is finite, so the extended divergence permits the appropriate new endpoint mass.

### Total variation

Uses

$$
\frac12\sum_i|q_i-p_i|.
$$

This is solved as a linear program. It also permits endpoint mass and may have multiple optimizers. The interface uses a deterministic secondary rule: **among TV minimizers, choose the solution with the smallest sum of squared changes in grid-point probabilities**. This secondary problem is solved on the optimal-TV face. It is not described as the “largest” law in first order stochastic dominance: distinct zero-scaled-mean laws cannot strictly dominate one another, because the scale function is strictly increasing. Thus all zero-scaled-mean feasible laws are first order stochastic dominance-maximal, but there need not be a first order stochastic dominance-greatest TV minimizer.

## Numerical method and checks

- With moderately spread scales, feasibility is checked by SciPy HiGHS; KL, Rényi and Hellinger use constrained SLSQP and a convex optimality check. Hellinger and Rényi orders below 1 receive positive-density feasible initial guesses.
- **Numerically stiff drift:** when the scale function spans large magnitudes, optimize in variables zᵢ = qᵢ max(1, |S(xᵢ)|). This keeps both moment-equation coefficient rows bounded by one, even when the required qᵢ are extremely small. A cutting-plane linear program minimizes piecewise-linear lower approximations of each convex divergence objective. Its evaluated objective minus the LP lower bound certifies the reported answer to the configured tolerance. If the direct solver fails its check at an intermediate drift, it also retries this rescaled method.
- Total variation uses a scale-rescaled LP with absolute-value variables and a second cutting-plane LP to select the minimum-squared-mass-change solution on its minimum-TV face. The tie-break gap is checked rather than silently accepting a solver-dependent vertex.
- Every smooth projection is checked against either the direct convex optimality bound or the cutting-plane lower-bound gap; a failure is not presented as a valid projection.
- Constraints are linear: probability normalization, zero scaled mean and, for CM′, cumulative FOSD inequalities.
- Every returned result is checked for non-negativity, normalization, scaled-mean residual and stochastic-dominance residual. The zero-mean shortcut uses a weighted numerical error bound rather than the largest scale value in the support; a nonzero mean must not be returned as a zero-distance solution.
- Infeasible support is reported rather than silently relaxing constraints. If the scale exponent itself exceeds floating-point range, the model reports that limitation rather than clipping the scale function and silently changing the problem.

## Reading the chart

- The solid green shape is the user-drawn target law.
- The orange outline is the projected law.
- Vertical height is shown in state-equivalent units: a projected probability q has height qN and can therefore be non-integer.
- The result displays both the projected scaled-mean residual and the largest change in probability at any support point. Very small new endpoint masses use scientific notation. Under strong drift relative to volatility, the scale function may be enormous and a genuinely necessary mass at zero or the cap can be smaller than a pixel: the UI says when the difference is below chart resolution. A calculated distance that rounds to zero is shown as **approximately zero, below numerical precision**, not as exact equality of distributions.
- A displayed endpoint mass is new mass at zero or at the compact support cap, not a new target state selected by the user.

## Scope and limitations

1. The builder discretizes the target law, so increasing the number of states and refining the payoff range can improve resolution.
2. CM′ is a compact-support problem. Changing the visible upper cap changes the mathematical feasible set and can change the optimizer.
3. KL and Rényi with α > 1 can be infeasible if the original target support alone cannot achieve zero scaled mean. Rényi with α < 1 permits endpoint mass.
4. The numerical projection constructs ν only. Implementing an actual stopping policy requires the Azéma–Yor barriers and, for negative scaled mean, the paper's preliminary hitting-and-restart construction.
5. Attainability is before or at ruin with no fixed sale deadline. Adding a finite horizon would be a different stopping problem.
