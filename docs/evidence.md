# Evidence: keeping a campaign after the data is gone

A night of recording is large and a likelihood factor is small. If the factor is
sufficient, the recording can be archived and the campaign still sampled.

---

## A campaign, after the recordings are gone

Every engine [on the other pages](inference.md) holds the data. A multi-year
campaign cannot: the recordings
are archived, or deleted, long before the last night is observed. What survives
is a **memory** — one epoch's data compressed into a factor of the campaign
likelihood, accumulated as they arrive.

`compress_linear` does the compressing. For a model affine in every global
latent with Gaussian noise, the factor it returns is a *sufficient statistic*:
no anchor, no validity region, order-invariant, exact. Per-epoch nuisances are
integrated out right there, analytically. `BayesMemory` accumulates the factors
by QR in square-root information form, and applies the prior **exactly once**,
at the end — which is why a stored term carries no prior at all and a tempered
one is refused by name.

```python
from rheplicant.inference import BayesMemory, Factorization, compress_linear

memory = BayesMemory(Factorization(space))
for night in nights:                       # one night, then archive the recording
    memory = memory.remember(
        compress_linear(
            design=night.design, observed=night.data, noise_std=night.sigma,
            shapes=shapes, epoch_id=night.data_hash,
        )
    )
mcmc.run(key)                              # kernel over memory.to_numpyro_model()
print(memory.audit())
```

A sketch, not a runnable block: `space`, `nights`, `shapes`, `mcmc` and `key`
are yours. `examples/` has no campaign demo yet; `tests/evidence/` runs the real
thing.

`Latent(scope=...)` is what makes the loop possible — `"global"` for the
quantities sampled against the whole campaign, `"per_epoch"` for the nuisances
integrated away inside one, `"linked"` for a Markov chain across them.
`Factorization` partitions the space by scope and exposes the global view;
`memory.to_numpyro_model()` opens a sample site per global latent and adds the
accumulated factor. There is no pipeline and no `observed=` anywhere in that
call, because the terms already absorbed the forward evaluation, the data and
the noise. `save_memory` / `load_memory` put the whole campaign on disk and get
it back.

The evidence layer needs float64 — a stored factor's offset is the
time-bandwidth product, ~7.2e11 for one RHINO night, against a difference of
~1e5, which float32 annihilates rather than merely rounds. Set
`jax.config.update("jax_enable_x64", True)`, or run under `JAX_ENABLE_X64=1`.

:::{danger}
**Do not point `fisher_information` at `memory.log_posterior`.** It type-checks,
runs, and returns a matrix. `fisher_information` forms `J^T N^-1 J` from
`jacfwd` of a *forward function*, and a log-density is a scalar, so `J` is
`(1, n)` and the product is a rank-1 outer product — one number's worth of
information, and at the mode that number is the gradient, which is zero.

Measured on the eight-epoch memory in `tests/evidence/test_memory_numpyro.py`,
evaluated at its own mode: the returned 2×2 matrix has eigenvalues
`[3.7e-44, 1.3e-25]` and rank 1, `parameter_covariance` inverts it to a matrix
with **negative** diagonal (≈ −4.7e43), and `sigma("x")` comes back
`[nan, nan]`. **No exception is raised anywhere in that chain.**

Use `memory.fisher()`, which is `sum_e R_e^T R_e` with named rows, permuted into
flatten order. On the same memory it gives `sigma = [0.0143, 0.0134]`, matching
the NUTS posterior's standard deviations to better than 3 %.

`GradientCalibrator` and `NeuralPosterior` do not apply either — they consume
predictions and simulations, and a memory has neither. Fitting or simulating
again means going back to the data, which is the thing that is gone.
:::

### When the model is not linear

`compress_linear` needs the prediction to be affine in every global latent. When
it is not — a running spectral index, a beam shape, anything with a product in
it — expand the prediction in a **reduced basis** instead and store the
coefficients:

```python
from rheplicant.inference import (
    basis_fidelity, build_reduced_basis, compress, score_directions,
)

basis = build_reduced_basis(
    space, pipeline, state, noise=noise, bank=prior_draws, n_basis=6,
    support={"t21_depth": (-0.5, 0.1), ...},   # the region the bank populated
)
term = compress(
    basis=basis, observed=night.data, noise=noise, epoch_id=night.data_hash,
)
memory = memory.remember(term)
```

`n_basis` is refused above the whitened bank's **numerical rank** — 6 for the
four-latent fixture in `tests/evidence/rhino_bank.py`, and asking for more is an
exception rather than a weaker answer. Above that cut the retained Gram matrix
is singular in float64 and `c^T G c` returns a finite, occasionally negative
number instead of raising, which is the failure the refusal is placed in front
of.

The dictionary is seeded with `dmu/dtheta_j` for every named latent, and that is
not an optimisation. A plain SVD of a bank of prior draws orders modes by
prior-induced amplitude, and at RHINO's band the foreground sits three orders
above the 21 cm trough: measured, the residual fraction of the `t21_depth` score
direction against an unseeded basis is **0.562 at `n_S = 3`** and 1.7e-4 at 4,
while a richer pre-planning bank measured 0.3147 at 3 and 0.0000 only at 13.
Seeded, it is **1.5e-16 at `n_S = 3`** — a complete repair rather than an
improvement, and one that does not depend on guessing `n_S` large enough. Check
it directly, and get a *name* rather than a scalar:

```python
scores = score_directions(space, pipeline, state)
basis_fidelity(basis, scores).refuse_above(0.01)
```

A basis that has deleted a direction is otherwise clean throughout — residual
chi-square, conditioning and bank reproduction all normal — while the stored
term's Fisher has a collapsed eigenvalue and its marginal reverts toward the
prior.

`memory.audit(at=values)` reports `bias_over_sigma` per named direction: the
theta-gradient of the compression error, divided by the campaign's own width.
The gradient and not the magnitude, because a constant offset has exactly zero
effect on a posterior. One basis serving every epoch makes the error coherent,
so the bias is N-independent while the width falls as `N^-1/2` and the ratio
grows as `sqrt(N)` — measured 1.571e-11 over four epochs and 6.286e-11 over
sixty-four. A ratio that is comfortable at N = 10 need not be at N = 1000. Pass
`bias_tolerance=` to make it a refusal; directions the campaign does not yet
constrain are listed under `unconstrained` rather than refused, since their
ratio is `0/0`.

Under `RadiometerNoise` the covariance is frozen at the basis's reference
prediction, and what that cost is measured per epoch rather than argued:
`term.frozen_noise_residual` is the worst gap in nats over the declared support,
and `frozen_tolerance=` refuses above a declared one. Its remedy is a narrower
support or a re-anchored basis — never a larger `n_S`, which is the remedy for
the *projection* error the bias budget measures, which is why the two are stored
as separate numbers rather than summed into one.

`RawLikelihood` is the oracle the tiers are validated against. It keeps the raw
data and a live forward model, so it belongs in a test and not in a campaign,
and it refuses to be remembered or archived by name.
