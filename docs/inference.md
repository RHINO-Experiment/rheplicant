# Inferring anything: parameter spaces

A digital twin is only half a research tool. The other half is running it
backwards: given data, what were the sky, the beam, the gain? RHEPLICANT does
that through one object — a **parameter space** — that every inference engine
reads: the two calibrators, the NumPyro bridge, Fisher forecasting, and the
conjugate-Gaussian solver.

The design question it answers is not "how do I fit a parameter", which JAX
already answers. It is: **what if the thing you want to infer is not a number
the model happens to store?** A beam is described by a width and a pointing
offset; the model holds a response matrix. A gain is one physical quantity;
the model holds it in three stages. A positive quantity is best explored in
its logarithm. In every case the parameters you want are a *function* of the
numbers the model has — and re-deriving them is an inference concern, so it
belongs in the inference layer, not in the instrument description.

---

## Two words

Everything rests on separating two ideas that are easy to conflate.

`Latent`
: **A named quantity you infer.** It is what a sampler draws or an optimizer
  steps: a name, an initial value (which fixes its shape and dtype), optionally
  a prior, optionally a declaration that it enters the model linearly. A latent
  knows *nothing* about the pipeline. `log_gain` is a latent; so is a
  10⁴-element vector of sky `alm` coefficients.

`Bind`
: **A rule turning latents into pipeline leaf values.** It names the latents it
  consumes, the leaves it writes, and optionally the function between them. A
  bind knows nothing about priors.

A pipeline leaf is *what the instrument model holds*. A latent is *what you
chose to infer*. Keeping them apart is what makes re-parameterization free.

```{mermaid}
flowchart LR
    subgraph L["what you infer — Latent"]
        A["fwhm"]
        B["log_e"]
        C["log_gain"]
    end
    subgraph BD["how it enters — Bind"]
        D["fn: gaussian_beam_alms"]
        E["fn: exp"]
    end
    subgraph P["what the model holds — pipeline leaves"]
        F["projector.beam_alms<br/>(n_freq, n_alm)"]
        G["gain_lna.gain"]
        H["gain_backend.gain"]
    end
    A --> D
    B --> D
    D --> F
    C --> E
    E --> G
    E --> H
```

Note what the diagram makes obvious: the arrows are **many-to-many**. Two
latents produce one leaf; one latent produces two. That is exactly what a
scheme attaching priors directly to leaves cannot express.

---

## The three shapes

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Direct
:class-header: sd-font-weight-bold

One latent, one leaf, unchanged.
^^^
```python
Bind("sky_alms",
     into=lambda p: p["sky"]
                     .sky_model.alms)
```
+++
The case the old positional priors covered.
:::

:::{grid-item-card} Tied
:class-header: sd-font-weight-bold

One latent, several leaves, transformed.
^^^
```python
Bind("log_gain",
     into=(lambda p: p["gain_lna"].gain,
           lambda p: p["gain_bk"].gain),
     fn=jnp.exp)
```
+++
One physical quantity, three stages.
:::

:::{grid-item-card} Derived
:class-header: sd-font-weight-bold

Several latents, one leaf, through a model.
^^^
```python
Bind(("fwhm", "log_e"),
     into=lambda p: p["sky"]
                     .projector.beam_alms,
     fn=beam_alms_from)
```
+++
Two scalars driving 10⁴ numbers.
:::
::::

A `fn` returning a single array is written to **every** selector in `into` —
that is what makes tying one line. Returning a tuple addresses each selector
separately.

### The container type is the physics, so `fan=` lets you say which

Read that last paragraph again as someone holding two leaves and a 2-vector.
Both readings are available, they are **different physics**, and with the
default `fan=None` the only thing separating them is a Python container type.
Measured on two scalar leaves — an antenna efficiency and a gain, arranged so
both enter the prediction as a bare product — driven by the same `[2, 5]`:

| `fn` returns | mode | what each leaf gets | `pred[0, 0]` |
|---|---|---|---|
| `lambda v: v` | broadcast | both leaves get `[2, 5]` | **4.0** |
| `lambda v: list(v)` | distribute | efficiency `2`, gain `5` | **10.0** |

`v` and `list(v)` are the same numbers. One is a JAX array and one is a Python
list of its elements, and that difference — invisible in the values, invisible
in every shape, invisible to `check_linearity` and to `identifiability` —
selects between a tie and an element-wise split. Someone who meant "write this
whole vector into both leaves" and reached for `list(v)` gets a finite,
correctly-shaped, silently wrong model: off by a factor of 2.5 here, and by
whatever the leaves happen to be worth in general.

`fan=` is that intent written down, and therefore checkable:

```python
into = (lambda p: p["loss"].efficiency, lambda p: p["gain"].gain)

Bind("v", into=into, fn=lambda v: v, fan="broadcast")   # both leaves [2, 5] -> 4.0
Bind("v", into=into, fn=list,        fan="distribute")  # 2 and 5        -> 10.0
```

A declaration that contradicts what `fn` actually produced is refused, naming
both sides:

```text
Bind for ('v',) declares fan='broadcast' — ONE value written into all 2 `into`
selectors — but `fn` returned a list of 2, which is a distribution. A tie writes
one value into every leaf and a distribution gives each leaf its own, which is
different physics — and the two spellings differ only by a Python container
type, so nothing downstream would report the mismatch: the model would be
finite, correctly shaped and wrong. Fix `fn`, or change the declaration to the
mode you meant.
```

The mirror case — `fan="distribute"` where `fn` returned a single array, "which
would be tied into all of them" — is refused the same way. An unknown mode is
refused at construction, before anything evaluates, and names the two that
exist; those two are also exported as `BROADCAST` and `DISTRIBUTE` if you prefer
constants to literals.

`fan=None` remains the default and keeps the inference. `Bind` is public and
appears in every example on this page, so refusing by default would break all of
them. What `fan=` buys is that the guess becomes a *claim* — and a claim can be
wrong out loud.

:::{note}
**One case no inference can decide, even in principle.** With a *single* `into`
selector, the length test that separates the modes — `len(produced) ==
len(into)` — is satisfied by a length-1 container under **either** intent. So
the container is unwrapped on a guess, and the guess is warned about rather than
refused:

```text
AmbiguousFanWarning: Bind for ('v',) has one `into` selector and `fn` returned a
list of 1, so which fan-out mode was meant cannot be told from the length: a
1-container matches a single selector under 'distribute' AND under 'broadcast'.
Unwrapping it, which is the only reading that can reach an array leaf — but say
fan='distribute' to declare that, or fan='broadcast' to be refused here if this
ever stops being a container.
```

A warning and not an error, deliberately. A Python list is not an array leaf, so
unwrapping is the only reading that can yield a valid pipeline at all;
broadcasting the container would change the pipeline's pytree structure and be
refused by `validate()` a moment later. There is no wrong answer to prevent
here, only an undeclared one — so refusing would break working code and buy no
correctness. `fan="distribute"` declares it and the warning goes.
:::

---

## A beam from two numbers

The whole point, in one figure. The latents are a beam width and a pointing
offset; the pipeline leaf they feed is the full response matrix. Nothing in the
instrument description changed to make this possible.

```python
from rheplicant.inference import Bind, Latent, ParameterSpace

space = ParameterSpace(
    latents=[
        Latent("fwhm",   init=0.50, prior=dist.Uniform(0.15, 0.70)),
        Latent("offset", init=0.00, prior=dist.Normal(0.0, 0.4)),
    ],
    bindings=[
        Bind(("fwhm", "offset"),
             into=lambda p: p["sky"].projector.matrix,
             fn=beam_matrix),
    ],
)
model = to_numpyro_model(twin, state, space, noise_std=0.02)
```

:::{figure} _static/inference-posterior-light.svg
:figclass: only-light
:alt: NUTS posterior over two beam latents, recovering the truth
:width: 100%

Left: the NUTS posterior over the two latents, with the truth marked. Right:
the same posterior against the **likelihood** Fisher forecast computed at the
truth — measured, `sd(NUTS) / sd(Fisher)` is 0.985 for `fwhm` and 0.975 for
`offset`. They agree because the model is this close to linear in its
parameters *and* because these two priors say almost nothing at this noise
level; see below for the second half, which is not general.
:::

:::{figure} _static/inference-posterior-dark.svg
:figclass: only-dark
:alt: NUTS posterior over two beam latents, recovering the truth
:width: 100%

Left: the NUTS posterior over the two latents, with the truth marked. Right:
the same posterior against the **likelihood** Fisher forecast computed at the
truth — measured, `sd(NUTS) / sd(Fisher)` is 0.985 for `fwhm` and 0.975 for
`offset`. They agree because the model is this close to linear in its
parameters *and* because these two priors say almost nothing at this noise
level; see below for the second half, which is not general.
:::

### Why those two agree, and when they will not

A Fisher matrix and a posterior are only the same object when the prior is
flat over the region the data picks out. That is the case here, and it is worth
measuring rather than assuming: `offset`'s declared `Normal(0, 0.4)` contributes
**5×10⁻⁷** of the total precision, and `fwhm`'s `Uniform(0.15, 0.70)` is flat
across a range **512×** the posterior width. Neither prior is doing any work,
so the likelihood's information is the whole story and the two panels land on
top of each other.

Change either declaration to something informative — the `Normal(250.0, 50.0)`
noise-wave temperature further down this page is one, and `wiener_solve` solves
with it as `S` — and the two part company, silently, because
`fisher_information` builds `F = JᵀN⁻¹J` and that is a statement about the
**data alone**. Pass the space and it adds each declared Gaussian prior's own
curvature at that latent's span:

```python
# likelihood only: the Cramér-Rao bound, no prior in it
fisher_information(forward, fitted, noise_std=0.02)              # kind="fisher"

# the posterior precision NUTS is sampling from, same declaration
fisher_information(forward, fitted, noise_std=0.02, space=space) # kind="posterior_precision"
```

`kind` travels through `parameter_covariance` into `"covariance"` and
`"posterior_covariance"`, so which quantity a `sigma()` reports is a property
of the object rather than of what the caller remembers passing.

The beam space above cannot use `space=`: `fwhm` is declared `Uniform`, which
has no quadratic form, and substituting its variance would report a crisp
Gaussian posterior for a prior with no curvature at all. That raises by name
rather than being approximated — the same refusal, in the same words,
`wiener_solve` already gives a non-conjugate prior. For a space like this one
the likelihood Fisher *is* the answer available, and the figure says so.

Sample sites are named by their **latents**, so the samples come back keyed by
the coordinates the model was declared in. A `log_gain` latent is one site
named `log_gain`, even though its value reaches a leaf called `gain`, and it
stays one site if it drives five stages.

That naming reaches the error bars too:

```python
cov = parameter_covariance(fisher_information(forward, fitted, noise_std=0.02))
cov.sigma("fwhm")             # not cov.matrix[0, 0]
cov.block("fwhm", "log_gain")  # the cross-covariance, by name
```

---

## What gets checked

Every one of these failure modes produces a **finite, correctly-shaped, wrong**
inference rather than an exception. So they are all errors, and all of them are
caught at declaration or build time — `validate()` runs on shapes alone
(`jax.eval_shape`), once per build rather than per evaluation, so there is no
reason to make it skippable.

| Checked | Without the check |
|---|---|
| Latent names are unique | NumPyro sites silently overwrite each other |
| Every `Bind` names a declared latent | `KeyError` from deep inside a trace |
| **Every latent reaches the model** — named in a binding, or (for a raw bind) shown to move it | It samples happily and returns the prior |
| No leaf is written twice | `eqx.tree_at` lets the last write win, silently |
| A raw bind is not combined with declarative bindings | `bind()` would apply only the raw one |
| Binding preserves every leaf's shape and dtype kind | A treedef encodes neither; the value is broadcast |
| Every selector reaches a real array leaf | Confusing failure inside `tree_at` |
| Produced shape matches its target | A broadcast that is shaped right and means nothing |
| Produced dtype *kind* matches | Complex into a real leaf is a modelling error, not a cast |
| A prior's shape matches its latent's | Sampling a different-sized thing than you bind |
| Binding preserves the pytree structure | `vmap`, `jit` and Fisher flattening all break |

Every one of them raises `ParameterSpaceError`, and so does most of the rest of
this page. It is exported exactly where it is raised and nowhere else — from
`rheplicant.inference`, but **not** from the top-level `rheplicant`, which is
the mirror image of its sibling error classes (`AssemblyError`, `PipelineError`,
`StateValidationError` and the rest are re-exported from `rheplicant` and not
from `rheplicant.inference`). Either of these works; the second is the path that
is uniform across all of them, since `rheplicant.core.errors` is where every one
is defined:

```python
from rheplicant.inference import ParameterSpaceError
from rheplicant.core.errors import ParameterSpaceError
```

### More refusals, at moments the table above cannot cover

The table is `validate()`'s inventory, and `validate()` sees shapes at build
time. Some failure modes are invisible to it and are checked where they *can* be
seen — all with the same signature as everything above: finite, correctly
shaped, and wrong.

**A `fan=` that contradicts what `fn` produced** is caught when the binding
evaluates, because that is the first moment anything knows what `fn` returned —
see [the container type is the
physics](#the-container-type-is-the-physics-so-fan-lets-you-say-which) above.

**A forward model that draws its own randomness** is refused on the way *into*
every inference exit. Inference closes the model over one template `State`, so a
stage consuming the PRNG key draws **one** realisation and adds that same frozen
field to every prediction compared against the data:

```text
build_forward_fn was given a forward model containing NoiseOperator at 'noise',
which declares 'key' in `requires` and therefore draws randomness from the
state's PRNG key. Inference closes the model over one template state, so that
draw is made ONCE and the same frozen realisation is added to every prediction
compared against the data — a bias in the fitted parameters that is reported
with an unchanged error bar, because adding a constant field is exactly affine
and so passes check_linearity, identifiability and every shape check untouched.
Drop the stage from the twin you infer with: Assembly.without(node_id) for a
graph assembly, or rebuild the Pipeline without it. Keep the stochastic pipeline
for GENERATING data — that is where the noise belongs — and give the inference
exits the deterministic model plus a noise_std / NoiseModel, which is how the
scatter is meant to enter.
```

Read the middle clause: the corruption is *exactly affine*, so every guard on
this page reports green through it. Measured on an 8×8 grid — sky 100 K, truth
`g = 1.1`, 2 K measurement scatter, a `NoiseOperator(sigma=20)` left in the twin
and nothing else changed — through `wiener_solve` for the estimate and
`fisher_information` → `parameter_covariance` for the bar:

| twin handed to the exit | `g` | reported σ |
|---|---|---|
| deterministic (`.without("noise")`) | 1.1002 | 0.002500 |
| stochastic stage left in | **1.0735** | 0.002500 |

The estimate moved by 10.6 of its own error bars, and the two bars are equal
*bit for bit* — not to three digits, `sd_clean == sd_corrupt` is `True`. That is
the failure mode in one line: both exits of the workflow wrong by the same
amount, with no diagnostic moving. The magnitude is a property of the draw; the
invisibility is structural.

The repair the message names is one line, and it is the supported one:

```python
twin_for_inference = twin.without("noise")   # Assembly.without: re-runs assemble()
```

The detector is the operators' own declaration — `RANDOMNESS` in `requires` — so
a new stochastic operator is covered the day it declares what it reads. What it
therefore *cannot* see is an operator that draws without declaring `"key"`, or a
draw closed over inside a static field — stated rather than implied, because
there is no numerical symptom to fall back on. That the shipped operators
declare honestly is itself checked, mechanically, in
`tests/test_operator_declarations.py`.

**An `observed` the prediction would have to broadcast against** is the third,
and it is the one you are least likely to think about, because every shipped
exit already applies it for you. `(24, 8) - (8,)` is legal NumPy and a wrong
residual, with no symptom: the loss converges, the Fisher matrix inverts, the
posterior looks healthy.

Both of these last two are public and are the two you want if you are writing an
exit of your own — `refuse_stochastic_stages(pipeline, caller)` and
`check_observed_shape(prediction_shape, observed, predictor=...)`. The shipped
exits call them already, so reach for them only when writing a new one.

---

## The noise model

Every inference route needs one number per sample: how noisy is it? The
likelihood needs it, a Wiener solve and a GCR draw need it as a weight, the
Fisher matrix needs it, a NumPyro observation site needs it as a scale. Passing
it to each of them as a bare `noise_std` quietly assumes the answer is *given*
and *constant*. For a radiometer it is neither:

$$\sigma(d) = \frac{|d|}{\sqrt{\Delta\nu\,\tau}}$$

σ is a function of the very thing being inferred. So there is one object that
answers the question, and every route takes it:

:::{list-table}
:header-rows: 1
:widths: 26 40 34

* - Model
  - σ
  - `depends_on_prediction`
* - `HomoscedasticNoise(sigma)`
  - a constant — what a bare `noise_std` always meant
  - `False`
* - `RadiometerNoise(Δν, τ)`
  - `|prediction| / √(Δν·τ)`
  - `True`
* - `FlaggedNoise(base, flags)`
  - the wrapped model, `∞` where flagged
  - inherited
:::

Wherever an exit **has a prediction to evaluate the model at**, `noise_std=`
takes one in the bare array's place and no signature changed to allow it —
`fisher_information` and `to_numpyro_model` both normalize through
`as_noise_model`:

```python
from rheplicant.inference import RadiometerNoise, FlaggedNoise

noise = FlaggedNoise(RadiometerNoise(channel_width=61e3, integration_time=1.0),
                     flags=state.aux["flags"])

fisher_information(forward, params, noise_std=noise)
```

The conjugate solves are the exception, and the exception is the interesting
part: they have no prediction — it is what they solve for — so `wiener_solve`,
`gcr_sample` and `condition_estimate` take a decided σ array and refuse a model
rather than freezing it at an arbitrary point. That is the whole of the
[`noise=` / `noise_std=` split](#noise-or-noise_std-the-keyword-is-the-type),
below.

`FlaggedNoise` is how RFI flags reach the covariance: by **wrapping a noise
model**, not by threading a `flags=` keyword through five separate functions. An
infinite σ is a self-describing encoding of "this sample was not observed", and
every consumer turns it into a clean zero rather than a NaN.

`depends_on_prediction` is not a hint. It is the claim a solver branches on:
`False` means one solve, `True` means the covariance has to be found before it
can be used.

### The term that is not a constant

The Gaussian log-density is

$$\log p = -\tfrac{1}{2}\sum_i\left[\frac{r_i^2}{\sigma_i(\theta)^2}
  + \log 2\pi\sigma_i(\theta)^2\right]$$

When σ is constant the second term is an additive constant and dropping it
changes nothing. When σ depends on the prediction, dropping it — which is
exactly what generalized least squares does — gives a **different estimator**,
one with no penalty for shrinking the prediction to make the variance small.

For the multiplicative model both are solvable by hand. With $d_i = \theta(1+w_i)$
and $w\sim\mathcal N(0,f^2)$:

| Objective | Stationary point | Expectation |
|---|---|---|
| GLS (log-det dropped) | $\hat\theta = \sum d^2 / \sum d$ | $\theta_{\rm true}(1+f^2)$ — biased high |
| Full Gaussian | $nf^2\theta^2 + \theta\sum d - \sum d^2 = 0$ | $\theta_{\rm true}$ — unbiased |

So the term GLS discards as a normalization is precisely the one that removes
the bias. `NoiseModelLikelihood` keeps it by default; `include_logdet=False` is
the explicit, documented GLS variant rather than an oversight.

:::{warning}
The same thing happens to the Fisher matrix, and more quietly. When the
covariance carries parameter dependence, $J^\top N^{-1} J$ **is not** the Fisher
information — there is a second term,

$$F = J^\top\Sigma^{-1}J + \tfrac{1}{2}
  \operatorname{tr}\!\left(\Sigma^{-1}\partial\Sigma\,\Sigma^{-1}\partial\Sigma\right),$$

which for a diagonal covariance is $2\,(\partial\log\sigma)^\top(\partial\log\sigma)$.
`fisher_information` includes it whenever the noise model reports
`depends_on_prediction`. Under `RadiometerNoise` it is a clean factor,
$F = (1+2f^2)\,J^\top N^{-1} J$ — so reporting only the first term forecasts
error bars too wide by $\sqrt{1+2f^2}$: a plausible number, and the wrong one.
:::

---

## Linear blocks

Some parameters enter the model linearly — sky `alm` coefficients, noise-wave
amplitudes, anything whose contribution is a matrix acting on it. Those are
also the *big* ones: a sky at `lmax` 191 across 32 channels is ~10⁶ real
degrees of freedom, where gradient samplers are hopeless and a conjugate
Gaussian solve is exactly right.

Declare it, and the declaration is checked before anything exploits it:

```python
space = ParameterSpace.direct(
    "sky_delta", init=jnp.zeros_like(maps),
    into=lambda p: p["sky"].sky_model.maps,
    fn=lambda delta: mean_sky + delta,      # affine, not just linear
    linear=True,
)
check_linearity(space, twin, state, names=("sky_delta",))   # against its own linearization
block = linear_operator(space, twin, state, names=("sky_delta",))  # A, Aᵀ, offset
solved, residual = wiener_solve(block, observed, noise_std=0.02,
                                prior_std={"sky_delta": 1.0})
solved["sky_delta"]                          # the answer, under its own name
```

`names=` is the spelling to reach for, including — as here — for a block of
**one** latent. Its answer is a `{name: array}` dict, which is the shape
everything downstream reads; the singular `name="sky_delta"` is legitimate and
different, and is covered [below](#one-latent-or-a-group). Note what the plural
costs and what it buys: `prior_std` becomes one entry per member, because `S` is
block-diagonal over a group rather than a multiple of the identity — and one
number spread across a noise-wave temperature in kelvin and a gain of order one
would be a prior nobody declared. Omit it entirely and each latent's own
`Latent(prior=...)` drives the solve.

That snippet is a fragment — `maps`, `mean_sky`, `twin`, `state` and `observed`
come from your own model — so it is not run as written. The four calls in it, at
exactly this spelling, are what
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
executes end to end; the [next section](#one-latent-or-a-group) has the
self-contained version with its real output.

:::{figure} _static/inference-linear-light.svg
:figclass: only-light
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band is the honest part — it widens exactly where the ~20°-wide
beam stops constraining the sky, and the mean's small-scale ripple lives inside
it rather than being mistaken for signal. Right: RMS error per channel, before
and after. Both the mean and the draws are conjugate-gradient solves; the same
calls take sky alms, where a gradient sampler is not an option.
:::

:::{figure} _static/inference-linear-dark.svg
:figclass: only-dark
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band is the honest part — it widens exactly where the ~20°-wide
beam stops constraining the sky, and the mean's small-scale ripple lives inside
it rather than being mistaken for signal. Right: RMS error per channel, before
and after. Both the mean and the draws are conjugate-gradient solves; the same
calls take sky alms, where a gradient sampler is not an option.
:::

:::{admonition} Probe at extreme scales, not reasonable ones
:class: important

`check_linearity` probes at 10⁻³, 1 and 10³ times the latent's own magnitude,
taken from `max|init|`. The span is the point: a knee, a saturation, or a small
quadratic is indistinguishable from linear below some scale and grossly
nonlinear above it, so a suite of "reasonable" probes signs off on exactly the
blocks that fail in a sampler's tails.

One sharp edge, since the examples above walk straight into it: an **all-zero
`init` has no scale to take**, so the probes fall back to absolute. If the
latent lives at 10⁶ — sky alms in kelvin — give a representative `init` or pass
`scales=` explicitly, or the sweep never reaches the regime a sampler will.

A block fails only if it exceeds both a relative tolerance *and* an absolute
floor set by the arithmetic's own roundoff. Without that floor the relative
measure explodes at small probes — where the response variation is vanishing
but roundoff is not — and rejects perfectly linear blocks. That false positive
would be worse than no check, because the cure a user reaches for is to switch
the check off.
:::

`linear_operator` never forms a matrix: `A` comes from `jax.linearize` and `Aᵀ`
from `jax.vjp`, so applying a 10⁶-dimensional block costs one forward
evaluation.

### One latent, or a group

`linear_operator` takes `name=` for one latent and `names=` for several exported
as **one** block. Both are legitimate, and they are not interchangeable — which
is why passing both is refused rather than guessed.

```python
block = linear_operator(space, twin, state, names=("t_unc", "t_cos", "t_sin"))
solved, residual = wiener_solve(block, observed, noise_std=0.02)
solved            # {"t_unc": Array, "t_cos": Array, "t_sin": Array}
```

A group's `x` is a `{name: array}` dict, and so is the answer. That is the
point: the physical names survive the solve instead of the caller slicing an
anonymous stacked vector — and the dict is the shape the rest of the package
consumes. Run both spellings against one model and the difference is not in the
number, it is in what the number can be handed to:

```python
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import GainOperator, SkyOperator

state = State(coords=Coordinates(time=jnp.linspace(0.0, 60.0, 8),
                                 freq=jnp.linspace(60e6, 85e6, 4)),
              env=Environment(temperature=jnp.array(280.0)), key=jax.random.key(0),
              meta={"telescope": "my-antenna", "obs_id": "tour-001"})
twin = Pipeline(SkyOperator(amplitude=jnp.array(100.0)),
                GainOperator(gain=jnp.array(1.0)), names=("sky", "gain"))
space = ParameterSpace.direct("gain", init=1.0, into=lambda p: p["gain"].gain,
                              prior=dist.Normal(1.0, 0.3), linear=True)
forward, _ = space.forward_fn(twin, state)
observed = forward({"gain": jnp.array(1.1)})

grouped = linear_operator(space, twin, state, names=("gain",))
singular = linear_operator(space, twin, state, name="gain")
many, _ = wiener_solve(grouped, observed, noise_std=0.5)
one, _ = wiener_solve(singular, observed, noise_std=0.5)

print("many:", many, "| forward(many) ->", jnp.shape(forward(many)))
print("one: ", repr(one))
forward(one)
```

```text
many: {'gain': Array(1.099999, dtype=float32)} | forward(many) -> (8, 4)
one:  Array(1.099999, dtype=float32)
TypeError: JAX does not support string indexing; got idx='gain'
```

`names=("gain",)` is a legitimate **group of one**, and is how a partition holds
one-latent and many-latent blocks without special-casing either; the plan's own
engine always spells it that way. Reach for `names=` by default. The singular is
not deprecated and is not going away — it is the one-latent shorthand, and its
bare array is the right thing when you are about to do linear algebra with it
rather than put it back into the model.

When you do have a bare one, wrap it as `{block.name: x}` before it meets
anything else. `LinearBlock.as_dict` is that call, and it is a no-op on the
grouped form, so it is correct whichever spelling built the block:

```python
print(singular.as_dict(one), "| forward(...) ->",
      jnp.shape(forward(singular.as_dict(one))))
print("grouped.as_dict(many) == many:", grouped.as_dict(many) == many)
```

```text
{'gain': Array(1.099999, dtype=float32)} | forward(...) -> (8, 4)
grouped.as_dict(many) == many: True
```

Six consumers need that wrap, and none of their exceptions names the actual
mistake: `space.forward_fn`'s `forward` and `space.bind` raise the `TypeError`
above, `identifiability(at=)` and `linear_operator(at=)` raise `TypeError:
iteration over a 0-d array`, `conditional_potential` raises `TypeError:
'jaxlib._jax.ArrayImpl' object is not a mapping`, and `fisher_information`
raises the string-indexing one again from inside a `jacfwd` trace.
`tests/inference/test_linear_block_as_dict.py` pins all six, so if any of them
ever starts accepting the bare form, that is a test going red rather than a
paragraph going quietly stale.

**Solving a group jointly is not the same as alternating over its members.** Two
latents the data barely tells apart are resolved in one CG here, where
alternation converges at the rate of their correlation while reporting a
converged residual and a well-conditioned block at every step. The joint κ that
`condition_estimate` reports for the group is the honest one — and "group the
correlated latents into ONE `Block`" is exactly what `SamplingPlan`'s own
non-convergence message recommends.

Not everything can be grouped, and the check says so: for a group,
`check_linearity` verifies **joint** affinity, which a bilinear pair fails. A
`gain × T_ant` model is refused as a group and belongs in two blocks of one
[plan](#a-plan-one-partition-two-exits).

### Sampling it, exactly

`wiener_solve` gives the posterior **mean**. `gcr_sample` gives a posterior
**draw** — by adding two white-noise terms to that same right-hand side:

```text
(AᵀN⁻¹A + S⁻¹) x  =  AᵀN⁻¹(d − offset)  +  AᵀN⁻¹ᐟ² ω₁  +  S⁻¹ᐟ² ω₂
```

with `ω₁`, `ω₂` standard normal on the data and on the latent. The right-hand
side then has mean `AᵀN⁻¹(d − offset)` and covariance equal to the operator
itself, so `x = M⁻¹b` carries the posterior mean **and** covariance
`M⁻¹ M M⁻¹ = M⁻¹` — exactly.

```python
sample, residual = gcr_sample(block, observed, noise_std=0.02,
                              prior_std={"sky_delta": 1.0}, key=jax.random.key(0))
sample["sky_delta"]                         # same shape the solve returned
```

Both take `prior_mean=`, which defaults to zero — wrong for most physical
quantities, since a noise-wave temperature sits near 250 K, not near zero. An
affine binding that adds the same offset gives the identical Gaussian, but
putting it on the prior says what it means.

### Where `S` comes from

Both keywords default to the latent's own declaration. `Latent(prior=...)` is
the one place this package says what a quantity is a priori, and it is what
`to_numpyro_model` reads — so it is what these solves read too, and a space
handed to NUTS and to `gcr_sample` targets one posterior rather than two:

```python
space = ParameterSpace.direct(
    "t_nw", init=jnp.zeros((3, N_FREQ)),
    into=lambda p: p["rx"].noise_wave_temps,
    prior=dist.Normal(250.0, 50.0), linear=True,
)
block = linear_operator(space, pipeline, template, names=("t_nw",))
mean, _ = wiener_solve(block, observed, noise_std=sigma)   # S is already known
```

The keywords remain, for a latent with no declared prior. What is *not*
allowed is passing one that contradicts the declaration — that raises, naming
both numbers, rather than letting one of the two silently win. A declared
prior with no conjugate Gaussian form (a Half-Normal, a Uniform, a LogNormal)
also raises here: these routines solve `(AᵀN⁻¹A + S⁻¹)x = b`, and substituting
such a prior's mean and variance would hand back a finite, confident posterior
for a model you did not declare. Sample that space with NUTS instead.

With several latents in the space, `linear_operator(..., names=("gain",))`
carries **that** latent's declaration — each block gets its own `S`, which is
what makes the Gibbs sweep below sound. And the contradiction check reads the two
values, not the context: concrete numbers are compared normally under `jit`,
`eqx.filter_jit` and inside `iterative_gls`'s reweighting loop. Only a keyword
that is *itself* a tracer is refused as undecidable, because then there is no
number yet to compare.

This is a constrained realization, not a Markov chain: every call is an
independent draw, with no burn-in and no convergence to diagnose. It costs the
same single CG solve as the mean, because the fluctuation enters the
right-hand side and never the operator — which is what makes a 10⁶-dimensional
block samplable at all.

:::{tip}
**Gibbs.** A block is only linear *given* the other latents, so pass `at=` to
rebuild it wherever they currently are, and check the linearity claim once
outside the loop:

```python
check_linearity(space, twin, state, names=("sky_alms",))   # once
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, names=("sky_alms",),
                            at=values, check=False)         # every sweep
    drawn, _ = gcr_sample(block, observed, noise_std=sigma,
                          prior_std={"sky_alms": s}, key=next(keys))
    values = {**values, **drawn}                            # the dict merges straight in
    values = update_the_nonlinear_ones(values)              # NUTS, MH, optimize...
```

That merge is the grouped spelling paying for itself: `drawn` is already keyed
by latent, so the update is one `{**values, **drawn}` and stays right when the
block later grows a second member. It is exactly what `SamplingPlan`'s conjugate
engine does internally, which is why that engine always spells the block
`names=` even for one latent.

Omitting `at=` is silent, not loud: the block keeps describing the model at its
declared starting point, which is right for exactly one sweep.

The sketch is not runnable as written — `update_the_nonlinear_ones` is yours —
but its shape is: a grouped draw merges into `values` and `at=values` rebuilds
the block from it, with no unpacking anywhere in the loop.

That loop is what [`SamplingPlan`](#a-plan-one-partition-two-exits) declares, so
you do not write it — including the two things a hand-rolled version leaves out,
which are the ones that go wrong.
:::

---

### When the covariance is not given

Both solvers above take `noise_std` and neither cares where it came from. Under
`HomoscedasticNoise` it comes from you and there is nothing more to say. Under
the default `RadiometerNoise` there is: σ tracks the prediction, so the weights
depend on the solution and the solution depends on the weights. Neither is
available first.

`iterative_gls` supplies the missing half by fixed point — solve at the current
σ, recompute σ at the new prediction, repeat — and **nothing about
`gcr_sample` changes**:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau),
                      prior_std=PRIOR)

draw, _ = gcr_sample(block, observed, noise_std=found.noise_std,
                     prior_std=PRIOR, key=key)
```

It is the same iteratively-reweighted GLS as hydra-tod's
`hydra_tod.linear_sampler.iterative_gls` — a test checks the two agree — but
**matrix-free**: hydra-tod forms a dense `U` and `N_inv`, while here the
algorithm runs on the block's JVP and VJP, which is what makes 10⁶ degrees of
freedom possible at all.

What comes back is a `GLSResult`, and it carries the fixed point's whole
provenance rather than just the answer — on a 64×4 design at
`RadiometerNoise(1e6, 1.0)`:

```text
solution   [11.999567 -4.997814  3.001249  7.997115]   # truth [12, -5, 3, 8]
noise_std  shape (64,), 0.0054 .. 0.0625               # the σ the fixed point found
residual   2.009e-07     iterations 5
delta      2.394e-07     converged  True
```

`solution` and `noise_std` are the two halves of the answer — the second is what
you hand to `gcr_sample` above. The other two numbers are separate fields
because they measure different loops: `residual` is the *inner* CG residual of
the final solve, `delta` the *outer* reweighting step `‖x_new - x‖ / ‖x_new‖`.
`converged` reports on `delta` only — a tight CG residual says nothing about
whether the covariance reached a fixed point, which is the whole distinction the
`reweight_tol` warning below turns on.

Check `found.converged`. A covariance that is not a fixed point is still a
number, and a draw conditioned on it is still a draw.

:::{tip}
**The mean can be weight-independent while the width is not.** In
[`examples/gls_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/gls_gcr.py)
three switched loads meet three per-channel noise-wave unknowns, so the reduced
system is **square** — one solution, and the weights cancel out of it. The
point estimate is unmoved by reweighting, exactly, not approximately.

The posterior covariance $(A^\top\Sigma^{-1}A + S^{-1})^{-1}$ depends on Σ all
the same, and a GCR draw is precisely a draw of that width. In that example a
frozen σ reports error bars wrong by −8 % to +8 %, in both directions, on a
point estimate that was already right. "The fit came out the same" is not
evidence the covariance did not matter.

Where the system *is* over-determined across genuinely different noise levels,
the estimate moves too: on a prediction spanning a decade, freezing σ costs a
factor of ≈2.3 in recovered RMS error.
:::

:::{warning}
`reweight_tol` cannot have a fixed default, and neither should yours. Two
independent floors bound how small a step is measurable: the arithmetic's
epsilon (`1.2e-7` in float32 — so a plausible-looking `1e-8` is exactly this
trap), and **the inner CG tolerance**, since consecutive solves differ by their
own residual whatever the outer iteration does. The latter binds in float64,
where a tight `tol=1e-10` sits five orders of magnitude above `eps`. The default
is `max(8·eps, tol)`. Ask for less than either and the run does not fail
quietly — it spends `max_reweights` steps and reports `converged=False` for a
fixed point it had reached.
:::

**What this estimator is.** Freezing σ inside each solve is what makes every
step a linear-Gaussian problem, and it is also what makes the converged answer
*generalized least squares* rather than the maximum of the full Gaussian
likelihood: the log-determinant's dependence on the solution is held fixed
rather than differentiated ([above](#the-term-that-is-not-a-constant)). GLS is
the right thing to condition a constrained realization on, because a GCR draw
*is* a draw from a linear-Gaussian posterior at a given covariance. If you want
the full likelihood's mode or posterior, that is a gradient sampler's job.

### `noise=` or `noise_std=`: the keyword is the type

Two spellings have now run past each other on this page — `iterative_gls(...,
noise=RadiometerNoise(...))` on one line and `gcr_sample(...,
noise_std=found.noise_std)` on the next — and they are **not** two names for one
argument.

* **`noise_std=`** names a σ that has already been decided: an array.
  It appears on `wiener_solve`, `gcr_sample`, `condition_estimate`,
  `fisher_information` and `to_numpyro_model`.
* **`noise=`** names the *rule that decides* one: a `NoiseModel`.
  It appears on `iterative_gls`, `SamplingPlan.estimate` and
  `SamplingPlan.sample`.

A previous recommendation was to rename `noise_std=` to `noise=` throughout,
keeping the old spelling as a deprecated alias. **It is rejected**, and the
reason is written here rather than left in a review nobody can find: at the
conjugate solves the two are not interchangeable, because there is nothing for a
rule to be evaluated *at*. `noise.std(prediction)` needs a prediction; a Wiener
solve's prediction is exactly what it is solving for. Renaming the keyword would
make the wrong call type-check without making it meaningful — the solve would
have to freeze σ at some arbitrary point and hand back the result as a
posterior.

So the conjugate seam refuses a model **by name**, and says which of the two
problems it is. Both calls below were run against the self-contained model in
[One latent, or a group](#one-latent-or-a-group); the refusal fires on the
argument's *type* and the exit's own name, before anything block-specific, so it
reads the same for any block:

```python
from rheplicant.inference import HomoscedasticNoise

wiener_solve(block, observed, noise_std=HomoscedasticNoise(sigma=0.5))
```

```text
ParameterSpaceError: wiener_solve takes a plain sigma array, not a
HomoscedasticNoise. The conjugate solves compute 1/sigma**2 directly; pass
`noise.std(...)`, or the sigma you built the model from.
```

```python
wiener_solve(block, observed,
             noise_std=RadiometerNoise(channel_width=61e3, integration_time=1.0))
```

```text
ParameterSpaceError: wiener_solve was given RadiometerNoise, whose sigma
depends on the prediction — but a conjugate solve has no prediction to
evaluate it at, because the prediction is what it solves for. Freeze it
yourself at the parameter tuple you mean (`noise.std(prediction)`) and pass
that array, which also makes explicit that the result is an exact draw at
THAT covariance and not from the full model's conditional. A SamplingPlan
does this per sweep.
```

The second message is longer because that case is not a packaging problem.
Freezing σ is a real statistical choice with a stated consequence — an exact
draw at *that* covariance, which is not the full model's conditional — and it
belongs to whoever knows which parameter tuple to freeze at. `iterative_gls`
is that choice made by fixed point, `SamplingPlan` is it made per sweep, and
both take `noise=` for precisely that reason.

Measured across the exits, one call per cell:

| exit | keyword | bare σ array | `HomoscedasticNoise` | `RadiometerNoise` |
|---|---|---|---|---|
| `wiener_solve` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `gcr_sample` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `condition_estimate` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `fisher_information` | `noise_std=` | ✅ | ✅ | ✅ |
| `to_numpyro_model` | `noise_std=` | ✅ | ✅ | ✅ |
| `iterative_gls` | `noise=` | ❌ named refusal | ✅ | ✅ |
| `SamplingPlan.estimate` | `noise=` | ✅ | ✅ | ✅ |
| `SamplingPlan.sample` | `noise=` | ✅ | ✅ | ✅ |

Three edges that table makes visible, and none of them softens the split:

* `fisher_information` and `to_numpyro_model` write `noise_std=` and take a
  model anyway. They can — both *have* a prediction, so `as_noise_model`
  normalizes and `noise.std(prediction)` is answerable. The keyword says what
  the argument is; where both readings are usable, both are accepted.
* `condition_estimate` refuses a model with the same sentence its two siblings
  give. It used to refuse with `TypeError: Value 'HomoscedasticNoise(...)' with
  dtype object is not a valid JAX array type`, because it never ran the shared
  `_check_solve_arguments` — so neither the seam refusal nor the 1-D axis check
  reached it. Both run now, and the second matters more than the message did:
  this is the function a caller is told to consult to pick `tol` for those
  solves, and a κ computed under a different reading of the same 1-D σ answers
  a different question than the solve it was computed for. Measured, the two
  explicit readings give different condition numbers.
* `iterative_gls` writes `noise=` and takes **only** a model — a bare array is
  refused by name. It used to raise `AttributeError: 'ArrayImpl' object has no
  attribute 'depends_on_prediction'`, an attribute the caller never wrote, from
  a layer they were not thinking about. It is the one `noise=` exit that does
  not route through `as_noise_model`, and that is deliberate rather than an
  oversight: its whole subject is the fixed point a prediction-dependent σ
  implies, so a decided array leaves it nothing to iterate. Accepting one
  silently would answer a question nobody asked. The refusal names both ways
  out — `wiener_solve` for what a constant σ actually wants, or
  `HomoscedasticNoise(sigma)` if you want the fixed-point machinery anyway, in
  which case it returns after one step with `converged=True`.

  Note the direction: this exit refuses an **array** and wants a model, while
  the conjugate solves refuse a **model** and want an array. Both are the same
  rule seen from two sides — whether the exit has a prediction at which a
  prediction-dependent σ could be evaluated.

The rest of the argument lives in two docstrings, and was measured rather than
asserted. `_refuse_a_noise_model_at_the_conjugate_seam` carries the messages
above; `_check_solve_arguments` carries why this seam is deliberately *not*
routed through `as_noise_model`, which is that `1/σ²` and `inverse_variance`
agree on every finite σ, on `inf` (both exactly `0`), on `0` and on a negative
σ — and disagree on **NaN**, where the conjugate solves propagate it and the
caller finds out, while `inverse_variance` maps it to weight `0.0`, which
*means* "unobserved". In a conjugate solve a silently dropped sample moves the
posterior **width**, not only the point, with nothing reporting how many went.

---

## Conditioning: why a residual is not an accuracy

CG only ever reports on itself: `‖M x̂ - b‖`, the **residual** — how well `x̂`
satisfies the equation it was asked to solve. What a caller actually needs
bounded is `‖x̂ - x*‖`, the **error** — how close `x̂` is to the truth. The two
are related through the condition number of the normal operator
`M = AᵀN⁻¹A + S⁻¹`:

```text
‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖
```

For a well-conditioned block (κ ≈ 1) the two coincide and a small residual is
a small error — no further thought required. But κ is large **by design**
exactly when these solvers matter most: whenever the data does not fully
identify some direction in the block — one calibration load against three
per-channel unknowns, a flagged channel, a short integration — the prior is
the only thing holding that direction down, so `λ_min(M)` is exactly
`1 / prior_std²` and κ runs past `1e6`. CG happily converges its residual on
the well-constrained directions, which dominate the aggregate norm, while the
prior-dominated directions sit at their starting value: a residual that
*looks* converged is not, and a draw built from it comes back with far too
little scatter. That is exactly what used to happen before this fix:
`gcr_sample` on a badly-conditioned block reported a posterior σ three
orders of magnitude too narrow, while its residual sat comfortably under the
old, residual-only guard.

`condition_estimate(block, noise_std=..., prior_std=...)` reports κ,
matrix-free — the same two power iterations `wiener_solve`/`gcr_sample`
already run internally to guard themselves, exposed so a caller can choose
`tol` instead of guessing it:

```python
kappa = condition_estimate(block, noise_std=0.5, prior_std=100.0)
target_error = 1e-3
solved, residual = wiener_solve(block, observed, noise_std=0.5, prior_std=100.0,
                                tol=target_error / kappa, maxiter=4000)
```

`require_convergence` (default `1e-3`) already bounds `κ · relative_residual`
rather than the residual alone, so a block the data does not identify raises
instead of returning a silently wrong answer, and names the remedy in the
error. [`examples/noise_wave_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/noise_wave_gcr.py)
shows both ends: its three-load block (κ ≈ 2.69e1) passes at the library's
default `tol=1e-6`; its `--one-source` variant (κ ≈ 4.35e6 — one load
against three per-channel unknowns, so two of every three directions are
prior-dominated) raises at that same default, and needs `tol=1e-10,
maxiter=4000` to converge — which then reports per-channel σ ≈ 71–106 K
against the 100 K prior, i.e. the prior width recovered *honestly* where the
data says nothing, rather than the ≈0.03 K a residual-only guard used to let
through.

:::{admonition} The guard is not free
:class: note

Estimating κ costs `2 · POWER_ITERATIONS` operator applications — two power
iterations per end of the spectrum — on top of the CG solve itself, which
roughly **doubles** a well-conditioned solve where CG converges in a handful
of iterations. In a Gibbs loop, where the conditioning barely moves sweep to
sweep, estimate κ once outside the loop and pass `require_convergence=None`
inside, the same bargain `linear_operator`'s `check` argument offers for
`check_linearity`:

```python
kappa = condition_estimate(block, noise_std=sigma, prior_std=s)  # once
tol = target_error / kappa
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, names=("sky_alms",),
                            at=values, check=False)
    drawn, _ = gcr_sample(block, observed, noise_std=sigma,
                          prior_std={"sky_alms": s}, tol=tol, maxiter=4000,
                          require_convergence=None, key=next(keys))
    values = {**values, **drawn}
```
:::

---

## A plan: one partition, two exits

Everything above answers for **one** block. `wiener_solve` is a linear-Gaussian
block's posterior mean, `gcr_sample` an exact draw from that same conditional,
and the [Gibbs tip](#sampling-it-exactly) above sketches the loop you would
write by hand to put several blocks together. `SamplingPlan` is that loop,
declared rather than written:

```python
plan = SamplingPlan(
    space,
    Block("gain"),                  # conjugate — gain is Latent(..., linear=True)
    Block("t_coeff"),               # conjugate
    Block("beam_fwhm", steps=20),   # not declared linear → gradient
)

est   = plan.estimate(twin, state, observed, noise=noise)
draws = plan.sample(twin, state, observed, noise=noise, key=k, n_sweeps=200)
```

Every number quoted below is measured on the package's own fixture in
`tests/inference/test_plan.py`: a bilinear `gain × T_ant` model, a 6-element
gain against a `(3, 4)` time × frequency coefficient basis, 54 data points and
18 parameters. It is asymmetric in every dimension on purpose.

### The engine is derived, never restated

`Latent(..., linear=True)` already says which machinery a latent can take, so
`Block` does not ask again. A block whose members are all declared linear is
solved by the conjugate routines above; anything else is stepped by gradient.
The plan's `repr` reports what it derived:

```text
SamplingPlan(('gain'):conjugate, ('t_coeff'):conjugate)
SamplingPlan(('amp'):conjugate, ('centre'):gradient)
```

Exactly one case is genuinely ambiguous — a block mixing declared-linear and
non-linear members — and it is refused rather than guessed:

```text
Block('amp', 'centre') mixes declared-linear latents ['amp'] with non-linear
ones ['centre'], so which engine it takes cannot be derived. A conjugate solve
needs the whole block affine; a gradient step does not exploit the linear
members' structure at all, which for a high-dimensional linear block is the
difference between tractable and hopeless. Split them into separate blocks, or
say engine='gradient' to step the whole block by gradient deliberately.
```

`engine=` exists for that override and for nothing else. `steps=` on a
conjugate block raises for the same reason: a Wiener solve has no inner steps,
so accepting the argument would silently ignore it.

### The partition is checked, and the check is the point

Every latent of the space in exactly one block. Both ways of getting that wrong
are refused by name, and the dangerous one is the omission:

```text
This plan does not cover latent(s) ['t_coeff']: every latent of the space must
be in exactly one block. An omitted latent is silently frozen at its declared
init for the whole run — the sweep converges, the joint chi-squared settles, and
nothing anywhere reports that a parameter you declared was never inferred. Add
it to a block, or drop it from the space.
```

A latent in *two* blocks is refused too: the second update each sweep would be
solving a conditional the first had just invalidated, and every diagnostic would
report the second's answer as if the first had never run.

### Two methods, not a mode flag

`key=None | key` is the right *implementation* and the wrong *interface*. A
caller's intent is "give me the best fit" or "give me draws", not "here is a
PRNG key" — and two methods make the invalid combinations unrepresentable
rather than merely validated: `key` is required on `sample` and absent from
`estimate`, so "asked for draws and forgot the key" cannot be written down.
`n_sweeps` and `warmup` belong to one, `max_iter` and `tol` to the other,
because they mean nothing to the other. Both return the same currency —
`result.diagnostics` and `result.names` — so a caller can log or assert on a run
without knowing which exit produced it.

That shared currency has a name. `PlanResult` is the protocol both exits
satisfy: `.diagnostics` is a `PlanDiagnostics`, `.names` are the latents. What
differs is what the answer *is*, which is the honest difference between the two:

| exit | returns | the answer |
|---|---|---|
| `plan.estimate` | `Estimate` | `.values` — one array per latent |
| `plan.sample` | `Draws` | `.samples` — a chain per latent, plus `.n_draw`, `.mean`, `.std` (properties, not calls) |

Annotate against `PlanResult` when a function of yours should take either, and
against `Estimate` or `Draws` when it genuinely needs the values or the chain.

Both exits on the fixture above, at `HomoscedasticNoise(1.0)`:

```text
plan.estimate   sweeps 93   converged True   chi2 1.16085e+08 -> 0.0136245
                block residuals {('gain',): 7.56e-07, ('t_coeff',): 1.03e-06}
                max |T_ant - truth| = 0.0332 K

plan.sample     n_sweeps  60   kept  30   rhat 1.434   converged False
                n_sweeps 200   kept 100   rhat 0.99    converged True
                n_sweeps 600   kept 300   rhat 1.001   converged True
```

The 93 sweeps and the `rhat = 1.434` are the same fact seen twice: these two
blocks are strongly correlated, so the alternation moves slowly, and 30 kept
draws are nowhere near stationarity. Neither exit hides it — `converged` is
`False` on the short run and `PlanDiagnostics.rhat` says by how much.

That diagnostic is computed by `split_rhat`, which is exported and worth
reaching for on any chain you hold — cut the trace in half, treat the halves as
two chains, compare the variance between them against the variance within:

```python
import jax, jax.numpy as jnp
from rheplicant.inference import split_rhat

split_rhat(jax.random.normal(jax.random.key(0), (200,)))   # 1.0043  — iid
split_rhat(jnp.arange(200.0))                              # 2.6326  — pure drift
```

Two degenerate readings are deliberate rather than defensive: halves that are
each constant at the *same* value give `1.0` (nothing to mix), and halves each
constant at *different* values give `inf` (a chain that moved once and stopped).

**A trace too short to halve is refused, not answered**, and that is the part
worth knowing before you call it. The minimum is `MIN_DRAWS = 4` — two halves of
two, and it lives in `rheplicant.inference.plan`, not on the package surface.
Below it the diagnostic is not weak but undefined:

```text
split_rhat was given 3 value(s) and a split-r_hat needs at least 4 — two halves
of two. Below that the mixing diagnostic is not weak, it is undefined: halves of
one have no variance within them to divide by, so the answer came back as nan
rather than as this refusal — and a nan passes no threshold test in either
direction, which makes an undefined diagnostic read as whichever verdict the
caller tested for. SamplingPlan.sample refuses the same count on
(n_sweeps - warmup); this is that refusal, for the trace you brought yourself.
```

The middle clause is the reason this is an exception and not a `nan`: `rhat <=
rhat_max` is `False` for a nan and so is `rhat > rhat_max`, so a threshold guard
reads an undefined diagnostic as whichever answer it happened to test for.
`SamplingPlan.sample` enforces the same minimum on `n_sweeps - warmup`, so
reaching this by that route is not possible; `split_rhat` enforces it anyway,
because it is public and does not trust its one in-package caller.

### Convergence is monitored on the joint χ², never a per-block residual

This is the module's reason for existing. A hand-rolled alternating solve over
this same bilinear model, with a free antenna temperature per `(time, frequency)`
cell, lands hundreds to thousands of kelvin from the truth while **every
per-block guard this package ships reports green**: `check_linearity` passes at
every sweep, because each conditional genuinely *is* affine; the per-block
condition number is ≈1.47; and the CG residual reads ~`1e-7`. Nothing in the
sweep is wrong. The *partition* is, and no per-block number is entitled to
notice — a residual and a condition number are both computed from the block
being solved.

How far from the truth is not a property of the degeneracy but of where the
solve started, because the answer is the initial offset carried along the null
direction and left there. Measured in
`tests/inference/test_degenerate_partition.py`:

| start | rms error | CG residual | κ |
|---|---|---|---|
| at the truth | 0.014 K | 1.3e-07 | 1.467 |
| 1 % off | 27.4 K | 1.1e-07 | 1.466 |
| 25 % off | 704 K | 1.0e-07 | 1.452 |
| 100 % off | 2962 K | 9.6e-07 | 1.432 |

Four decades of error, and the guards read alike down the column — including
across the gap between the row that is right and the row that is
catastrophically wrong. There is no threshold to place between them, which is
why the remedy is a different measurement rather than a tighter tolerance.
Iterating is not one either: five sweeps and two hundred agree to four figures,
because the solve reaches the solution manifold at once and then has nowhere
left to move.

So the monitored quantity is the **joint** χ² at the current parameter tuple,
across sweeps. When it has not settled, the refusal says so and names what the
per-block numbers were doing at the time:

```text
SamplingPlan.estimate did not converge: after 4 sweeps the JOINT chi-squared is
still falling by 728774 per sweep (chi2 = 2.31316e+06), which is above
tol=1e-08. Note what this does NOT show up in: every conjugate block's own CG
residual is 4.18e-07 or better, because a per-block residual is computed from
the block and converges at every sweep of an alternation that is going nowhere.
```

It is a *decrease* that is tested, not a change: block-coordinate descent cannot
increase the objective, so once a sweep stops reducing it there is nothing left
to reach. Testing `|χ²[k] − χ²[k−1]|` instead walks into the floor
[`iterative_gls`](#when-the-covariance-is-not-given) documents for its own
`reweight_tol` — consecutive sweeps differ by roughly the inner solver's noise
whatever the outer iteration does. Measured in float32 on the motivating model,
the plateau sits at χ² = 2.7e-3 and jitters by 1.2e-3 a sweep, so a converged run
would have been refused for 300 sweeps and counting.

:::{important}
**The joint χ² catches a slow partition, not a degenerate one.** Running the
free-per-cell parameterization with `check_identifiability=False` and `tol=None`
for 40 sweeps:

```text
worst per-block residual  5.55e-07
joint chi2                8.88e-06
max |T_ant - truth|        1044.69 K
max |gain  - truth|            0.308
```

The joint χ² is *also* tiny, because a degenerate model fits the data exactly —
it just does so at an arbitrary point of the null space. χ² is the right monitor
for blocks that are identified but correlated; the rank test below is the only
thing that sees the other failure. They are two guards, not one guard twice.
:::

### The identifiability check, and its cadence

`identifiability()` looks across blocks at the joint Jacobian and refuses a
model with a null space before a sweep runs, naming the degenerate directions as
combinations of **latents** — "you have 6 blind directions" tells a user they
have a problem and nothing about which:

```text
SamplingPlan.estimate refuses this model: its joint Jacobian has nullity 6 of 60
parameters, so that many independent directions leave the prediction unchanged
and any answer along them is arbitrary. No per-block guard can see this — a
residual and a condition number are both computed from the block being solved —
so the run would otherwise converge quietly onto one arbitrary point of the null
space. The degenerate directions, as shares of each latent:
  direction 0: t_ant 0.50, gain 0.50
  direction 1: t_ant 0.50, gain 0.50
  direction 2: t_ant 0.50, gain 0.50
  direction 3: gain 0.50, t_ant 0.50
  ... and 2 more
```

Read the shares: each blind direction is half gain and half `t_ant`, which is
the bilinear degeneracy `gain × T_ant = (c·gain) × (T_ant/c)` written down. The
repair is a re-parameterization — the `(3, 4)` basis — not a tighter tolerance.

Called directly, `identifiability()` hands back an `IdentifiabilityReport`
rather than raising, which is how you ask the question before committing to a
partition. On a healthy 64×4 design:

```text
names ('coeffs',)   n_par 4   n_data 64
rank  4             nullity 0
rtol  1e-08         threshold 1.2210e-08
singular_values [1.22098371 1.01958727 0.90795122 0.80328398]
```

`rank` and `nullity` are the verdict; `singular_values`, `null_space` and
`column_norms` are what it was read off, so a borderline case can be inspected
instead of argued about. `threshold` is `rtol × σ_max`, and `rtol` defaults to
`DEFAULT_RANK_RTOL` (`1e-8`) — a *relative* cut, which is why it does not need
retuning when the design's overall scale changes.

`check_identifiability=` takes three values, and there is no size heuristic on
purpose, because the cost is a dense Jacobian and a dense SVD, `n_data × n_par`
float64 words:

| value | when the rank test runs |
|---|---|
| `"once"` (default) | at the starting values, before the first sweep |
| `"each_sweep"` | at every parameter tuple the run visits |
| `False` | never |

`"each_sweep"` is cheap for a small model and strictly more informative: a
nonlinear model's identifiability is a property of *where you are*, so a check
only at the start misses a degeneracy that opens up near the parameters you
actually reach. Both exits check by default, and the point estimate is the
**more** dangerous one to skip — a chain at least has `r_hat` to scream with,
while a point estimate has no diagnostic at all and CG converges quietly onto an
arbitrary point of the null space.

### Two limitations that are real

:::{warning}
**`identifiability()` refuses a complex latent, so a complex sky-`alm` block must
pass `check_identifiability=False`.**

```text
Latent(s) ['alm'] are complex. The prediction is real, so the map from complex
coefficients to data is R-linear but not C-linear and its rank over C is not the
number you want — a block with n complex coefficients has 2n real degrees of
freedom, and they can be identified separately. Declare the real and imaginary
parts as separate latents, or ask about a different block with names=.
```

Two independent reasons point the same way for the 10⁶-coefficient sky block
that `linear.py` exists for: the rank test cannot *analyse* it, and it could not
*afford* to — a dense `n_data × n_par` SVD is exactly what a matrix-free solver
was built to avoid. So that plan runs with the guard off, and the run has no
cross-block check at all. Declaring the real and imaginary parts as separate
real latents buys the check back on a small block; on a big one, nothing does.
:::

:::{danger}
**A block taking a finite number of NUTS steps is Metropolis-within-Gibbs, not
an exact conditional draw.** A conjugate block's GCR draw *is* exact, so a plan
of conjugate blocks is an exact Gibbs sampler with nothing to tune. The moment
one block takes `steps` gradient steps, the scheme is still valid and still
targets the right stationary distribution — and is no longer exact, because the
inner step count now sets the mixing. `steps=` reads as a performance knob and
is a statistical assumption.

Measured on the plan's other fixture (`amp` conjugate, `centre` gradient, 120
sweeps, truth `centre = 0.350`, `init = 0.100`, `key=jax.random.key(1)` — the
one table on this page that does not use `key(0)`, because a stuck chain is a
sampling accident and `key(0)` happens to escape at `steps=2`; that it depends
on the seed is the point, not a caveat):

| `steps` | `rhat` | `converged` | posterior `centre` |
|---|---|---|---|
| 2 | 1.070 | **False** | 0.1000 ± 0.0000 |
| 10 | 0.983 | True | 0.3500 ± 0.0026 |
| 50 | 0.991 | True | 0.3502 ± 0.0026 |

At `steps=2` the chain **never left its initial value** — a posterior reported as
a point mass 0.25 away from the truth, at zero width. Note how nearly it passed:
`r_hat` came in at 1.070 against a 1.05 threshold, and it is `r_hat` of the joint
χ², which still moves because the *other* block is moving. Read
`diagnostics.rhat`, and look at the draws of the latent you care about.
:::

:::{note}
**"Group the correlated latents into ONE Block" is not always available.** The
non-convergence message above names that remedy, and for the bilinear model it
is measured on, `check_linearity` refuses it — correctly:

```text
Latents ['gain', 't_coeff'] are each declared linear=True, but the prediction is
not affine in them JOINTLY [...]. Each conditional of a bilinear model is affine
on its own, which is why this is not caught one latent at a time — and why these
two cannot share one linear block. Split them into separate blocks and alternate,
or re-parameterize so the joint map really is affine. identifiability(space,
pipeline, state) will tell you what the split costs before you choose it.
```

Grouping works when the joint map really is affine — several noise-wave
amplitudes, a sky block and an offset. When the coupling between the blocks is
what makes them correlated, the remedy is the other one the message names: more
sweeps, and `identifiability(space, pipeline, state, names=...)` to see what the
split is costing before you choose it.
:::

---

## One space, every engine

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Optimize
```python
forward, start = space.forward_fn(twin, state)
fitted, losses = AdamCalibrator(
    n_steps=2000).fit(forward, start, data)
```
+++
Needed no changes: a dict is a pytree.
:::

:::{grid-item-card} Sample
```python
model = to_numpyro_model(
    twin, state, space, noise_std=0.02)
mcmc.run(key, observed=data)
```
+++
Sites named by latent.
:::

:::{grid-item-card} Forecast
```python
cov = parameter_covariance(
    fisher_information(forward, start,
                       noise_std=0.02))
cov.sigma("fwhm")
```
+++
Rows carry their names.
:::

:::{grid-item-card} Solve or sample exactly
```python
block = linear_operator(
    space, twin, state, names=("sky_delta",))
mean, _ = wiener_solve(
    block, data, noise_std=0.02)
draw, _ = gcr_sample(
    block, data, noise_std=0.02,
    key=jax.random.key(0))
```
+++
Answers keyed by latent; too big for a gradient sampler.
:::
::::

:::{note}
[`build_forward_fn`](api.md) is not superseded by this and stays. The two
answer different questions: `build_forward_fn` partitions a whole subtree into
trainables, which is what a neural surrogate's MLP weights want;
`forward_fn` carries parameters that were chosen, transformed, or shared, which
is what physical fitting wants.
:::

---

## Inference without a likelihood

Every engine above evaluates a likelihood. Simulation-based inference does not:
it draws pairs $(\theta, x)$ from the prior and the simulator, fits a
conditional density $q(\theta \mid x)$ to them, and reads the posterior off $q$
at the data actually observed.

```python
thetas, bank = simulate_pairs(twin, state, space, noise=noise,
                              key=jax.random.key(0), n_simulations=32_768)
q = NeuralPosterior.create(thetas, bank, key=jax.random.key(1))
q, history = train_posterior(q, thetas, bank, key=jax.random.key(2))

draws = q.sample(observed, key=jax.random.key(3), n_samples=4000)
```

Nothing there needs the noise to be Gaussian, the model to be differentiable,
or a normalization to be tractable — only a simulator, which the twin already
is. And the cost is **amortized**: a second observation is a forward pass, not
another chain.

The density is a conditional Gaussian mixture (an MLP → weights, means,
scales). A normalizing flow is more expressive; a mixture is a few dozen lines,
is exact for a Gaussian posterior at one component, and keeps the failure modes
legible.

:::{danger}
**An approximate posterior has no internal notion of being wrong.** A
badly-fitted $q$ returns a smooth, confident, correctly-centred, incorrect
distribution and reports nothing amiss. Both failure modes are real, and they
push in opposite directions — measured on the package's own linear-Gaussian
test problem, where the exact answer is available from `gcr_sample`:

| simulations | steps | components | width / exact |
|---|---|---|---|
| 8 192 | 1 500 | 1 | 0.88 |
| 8 192 | 4 000 | 1 | 0.84 |
| 8 192 | 4 000 | 2 | **0.60** |
| 32 768 | 1 500 | 1 | **0.98** |
| 32 768 | 1 500 | 2 | 1.07 |

*Too few simulations*: draws come from the prior, so only a fraction
$\sigma_\text{post}/\sigma_\text{prior}$ of them land near any given
observation, and the width comes out wrong. *Too many steps on a small bank*:
over-fitting, which makes $q$ too **narrow** — the failure that looks like a
better answer.

`train_posterior` holds out a validation split by default and **returns the
best validation step, not the last**, because the training loss falls
monotonically straight through the point where the fit stops being a
posterior. Prefer few components. And validate on a problem you can solve
exactly before trusting one you cannot — which is what
`tests/inference/test_npe.py` does, and why NUTS was wired to the noise model
first.
:::

---

## A campaign, after the recordings are gone

Every engine above holds the data. A multi-year campaign cannot: the recordings
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

---

## When the blocks are not enough

`ParameterSpace.raw` takes a bind function outright:

```python
space = ParameterSpace.raw(
    latents=[Latent("x", init=x0, prior=dist.Normal(0.0, 1.0))],
    bind=lambda pipeline, values: ...,   # anything, as long as the treedef survives
)
```

The structural checks still apply — a bind function that quietly changes the
pipeline's pytree structure is still caught — but the per-selector checks
cannot run, because there are no selectors to inspect. Reach for it when the
declarative form genuinely cannot express what you need, not to save a line.

---

## Tutorials

The reference above says what each piece is. These walk one problem through end
to end, in the order you would actually do it, with the scripts' real output:

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Tutorial
  - What it covers
* - [An exact posterior for a big linear block](tutorial-gcr.md)
  - 256 sky pixels, no chain. Checking the linearity claim, reading κ before
    choosing `tol`, `iterative_gls` for the covariance, `gcr_sample` for the
    draws — then widening the beam until the prior takes over, and watching
    every diagnostic say so.
* - [A gradient posterior, and how to tell it is wrong](tutorial-nuts.md)
  - Three nonlinear beam parameters. **It fails first**: `r_hat = 840`,
    `n_eff = 2`, nothing raised. Diagnosing that — two hypotheses, one right —
    is most of the page, and the fix is one line.
:::

---

## Run it

[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
does all three on one twin, from truth to recovery: a beam derived from two
scalars, a gain tied across two stages in log space, and a sky map declared
linear and solved by CG. The instrument description is written once at the top
and never edited.

```bash
uv run python examples/inferring_anything.py
```

The figures on this page come from
[`docs/_generate_inference_figures.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/_generate_inference_figures.py),
which runs the same code rather than illustrating it.
