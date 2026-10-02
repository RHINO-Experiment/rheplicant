# Parameter spaces: what you infer, and how it enters

```{include} _bayesmith-note.md
```

An inference problem is declared in two separate statements. Keeping them
separate lets one latent drive several stages, or a leaf be a transform of
several latents, without a new operator for each combination.

- [Two words](#two-words)
- [The three shapes](#the-three-shapes)
- [A beam from two numbers](#a-beam-from-two-numbers)
- [What gets checked](#what-gets-checked)
- [When the blocks are not enough](#when-the-blocks-are-not-enough)

---

## Two words

The two statements are `Latent` and `Bind`.

`Latent`
: **A named quantity you infer.** It is what a sampler draws or an optimizer
  steps: a name, an initial value (which fixes its shape and dtype), optionally
  a prior, optionally a declaration that it enters the model linearly. A latent
  knows nothing about the pipeline. `log_gain` is a latent; so is a
  10⁴-element vector of sky `alm` coefficients.

`Bind`
: **A rule turning latents into pipeline leaf values.** It names the latents it
  consumes, the leaves it writes, and optionally the function between them. A
  bind knows nothing about priors.

A pipeline leaf is what the instrument model holds. A latent is what you
chose to infer. Keeping them apart lets you re-parameterize without changing
the instrument model.

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

The arrows are many-to-many: two latents produce one leaf, and one latent
produces two. A scheme that attaches priors directly to leaves cannot express
that.

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
One physical quantity, two stages.
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

A `fn` returning a single array is written to every selector in `into`, so a
tie is one line. Returning a tuple addresses each selector separately.

### Broadcast or distribute: `fan=`

With two leaves and a 2-vector, both readings are available and they are
different physics. With the default `fan=None` the only thing separating them
is a Python container type. Measured on two scalar leaves (an antenna
efficiency and a gain, arranged so both enter the prediction as a bare
product) driven by the same `[2, 5]`:

| `fn` returns | mode | what each leaf gets | `pred[0, 0]` |
|---|---|---|---|
| `lambda v: v` | broadcast | both leaves get `[2, 5]` | 4.0 |
| `lambda v: list(v)` | distribute | efficiency `2`, gain `5` | 10.0 |

`v` and `list(v)` are the same numbers. One is a JAX array and one is a Python
list of its elements. That difference is invisible in the values, in every
shape, and to `check_linearity` and `identifiability`, and it selects between
a tie and an element-wise split. A caller who meant "write this whole vector
into both leaves" and wrote `list(v)` gets a finite, correctly-shaped, wrong
model with no error: off by a factor of 2.5 here, and by whatever the leaves
happen to be worth in general.

`fan=` declares the intent, so it can be checked:

```python
into = (lambda p: p["loss"].efficiency, lambda p: p["gain"].gain)

Bind("v", into=into, fn=lambda v: v, fan="broadcast")   # both leaves [2, 5] -> 4.0
Bind("v", into=into, fn=list,        fan="distribute")  # 2 and 5        -> 10.0
```

A declaration that contradicts what `fn` produced is refused, and the message
names both sides:

```text
Bind for ('v',) declares fan='broadcast' — ONE value written into all 2 `into`
selectors — but `fn` returned a list of 2, which is a distribution. A tie writes
one value into every leaf and a distribution gives each leaf its own, which is
different physics — and the two spellings differ only by a Python container
type, so nothing downstream would report the mismatch: the model would be
finite, correctly shaped and wrong. Fix `fn`, or change the declaration to the
mode you meant.
```

The mirror case, `fan="distribute"` where `fn` returned a single array ("which
would be tied into all of them"), is refused the same way. An unknown mode is
refused at construction, before anything evaluates, and the message names the
two that exist. Those two are also exported as `BROADCAST` and `DISTRIBUTE`, for
callers who prefer constants to literals.

`fan=None` remains the default, and the mode is then inferred from the
container type. `Bind` is public and appears in every example on this page, so
refusing by default would break all of them. With `fan=` set the mode is
declared, and a declaration that `fn` contradicts raises.

:::{note}
**A single `into` selector.** The length test that separates the modes,
`len(produced) == len(into)`, is satisfied by a length-1 container under either
intent, so the mode cannot be inferred. The container is unwrapped on a guess,
and the guess raises a warning, not an error:

```text
AmbiguousFanWarning: Bind for ('v',) has one `into` selector and `fn` returned a
list of 1, so which fan-out mode was meant cannot be told from the length: a
1-container matches a single selector under 'distribute' AND under 'broadcast'.
Unwrapping it, which is the only reading that can reach an array leaf — but say
fan='distribute' to declare that, or fan='broadcast' to be refused here if this
ever stops being a container.
```

A Python list is not an array leaf, so unwrapping is the only reading that can
yield a valid pipeline; broadcasting the container would change the pipeline's
pytree structure and be refused by `validate()`. There is no wrong answer to
prevent here, only an undeclared one, so refusing would break working code
without preventing a wrong result. `fan="distribute"` declares it and the
warning goes.
:::

---

## A beam from two numbers

The latents are a beam width and a pointing offset; the pipeline leaf they
feed is the full response matrix. The instrument description is unchanged.

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
the same posterior against the likelihood Fisher forecast computed at the
truth. Measured, `sd(NUTS) / sd(Fisher)` is 0.985 for `fwhm` and 0.975 for
`offset`. They agree because the model is close to linear in its parameters
and because these two priors say almost nothing at this noise level. The
second condition is not general; see below.
:::

:::{figure} _static/inference-posterior-dark.svg
:figclass: only-dark
:alt: NUTS posterior over two beam latents, recovering the truth
:width: 100%

Left: the NUTS posterior over the two latents, with the truth marked. Right:
the same posterior against the likelihood Fisher forecast computed at the
truth. Measured, `sd(NUTS) / sd(Fisher)` is 0.985 for `fwhm` and 0.975 for
`offset`. They agree because the model is close to linear in its parameters
and because these two priors say almost nothing at this noise level. The
second condition is not general; see below.
:::

### Fisher forecast and posterior

A Fisher matrix and a posterior are the same object only when the prior is
flat over the region the data picks out. That is the case here. Measured,
`offset`'s declared `Normal(0, 0.4)` contributes 5×10⁻⁷ of the total precision,
and `fwhm`'s `Uniform(0.15, 0.70)` is flat across a range 512× the posterior
width. Both priors are negligible, so the likelihood's information determines
the posterior and the two panels coincide.

With an informative declaration the two differ and nothing reports it, because
`fisher_information` builds `F = JᵀN⁻¹J`, which describes the data alone. The
`Normal(250.0, 50.0)` noise-wave temperature in
[the linear machinery](inference-linear.md) is such a declaration, and
`wiener_solve` solves with it as `S`. Pass the space and `fisher_information`
adds each declared Gaussian prior's own curvature at that latent's span:

```python
# likelihood only: the Cramér-Rao bound, no prior in it
fisher_information(forward, fitted, noise_std=0.02)              # kind="fisher"

# the posterior precision NUTS is sampling from, same declaration
fisher_information(forward, fitted, noise_std=0.02, space=space) # kind="posterior_precision"
```

`kind` travels through `parameter_covariance` into `"covariance"` and
`"posterior_covariance"`, so the object records which quantity a `sigma()`
reports.

The beam space above cannot use `space=`: `fwhm` is declared `Uniform`, which
has no quadratic form, and substituting its variance would report a Gaussian
posterior for a prior with no curvature. That is refused by name, in the same
words `wiener_solve` uses for a non-conjugate prior. For a space like this one
the likelihood Fisher is the quantity available, and it is what the figure
shows.

Sample sites are named by their latents, so the samples come back keyed by
the coordinates the model was declared in. A `log_gain` latent is one site
named `log_gain`, even though its value reaches a leaf called `gain`, and it
stays one site if it drives five stages.

The covariance is addressed by the same names:

```python
cov = parameter_covariance(fisher_information(forward, fitted, noise_std=0.02))
cov.sigma("fwhm")             # not cov.matrix[0, 0]
cov.block("fwhm", "log_gain")  # the cross-covariance, by name
```

---

## What gets checked

Most of these failure modes produce a finite, correctly-shaped, wrong
inference, and the rest raise far from their cause. So they are all errors, and
all of them are caught at declaration or build time. `validate()` runs on shapes alone
(`jax.eval_shape`), once per build rather than per evaluation, so there is no
reason to make it skippable.

| Checked | Without the check |
|---|---|
| Latent names are unique | NumPyro sites silently overwrite each other |
| Every `Bind` names a declared latent | `KeyError` from deep inside a trace |
| Every latent reaches the model: named in a binding, or (for a raw bind) shown to move it | It samples and returns the prior |
| No leaf is written twice | `eqx.tree_at` lets the last write win, silently |
| A raw bind is not combined with declarative bindings | `bind()` would apply only the raw one |
| Binding preserves every leaf's shape and dtype kind | A treedef encodes neither; the value is broadcast |
| Every selector reaches a real array leaf | Confusing failure inside `tree_at` |
| Produced shape matches its target | A broadcast that is shaped right and means nothing |
| Produced dtype kind matches | Complex into a real leaf is a modelling error, not a cast |
| A prior's shape matches its latent's | Sampling a different-sized thing than you bind |
| Binding preserves the pytree structure | `vmap`, `jit` and Fisher flattening all break |

Every one of them raises `ParameterSpaceError`, and so does most of the rest of
this page. It is exported from `rheplicant.inference`, where it is raised, and
not from the top-level `rheplicant`. Its sibling error classes are the reverse:
`AssemblyError`, `PipelineError`, `StateValidationError` and the rest are
re-exported from `rheplicant` and not from `rheplicant.inference`. Either of
these works. The second path is the same for all of them, since
`rheplicant.core.errors` is where every one is defined:

```python
from rheplicant.inference import ParameterSpaceError
from rheplicant.core.errors import ParameterSpaceError
```

### Refusals outside `validate()`

The table is `validate()`'s inventory, and `validate()` sees shapes at build
time. Some failure modes are invisible to it and are checked where they can be
seen. Unchecked, each gives the same kind of result as the ones above: finite,
correctly shaped, and wrong.

**A `fan=` that contradicts what `fn` produced** is caught when the binding
evaluates, because that is the first moment anything knows what `fn` returned;
see [the container type is the
physics](#broadcast-or-distribute-fan) above.

**A forward model that draws its own randomness** is refused on entry to every
inference exit. Inference closes the model over one template `State`, so a
stage consuming the PRNG key draws one realisation and adds that same frozen
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

The corruption is exactly affine, so every guard on this page passes it.
Measured on an 8×8 grid (sky 100 K, truth `g = 1.1`, 2 K measurement scatter, a
`NoiseOperator(sigma=20)` left in the twin and nothing else changed), through
`wiener_solve` for the estimate and `fisher_information` →
`parameter_covariance` for the bar:

| twin handed to the exit | `g` | reported σ |
|---|---|---|
| deterministic (`.without("noise")`) | 1.1002 | 0.002500 |
| stochastic stage left in | **1.0735** | 0.002500 |

The estimate moved by 10.6 of its own error bars, and the two bars are equal
bit for bit: `sd_clean == sd_corrupt` is `True`. Both exits of the workflow are
wrong by the same amount, and no diagnostic moves. The magnitude is a property
of the draw; the invisibility is structural.

The repair the message names is the supported one, and it is one line:

```python
twin_for_inference = twin.without("noise")   # Assembly.without: re-runs assemble()
```

The detector is the operators' own declaration, `RANDOMNESS` in `requires`, so
a new stochastic operator is covered once it declares what it reads. The
detector cannot see an operator that draws without declaring `"key"`, or a draw
closed over inside a static field, and neither has a numerical symptom.
`tests/test_operator_declarations.py` checks that the shipped operators'
declarations are accurate.

**An `observed` the prediction would have to broadcast against** is the third.
Every shipped exit already applies this check. `(24, 8) - (8,)` is legal NumPy
and a wrong residual, with no symptom: the loss converges, the Fisher matrix
inverts, the posterior looks healthy.

The checks for these last two are public:
`refuse_stochastic_stages(pipeline, caller)` and
`check_observed_shape(prediction_shape, observed, predictor=...)`. The shipped
exits call them already, so call them yourself only when writing a new exit.

---

## When the blocks are not enough

`ParameterSpace.raw` takes a bind function outright:

```python
space = ParameterSpace.raw(
    latents=[Latent("x", init=x0, prior=dist.Normal(0.0, 1.0))],
    bind=lambda pipeline, values: ...,   # anything, as long as the treedef survives
)
```

The structural checks still apply: a bind function that changes the pipeline's
pytree structure is still caught. The per-selector checks cannot run, because
there are no selectors to inspect. Use it when the declarative form cannot
express what you need, not to save a line.
