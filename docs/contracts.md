# Contracts between stages

Three things in this package are agreements between stages that never import
each other: the shape every refusal takes, an `aux` key a calibrator writes and
a flagger reads, and the way a composite reads its own children's declarations.
None of the three is an operator, so none of them appears in
[the operator catalog](operators.md) — but all three are public, and one of them
decides whether your calibrator survives its first observation.

## One base class for every refusal

Every error this package raises derives from `DirtError`, so one `except`
clause catches the whole family. Each subclass *additionally* derives from the
closest builtin, so a generic handler written before rheplicant existed keeps
working.

:::{list-table}
:header-rows: 1
:widths: 24 16 60

* - Exception
  - Also a
  - Raised when
* - `DirtError`
  - —
  - base of everything below; never raised on its own
* - `StateValidationError`
  - `ValueError`
  - a *structural* problem: wrong ndim or dtype, an `aux` mask that cannot
    compose, a `coords.time` that cannot resolve its own sampling. Never for
    traced array *values*, so validation stays jit-safe
* - `PipelineError`
  - `ValueError`
  - a `Pipeline` was misconfigured: empty, a bad stage type, a name collision,
    a `must_precede` violated among its own `names`
* - `AssemblyError`
  - `ValueError`
  - a provided operator set cannot be assembled on the template
* - `AmbiguousNodeError`
  - `AssemblyError`
  - a node id was used as an address but holds more than one operator
* - `MissingKeyError`
  - `RuntimeError`
  - an operator needed randomness and `State.key` is `None`
* - `DataIngestionError`
  - `ValueError`
  - a file could not be read, or its contents contradict what the caller
    declared about them
* - `ParameterSpaceError`
  - `ValueError`
  - a parameter space was declared inconsistently — the one member of the
    family that is **not** on `rheplicant.core`'s surface (see below)
* - `LinearityRefused`
  - `ParameterSpaceError`
  - `check_linearity` measured a departure from linearity. Catch it as a
    `ParameterSpaceError` as before; catch it by name to read `.errors`,
    `.rtol` and `.failed` instead of its sentence
:::

Four of them, raised by four different layers and caught by one clause:

```python
import jax.numpy as jnp
from rheplicant import Coordinates, DirtError, Pipeline, State
from rheplicant.core import At
from rheplicant.radio import (CWCalibrationOperator, GainOperator, NoiseOperator,
                              SkyOperator, assemble)

state = State(coords=Coordinates(time=jnp.linspace(0.0, 60.0, 8),
                                 freq=jnp.linspace(60e6, 85e6, 16)))   # no key

def attempt(label, thunk):
    try:
        thunk()
    except DirtError as e:
        print(f"{label:12} {type(e).__name__:22} also ValueError: {isinstance(e, ValueError)}")

attempt("assembly", lambda: assemble(                     # cw_tone must precede gain
    SkyOperator(amplitude=jnp.array(1.0)), GainOperator(gain=jnp.array(1.0)),
    At("noise", CWCalibrationOperator(amplitude=5e3, tone_freq=70e6, line_width=1e6))))
attempt("pipeline", lambda: Pipeline())                   # nothing to run
attempt("state",    lambda: Coordinates(time=jnp.zeros((2, 2)), freq=jnp.zeros(4)))
attempt("randomness", lambda: NoiseOperator(sigma=jnp.array(0.5))(
    state.with_data(jnp.zeros((8, 16)))))                 # state.key is None
```

```text
assembly     AssemblyError          also ValueError: True
pipeline     PipelineError          also ValueError: True
state        StateValidationError   also ValueError: True
randomness   MissingKeyError        also ValueError: False
```

**Read the last column.** `MissingKeyError` is the one refusal that is not a
`ValueError`, and the difference is not cosmetic: it reports a state that was
never given a key at all, which is a missing precondition rather than a bad
value, so it derives from `RuntimeError`. A handler spelled `except ValueError`
therefore catches every other refusal in the package and walks past exactly
that one — into a traceback from wherever the state came from, which is not
where the fix is. `except DirtError` catches all of them.

**Where the names live.** All of them are defined in `rheplicant.core.errors`.
Every one except `ParameterSpaceError` is re-exported from `rheplicant` and
`rheplicant.core`; `AssemblyError` and `AmbiguousNodeError` are *additionally*
re-exported from `rheplicant.core.graph`, which is where they used to be
defined, so `from rheplicant.core.graph import AssemblyError` still resolves.
`ParameterSpaceError` is deliberately absent from `core`'s surface — a
parameter space is not a concept `core` has — and reaches you through
`rheplicant.inference`. That is worth knowing before you write
`except ParameterSpaceError` around a `core` call: you would catch nothing, and
could not import the name from there to try.

## Protected channels: keeping a known calibrator out of the flags

A continuous-wave calibration tone is a narrow, bright, persistent line —
which is, from a flagger's point of view, the definition of RFI. Both shipped
flaggers duly flag it, and `flagging` sits **downstream** of `cw_tone` on the
same trunk. Without a contract between them, the pipeline that is supposed to
*use* the calibrator destroys it on the first observation, and the symptom is a
slightly worse calibration rather than an error.

The mechanism is an `aux` channel rather than a flagger setting, and that is
the whole design decision: the operator that **injects** the tone knows which
channels it went into and writes the protection itself; the flaggers **read**
it if it is there. A flagger has no way to tell a calibration tone from RFI,
and the injecting operator has no way *not* to know. Put the switch on the
flagger instead and it becomes a setting the user must remember to turn on for
every run — the kind that gets forgotten exactly once and then never noticed.

:::{list-table}
:header-rows: 1
:widths: 34 66

* - Name
  - What it is
* - `PROTECTED_KEY`
  - the `aux` key itself, `"protected"`. A boolean `(n_freq,)` channel mask, or
    a full `(n_time, n_freq)` waterfall for a calibrator that drifts
* - `protect(aux, mask)`
  - the **write** side: a new `aux` with `mask` OR-ed into whatever is already
    there
* - `unflag_protected(flags, aux)`
  - the **read** side: `flags` with the protected channels cleared
* - `reduce_protection(mask, n_chunk)`
  - bring a waterfall mask across an average into chunks of `n_chunk`
:::

All four are exported from `rheplicant.radio`. Nothing in a normal pipeline
calls them by hand — `CWCalibrationOperator` calls `protect`, both flaggers
call `unflag_protected`, and `BackendOperator` calls `reduce_protection` — so
the first thing to see is the contract working with no code of yours in it:

```python
import jax, jax.numpy as jnp
from rheplicant import Coordinates, Environment, State
from rheplicant.radio import (CWCalibrationOperator, FlaggingOperator,
                              PROTECTED_KEY, SkyOperator, assemble)

state = State(coords=Coordinates(time=jnp.linspace(0.0, 500.0, 6),
                                 freq=jnp.linspace(60e6, 85e6, 64)),
              env=Environment(temperature=jnp.array(280.0)),
              key=jax.random.key(0))

twin = assemble(SkyOperator(amplitude=jnp.array(300.0)),
                CWCalibrationOperator(amplitude=5000.0, tone_freq=70e6,
                                      line_width=0.5e6),
                FlaggingOperator(threshold=1000.0))
out = twin(state)
print("mask", out.aux[PROTECTED_KEY].shape,
      "protected:", int(out.aux[PROTECTED_KEY].sum()),
      "flagged:", int(out.aux["flags"].sum()))

stripped = out.replace(aux={k: v for k, v in out.aux.items() if k != PROTECTED_KEY})
unprotected = FlaggingOperator(threshold=1000.0)(stripped).aux["flags"]
print("same flagger, no mask -> flagged:", int(unprotected.sum()),
      "in channels", jnp.where(unprotected.any(axis=0))[0].tolist())
```

```text
mask (64,) protected: 5 flagged: 0
same flagger, no mask -> flagged: 12 in channels [25, 26]
```

Twelve flagged samples — every sample of the two channels the 5000 K line
actually peaks in. That is the calibrator, gone.

**"Narrow" is not "one channel", and it is not always the same channels.** The
tone is observed through the spectrometer's channel response, so it wets a set
of channels (five, above, at the default `protect_floor` of 1e-2 of the peak);
if it drifts, that set moves during the run. Both mask shapes therefore matter:
`(n_freq,)` for a line that stays put, `(n_time, n_freq)` for one that does not.
Which channels a given tone wets is `CWCalibrationOperator`'s to decide — it is
the only thing on the path that knows the lineshape.

A waterfall mask is **bound to the time axis it was written on**: row `i` names
the channels the calibrator wet at sample `i` of the axis that existed when the
mask was built. Any stage that changes the number of samples leaves it stale,
and `unflag_protected` refuses a stale one rather than broadcasting it:

```python
from rheplicant import StateValidationError
from rheplicant.radio import BackendOperator, reduce_protection

drifting = assemble(SkyOperator(amplitude=jnp.array(300.0)),
                    CWCalibrationOperator(amplitude=5000.0, tone_freq=70e6,
                                          line_width=0.5e6, drift_rate=4.0e3))(state)
print("drifting mask:", drifting.aux[PROTECTED_KEY].shape)

def decimate(s, n=2):                      # a stage that knows nothing about aux
    n_out = s.data.shape[0] // n
    return s.replace(data=s.data.reshape(n_out, n, -1).mean(axis=1),
                     coords=s.coords.replace(
                         time=s.coords.time.reshape(n_out, n).mean(axis=1)))

stale = decimate(drifting)
try:
    FlaggingOperator(threshold=1000.0)(stale)
except StateValidationError as e:
    print("REFUSED:", str(e)[:96], "...")

fixed = stale.replace(aux={**stale.aux,
                           PROTECTED_KEY: reduce_protection(stale.aux[PROTECTED_KEY], 2)})
print("re-derived:", fixed.aux[PROTECTED_KEY].shape,
      "flagged:", int(FlaggingOperator(threshold=1000.0)(fixed).aux["flags"].sum()))
```

```text
drifting mask: (6, 64)
REFUSED: aux['protected'] is a waterfall mask over 6 time samples but the flags cover 3. A waterfall mask ...
re-derived: (3, 64) flagged: 0
```

The full message names both ways out — *drop the mask or re-derive it* — and
only one of them keeps the calibrator protected. `reduce_protection` is that
re-derivation, which is why it lives next to the convention it depends on
rather than being re-invented by every stage that reshapes a run. The two mask
shapes go different ways, and that asymmetry is the whole content of the
function: a `(n_freq,)` channel mask comes back unchanged, because it names
channels and no change to the time axis can stale it; a waterfall mask is
reduced with **`any`** over each chunk, because the chunk's average carries a
contaminated sample's power whether or not the tone was on for the rest of the
chunk. `all` would unprotect a chunk the tone contaminated in two rows of
three, which is the failure protection exists to prevent.

On the shipped trunk you never write that line: `BackendOperator` reduces a 2-D
`aux["protected"]` itself, with the same `any`, alongside `aux["flags"]` —

```python
with_flags = drifting.replace(aux={**drifting.aux,
                                   "flags": jnp.zeros(drifting.data.shape, bool)})
averaged = BackendOperator(n_chunk=2)(with_flags)
print("BackendOperator(n_chunk=2):", averaged.data.shape,
      averaged.aux[PROTECTED_KEY].shape)
```

```text
BackendOperator(n_chunk=2): (3, 64) (3, 64)
```

— so `reduce_protection` is for the stages `BackendOperator` is not: your own
decimation, a subset selection, anything that reshapes time on its way past.

:::{admonition} Two things protection is not
:class: warning
**It removes the flag, not the tone's influence on the flagger's own fit.**
`MomentRFIFlaggingOperator` fits a surface to the waterfall, and a 5000 K spike
biases that fit near the tone whether or not the spike is flagged afterwards.
Protecting a channel is not the same as excluding it from the estimator, and
only the first is claimed here.

**It is not free.** A channel that is protected is a channel where genuine RFI
now survives into the data. That is the deliberate trade — the tone channel is
known-bright by construction, so a flagger's verdict there carries no
information anyway — but it is a trade, and the raw data still shows what
happened.
:::

## Reading a tree's own declarations

Every operator carries two ClassVars, `requires` and `provides`, naming the
`State` paths it reads and writes. `rheplicant.core.contract` is where they
stop being prose: it walks a *built* operator tree and answers **which stages
declare a given path**, so a caller can refuse a composition on the strength of
what its stages say about themselves rather than on a hard-coded list of
classes.

```python
import jax.numpy as jnp
from rheplicant.core import RANDOMNESS, stages_requiring, walk_operators
from rheplicant.radio import (ForegroundOperator, GainOperator, GlobalSignalOperator,
                              NoiseOperator, assemble)

twin = assemble(
    GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5),
                       ref_freq=70e6),
    GainOperator(gain=jnp.array(1.1)),
    NoiseOperator(sigma=jnp.array(0.5)),
)
for label, op in walk_operators(twin):
    print(f"{label!r:26} {type(op).__name__}")
print("stochastic:", [(l, type(o).__name__) for l, o in stages_requiring(twin, RANDOMNESS)])
```

```text
''                         Assembly
''                         Pipeline
'astro_sum'                SumOperator
'astro_sum/global_signal'  GlobalSignalOperator
'astro_sum/foregrounds'    ForegroundOperator
'gain'                     GainOperator
'noise'                    NoiseOperator
```
```text
stochastic: [('noise', 'NoiseOperator')]
```

`walk_operators` yields `(label, operator)` for the root and everything nested
in it; `stages_requiring(op, path)` is the filter over that walk. Labels are
`/`-joined stage names, which is what lets a refusal quote a graph node id
instead of a class name — `'astro_sum/foregrounds'` above. The `Assembly`
wrapper and the trunk `Pipeline` both come back as `''` because they are
structural spine with no name of their own; only named composites (`Pipeline`,
`SumOperator`, `SelectOperator`) contribute a segment.

The walk is by **pytree position**, not by the composite spine `assemble` folds
along. That is deliberate: this is a safety check, so it must not miss a stage
held by a composite type nobody taught it about.

**One path is enforced, and it is `"key"`** — spelled `RANDOMNESS`. An operator
that names it in `requires` draws randomness through `State.next_key()`, and
that is a property no shape check, no linearity check and no rank test can see.
It is the reason every inference exit refuses a twin containing one: inference
closes the model over *one* template state, so the draw happens once and the
same frozen realisation rides into every prediction compared against the data.
Adding a constant field is exactly affine, so `check_linearity` reports residual
0.0 and `identifiability` reports full rank — both exits wrong by the same
amount with no diagnostic moving. `stages_requiring(model, RANDOMNESS)` is how
that refusal finds the drawing stage, and it is how you would write your own:

```python
stochastic = stages_requiring(twin, RANDOMNESS)
if stochastic:
    raise ValueError(f"not a deterministic model: {stochastic[0][0]!r} draws")
```

```text
ValueError: not a deterministic model: 'noise' draws
```

Because the detector is the operators' own declaration, a new stochastic
operator is covered the day it declares what it reads, with nothing to update.
What it therefore cannot catch, stated rather than implied: an operator that
draws randomness *without* declaring `"key"`, and one hiding a draw inside a
static field. Nothing static can see either — there is no numerical symptom,
which is the premise of the whole guard. `tests/test_operator_declarations.py`
checks mechanically that the shipped operators declare honestly; a user-written
operator is the user's declaration to make.

**The rest of `requires`/`provides` stays descriptive, deliberately.** The
obvious next step — refusing an operator whose `requires` names a path the
template state does not supply — is not implementable against the shipped set,
and the counter-example is in the package: `GroundPickupOperator` declares
`"env.temperature"` and then documents a `t_ground` fallback for when it is
missing, so the declaration means "reads if present", not "needs". A blanket
availability rule would refuse a model the package itself describes as
legitimate. `provides` is weaker still — 25 of the 31 classes that declare it
declare exactly `("data",)`, which distinguishes almost nothing the graph's own
source/transform kinds do not already say.
