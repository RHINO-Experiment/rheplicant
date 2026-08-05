# Changelog

## Unreleased

### Averaging brings `aux` across the chunk axis, or refuses the key it cannot

`BackendOperator` averaged `data` into time chunks and rewrote `coords.time`,
and left `state.aux` alone. Every per-time array in `aux` was stale the moment
it did. Measured on one `(6, 4)` fixture at `n_chunk=3`:

| `aux` entry | before | after |
|---|---|---|
| `aux["flags"]` | carried at `(6, 4)`; `FlaggedNoise.std` refused two stages later | reduced to `(2, 4)` |
| `aux["protected"]`, 2-D waterfall | carried at `(6, 4)`; the next `FlaggingOperator` refused it, naming the staleness | reduced to `(2, 4)` |
| `aux["switch"] = [0,1,0,1,0,1]` | carried at `(6,)`, **no error anywhere**, and each chunk spanned both switch positions | refused, by name |

The third is why this is a defect rather than an inconvenience. The first two
are loud only because something downstream knows what shape those keys are
supposed to have; a key the package has never heard of has no such consumer, so
a wrong-length array rides to the end of the run misaligned with `data` and
`coords.time` and nothing can tell.

Both reductions are `any` over the chunk, and for the same reason: this
placeholder averages every sample in the chunk, flagged ones included, so the
chunk mean is contaminated if ANY sample in it was. `all` would call a chunk
clean with two of three samples RFI-blasted. When the mean learns to exclude
flagged samples the two must change together. The protection half lives in
`rheplicant.radio.protection.reduce_protection` — the re-derivation that
module's own refusal already told the caller to perform — where a `(n_freq,)`
channel mask comes back unchanged because it names channels, not samples.

**The refusal is on shape, not on values.** The register's proposal was to
refuse an integer per-time array whose chunk is not constant; that is a value
check, so it could not run under `jit`, and NaN walks through any comparison
built on it. Refusing by leading axis is simpler, jit-safe, and strictly
stronger — a constant `[7,7,7,9,9,9]` is refused too, because averaging it
would hand back `[7.0, 9.0]`, a float where an index was.

It is deliberately conservative: an array whose leading axis merely *coincides*
with `n_time` is refused as well, since nothing distinguishes it from a
genuinely per-time one. Two escapes are in the message (reduce it first, or pop
it and put the reduced version back), and two cases are exempt — `n_chunk=1`,
which does not change the axis at all, and `aux["snapshot/..."]`, which is
deliberately a record of the axis that existed before and is the one entry
whose staleness is the point.

`provides` and `requires` now name `aux.flags` and `aux.protected`.

### A non-finite pointing is refused, because the adjoint would not be loud

`GeneralPointingProjector` validated that `coords.pointing` and
`coords.extra["lst_deg"]` were *present*, never that they were finite. The two
directions then failed asymmetrically, measured on one fixture:

| direction | result on all-NaN pointing |
|---|---|
| `forward` | every TOD sample NaN — loud, a user notices |
| `adjoint` | a finite, correctly shaped, identically **zero** map (the honest answer is 11.4) |

Map-making on corrupted pointing returned a clean-looking empty result and every
downstream `isfinite` check passed. Both directions now refuse, from the entry
they share; the message names the field and counts the bad samples, since one
dropped ephemeris row in an otherwise good run is the realistic case.

The check reads concrete values, so it steps over traced arrays **per field** —
a mixed call, where the jitted argument is traced and the rest of the
coordinates are concrete, is the common one. With every field traced nothing is
checked, which is the honest limit of a value check on a differentiable path and
is pinned as such.

### `CalLoadOperator` takes a per-sample temperature, and a recording can supply it

`read_rhino_observation` parsed the thermistor log, interpolated it onto the SDR
axis and refused a recording whose readings were short or non-finite — and then
`to_state` dropped it, so the loads' temperatures were parsed, validated and
discarded and the warm/hot-load noise-wave path had no route from a recording.

`t_load` now also accepts an explicit `(n_time, 1)` column, and
`rheplicant.radio.cal_load_operators(obs)` builds one operator per switched
load from a recording. A bare 1-D array is still read as per-**frequency**,
matching `NoiseWaveOperator`'s temperature leaves; `(n_time, n_freq)` is
deliberately **not** accepted.

**Breaking:** `rhino.to_state` now stores `coords.time` as seconds since the
first kept sample, with the absolute epoch in `meta["time_epoch_unix_s"]`. Code
reading `coords.time` as a unix epoch is wrong by ~1.75e9 and nothing raises.
The change is what makes a float32 time axis usable at all — a unix-second axis
has 128 s of float32 resolution, which merged samples before averaging.

### Eighteen construction-time refusals had never been executed — and eight of them let NaN through

The raise audit
(`tools/raise_audit.py`) found 18 `raise` statements in the construction-time
configuration-validation family that the suite had never run: the guards that
refuse `n_pix=0`, `validation_fraction=1.0`, an empty `ParameterSpace`, a
`cg_maxiter` that is not an int. They are now covered by
`tests/inference/test_inference_construction_guards.py` (57 tests) and
`tests/radio/test_radio_construction_guards.py` (20).

**This family is not the coords family, and the difference was measured rather
than assumed.** `tests/radio/test_coords_guard_family.py` derives its population
from the source because its guard is one sentence copy-pasted nine times, so one
case per carrier reaches every raise in the family. Enumerating `__check_init__`
with `ast` here finds **31 classes carrying 70 raises across 62 distinct
messages** — one case per class would reach 31 of 70, and a table claiming to be
the family would be honest about under half of it. Five of the eighteen raises
covered are not in a `__check_init__` at all (`NeuralPosterior.create` validates
in a classmethod, `train_posterior` in a module-level function,
`ParameterSpace._resolve_targets` at validate time), so a `__check_init__`-derived
table would also have excluded them while looking complete. The cases are
therefore hand-written — with one exception. Seven of the 62 messages *are*
copy-pasted, four of them in this batch, and those get the derived treatment:
`test_the_copied_guards_have_not_grown_a_new_copy` enumerates the carriers of
`ref_freq must be > 0` (three copies, two previously tested), `n_pix must be a
positive int`, `learning_rate must be > 0` and `n_steps must be a positive int`
(two copies each, one previously tested each) and fails, naming the offender, if
a further copy appears.

**`nan` defeated every comparison-based guard here, and it is now fixed.**
`nan <= 0` is `False`, so `if x <= 0: raise` did not fire and the NaN became
configuration. Eight sites, all now inverted to `if not x > 0`:

| site | what used to happen |
|---|---|
| `AdamCalibrator.learning_rate` | every Adam iterate NaN |
| `GradientCalibrator.learning_rate` | same guard, copied |
| `RadiometerNoise.floor` | **the floor was silently dropped** — `if self.floor > 0.0` is also False, so `std` came back finite, correctly shaped and simply un-floored |
| `RadiometerNoise.channel_width` / `integration_time` | every noise weight NaN |
| `NeuralPosterior.create(n_components=)` | refused three frames deeper by `eqx.nn.MLP`, with a `TypeError` about shape sequences instead of the sentence written for the caller |
| `IonosphereOperator.ref_freq` | the whole band NaN |
| `PowerLawSkyModel.ref_freq` / `ForegroundOperator.ref_freq` | the same guard, copied twice more |

`RadiometerNoise.floor` is the one worth reading twice, and it is why "NaN
poisons the output anyway" was not a defence: it was the only case producing no
NaN downstream at all, so a caller passing `floor=nan` got a working noise model
with their argument quietly discarded.

The package already contained the one-line fix and used it three times —
`if not 0.0 <= self.beta1 < 1.0`, `if not 0.0 <= validation_fraction < 1.0` and
`if not 0.0 <= self.occupancy <= 1.0`. The tests that pinned the gap were
written to fail when it was closed, with the remedy named in each docstring, so
the fix arrived as a deliberate test change rather than a silent one.

Two smaller notes. `n_components`, `n_steps` in `train_posterior` and the
`isinstance`-typed guards (`n_pix`, `cg_maxiter`, `AdamCalibrator.n_steps`) do
refuse NaN, but the last three do it by the type check rather than the
comparison beneath it — correct by accident, so pinned separately. And
`AdamCalibrator(n_steps=True)` is accepted as one step, since `isinstance(True,
int)`.

**Test file basenames must be unique across `tests/`.** There is no
`__init__.py` under `tests/`, so pytest imports test modules by bare basename
and two files sharing one cannot both be collected — an `EXIT=2` at collection
that appears only when both are in the same run, which is why the directory
name is carried in the basename of both files above.

### `coords.time` is refused when the dtype it is stored in cannot carry it

**Breaking, and deliberately so.** `Coordinates.__check_init__` now raises
`StateValidationError` for a concrete `time` axis whose stored representable
resolution exceeds `MAX_TIME_RESOLUTION_IN_SAMPLES` (1e-2) of its own smallest
distinct sample gap, and for a non-finite `time` axis. This is the first *value*
check in that container, which is otherwise structural and value-independent.

**What was wrong.** `Coordinates` stores through `jnp.asarray` — float32 unless
x64 is enabled — and a unix second near 1.75e9 has a float32 resolution of
128 s. `read_rhino_observation` produces a unix-epoch axis and `to_state` put it
straight in. Measured, 8 samples 100 s apart, through `BackendOperator(n_chunk=2)`:

```
stored       [1750000000, 1750000128, 1750000256, 1750000256,
              1750000384, 1750000512, 1750000640, 1750000640]   6 distinct of 8
chunk times  [1750000128, 1750000256, 1750000384, 1750000640]
float64 truth[1750000050, 1750000250, 1750000450, 1750000650]
error [s]    [       +78,         +6,        -66,        -10]
```

Two of eight samples merged *before* the average ran, and every chunk timestamp
is wrong — by 78 s of a 100 s cadence at worst. No exception, no NaN, every
shape right. The rounding happens at store time, so nothing downstream can
recover or even detect it: a consumer's own consistency checks compare the
corrupted values against each other and see nothing.

**Why the container and not the consumers.** `CWCalibrationOperator` already
refused this exact axis, and was the only thing in the package that did — one
consumer of two that do arithmetic on `coords.time` (`BackendOperator` is the
other; every remaining `requires=("coords.time", ...)` reads only its length).
A per-consumer check scales with consumers and is a fresh omission each time.

**A previous decision is overturned.**
`tests/radio/test_cw_time_axis.py::test_a_static_tone_does_not_care_what_epoch_the_axis_uses`
asserted that a unix-second axis must be *accepted* when the tone does not
drift, on the grounds that refusing it "would break a pipeline that is entirely
fine". That holds for the operator and not for the axis: the same axis is
already lying about its own samples, and `Coordinates` cannot know which
consumer comes next. The test is rewritten around what is still genuinely that
operator's own — the check runs only when the tone drifts, and it counts a zero
gap that the container deliberately allows.

**Where the two checks differ, on purpose.** The container takes the smallest
*distinct* gap; `CWCalibrationOperator` takes the smallest gap including zero. A
container cannot tell a genuine repeated timestamp from a collision and has no
business refusing the first. Nothing is lost on the motivating defect: rounding
makes every surviving gap a multiple of the resolution, so a uniformly quantised
axis that has collided still shows a smallest distinct gap of one or two grid
steps. What does escape is one isolated close pair merging in an otherwise
coarse axis — the pre-conversion values are gone by then, and only float64 or a
relative axis defends against that.

**Two traps this is built around**, each with a test on both sides.
`np.spacing(float(times.max()))` promotes to a Python float and answers for
float64 (2.4e-7 s at unix seconds) whatever the array holds — blind to the one
thing being guarded; the array's own scalar is used instead. And NaN compares
False against everything, so a NaN gap is not positive and drops out of "the
smallest distinct gap", leaving an all-NaN axis with no gap left to test and a
clean pass through any purely comparison-based guard; non-finiteness is named
first, before any comparison. Traced axes are stepped over rather than forced.

**What this costs a long run, stated rather than papered over.** For a uniform
axis measured from its own start the ratio is `spacing(n·cadence)/cadence`, and
because `spacing(x)` is within a factor of two of `x·2⁻²³` the cadence nearly
cancels: the constraint is on the sample COUNT, of order 1e5 for float32 (in
[8.4e4, 1.7e5]; exactly 2¹⁷ = 131072 at 1 s cadence). A four-hour RHINO run at
1 s is 1.4e4 samples and an order of magnitude clear; the same run at 0.05 s is
2.9e5 and is refused, and needs x64. Making the axis relative buys about five
decimal orders over a unix epoch, not unlimited range.

Cost in eager mode, measured on a 1e5-sample axis: 0.14 ms per
`Coordinates.replace` against 0.03 ms without a time axis. Zero under jit, where
the axis is a tracer and the check steps aside.

`MAX_TIME_RESOLUTION_IN_SAMPLES` moved to `rheplicant.core.coordinates`, since it
describes how `coords.time` is stored rather than what any operator does with
it. `rheplicant.radio.instrument.calibration` re-exports the name unchanged.

### `rhino.to_state` stores time from the start of the run, not from the epoch

**Breaking, on a public function.** `state.coords.time` is now **seconds since
the first kept sample**, and `state.meta["time_epoch_unix_s"]` holds the unix
second it is measured from, so `meta[TIME_EPOCH_META_KEY] + coords.time`
recovers `obs.time_s` exactly. `obs.time_s` on the recording is unchanged and
still absolute. The name is `rheplicant.radio.rhino.TIME_EPOCH_META_KEY`.

This makes the precision defect above structurally impossible on the documented
ingestion path rather than merely detected on it. The subtraction happens in
numpy float64, before the store — subtracting after the store reads the
already-rounded values and is exactly the failure, not a cheaper fix. Measured
on six samples at offsets [0, 100, 250, 450, 700, 1000] s from a 1.75e9 epoch:

```
stored offsets  [0, 128, 256, 512, 640, 1024]
error [s]       [0, +28,  +6, +62, -60,  +24]
```

All six values stay *distinct* here, so no shape, count, dtype or finiteness
check could see it, while individual timestamps are wrong by up to 62 s.
Relative, the offsets are small integers and float32 holds them exactly.

The epoch is the first **kept** sample, not the first in the file: the leading
drop removes samples with no defined switch state, and they are not part of the
run the State describes. `to_state` also now refuses a recording with no
samples, which previously surfaced as a bare `IndexError` from `obs.time_s[0]`.

Every `coords.time` consumer in `src/` was audited for an absolute-time
assumption; there is none. `SiderealFilter` bins by index and `DriftScan` reads
`coords.extra["lst_deg"]`, not `coords.time`.

### `read_rhino_observation` no longer refuses a file over data it discards

`thermistor_columns` is now optional. Omitting it skips the thermistor log
entirely — nothing under `/temperatures` is read, so a file with a malformed
table, a log ending short of the SDR axis, a non-finite reading, or no
temperature group at all is readable for its waterfall. `thermistor_k` comes
back `{}`.

`_thermistors_in_kelvin` was called unconditionally and refused a whole
observation for a thermistor log ending 1 ms short of the SDR axis, or one NaN
row in a used column. Both refusals are argued in that function and both are
right for a caller who wants the temperatures. But `to_state` carries only
`data`, `coords.time`, `coords.freq`, `coords.extra["receiver_input"]`,
`aux["flags"]` and the epoch: `thermistor_k`, `transitions`,
`n_leading_dropped`, `adc_max_i` and `adc_max_q` reach no operator anywhere in
`src/` — audited by grepping each field name. The file was being refused over a
column nothing downstream consumes, while the waterfall, the switch log and the
settling mask in it were intact.

Omitting the map is not a default guess, which is the thing
`_thermistors_in_kelvin` argues against: the label-to-column convention is
shared between writer and reader with nothing in the file to enforce it, so a
declaration is still required to get temperatures at all. Declaring it runs
every check that ran before, unchanged. `read_rhino_observation`'s docstring now
states which fields reach the signal path and which are diagnostic.

**Still open:** wiring `obs.thermistor_k` onto `CalLoadOperator.t_load`, so the
defended quantity does reach the signal path and switched-load noise-wave
calibration works on real recordings. `t_load` is a scalar or `(n_freq,)` while
a load temperature is per-sample, so it needs a `(n_time,)` case that is not
free to disambiguate; and `to_state` returns a State, so wiring an operator from
it either changes its return type or moves the temperature into the State and
so changes that operator's `requires` declaration.

### `fisher_information(space=...)`: the declared prior can reach the one exit that ignored it

`Latent(prior=...)` is the package's single statement of what a latent is a
priori, and every other exit reads it — `to_numpyro_model` samples it,
`wiener_solve` and `gcr_sample` solve with it as `S` and refuse a prior-free
linear latent by name. `fisher_information` never received the `ParameterSpace`
at all, so the declaration could not reach it:

```
declared Normal(10, 5.0)   -> sigma('amp') = 0.00568182
declared Normal(10, 1e-06) -> sigma('amp') = 0.00568182
```

A 5,000,000x tightening of the prior moved the reported error bar by exactly
zero, and nothing in the result said the matrix was likelihood-only.

`space=` adds each declared Gaussian prior's own curvature (`1/scale²`) at that
latent's span, giving the posterior precision at `params`:

```
declared Normal(10, 5.0)   -> 0.00568181   kind='posterior_covariance'
declared Normal(10, 1e-06) -> 0.00000100   kind='posterior_covariance'
```

`space=None` is unchanged and is a real answer rather than a missing argument:
`F = JᵀN⁻¹J` is the **likelihood** Fisher, the standard forecasting quantity,
and it now says so — `kind="fisher"`, and the docstring leads with it. The
distinction travels: `parameter_covariance` carries `kind` across into
`"covariance"` (Cramér-Rao, no prior) or `"posterior_covariance"`, and `sigma()`
refuses both precision kinds rather than only the one it used to know about.

A prior with no quadratic form (a `Uniform`, a `LogNormal`) and a prior-free
latent are both refused by name rather than approximated — the same refusal, in
the same words, that the conjugate exits already give. That is not hypothetical:
the beam space in `docs/inference.md` declares `fwhm` as `Uniform`, so its
Fisher forecast is necessarily likelihood-only, and the page now says so.

**Re-measured, not assumed:** that page's "Fisher and NUTS agree" claim
survives — `sd(NUTS)/sd(Fisher)` is 0.985 for `fwhm` and 0.975 for `offset`.
The reason is now written down rather than implied: `offset`'s declared
`Normal(0, 0.4)` contributes 5e-7 of the total precision and `fwhm`'s `Uniform`
is flat over 512x the posterior width, so neither prior does any work at this
noise level. With the informative priors the rest of the package reads, the two
would not agree.

### `noise_std` has an axis contract, because a square grid has two readings

Every `noise_std` docstring said "scalar or broadcastable to the data", under
which a 1-D sigma vector against a square `(n_time, n_freq)` grid means one
sigma per time sample *and* one per frequency channel. Both are legitimate,
which is the defect. Measured on an 8x8 grid with a `(8,)` per-time gain latent
and `sigma = linspace(0.01, 1.0, 8)`:

```
noise_std (8,)   -> sigma('gt') [0.00010 .. 0.00010]
noise_std (8,1)  -> sigma('gt') [0.00004 .. 0.00354]
noise_std (1,8)  -> sigma('gt') [0.00010 .. 0.00010]
```

All three succeeded and returned the same shape. NumPy settles the tie by
aligning trailing axes, so the per-**time** vector was applied per-**frequency**
and an error bar that genuinely spans ~90x came back internally flat.

`check_noise_std_axis` refuses a 1-D `noise_std` whose length matches more than
one axis of the prediction, quoting both readings and naming the two shapes that
settle it. It reads **shapes only** — a NaN defeats every comparison-based guard,
so a value check here would be the one thing a poisoned sigma sails past. A
constant sigma wrapped in `HomoscedasticNoise` or `FlaggedNoise` is unwrapped and
checked; `RadiometerNoise` is exempt, its sigma being the prediction's shape by
construction. Reached through `as_noise_model(..., prediction_shape=...)`, which
existing callers do not pass and are therefore untouched.

`inference/linear.py` does not route through `as_noise_model`, so the rule has a
second home in `_check_solve_arguments`, passed by `wiener_solve`. **Owed:**
`gcr_sample` does not pass it yet, so the draw is not covered while the mean is;
the unification is named in the code rather than left implicit.

### `Bind(fan=...)`: the fan-out mode is declarable, because the container type is not evidence

`Bind` decided between *tying* one produced value into every `into` selector
and *distributing* one value per selector by asking whether `fn` returned a
Python `tuple`/`list`. Measured on two scalar leaves — an
`AntennaLossOperator.efficiency` and a `GainOperator.gain` — fed the same
2-vector `[2, 5]`:

```
fn = lambda v: v          ACCEPTED   pred[0,0] = 4.0    (2 * 2, tied)
fn = lambda v: [v[0],v[1]] ACCEPTED  pred[0,0] = 10.0   (2 * 5, distributed)
fn = lambda v: list(v)    ACCEPTED   pred[0,0] = 10.0   (2 * 5, distributed)
```

**Row three is the defect.** `v` and `list(v)` are the same data. One is a JAX
array and the other a Python list of its elements, and that difference — which
survives no shape check, no `check_linearity`, no `identifiability` — is the
whole basis on which the two readings were told apart. A user who meant "write
this whole 2-vector into both leaves" and reached for `list(v)` got element-wise
distribution: finite, correctly shaped, and 2.5x off.

`fan="broadcast"` and `fan="distribute"` write that intent down.
`fan=None` is the default and **keeps the old inference exactly** — `Bind` is
public and appears in every example and doc page, so nothing existing changes.
Declared, a contradiction between the declaration and what `fn` did is refused
by name: broadcast that produced a container, distribute that produced a single
value. `ParameterSpace.direct(..., fan=...)` threads it through, since `into`
there takes a tuple of selectors too.

**The one case no rule can decide, and why it warns rather than refuses.** With
a single `into` selector, the length test that separates the modes —
`len(produced) == len(into)` — is satisfied by a length-1 container under either
intent, so the container was silently unwrapped. It still is, now under a new
`AmbiguousFanWarning`. Not a refusal, because there is no wrong answer to
prevent: a Python list is not an array leaf, so unwrapping is the only reading
that can reach one, and broadcasting the container instead would change the
pipeline's pytree structure and be refused by `ParameterSpace.validate` a moment
later. Refusing would break working code and buy no correctness. Declaring
`fan="distribute"` silences it.

`Bind` carries no array leaves — every field is static — so the new field lands
in the treedef's aux data and changes no leaf count. `ParameterSpace.bindings`
is itself static, which puts that aux data in a jit cache key; `str | None`
keeps it hashable. Pinned by a test.

### `split_rhat` refuses a trace it cannot halve, instead of returning NaN

`MIN_DRAWS = 4` was enforced by `SamplingPlan.sample` and not by `split_rhat`,
which is public, exported, and documented no minimum. Called directly:

```
len 0 -> ZeroDivisionError: division by zero
len 1 -> ValueError: all input arrays must have the same shape
len 2 -> nan
len 3 -> nan
len 4 -> 2.1213203435596424
```

Sizes 0-3 now raise `ParameterSpaceError` carrying the same "two halves of two"
sentence `MIN_DRAWS` and `sample()` already use. **The NaN is the one that
mattered.** It arrived with only a numpy `RuntimeWarning` that nothing surfaces,
and a NaN passes no threshold test in either direction — both `rhat <= rhat_max`
and `rhat > rhat_max` are `False` — so an undefined diagnostic reads as whichever
verdict the caller happened to test for.

Independently: the second half is now sliced `values[values.size - half:]` and
not `values[-half:]`. Those agree for every positive `half` and disagree at 0,
where `values[-0:]` is the *whole* trace — which is how a length-1 trace came to
be compared against itself. Extracted as `_halves` so the slice stays testable
at the lengths the new guard forbids; a guard that hides a bug is not a fix.

### A twin that draws its own noise is refused at every inference exit

**Breaking, for one shape of call.** Handing a forward model containing a
stochastic stage to any inference entry point now raises `ParameterSpaceError`
naming the stage. `NoiseOperator` and `RFIOperator` are the two shipped
operators affected. Generating data with them is unchanged and is what they are
for — it is the *fit target* that may not contain one.

**What was wrong.** Inference closes the model over one template state, so an
operator consuming the PRNG draws ONE realisation and adds that same frozen
field to every prediction compared against the data. Measured on an 8×8 grid in
float64, `observed` generated from the honest model at g = 1.1 plus independent
2 K scatter, the only difference between the rows being whether
`NoiseOperator(sigma=20)` sits in the twin:

```
clean twin: estimate 1.101511  posterior 1.101545 +/- 0.002451   (truth 1.100000)
noisy twin: estimate 1.082393  posterior 1.082427 +/- 0.002451   (truth 1.100000)
```

**7.8 σ of bias, reported with an error bar identical to every digit.** The
magnitude is the draw; the invisibility is structural. Adding a constant field
is exactly affine, so `check_linearity` sees residual 0.0 and `identifiability`
reports full rank. There is no numerical symptom for any diagnostic to find,
which is why the detector is the operators' own declaration.

**`'key'` in `requires` is now a contract rather than a note.** `requires` and
`provides` were declared by 31 classes and read by nothing;
`rheplicant.core.contract` is the consumer. `refuse_stochastic_stages` walks the
assembled tree and refuses, and two call sites —`ParameterSpace.validate` and
`build_forward_fn` — cover all ten pipeline-accepting entry points:
`check_linearity`, `linear_operator` (hence `wiener_solve`, `gcr_sample`,
`iterative_gls`), `identifiability`, `to_numpyro_model`, `predict_from_samples`,
`SamplingPlan.estimate`, `SamplingPlan.sample`, `build_forward_fn` (hence both
calibrators) and `simulate_pairs`. A new stochastic operator is covered the day
it declares what it reads.

The rest of `requires`/`provides` stays descriptive, and the class docstring now
says so with its reasons instead of promising a checker. Threading available
State paths forward is not implementable against the shipped set:
`GroundPickupOperator` declares `"env.temperature"` and documents a `t_ground`
fallback for when it is absent, so the declaration means "reads if present".
And `provides` is `("data",)` on 26 of the 31 declaring classes.

### `SelectOperator` selects, and says what selecting cannot repair

`leaf * mask` is not `data[t] = contribution[switch[t]][t]`; it is an identity
that holds only while every branch is finite everywhere. Every branch runs at
every sample, so a branch returning `inf` where it is switched OFF entered the
sum as `inf * 0 -> nan`. Whether it did depended on the execution mode, because
XLA may rewrite a multiply by a predicate into a select and does so in some
fusion contexts and not others:

```
             before                     after
eager        [5.  1.  0.5  0.25]        [5.  1.  0.5  0.25]
filter_jit   [5.  1.  0.5  0.25]        [5.  1.  0.5  0.25]
disable_jit  [nan 1.  0.5  0.25]        [5.  1.  0.5  0.25]
```

The forward answer was correct by optimiser luck. It is a `jnp.where` now,
which never reads the unselected value, in any mode.

**The gradient is a separate claim, and it is a precondition rather than a
fix.** Reverse mode differentiates every branch at every sample. A branch that
returns `inf` at a switched-off sample has an infinite residual there —
`d(a/t)/da = 1/t` — and the selector's zero cotangent for that sample meets it
as `0 * inf` *inside that branch's own backward pass*, upstream of anything
`SelectOperator` does. Measured: the naive `jnp.where` swap and the two-step
"sanitise then select" produce the identical VJP, and both still return `nan`.
So the class docstring states the precondition and the remedy that works — the
branch guards its own singularity, using the coordinates it already receives —
and `tests/core/test_select_finiteness.py` pins the limitation and the remedy
side by side. Nothing shipped trips it — the divisions in the radio operators
are by configuration (`width`, `ref_freq`), never by a coordinate that can be
zero on an observing grid — so this is written for a user-supplied
calibration-load branch with a reciprocal in it.

### A placed operator must agree with its node about who creates the data

`Assembly.has_source` was read off the template's node kinds, and `At` will put
any operator at any node — so the one fact the `__call__` guard is built on was
a fact about the *graph*, and it was wrong in both directions:

```
a SOURCE operator on a TRANSFORM node
  has_source=False, lit ('t',);  caller data 100.0 in -> [777. 777. 777.]
a TRANSFORM operator on a SOURCE node
  has_source=True,  lit ('s',);  data=None -> TypeError on NoneType
                                 data given -> "Pass a state with data=None"
```

The first is the guard whose entire sentence is "caller-supplied `state.data`
would be discarded" passing in exactly the case where it is discarded — along
with whatever the upstream branch computed. The second is the guard refusing
the only input that runs, and naming the input that crashes.

`assemble` now refuses the disagreement at placement, which fixes both at once:
with the kinds in agreement the node kind IS the operator kind and the existing
derivation is sound. `At` still moves an operator freely between nodes of the
same kind. The check reads `graph_node` only, costs 2.5 µs on a 262 µs
ten-operator assembly, and refuses nothing the shipped templates do.

Deriving the answer instead by running each operator under `jax.eval_shape` at
assemble time — asking whether it produces data from `data=None` — was measured
and does not work here: `assemble` has no coordinates to build a probe state
from, and every shipped source raises `StateValidationError` without them. 0 of
10 shipped operators classify correctly against `State()`, so every assembly
would report itself source-free and the guard would then refuse every
legitimate forward run. With a fully-populated state, which `assemble` does not
have, it is 5 of 10, at ~1.6 ms per operator. What the declaration check cannot
see is an operator declaring no `graph_node`; that hole is pinned in
`tests/core/test_placement_kind.py`.

### `must_precede` is checked on the hand-built route too

`assemble()` enforced the ordering constraint by reachability and was the only
thing that did. Both routes compile to the same composition, so the refusal was
one line of call site away from silence:

```
assemble(..., At('noise', tone))   AssemblyError: 'bandpass' is not reachable
Pipeline(sky, band, gain, tone)    no error; tone channel 2678.5
Pipeline(sky, band, tone, gain)              tone channel 7435.6   (g = 3.0)
```

A tone injected after the gain has a gain response of exactly 1.0 — it monitors
nothing, which is the sentence `must_precede_because` exists to say — and the
run converges and reports healthy diagnostics.

`Pipeline` has no graph, so it cannot ask what is reachable from where; it has
`names`, so `check_stage_ordering` asks the sequence-local question: **if a
named stage is present, it must come after me.** A stage that is not present is
not a violation — the same rule `_check_ordering` applies to a node that was
never lit, and anything stricter would refuse every partial model.

It runs from `Pipeline.__init__`, and equinox rebuilds a Module through
`tree_unflatten` rather than `__init__`: measured, a `filter_jit` trace, a
`filter_grad` and an `eqx.tree_at` edit re-run it **zero** times. Construction
costs 1.5 µs more (8.0 → 9.5 µs), at the one place the order can be chosen.

Three things it cannot do, each pinned as a test: refuse a constraint naming a
stage nothing is called (a Pipeline has no node list, so a typo and a
legitimately absent stage are one observation, where `assemble` refuses the
unknown node id); see into a nested composite (`names` is one level deep); mean
anything for `SumOperator`/`SelectOperator`, whose branches are parallel. Note
that the constraint binds through stage *names*, so auto-derived names bind it
only where they coincide with the node ids — `GainOperator` auto-names to
`gain`, `ReceiverOperator` to `receiver` and not `bandpass`. Pass `names=` for
a pipeline whose stages carry ordering constraints.

### `Assembly.without(node_id)`, and two entry points that took objects they could not use

`replace_node(node, None)` returned a live `Assembly` whose `lit` and whose
mermaid rendering both still claimed the stage was there, and which then died
with `TypeError: 'NoneType' object is not callable`. There was no `without()`
anywhere, so "take this stage out" had no supported spelling. `replace_node`
now refuses `None` by name and points at `without()`; any other non-operator
goes through the same `validate_operators` screen `Pipeline` and the combinators
use.

`assemble()` calls that screen too. Before `must_precede` landed, a
non-operator reached the fold and failed there; the ordering work turned it into
`AttributeError: 'Bare' object has no attribute 'must_precede'`.
`assemble(At('t1', Bare()))` now says what `Pipeline(Bare())` has always said.

`Assembly.without(node_id)` re-runs `assemble()` over the remaining operators
rather than editing the tree, so the result is exactly the assembly that not
providing the operator would have given — honest
`lit`/`skipped`/`has_source`/`materialized`, and every assembly-time refusal
re-run. Dropping a source a summed branch needed gives assemble's own "no live
source upstream". A region is dropped whole and re-placed over its whole path;
a `many` node is dropped whole. The new static `Assembly.placements` field
records the recipe as addresses, sorted by template order so that argument
order stays irrelevant.

### The CW tone is a line now, and both ends of "how wide" are guarded

**Breaking, twice.** `CWCalibrationOperator` no longer models the tone as one
number in one channel.

* `line_width` is now a **required** argument. Every existing
  `CWCalibrationOperator(amplitude=..., tone_freq=...)` call raises `TypeError`
  until it is given one. There is deliberately no default: the width is a
  property of the spectrometer, nothing in this repository, `rhino-cal` or
  `rhino_cal_jax` establishes RHINO's, and guessing it silently mis-sizes the
  protection mask — the failure the operator exists to avoid. For a critically
  sampled unwindowed FFT it is one channel spacing, `band / (n_freq - 1)` on a
  linear grid.
* `amplitude` **changed meaning**: it is the tone's TOTAL contribution summed
  over channels, not the level added to one channel. The lineshape is
  normalised over the sampled channels so the injected total is `amplitude`
  whatever the width and wherever the line falls between two channels — the
  tone's level is the one thing this operator knows, and a total that moved
  with the channelisation would make the known quantity unknown. The price is
  that the peak channel is no longer `amplitude`: a line halfway between two
  channels keeps ~0.42 of it (half-bin scalloping, −3.8 dB), which is real and
  is exactly the bias the delta-on-one-channel model hid. An on-centre `sinc2`
  tone of one channel width is unchanged to 1e-6, so on-centre call sites keep
  their old numbers.

`lineshape` (`"sinc2"` | `"gaussian"`), `drift_rate` and
`amplitude_drift_rate` are new, all KNOWN static settings rather than
differentiable leaves. A tone that drifts in frequency writes a
`(n_time, n_freq)` **waterfall** protection mask instead of a `(n_freq,)`
channel mask, because the contaminated channels move with the line.

**What a tone with width actually measures.** A delta probes `b(ν_cw) g(t)` —
one bandpass value. A line with width probes `Σ_k w_k b(ν_k) g(t)`, the
lineshape-weighted average of the bandpass across its wings. Measured on a
realistically curved band for a 1.5-channel gaussian: **2.37 %** apart
(0.9213 against 0.9000), in float32 and float64 alike. Widening the line also
COSTS the tone leverage rather than adding any — the residual of its channel
profile outside a degree-4 polynomial basis falls 0.84 → 0.038 from a quarter
channel to the widest line the band admits. Realism here is not extra
information.

**`line_width` is now guarded from both sides.** `MIN_WIDTH_IN_CHANNELS` was
already there; `MAX_WIDTH_IN_BAND_FRACTION` (0.25 of the band, never below
`MIN_CEILING_IN_CHANNELS` = 2 channel spacings, so a coarse grid cannot make a
one-channel line "too wide") is new. Past the ceiling the injection is not a
line at all but a pedestal: measured on an 11-channel band, a width of 5
channels puts every channel above `protect_floor`, so the mask covers the band
and the RFI flagger is off for the whole run — genuine RFI surviving into the
data. At 100000 channels the "tone" is a uniform `1/n_freq` floor. The number
is where both shapes cross the default 1e-2 floor for the worst-case placement
(gaussian `exp(-1/2f²)`, `sinc2` envelope `(f/π)²`, both ≈ 1.1e-2 at `f = 1/3`).
`line_width = 25e6` where `25e6 / (N_FREQ - 1)` was meant is one keystroke, and
it is now a refusal.

**Fixed: a drift on a unix-second time axis was silently wrong.** `coords.time`
is stored through `jnp.asarray` — float32 unless x64 is on — so an axis of unix
seconds (~1.75e9, which is exactly what `read_rhino_observation` puts there
from `obs.time_s`) is quantised onto a **128 s grid before** `t - t[0]` ever
runs. The anchor cannot recover what the store already threw away, and because
the injection and the band guard read the same corrupted elapsed values, the
guard could not see it. Measured, 4 samples 100 s apart, `drift_rate` 1e4 Hz/s:

```
coords.time = [0,100,200,300]          elapsed [0,100,200,300]  peak ch [4,5,6,7]  protected 1,1,1,1 of 11
coords.time = 1.75e9 + the same        elapsed [0,128,256,256]  peak ch [4,5,7,7]  protected 1,6,8,8 of 11
```

Two of four samples collapse onto the same time, the tone lands in the wrong
channel, and the mask blows out from one channel to eight of eleven. Nothing
raised, nothing was NaN, every shape was right; the same run under
`JAX_ENABLE_X64=1` gives `[4,5,6,7]`, so the cause was precision, not logic. A
drifting tone now **refuses** a time axis whose stored resolution exceeds
`MAX_TIME_RESOLUTION_IN_SAMPLES` (1e-2) of the run's smallest sample interval,
and the message names both remedies: pass times measured from the start of the
run, or enable float64. Seconds from the start of the run, the day or the month
all pass; seconds from the start of the year (2 s of error) do not.

**Fixed: a waterfall protection mask went stale through a shape-changing
stage.** `BackendOperator` updates `data` and `coords.time` together and leaves
`aux` alone, so `Pipeline(CWCalibrationOperator(drift_rate=...),
BackendOperator(n_chunk=2))` produced `(2, 11)` data beside a `(4, 11)` mask.
`unflag_protected` checked only the channel axis and then broadcast: the
mismatched case raised a raw `TypeError` from `&`, and a **one-row** mask —
what a single-chunk average leaves behind — broadcast silently over every
sample and unflagged the entire run (measured: 0 of 44 flags left). It now
checks the time axis too and refuses both as a sentence naming the stale mask.
A waterfall mask is bound to the time axis it was written on; a stage that
changes the number of samples must drop or re-derive it.

### A smooth (ν, t) basis, and the antenna temperature that uses it

`identifiability()` refuses a free-per-cell model and tells the caller what to
do about it — "a smooth basis in place of one free parameter per cell is the
usual repair". `rheplicant.core.basis` is that basis, and
`rheplicant.radio.BasisTemperatureOperator` puts it on the reserved
`t_sys_extra` node, where it is parameterized by **coefficients, not cells**.

```python
basis = SeparableBasis(
    time=basis_matrix("legendre", n=n_time, n_basis=3),
    freq=basis_matrix("legendre", n=n_freq, n_basis=2),
)
twin = assemble(BasisTemperatureOperator.from_basis(basis, basis.fit(t_ant0)),
                CWCalibrationOperator(...), GainOperator(gain=g0))
plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
```

**Why these are one change and not two.** Measured through the operator, on the
assembled graph, with a known 5000 K CW tone against a gain free per time
sample (`n_time=7`, `n_freq=5`, at a generic coefficient point):

```
free-per-cell T_ant,  tone ON  (5000 K)   n_par=42 rank=35 nullity=7
free-per-cell T_ant,  tone OFF            n_par=42 rank=35 nullity=7
(3,2)-basis T_ant,    tone ON  (5000 K)   n_par=13 rank=13 nullity=0
(3,2)-basis T_ant,    tone OFF            n_par=13 rank=12 nullity=1
```

Against a free-per-cell antenna temperature **the tone buys exactly nothing** —
nullity is `n_time` either way, because the free cells absorb the whole of
`g[t] × (tone profile)` sample by sample. So the basis has to reach `T_ant`
itself; smoothing the noise waves alone would leave the tone useless. This
reproduces the synthetic probe quoted in `identifiability`'s own docstring
(there at `n_time=8`: nullity 8 either way, `s[rank-1]/s[0]` = 7.071e-01 — the
identical number, since the free-per-cell null space is exactly null).

**And it is the FREQUENCY axis that does the work** — new, and not something
the earlier probe could see. Varying which axis is restricted:

```
n_k              n_j                nullity, tone ON   tone OFF
3                2                  0                  1
7 (complete)     2                  0                  7
3                5 (complete)       1                  1
7 (complete)     5 (complete)       7                  7
```

A basis complete in *frequency* makes the tone worth nothing whatever the time
axis does: the tone's profile is then inside the span and is reabsorbed,
nullity 1 with it and 1 without. A basis complete in *time* is still rescued by
it. So "frequency-smooth" is the condition, and `n_j < n_freq` is what it
means. `n_basis == n` is therefore legal rather than refused — the matrix is
perfectly well conditioned, and whether it costs anything is a joint property
of the model that only `identifiability()` can answer.

**The framework needed nothing new.** A basis expansion was already a `Bind`
with an `fn` over a `linear=True` latent, which `check_linearity` verifies and
both conjugate exits drive. What was missing was the matrices, which are a
modelling choice. `SeparableBasis.expand` is a drop-in `Bind` function — that
is the route the noise-wave temperatures need, since those leaves belong to
`NoiseWaveOperator` and are full-grid by its contract — and it always returns
`(n_time, n_freq)`, the one temperature shape that operator's guards can never
misread.

**`legendre` and `polynomial` span the same functions and are not
interchangeable.** Measured `cond(design)` at `n=32, n_basis=16`: 7.86 against
2.81e+05. That number lands on κ of the block's normal operator, which is what
`wiener_solve`'s error guard divides by.

**Two conditionally-linear blocks, not one.** `gain × T_ant` is bilinear, so
`check_linearity(names=("gain", "t_coeff"))` refuses the group — verified, not
assumed — and the two are separate `Block`s of one plan. `plan.estimate` and
`plan.sample` then agree with each other and with the truth, while the
free-per-cell version of the same model is refused before a sweep runs with the
degenerate direction named as `gain 0.50, t_coeff 0.50`.

**Where it lives.** `rheplicant.core.basis`, not `rheplicant.inference.basis`.
It reads like an inference utility, but the design matrices are held by an
operator that sits on the signal path, so `radio` would have had to import
`inference` — which nothing in this package does, and which the inference
layer's own premise forbids. `core` is the one layer both may depend on, and it
fits: no `State`, no radio physics, and `Coordinates` already names the `time`
and `freq` axes there.

### `SamplingPlan`: one declared partition, two exits

A point estimate and a posterior sample are two exits from one workflow, not
two workflows. That was already true in exactly one place — `_conjugate_solve`'s
`key=None | k`, which `wiener_solve` and `gcr_sample` share — and this promotes
it to the level a whole model is inferred at.

```python
plan = SamplingPlan(space, Block("t_nw", "t_ant"), Block("gain"),
                    Block("beam_fwhm", steps=20))
est   = plan.estimate(twin, state, observed, noise=noise)
draws = plan.sample(twin, state, observed, noise=noise, key=k, n_sweeps=200)
```

**Two methods rather than a mode flag.** `key=k` says "pass a PRNG key", which
is an implementation detail and not an intent. Two signatures make the invalid
combinations unrepresentable instead of validated: `key` is required on one and
absent from the other, `n_sweeps`/`warmup` belong to one and `max_iter`/`tol` to
the other. What the two share is the implementation — conditioning, the
partition, sigma, the joint chi-squared, the identifiability check — and they
diverge at the last step, which is where the layer below already diverges.

**The engine is derived, never restated.** `Latent(..., linear=True)` already
says which exit a latent takes, so `Block("t_nw", "t_ant")` needs no `engine=`.
A block mixing declared-linear and non-linear latents is the one genuinely
ambiguous case and is refused unless the caller downgrades the whole block to
gradient deliberately; `engine="conjugate"` on a latent nobody declared linear
is refused outright, because that claim belongs in the declaration where
`check_linearity` verifies it.

**The partition is checked to be strict.** Every latent of the space in exactly
one block, and a latent omitted, duplicated, or undeclared is refused *by name*.
An omitted latent is the dangerous one: it sits frozen at its declared init for
the whole run while the sweep converges and every other number looks healthy.

**Two guards see what no per-block number can.** `identifiability()` runs at
**both** exits before a sweep and refuses a degenerate model, printing the null
directions as named combinations of latents. The point estimate is the exit that
needs it more, not less — a chain at least has `r_hat` to scream with, while CG
converges quietly onto an arbitrary point of the null space. And the convergence
monitor is the **joint** chi-squared across sweeps, never a per-block residual,
which is precisely the number that read `1.7e-07` on an answer 2288 K wrong.
`check_identifiability="once" | "each_sweep" | False` is the caller's explicit
choice with the cost documented at both ends, not a size heuristic: the sky is
not always 10⁶ coefficients, and for a small model the per-sweep check is cheap
*and* strictly more informative, since a nonlinear model's identifiability is a
property of where you are.

Measured on the bilinear `gain × T_ant` model that motivated all of this: the
free-per-cell parameterization is refused (nullity 6 of 60, direction named as
`gain 0.50, t_ant 0.50`); the basis one runs, and `estimate` and `sample` agree
with each other and with the truth to well inside the posterior width; and at
sweep three the joint chi-squared is still falling by 3e7 while every block's own
CG residual is below 1e-5.

**The chi-squared test is on the decrease, not the change.** Testing
`|chi2[k] - chi2[k-1]|` walks into the trap `iterative_gls` documents for its own
`reweight_tol`: consecutive sweeps differ by the inner solver's own noise
whatever the outer iteration is doing. Measured in float32, the plateau sits at
`chi2 = 2.7e-3` and jitters by `1.2e-3` a sweep, so a converged run was refused
for 300 sweeps and counting.

**A plan does not nest `iterative_gls`.** Sigma is re-evaluated at the current
joint prediction before every block update, so for a `RadiometerNoise` the sweep
*is* the reweighting iteration; nesting would run one fixed point inside another
at the product of their costs. `PlanDiagnostics.noise_depends_on_prediction`
records when that applied, because freezing sigma inside a draw makes it exact
for the linear-Gaussian conditional *at that covariance* and not for the full
model's.

**NUTS-within-Gibbs is stated, not hidden.** A conjugate block's GCR draw is an
exact conditional draw, so a plan of conjugate blocks is an exact Gibbs sampler.
A finite number of NUTS steps is a transition that merely leaves the conditional
invariant, which makes the scheme Metropolis-within-Gibbs — still valid, and the
inner step count now affects mixing. `Block(..., steps=20)` looks like a
performance knob and is a statistical assumption; it says so in its own
docstring. The kernel adapts through warmup and is **frozen** afterwards, since
a kernel that keeps adapting from the states it visits is no longer a valid
transition.

### The CW tone: a position the graph checks, a level it knows, channels it keeps

`calibration.py` stated its ordering constraint in a module docstring — the
tone must sit *before* the bandpass and gain, because it tracks `g(t)` only by
passing through it. Nothing enforced it. `At("noise", cw)` assembled cleanly
and the tone's gain response dropped to **exactly 1.0**: a calibrator that
monitors nothing, in a model that runs, differentiates, and looks healthy.

**`must_precede` on `AbstractOperator`**, checked by `assemble()`. An operator
whose physics depends on where it sits declares that in the graph's own nouns
(node ids), with an optional `must_precede_because` the refusal quotes back.
The check is **reachability, not a toposort index**: a toposort totally orders
a DAG, so it also orders nodes on branches that never meet, and "sorts earlier"
would be satisfied by a placement whose output never reaches the constrained
stage at all. An absent stage is not a violation (nothing to pass through); a
node id the template does not have *is* one, because an unenforceable
declaration is prose in a ClassVar. This is a third declaration alongside
`requires`/`provides`, not a consumer of them — those speak in State paths, and
every operator on the receiver chain reads `"data"` and writes `"data"`, so
"before the gain" is not a sentence that vocabulary can form.

**The tone's amplitude is now static.** It was a differentiable leaf, which
quietly contradicted the premise of the whole calibration route: a tone of
unknown level constrains nothing, because the gain it is meant to track absorbs
it exactly. `eqx.field(static=True)` with a validating converter, so
`eqx.partition(op, eqx.is_inexact_array)` cannot pick it up. A non-scalar
amplitude is refused (it would broadcast into the tone's channel and model a
per-sample tone), and so is a traced one.

**And `tone_freq` is checked against the observing band.** `argmin` always
returns *some* channel, so a tone at 200 MHz against a 60–85 MHz band landed on
the top edge channel and modelled a bright spike that calibrates nothing. The
band arrives per call, so the check is at the same boundary and uses the same
concreteness escape as `DriftScanProjector._validate_uniform_grid`: it fires at
trace time for a closed-over band and skips, rather than crashing, for a
genuinely traced one.

### The bandpass/gain scalar degeneracy, measured

`b -> c*b, g -> g/c` leaves every prediction invariant, so a free bandpass and
a free gain are separately unidentifiable up to one scalar. Nothing in the
design record mentions it. `identifiability()` now says so by name:

```
b free (5 ch) + g free (6 samples), T_ant known
    n_par=11 rank=10 nullity=1     null singular value 8.4e-17 of s_max
    participation                  {'bandpass': 0.50, 'gain': 0.50}
    direction(0)['bandpass'] / b   +0.2206 in every channel
    direction(0)['gain']     / g   -0.2206 in every sample
```

That last pair *is* the trade, read off the report rather than asserted.

**The convention: the bandpass carries only shape (mean 1), the gain carries
the level.** `unit_mean_bandpass` (and its companion `unit_mean_free`) takes
`n_freq - 1` free values and expands them onto the mean-1 hyperplane —
`n_par=10 rank=10 nullity=0`, weakest identified direction 0.41. Unit mean
rather than pinning a reference channel because the reference is then a band
average: its noise falls as `sqrt(n_freq)`, and it survives the channel being
flagged. RHINO flags channels; a convention anchored to one channel makes the
absolute gain scale hostage to whichever channel RFI sits in.

The trap is documented because it is measured: normalising *inside the binding*
(`fn=lambda b: b / mean(b)`) with all `n_freq` parameters kept does **not**
remove the degeneracy — `n_par=11 rank=10 nullity=1`, with `participation` now
`{'bandpass': 1.00, 'gain': 0.00}`. It moved out of the b/g trade and into the
bandpass latent's own scale ray. Removing a degeneracy means removing a
parameter.

### The flaggers no longer erase the tone

`FlaggingOperator` and `MomentRFIFlaggingOperator` both flagged the tone's
channel at fraction **1.0** — correctly, from their point of view, since a
narrow persistent spike is what RFI looks like — and flagging sits downstream
of `cw_tone` on the same trunk.

New `rheplicant.radio.protection`: `PROTECTED_KEY`, `protect`,
`unflag_protected`. The mask rides in `state.aux` and is written by the
operator that **injected** the tone, because that operator is the one that
knows which channel it went into; a flagger has no way to tell a calibration
tone from RFI, and a `protected=` setting on the flagger is one the user must
remember for every run, failing as a slightly worse calibration rather than as
an error. Measured on the real MomentRFI flagger: fraction 1.0 → 0.0 on the
tone's channel, with genuine RFI elsewhere still flagged.

Two things it does not claim. It removes the flag, not the tone's influence on
MomentRFI's own surface fit. And a protected channel is one where real RFI now
survives — the deliberate trade, since a flagger's verdict on a known-bright
channel carries no information anyway.

### An ingestion layer, and what checking it against the original found

rheplicant can now read real RHINO recordings. Two thin adapters, flat under
`radio/` alongside `beams.py` because neither is an operator:

**`rheplicant.radio.touchstone`** — `read_touchstone`, `Touchstone`,
`interpolate_onto`. Touchstone v1 `.s1p`/`.s2p` measured S-parameters, in Hz.

**`rheplicant.radio.rhino`** — `read_rhino_observation`, `RhinoObservation`,
`to_state` (exported as `rhino_to_state`). The observation HDF5 — waterfall,
switch log, thermistor log — in Hz, unix seconds and Kelvin, and then onto the
signal graph. Needs the new `rheplicant[rhino]` extra (h5py).

Plus **`DataIngestionError`**, for a file that cannot be read or that
contradicts what the caller declared about it.

**Both readers refuse what the originals accepted.** `read_s2p` skipped any row
whose column count was not exactly nine, so a trailing `!` comment — legal
Touchstone — silently removed that frequency point, and a 1-port file came back
as *empty arrays*. `DataHandler` defaulted `freq_unit` to MHz, which is wrong
for its own simulator's output and silent about it. `interp_vals_to_new_freq`
let `np.interp` clamp, so a sweep narrower than the observing band returned a
constant Γ at the edges. Every one of those is now an error naming the file and
the line.

Two policies are deliberately stricter than the standard, for the same reason
in both cases — a unit error is invisible downstream and unrecoverable:
`freq_unit` on the HDF5 side is a required argument with no default, and a
Touchstone file omitting its frequency-unit token is refused rather than
silently taking the spec's GHz default.

**`tests/radio/test_ingestion_vs_reference.py` reads the same bytes through
both implementations and demands they agree.** Where both answer, every value
is bit-identical — `rtol=0, atol=0`. Skipped when the rhino-cal checkout is
absent.

That comparison also turned up a live bug in the reference:
`DataHandler(temperature_unit=un.K)` ignores the argument and adds 273.15
anyway, so a file already in Kelvin reads back 293.15 K for a 20 K reading.

**What review caught that tests did not.** `_interp_strict`'s tolerance scaled
with the axis *magnitude*, which is harmless at 60–85 MHz and gives 1.75 s of
slack on a unix-epoch axis — silently clamping a thermistor log that stopped
1.5 s short, in the one caller it was written for. `aux["flags"]` was built
per-sample when every consumer in this package requires it data-shaped. And
NaN defeated three independently-written boundary guards, because `nan <= 0`
and `nan > hi` are both `False`: a NaN channel walked through the frequency
plausibility check, a NaN timestamp walked through the ascending check and made
the sample-drop mask non-contiguous, and a NaN axis walked through
`_interp_strict`'s own ordering guard.

### A `many=True` node id is now a stable identifier

A second operator placed at a `many` node used to take over the first one's
name. Measured, on a two-source graph with `f` a `many` source:

```
one instance  forward: [10. 10. 10.]  lit: ('f', 'g')
two instances forward: [30. 30. 30.]  lit: ('f', 'g')   <- lit IDENTICAL
A2['f']  type: SumOperator    A2['f_2'] type: Src
replace_node('f', amp=0) forward : [0. 0. 0.]    correct would be 0 + 20 = 20.0
```

`_dedup` gave instance 1 the bare node id, which is also the label the fold
over the instances carries, and the breadth-first lookup resolved the
collision to the fold — so `replace_node` overwrote *both* instances with one
operator. A component of the instrument disappeared with no shape change, no
error, and an unchanged `lit` and rendering.

The same defect reached inference through `t_sys_extra`, the `many` node the
graph designates for a generic effective-T_sys contribution:

```
one instance : space.validate PASSED
two instances: space.validate RAISED ParameterSpaceError: Bind for ('C',): `into` selector 0
               failed against the pipeline ('SumOperator' object has no attribute 'coeff').
```

Every `ParameterSpace` written against such a node was invalidated by the next
component someone added there — the stated design intent ("re-parameterizing
never requires editing the instrument description") failing in reverse.

**What changed.** With two or more instances at a node they are now named
`x_1`, `x_2`, … — *including the first*, so the bare id names no instance and
cannot collide with the fold — and the bare id raises the new
**`AmbiguousNodeError`**, which lists the instance ids to use instead.
`replace_node` therefore refuses rather than deleting; it also refuses on a
junction/selector that assembly materialized as a combinator, which silently
discarded every branch feeding it for the same reason. `Assembly.instances`
records the multiplicity, `repr` shows `t_sys_extra x2`, and the mermaid/SVG
renderings label such a node `(x2)` — `lit` alone said the same thing for one
instance and for two.

**A single instance is untouched**, bitwise: `_instance_names(x, 1) == (x,)`,
so existing spaces, examples and tests keep working unchanged. The one
user-visible rename is the *first* of several siblings:
`twin["receiver_input"].names` for two calibration loads is now
`('observed_astro_sky', 'cal_loads_1', 'cal_loads_2')`, was `(..., 'cal_loads',
'cal_loads_2')`. Branch order, switch indices and every forward value are
unchanged — the names are static metadata.

Note that one `ParameterSpace` still cannot validate against both the
one-instance and the two-instance assembly: that is the point. Answering the
bare id with instance 1 would be the finite, correctly-shaped, wrong binding
this package exists to refuse.

**Follow-up: the ids the error message hands out are now checked.** The
per-instance ids `x_1..x_n` above and the `x, x_2, x_3…` that `_dedup` mints
for repeated branch labels are the same format, so on a graph where a node
reaches a junction by two paths they collided — and the breadth-first lookup
resolved the collision to the fold, as before. Measured on `x[many] -> p`,
`x -> q`, `p -> j`, `q -> j`, with sources 10 and 20 at `x` (forward 60):

```
asm['x_1'] -> Src(10)        correct
asm['x_2'] -> a fold over the q-path, NOT instance 2
replace_node('x_2', Src(0))  <- literally what AmbiguousNodeError said to write
           -> forward 60 -> 30    correct for dropping instance 2 is 20
```

`assemble()` now closes that loop on the *built* tree: every id it is about to
promise must resolve to the very operator placed there, and no operator may
occupy more than one position. It refuses with `AssemblyError` otherwise, so
no caller is handed an id that rewrites the wrong subtree.

The same probe found the identity check alone is not enough: `x_1` *does*
round-trip, yet `replace_node('x_1', Src(0))` still returned 50 where dropping
instance 1 is 40, because `eqx.tree_at` rewrites the one position the lookup
reaches and the fork had folded the operator in twice. That is not specific to
`many` — a *single* operator at such a node had the same defect all along
(`replace_node('x', 0)` gave 10.0 where 0.0 is correct). `Assembly.aliased`
now records those nodes and `replace_node` refuses on them; reading still
works, since `assembly[nid]` genuinely is the operator sitting there.

No shipped graph is affected: every node of the radio template reaches the
sink by exactly one path, so `aliased` is always empty there and the `assemble`
check never fires.

**Follow-up: the same rule now covers binding, which is the route that
mattered.** `replace_node` was guarded; `ParameterSpace.bind` was not, and
`into` selectors are documented as `lambda p: p["gain"].gain`, so they go
through `Assembly.__getitem__` and never through `replace_node`. On the
fork-rejoin graph with one `Src(10)` at `x` (forward 20.0) the space
*validated*, and binding `V=0` gave 10.0 where 0.0 is correct — silent,
finite, correctly shaped, wrong, in the inference path.

`ParameterSpace.validate` now refuses a binding whose `into` selector lands
inside an aliased node, naming the latent and the node and saying what to do
instead (bind downstream of the fork, or restructure the graph). Every copy is
checked, not only the one `__getitem__` reaches, so a selector spelled out by
hand to the second copy is refused too. Gradients were never the problem here:
both copies carry their own and the two sum to the true derivative — the
pytree is a correct *two*-parameter model where a one-parameter model was
declared, which is why `validate` is a complete gate for it.

**`Assembly.__getitem__` stays an inspection API.** Reading an aliased node
returns the operator that genuinely sits there; only writing refuses. That is
pinned by tests on both sides so a later editor does not "fix" the read.

**Known limit.** A hand-rolled `eqx.tree_at(lambda a: a[nid].x, ...)` goes
through no framework call, so nothing can intercept it and it still rewrites
one copy only. `Assembly.aliased` is documented as public API for exactly
this: check it yourself before writing such a selector.

**Stated envelope.** The design promise — any assembled graph serves both
forward modelling and inference — holds while every node reaches the sink by
exactly one path, because assembly folds the graph to a *tree*. That is now
written down in `assemble()` as a limitation with its reason rather than left
as a silent assumption. Not making the fold duplicate operators at all is the
real root-cause fix; it is an architecture change (evaluating the graph as a
memoised DAG) with consequences beyond rewriting — a duplicated *stochastic*
operator is evaluated twice with different randomness — and is left to its own
decision.

### Fixed: `Latent(prior=...)` had no effect at the conjugate exits

`ParameterSpace` is the one place this package says what a latent is, and
`to_numpyro_model` reads the `prior=` on it. `wiener_solve`, `gcr_sample`,
`iterative_gls` and `condition_estimate` did not — `grep '\.prior'` over
`inference/linear.py` and `inference/gls.py` returned nothing. Same twin, same
space, only the declaration changed:

```
declared prior Normal(1.0, 0.05)  -> wiener_solve(prior_std=1, mean=0) = 0.999997795
declared prior Normal(9.9, 1e-4)  -> wiener_solve(prior_std=1, mean=0) = 0.999997795
declared prior None               -> wiener_solve(prior_std=1, mean=0) = 0.999997795
```

Bit-identical. The numbers had to be hand-passed as keywords and hand-synced
across every re-parameterization, and one space could be given to NUTS and to
`gcr_sample` and target two different posteriors with nothing raised.

`LinearBlock` now carries the declaration, and `prior_std=` / `prior_mean=`
default from it. Three consequences, all of them refusals rather than guesses:

* a keyword that **contradicts** the declaration raises, naming both numbers —
  neither silently wins;
* a declared prior with **no conjugate Gaussian form** (Half-Normal, Uniform,
  LogNormal, MultivariateNormal, any truncation) raises at these exits and says
  that `to_numpyro_model` + NUTS is where that space belongs. It is not
  approximated by its first two moments;
* the Gaussian is identified by **type, never by attribute**. `LogNormal`
  carries `.loc`, `.scale` and a `.base_dist` that really is a `Normal`, so
  duck-typing would have read a lognormal prior as a Gaussian one and returned
  a finite, confident posterior for a parameterization nobody declared.
  `Independent` and `.expand()` *are* unwrapped, being re-shapings of a Normal
  and nothing more.

Keyword-only use is unchanged, and was checked by running rather than by
reasoning: `examples/gls_gcr.py`, `examples/noise_wave_gcr.py`,
`examples/tutorial_gcr.py` produce byte-identical output before and after, and
`examples/three_ways_to_a_posterior.py` — which declares a prior *and* passes
the matching keywords — differs only in its wall-clock timings.

**The reconciliation is decided on the values, not on whether a trace is
open.** Two concrete numbers are the same two numbers inside `jit` as outside
it, so the check is evaluated where it is written rather than emitted into the
enclosing trace. Staged, `bool()` on the comparison raises, a settled `True`
comes back as unanswerable, and the guard refuses a correct call while blaming
a tracer that is not there. Two calls hit exactly that, and neither was covered
by the tests above:

* `iterative_gls` with a declared prior under **any prediction-dependent noise
  model** — `RadiometerNoise`, the default, and the entire reason the function
  exists. It resolves the prior once and re-passes it into `wiener_solve` from
  inside its `lax.while_loop`, so the guard met a live trace on every
  reweighting step. `HomoscedasticNoise` has nothing to reweight and returns
  before the loop, which is why the first round of tests missed it;
* any agreeing keyword under `jax.jit` or `eqx.filter_jit`, with both numbers
  plain Python floats closed over by the traced function.

Only a **genuine tracer** is undecidable now, and it is recognised structurally
— over the pytree leaves, so a tracer inside a list is still one. Deciding it
from the failure instead cannot work: `TracerArrayConversionError` is a
`TypeError`, so an unanswerable comparison would come back as a settled
*disagreement*, naming two numbers as contradicting when one of them has no
value yet. The refusal message now says which side is traced, and says that
being inside a trace is not itself the problem.

The comparison stays in `jnp`, which canonicalizes both sides to the working
precision. Deciding it in NumPy instead widens a declared `float32` scale to
`float64` and makes `prior_std=0.05` a contradiction of
`Normal(jnp.asarray(1.0), jnp.asarray(0.05))`, whose scale reads
`0.05000000074505806` once widened — the same false refusal, moved rather than
removed.

Separately, `linear_operator` carries the prior of the latent it **resolved**,
not `space.latents[0]`'s. The two coincide for every single-latent space, which
is what `ParameterSpace.direct` builds and what all the tests above used; they
come apart in the Gibbs sweep over several latents this module is designed
around, where the block for `'gain'` would have silently solved with another
latent's prior.

### The calibrators and the sampler now refuse an `observed` they would broadcast

`wiener_solve` already refused a mis-shaped `observed`, and its message already
explained why. The other three exits did not. Passing `observed[0]`, shape
`(8,)`, to a model predicting `(24, 8)` — one slicing mistake — produced this:

```
  AdamCalibrator     : NO ERROR. final loss = 0.216759, max|g-g_true| = 0.4576
                       g[:4] = [1.0024, 1.0024, 1.0024, 1.0024]  truth [1.0, 1.02, 1.04, 1.06]
  GradientCalibrator : NO ERROR. final loss = 0.257105
  NUTS               : NO ERROR. max|mean-g_true| = 0.4575
```

A converged loss of 0.22, a healthy chain, and every recovered gain wrong. The
loss history was the only evidence available, and it said the fit worked.

The refusal now lives in **one** place,
`rheplicant.inference.check_observed_shape`, and the message that was already
right is now the message all of them give. `GradientCalibrator.fit` and
`AdamCalibrator.fit` check at entry (one `jax.eval_shape`, before `lax.scan`,
so a 10⁶-step fit fails as fast as a 10-step one); the NumPyro observation site
checks while the model is traced, so it costs nothing at run time and fires
before the first draw. `observed=None` is still the prior-predictive call.

The test is shape *equality* — not rank, and not broadcast-compatibility —
because the dangerous slices are exactly the compatible ones. `(24, 1)` (one
frequency channel where the whole band was meant) and `(1, 8)` (one time sample
where the whole record was meant) preserve rank and broadcast just as silently
as `(8,)` does; a rank-based check would admit both, and then a calibrator
converges and `wiener_solve` returns a well-shaped `(24,)` gain estimate from a
single column of data. All three shapes are refused at all four exits, and
`tests/inference/test_observed_shape.py` sweeps all three through each of them.

Nothing that broadcast legitimately changed: `noise_std` is still documented as
broadcastable to the prediction, and still is.

`fisher_information` was named in the same audit finding. It has no `observed`
argument — a Fisher forecast is a function of the model and the noise alone — so
there is nothing there to check; its one data-shaped argument, `flags`, was
already guarded by `FlaggedNoise`. `tests/inference/test_observed_shape.py`
pins that reading so the signature cannot gain an unchecked `observed` quietly.

### `identifiability()` — the diagnostic that can see across Gibbs blocks

New **`rheplicant.inference.identifiability`**. Every convergence guard in
`inference.linear` is computed *from one block*: a CG residual and
`condition_estimate`'s κ are both properties of the operator for the block
being solved, and `check_linearity` asks about one latent at a time. None of
them can see a degeneracy whose two halves live in different blocks — and they
are not wrong to pass, because each conditional of a bilinear model genuinely
*is* affine.

The failure that leaves behind is silent and large. An alternating solve over
gain × antenna temperature reports **κ = 1** and a CG residual of **1.7e-7**
while sitting **2288 K** from the truth.

`identifiability(space, pipeline, state)` column-normalises the Jacobian of the
prediction with respect to *all* the declared latents and takes its SVD. On the
same two parameterizations, with and without a known 5000 K calibration tone:

```
free-per-cell T_ant,  tone ON    n_par=72 rank=64 nullity=8   weakest identified 7.071e-01
free-per-cell T_ant,  tone OFF   n_par=72 rank=64 nullity=8   weakest identified 7.071e-01
(3,3)-basis T_ant,    tone ON    n_par=17 rank=17 nullity=0   weakest identified 6.790e-02
(3,3)-basis T_ant,    tone OFF   n_par=17 rank=16 nullity=1   null direction at  6.647e-17
```

The tone buys **exactly nothing** against a free-per-cell antenna temperature —
the free cell at the tone's channel absorbs the gain sample by sample — and
**everything** against a frequency-smooth one. And the same partition, asked one
block at a time, reports `gain` rank 8 of 8 and `t_coeff` rank 9 of 9: two clean
bills of health on a model that is degenerate.

The null space comes back **named**. `report.participation(0)` returns
`{'gain': 0.5014, 't_coeff': 0.4986}` — the bilinear degeneracy, read as the
combination it actually is — and `report.direction(0)` returns arrays shaped
like the latents, in raw units, such that adding a small multiple of them to
the parameters leaves the prediction where it was.

Three parts of the method are load-bearing, and all three are pinned by tests
that fail when they are removed:

- **Column normalisation.** Two perfectly identified latents whose scales
  differ by 1e10 have a raw singular-value ratio of 1e-10; without the scaling
  the rank test reports the choice of units and calls the model degenerate.
- **float64, whatever the caller has configured.** In single precision the
  degenerate model's null direction surfaces at **3.1e-8** — above any usable
  tolerance — and is reported as identified. The function forces
  `jax_enable_x64` for the duration and restores it afterwards, including on
  the way out of an exception, and hands back numpy arrays so the precision
  survives leaving the context.
- **A threshold that is exposed and argued for.** `DEFAULT_RANK_RTOL = 1e-8`
  sits in an 8.7-decade window (1e-13 to 4.8e-5) where every choice returns the
  same verdict on the measured model. `report.weakest_identified` is there to
  be read before the verdict is trusted.

`names=` asks the conditional question a Gibbs block faces, `at=` moves the
evaluation point the way `linear_operator`'s `at=` does, and the function
refuses rather than guesses on an empty or repeated selection, a complex or
integer latent, and a model that pins its own prediction to float32.

### Two inference tutorials, and a sampler bug they found

New [`docs/tutorial-gcr.md`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/tutorial-gcr.md)
and [`docs/tutorial-nuts.md`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/tutorial-nuts.md),
backed by
`examples/tutorial_gcr.py` and `examples/tutorial_nuts.py`. One ring toy, two
opposite questions: 256 sky pixels by exact conjugate solve, three beam
parameters by gradient MCMC. Every number in both pages is the script's real
output.

**The GCR tutorial ends by breaking itself.** With a beam that resolves the sky
the RMS error equals the posterior σ and coverage is exactly 68 %. Widen the
beam past the structure and κ rises 13×, the posterior keeps 63 % of the prior
instead of 4 %, and coverage climbs to 100 % — nothing fails, and every
diagnostic says the answer is now mostly prior. An estimator without error bars
would return a similar map and report none of it.

**The NUTS tutorial fails first, on purpose, because it really did.** Written
the obvious way it produces `r_hat = 840` and an effective sample size of **2**
out of 8000 draws, with no exception, no NaN and means that look like numbers.
Two hypotheses were tested: the likelihood is multimodal (**wrong** — a scan
finds one maximum), and the posterior is a needle inside its prior (**right** —
500× narrower).

That produced a real fix in the library. New
**`rheplicant.inference.init_to_declared(space)`**: `ParameterSpace` already
declares where to start, the calibrators and `check_linearity` both use it, and
`to_numpyro_model` was dropping it on the floor. One line takes `r_hat` from 840
to **1.002** and `n_eff` from 2 to **1327** — and the run gets *faster*.
Measured on the same problem, tightening the priors and tripling the warmup
each change nothing whatever (`r_hat` 1123 and 1124).

The NUTS page also documents what the reference pages never did: `r_hat`,
`n_eff` and divergences, what each one means, and a six-point checklist for any
run.

### The CST beam reader moved to limTOD

D20 moved the horizon partition upstream and left the CST far-field reader
behind, calling the distinction "a follow-up, not a distinction worth
defending". It was not worth defending: reading a measured horn into a beam map
is the same kind of thing as deciding how that beam weights the sky, and limTOD
already had the slot — `limTOD.uvbeam` does exactly this job for pyuvdata.

`limTOD.cstbeam` is now its sibling, and gained something it could not have
while living downstream: a **`cst_beam_func`** satisfying `TODSim`'s
`beam_func(freq=..., nside=...)` contract, so a CST horn drops straight into
limTOD's own simulator. It caches each file's HEALPix resampling too — a
200-channel sweep over a 61-file directory now parses each file once rather
than hundreds of times.

`read_cst_farfield`, `cst_frequency_table` and `cst_beam_maps` remain importable
from `rheplicant.radio` and are unchanged in signature. They are pass-throughs
now, and the seam has exactly one job: **units**. `Coordinates.freq` is in Hz;
limTOD is in MHz throughout. Each package keeps its house convention and the
adapter is where they meet.

Consequently a bad export raises `ValueError` (limTOD's) rather than
`StateValidationError`, as already happened for `horizon_truncated_beam` in
D20. rheplicant's beam tests now cover the seam and RHINO's real horn; the
synthetic convention tests — the theta-fastest reshape, the `phi_sense`
reflection, the frequency interpolation — went upstream with the reader rather
than being duplicated across two repositories.

The `limtod` extra is floored at **1.10**, which is the release the reader
landed in. The runtime gates still check the *symbol* rather than the version,
and that is not redundancy: an editable install reports whatever version its
dist metadata was written with, and this repository's own environment was found
sitting at a recorded `1.8.0` while running 1.10.0 source, `limTOD.cstbeam`
importable throughout. A version check would have refused a fully capable
install. The floor says what to install; the symbol says what is there.

### NUTS through the noise model, then inference without a likelihood at all

**`to_numpyro_model` accepts a `NoiseModel`.** With `RadiometerNoise` the
observation scale is a function of the sampled parameters, so
`Normal(loc, scale).log_prob`'s `-log scale` puts the log-determinant in the
potential automatically — this is the *full* Gaussian density, not the
generalized least squares `iterative_gls` converges to. An unobserved sample
(infinite σ) is masked rather than given an infinite scale, which would send
the whole potential to `-inf`; `flags=` and wrapping in `FlaggedNoise` are now
the same code path.

*Validated against an exact sampler.* On a linear-Gaussian problem
`gcr_sample` draws the posterior in closed form with no chain and nothing to
diagnose, and NUTS must reproduce it. It does: **mean to 0.00σ, width to
1.00×**.

**New `rheplicant.inference.npe` — amortized neural posterior estimation.**
`simulate_pairs` draws (θ, x) from the prior and the simulator;
`NeuralPosterior` is a conditional Gaussian mixture; `train_posterior` fits it.
No likelihood is written anywhere, and a second observation costs a forward
pass — measured at **3.5 ms against 66 ms** for a fresh exact solve.

**An approximate posterior has no internal notion of being wrong**, and both
failure modes appeared while building this, pushing in opposite directions:

| simulations | steps | components | width / exact |
|---|---|---|---|
| 8 192 | 1 500 | 1 | 0.88 |
| 8 192 | 4 000 | 2 | **0.60** |
| 32 768 | 1 500 | 1 | **0.98** |

Too few simulations and the width is wrong; too many steps on a small bank and
it over-fits, which makes `q` too **narrow** — the failure that presents as a
better answer. So `train_posterior` holds out a validation split by default and
**returns the best validation step, not the last**: in the worked example that
is step 489 of 2000, with the training loss still falling throughout.

New `examples/three_ways_to_a_posterior.py` puts all three engines on one
problem, in the order that makes them checkable.

### `iterative_gls`: finding the covariance a GCR draw is conditioned on

`gcr_sample` is a linear sampler *given* a covariance. Under `RadiometerNoise`
the covariance is not given — σ tracks the prediction, so the weights depend on
the solution and the solution depends on the weights.

**`wiener_solve` and `gcr_sample` did not change.** They already accepted an
array `noise_std`, and their linear-Gaussian correctness was already tested.
The new module supplies only the thing that produces σ:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau), prior_std=P)
draw, _ = gcr_sample(block, observed, noise_std=found.noise_std, prior_std=P, key=k)
```

A **matrix-free** port of hydra-tod's `hydra_tod.linear_sampler.iterative_gls`
— that one forms a dense `U` and `N_inv`; this runs the same fixed point on the
block's JVP and VJP. A transcription of the numpy original is the test oracle.

**The convergence tolerance cannot have a fixed default**, and the first one
tried (`1e-8`) violated both bounds in turn. Two independent floors limit how
small a step is measurable: the arithmetic's epsilon (float32's is `1.2e-7`, so
`1e-8` is rounding — the loop ran to its cap reporting `converged=False` for a
run settled at `delta = 7e-8`), and the **inner CG tolerance**, since
consecutive solves differ by their own residual whatever the outer iteration
does. The latter binds in float64, where a tight `tol=1e-10` made `8·eps` far
*too tight* and it failed the same way. The default is `max(8·eps, tol)` — both
derived, neither tuned — and a test pins the failure mode.

**The mean can be weight-independent while the width is not.** New
`examples/gls_gcr.py`: three switched loads against three per-channel
noise-wave unknowns is a **square** system, so reweighting moves the point
estimate by nothing, exactly. The posterior covariance depends on Σ regardless,
and a GCR draw is a draw of that width — a frozen σ there reports error bars
wrong by −8 % to +8 %, in both directions, on an estimate that was already
right. Where the system is over-determined across genuinely different noise
levels the estimate moves too: a factor of ≈2.3 in recovered RMS on a
prediction spanning a decade.

### MomentRFI flagging: the bridge now actually runs

`MomentRFIFlaggingOperator` had been in the tree with tests since the rename,
and **every one of those tests was skipped** — MomentRFI had no packaging, so
it could not be installed anywhere, so `skipif(find_spec(...) is None)` was
true in every environment including CI. The bridge was written against an API
nobody had called.

MomentRFI is packaged now (upstream), and the tests pass unmodified: the
assumed `fit(waterfall, kernels=..., prior_mask=...)` signature was right.
What first execution added:

- **jit gives bit-identical flags** — asserted by the docstring before, by a
  test now. `jax.pure_callback` is the permanent integration for a boolean
  decision, not a stopgap.
- **`kernel_shapes` earns the matched filter's √K.** On a 3σ-per-pixel blob
  under the fitter's default 4σ cut, round 0 alone recovers *none* of it and a
  single 3×3 box recovers *all* of it, at a 0.15 % false-positive rate.
- **The flags reach the noise covariance, and it matters.** A persistent
  narrow-band emitter on 2 of 32 channels biases a maximum-likelihood amplitude
  **+5.8 %**; wrapping MomentRFI's flags in `FlaggedNoise` recovers the truth
  and agrees to six digits with flagging those channels by hand. The test
  asserts the bias removed, not the mask produced.

New `rheplicant[rfi]` extra. Like `limtod`, it names the requirement rather
than resolving it — MomentRFI is not on PyPI — and the operator's ImportError
now gives both install routes.

### The noise level became a model instead of an argument

`noise_std` was a bare scalar handed separately to five places —
`GaussianLikelihood`, `to_numpyro_model`, `fisher_information`, `wiener_solve`
and `gcr_sample` — each assuming the answer was given and constant. **The
radiometer equation says it is neither**: `sigma = |d| / sqrt(delta_nu * tau)`,
so the noise level is a function of the very thing being inferred.

New `rheplicant.inference.noise`:

- `HomoscedasticNoise(sigma)` — what a bare `noise_std` always meant, now named.
- `RadiometerNoise(channel_width, integration_time)` — the multiplicative
  default, `sigma = |prediction| / sqrt(delta_nu * tau)`. A test asserts its
  sigma matches the one `rhino_cal_jax.power.add_radiometer_noise` actually
  draws with, so the side that scores the data and the side that generates it
  cannot drift apart.
- `FlaggedNoise(base, flags)` — RFI flags reach the covariance by *wrapping a
  noise model*, not by threading a `flags=` keyword through five signatures.
- `NoiseModelLikelihood`, `inverse_variance`, `as_noise_model`.

Every `noise_std=` argument accepts a noise model in place of a number, so no
downstream signature changed. `GaussianLikelihood` and
`MaskedGaussianLikelihood` are unchanged and tested to agree with the general
form to roundoff.

**The log-determinant is not a constant when sigma depends on the prediction,
and dropping it is a different estimator.** For the multiplicative model both
are solvable in closed form: generalized least squares returns
`sum d^2 / sum d`, biased high by `(1 + f^2)`, while the full Gaussian density
is asymptotically unbiased. `NoiseModelLikelihood` keeps the term by default;
`include_logdet=False` is the explicit GLS variant. The tests assert the
implemented log-density has a vanishing gradient at each closed form, so the
claim is pinned rather than measured.

**`fisher_information` was reporting the wrong matrix for prediction-dependent
noise** — `J^T N^-1 J` is not the Fisher information when the covariance
carries parameter dependence. The variance term
`2 (d log sigma/d theta)^T (d log sigma/d theta)` is now included whenever the
noise model reports `depends_on_prediction`. Under `RadiometerNoise` it is
exactly the factor `(1 + 2 f^2)`; without it a forecast reports error bars too
wide by `sqrt(1 + 2 f^2)`.

### Docs: the equations were never rendering, and the framing was in the wrong place

**Every equation in the physics pages shipped as literal `$$...$$` source
text.** MyST's `dollarmath` extension was not enabled and MathJax was not
loaded, so `sky-to-receiver.md` and `sky-engines.md` — both written in TeX —
rendered their maths as dollar signs. A docs build cannot warn about that; only
a reader can. `dollarmath`, `amsmath` and `sphinx.ext.mathjax` are on now, and
Eq. 1, the four coupling spectra, the rank counting and the horizon split all
display as equations.

**"Operators plus three structures" now appears where it belongs.** Cascade
(`Pipeline`), sum (`SumOperator`) and switch (`SelectOperator`) are the entire
composition vocabulary, and the four node kinds map onto them one-to-one. That
framing is now stated in the docs landing page and in the tour's composition and
graph-assembly sections, and developed in full on
[the canonical signal path](https://rheplicant.readthedocs.io/en/latest/signal-path.html)
— rather than being introduced incidentally on a physics page. Two rules fall
out of it rather than being extra: a junction or selector with one live input is
traversed as identity, and `many` instances compose the way their consumer
composes.

It also makes explicit something the code always allowed and the docs never
said: **`RADIO_GRAPH` is RHINO's template, not the framework.** `SignalGraph`
and `register_graph` are public and domain-agnostic; another instrument is
another template. A documented end-to-end path for supplying one is flagged as
planned.

**`docs/sky-to-receiver.md` rebuilt** around three figures generated by running
the documented path (`docs/_generate_receiver_figures.py`, same contract as the
engine and inference generators — transparent, theme-paired SVGs from live
code):

- *the horn against the horizon* — directivity versus zenith angle with the
  azimuthal spread as a band, and the below-horizon share per frequency;
- *the cascade* — 3026 K collected → 2975 K after the horizon split → 2895 K
  after the horn's ohmic loss → 768 K delivered, each labelled with what makes
  it different from the others;
- *the recovery* — truth against the Wiener mean for all three noise-wave
  spectra, with residuals on a ±0.45 K scale.

Plus a mermaid diagram of the assembled path showing all three structures at
once, and the three unguarded joins kept as call-outs where a reader meets them.

The diagram labels each box with what its operator *does*, and does **not**
label the edges: a cascade is a run of arrows, while a sum and a switch are
properties of the node that collects them, so writing all three as edge labels
implied a relationship between two boxes that does not exist. It read as though
`beam_spill` were a contribution added to the sky rather than a transform of it
— which it is not, and a call-out now says why (a mixture whose weights add to
one, tested by the isothermal invariant, versus the atmosphere's independent
contribution).

### Changed: the horizon physics moved to limTOD, where it belongs

`horizon_truncated_beam` and `DriftScanProjector.horizon_fraction()` were
implemented here. They should not have been: how a beam weights the sky, where
the horizon falls in it and what share of its solid angle survives are limTOD's
subject — exactly as the noise-wave data model is `rhino_cal_jax`'s (D15).

Now in **limTOD 1.9** (`limtod_jax.driftscan`): `horizon_partition_weights`,
`horizon_truncated_beam`, `horizon_beam_fraction`, with 25 tests including the
painted-ground closure that decides the conventions. `horizon_weights` is
unchanged — its masking semantics were never wrong; the partition is a
different object and now says so.

What stays here is **placement**. `BeamSpillOperator` consumes `f_sky` and puts
it at the `beam_spill` node; it does not compute it.
`rheplicant.radio.beams.horizon_truncated_beam` and
`DriftScanProjector.horizon_fraction()` are pass-throughs whose only added value
is inferring `nside` from maps the caller already has, and both feature-gate on
the upstream symbol so an outdated install is named at the boundary rather than
raising `AttributeError` midway. The `limtod` extra is floored at **1.9**.

The tests moved with the physics: this side no longer re-asserts the partition,
the half-counted horizon ring or the zenith-only exactness. Duplicating them
would be two copies of a moving target. What is tested here is the seam. See
D20.

### Changed: the graph knows how switches compose, and a fixed mask is a constant

Two claims in the previous entry were wrong, and both were wrong in the same
direction — treating a limitation as a fact about the physics.

**`many` instances now compose the way their CONSUMER composes.** `many=True`
at a source folded its instances into a `SumOperator`, always. That is right for
a junction and wrong for a selector: a switch picks one source per sample, it
does not add them up. So multi-load switching — three distinct sources, the
minimum for an identifiable per-channel noise-wave fit — could not be expressed
through `assemble()`, and the documented workaround was to hand-build the
`SelectOperator`. That workaround is how a `Pipeline`-instead-of-`SumOperator`
bug got into an example and survived until a gradient came back exactly zero.

`cal_loads` is `many=True` now, and the whole switching cycle comes out of
`assemble()`:

```python
twin = assemble(SkySourceOperator(...), BeamSpillOperator(...),
                AtmosphericEmissionOperator(...), AntennaLossOperator(...),
                CalLoadOperator(t_load=...), CalLoadOperator(t_load=...),
                NoiseWaveOperator(...))
twin["receiver_input"].names   # ('observed_astro_sky', 'cal_loads', 'cal_loads_2')
```

Six operators, four different composition relationships (sum, chain, trunk,
switch), and not one line saying what connects to what. The other half of the
rule already held and is now pinned: a selector with one live branch is
traversed as identity — no switch array required, no `SelectOperator` left
behind. A source feeding both a selector and a junction keeps the Sum (no
single right answer; no shipped graph has one), and the fan-out is kept beside
the fold state so every other path folds bit-for-bit as before — `SumOperator`
splits the PRNG key per branch, so a flatter tree would be a different seeded
run, not just a different shape. See D18.

`Assembly.__getitem__` was fixed alongside: a branch spanning
`observed_astro_sky → beam_spill` is labelled by its first node, so the lookup
returned the *fold rooted at* the node instead of the operator *at* it —
contradicting its own documented contract and making
`eqx.tree_at(lambda a: a["observed_astro_sky"].sky_model.maps, ...)` fail on an
attribute the caller could see in the source. Reaching any parameter by its
graph node now works wherever the fold put it.

**The horizon mask does not cost 8×; masking the alms per call does.**
`DriftScanProjector(horizon_mask=True)` rotates into the horizontal frame,
synthesizes, multiplies, re-analyzes three times and rotates back — on every
call. Measured at nside 16 / lmax 47: **14.6 ms against 1.79 ms unmasked, 8.2×**
(the previous entry called it 7× and treated it as the price of asking for the
horizon).

It is not. The horizon is static in the horizontal frame and a drift scan's
pointing is fixed by definition, so the masked beam is a **constant**.
`rheplicant.radio.beams.horizon_truncated_beam` truncates the beam MAP once,
before analysis:

```python
beam_maps, f_sky = horizon_truncated_beam(beam_maps, el_deg=90.0, apod_deg=3.0)
```

**1.04×** — free — and the same instrument to 2.8e-5, the residual being the
alm→map→alm round trip the masking path takes *before* it masks. At a zenith
pointing no rotation is needed at all, and that is provable rather than assumed:
`horizon_weights` is a pure function of elevation, limTOD's horizontal chart
puts the zenith at the pole and the beam-local chart the boresight, so at zenith
they share a pole and a pure-elevation mask is invariant under the rotation that
still separates them. Away from zenith the function refuses rather than
hand-derive the rotation, and `horizon_mask=True` keeps its place for that case.
It returns `f_sky` with the maps because they are the same sum. See D19.

`examples/sky_to_noise_wave.py` uses both, and lost its entire hand-wired
selector as a result.

### Added: RHINO's horn on the sky, feeding the noise-wave receiver

The two halves were already there — limTOD says what the antenna sees, the
Noise-Wave GCR draft's Eq. 1 says what the receiver does with it — and they meet
at one quantity, `T_src`. `SkySourceOperator` upstream of the `receiver_input`
selector now feeds `NoiseWaveOperator` directly, and the path is tested end to
end against Eq. 1 written out by hand (4.0e-16 relative).

**`AntennaLossOperator`** and the `antenna_loss` graph node (graph v1.3) close
the physics gap that separated them. A real antenna dissipates part of what it
collects and re-emits it at its own temperature, `T = eta T_collected +
(1 - eta) T_phys`. This is a *different* loss from the noise-wave stage's
`c_s = (1 - |Gamma|^2)|F|^2`: ohmic dissipation inside the antenna versus
impedance mismatch at the receiver input. They multiply, and only the ohmic one
adds emission of its own — an efficiency folded into the noise-wave couplings
would be indistinguishable from a mismatch in the fit while silently dropping
that term.

The node sits on the trunk between `t_ant_sum` and the switch: it acts on
everything the beam collected (unlike atmospheric opacity — D13 moved the
atmosphere *off* the trunk for exactly the reason that does not apply here) and
on nothing that connects downstream, so the calibration loads arrive
unattenuated. The pairing is pinned by its thermodynamic fixed point — an
antenna at `T` looking at a sky at `T` delivers `T` for any efficiency — which
`(eta, eta)`, `(eta, 1)` and `(1, 1 - eta)` all fail. See D16 in `DESIGN.md`.

**`rheplicant.radio.beams`** reads RHINO's horn as it is actually shipped: CST
Studio far-field ASCII exports, one file per frequency, total directivity in dBi
on a regular (theta, phi) grid. `cst_beam_maps` samples them onto HEALPix in
limTOD's beam-local convention as linear power and interpolates in frequency
(extrapolation beyond the simulated band is refused). Validated against the real
horn: the directivity still integrates to 4*pi after resampling, the boresight
lands on the pole, and the below-horizon fraction survives. CST azimuth is
measured from the model's `+x` axis, which is a fact about the as-built horn and
not about the file — `phi0_deg`/`phi_sense` expose the offset and the handedness
as assumptions to check, not results.

**`examples/sky_to_noise_wave.py`** runs the whole path on the real horn:
CST → HEALPix → drift-scan m-mode sky with `normalize_beam=True`, ground spill
and atmospheric emission, ohmic loss, a three-source switching cycle
(antenna + ambient + hot load), Eq. 8 fractional radiometer noise, a closed-form
recovery of the per-channel noise-wave temperatures with the sky treated as
known data (0.11–0.14 K RMS, kappa ~ 40), and one gradient from the HEALPix sky
map through the beam convolution, the switch and Eq. 1.
`docs/sky-to-receiver.md` walks through it.

**Fixed — an out-of-range switch value now refuses instead of clamping.**
`SwitchCycle`'s range check needs concrete values and is skipped under tracing,
which is the production path. There the two consumers of one switch array
disagreed: `SelectOperator` selected no branch (`T_src = 0`) while
`SwitchCycle.gather` clamped to a neighbouring coupling row, leaving a sample
with a receiver contribution, no source, and another load's `Gamma` — finite and
correctly shaped throughout. `gather` now fills out-of-range samples with NaN.
(In `rhino_cal_jax`; `one_hot` needs no change, since an unmatched sample
already gets an all-zero row.)

**Three joins that carry no structural guard** are documented and pinned by
`tests/radio/test_sky_noise_wave_integration.py` (21 tests), alongside the
physics — a matched antenna passes the sky through untouched, a mismatched one
attenuates it by exactly `c_s`, the receiver terms do not scale with the sky,
load samples never see it and antenna samples never see the load:

- **`normalize_beam` decides whether `T_src` is a temperature at all.** Both sky
  engines default to `False` (numpy limTOD's convention), returning `∫BT` rather
  than `∫BT/∫B`. Against a uniform 200 K sky: `True` gives 200.0000 K exactly, a
  raw beam gives 32838 K, and a beam normalized by hand to unit pixel sum still
  gives 200.4113 K — **+0.21 %**, growing to ~4 % at nside 8 with a 20° beam,
  because the band-limit truncates the denominator too. The hand-normalized case
  is the one that hides. Now documented in `docs/sky-engines.md`, which had no
  mention of `normalize_beam` at all.
- **`gamma_src`'s row order must match the selector's branch order.** Both are
  `(n_source, n_freq)`, so a transposition is shape-legal; measured cost, 46 K
  peak / 28 K mean on a 545 K signal. Read the order off the assembly
  (`twin["receiver_input"].names`) rather than assuming it.
- **A hand-wired antenna branch needs `SumOperator`, not `Pipeline`.** A
  `Pipeline` of *source-type* operators replaces the data at each stage, so
  `Pipeline(sky, ground, atmosphere)` returns the atmosphere alone — the sky
  silently gone, the result finite and correctly shaped. This was a live bug in
  the first draft of the example, caught only because the gradient with respect
  to the sky map came back exactly zero. Whenever `assemble()` could have built
  the branch, check the hand-wiring against it.

**The horizon split (`BeamSpillOperator`, `beam_spill`, graph v1.4).** 1–3 % of
RHINO's horn response is below the horizon and sees ground, not sky, so the
antenna collects `f_sky <T_sky>_masked + (1 - f_sky) T_ground`.
`DriftScanProjector(horizon_mask=True)` supplied the masked average and
`GroundPickupOperator` could supply the ground term, but nothing applied the
`f_sky` weight to the sky branch — which made masking, on its own, no better
than not masking: at a 3000 K sky either choice is a ~200 K bias.

`BeamSpillOperator` applies both halves, so the weights sum to one by
construction, and `BeamSpillOperator.from_projector(projector, t_ground=...)`
reads `f_sky` off the same beam that will supply the sky —  the one call that
cannot get the two out of step. `DriftScanProjector.horizon_fraction()` computes
it; call it before `to_reference_frame()`, which folds the mask into the cached
alms and leaves no unmasked denominator (it raises if asked afterwards).

Every decision here was settled by measurement, against a projector run on a sky
map with the ground painted in at latitude 90, where the local horizon coincides
with the celestial equator and stops moving with LST. Residual on the ~200 K
effect, at nside 16:

| `f_sky` from | residual |
|---|---|
| the masked beam's harmonic integral | −17 K |
| pixel partition, horizon ring dropped (`horizon_weights` as shipped) | −8.6 K |
| pixel partition, horizon ring counted as all sky | +8.7 K |
| **pixel partition, horizon ring counted half** | **+0.005 K** |

Two findings in that table. The band-limited masked beam's solid-angle integral
is not `f_sky` — `map2alm` of a sharply cut map does not preserve the mean, so
it is off by ~0.7 %. And `limtod_jax.horizon_weights` uses a strict `el > 0`,
dropping the whole ring of pixels centred exactly on the horizon (64 of 3072 at
nside 16); a pixel centred on the horizon is half sky and half ground, and the
two one-sided alternatives are symmetric and halve with nside — a miscounted
ring, not anything harmonic. The first implementation used the strict cut and
looked entirely reasonable.

`beam_spill` sits on the ASTRO branch (`beam | observed_astro_sky →
astro_ant_sum → beam_spill → t_ant_sum`), not the trunk: the split applies to
the thing that genuinely is a beam integral over the celestial sphere, while the
other `t_ant_sum` leaves are effective temperatures by D13's construction and
`ground_pickup` in particular IS a below-horizon share. It shares the arithmetic
`a x + (1-a) b` with `AntennaLossOperator` and is deliberately not the same
operator: a spill is a *mixture* (sky and ground at one temperature give that
temperature — no loss), ohmic loss dissipates and re-emits. See D17.

### Added: parameter spaces — infer anything, however it is parameterized

The inference seam could already fit any pipeline leaf. It could only fit a
*leaf*: priors attached positionally and had to match that leaf's shape
exactly. So the models it could express were the models whose parameters
happened to already be stored numbers. A beam described by a width and a
pointing offset, a gain tied across three stages, a positive quantity explored
in its logarithm — each of those required writing a new operator whose leaves
were your parameters, which is exactly the "calibration contaminates the
instrument description" the design exists to prevent.

Two objects now carry the two ideas that were conflated:

- **`Latent`** — a named quantity you infer. Name, initial value (which fixes
  shape and dtype), optional prior, optional `linear=True`. Knows nothing
  about the pipeline.
- **`Bind`** — a rule turning latents into pipeline leaf values. Which latents,
  which leaves, and the function between them. Knows nothing about priors.

`ParameterSpace` holds both. It covers the three shapes a parameterization
takes — direct, tied (one latent to several leaves), derived (several latents
to one leaf) — and `ParameterSpace.raw` takes a bind function outright for
anything they cannot express.

```python
space = ParameterSpace(
    latents=[Latent("fwhm", init=0.5, prior=dist.Uniform(0.15, 0.70)),
             Latent("offset", init=0.0, prior=dist.Normal(0.0, 0.4))],
    bindings=[Bind(("fwhm", "offset"),
                   into=lambda p: p["sky"].projector.matrix, fn=beam_matrix)],
)
forward, start = space.forward_fn(twin, state)   # {"fwhm": ..., "offset": ...}
```

Neither calibrator needed a single change — a dict is a pytree — and posterior
predictive still `filter_vmap`s over the posterior, because bindings are
required to preserve the pipeline's treedef.

**Validation, because every failure mode here is silent.** Duplicate names, a
binding naming an undeclared latent, a latent nothing binds (it samples
happily and returns the prior), two bindings on one leaf, a selector landing
on static configuration, a produced shape or dtype-kind that does not fit its
target, a prior sized differently from its latent, a bind function that
changes the treedef. Each of those otherwise yields a finite,
correctly-shaped, wrong inference. All are caught, almost all on
`jax.eval_shape` — so validation computes nothing and happens once per build
rather than per evaluation, which is why it is not made skippable.

### Added: declared-linear blocks, checked and then exploited

`linear=True` asserts the prediction is affine in one latent — the case that
matters for sky alms and noise-wave amplitudes, ~1e6 real degrees of freedom
where gradient samplers are hopeless and a conjugate-Gaussian solve is exactly
right.

The assertion is checked before anything uses it. `check_linearity` compares
the model against its own linearization at probes spanning 1e-3 to 1e3 times
the latent's magnitude; the span is the point, since `x + eps*x^2` is
indistinguishable from linear near the origin and grossly nonlinear far from
it. A block fails only if it exceeds a relative tolerance AND an absolute
roundoff floor — without the floor the relative measure explodes at small
probes and rejects correct blocks, and the cure a user reaches for on a
spuriously-failing check is to switch it off.

`linear_operator` exports `A`, `A^T` and the offset without forming a matrix
(`jax.linearize` + `jax.vjp`), so applying a 1e6-dimensional block costs one
forward evaluation. `wiener_solve` gives the posterior mean by CG. GCR
sampling adds a fluctuation term to the same right-hand side and is what this
operator is for next.

Two facts measured rather than assumed. `jax.vjp` returns the conjugate
gradient for complex latents, so the identity that holds is
`Re sum(x * adjoint(y)) == sum(forward(x) * y)` — the adjoint of the REAL
inner product, which is the pairing a Gaussian likelihood forms. And
`wiener_solve` carries complex latents as `(real, imag)`, because a real
prediction makes the map R-linear but not C-linear and a Krylov method over C
would be solving a different problem.

### Added: exact posterior draws from a linear block (GCR)

`gcr_sample` turns the Wiener solve into a constrained realization by adding
two white-noise terms to the same right-hand side:

```
(A^T N^-1 A + S^-1) x = A^T N^-1 (d - offset) + A^T N^-1/2 w1 + S^-1/2 w2
```

The right-hand side then has covariance equal to the normal operator itself, so
`x = M^-1 b` carries the posterior mean AND covariance `M^-1 M M^-1 = M^-1`
exactly. Every call is an independent draw — no chain, no burn-in, nothing to
diagnose — for the same single CG solve the mean costs, because the fluctuation
enters the right-hand side and never the operator. That is what makes a
1e6-dimensional block samplable at all.

Both moments are tested against a densely-constructed posterior: 6000 draws
reproduce the covariance to 12% in Frobenius norm, and with uninformative data
the draws reproduce the PRIOR width to 5%. Testing the mean alone would not
have been enough — a sampler that gets the mean right and the covariance wrong
looks entirely healthy.

`linear_operator` and `check_linearity` also gained `at=`, which fixes the
OTHER latents. A block is linear only *given* them, so a Gibbs sweep must
rebuild it wherever they currently are; without `at=` the block silently kept
describing the model at its declared starting point, which is right for exactly
one sweep.

### Changed: Fisher and covariance matrices carry named rows

`cov.sigma("fwhm")` instead of `cov.matrix[0, 0]`; `cov.block("fwhm",
"log_gain")` for a cross-covariance. Spans come from the actual flattening,
not from an assumption about dict ordering. `FlatMatrix.kind` also
distinguishes Fisher from covariance, and `sigma()` refuses on a Fisher
matrix: `sqrt(diag(F))` looks exactly like an error bar and ignores every
parameter degeneracy.

### Removed: `prior_template` / `set_prior` (breaking)

`to_numpyro_model` and `predict_from_samples` take a `ParameterSpace` where
they took a positional prior pytree. Sample sites are named by their latents,
which deleted `_site_name` outright — 50 lines that walked pytree paths and
recognised Assembly/Pipeline/SumOperator containers to reconstruct a readable
name for a parameter that never had one. Once parameters have their own
namespace, the problem does not exist. Net ~90 lines removed from the bridge.

`build_forward_fn` is NOT removed and is not deprecated: it answers "train
everything under this subtree", which is what a neural surrogate's weights
want, while `ParameterSpace.forward_fn` answers "these specific quantities,
possibly transformed or shared".

New: [`docs/inference.md`](https://rheplicant.readthedocs.io/en/latest/inference.html),
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py), and
`docs/_generate_inference_figures.py`, which produces the page's figures by
running the code rather than illustrating it.

### Added: a `limtod` extra — the sky engines install from PyPI

limTOD is [published on PyPI](https://pypi.org/project/limTOD/), so the
engine it backs is now a declared optional dependency rather than a
build-it-yourself instruction:

```bash
pip install "rheplicant[limtod]"
```

The extra floors at `limTOD[jax]>=1.8`, so a fresh install gets every fast
path including the hoisted Wigner plane. The runtime feature check stays, so
an environment already holding 1.7 keeps working and only forgoes that
speed-up — the floor expresses what a new install *should* take, not the
bare minimum the code tolerates. Every `pip install -e '<limTOD>[jax]'` in the
docs, examples and error messages is updated accordingly.

### Performance: the drift-scan engine stops repeating itself

Three hoists, none of which change a number. Measured at nside 64 / lmax 191
/ 512 samples / 32 channels in x64, on the cached-beam projector.

- **`forward_alms()` / `sky_to_alms()` / `mmodes_alms()`** — the sky may now
  enter as pre-analysed quadrature alms. `AbstractSkyProjector`'s contract is
  maps-in, so `forward()` re-ran the analysis on every call; once the beam
  rotation is cached that IS the engine — **176 ms of a 193 ms forward and
  114 MB of its 114 MB peak**. Analysed once instead, a forward drops from
  **163 ms / 122 MB to 1.1 ms / 11 MB** (153x), agreeing to 5e-16.
  `forward()` and `mmodes()` are now thin wrappers, so the map-based contract
  is unchanged. Note `sky_to_alms` uses the QUADRATURE transform, not the
  true-alm one `from_beam_maps` uses for the beam — the two differ by
  `npix/4pi`, which is why this is a method rather than a docstring.
- **The Wigner-d plane is built once per projector.** It depends on the
  pointing alone — LST enters the zyz composition in the first-applied slot,
  so it moves `psi` and never the plane — but `limtod_jax` rebuilt it on every
  call. Hoisted through limTOD's new `dl_array` parameter: **44.0 ms -> 2.8 ms
  (15.6x)** on the rotation at lmax 127, bit-for-bit identical. This is the
  only amortization available when the BEAM is the fitted parameter, where
  `to_reference_frame()` cannot be used because gradients must reach the
  beam-local alms. Skipped gracefully on a limTOD without the parameter.
- **`freq_chunk`** (static, default `None`) walks the frequency axis in
  batches instead of all at once. Peak memory is linear in `n_freq` — 3.4 MB
  per channel, so 114 MB at 32 channels but ~1.8 GB at nside 256. Chunk 8 cut
  the peak 3.2x for 1.7x the time; chunk 1, 9.4x for 7.9x. Off by default
  because below the memory ceiling it is a pure loss.

Requires the matching limTOD (`dl_array` / `dl_plane_for_pointing`) for the
second item only; the others are self-contained.

The docs benchmark was re-timed afterwards, and the split is exactly where the
optimizations predict. The **general** engine is untouched (its per-sample
rotation cannot hoist a plane, and it never enters the m-mode synthesis):
176.4 / 1354.8 / 10150.6 / 65569 ms across lmax 23 / 47 / 95 / 191, all within
the ±7 % run-to-run scatter of the previous run. The **un-cached m-mode** path
gained 1.2–1.8x (115.2 -> 62.6 ms at lmax 191), and the **fully optimized**
path barely moved (63.7 -> 60.4 ms) because its rotation was already cached
and it was already on the FFT synthesis.

One consequence worth knowing: plain m-mode now runs within **4 %** of the
cached-beam + FFT path at lmax 191, where it used to be 81 % slower. What is
left in those 60 ms is the sky analysis, not the rotation — which is precisely
what `forward_alms()` removes.

### Fixed: the normalization denominator was recomputed on every call

`_ones_alm` — the quadrature alms of the ones map, used when
`normalize_beam=True` — is a pure function of the static `(nside, lmax)`, and
a comment in both engines asserted XLA constant-folds it. **It does not.** It
is a full s2fft analysis (63 unrolled ring FFTs plus a loop), which exceeds
XLA's constant-folding budget, so it was traced into every call. Measured at
nside 64 / lmax 191 in x64: **64 ms and 10 MB per call, 7700 lines of HLO**.

Hoisted into JAX's compile-time constant-evaluation context, which computes it
eagerly at trace time: **0.04 ms, 0.15 MB, 48 lines of HLO** — a 1600x
reduction, with bitwise-identical output. End to end on a 32-channel /
512-sample forward the normalization overhead drops from ~51 ms to ~11 ms (the
remainder being the genuine second synthesis for the denominator). Both
comments now state the measured behaviour instead of the assumed one.

Also documented: `horizon_mask=True` on the `"local"` beam frame reads like a
physics switch but is the most expensive path in the projector — it adds a map
synthesis, `mask_iterations` analysis+synthesis rounds and a second Wigner
rotation to every call, roughly 7x the cached unmasked cost. The remedy
already existed (`to_reference_frame()` folds the mask in once); nothing said
so.

### Removed: `MModeProjector` and `LimTODProjector`

**Breaking.** Three sky engines remain, no placeholders: two that compute the
physics (`GeneralPointingProjector`, `DriftScanProjector`, named for the
observation geometry) and `MatrixProjector`, which takes the projection as
data. `rheplicant.radio.sky.projection` is down from five classes to two.

`MModeProjector` was a placeholder that took m-mode transfer matrices as
given. `DriftScanProjector` computes them from the beam, so the placeholder's
only remaining job was explaining that it was not the one to use — which four
separate docs had to say. It could not have grown into the real thing either:
its `(n_freq, n_m, n_pix)` layout is ~6.4 GB at nside 64 / 32 channels / 512
samples, and it required `n_m == n_time`, whereas m-mode pipelines store
`(n_freq, n_m, n_alm)` with `n_m = 2*lmax+1` independent of the sampling.

`LimTODProjector` bridged to numpy limTOD through `jax.pure_callback`: not
differentiable, not vmappable, no adjoint — so it could feed neither
calibration, nor Fisher forecasts, nor sky-space map-making, which is most of
what this framework is for. Now that both real engines are native JAX, anyone
wanting a numpy reference should call `limTOD.simulator.generate_TOD_sky`
directly, which is also the more trustworthy comparison: XLA runs callback
threads with FTZ/DAZ set, so healpy inside one lands ~1e-7 relative from the
same numpy code on the main thread. (This also retires the
`truncate_frac_thres` knob added moments earlier in this same Unreleased
cycle.) D10's host-callback policy is narrowed to match: callbacks are for
inherently non-differentiable steps such as RFI flagging, not for borrowing
numpy physics.

### Renamed: `NativeLimTODProjector` -> `GeneralPointingProjector`

**Breaking, no alias.** The two real sky engines were named on different
axes: one for its provenance (`NativeLimTOD`), one for its applicability
(`DriftScan`). Since BOTH are ports of `limtod_jax`, provenance does not
distinguish them — it just spent the name. Engines are now named for the
observation geometry they serve, which is the question a user actually asks:

| engine | named for | when |
|---|---|---|
| `GeneralPointingProjector` | observation geometry | pointing varies per sample |
| `DriftScanProjector` | observation geometry | pointing is fixed; the Earth scans |
| `MatrixProjector` | the data you supply | the projection is already built |

Module `rheplicant.radio.sky.native` -> `rheplicant.radio.sky.general_pointing`.
No compatibility alias: the old name is gone, so a stale import fails loudly
at import time rather than silently working until it does not.

- **Docs: [sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)**
  — a new page comparing the general and drift-scan engines side by side,
  with figures generated from live code by `docs/_generate_engine_figures.py`
  (one sidereal day of GSM sky through a zenith beam at latitude 53.2°):
  the waterfall the twin produces, the engine-to-engine agreement, the
  wall-clock scaling, and the m-mode spectrum. Every figure ships in light
  and dark variants and is theme-switched by furo. Measured at nside 64
  (lmax 191), 512 samples, 32 channels: the engines agree to **1.4e-15**
  relative, and one forward evaluation costs **66 s** on the general engine
  against **60 ms** with the cached beam and FFT synthesis — about a
  thousandfold. (Both are `forward()` on sky maps, so they include the sky
  analysis; the general engine's timing carries a few per cent of
  run-to-run scatter, so the ratio is not four significant figures.)
  `sphinx-design` is a new docs dependency.
  `--replot` redraws every figure from a cached run, so a purely visual
  change never costs another 20-minute generic-engine sweep.
- `examples/driftscan_mmode.py`: the end-to-end drift-scan demo — build a
  twin from a beam map, cross-check it against the general engine, time all
  three configurations, read off the m-modes, differentiate w.r.t. the beam.
- `DriftScanProjector.from_beam_maps(...)`: build the projector from HEALPix
  beam **maps** rather than alms, with `nside` inferred from the map length.
  The analysis runs in JAX (`limtod_jax.map2alm_iter`), so gradients reach
  the beam map — and it removes a genuine footgun: the quadrature transform
  the *sky* uses (`map2alm_quad`, the one visible in `forward`) silently
  rescales a beam by `npix/4pi`, and nothing previously said which transform
  the beam wanted.
- `DriftScanProjector.uniform_lst_grid(n_time, lst0_deg)`: the LST grid
  `uniform_sampling=True` requires. The natural `jnp.linspace(0, 360,
  n_time)` includes the endpoint and is a turn *plus one step* — a mistake
  this package has already been bitten by — so the correct grid is now
  provided rather than described in an error message.
- `DriftScanProjector` now REJECTS coords whose `pointing` or
  `selfrot_deg` disagrees with its own fixed pointing. Those entries are
  configuration here, not data, so a scan handed to the drift engine used to
  be silently discarded and a different observation simulated: finite,
  correctly shaped, and wrong. Constant pointing that agrees still passes —
  reusing a `GeneralPointingProjector`'s coords is the expected way to switch
  engines — and traced values, which carry nothing to compare, are left
  alone. Uniform-grid violations are also re-raised as `StateValidationError`
  so the whole boundary is catchable as one family.
- Documented the m-mode formalism's source (*M-mode RIME explicit in beam,
  fringe and sky modes*) in the module and the docs; the last line of its
  Eq. (13) is the identity the fast path implements.
- Projector cross-references: the `projection` module ladder, the `adjoint`
  error message, `MatrixProjector`, and `GeneralPointingProjector` now all point
  at the drift-scan engine, and `rheplicant.radio.sky.driftscan` was added to
  the API reference — its ~200 lines of contract documentation were missing
  from the rendered docs entirely. The stale engine ladders in
  `docs/tour.md`, `README.md`, and `DESIGN.md` (D8), which pointed at a
  placeholder as the drift-scan answer, are corrected.
- `DriftScanProjector` (`rheplicant.radio.sky.driftscan`): the m-mode fast
  path for drift scans. Derives the m-mode projection from the beam alms via
  `limtod_jax.driftscan` (one Wigner rotation for the whole scan plus per-m
  phases): equal to `GeneralPointingProjector` with constant pointing to float64
  roundoff at O(lmax^3 + n_time*lmax) instead of O(n_time*lmax^3). The drift
  pointing (az/el/selfrot) is static projector configuration; `coords` only
  supplies `lst_deg`. Ships the exact sky-slot adjoint, an `mmodes` accessor
  (per-frequency Fourier coefficients of the sidereal TOD), and the optional
  horizon mask with cosine apodization (see the limTOD ringing study).
  Requires limTOD >= 1.6 (`limtod_jax.driftscan`); guarded lazy import,
  with a second feature level for `uniform_sampling=True`, which needs
  the FFT fast path and public grid check added in limTOD 1.7 — an
  outdated install now fails at the boundary with a clear message
  instead of an AttributeError inside a traced call.
- `DriftScanProjector.to_reference_frame()` pays the O(lmax^3) Wigner
  rotation once and returns an equivalent projector (new static
  `beam_frame="reference"`) that skips it on every later
  forward/adjoint/mmodes — the difference between rotating once and
  rotating per likelihood evaluation (measured 73-878x fewer per-evaluation
  FLOPs depending on lmax and n_time). A configured horizon mask is folded
  into the cached alms and its flag cleared, so it can be neither lost nor
  applied twice. The precompute is itself pure JAX, so a caller who wants
  beam-LOCAL gradients can still differentiate through it; keeping the
  `"local"` projector remains the way to get gradients w.r.t. pointing.
- `DriftScanProjector(uniform_sampling=True)` routes the time synthesis and
  its adjoint through real FFTs (limTOD's new `uniform=` fast path):
  O(n_time*log n_time) independent of lmax, 19-51x faster than the direct
  phase sum, identical to roundoff. Static by design — dispatching on the
  values of the LST grid is impossible under jit. The projector validates
  the RAW `lst_deg` per call, which stays concrete even inside a jit trace
  (deriving `dphi` there would not), so a bad grid is a clear ValueError at
  trace time rather than limTOD's NaN-poisoned fallback.
- `DriftScanProjector` gained a `beam_ref_lst_deg` invariant: it records the
  LST the cached beam was actually rotated to and only `to_reference_frame()`
  sets it, so `dataclasses.replace(cached, lst_ref_deg=...)` — which would
  measure the phases from a reference the baked-in rotation does not
  correspond to, silently — now fails validation instead. The uniform-grid
  check also stopped upcasting the LST grid to float64 before validating:
  the cast hid the grid's real precision, and limTOD's dtype-scaled
  tolerance rejects a legitimate float32 grid when checked at the f64 bound.
- Test coverage strengthened after adversarial review: every cached-beam and
  FFT test now also runs with a reference LST far from `lst_deg[0]` (they
  were all degenerate at 12.0, so a "use lst[0] regardless" bug was
  invisible — mutation-verified as killed now), compared against the
  general `GeneralPointingProjector` as an independent oracle; and the
  `to_reference_frame` gradient is compared for equality with the uncached
  projector's rather than only checked finite.
- `DriftScanProjector.mmodes()` now rejects `normalize_beam=True`: the
  normalization divides by the ones-map denominator, which is not part of
  the m-mode expansion, so the returned coefficients would silently not be
  the spectrum of `forward()` (~18x off).

### Changed: `NoiseWaveOperator` implements the full Eq. 1 (breaking)

`NoiseWaveOperator` now runs Eq. 1 of the noise-wave GCR note through
`rhino_cal_jax` (see D15 in `DESIGN.md`) instead of the `F -> 1` placeholder
that summed one scalar `Gamma` straight into the sky-side temperature.
`t_zero` is renamed `t_rx`; the scalar `gamma_re`/`gamma_im` pair is replaced
by per-source `gamma_src_re`/`gamma_src_im` of shape `(n_source, n_freq)` plus
`gamma_rec_re`/`gamma_rec_im` of shape `(n_freq,)` — reflection coefficients
belong to *sources*, and the `receiver_input` selector discards source
identity before this node sees the data, which the old single-`Gamma` shape
could not represent. The operator now reads which source is connected from
`coords.extra["receiver_input"]` and raises if it carries more than one source
and that array is absent, rather than silently defaulting to the first.

### Added: the noise-wave temperatures as a checked linear block (worked example)

`examples/noise_wave_gcr.py`: the per-channel noise-wave temperatures
(`t_unc`, `t_cos`, `t_sin`) declared as ONE `linear=True` latent, solved in
closed form (`wiener_solve`, GCR note Eq. 30) and sampled exactly
(`gcr_sample`, Eq. 31). A `--one-source` mode shows what switching buys: with
one load the per-channel design matrix is deficient by a factor of three and
two of every three directions fall back to the prior; with three genuinely
different loads it is square and every direction is data-constrained.

### Fixed: `wiener_solve`/`gcr_sample`'s convergence guard now certifies the ERROR, not the residual

`require_convergence` bounded the relative *residual* of the CG solve, but
what a caller needs bounded is the *solution* error, and jax's `cg` gives no
other convergence signal to check. The two differ by the condition number of
the normal operator `M = AᵀN⁻¹A + S⁻¹`, and κ is large by *design* in exactly
the case these solvers exist for: whenever the data does not fully identify
the block, `λ_min(M)` is exactly `1/prior_std²` and κ runs past 1e6. On a
48-dimensional calibration block (two of every three directions
prior-dominated, `cond(M) ~ 4e8`) CG stopped on a residual dominated by the
one well-constrained direction, having left the prior-dominated directions at
their starting value of zero — `gcr_sample` reported a posterior sigma of
0.026 K where 81 K was correct, understating the width by a factor of ~3000,
always toward false confidence, while the residual sat at ~1e-7 and
`require_convergence`'s old default of `1e-3` never fired.

Tightening `tol` alone would only have moved the threshold at which this same
failure mode recurs — it does not detect anything, so a block conditioned a
few orders of magnitude worse would again pass silently. `require_convergence`
now bounds `κ · relative_residual` instead, with κ estimated by two power
iterations on the operator the solver already applies (`λ_min` measured, not
assumed worst-case, since the rigorous bound `λ_min ≥ 1/prior_std²` would flag
every healthy full-rank block as ill-conditioned). A second, separate error
covers the case where `κ · eps` already exceeds the target: no tolerance or
iteration count helps there, only precision, and the natural response to the
first message — tighten `tol`, raise `maxiter` — would burn iterations to
arrive at an equally wrong answer.

- **`condition_estimate(block, noise_std=..., prior_std=...)`** is new and
  public: it reports κ, which is what a caller needs to choose `tol`
  (`tol ≈ require_convergence / κ`) — matrix-free, like everything else here.
- **`rheplicant.inference.conditioning`** is new: `tree_norm`,
  `largest_eigenvalue`, and `extreme_eigenvalues`, matrix-free spectral
  diagnostics over pytrees with no dependency on the block machinery above
  them.
- **`require_convergence` changed meaning**, from a bound on the relative
  residual to a bound on the relative error. For a well-conditioned block
  κ≈1 and the two coincide, so healthy solves are unaffected; a block the
  data does not identify, which used to return silently wrong, now raises
  and names the remedy (tighten `tol`, or strengthen the prior). `tol`'s
  default is unchanged at `1e-6` — it is the guard that changed, not the
  starting point it checks. `require_convergence=None` still disables the
  guard entirely, and also skips its cost.

## 0.1.4 (2026-07-25)

- Add project logos (a rhino dissolving into digital pixels — the
  differentiable digital-twin motif): a banner heads the README and the docs
  landing page, and the single-rhino mark is the docs sidebar logo.

## 0.1.3 (2026-07-25)

- Repository moved to the [`RHINO-Experiment`](https://github.com/RHINO-Experiment)
  GitHub organization; package metadata URLs (Homepage, Repository, Changelog)
  now point there. Publishing continues via Trusted Publishing under the new
  owner. No code or API changes.

## 0.1.2 (2026-07-25)

- `__version__` is now read from the installed distribution metadata
  (`importlib.metadata.version`) instead of a hardcoded string, so
  `pyproject.toml` is the single source of truth for the version. Falls back
  to `0.0.0+unknown` when run from an uninstalled source tree.

## 0.1.1 (2026-07-25)

- Credit the developers and maintainers (Zheng Zhang, Phil Bull, Jordan
  Norris, Rashi Srivastava) in the package metadata (`authors`) and README.
  First release carrying the full author list on the PyPI page.

## 0.1.0 (2026-07-25)

First PyPI release (`pip install rheplicant`), published via Trusted
Publishing (OIDC). Highlights of the 0.1.0 development line:

### Renamed: REPLICANT -> RHEPLICANT (RHino + REPLICa + ANTenna)

Same portmanteau (REPLICa + ANTenna = a differentiable replica of a radio
antenna), now with the **RH** of RHINO in front — the horn antenna the
framework was first built for. Distribution *and* import name are now both
`rheplicant` (the bare name is free on PyPI, so the earlier `-telescope`
suffix is dropped); import path `replicant.*` -> `rheplicant.*`; source dir
`src/replicant` -> `src/rheplicant`. GitHub repo and RTD project renamed to
`rheplicant`; old URLs redirect.

### Renamed: DIRT -> REPLICANT (a portmanteau of REPLICa + ANTenna)

A digital twin *is* a replica, and this one is of a radio antenna — so the
package is now **REPLICANT** (`REPLIC`a ⊕ `ANT`enna, overlapping the shared
`A`). Distribution name: `replicant-telescope`; import name: `replicant`
(was `dirt` / `dirt-telescope`). A PyPA packaging sample owns the bare
`replicant` name on PyPI, hence the `-telescope` suffix. Old `dirt-telescope`
GitHub/RTD URLs redirect after the rename.

### Rendering: embeddable SVG + documented lit/dim examples

`Assembly.to_svg()` / `SignalGraph.to_svg()` return a self-contained
`<svg>` (opacity classes styled inside the figure), so lit/dim signal-path
renders embed anywhere a plain image does. The docs signal-path page now
shows two real example renders, generated from live assemblies at build
time.

### Graph v1.2: atmosphere as an equivalent-entry pair (D13)

The `atmosphere` node moved from a trunk transform (between `t_ant_sum` and
the receiver-input switch) to a **source leaf** of `t_ant_sum`, parallel to
`ground_pickup`/`t_sys_extra`: `SystemTemperatureOperator` (transform,
`t_sys`) is replaced by `AtmosphericEmissionOperator` (source, `t_atm`, in
`replicant.radio.environment`). A reserved `atmosphere_field` transform on the
astro branch (between `ionosphere` and `field_sum`) marks the strict
radiative-transfer entrance — opacity acts on the astro sky alone, never on
ground pickup. Numerically identical for the additive placeholder; see
DESIGN.md D13 for the rationale.

### Renamed: e-RHINO -> DIRT (Differentiable Instrument Response Twin)

The framework applies to any single-antenna radio telescope (horns, dipoles,
dishes), so the RHINO-specific name was retired. Distribution name:
`dirt-telescope`; import name: `dirt` (was `erhino`). The GitHub repository
moved to `zzhang0123/dirt-telescope` (old URLs redirect). The canonical graph
template is now named "single-antenna".

Initial architecture of the differentiable scientific pipeline framework.

### Inference layer completed (D12)

- **NumPyro bridge** (`to_numpyro_model` — the last stub is gone): pytree
  priors via `prior_template`/`set_prior`, semantic sample-site names from
  stage names, masked Gaussian likelihood (flags -> zero weight), optional
  noise-std inference, `predict_from_samples` posterior predictive.
- **Uncertainty propagation** (`dirt.inference.uncertainty`):
  `fisher_information` (exact Jacobians via jacfwd), `parameter_covariance`
  (Cramer-Rao), `propagate_covariance` (delta-method prediction bands),
  `push_forward` (Monte Carlo). Fisher matches NUTS posterior widths on the
  demo problem.
- **Neural surrogates**: `NeuralOperator` (eqx.nn.MLP as a positive spectral
  response) — hybrid physics+ML with zero special machinery; placed
  explicitly (e.g. `At("bandpass", ...)`). `AdamCalibrator` (pure JAX)
  added; it recovers a rippled bandpass to <1% where fixed-step GD diverges.
- Examples: `bayesian_and_uncertainty.py`, `neural_surrogate.py`.

### Graph-guided assembly (D11)

- `dirt.core.graph`: `SignalGraph` declarative signal-path templates
  (validated DAG, single sink, typed nodes) and `assemble` — compiles a set
  of operator instances into the induced `Pipeline`/`SumOperator` nesting
  (absent sources pruned, absent transforms skipped as identity, junctions
  materialized as sums; deterministic branch order = graph declaration
  order). Result is an `Assembly` operator with lit/skipped metadata,
  node-id access (`assembly["gain"]`, `replace_node`), caller-data guards,
  and lit/dim `to_mermaid` rendering.
- `dirt.radio.graph`: the canonical single-antenna graph (26 nodes) with
  equivalent-entry leaves (`observed_astro_sky` — served by
  `SkySourceOperator`; reserved placeholders `ground_field`, `t_sys_extra`)
  and `graph_node` slots on every radio operator;
  `assemble(*ops)` convenience. Full-set assembly is regression-tested
  bitwise against the hand-built twin.
- `SumOperator`: branch input data now stripped to `None` (D6 enforced);
  added `replace_branch`.
- **Selector nodes** (`SelectOperator` + the `"selector"` NodeSpec kind):
  switched signal paths — one branch selected per time sample via
  `coords.extra[<node_id>]`. The canonical graph gains `cal_loads`
  (`CalLoadOperator` placeholder) and the `receiver_input` antenna/load
  switch, modeling the elements taxonomy's switched calibration signals;
  pass-through (zero cost) when no load is provided.
- **Region coverage**: `graph_node`/`At` accept a tuple of node ids — one
  operator implementing a contiguous template path atomically (disjointness
  and interior-feed validation; addressed by its last covered node).
- **HTML rendering**: `SignalGraph.to_html()` / `Assembly.to_html()` produce
  a standalone lit/dim signal-path page (`examples/render_signal_path.py`).

### Integration seams (added after initial architecture)

- **Modular sky** (`dirt.radio.sky`): `AbstractSkyModel` (params → maps) ×
  `AbstractSkyProjector` (maps → TOD, with `adjoint` for linear engines),
  composed by `SkySourceOperator`. Engines: `MatrixProjector` (precomputed
  `generate_sky2sys_projection` matrix — differentiable today),
  `LimTODProjector` (pure_callback oracle into numpy limTOD),
  `MModeProjector` (m-mode transfer, drift scans). Port task book for the
  native JAX limTOD rewrite: `docs/limtod-port-contract.md`.
- **Native limTOD projector** (`NativeLimTODProjector`): the port contract
  delivered — pure-JAX sky→TOD chain (Wigner rotation + harmonic beam sum
  from the `limtod_jax` package in the limTOD repo), general pointing,
  jit/vmap-safe, differentiable w.r.t. both sky maps and beam alms, exact
  adjoint for `SkySpaceFilter` map-making. Matches numpy
  `generate_TOD_sky(..., truncate_frac_thres=0.0)`; enable x64 for
  quantitative accuracy. Optional dependency: `pip install -e '<limTOD>[jax]'`.
- **MomentRFI** (`dirt.radio.backend`): `MomentRFIFlaggingOperator`
  (host-callback into `IterativeSurfaceFitter`; existing flags become
  `prior_mask`) + `MaskedGaussianLikelihood` (flags → noise covariance).
- **Filters** (`dirt.radio.filters`): `AbstractLinearFilter`
  (extract/remove projection semantics) with `SiderealFilter` (day-repeating
  subspace), `SkySpaceFilter` (CG map-make/reproject through any linear sky
  projector), `FourierBandFilter` (fringe-rate/delay bands); plus
  `ApplyCalibrationOperator` and raw-data preservation via
  `State.checkpoint` / `SnapshotOperator`.

### Core (`dirt.core`)

- `State`: immutable pytree container (traced `data`/`coords`/`env`/`aux`/`key`,
  static hashable `meta` via `FrozenMapping`); functional updates
  (`replace`/`with_data`) and the PRNG protocol
  (`subkey, state = state.next_key()`).
- `AbstractOperator` / `LambdaOperator`: the universal `State -> State`
  contract with declarative `requires`/`provides`.
- `Pipeline`: sequential named composition (composite pattern — nests freely);
  `run_with_intermediates`, `replace_stage`, name/index access.
- `SumOperator`: parallel additive composition for source-type branches;
  per-branch PRNG subkeys; leafwise pytree accumulation with loud trace-time
  errors on shape/structure mismatch and dataless branches.

### Radio (`dirt.radio`) — placeholder physics, real contracts

- Reorganized by the single-antenna element taxonomy:
  `sky/` (uniform, global signal, foregrounds, point sources),
  `environment/` (ionosphere, ground pickup, RFI),
  `instrument/` (beam, sky-side system temperature, noise-wave/reflection
  terms, CW calibration tone, bandpass, gain, thermal noise, EMI, ADC),
  `backend/` (flagging, averaging). Flat `dirt.radio` API preserved.
- Chain ordering follows the RHINO system equation
  `P_rec = g (T_ant + T_nw + T_cw) + T_n`: CW tone before bandpass/gain
  (it tracks gain drift only through the gain); sky-side temperatures before
  the reflection/noise-wave terms.
- `NoiseWaveOperator` preserves linearity in `t_nw = (T_unc, T_cos, T_sin)` —
  the `d = H t_nw` structure GCR sampling relies on.

### Inference (`dirt.inference`)

- `build_forward_fn(pipeline, state_template, filter_spec)`: the single seam
  between forward models and inference (Equinox partition/combine).
- `Likelihood` protocol + `GaussianLikelihood`; minimal working
  `GradientCalibrator`; `to_numpyro_model` stub (NumPyro optional extra).

### Project

- src layout, hatchling, uv-native; pytest with 80% coverage floor
  (currently ~97%); ruff clean; runnable end-to-end demo
  (`examples/radio_digital_twin.py`) including gradient recovery of a known
  gain.
