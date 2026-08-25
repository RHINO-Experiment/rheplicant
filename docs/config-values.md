# Values in a config document

A twin can be built in Python: you construct operators, hand them to
[the graph](operators.md), and run it. `rheplicant.config` is the layer that
lets a *document* say the same thing — and underneath all of it is the
**value grammar**: the rules by which a fragment of YAML becomes a number,
an array, or an object a field can hold.

This page is that grammar and nothing above it. What a whole document is made
of is [the anatomy](config-anatomy.md); turning a parsed one into a run is
`load_document`, and reading a whole YAML file off disk is [the command
line](config-cli.md). A single value node is small enough to exercise through
the resolver alone, which is what this does:

```bash
.venv/bin/python -c "
from rheplicant.config import ResolutionContext, resolve_value
print(resolve_value('60 MHz', ResolutionContext()))
"
```

```text
ResolvedValue(value=60000000.0, unit=Unit(canonical='Hz', factor=1000000.0,
offset=0.0, numerator=('Hz',), denominator=(), dimension='frequency'),
source='scalar', modifiers={'unit': 'MHz'})
```

(One line, wrapped here.) Three things in it are the whole design. The value came back in
**canonical units** — Hz, not MHz — because a factor of 1e6 on a frequency grid
produces a finite, correctly shaped, wrong answer and nothing downstream can
tell. The **form** that produced it is recorded, because a later step refuses
some forms in some places. And the resolver was told nothing about where the
number was going: that is a separate decision, made against the destination
field, and it is the subject of [the delivery rule](#the-delivery-rule) below.

For the signatures, see [the API reference](api.md).

## What a value node is

A value node is any one of three things:

* a **bare number** — `290`, `1.0e8`;
* a **`"<number> <unit>"` string** — `290 K`, `60 MHz`, `2.5 adc_count/K`.
  Exactly one number, whitespace, one unit token, and nothing else;
* a **mapping holding exactly one form key**, plus any modifiers —
  `{linspace: {...}, unit: MHz}`.

Everything else is refused, and refused by name. A mapping with two form keys
has no defined order and no defined result. A mapping with none is a
description of a value that does not make one. A key that is neither a form nor
a modifier nor an argument of the form present is refused rather than ignored,
because a mistyped modifier that is silently dropped makes the run differ from
the document by exactly the thing that key was there to say.

A bare YAML sequence is **not** a value node. `[1, 2, 3]` could be a `list` or a
`stack`, and the two produce different things, so the form is stated rather than
inferred from the Python type: write `{list: [1, 2, 3]}`.

The units are a closed table — `Hz`, `s`, `K`, `deg`, `m`, `ohm`,
`dimensionless`, `count`, `samples`, `bits`, `channels`, `cycles`, `adc_count`,
plus the prefixed and alternative spellings the package's own readers already
accept (`MHz`, `ms`, `celsius`, `rad`, …). A compound unit is a product, or a
quotient with at most one `/`: `adc_count/K` is a unit, `K/s*s` is how a second
denominator is spelled, and `adc_count/K × K = adc_count` is an identity a
validator can check. There are no exponents and no parentheses — `m*m` for an
area, because this is an alphabet and not an expression language.

## The eight forms

Every form key is listed in `rheplicant.config.VALUE_FORMS` — eighteen of them.
They group into the eight forms below, one per subsection, and that grouping is
what the schema numbers.

### 1. A scalar

```yaml
t_load: 290 K                       # shorthand
t_load: {value: 290, unit: K}       # longhand — identical result
```

The longhand exists for the cases the shorthand cannot spell: a value carrying
modifiers, and a value whose number is not a bare literal.

### 2. An array constructor

```yaml
gain:      {ones: [n_time, n_freq]}
freq_grid: {linspace: {start: 60, stop: 85, num: n_freq, endpoint: false},
            unit: MHz}
```

Also `zeros`, `full`, `list`, `arange`, `modulo` and `from_grid`. Every shape
position takes an integer *or* a shape symbol — `n_time`, `n_freq`, `n_source`,
`n_pix`, `n_alm`, `n_load` — optionally with an integer multiple and an integer
offset: `2 * n_freq - 1`. That is a symbol table, not an expression language.

`endpoint:` on `linspace` is required and has no default: it changes the channel
spacing, so a document that also declares `channel_width = band / n_freq` agrees
with one convention and not the other.

Where a literal integer in a shape position happens to equal one of the run's
own extents, the node records it rather than refusing — a literal `8` may
genuinely be `8`, and what it cannot be is *tied* to the grid.

### 3. A random draw

```yaml
noise: {normal: {shape: [n_time, n_freq], scale: 0.01,
                 seed: {from: runtime.seeds.receiver}}}
```

Also `uniform`. `seed:` is required and must *name* an entry of
`runtime.seeds` — never a literal. A literal seed appears in one value node and
nowhere else, so a second run of the same file cannot be shown to be the same
run.

### 4. A file reference

```yaml
bandpass: {file: {path: data/bandpass.npy, format: npy}}
```

The format is stated, never guessed from the extension: two producers of the
same extension disagree often enough that guessing is how a run reads the wrong
thing quietly. Every file reference is hashed on read, and an optional
`sha256:` is checked against the bytes actually seen. Paths expand `~` and
`${ENV}`, and are tried against the document's directory, then the declared
roots, then as written.

### 5. A reference to a named resource

```yaml
bandpass: {ref: resources.arrays.bandpass}
```

`ref` returns *the object itself*, not a copy. Named resources plus `ref` are
how this grammar composes: a DAG of named quantities, and nothing more.

### 6. A derivation

```yaml
channel_width: {from: channel_spacing}
design:        {from: basis_matrix, kind: polynomial, n_basis: 6, axis: freq}
```

A derivation names a function this package already computes — the registry is
in `rheplicant.config.DERIVATIONS` — so the document *refers* rather than
calculating. `{from: channel_spacing}` measures the median gap of the run's own
frequency grid instead of restating a division that can disagree with it.
`basis_matrix` refuses an `n:` written by hand, because a design matrix built
for the wrong number of samples returns a smooth, plausible, wrong temperature.

### 7. A stack

```yaml
gamma_src: {from_switch_order: {resource: resources.s_params, part: re}}
rows:      {stack: [{ref: resources.arrays.a}, {ref: resources.arrays.b}],
            axis: 0}
```

`stack` is a container, not a computation: its only result type is "one more
axis". `from_switch_order` is sugar for the stack that
`observation.switching.order` implies, matched **by name** rather than by
position — a transposed `gamma_src` is shape-legal and costs tens of kelvin.

### 8. The `python:` hatch

```yaml
window: {python: "numpy:hanning", args: {M: 128}}
fn:     {python: "mypkg.ops:my_fn"}          # the function itself, uncalled
```

Everything the grammar deliberately cannot do goes here. Writing `args:` or
`literal:` — either of them, even empty — calls the attribute; writing neither
delivers the attribute itself. `args:` values are resolved through this same
grammar; `literal:` values are forwarded untouched. It costs reproducibility
and it costs trust; see [what a document is trusted to
do](#what-a-document-is-trusted-to-do).

## The modifiers

Nine keys in eight rows — `scale:` and `offset:` are one affine step, applied
together. They are listed in `rheplicant.config.VALUE_MODIFIERS`.

| Modifier | Takes | Does |
| --- | --- | --- |
| `unit` | a unit token | converts to canonical units. Applied by the form, since it decides what the number *means* |
| `dtype` | `float32`, `float64`, `complex64`, `complex128` | casts. Widening a real value to complex is legal; narrowing a complex one is refused, because it deletes the phase and leaves a plausible magnitude |
| `part` | `re`, `im`, `abs`, `angle` | takes a component of a complex value. A declaration, not a cast — which is why it, and not `dtype`, is how you get a real number out of a reflection coefficient |
| `scale` / `offset` | numbers | `value * scale + offset` |
| `normalize` | `none`, `mean1`, `pixel_sum`, `max1` | states a convention the array could not have implied |
| `column` | `true` | reshapes a 1-D array to a column |
| `as` | a delivery mode | recorded and cross-checked against the destination, never applied |
| `axis` | `time`, `freq`, `none` | recorded for the noise model, never applied |

The order is fixed — `dtype` → `part` → `scale`/`offset` → `normalize` →
`column` — rather than taken from the document, so two documents with the same
keys cannot mean two different things.

## The delivery rule

Resolution knows nothing about the destination. Whether a resolved value
reaches an operator as a Python `int` or as a traced `jnp` array is not the
document's choice and not the config layer's — it is written on the target
class, in the field metadata equinox populates, and `deliver` reads the class
first.

That split is not tidiness. It is there because getting it wrong is *invisible*:

> `ADCOperator(n_bits=jnp.asarray(12))` warns `A JAX array is being set as
> static!` and then raises. `ForegroundOperator(ref_freq=jnp.asarray(1.4e8))`
> only warns: it *constructs*, the forward numbers are bit-identical, and
> `eqx.filter_grad` then returns `1.4e+08` where a gradient belongs.
> `FlaggingOperator.threshold` has no `__check_init__` at all and takes a whole
> array, detonating later at an unrelated pytree comparison.

The same measurement runs the other way. `AntennaLossOperator(efficiency=1)`
stores `int32`; an integer array is not an *inexact* array, so
`eqx.partition(op, eqx.is_inexact_array)` returns `[]` and the field is silently
untrainable. A YAML `1` and a YAML `1.0` must not differ in what can be
inferred, so a traced field always arrives floating.

Three rules follow, and all three are enforced:

* a form that produces an **array** may not land on a **static** field, and the
  refusal names the form and the field rather than waiting for equinox;
* an `as:` the document writes is checked against what the field's own metadata
  says, and a disagreement is refused — one of the two is out of date;
* `float64` is refused when `jax_enable_x64` is off in this process, because
  `astype("float64")` then silently returns `float32`, every later dtype check
  compares downcast values against each other and agrees, and the sky transforms
  carry O(10%) errors at float32. Set `runtime.jax_enable_x64: true`, or export
  `JAX_ENABLE_X64=1`, before any array exists.

## Three things this grammar will not do

**It will not do arithmetic.** There is no operator, no precedence and no
evaluation order. `30*cos(linspace(0, 3, 8))` needs an evaluator and a
namespace, which is a programming language rather than a schema. *Write
instead:* bind the quantity to a `resources.arrays` entry and read it back with
`{ref: ...}` — that gives composition of named quantities, which is what is
actually wanted most of the time — or, if it really is arithmetic, build it
behind the `python:` hatch, which states its own cost.

**It will not evaluate an expression in a shape position.** A shape takes an
integer, a symbol, an integer multiple and an integer offset, and nothing
further. *Write instead:* the integer, if it is a constant; or name the array as
a resource and take its shape from there.

**It will not convert a unit it does not know.** The table is small on purpose,
and a token outside it is refused by name rather than passed through, because a
wrong factor produces a finite, correctly shaped, wrong answer. *Write instead:*
convert the value yourself and declare the canonical unit.

The shape of all three answers is the same, and it is deliberate: the grammar
would rather be *small and refuse* than large and quietly wrong. Every refusal
names what it wanted, what it got, and what to write.

## What a document is trusted to do

A config document is executable input, and the assumption underneath this layer
is that **whoever wrote the document is whoever is running the pipeline** — in
which case it can do nothing its author could not do at a shell. Two places
depend on that and are worth knowing before you load somebody else's file.

`file:` applies no containment: `~` and `${ENV}` expand, an absolute path is
taken as written, and a relative one may climb out of the document's directory.
That is what makes `~/data/beams/...`, `${SCRATCH}/...` and `../data/...` all
work, and it means a `file:` entry naming a private key is read by this process
and its digest recorded in the resolved config.

`python:` is stronger: it runs arbitrary named code with the loading process's
privileges, and **importing alone is sufficient** — resolving
`{python: "pkg.mod:anything"}` executes `pkg/mod.py` even with no `args:` key,
so merely resolving a document to report on it has already run its code. The
resolved config records the target *string*, which says which code ran and
nothing about what that code was; there is no file hash to pin it, because
there is no single artefact to hash.

None of this is a defect to be fixed — an escape hatch that could be made safe
would not be an escape hatch. It is a reason to read the `plugins:` and
`python:` targets of a document that arrived from a shared root, a CI artefact
or a collaborator before handing it to a process that can afford what they do.
