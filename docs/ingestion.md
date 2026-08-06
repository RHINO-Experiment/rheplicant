# Ingestion: a recording on the graph

Everything else in these docs starts from a `State` that already exists. This
page is where one comes from: a RHINO spectrometer HDF5 file, and the
Touchstone `.sNp` sweeps that supply the reflection coefficients the noise-wave
model consumes. Two readers, both exported from `rheplicant.radio`, both
raising {class}`~rheplicant.core.errors.DataIngestionError` rather than
absorbing anything.

The RHINO reader needs `h5py`, which comes with the optional extra:

```bash
pip install "rheplicant[rhino]"
```

Every code block below was executed to produce the output shown beside it. The
fixture in [A file to read](#a-file-to-read) is written by the first block, so
the page runs end to end with no real recording in hand — it is the schema of
`tests/radio/test_rhino.py`'s `make_file` helper, cut down to what these
examples use.

:::{admonition} What is not on this page
:class: note
**Beam ingestion is documented where the beam is used.** `read_cst_farfield`,
`cst_frequency_table`, `cst_beam_maps` and `horizon_truncated_beam` are
pass-throughs to limTOD, so their arguments are in
[the beam-data table](operators.md#beam-data) and reading a real CST export onto
HEALPix is in [sky-to-receiver](sky-to-receiver.md) and
[sky engines](sky-engines.md). The two readers on *this* page are pass-throughs
to nothing.
:::

- [Two layers, and where the second one stops](#two-layers-and-where-the-second-one-stops)
- [A file to read](#a-file-to-read) — the HDF5 schema, and `freq_unit`
- [Onto the graph](#onto-the-graph) — `to_state`, and why `coords.time` is relative
- [`thermistor_columns` is opt-in](#thermistor_columns-is-opt-in)
- [`cal_load_operators`: the route from file to model](#cal_load_operators-the-route-from-file-to-model)
- [Touchstone sweeps](#touchstone-sweeps) — column order, `flipped=`, and interpolation

## Two layers, and where the second one stops

```{mermaid}
flowchart LR
  file[("obs.hd5f")] --> read["read_rhino_observation<br/>numpy, Hz, unix s, K"]
  read --> obs["RhinoObservation"]
  obs --> tostate["rhino_to_state"]
  obs --> calops["cal_load_operators"]
  tostate --> state["State<br/>the measurement"]
  calops --> ops["CalLoadOperator, one per load<br/>part of the model"]
```

The split is deliberate. {func}`~rheplicant.radio.rhino.read_rhino_observation`
returns a {class}`~rheplicant.radio.rhino.RhinoObservation` — plain numpy that
knows nothing about the signal graph, so a waterfall can be plotted and a
switch log inspected without constructing an operator.
{func}`~rheplicant.radio.rhino.to_state`, exported as `rhino_to_state`, is the
separate seam that places it on the graph.

**`to_state` carries five things and one number, and drops the rest.** This is
the single most useful fact about the layer:

| on the `State` | from the recording |
|---|---|
| `data` | `waterfall` |
| `coords.time` | `time_s`, **relative** — see [below](#coordstime-is-relative) |
| `coords.freq` | `freq_hz` |
| `coords.extra["receiver_input"]` | `switch_label`, as an integer index |
| `aux["flags"]` | `settled`, **inverted** and broadcast |
| `meta["time_epoch_unix_s"]` | the unix second `coords.time` is measured from |

Everything else on a `RhinoObservation` — `thermistor_k`, `transitions`,
`n_leading_dropped`, `adc_max_i`, `adc_max_q` — is **diagnostic**. Nothing in
`rheplicant` consumes it, and a caller who wants it reads it off the recording
directly. The one exception is `thermistor_k`, which reaches the model by a
route of its own: {func}`~rheplicant.radio.rhino.cal_load_operators`, argued
[below](#cal_load_operators-the-route-from-file-to-model).

## A file to read

```python
import h5py
import numpy as np

T0 = 1.75e9                                   # a plausible unix second
times = T0 + np.arange(14) * 10.0             # 14 samples, 10 s apart
switch_times = T0 + np.array([20.0, 60.0, 100.0])
states = np.array([b"antenna", b"internal_load", b"heated_load"], dtype="S16")
temps = np.stack([np.full(14, 20.0),                   # column 0: ambient [C]
                  100.0 + 0.1 * np.arange(14)], 1)     # column 1: heated, drifting

with h5py.File("obs.hd5f", "w") as f:
    sdr = f.create_group("sdr")
    sdr.create_dataset("sdr_freqs", data=np.array([60.0, 70.0, 80.0]))    # MHz
    sdr.create_dataset("sdr_times", data=times)
    sdr.create_dataset("sdr_waterfall", data=np.arange(42.0).reshape(14, 3))
    sw = f.create_group("switches")
    sw.create_dataset("switch_times", data=switch_times)
    sw.create_dataset("switch_states", data=states)
    tg = f.create_group("temperatures")
    tg.create_dataset("temperature_times", data=times)
    tg.create_dataset("temperatures", data=temps)
```

The switch log starts 20 s after the SDR does, so the first two samples have no
defined switch state. That is not contrived — it is the ordinary case, and it
is why the reader has a leading drop at all.

```python
from rheplicant.radio import read_rhino_observation

obs = read_rhino_observation("obs.hd5f", freq_unit="MHz")
print(obs.waterfall.shape, obs.freq_hz)
print(obs.n_leading_dropped)
print(obs.switch_label)
print(obs.settled)
print(obs.thermistor_k)
```

```
(12, 3) [60000000. 70000000. 80000000.]
2
['antenna' 'antenna' 'antenna' 'antenna' 'internal_load' 'internal_load'
 'internal_load' 'internal_load' 'heated_load' 'heated_load' 'heated_load'
 'heated_load']
[False  True  True  True False  True  True  True False  True  True  True]
{}
```

Fourteen samples in, twelve out: the two that preceded the first transition are
gone, and `n_leading_dropped` says so. Every per-sample array is cut by the same
mask, so `switch_label[i]` always describes the sample in `waterfall[i]`.

`settled` is `False` for the first sample of each block because
`settle_seconds` defaults to 5 s and the samples are 10 s apart, so only the
sample sitting exactly on a transition is inside the window. **Note the
polarity: `settled` is True-means-usable**, the opposite of `aux["flags"]`.

### `freq_unit` has no default, on purpose

The file does not record its own frequency unit and its two producers disagree:
`rhino-cal`'s `ObservationHandler.save_to_hdf5` writes Hz, the
`RHINO_fully_simulated_calibration` notebook writes MHz. The reference reader
defaults to MHz — wrong for its own simulator's output, and silent about it,
because the consequence is a reflection coefficient interpolated onto a band
10⁶ away, which then *clamps to constant edge values* rather than raising.

So `freq_unit` is required, and the declaration is checked against the file's
own values:

```python
read_rhino_observation("obs.hd5f", freq_unit="Hz")
```

```
DataIngestionError: declared freq_unit='Hz', which puts this file's channels at
[60, 80] Hz -- outside the plausible range [1e+06, 1e+10] Hz. The file's raw
values span [60, 80]; the other unit is likely right.
```

The plausible band is wide on purpose. Its job is to catch a 10⁶ unit error,
not to police which telescope wrote the file.

## Onto the graph

```python
from rheplicant.radio import rhino_to_state
from rheplicant.radio.rhino import TIME_EPOCH_META_KEY

state = rhino_to_state(obs, source_order=("antenna", "internal_load", "heated_load"))
print(state.data.shape, sorted(state.aux), sorted(state.meta))
print(state.coords.time)
print(state.coords.extra["receiver_input"])
print(state.aux["flags"][:, 0])
```

```
(12, 3) ['flags'] ['time_epoch_unix_s']
[  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100. 110.]
[0 0 0 0 1 1 1 1 2 2 2 2]
[ True False False False  True False False False  True False False False]
```

Three things happened there, and two of them are easy to get backwards.

**`source_order` is the graph's in-edge order, not yours.** It maps switch
labels onto the integer indices `SwitchCycle` and the `receiver_input` selector
use, and it is the same order `NoiseWaveOperator`'s `gamma_src` rows must match.
A transposition there is shape-legal and costs tens of kelvin. Read it off the
assembled twin rather than assuming it —
[below](#the-whole-route-and-where-source_order-comes-from) does exactly that. A
label the recording switches to that `source_order` does not name is refused
here rather than deferred:

```python
rhino_to_state(obs, source_order=("antenna", "internal_load"))
```

```
DataIngestionError: the recording switches to ['heated_load'], which
source_order does not name (it lists ['antenna', 'internal_load']). Deferring
this makes SwitchCycle.gather return NaN much later, where the cause is no
longer visible.
```

**The settling mask is inverted on the way in.** `aux["flags"]` is
True-means-*flagged* — that is what `radio/backend/flagging.py` produces and
what `FlaggedNoise` consumes — while `settled` is True-means-*usable*. Compare
the two arrays printed above: they are complements. Getting this backwards
yields a finite, correctly-shaped result that discards every good sample and
keeps every transient, and nothing about the shape or dtype would reveal it.

**`flags` is broadcast to `(n_time, n_freq)`.** Settling is inherently
per-time — every channel of an unsettled sample is unsettled — so the broadcast
changes nothing about what is being said. The shape is not `to_state`'s choice;
it is set by every consumer (`FlaggedNoise.std` raises if `flags` disagrees in
shape with the prediction it masks, `SkySpaceFilter` multiplies `1 - flags`
elementwise against the data, and both flagging operators produce and expect
2-D). `obs.settled` stays `(n_time,)` for a caller who wants the per-time form.

### `coords.time` is relative

```python
print(state.meta[TIME_EPOCH_META_KEY], float(obs.time_s[0]), obs.n_leading_dropped)
recovered = state.meta[TIME_EPOCH_META_KEY] + np.asarray(state.coords.time, np.float64)
print(np.array_equal(recovered, obs.time_s), state.coords.time.dtype)
```

```
1750000020.0 1750000020.0 2
True float32
```

`coords.time` is **seconds since the first kept sample**, and
`meta["time_epoch_unix_s"]` holds the unix second it is measured from, so
`meta[key] + coords.time` recovers `obs.time_s` exactly. The epoch is the first
*kept* sample, not the first sample in the file — 1750000020, twenty seconds
after this recording's `T0`, because the leading drop removed two samples that
were never part of the run the `State` describes.

This is a behaviour change from an earlier version that stored the raw unix
axis, and it is a fix rather than a convenience.
{class}`~rheplicant.core.coordinates.Coordinates` stores its axes through
`jnp.asarray`, which is float32 unless x64 is enabled, and a unix second near
1.75e9 has a float32 resolution of 128 s. Hand six samples over absolute:

```python
import jax.numpy as jnp

offsets = np.array([0.0, 100.0, 250.0, 450.0, 700.0, 1000.0])
stored = np.asarray(jnp.asarray(1.75e9 + offsets), dtype=np.float64) - 1.75e9
print("stored offsets", stored)
print("error [s]     ", stored - offsets)
print("all distinct  ", len(set(stored.tolist())) == len(stored))
print("relative      ", np.asarray(jnp.asarray(offsets), dtype=np.float64) - offsets)
```

```
stored offsets [   0.  128.  256.  512.  640. 1024.]
error [s]      [  0.  28.   6.  62. -60.  24.]
all distinct   True
relative       [0. 0. 0. 0. 0. 0.]
```

Read the third line. **All six values stay distinct**, so no shape, count,
dtype or finiteness check can see the corruption — while the axis is wrong by
up to a minute. Subtracting the epoch before the store removes the cause
outright: the offsets become small integers, which float32 holds exactly.

That is not a hypothetical cost. The two places in the package that do
*arithmetic* on `coords.time` rather than reading its length both read the
rounded values. On eight samples 100 s apart at unix magnitude, averaged in
chunks of two the way `BackendOperator` does:

```python
t = 1.75e9 + np.arange(8) * 100.0
print("distinct stored", len(set(np.asarray(jnp.asarray(t), np.float64).tolist())), "of 8")
chunk = np.asarray(jnp.asarray(t).reshape(4, 2).mean(1), np.float64)
print("error [s]      ", chunk - t.reshape(4, 2).mean(1))
```

```
distinct stored 6 of 8
error [s]       [ 78.   6. -66. -10.]
```

Two of the eight samples had already merged *before* the average ran, and every
chunk timestamp comes out wrong — by 78 s out of a 100 s cadence at worst.
Nothing raised, nothing was NaN, every shape was right. The same axis sends a
drifting `CWCalibrationOperator` tone into the wrong channel.

Detecting this after the fact is impossible from the stored values alone, so
`Coordinates.__check_init__` now refuses such an axis outright rather than
repairing it:

```python
from rheplicant.core.coordinates import Coordinates

Coordinates(time=jnp.asarray(1.75e9 + np.arange(8) * 100.0), freq=jnp.asarray([60e6]))
```

```
StateValidationError: coords.time is stored as float32 and reaches
1.75000064e+09, where consecutive representable numbers are 128 apart — but the
closest two distinct samples on this axis are 128 apart, and coords.time must
resolve its own sampling to at most 0.01 of that. The rounding happens when the
axis is STORED, so no later subtraction recovers it: samples merge, and every
consumer that does arithmetic on the values then reads the rounded ones — ...
```

The ingestion path no longer trips it, because it no longer produces that axis.
For what the guard means for a *hand-built* axis — including that it is
unit-agnostic, so MJD is not exempt, and roughly how many samples a relative
float32 axis carries before you need `JAX_ENABLE_X64=1` — see
[the operator guide](operators.md#coordstime-is-checked-where-it-is-stored-and-twice), which
also states the tone's own, stricter check.

## `thermistor_columns` is opt-in

Pass the map and the temperature log is read, validated and interpolated onto
the SDR axis. Omit it and **the log is not read at all**:

```python
COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}
obs_t = read_rhino_observation("obs.hd5f", freq_unit="MHz", thermistor_columns=COLUMNS)
print(sorted(obs_t.thermistor_k))
print(obs_t.thermistor_k["antenna"][:3])
print(obs_t.thermistor_k["heated_load"][[0, 6, 11]])
print(np.array_equal(obs_t.thermistor_k["antenna"], obs_t.thermistor_k["internal_load"]))
```

```
['antenna', 'heated_load', 'internal_load']
[293.15 293.15 293.15]
[373.35 373.95 374.45]
True
```

Celsius in the file, Kelvin out. Two labels may share a column and then hold
equal arrays, which is what `antenna` and `internal_load` do here.

The map is required rather than defaulted for the same reason `freq_unit` is.
The reference reader takes `heated_load_index=1, ambient_load_index=0` and
routes every other state to the ambient column; those indices are the
positional order of `save_to_hdf5`'s `save_temps` argument — a convention shared
between writer and reader with **nothing in the file to enforce it**. A file
written with a different order reads back with hot and ambient swapped and
nothing raises. Requiring the map makes that a declaration.

`thermistor_k` is keyed by the labels this file's switch log actually visited,
not by every key in the map. A label the map declares but the recording never
switched to simply has no entry, and `thermistor_k[label]` raises `KeyError`
rather than returning `None` or an empty array.

### Why opt-in, and not simply always on

Because the checks are strict, correct, and defend something that does not
reach the signal path. Two of them refuse a whole file:

```python
import shutil

shutil.copy("obs.hd5f", "short.hd5f")
with h5py.File("short.hd5f", "r+") as f:        # log stops one millisecond early
    del f["temperatures/temperature_times"]
    f["temperatures"].create_dataset("temperature_times", data=times - 0.001)

print(read_rhino_observation("short.hd5f", freq_unit="MHz").waterfall.shape)
read_rhino_observation("short.hd5f", freq_unit="MHz", thermistor_columns=COLUMNS)
```

```
(12, 3)
DataIngestionError: thermistor column 0 for 'antenna': the target range is
outside the sampled range [1.75e+09, 1.75e+09] by 0.000999928 above the high
end. np.interp would clamp to the edge values and report nothing. The fix is
usually to trim the target axis to the sampled coverage, or to extend what was
measured -- a wider sweep, a longer log -- to cover it; pass
allow_extrapolation=True only when clamping to the edge value is actually what
you want.
```

(The message states the *overshoot*, not the four raw bounds, and the reason is
visible in the output above: at unix-epoch magnitude `%.6g` renders `1.75e+09`
for both ends of the sampled range. Six significant figures cannot resolve a
millisecond against a 1.75e9 base. The gap is a small number on its own scale
whatever the axis's magnitude.)

The other refusal is a non-finite reading in a used column, which raises rather
than propagating: a linear interpolant spreads one bad row into every sample
whose bracketing interval touches it — a dropout wider than the dropout itself
— and by the time that value reaches `T_sys` and the noise-wave solve there is
nothing left pointing back at the thermistor log it came from.

Both refusals are right *for a caller who wants the temperatures*. Running them
unconditionally made a whole recording unreadable over a column nothing
downstream consumes — note the first line of that output: the waterfall, the
switch log and the settling mask are all intact in that file, and they are what
a forward model needs. So the map is opt-in, and when it is given, every check
runs unchanged.

Nothing under `/temperatures` is read at all when the map is omitted, so a file
with no such group — or a malformed one — is still readable for its waterfall.

## `cal_load_operators`: the route from file to model

`to_state` does not carry the load temperatures, so for a while they were
parsed, validated, and then discarded: the warm/hot-load noise-wave path had no
way to reach a model from a recording.
{func}`~rheplicant.radio.rhino.cal_load_operators` is that route.

```python
from rheplicant.radio import cal_load_operators

loads = cal_load_operators(obs_t, labels=("internal_load", "heated_load"))
print(list(loads))
print(loads["heated_load"].t_load.shape)
print(np.asarray(loads["heated_load"].t_load[:, 0])[[0, 6, 11]])
```

```
['internal_load', 'heated_load']
(12, 1)
[373.35 373.95 374.45]
```

**It is a separate function rather than part of `to_state`, and the reason is a
type rather than a preference.** `to_state` returns a `State`. Wiring operators
out of it would either change that return type — collapsing the two-layer split
this module exists to keep — or move the load temperature *into* the `State` for
the operator to read. The second is what a reader reaches for first, since it
makes the dependency explicit in `CalLoadOperator.requires`, and it is the wrong
move: a load's physical temperature is a property of the *instrument's*
configuration, not of the observation. It belongs to one load among several, and
a `State` has no place to key it by load without inventing one. Worse, it would
make the operator's declared requirement unsatisfiable for any two-load model
built from this reader, since both loads would read the same path. A separate
function costs the caller one line and changes neither.

**The temperature is an explicit `(n_time, 1)` column, never a bare
`(n_time,)`.** A load's physical temperature drifts through a run, so it *is*
per-sample — but
{class}`~rheplicant.radio.instrument.calibration.CalLoadOperator` always reads a
bare 1-D array as per-*frequency*, matching `NoiseWaveOperator`'s temperature
leaves. Spelling the column is what keeps this function correct on a square
grid, where NumPy would otherwise settle the ambiguity silently by aligning
trailing axes.

Asking for load operators from a recording read without the map raises, rather
than handing back an empty mapping that would build a model with no loads and
no warning:

```python
cal_load_operators(obs)          # obs was read without thermistor_columns
```

```
DataIngestionError: This observation carries no thermistor temperatures, so
there is nothing to build a load operator from. Pass `thermistor_columns=` to
read_rhino_observation; without it the temperature log is not read at all (see
read_rhino_observation).
```

### The whole route, and where `source_order` comes from

`labels=` is how a caller pins the *order* the loads are built in, and that
order is what pairs with the graph's branch order. Here is the join, run:

```python
from rheplicant.radio import ForegroundOperator, GainOperator, assemble

twin = assemble(
    ForegroundOperator(amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5),
                       ref_freq=70e6),
    *loads.values(),
    GainOperator(gain=jnp.array(1.1)),
)
print(twin["receiver_input"].names)
print(("antenna",) + tuple(loads))
```

```
('foregrounds', 'cal_loads_1', 'cal_loads_2')
('antenna', 'internal_load', 'heated_load')
```

Those two tuples are the same ordering, expressed twice: the graph names its
branches, and you name the switch labels that occupy them. The pairing is
**positional**, and the second tuple is exactly what `source_order` wants. Read
the first off the assembly rather than assuming it — that is the one line that
keeps the switch index, the selector, and `gamma_src`'s rows from disagreeing.

The `State` from `rhino_to_state` holds the *measurement*. A twin containing
source operators generates its own data, so it refuses to be handed one:

```python
twin(state)
```

```
AssemblyError: This assembly contains source operators and generates its own
data; caller-supplied state.data would be discarded. Pass a state with
data=None (or drop the sources to build a transform chain).
```

Strip `data` and the twin runs on the recording's axes, switch cycle and flags,
producing the prediction to compare the measurement against:

```python
prediction = twin(state.replace(data=None))
print(prediction.data[0], prediction.data[4])
print(prediction.data[8], prediction.data[11])
```

```
[1617.1848 1100.      787.7944] [322.465 322.465 322.465]
[411.565 411.565 411.565] [411.89502 411.89502 411.89502]
```

Row 0 is an antenna sample and carries a spectrum; rows 4, 8 and 11 are load
samples and are flat in frequency, as a load is. Rows 8 and 11 are both
`heated_load` and they **differ** — 411.565 against 411.895 — because the
recorded temperature drifted 0.3 K between them and the `(n_time, 1)` column
carried it through. That difference is the whole point of the route: with a
scalar `t_load` it would be zero.

## Touchstone sweeps

{func}`~rheplicant.radio.touchstone.read_touchstone` turns a Touchstone v1
`.s1p` or `.s2p` file into a {class}`~rheplicant.radio.touchstone.Touchstone` —
complex S-parameters, frequencies in Hz. Like `rheplicant.radio.beams`, it is a
thin adapter and adds nothing else: what an S-parameter *means*, how a
reflection coefficient becomes a coupling spectrum, is `rhino_cal_jax`'s
subject and [sky-to-receiver](sky-to-receiver.md)'s.

It is ported from `rhino-cal`'s `read_s2p` with that function's silent failure
modes turned into errors. The one that matters most: `read_s2p` skips any data
row whose column count is not exactly nine, so a trailing `!` comment or a
truncated line removes a frequency point without a word — and the caller gets a
shorter sweep that still interpolates cleanly.

Two files, so the blocks below run as written — a 1-port horn sweep in
magnitude/angle, and a 2-port in real/imaginary:

```python
open("antenna.s1p", "w").write(
    "! horn return loss, magnitude/angle\n"
    "# MHZ S MA R 50\n"
    "50.0   0.30  -20.0\n"
    "75.0   0.22  -95.0\n"
    "100.0  0.35  170.0\n"
)
open("lna.s2p", "w").write(
    "! freq  S11        S21        S12        S22\n"
    "# MHZ S RI R 50\n"
    "50.0    0.10 0.20  0.30 0.40  0.50 0.60  0.70 0.80\n"
    "100.0   0.11 0.21  0.31 0.41  0.51 0.61  0.71 0.81\n"
)
```

```python
from rheplicant.radio import interpolate_onto, read_touchstone

antenna = read_touchstone("antenna.s1p")
print(antenna.n_port, antenna.z0, antenna.s.shape)
print(antenna.freq_hz)
print(antenna.s11)
```

```
1 50.0 (3, 1, 1)
[5.0e+07 7.5e+07 1.0e+08]
[ 0.28190779-0.10260604j -0.01917426-0.21916283j -0.34468271+0.06077686j]
```

`s` is always `(n, p, p)`; the four accessors `s11`/`s12`/`s21`/`s22` index into
it, and asking a 1-port file for one it does not have raises rather than
returning a zero that would read as a perfectly isolated port:

```python
antenna.s21
```

```
DataIngestionError: s21 was requested from a 1-port file. A 1-port measurement
carries only s11; there is no transmission term to return and a zero would read
as a perfectly isolated port.
```

**No unstated frequency unit.** Touchstone v1 defaults an omitted unit to GHz;
this reader raises instead of applying a 10⁹ rescaling nobody wrote down. The
stakes are higher here than for the HDF5 reader's `freq_unit` — 10⁹ rather than
10⁶ — and VNA exports that omit the token are rare enough that raising costs
less than a silently rescaled calibration sweep would:

```python
open("nounit.s1p", "w").write("# S RI R 50\n50.0 0.1 0.2\n")
read_touchstone("nounit.s1p")
```

```
DataIngestionError: nounit.s1p:1: the option line names no frequency unit.
Touchstone v1 defaults an omitted unit to GHz, but this reader will not apply an
unstated 10⁹ rescaling -- name one of ('HZ', 'KHZ', 'MHZ', 'GHZ') explicitly.
Line: '# S RI R 50'
```

An implicit **50 ohm** reference impedance, by contrast, *is* applied when the
option line has no `R` clause. That one is the near-universal case rather than a
plausible order-of-magnitude error, which is the whole distinction.

### Column order, and `flipped=`

A Touchstone 2-port data row is `freq S11 S21 S12 S22`. **The second pair is
S21, not S12.** This is the single most likely thing to get wrong here, because
every other 2×2 convention in this package is row-major, and because a test that
only checks `s11` cannot see the error:

```python
lna = read_touchstone("lna.s2p")
print(lna.s21, lna.s12)
print(lna.s[0])
```

```
[0.3 +0.4j  0.31+0.41j] [0.5 +0.6j  0.51+0.61j]
[[0.1+0.2j 0.5+0.6j]
 [0.3+0.4j 0.7+0.8j]]
```

`flipped=True` swaps the two ports. Set it when the device under test was wired
to the VNA with this codebase's port 1 and port 2 reversed. It is a genuine
reversal of the parsed matrix — `s[:, ::-1, ::-1]` — not a relabelling of the
return values, so `Touchstone.s` itself and all four accessors agree:

```python
flipped = read_touchstone("lna.s2p", flipped=True)
print(flipped.s[0])
print(np.array_equal(flipped.s, lna.s[:, ::-1, ::-1]))
```

```
[[0.7+0.8j 0.3+0.4j]
 [0.5+0.6j 0.1+0.2j]]
True
```

Reading `s22` at the call site and *calling* it `s11` would give the same four
scalars while leaving `ts.s` un-reversed and disagreeing with them. On a 1-port
file `flipped=True` raises: port reversal exchanges two ports, and there is only
one.

The filename is cross-checked against the port count the rows carry, because
nothing inside a Touchstone file states its own port count independently of the
column count — the suffix is the only second opinion available, and a
mislabelled file would otherwise put a transmission term where a reflection
coefficient belongs. A file with no recognised suffix has no second opinion and
passes through untouched.

### `interpolate_onto`, and why it refuses to extrapolate

A calibration sweep is measured on the VNA's grid; the model needs it on
`coords.freq`. {func}`~rheplicant.radio.touchstone.interpolate_onto` does that
for one named component:

```python
band = np.linspace(60e6, 90e6, 4)
print(interpolate_onto(band, antenna, component="s11"))
```

```
[ 0.16147497-0.14922876j  0.04104215-0.19585148j -0.08427595-0.16317489j
 -0.21447933-0.05119902j]
```

Complex in, complex out, with real and imaginary parts interpolated
*separately* — not magnitude and phase. Across a mismatch resonance the phase
wraps, and interpolating a wrapped angle is worse than interpolating Cartesian
components, not better.

**Extrapolation is refused by default**, and this is the guard the whole module
is built around. `np.interp` *clamps* outside its sampled range rather than
raising, so a reflection coefficient measured over a narrower band than the one
being observed comes back as a constant at the edges with no diagnostic — the
reference implementation has exactly that behaviour:

```python
interpolate_onto(np.linspace(40e6, 90e6, 4), antenna)     # sweep starts at 50 MHz
```

```
DataIngestionError: s11 interpolation: the target range is outside the sampled
range [5e+07, 1e+08] by 1e+07 below the low end. np.interp would clamp to the
edge values and report nothing. The fix is usually to trim the target axis to
the sampled coverage, or to extend what was measured -- a wider sweep, a longer
log -- to cover it; pass allow_extrapolation=True only when clamping to the edge
value is actually what you want.
```

`allow_extrapolation=True` is the escape hatch, and it does what the message
says — clamps, rather than extending the trend:

```python
print(interpolate_onto(np.array([40e6]), antenna, allow_extrapolation=True), antenna.s11[0])
```

```
[0.28190779-0.10260604j] (0.2819077862357725-0.10260604299770061j)
```

Pass it only when the edge value is genuinely what you want. The tolerance band
around "outside the sampled range" is a relative term on the span plus a few
ULPs at the axis's own magnitude — deliberately tiny, sized to swallow a unit
conversion's rounding and nothing more. The same helper serves the thermistor
log, which is why a temperature series stopping a millisecond short of the SDR
axis refuses [above](#why-opt-in-and-not-simply-always-on).
