# RHINO data ingestion: Touchstone and observation HDF5 — Design

**Goal:** give rheplicant the two readers it needs to consume real RHINO
recordings — Touchstone `.s1p`/`.s2p` files (the measured reflection
coefficients) and the RHINO observation HDF5 (the power waterfall, the switch
log and the thermistor log) — as thin adapters that produce numpy, plus one
explicit seam that places the result on the signal graph.

**Non-goals, deliberately.** The Dicke `q`-ratio data reduction
`(P_src − P_l)/(P_ns − P_l)`, its error propagation, the `(ν, t)` basis
expansion and the `T_ant` back-solve are all the *next* step. This spec covers
ingestion only. Nothing here computes a temperature from a power.

---

## Prior context an implementer needs

### Repositories

- **rheplicant** — `/Users/zzhang/projects/e-RHINO`, package `src/rheplicant`.
  All work happens here, on branch `feat/rhino-ingestion`.
- **Reference implementation** — `/Users/zzhang/projects/rhino-cal`
  (`git@github.com:RHINO-Experiment/rhino-cal.git`). The code being ported:
  - `utils/utils.py:8` `read_s2p`
  - `utils/utils.py:99` `write_s2p` (**not** ported — see "What is not ported")
  - `utils/utils.py:196` `interp_vals_to_new_freq`
  - `gcr/data_processing.py:19` `DataHandler.__init__` — the reference *reader*
  - `simulation/observation_handler.py:188` `save_to_hdf5` — the reference *writer*
- **`Rashi-Srivastava/RHINO-CAL`** (private, notebooks only, no package) — the
  authority on what real files look like. `RHINO_fully_simulated_calibration.ipynb`
  §8 writes the schema from scratch and is the clearest statement of it;
  `NoiseWaveSolver.ipynb` and `RFI.ipynb` read genuine recordings from
  `../data/long_data/*_obs.hd5f` and `../data/16DecHotLoad/*_obs.hd5f`.

### The HDF5 schema, as both producers write it

```
/sdr/sdr_freqs          (n_freq,)             float
/sdr/sdr_times          (n_time,)             float, unix seconds
/sdr/sdr_waterfall      (n_time, n_freq)      float, raw power (arbitrary scale)
/sdr/max_i_adc          (n_time,)             float   — notebook only
/sdr/max_q_adc          (n_time,)             float   — notebook only
/switches/switch_times  (n_switch,)           float, unix seconds
/switches/switch_states (n_switch,)           bytes (`S` dtype), e.g. b"open"
/temperatures/temperatures       (n_temp_time, n_column)  float, CELSIUS
/temperatures/temperature_times  (n_temp_time,)           float, unix seconds
/aux_sdr/...            — written empty, see below
/obs_config             — written empty
/simulation_truth/...   — notebook-generated files only
```

`switch_times` / `switch_states` are a **transition log**, not a per-sample
array: the state named at `switch_times[i]` holds until `switch_times[i+1]`.

### Five defects in the reference path that this design exists to fix

These are established by reading the code, not conjecture. Each becomes a test.

**1. The frequency unit disagrees between the two producers, and the file does
not record which.** `ObservationHandler.save_to_hdf5` writes
`np.array(self.freqs)` where `self.freqs` is an astropy `Quantity` in **Hz**;
`RHINO_fully_simulated_calibration.ipynb` §8 writes `freqs_MHz_truth`, in
**MHz**. The reference reader `DataHandler.__init__` defaults to
`freq_unit=un.MHz` — so its default is wrong for its own simulator's output,
and nothing raises. The consequence is Γ interpolated onto a band 10⁶ off,
which then silently extrapolates (defect 2) into constant edge values.

**2. Interpolation extrapolates silently.** `interp_vals_to_new_freq` uses
`np.interp`, which *clamps* outside the sample range rather than raising. If a
`.s2p` sweep does not cover the SDR band you get a constant Γ at the edges and
no diagnostic. The notebook knows this — §3 states it "refuses to extrapolate"
and intersects every band first — but the packaged path does not.

**3. The thermistor column mapping is a magic index paired with an implicit
rule.** `DataHandler.__init__` takes `heated_load_index=1, ambient_load_index=0`
and assigns *every* switch state except `heated_load` to the ambient column.
Those indices are the positional order of `save_to_hdf5`'s
`save_temps=['internal_load', 'heated_load']` argument — a convention shared
between writer and reader with **nothing in the file to enforce it**. A file
written with a different `save_temps` order reads back with hot and ambient
swapped, silently.

**4. `temperature_unit` is accepted and never used.** `DataHandler` takes
`temperature_unit=un.Celsius`, then hard-codes `self.temperatures + 273.15`
(the `.to(un.K, equivalencies=...)` line is commented out). A caller who passes
`un.K` gets Kelvin-plus-273.15 and no error.

**5. `read_s2p` drops malformed rows silently.** `if len(values) != 9: continue`
means a trailing `!` comment, a continuation line, or a truncated row removes
that frequency point without a word. The function also builds a full
`s_params[:, 2, 2]` array and then never returns it — dead code that happens to
be the right data structure.

Two further notes that are not defects but constrain the design:

- `save_to_hdf5` creates `/aux_sdr/aux_sdr_waterfall` with `dtype="f"` and
  **no `data=` and no `shape=`**, which makes a *scalar* dataset, not an array.
  Any reader that assumes `/aux_sdr` holds a waterfall will break on real
  files. This design ignores `/aux_sdr` and `/obs_config` entirely.
- The settling time after a switch transition is inconsistent across the
  reference: 5 s in the notebook, `switch_buffer=2*un.s` in
  `recover_source_temperatures`, `1*un.s` in `produce_nw_fitting_data`. This
  design takes 5 s as the default — the most conservative of the three — and
  makes it a parameter.

### Two rheplicant conventions this must not get wrong

- **`aux["flags"]` is True-means-bad.** `radio/backend/flagging.py:62` states it
  and `FlaggedNoise` consumes it that way. The natural name for a settling mask
  (`settled`) is True-means-good, so `to_state` inverts. Getting this backwards
  produces a finite, correctly-shaped result that throws away every good sample
  and keeps every transient.
- **`gamma_src`'s row order is the graph's in-edge order**, readable off
  `assembly["receiver_input"].names` — see `radio/instrument/noise_wave.py`'s
  module docstring. `to_state` takes that order as `source_order` and is the
  place where a label-to-index mismatch must be caught.

---

## Decisions taken

| Question | Decision |
|---|---|
| Reader output shape | Two layers: a neutral numpy dataclass, plus a separate explicit `to_state()`. Reading a file and placing it on the signal graph are different seams. |
| Frequency unit | **Required argument, no default.** The caller must declare it. A self-consistency check then validates the declaration against the file's actual values. |
| Scope | Read + per-sample switch index + thermistor mapping + settled mask. No block aggregation, no `q` ratio, no error propagation. |
| Placement | Flat modules under `radio/`, alongside `beams.py`. |
| Units at the seam | Bare floats: Hz, unix seconds, Kelvin. **No astropy.** |

**Why flat rather than a `radio/io/` subpackage.** Every subpackage under
`radio/` (`backend/ environment/ filters/ instrument/ sky/`) is a group of
*operators*; every flat module (`beams.py graph.py surrogate.py`) is not. These
two are not operators. `beams.py` is also already a file→numpy adapter, which
makes it the governing precedent. Grouping by `io/` would also be organising by
type rather than by domain, which the project's style rules argue against —
S-parameters and observation recordings are different domains that happen to
share a direction of data flow.

**Why no astropy.** The core package depends on `jax` and `equinox` only, and
`beams.py` already establishes bare-Hz at the boundary. It also sidesteps the
unsimplified-unit trap documented in the rhino-cal README, where dividing a
Hz-valued `Quantity` by `210 * un.MHz` yields a dimensionless number wrong by
`(10⁶)^2.6 ≈ 4 × 10¹⁵` with no error raised.

---

## Component 1 — `src/rheplicant/radio/touchstone.py`

```python
@dataclass(frozen=True)
class Touchstone:
    freq_hz: np.ndarray   # (n,) strictly ascending, Hz
    s: np.ndarray         # (n, p, p) complex, p in {1, 2}
    z0: float             # reference impedance from the option line
    # .s11 always; .s12 / .s21 / .s22 raise DataIngestionError on a 1-port file

def read_touchstone(path, *, flipped: bool = False) -> Touchstone

def interpolate_onto(
    freq_hz: np.ndarray,
    source: Touchstone,
    *,
    component: str = "s11",          # "s11" | "s12" | "s21" | "s22"
    allow_extrapolation: bool = False,
) -> np.ndarray

def _interp_strict(x_new, x, y, *, allow_extrapolation, what: str) -> np.ndarray
```

`_interp_strict` is the shared guard: linear interpolation that raises unless
`x_new` lies inside `x`'s range, with `what` naming the quantity in the error.
`interpolate_onto` is a thin wrapper over it, and `rhino.py` imports it for the
thermistor axis, so the two readers cannot drift apart on extrapolation policy.
`component` outside the four names raises, as does a 2-port name against a
1-port file.

### Changes relative to `read_s2p`

| # | Reference behaviour | This design |
|---|---|---|
| 1 | 2-port only | `.s1p` and `.s2p`. Port count inferred from the data column count and cross-checked against the suffix; disagreement raises. |
| 2 | `len(values) != 9: continue` — silent drop | Column count mismatch raises, quoting the line number and its text. |
| 3 | A trailing `!` comment loses the whole row | `!` and everything after it is stripped before parsing. |
| 4 | Reference impedance not parsed | `R <z0>` parsed off the option line; defaults to 50.0 when absent, as Touchstone v1 specifies. |
| 5 | Missing option line surfaces only at the first data row | Checked before the data section; the error quotes the file's first data line. |
| 6 | Frequency monotonicity unchecked | Strictly-ascending enforced — it is `np.interp`'s precondition — otherwise raise. |
| 7 | `s_params` built, never returned | It *is* the return value; the named accessors are views onto it. |
| 8 | `flipped` implemented by permuting the return tuple | Implemented as a genuine port reversal, `s[:, ::-1, ::-1]`, so the meaning does not depend on the caller remembering positions. |
| 9 | No extrapolation guard | `interpolate_onto` refuses extrapolation by default; the error names the target band, the file's band, and which end is short. |

`interpolate_onto` interpolates real and imaginary parts separately with
`np.interp` (numpy only, no scipy). The docstring must record why not
magnitude/phase: across a mismatch resonance the phase wraps, and interpolating
a wrapped angle is worse than interpolating the Cartesian components, not
better. The notebooks use PCHIP; on a VNA sweep far denser than the SDR channel
grid — which is the operating regime — the difference is below the level of
anything downstream. If that stops being true, the fix is a `method=` argument
and a scipy import gated like `beams.py` gates limTOD, not a silent upgrade.

`MA` and `DB` formats convert as `mag · exp(i·angle)` and
`10^(dB/20) · exp(i·angle)` respectively, angles in degrees — matching the
reference, which is correct.

## Component 2 — `src/rheplicant/radio/rhino.py`

```python
@dataclass(frozen=True)
class RhinoObservation:
    freq_hz: np.ndarray                  # (n_freq,)
    time_s: np.ndarray                   # (n_time,) unix seconds
    waterfall: np.ndarray                # (n_time, n_freq) raw power
    switch_label: np.ndarray             # (n_time,) str, per-sample
    settled: np.ndarray                  # (n_time,) bool, True = usable
    thermistor_k: dict[str, np.ndarray]  # label -> (n_time,) K, on time_s
    transitions: tuple[np.ndarray, np.ndarray]   # raw (times, labels) log
    n_leading_dropped: int               # samples before the first transition
    adc_max_i: np.ndarray | None         # (n_time,) if present in the file
    adc_max_q: np.ndarray | None

def read_rhino_observation(
    path,
    *,
    freq_unit: str,                        # "Hz" | "MHz" — REQUIRED
    thermistor_columns: Mapping[str, int], # REQUIRED
    settle_seconds: float = 5.0,
    thermistor_unit: str = "celsius",      # "celsius" | "kelvin"
) -> RhinoObservation

def to_state(obs: RhinoObservation, *, source_order: Sequence[str]) -> State
```

### Behaviour

**`freq_unit` is required and then verified.** Accepted values are `"Hz"` and
`"MHz"`, matched case-insensitively; anything else raises. After conversion to
Hz, the values must fall in 1 MHz – 10 GHz. Outside that, raise, quoting the
declared unit and the resulting range. The band is deliberately far wider than
RHINO's 60–85 MHz: its job is to catch a 10⁶ unit error, not to police which
telescope wrote the file. This is not second-guessing the caller: a
declaration of MHz against an Hz-valued file lands at ~7 × 10¹³ Hz, which is
not a radio band, and stopping there is strictly better than interpolating Γ
against it.

**`thermistor_columns` is required**, replacing both the magic indices and the
"everything that is not `heated_load` uses the ambient column" rule. It maps
switch label → column index. Every label appearing in the switch log must have
an entry; a missing one raises and lists the labels it did not cover. Two
labels may name the same column — that is how the reference's "ambient covers
everything but the hot load" convention is expressed once the caller has to
write it down. `thermistor_k` is therefore keyed by switch label, and distinct
keys may hold equal arrays. Columns are interpolated from `temperature_times`
onto `time_s` through `_interp_strict`, so a thermistor log that does not span
the SDR time axis raises rather than clamping.

**`thermistor_unit` is honoured** rather than accepted and ignored: `"celsius"`
adds 273.15, `"kelvin"` does not, anything else raises.

**Per-sample labels come from `np.searchsorted`** on the transition log, not
from a Python loop building one boolean mask per switch block — the reference's
approach is `O(n_switch × n_time)` and a four-hour recording has thousands of
transitions.

**Samples before the first transition are dropped**, and the count is recorded
in `n_leading_dropped` rather than discarded quietly. They have no defined
switch state, and inventing one — a sentinel label, or assuming the first — is
exactly the finite-shaped-wrong-answer failure this codebase refuses elsewhere.

**`settled` is False for `settle_seconds` after each transition**, True
otherwise — including the first transition that survives the leading drop, so
the recording never opens with an unsettled sample marked usable.

**`adc_max_i` / `adc_max_q` are read when present, `None` when not.** Only the
notebook writes them; `save_to_hdf5` does not. They are ADC saturation
monitors, and saturation is a data-quality fact a future flagging step will
want. Reading them costs two lines and nothing consumes them yet — `to_state`
ignores them.

**`to_state(obs, source_order)`** produces a `State` whose:
- `data` is `obs.waterfall`,
- `coords.freq` is `obs.freq_hz`, `coords.time` is `obs.time_s`,
- `coords.extra["receiver_input"]` is the integer index (`(n_time,)`, integer
  dtype — `NoiseWaveOperator._source_index` requires exactly that shape) of
  each sample's label within `source_order`,
- `aux["flags"]` is `~obs.settled` — **inverted**, because flags are
  True-means-bad.

A label present in the data but absent from `source_order` raises immediately,
naming both sets. Deferring it means `SwitchCycle.gather` returns NaN much
later, at a point where the cause is no longer visible.

## Component 3 — errors and packaging

`src/rheplicant/core/errors.py` gains:

```python
class DataIngestionError(DirtError, ValueError):
    """A data file could not be read, or its contents contradict what the
    caller declared about them."""
```

`StateValidationError` is documented as covering *structural* problems with a
State — wrong ndim, wrong dtype, bad key types. A malformed Touchstone line and
a frequency unit that contradicts the file are neither, so they get their own
class. It is exported from `rheplicant` alongside the existing error types.

`pyproject.toml` gains an optional extra, written like `limtod` and `rfi`:

```toml
rhino = ["h5py"]
```

`touchstone.py` needs numpy only, which is already a transitive dependency of
jax, so it carries no new requirement. `rhino.py` gates its `h5py` import
through a `_require_h5py()` helper following `beams.py:42`'s pattern: raise
`ImportError` naming the install command, at the boundary, rather than an
`AttributeError` midway.

---

## Testing

`tests/radio/test_touchstone.py`, `tests/radio/test_rhino.py`. Fixtures are
written into `tmp_path` — Touchstone files as inline text, HDF5 files with
`h5py` — so no test data is committed and no writer is needed.

1. **Three formats agree.** One set of physical S values rendered as RI, MA and
   DB parses to the same complex array within floating-point tolerance.
2. **Each rejection has a case:** wrong column count, missing option line,
   non-ascending frequencies, suffix disagreeing with port count, `.s1p`
   accessed through `.s21`.
3. **Comment and whitespace handling:** a trailing `!` comment, blank lines and
   `!`-only lines do not change the parsed result.
4. **Extrapolation:** a target band wider than the file's raises by default;
   `allow_extrapolation=True` clamps without raising.
5. **Unit declaration:** declaring MHz for an Hz-valued file raises, and the
   message contains the actual range.
6. **`settled` polarity:** with known transition times, `settled` is False for
   exactly `settle_seconds` after each, and `to_state(...).aux["flags"]` is its
   exact complement.
7. **`to_state` label coverage:** a label outside `source_order` raises rather
   than producing an index.
8. **`n_leading_dropped`** counts correctly and the surviving arrays are
   consistent in length across `time_s`, `waterfall`, `switch_label`,
   `settled` and every `thermistor_k` entry.
9. **Thermistor mapping:** a file written with columns in a non-default order
   reads back correctly given the matching `thermistor_columns`, and raises
   when a switch label has no column.
10. **Cross-check against the reference implementation** — skipped when
    `rhino-cal` is not importable. The same `.s2p` read through both paths must
    agree pointwise in the complex plane; the same HDF5 read through both must
    agree on waterfall, times and frequencies. This is the same class of
    verification `rhino_cal_jax` was held to against the numpy `simulation/`
    module, and it is the only test that can show the rewrite preserved
    meaning rather than merely being self-consistent.

## What is not ported

- **`write_s2p`.** Nothing in rheplicant needs to emit Touchstone, and the
  tests build fixture text directly. Porting it would mean maintaining a
  formatter with no consumer. Its `np.nan_to_num` behaviour — silently turning
  a NaN S-parameter into 0, which reads as a perfectly matched port — is also
  not a behaviour worth carrying forward.
- **`set_up_switch_cycle_indices` / `states_array_generator` / `assign_states`.**
  These *generate* a switching schedule for the simulator. rheplicant expresses
  a schedule as `coords.extra["receiver_input"]` and already has everything it
  needs to build one; this spec only *reads* a schedule that a recording
  already contains.
- **`chebyshev_model` / `chebyshev_model_2d`.** These belong to the `(ν, t)`
  basis expansion, which is the next step's subject.
- **`gamma_db_to_impedence`.** No consumer.
