# RHINO ingestion layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two file readers to rheplicant — Touchstone `.s1p`/`.s2p` for measured reflection coefficients, and the RHINO observation HDF5 for the power waterfall / switch log / thermistor log — plus one explicit `to_state()` seam that places a recording on the signal graph.

**Architecture:** Two flat modules under `src/rheplicant/radio/`, alongside `beams.py`, which is the governing precedent: a file→numpy adapter that is not an operator. Both produce plain numpy in SI-at-the-seam units (Hz, unix seconds, Kelvin) with no astropy. `to_state()` is separate from reading, so a raw waterfall can be inspected without touching the signal graph.

**Tech Stack:** numpy (already a transitive jax dependency), h5py (new optional extra `rhino`), pytest. Design spec: `docs/superpowers/specs/2026-08-03-rhino-ingestion-design.md`.

---

## Prior context an implementer needs

**Run tests with `.venv/bin/python -m pytest`.** `uv run` cannot resolve this project's environment; the venv at `.venv/` already has `h5py` 3.16 and numpy 2.5 installed.

**Partial test runs need `--no-cov`.** `pyproject.toml`'s
`addopts = "-q --cov=rheplicant --cov-fail-under=80"` measures coverage over
the whole package, so running one file exits non-zero on the coverage gate even
when every test passes (a single-file run scores ~31 %). Every per-task command
below carries `--no-cov`; only Task 12's full-suite run leaves the gate on.

**Ruff runs with `select = ["E", "F", "I", "W", "UP", "B"]`.** `F401` is
therefore active: do not add an import in one task and first use it in the next.
The tasks below are ordered so that never happens. Check with
`.venv/bin/python -m ruff check src tests` before each commit.

**Branch:** `feat/rhino-ingestion`, already created. The spec is already committed on it (`e6e1482`).

**The reference implementation being ported** lives at `/Users/zzhang/projects/rhino-cal`:
`utils/utils.py:8` (`read_s2p`), `utils/utils.py:196` (`interp_vals_to_new_freq`),
`gcr/data_processing.py:19` (`DataHandler.__init__`),
`simulation/observation_handler.py:188` (`save_to_hdf5`).
Read the spec's "Five defects" section before starting — every one of them is a
test in this plan.

**Touchstone column order is not row-major.** A 2-port data line is
`freq S11 S21 S12 S22` — the second pair is **S21**, not S12. The reference gets
this right (`utils/utils.py:75-78`); a rewrite that assumes row-major will
transpose the off-diagonal and no test on `s11` alone will catch it.

**`aux["flags"]` is True-means-bad** (`radio/backend/flagging.py:62`), while
`settled` is True-means-good. `to_state` inverts. See Task 12.

**Existing conventions to match:**
- Optional dependency gating: `radio/beams.py:42` `_require_limtod_jax` — raise
  `ImportError` naming the install command at the boundary.
- Module docstrings in this package explain *why*, and name the failure a wrong
  choice would produce. Match that register; see `radio/instrument/noise_wave.py`.
- Errors come from `rheplicant.core.errors`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/rheplicant/core/errors.py` (modify) | Add `DataIngestionError`. |
| `src/rheplicant/core/__init__.py` (modify) | Re-export it. |
| `src/rheplicant/__init__.py` (modify) | Re-export it. |
| `src/rheplicant/radio/touchstone.py` (create) | Touchstone parse + strict interpolation. No h5py, no jax. |
| `src/rheplicant/radio/rhino.py` (create) | RHINO HDF5 → `RhinoObservation` → `State`. Imports `_interp_strict` from `touchstone.py`. |
| `src/rheplicant/radio/__init__.py` (modify) | Re-export both modules' public names. |
| `pyproject.toml` (modify) | Add the `rhino = ["h5py"]` extra. |
| `tests/radio/test_touchstone.py` (create) | Parser and interpolation surface. |
| `tests/radio/test_rhino.py` (create) | Reader, `to_state`, and the polarity of `settled` vs `flags`. |
| `tests/radio/test_ingestion_vs_reference.py` (create) | Cross-check against rhino-cal; skipped when it is not importable. |
| `CHANGELOG.md` (modify) | Entry. |

---

### Task 1: `DataIngestionError`

**Files:**
- Modify: `src/rheplicant/core/errors.py` (append)
- Modify: `src/rheplicant/core/__init__.py:10-15`, and its `__all__`
- Modify: `src/rheplicant/__init__.py` import block and `__all__`
- Test: `tests/core/test_errors.py` (create if absent, else append)

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_errors.py
import pytest

from rheplicant import DataIngestionError, DirtError


def test_data_ingestion_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(DataIngestionError, DirtError)
    assert issubclass(DataIngestionError, ValueError)
    with pytest.raises(DirtError):
        raise DataIngestionError("bad file")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_errors.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'DataIngestionError'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/rheplicant/core/errors.py`:

```python
class DataIngestionError(DirtError, ValueError):
    """A data file could not be read, or its contents contradict what the
    caller declared about them.

    Distinct from :class:`StateValidationError`, which covers *structural*
    problems with an in-memory State — wrong ndim, wrong dtype, bad key types.
    A malformed Touchstone line and a declared frequency unit that disagrees
    with the file's own values are neither: nothing is wrong with the shape of
    what was read, only with what it means. Both would otherwise propagate as a
    finite, correctly-shaped, wrong answer.
    """
```

In `src/rheplicant/core/__init__.py`, add `DataIngestionError,` to the
`from rheplicant.core.errors import (...)` block (alphabetically first, before
`DirtError`) and `"DataIngestionError",` to `__all__`.

In `src/rheplicant/__init__.py`, add `DataIngestionError,` to the
`from rheplicant.core import (...)` block and `"DataIngestionError",` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_errors.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/core/errors.py src/rheplicant/core/__init__.py src/rheplicant/__init__.py tests/core/test_errors.py
git commit -m "feat: DataIngestionError, for files that contradict their declaration"
```

---

### Task 2: `Touchstone` and a 2-port RI parse

**Files:**
- Create: `src/rheplicant/radio/touchstone.py`
- Test: `tests/radio/test_touchstone.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/radio/test_touchstone.py
"""The Touchstone adapter's own surface.

What an S-parameter *means* is rhino_cal_jax's subject. What belongs here is
the seam: that a Touchstone file's bytes become the complex array the rest of
the package expects, in Hz, with every way of getting that wrong named rather
than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError
from rheplicant.radio.touchstone import Touchstone, read_touchstone

RI_2PORT = """\
! a two-port, real/imaginary
# MHZ S RI R 50
60.0   0.10  0.20   0.30  0.40   0.50  0.60   0.70  0.80
70.0   0.11  0.21   0.31  0.41   0.51  0.61   0.71  0.81
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_ri_two_port_parses_to_hz_and_touchstone_column_order(tmp_path):
    ts = read_touchstone(write(tmp_path, "cal.s2p", RI_2PORT))

    assert isinstance(ts, Touchstone)
    np.testing.assert_allclose(ts.freq_hz, [60e6, 70e6])
    assert ts.s.shape == (2, 2, 2)
    # Touchstone 2-port column order is S11 S21 S12 S22 -- the SECOND pair is
    # S21. A row-major reading transposes the off-diagonal silently.
    np.testing.assert_allclose(ts.s11, [0.10 + 0.20j, 0.11 + 0.21j])
    np.testing.assert_allclose(ts.s21, [0.30 + 0.40j, 0.31 + 0.41j])
    np.testing.assert_allclose(ts.s12, [0.50 + 0.60j, 0.51 + 0.61j])
    np.testing.assert_allclose(ts.s22, [0.70 + 0.80j, 0.71 + 0.81j])
    assert ts.z0 == 50.0
    assert ts.n_port == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'rheplicant.radio.touchstone'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rheplicant/radio/touchstone.py
"""Touchstone .s1p / .s2p files as complex S-parameter arrays.

A thin adapter, in the same sense as :mod:`rheplicant.radio.beams`: it turns
bytes on disk into numpy, in the units this package carries at its seams (Hz),
and adds nothing else. What an S-parameter *means* -- how a reflection
coefficient becomes a coupling spectrum -- is ``rhino_cal_jax``'s subject.

Ported from ``rhino-cal``'s ``utils/utils.py::read_s2p``, with its silent
failure modes turned into errors. The one that matters most: that function
skips any data row whose column count is not exactly nine, so a trailing ``!``
comment or a truncated line removes a frequency point without a word, and the
caller gets a shorter sweep that still interpolates cleanly.

**Column order.** A Touchstone 2-port data row is ``freq S11 S21 S12 S22``.
The second pair is S21, not S12. This is the single most likely thing to get
wrong here, because every other 2x2 convention in this package is row-major and
because a test that only checks ``s11`` cannot see the error.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from rheplicant.core.errors import DataIngestionError

_FREQ_MULTIPLIER = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_FORMATS = ("RI", "MA", "DB")
#: Columns in a data row, keyed by port count: 1 + 2 * n_port**2.
_COLUMNS = {1: 3, 2: 9}


@dataclasses.dataclass(frozen=True)
class Touchstone:
    """Parsed Touchstone contents.

    Attributes:
        freq_hz: ``(n,)`` strictly ascending frequencies [Hz].
        s: ``(n, p, p)`` complex S-parameters, ``p`` in ``{1, 2}``.
        z0: reference impedance [ohm] from the option line.
    """

    freq_hz: np.ndarray
    s: np.ndarray
    z0: float

    @property
    def n_port(self) -> int:
        return int(self.s.shape[1])

    def _entry(self, row: int, col: int, name: str) -> np.ndarray:
        if row >= self.n_port or col >= self.n_port:
            raise DataIngestionError(
                f"{name} was requested from a {self.n_port}-port file. A 1-port "
                "measurement carries only s11; there is no transmission term to "
                "return and a zero would read as a perfectly isolated port."
            )
        return self.s[:, row, col]

    @property
    def s11(self) -> np.ndarray:
        return self._entry(0, 0, "s11")

    @property
    def s12(self) -> np.ndarray:
        return self._entry(0, 1, "s12")

    @property
    def s21(self) -> np.ndarray:
        return self._entry(1, 0, "s21")

    @property
    def s22(self) -> np.ndarray:
        return self._entry(1, 1, "s22")


def _to_complex(fmt: str, a: float, b: float) -> complex:
    if fmt == "RI":
        return complex(a, b)
    if fmt == "MA":
        return a * np.exp(1j * np.deg2rad(b))
    return 10 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))


def read_touchstone(path, *, flipped: bool = False) -> Touchstone:
    """Read a Touchstone v1 ``.s1p`` or ``.s2p`` file.

    Args:
        path: the file.
        flipped: treat the measurement as port-reversed (see below).

    Raises:
        DataIngestionError: on any malformed content. Nothing is skipped.
    """
    path = Path(path)
    multiplier = 1.0
    fmt = None
    z0 = 50.0
    freq: list[float] = []
    rows: list[list[complex]] = []
    n_column = None

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue

        if line.startswith("#"):
            parts = line[1:].upper().split()
            for token in parts:
                if token in _FREQ_MULTIPLIER:
                    multiplier = _FREQ_MULTIPLIER[token]
                elif token in _FORMATS:
                    fmt = token
            if "R" in parts:
                z0 = float(parts[parts.index("R") + 1])
            if fmt is None:
                raise DataIngestionError(
                    f"{path}:{lineno}: the option line names no data format; one "
                    f"of {_FORMATS} is required."
                )
            continue

        values = line.split()
        if fmt is None:
            raise DataIngestionError(
                f"{path}:{lineno}: data before any '#' option line, so the "
                f"frequency unit and data format are unknown. First data line: "
                f"{line!r}"
            )
        if n_column is None:
            if len(values) not in _COLUMNS.values():
                raise DataIngestionError(
                    f"{path}:{lineno}: {len(values)} columns; a Touchstone data "
                    f"row has 3 (1-port) or 9 (2-port). Line: {line!r}"
                )
            n_column = len(values)
        elif len(values) != n_column:
            raise DataIngestionError(
                f"{path}:{lineno}: {len(values)} columns, but this file's first "
                f"data row had {n_column}. Line: {line!r}"
            )

        numbers = [float(v) for v in values]
        freq.append(numbers[0] * multiplier)
        pairs = list(zip(numbers[1::2], numbers[2::2]))
        rows.append([_to_complex(fmt, a, b) for a, b in pairs])

    if not freq:
        raise DataIngestionError(f"{path}: no data rows.")

    n_port = 1 if n_column == 3 else 2
    freq_hz = np.asarray(freq, dtype=float)
    flat = np.asarray(rows, dtype=complex)

    s = np.empty((len(freq_hz), n_port, n_port), dtype=complex)
    if n_port == 1:
        s[:, 0, 0] = flat[:, 0]
    else:
        # Touchstone 2-port order: S11 S21 S12 S22.
        s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1] = (
            flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
        )

    if flipped:
        if n_port == 1:
            raise DataIngestionError(
                f"{path}: flipped=True on a 1-port file. Port reversal exchanges "
                "two ports; there is only one."
            )
        s = s[:, ::-1, ::-1]

    return Touchstone(freq_hz=freq_hz, s=s, z0=z0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/touchstone.py tests/radio/test_touchstone.py
git commit -m "feat: Touchstone reader, with the 2-port column order pinned"
```

---

### Task 3: MA and DB formats agree with RI

**Files:**
- Test: `tests/radio/test_touchstone.py` (append)

No implementation change is expected — Task 2 already handles all three. This
task exists to prove it, because the conversion is where a degree/radian or a
20-vs-10 slip lives.

- [ ] **Step 1: Write the failing test**

```python
def test_ri_ma_and_db_render_the_same_physical_s(tmp_path):
    z = 0.5 * np.exp(1j * np.deg2rad(37.0))
    ri = f"# HZ S RI R 50\n1.0e8   {z.real!r}  {z.imag!r}\n"
    ma = "# HZ S MA R 50\n1.0e8   0.5  37.0\n"
    db = f"# HZ S DB R 50\n1.0e8   {20 * np.log10(0.5)!r}  37.0\n"

    got = [
        read_touchstone(write(tmp_path, f"one_{tag}.s1p", text)).s11[0]
        for tag, text in (("ri", ri), ("ma", ma), ("db", db))
    ]
    for value in got:
        assert value == pytest.approx(z, rel=1e-12)
```

- [ ] **Step 2: Run test to verify it passes (or reveals a bug)**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py::test_ri_ma_and_db_render_the_same_physical_s -v --no-cov`
Expected: PASS. If it fails, the bug is in `_to_complex` — `DB` must be
`10 ** (dB / 20)` (an amplitude ratio), not `10 ** (dB / 10)` (a power ratio),
and both `MA` and `DB` angles are in **degrees**.

- [ ] **Step 3: Commit**

```bash
git add tests/radio/test_touchstone.py
git commit -m "test: the three Touchstone number formats agree"
```

---

### Task 4: every malformed input is named, not skipped

**Files:**
- Test: `tests/radio/test_touchstone.py` (append)
- Modify: `src/rheplicant/radio/touchstone.py` — add the ascending-frequency
  check and the suffix cross-check

- [ ] **Step 1: Write the failing test**

```python
def test_a_short_row_raises_instead_of_vanishing(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2  0.3 0.4  0.5 0.6  0.7 0.8\n70.0  0.1 0.2\n"
    with pytest.raises(DataIngestionError, match="columns"):
        read_touchstone(write(tmp_path, "short.s2p", text))


def test_data_before_the_option_line_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="option line"):
        read_touchstone(write(tmp_path, "no_opt.s1p", "60.0  0.1  0.2\n"))


def test_non_ascending_frequencies_raise(tmp_path):
    text = "# MHZ S RI R 50\n70.0  0.1 0.2\n60.0  0.3 0.4\n"
    with pytest.raises(DataIngestionError, match="ascending"):
        read_touchstone(write(tmp_path, "back.s1p", text))


def test_suffix_disagreeing_with_the_port_count_raises(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n"
    with pytest.raises(DataIngestionError, match="name says"):
        read_touchstone(write(tmp_path, "mislabelled.s2p", text))


def test_comments_and_blank_lines_do_not_change_the_result(tmp_path):
    noisy = (
        "! leading comment\n"
        "\n"
        "# MHZ S RI R 50\n"
        "!\n"
        "60.0  0.1 0.2   ! trailing comment on a real row\n"
        "   \n"
        "70.0  0.3 0.4\n"
    )
    clean = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"
    a = read_touchstone(write(tmp_path, "noisy.s1p", noisy))
    b = read_touchstone(write(tmp_path, "clean.s1p", clean))
    np.testing.assert_allclose(a.freq_hz, b.freq_hz)
    np.testing.assert_allclose(a.s11, b.s11)


def test_s1p_has_no_transmission_term(tmp_path):
    ts = read_touchstone(write(tmp_path, "one.s1p", "# MHZ S RI R 50\n60.0  0.1 0.2\n"))
    with pytest.raises(DataIngestionError, match="1-port"):
        ts.s21
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: FAIL on `test_non_ascending_frequencies_raise` and
`test_suffix_disagreeing_with_the_port_count_raises`; the other four pass.

- [ ] **Step 3: Write minimal implementation**

In `read_touchstone`, immediately after `freq_hz = np.asarray(freq, dtype=float)`,
insert:

```python
    if np.any(np.diff(freq_hz) <= 0):
        bad = int(np.argmin(np.diff(freq_hz) > 0)) + 2
        raise DataIngestionError(
            f"{path}: frequencies are not strictly ascending (first offender at "
            f"data row {bad}). Interpolation onto this sweep assumes they are, "
            "and np.interp does not check."
        )
```

and after `n_port = 1 if n_column == 3 else 2`, insert:

```python
    suffix = path.suffix.lower()
    if suffix in (".s1p", ".s2p"):
        declared = int(suffix[2])
        if declared != n_port:
            raise DataIngestionError(
                f"{path}: the name says {suffix} but the rows carry {n_column} "
                f"columns, i.e. {n_port}-port data. One of the two is wrong and "
                "guessing which would silently mislabel a port."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/touchstone.py tests/radio/test_touchstone.py
git commit -m "feat: Touchstone rejections are named rather than skipped"
```

---

### Task 5: `_interp_strict` and `interpolate_onto`

**Files:**
- Modify: `src/rheplicant/radio/touchstone.py` (append)
- Test: `tests/radio/test_touchstone.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from rheplicant.radio.touchstone import interpolate_onto


def test_interpolation_lands_on_the_target_grid(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    got = interpolate_onto(np.array([60e6, 70e6, 80e6]), ts)
    np.testing.assert_allclose(got, [0.0 + 0.0j, 1.0 + 2.0j, 2.0 + 4.0j])


def test_extrapolation_is_refused_by_default_and_names_both_bands(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    with pytest.raises(DataIngestionError, match="outside"):
        interpolate_onto(np.array([50e6, 70e6]), ts)


def test_extrapolation_clamps_when_explicitly_allowed(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    got = interpolate_onto(np.array([50e6]), ts, allow_extrapolation=True)
    np.testing.assert_allclose(got, [0.0 + 0.0j])


def test_an_unknown_component_raises(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"
    ts = read_touchstone(write(tmp_path, "one.s1p", text))
    with pytest.raises(DataIngestionError, match="component"):
        interpolate_onto(np.array([65e6]), ts, component="s99")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'interpolate_onto'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/rheplicant/radio/touchstone.py`:

```python
def _interp_strict(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    allow_extrapolation: bool = False,
    what: str = "value",
) -> np.ndarray:
    """Linear interpolation that refuses to leave the sampled range.

    ``np.interp`` *clamps* outside ``x`` rather than raising, so a reflection
    coefficient measured over a narrower band than the one being observed comes
    back as a constant at the edges, with no diagnostic. That is the failure
    this guard exists for; the reference implementation
    (``rhino-cal/utils/utils.py::interp_vals_to_new_freq``) has it.

    Real and imaginary parts are interpolated separately. Not magnitude/phase:
    across a mismatch resonance the phase wraps, and interpolating a wrapped
    angle is worse than interpolating Cartesian components, not better.
    """
    x_new = np.asarray(x_new, dtype=float)
    x = np.asarray(x, dtype=float)
    lo, hi = float(x[0]), float(x[-1])
    if not allow_extrapolation:
        # A tolerance on the span, so that a target grid whose endpoint agrees
        # with the source to within a unit conversion's rounding still passes.
        tol = 1e-9 * max(hi - lo, abs(hi))
        if x_new.min() < lo - tol or x_new.max() > hi + tol:
            raise DataIngestionError(
                f"{what}: the target range [{x_new.min():.6g}, {x_new.max():.6g}] "
                f"lies outside the sampled range [{lo:.6g}, {hi:.6g}]. "
                "np.interp would clamp to the edge values and report nothing; "
                "pass allow_extrapolation=True only if that is what you want."
            )
    if np.iscomplexobj(y):
        return np.interp(x_new, x, np.real(y)) + 1j * np.interp(x_new, x, np.imag(y))
    return np.interp(x_new, x, np.asarray(y, dtype=float))


def interpolate_onto(
    freq_hz: np.ndarray,
    source: Touchstone,
    *,
    component: str = "s11",
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Interpolate one S-parameter of ``source`` onto ``freq_hz`` [Hz]."""
    if component not in ("s11", "s12", "s21", "s22"):
        raise DataIngestionError(
            f"component must be one of s11, s12, s21, s22; got {component!r}."
        )
    values = getattr(source, component)
    return _interp_strict(
        freq_hz,
        source.freq_hz,
        values,
        allow_extrapolation=allow_extrapolation,
        what=f"{component} interpolation",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/touchstone.py tests/radio/test_touchstone.py
git commit -m "feat: strict interpolation that refuses to extrapolate silently"
```

---

### Task 6: export Touchstone, and add the `rhino` extra

**Files:**
- Modify: `src/rheplicant/radio/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/radio/test_touchstone.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_the_public_names_are_reachable_from_the_subpackage():
    from rheplicant.radio import Touchstone as T
    from rheplicant.radio import interpolate_onto as i
    from rheplicant.radio import read_touchstone as r

    assert (T, r, i) == (Touchstone, read_touchstone, interpolate_onto)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -k public -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'Touchstone' from 'rheplicant.radio'`

- [ ] **Step 3: Write minimal implementation**

In `src/rheplicant/radio/__init__.py`, after the `from rheplicant.radio.surrogate import NeuralOperator` line (`:83`), add:

```python
from rheplicant.radio.touchstone import Touchstone, interpolate_onto, read_touchstone
```

and add to `__all__`:

```python
    "Touchstone",
    "interpolate_onto",
    "read_touchstone",
```

In `pyproject.toml`, after the `rfi = ["MomentRFI"]` entry, add:

```toml
# read_rhino_observation() reads the RHINO observation HDF5 through h5py. Unlike
# limTOD and MomentRFI this one IS on PyPI, so the extra resolves normally:
#   uv pip install -e ".[rhino]"
# The Touchstone reader needs none of it -- it is numpy only.
rhino = ["h5py"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_touchstone.py -v --no-cov`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/__init__.py pyproject.toml tests/radio/test_touchstone.py
git commit -m "feat: export the Touchstone reader; add the rhino extra"
```

---

### Task 7: `RhinoObservation` and the core HDF5 read

**Files:**
- Create: `src/rheplicant/radio/rhino.py`
- Test: `tests/radio/test_rhino.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/radio/test_rhino.py
"""The RHINO observation reader's own surface.

The file schema is fixed by two producers that do not agree with each other --
rhino-cal's ObservationHandler.save_to_hdf5 writes frequencies in Hz, the
RHINO_fully_simulated_calibration notebook writes them in MHz, and the file
records neither. Most of what is tested here is that disagreement being caught
rather than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.radio.rhino import RhinoObservation, read_rhino_observation  # noqa: E402

FREQ_MHZ = np.array([60.0, 70.0, 80.0])
#: Four samples per switch position, 1 s apart, three positions.
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0
SWITCH_TIMES = np.array([1000.0, 1004.0, 1008.0])
SWITCH_STATES = [b"antenna", b"internal_load", b"heated_load"]


def make_file(path, *, freqs=FREQ_MHZ, times=TIME_S, temps=None, with_adc=False):
    n_time, n_freq = len(times), len(freqs)
    if temps is None:
        # column 0 = ambient (20 C), column 1 = hot (100 C)
        temps = np.stack(
            [np.full(n_time, 20.0), np.full(n_time, 100.0)], axis=1
        )
    with h5py.File(path, "w") as f:
        sdr = f.create_group("sdr")
        sdr.create_dataset("sdr_freqs", data=freqs)
        sdr.create_dataset("sdr_times", data=times)
        sdr.create_dataset(
            "sdr_waterfall",
            data=np.arange(n_time * n_freq, dtype=float).reshape(n_time, n_freq),
        )
        if with_adc:
            sdr.create_dataset("max_i_adc", data=np.zeros(n_time))
            sdr.create_dataset("max_q_adc", data=np.ones(n_time))
        sw = f.create_group("switches")
        sw.create_dataset("switch_times", data=SWITCH_TIMES)
        sw.create_dataset("switch_states", data=np.array(SWITCH_STATES, dtype="S"))
        tg = f.create_group("temperatures")
        tg.create_dataset("temperature_times", data=times)
        tg.create_dataset("temperatures", data=temps)
    return path


COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}


def test_reads_the_waterfall_and_converts_mhz_to_hz(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    assert isinstance(obs, RhinoObservation)
    np.testing.assert_allclose(obs.freq_hz, [60e6, 70e6, 80e6])
    assert obs.waterfall.shape == (len(TIME_S), 3)
    np.testing.assert_allclose(obs.time_s, TIME_S)


def test_declaring_the_wrong_frequency_unit_raises_with_the_actual_range(tmp_path):
    with pytest.raises(DataIngestionError, match="plausible range") as excinfo:
        read_rhino_observation(
            make_file(tmp_path / "hz.hd5f", freqs=FREQ_MHZ * 1e6),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )
    # The message must show what the declaration implies (8e+13 Hz) so the
    # reader can see at a glance that the other unit is the right one.
    assert "8e+13" in str(excinfo.value)


def test_an_unknown_frequency_unit_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="freq_unit"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"), freq_unit="GHz", thermistor_columns=COLUMNS
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'rheplicant.radio.rhino'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rheplicant/radio/rhino.py
"""RHINO observation HDF5 files as a recording, and as a State.

Two layers on purpose. :func:`read_rhino_observation` produces
:class:`RhinoObservation`, plain numpy that knows nothing about the signal
graph, so a waterfall can be plotted and a switch log inspected without
constructing anything. :func:`to_state` is the separate seam that places it on
the graph.

**The file does not record its own frequency unit, and its two producers
disagree.** ``rhino-cal``'s ``ObservationHandler.save_to_hdf5`` writes an
astropy Quantity in Hz; the ``RHINO_fully_simulated_calibration`` notebook
writes MHz. The reference reader, ``rhino-cal/gcr/data_processing.py``'s
``DataHandler``, defaults to MHz -- wrong for its own simulator's output, and
silent about it, because the consequence is Gamma interpolated onto a band
10^6 away, which then clamps to constant edge values rather than raising.
``freq_unit`` is therefore required here, with no default, and the declaration
is checked against the file's values.

The schema, as both producers write it::

    /sdr/sdr_freqs          (n_freq,)
    /sdr/sdr_times          (n_time,)            unix seconds
    /sdr/sdr_waterfall      (n_time, n_freq)     raw power
    /sdr/max_i_adc          (n_time,)            notebook-written files only
    /sdr/max_q_adc          (n_time,)            notebook-written files only
    /switches/switch_times  (n_switch,)          unix seconds
    /switches/switch_states (n_switch,)          bytes
    /temperatures/temperatures       (n_temp_time, n_column)   CELSIUS
    /temperatures/temperature_times  (n_temp_time,)            unix seconds

``/aux_sdr`` and ``/obs_config`` are ignored. ``save_to_hdf5`` creates
``/aux_sdr/aux_sdr_waterfall`` with a dtype but no ``data=`` and no ``shape=``,
which makes a *scalar* dataset rather than an array; there is nothing there to
read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from rheplicant.core.errors import DataIngestionError

_FREQ_UNIT_HZ = {"hz": 1.0, "mhz": 1e6}
#: Wide on purpose: this band's job is to catch a 10^6 unit error, not to
#: police which telescope wrote the file.
_PLAUSIBLE_HZ = (1e6, 1e10)
_KELVIN_OFFSET = 273.15


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "rheplicant.radio.rhino needs h5py: pip install \"rheplicant[rhino]\""
        ) from exc
    return h5py


@dataclasses.dataclass(frozen=True)
class RhinoObservation:
    """One RHINO recording, in numpy, in Hz / unix seconds / Kelvin.

    Attributes:
        freq_hz: ``(n_freq,)`` channel frequencies [Hz].
        time_s: ``(n_time,)`` sample times [unix seconds].
        waterfall: ``(n_time, n_freq)`` raw power, arbitrary scale.
        switch_label: ``(n_time,)`` per-sample switch state.
        settled: ``(n_time,)`` bool, **True = usable**. Note the polarity: it is
            the opposite of ``aux["flags"]``, which is True-means-bad.
        thermistor_k: switch label -> ``(n_time,)`` physical temperature [K].
            Two labels may share a column and therefore hold equal arrays.
        transitions: the raw ``(times, labels)`` switch log, kept for diagnosis.
        n_leading_dropped: samples that preceded the first transition and were
            dropped, because they have no defined switch state.
        adc_max_i / adc_max_q: ADC monitors, ``None`` when the file has none.
    """

    freq_hz: np.ndarray
    time_s: np.ndarray
    waterfall: np.ndarray
    switch_label: np.ndarray
    settled: np.ndarray
    thermistor_k: dict[str, np.ndarray]
    transitions: tuple[np.ndarray, np.ndarray]
    n_leading_dropped: int
    adc_max_i: np.ndarray | None
    adc_max_q: np.ndarray | None


def _frequencies_in_hz(raw: np.ndarray, freq_unit: str) -> np.ndarray:
    key = str(freq_unit).strip().lower()
    if key not in _FREQ_UNIT_HZ:
        raise DataIngestionError(
            f"freq_unit must be 'Hz' or 'MHz' (case-insensitive); got "
            f"{freq_unit!r}. The file does not record its own unit and its two "
            "known producers disagree, so there is no default to fall back on."
        )
    freq_hz = raw * _FREQ_UNIT_HZ[key]
    lo, hi = _PLAUSIBLE_HZ
    if freq_hz.min() < lo or freq_hz.max() > hi:
        raise DataIngestionError(
            f"declared freq_unit={freq_unit!r}, which puts this file's channels "
            f"at [{freq_hz.min():.6g}, {freq_hz.max():.6g}] Hz -- outside the "
            f"plausible range [{lo:.0e}, {hi:.0e}] Hz. The file's raw values span "
            f"[{raw.min():.6g}, {raw.max():.6g}]; the other unit is likely right."
        )
    return freq_hz


def read_rhino_observation(
    path,
    *,
    freq_unit: str,
    thermistor_columns: Mapping[str, int],
    settle_seconds: float = 5.0,
    thermistor_unit: str = "celsius",
) -> RhinoObservation:
    """Read a RHINO observation HDF5 file.

    Args:
        path: the ``.hd5f`` / ``.hdf5`` file.
        freq_unit: ``"Hz"`` or ``"MHz"``. Required -- see the module docstring.
        thermistor_columns: switch label -> column of ``/temperatures``. Required.
        settle_seconds: samples within this long after a transition are marked
            unsettled. The reference is inconsistent here (5 s in the notebook,
            2 s and 1 s in two rhino-cal functions); 5 s is the most conservative.
        thermistor_unit: ``"celsius"`` (the file's convention) or ``"kelvin"``.
    """
    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        freq_raw = np.asarray(f["sdr/sdr_freqs"][:], dtype=float)
        time_s = np.asarray(f["sdr/sdr_times"][:], dtype=float)
        waterfall = np.asarray(f["sdr/sdr_waterfall"][:], dtype=float)
        switch_time = np.asarray(f["switches/switch_times"][:], dtype=float)
        switch_raw = f["switches/switch_states"][:]
        temps_raw = np.asarray(f["temperatures/temperatures"][:], dtype=float)
        temp_time = np.asarray(f["temperatures/temperature_times"][:], dtype=float)
        adc_i = np.asarray(f["sdr/max_i_adc"][:], dtype=float) if "sdr/max_i_adc" in f else None
        adc_q = np.asarray(f["sdr/max_q_adc"][:], dtype=float) if "sdr/max_q_adc" in f else None

    freq_hz = _frequencies_in_hz(freq_raw, freq_unit)
    switch_label_raw = np.array(
        [s.decode() if isinstance(s, bytes) else str(s) for s in switch_raw]
    )
    return RhinoObservation(
        freq_hz=freq_hz,
        time_s=time_s,
        waterfall=waterfall,
        switch_label=switch_label_raw,
        settled=np.ones(time_s.shape, dtype=bool),
        thermistor_k={},
        transitions=(switch_time, switch_label_raw),
        n_leading_dropped=0,
        adc_max_i=adc_i,
        adc_max_q=adc_q,
    )
```

Note the placeholder values for `switch_label`, `settled` and `thermistor_k` —
Tasks 8 and 9 replace them. They are deliberately *wrong shapes conceptually*
but the right types, so the module imports and Task 7's tests pass without
pretending the later work is done.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/rhino.py tests/radio/test_rhino.py
git commit -m "feat: RHINO HDF5 reader core, with the frequency unit declared and checked"
```

---

### Task 8: per-sample switch labels, the leading drop, and `settled`

**Files:**
- Modify: `src/rheplicant/radio/rhino.py`
- Test: `tests/radio/test_rhino.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_switch_labels_are_expanded_per_sample(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert obs.switch_label.shape == (12,)
    assert list(obs.switch_label[:4]) == ["antenna"] * 4
    assert list(obs.switch_label[4:8]) == ["internal_load"] * 4
    assert list(obs.switch_label[8:]) == ["heated_load"] * 4


def test_settled_is_false_for_settle_seconds_after_each_transition(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=2.0,
    )
    # transitions at t = 1000, 1004, 1008; samples are 1 s apart from t = 1000.
    expected = np.array([False, False, True, True] * 3)
    np.testing.assert_array_equal(obs.settled, expected)


def test_samples_before_the_first_transition_are_dropped_and_counted(tmp_path):
    early = np.concatenate([[998.0, 999.0], TIME_S])
    obs = read_rhino_observation(
        make_file(tmp_path / "early.hd5f", times=early),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert obs.n_leading_dropped == 2
    assert obs.time_s.shape == (12,)
    assert obs.waterfall.shape == (12, 3)
    assert obs.switch_label.shape == (12,)
    assert obs.settled.shape == (12,)


def test_non_ascending_sample_times_raise(tmp_path):
    scrambled = TIME_S.copy()
    scrambled[5], scrambled[6] = scrambled[6], scrambled[5]
    with pytest.raises(DataIngestionError, match="ascending"):
        read_rhino_observation(
            make_file(tmp_path / "scrambled.hd5f", times=scrambled),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: FAIL — `switch_label` still holds the 3 raw transition labels, so the
shape assertion fails.

- [ ] **Step 3: Write minimal implementation**

Add to `src/rheplicant/radio/rhino.py`, above `read_rhino_observation`:

```python
def _expand_switch_log(
    time_s: np.ndarray,
    switch_time: np.ndarray,
    switch_label: np.ndarray,
    settle_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-sample labels and a settling mask, plus the leading-drop count.

    ``np.searchsorted`` rather than one boolean mask per switch block: a
    four-hour recording holds thousands of transitions, and the reference's
    per-block masking is O(n_switch * n_time).
    """
    if np.any(np.diff(time_s) <= 0):
        raise DataIngestionError(
            "sdr_times are not strictly ascending. The switch log is a list of "
            "transitions, so assigning a state to each sample assumes an "
            "ordered time axis."
        )
    order = np.argsort(switch_time, kind="stable")
    edges, labels = switch_time[order], switch_label[order]

    index = np.searchsorted(edges, time_s, side="right") - 1
    keep = index >= 0
    n_dropped = int((~keep).sum())
    index = index[keep]

    elapsed = time_s[keep] - edges[index]
    return labels[index], elapsed >= settle_seconds, keep, n_dropped
```

Then, in `read_rhino_observation`, replace the `return RhinoObservation(...)`
block with:

```python
    per_sample_label, settled, keep, n_dropped = _expand_switch_log(
        time_s, switch_time, switch_label_raw, settle_seconds
    )
    time_s = time_s[keep]
    waterfall = waterfall[keep]
    if adc_i is not None:
        adc_i = adc_i[keep]
    if adc_q is not None:
        adc_q = adc_q[keep]

    return RhinoObservation(
        freq_hz=freq_hz,
        time_s=time_s,
        waterfall=waterfall,
        switch_label=per_sample_label,
        settled=settled,
        thermistor_k={},
        transitions=(switch_time, switch_label_raw),
        n_leading_dropped=n_dropped,
        adc_max_i=adc_i,
        adc_max_q=adc_q,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/rhino.py tests/radio/test_rhino.py
git commit -m "feat: expand the switch log per sample, with a settling mask"
```

---

### Task 9: thermistor mapping, units, and ADC passthrough

**Files:**
- Modify: `src/rheplicant/radio/rhino.py`
- Test: `tests/radio/test_rhino.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_thermistor_columns_are_mapped_and_converted_to_kelvin(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert set(obs.thermistor_k) == {"antenna", "internal_load", "heated_load"}
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0 + 273.15)
    np.testing.assert_allclose(obs.thermistor_k["heated_load"], 100.0 + 273.15)
    # Two labels sharing column 0 hold equal arrays -- that is how the
    # reference's "ambient covers everything but the hot load" rule is written
    # down once the caller has to state it.
    np.testing.assert_allclose(
        obs.thermistor_k["antenna"], obs.thermistor_k["internal_load"]
    )


def test_kelvin_input_is_not_offset_again(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        thermistor_unit="kelvin",
        settle_seconds=0.0,
    )
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0)


def test_an_unknown_thermistor_unit_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="thermistor_unit"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            thermistor_unit="fahrenheit",
        )


def test_a_switch_label_with_no_column_raises_and_names_it(tmp_path):
    with pytest.raises(DataIngestionError, match="heated_load"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"),
            freq_unit="MHz",
            thermistor_columns={"antenna": 0, "internal_load": 0},
        )


def test_a_non_default_column_order_reads_back_correctly(tmp_path):
    # Hot in column 0, ambient in column 1 -- the reverse of what
    # save_to_hdf5's default save_temps order produces. The reference's magic
    # indices would silently swap them; a declared map cannot.
    swapped = np.stack(
        [np.full(len(TIME_S), 100.0), np.full(len(TIME_S), 20.0)], axis=1
    )
    obs = read_rhino_observation(
        make_file(tmp_path / "swapped.hd5f", temps=swapped),
        freq_unit="MHz",
        thermistor_columns={"antenna": 1, "internal_load": 1, "heated_load": 0},
        settle_seconds=0.0,
    )
    np.testing.assert_allclose(obs.thermistor_k["heated_load"], 100.0 + 273.15)
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0 + 273.15)


def test_every_time_axis_array_keeps_the_same_length_after_the_leading_drop(tmp_path):
    early = np.concatenate([[998.0, 999.0], TIME_S])
    obs = read_rhino_observation(
        make_file(tmp_path / "early.hd5f", times=early),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    n = len(obs.time_s)
    assert n == 12
    assert obs.waterfall.shape[0] == n
    assert obs.switch_label.shape == (n,)
    assert obs.settled.shape == (n,)
    for label, series in obs.thermistor_k.items():
        assert series.shape == (n,), label


def test_adc_monitors_are_passed_through_when_present(tmp_path):
    without = read_rhino_observation(
        make_file(tmp_path / "a.hd5f"), freq_unit="MHz", thermistor_columns=COLUMNS
    )
    assert without.adc_max_i is None and without.adc_max_q is None

    with_adc = read_rhino_observation(
        make_file(tmp_path / "b.hd5f", with_adc=True),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    assert with_adc.adc_max_i.shape == (12,)
    np.testing.assert_allclose(with_adc.adc_max_q, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: FAIL — `thermistor_k` is still `{}`.

- [ ] **Step 3: Write minimal implementation**

First add the import this task's helper needs (ruff's `F401` means it could not
have been added earlier):

```python
from rheplicant.radio.touchstone import _interp_strict
```

Then add to `src/rheplicant/radio/rhino.py`, above `read_rhino_observation`:

```python
def _thermistors_in_kelvin(
    time_s: np.ndarray,
    temp_time: np.ndarray,
    temps: np.ndarray,
    labels_present: set[str],
    thermistor_columns: Mapping[str, int],
    thermistor_unit: str,
) -> dict[str, np.ndarray]:
    """Map switch labels onto thermistor columns, in Kelvin, on ``time_s``.

    The reference (``rhino-cal/gcr/data_processing.py``) takes
    ``heated_load_index=1, ambient_load_index=0`` and routes every state except
    ``heated_load`` to the ambient column. Those indices are the positional
    order of ``save_to_hdf5``'s ``save_temps`` argument -- a convention shared
    between writer and reader with nothing in the file to enforce it, so a file
    written with a different order reads back with hot and ambient swapped and
    nothing raises. Requiring the map makes that a declaration, not a default.
    """
    unit = str(thermistor_unit).strip().lower()
    if unit == "celsius":
        temps_k = temps + _KELVIN_OFFSET
    elif unit == "kelvin":
        temps_k = temps
    else:
        raise DataIngestionError(
            f"thermistor_unit must be 'celsius' or 'kelvin'; got {thermistor_unit!r}."
        )

    if temps_k.ndim != 2:
        raise DataIngestionError(
            f"/temperatures/temperatures must be (n_temp_time, n_column); got "
            f"ndim={temps_k.ndim}."
        )

    missing = sorted(labels_present - set(thermistor_columns))
    if missing:
        raise DataIngestionError(
            f"thermistor_columns has no entry for {missing}, which appear in the "
            f"switch log. Declared: {sorted(thermistor_columns)}."
        )

    out: dict[str, np.ndarray] = {}
    for label in sorted(labels_present):
        column = thermistor_columns[label]
        if not 0 <= column < temps_k.shape[1]:
            raise DataIngestionError(
                f"thermistor_columns[{label!r}] = {column}, but /temperatures "
                f"has {temps_k.shape[1]} columns."
            )
        out[label] = _interp_strict(
            time_s, temp_time, temps_k[:, column],
            what=f"thermistor column {column} for {label!r}",
        )
    return out
```

Then, in `read_rhino_observation`, replace `thermistor_k={},` with:

```python
        thermistor_k=_thermistors_in_kelvin(
            time_s, temp_time, temps_raw, set(per_sample_label.tolist()),
            thermistor_columns, thermistor_unit,
        ),
```

(place the call after the `keep`-slicing of `time_s`, so it interpolates onto
the surviving axis).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/rhino.py tests/radio/test_rhino.py
git commit -m "feat: declared thermistor column mapping, in Kelvin, on the sample axis"
```

---

### Task 10: `to_state`, and the flag polarity

**Files:**
- Modify: `src/rheplicant/radio/rhino.py` (append)
- Modify: `src/rheplicant/radio/__init__.py`
- Test: `tests/radio/test_rhino.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_to_state_indexes_sources_and_inverts_the_settling_mask(tmp_path):
    from rheplicant.radio.rhino import to_state

    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=2.0,
    )
    state = to_state(obs, source_order=("antenna", "internal_load", "heated_load"))

    index = np.asarray(state.coords.extra["receiver_input"])
    assert index.shape == (12,)
    assert np.issubdtype(index.dtype, np.integer)
    np.testing.assert_array_equal(index, [0] * 4 + [1] * 4 + [2] * 4)

    np.testing.assert_allclose(np.asarray(state.coords.freq), obs.freq_hz)
    np.testing.assert_allclose(np.asarray(state.coords.time), obs.time_s)
    np.testing.assert_allclose(np.asarray(state.data), obs.waterfall)

    # aux["flags"] is True-means-BAD; settled is True-means-GOOD.
    np.testing.assert_array_equal(
        np.asarray(state.aux["flags"]), ~obs.settled
    )
    assert np.asarray(state.aux["flags"]).sum() == 6


def test_to_state_rejects_a_label_outside_source_order(tmp_path):
    from rheplicant.radio.rhino import to_state

    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    with pytest.raises(DataIngestionError, match="heated_load"):
        to_state(obs, source_order=("antenna", "internal_load"))


def test_the_public_names_are_reachable_from_the_subpackage():
    from rheplicant.radio import RhinoObservation as R
    from rheplicant.radio import read_rhino_observation as r
    from rheplicant.radio import rhino_to_state as t

    assert (R, r) == (RhinoObservation, read_rhino_observation)
    assert callable(t)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/radio/test_rhino.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'to_state'`

- [ ] **Step 3: Write minimal implementation**

First add this task's imports, alongside the existing ones:

```python
from collections.abc import Mapping, Sequence   # Sequence is new

import jax.numpy as jnp

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.state import State
```

Then append to `src/rheplicant/radio/rhino.py`:

```python
def to_state(obs: RhinoObservation, *, source_order: Sequence[str]) -> State:
    """Place a recording on the signal graph.

    Args:
        obs: a recording.
        source_order: switch labels in the graph's in-edge order. Read it off
            the assembled twin -- ``assembly["receiver_input"].names`` -- rather
            than assuming it: it is the order ``NoiseWaveOperator``'s
            ``gamma_src`` rows must match, and a transposition there is
            shape-legal and costs tens of kelvin.

    The settling mask is **inverted** on the way in. ``aux["flags"]`` is
    True-means-flagged (``radio/backend/flagging.py``, and ``FlaggedNoise``
    consumes it that way) while ``settled`` is True-means-usable. Getting this
    backwards yields a finite, correctly-shaped result that discards every good
    sample and keeps every transient.
    """
    order = tuple(source_order)
    lookup = {label: i for i, label in enumerate(order)}
    unknown = sorted(set(obs.switch_label.tolist()) - set(lookup))
    if unknown:
        raise DataIngestionError(
            f"the recording switches to {unknown}, which source_order does not "
            f"name (it lists {list(order)}). Deferring this makes "
            "SwitchCycle.gather return NaN much later, where the cause is no "
            "longer visible."
        )
    index = np.array([lookup[label] for label in obs.switch_label], dtype=int)
    return State(
        data=jnp.asarray(obs.waterfall),
        coords=Coordinates(
            time=obs.time_s,
            freq=obs.freq_hz,
            extra={"receiver_input": jnp.asarray(index)},
        ),
        aux={"flags": jnp.asarray(~obs.settled)},
    )
```

In `src/rheplicant/radio/__init__.py`, add after the `touchstone` import:

```python
from rheplicant.radio.rhino import (
    RhinoObservation,
    read_rhino_observation,
    to_state as rhino_to_state,
)
```

and to `__all__`:

```python
    "RhinoObservation",
    "read_rhino_observation",
    "rhino_to_state",
```

The alias is deliberate: `to_state` is too generic a name to sit in
`rheplicant.radio`'s flat namespace, while `rhino.to_state` reads correctly at
its own module path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/radio/ -v --no-cov`
Expected: PASS (whole radio suite, including the pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/rheplicant/radio/rhino.py src/rheplicant/radio/__init__.py tests/radio/test_rhino.py
git commit -m "feat: rhino.to_state, with the settling mask inverted into flags"
```

---

### Task 11: cross-check against the reference implementation

**Files:**
- Create: `tests/radio/test_ingestion_vs_reference.py`

This is the only test that can show the rewrite preserved *meaning* rather than
merely being self-consistent — the same class of verification `rhino_cal_jax`
was held to against the numpy `simulation/` module. It is skipped when
`rhino-cal` is not importable, so CI stays green without it.

- [ ] **Step 1: Write the test**

```python
# tests/radio/test_ingestion_vs_reference.py
"""Agreement with rhino-cal's numpy readers, where both can read the same file.

Skipped unless the rhino-cal checkout is importable. Nothing here re-tests
rheplicant's own rejections -- those live in test_touchstone.py and
test_rhino.py. What this file establishes is that where the reference produces
an answer, so do we, and it is the same one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

RHINO_CAL = Path("~/projects/rhino-cal").expanduser()
if RHINO_CAL.is_dir() and str(RHINO_CAL) not in sys.path:
    sys.path.insert(0, str(RHINO_CAL))

reference = pytest.importorskip(
    "utils.utils", reason=f"rhino-cal not present at {RHINO_CAL}"
)
h5py = pytest.importorskip("h5py")

from rheplicant.radio.rhino import read_rhino_observation  # noqa: E402
from rheplicant.radio.touchstone import read_touchstone  # noqa: E402

# The HDF5 fixture is rebuilt here rather than imported from test_rhino.py:
# tests/radio/ has no __init__.py, so there is no package for a relative import
# to resolve against, and adding one to make two test modules share twenty
# lines is the wrong trade.
FREQ_MHZ = np.array([60.0, 70.0, 80.0])
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0
SWITCH_TIMES = np.array([1000.0, 1004.0, 1008.0])
SWITCH_STATES = [b"antenna", b"internal_load", b"heated_load"]
COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}


def make_file(path):
    n_time, n_freq = len(TIME_S), len(FREQ_MHZ)
    temps = np.stack([np.full(n_time, 20.0), np.full(n_time, 100.0)], axis=1)
    with h5py.File(path, "w") as f:
        sdr = f.create_group("sdr")
        sdr.create_dataset("sdr_freqs", data=FREQ_MHZ)
        sdr.create_dataset("sdr_times", data=TIME_S)
        sdr.create_dataset(
            "sdr_waterfall",
            data=np.arange(n_time * n_freq, dtype=float).reshape(n_time, n_freq),
        )
        sw = f.create_group("switches")
        sw.create_dataset("switch_times", data=SWITCH_TIMES)
        sw.create_dataset("switch_states", data=np.array(SWITCH_STATES, dtype="S"))
        tg = f.create_group("temperatures")
        tg.create_dataset("temperature_times", data=TIME_S)
        tg.create_dataset("temperatures", data=temps)
    return path


TWO_PORT = """\
# MHZ S RI R 50
60.0   0.10  0.20   0.30  0.40   0.50  0.60   0.70  0.80
70.0   0.11 -0.21   0.31  0.41  -0.51  0.61   0.71  0.81
80.0  -0.12  0.22   0.32 -0.42   0.52  0.62   0.72 -0.82
"""


def test_touchstone_agrees_with_read_s2p(tmp_path):
    path = tmp_path / "cal.s2p"
    path.write_text(TWO_PORT)

    s11, s12, s21, s22, freq = reference.read_s2p(str(path))
    ts = read_touchstone(path)

    np.testing.assert_allclose(ts.freq_hz, freq)
    np.testing.assert_allclose(ts.s11, s11)
    np.testing.assert_allclose(ts.s12, s12)
    np.testing.assert_allclose(ts.s21, s21)
    np.testing.assert_allclose(ts.s22, s22)


def test_flipped_agrees_with_read_s2p_flipped(tmp_path):
    path = tmp_path / "cal.s2p"
    path.write_text(TWO_PORT)

    s11, s12, s21, s22, _ = reference.read_s2p(str(path), flipped_measurement=True)
    ts = read_touchstone(path, flipped=True)

    np.testing.assert_allclose(ts.s11, s11)
    np.testing.assert_allclose(ts.s12, s12)
    np.testing.assert_allclose(ts.s21, s21)
    np.testing.assert_allclose(ts.s22, s22)


def test_hdf5_waterfall_times_and_frequencies_agree_with_datahandler(tmp_path):
    data_processing = pytest.importorskip("gcr.data_processing")

    path = make_file(tmp_path / "obs.hd5f")
    ours = read_rhino_observation(
        path, freq_unit="MHz", thermistor_columns=COLUMNS, settle_seconds=0.0
    )
    theirs = data_processing.DataHandler(
        filepath=str(path), gamma_src_dict={}, gamma_rec=None
    )

    np.testing.assert_allclose(ours.freq_hz, np.asarray(theirs.freqs.to("Hz").value))
    np.testing.assert_allclose(ours.time_s, np.asarray(theirs.times.to("s").value))
    np.testing.assert_allclose(ours.waterfall, theirs.waterfall)
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/radio/test_ingestion_vs_reference.py -v --no-cov`
Expected: PASS, or SKIPPED if `~/projects/rhino-cal` is absent.

If `test_hdf5_...` fails on `theirs.freqs`, check whether `DataHandler`'s
`freq_unit=un.MHz` default matches the fixture — the fixture writes MHz, so it
should. A failure there is a real disagreement worth reporting, not a test to
loosen.

- [ ] **Step 3: Commit**

```bash
git add tests/radio/test_ingestion_vs_reference.py
git commit -m "test: cross-check both readers against rhino-cal's numpy originals"
```

---

### Task 12: changelog, and the full suite

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, with the pre-existing count plus the new tests. Investigate any
pre-existing failure before continuing rather than assuming it is unrelated.

- [ ] **Step 2: Add the changelog entry**

Under the current unreleased heading in `CHANGELOG.md`, add:

```markdown
### Added

- `rheplicant.radio.touchstone` — Touchstone `.s1p`/`.s2p` reader
  (`read_touchstone`, `interpolate_onto`). Ported from rhino-cal's `read_s2p`
  with its silent row-skipping turned into errors, and with interpolation that
  refuses to extrapolate rather than clamping to the edge values.
- `rheplicant.radio.rhino` — RHINO observation HDF5 reader
  (`read_rhino_observation`, `RhinoObservation`, `to_state`). The frequency
  unit is a required argument, because the file does not record it and its two
  known producers disagree; the declaration is then checked against the file's
  own values. Thermistor columns must be declared rather than defaulted.
  Needs the new `rheplicant[rhino]` extra (h5py).
- `DataIngestionError`, for files that cannot be read or that contradict what
  the caller declared about them.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for the ingestion layer"
```

---

## What this plan does not build

Named here so a reader does not go looking for them:

- The Dicke `q`-ratio reduction `(P_src − P_l)/(P_ns − P_l)`, its quadrature
  error propagation, and the `(ν, t)` basis expansion. Next step.
- The `T_ant` back-solve.
- `write_s2p`. No consumer, and its `np.nan_to_num` turns a NaN S-parameter
  into 0, which reads as a perfectly matched port.
- Switch-schedule *generation* (`set_up_switch_cycle_indices` and friends).
  rheplicant expresses a schedule as `coords.extra["receiver_input"]` and can
  already build one; this plan only reads a schedule a recording contains.
