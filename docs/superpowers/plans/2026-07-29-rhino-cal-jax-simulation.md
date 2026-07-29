# rhino-cal-jax: Eq. 1 simulation core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the noise-wave data model (Noise_Wave_GCR draft Eq. 1, with the source-switching layer of Eqs. 11–12) as a differentiable JAX/Equinox package `rhino_cal_jax` living inside the `RHINO-Experiment/rhino-cal` repository, verify it against the existing numpy `simulation/` implementation, and wire it into rheplicant through a thin adapter operator.

**Architecture:** `rhino_cal_jax` is a standalone package that does **not** import rheplicant — the dependency runs one way, exactly as `limtod_jax` → `rheplicant`. It splits the model at its natural seam: the four *coupling spectra* (`c_s`, `κ_unc`, `κ_cos`, `κ_sin`) depend only on the reflection coefficients, while the model is **exactly linear** in the temperature vector `(T_src, T_unc, T_cos, T_sin, T_rx)`. That split is not cosmetic — the coupling array *is* the design matrix the draft's GCR (Eqs. 26/30/31) solves against, so getting it out as a first-class object is what lets rheplicant's existing `linear.py` (`check_linearity` / `wiener_solve` / `gcr_sample`) apply without new machinery. The switching layer (Eq. 12's `θ`) carries `Γ` **per source** and gathers onto the time axis, which is also the fix for the rank-1 defect described in "Prior context" below.

**Tech Stack:** JAX, Equinox, pytest. Reference implementation: numpy + astropy in the same repo (`simulation/`, `utils/`).

---

## Prior context an implementer needs

**Repositories.**
- Dev clone (writable, SSH): `/Users/zzhang/projects/rhino-cal` → `git@github.com:RHINO-Experiment/rhino-cal.git`. All Tasks 1–8 happen here.
- rheplicant: `/Users/zzhang/projects/e-RHINO` (package `src/rheplicant`). Tasks 9–11 happen here.
- A read-only reference snapshot of rhino-cal also sits at `e-RHINO/assets/rhino-cal`. **`assets/` is git-ignored and must never be committed** — it holds an unpublished draft.

**The blocking defect this plan fixes.** rheplicant's current `NoiseWaveOperator` holds a *single* `(gamma_re, gamma_im)` pair and sits downstream of the `receiver_input` selector, so every switched source shares one `Γ`. Draft Eqs. 11–12 say the right thing: `Γ` is a property of the *source*, and `θ(t_j, φ_h)` selects which one is connected. Carrying `Γ` per source and gathering through the switch is the fix.

**Why that matters — stated precisely, because the loose version is wrong.** Count equations *per frequency channel*, because the noise-wave temperatures are functions of frequency and nothing ties channels together a priori. Each switch position contributes exactly **one** equation per channel. So with `n_src` distinct reflection coefficients you get `n_src` equations per channel against the 3 noise-wave unknowns living there — and the design matrix has rank `min(n_src, 3) × n_freq`. Measured, for `n_freq = 4`:

| distinct loads | rank / unknowns |
|---|---|
| 1 | 4 / 12 — deficient |
| 2 | 8 / 12 — deficient |
| 3 | 12 / 12 — full rank |

which is why real experiments (EDGES, REACH) switch between four or five calibrators, not one.

Two corollaries worth keeping straight, both verified numerically:
- **Frequency structure in `Γ` does not rescue a single load** for per-channel temperatures. It *does* fully identify them if the temperatures are instead **scalars** (frequency-independent): one load then gives `n_freq` equations for 3 unknowns, well conditioned (`cond(JᵀJ) ≈ 6`). So any test that uses scalar temperatures cannot demonstrate what switching buys — it will pass with one source.
- The bridge between those two regimes is exactly the draft's basis functions `U_unc`, `U_cos`, `U_sin` (Eqs. 13–15): they tie channels together, cutting the per-channel parameter count so that fewer calibrators suffice. That is what those matrices are *for*.

Do not restate this as "one shared `Γ` makes the columns proportional" — that is not the mechanism, and a test built on it will contradict itself.

**Discrepancies already established between the draft and the numpy code.** These are decided, not open questions — implement the right-hand column and pin each with a test:

| # | Draft says | numpy code says | Implement | Why |
|---|---|---|---|---|
| D1 | Eq. 4: `κ_unc = \|Γ\|²\|F\|` | `t_unc * abs(gamma_src)**2 * abs(f_src)**2` | `\|Γ\|²\|F\|²` | Eq. 2 renders both squares (`(1−\|Γ\|2)\|F\|2`), Eq. 4 renders only one — a typesetting drop. `\|Γ\|²\|F\|²` is the Meys/EDGES/REACH form and is what `gcr/transfer_matrix_construction.py::construct_h_spectra` also uses, so the repo is self-consistent against the code, not the draft. |
| D2 | Eq. 1 puts `n_w` inside the bracket (additive in K); Eq. 8 writes `d = G T_sys (1+w)` (fractional) | fractional: `δP = P/√(τΔν)` | fractional (Eq. 8) | The two draft forms agree only if `n_w = T_sys·w`. The code and the radiometer equation both mean fractional. |
| D3 | — | `p_src = np.abs(p_src + noise)` | no fold; `fold_negative=False` default | Folding is not a physical model: it biases the mean upward whenever `P/√(τΔν)` is not ≫ 1, and it silently breaks the Gaussian likelihood the GCR assumes. Keep it reachable behind a flag purely so the consistency test can reproduce the reference bit-for-bit. |

Report D1–D3 to the draft's author (Jordan Norris) once the tests pin them; do not silently "fix" the draft.

**Precision.** The two repos differ, and it matters for every tolerance in this plan.
- **rhino-cal**: `tests/conftest.py` (Task 1) enables float64 for the whole suite, so the test modules in Tasks 2–7 must **not** repeat `jax.config.update("jax_enable_x64", True)` — drop it, put their imports at the top normally, and drop the `# noqa: E402` markers that only existed to accommodate it.
- **rheplicant**: its suite deliberately runs **float32**, and flipping `jax_enable_x64` mid-process is global, so it cannot be enabled per-module. Existing practice (see `tests/radio/test_sky_abstraction.py::test_oracle_x64_subprocess`) checks float64 criteria in a fresh interpreter with `JAX_ENABLE_X64=1`. Any tolerance in Task 9 must therefore be chosen from the active precision, never hard-coded to a float64 value.

**House rules that apply.**
- A failure mode that produces a finite, correctly-shaped, **wrong** answer must raise, not warn. `NaN`/`Inf` are loud and therefore acceptable; a silently-zeroed `κ_sin` is not.
- Structural validation only inside `__call__` (shapes, dtypes) — never traced values, or it breaks under `jit`.
- All comments, docstrings and commit messages in English.
- Commit format: `<type>: <description>` (feat, fix, refactor, docs, test, chore).

---

## File structure

**New, in `/Users/zzhang/projects/rhino-cal`:**

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package **only** `rhino_cal_jax` — in the wheel *and* the sdist; leave the existing numpy dirs unpackaged and importable from the repo root as they are today. |
| `tests/conftest.py` | Enable float64 once for the whole suite. |
| `rhino_cal_jax/__init__.py` | Public API surface. |
| `rhino_cal_jax/errors.py` | `RhinoCalError` / `ValidationError`. |
| `rhino_cal_jax/reflection.py` | Draft Eqs. 2–6: `F`, and the `Couplings` container. |
| `rhino_cal_jax/loads.py` | `Γ` construction (termination type, cable phase) + `Load` / `Receiver`. |
| `rhino_cal_jax/switching.py` | Draft Eqs. 11–12: `SwitchCycle` (`θ`), gather per-source → per-time. |
| `rhino_cal_jax/power.py` | Draft Eqs. 1/8: `system_temperature`, `radiometer_power`, radiometer noise. |
| `rhino_cal_jax/sky.py` | Synchrotron power-law source temperature. |
| `tests/test_reflection.py` | Eqs. 2–6 unit + boundary tests. |
| `tests/test_loads.py` | `Γ` construction tests. |
| `tests/test_switching.py` | `θ` tests. |
| `tests/test_power.py` | Eq. 1 unit tests + exact-linearity test. |
| `tests/test_sky.py` | Synchrotron tests. |
| `tests/test_consistency_with_numpy.py` | The cross-check against `simulation/` + `utils/`. |

**Modified, in `/Users/zzhang/projects/e-RHINO`:**

| File | Change |
|---|---|
| `src/rheplicant/radio/instrument/noise_wave.py` | Replace the placeholder with a real adapter over `rhino_cal_jax`, carrying `Γ` per source. |
| `src/rheplicant/radio/__init__.py` | Export unchanged name; docstring update. |
| `tests/radio/test_noise_wave.py` | New: adapter tests, incl. the rank-1 regression. |
| `examples/noise_wave_gcr.py` | New: couplings → `ParameterSpace(linear=True)` → `wiener_solve`/`gcr_sample`. |
| `pyproject.toml`, `CHANGELOG.md`, `DESIGN.md`, `docs/*` | Optional extra, decision record, docs. |

---

## Task 1: Package scaffolding

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/pyproject.toml`
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/__init__.py`
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/errors.py`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_import.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_import.py`:

```python
"""The package imports, and its error type is catchable as a ValueError."""

import pytest


def test_package_imports():
    import rhino_cal_jax

    assert rhino_cal_jax.__version__


def test_validation_error_is_a_value_error():
    from rhino_cal_jax.errors import RhinoCalError, ValidationError

    assert issubclass(ValidationError, RhinoCalError)
    assert issubclass(ValidationError, ValueError)
    with pytest.raises(ValueError):
        raise ValidationError("boom")


def test_numpy_reference_is_importable():
    """The consistency suite reads the numpy implementation from the repo root."""
    from simulation.radiometer_power import compute_radiometer_power

    assert callable(compute_radiometer_power)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax'`

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
# The distribution ships ONLY the JAX package, so it is named for that rather
# than for the whole pipeline. It also leaves the name `rhino-cal` free should
# the numpy side ever be packaged separately.
name = "rhino-cal-jax"
dynamic = ["version"]
description = "Differentiable JAX/Equinox implementation of the RHINO noise-wave data model"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
# Required, not optional: importing rhino_cal_jax re-exports jax-backed
# symbols, so there is no meaningful install of this distribution without them.
dependencies = ["jax>=0.5", "equinox>=0.13"]

[project.urls]
Repository = "https://github.com/RHINO-Experiment/rhino-cal"

[tool.hatch.version]
# Single source of truth, so pyproject and __init__ cannot drift apart.
path = "rhino_cal_jax/__init__.py"

[tool.hatch.build.targets.wheel]
# Only the JAX package ships. The numpy modules are run from a repo checkout
# (they import as `simulation.*`, `gcr.*`, `utils.*` namespace packages) and
# packaging them would mean rewriting those imports.
packages = ["rhino_cal_jax"]
# Editable installs must expose ONLY rhino_cal_jax. Hatchling's default "loose"
# mode puts the whole project root on sys.path via a .pth file, which with this
# repo's flat layout would claim `simulation`, `gcr`, `utils` and the top-level
# notebooks as global top-level modules for the entire interpreter.
dev-mode-exact = true

[tool.hatch.build.targets.sdist]
# The wheel setting above does NOT scope the sdist -- hatchling's sdist target
# defaults to the whole tracked tree, which here means every notebook and all
# of the collaborator's numpy code. Restrict it explicitly.
#
# `tests` is deliberately NOT shipped. The suite cross-checks against the numpy
# pipeline in simulation/ and utils/, which the sdist rightly omits, so those
# tests cannot pass from an extracted sdist. Shipping a suite that is
# guaranteed to fail is worse than not shipping one; the tests live in the
# repository, which is where they are meant to be run.
include = ["rhino_cal_jax", "README.md", "LICENSE", "pyproject.toml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Puts the repo root on sys.path so the consistency suite can import the numpy
# reference (simulation/, utils/) alongside the installed rhino_cal_jax.
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"
# Exclude the collaborator's pre-existing numpy pipeline and notebooks -- they
# predate any linter config here and report hundreds of findings, and linting
# them is their call, not a side effect of adding this package.
#
# A deny-list, not an allow-list: `include = [...]` would silently leave any
# NEW top-level file belonging to this package (a noxfile, a docs conf, a
# helper script) unlinted with no signal at all. This way the default is
# "lint it" and only the named legacy paths opt out.
extend-exclude = ["simulation", "gcr", "utils", "rfi_flagging", "*.ipynb"]

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
# E741: single-character names (l, I, O) are standard physics notation.
ignore = ["E741"]
```

Create `tests/conftest.py`:

```python
"""Suite-wide configuration.

float64 is enabled here, once, rather than in each test module. The
consistency suite compares against a numpy reference at ``rtol=1e-13``; under
JAX's default float32 that comparison cannot pass, and a module that simply
forgot the line would report a plausible-looking failure with no hint that
precision was the cause. Setting it in ``conftest.py`` means no test module can
forget it -- pytest imports this before collecting any of them.
"""

import jax

jax.config.update("jax_enable_x64", True)
```

Create `rhino_cal_jax/errors.py`:

```python
"""Exception hierarchy.

One root (:class:`RhinoCalError`) so callers can catch everything this package
raises, and leaf classes that also subclass the builtin a caller would
naturally reach for.
"""


class RhinoCalError(Exception):
    """Base class for every error raised by rhino_cal_jax."""


class ValidationError(RhinoCalError, ValueError):
    """An input failed a structural check (shape, dtype, or declared size)."""
```

Create `rhino_cal_jax/__init__.py`:

```python
"""Differentiable JAX/Equinox implementation of the RHINO noise-wave data model.

The model (Noise_Wave_GCR draft, Eq. 1) is the spectral power recorded by the
spectrometer when source ``k`` is connected to the receiver::

    d(nu, t) = G(nu, t) [ T_src c_s + T_unc k_unc + T_cos k_cos
                          + T_sin k_sin + T_rx ] (1 + w)

Everything to the right of a temperature depends only on the reflection
coefficients, and the bracket is *exactly linear* in the temperature vector.
That is the seam this package is built on: :mod:`rhino_cal_jax.reflection`
produces the couplings, :mod:`rhino_cal_jax.switching` gathers them onto the
time axis through the Dicke switch, and :mod:`rhino_cal_jax.power` contracts
them with the temperatures.
"""

from rhino_cal_jax.errors import RhinoCalError, ValidationError

# Read by hatchling via [tool.hatch.version]; keep it the single source.
__version__ = "0.1.0"

__all__ = ["RhinoCalError", "ValidationError"]
```

- [ ] **Step 4: Install and run the tests**

Run:

```bash
cd /Users/zzhang/projects/rhino-cal && pip install -e . && python -m pytest tests/test_import.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add pyproject.toml rhino_cal_jax tests
git commit -m "feat: scaffold the rhino_cal_jax package

Packages only rhino_cal_jax; the numpy pipeline in simulation/, gcr/ and
utils/ keeps its existing run-from-checkout workflow untouched."
```

---

## Task 2: Reflection couplings (draft Eqs. 2–6)

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/reflection.py`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_reflection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reflection.py`:

```python
"""Draft Eqs. 2-6: the four source-dependent coupling spectra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.reflection import Couplings, couplings, reflection_factor


class TestReflectionFactor:
    def test_matched_receiver_is_unity(self):
        """Gamma_rec = 0 makes F = 1 exactly (draft Eq. 3)."""
        f = reflection_factor(jnp.array([0.3 + 0.1j]), jnp.array([0.0 + 0.0j]))
        assert f == pytest.approx(1.0 + 0.0j)

    def test_matches_the_closed_form(self):
        g_src = jnp.array([0.2 - 0.1j, 0.0 + 0.4j])
        g_rec = jnp.array([0.05 + 0.02j, -0.1 + 0.0j])
        expected = np.sqrt(1 - np.abs(np.asarray(g_rec)) ** 2) / (
            1 - np.asarray(g_rec) * np.asarray(g_src)
        )
        np.testing.assert_allclose(np.asarray(reflection_factor(g_src, g_rec)), expected, rtol=1e-14)


class TestCouplings:
    def test_a_matched_source_kills_every_noise_wave_coupling(self):
        """Gamma_src = 0: all three noise-wave couplings vanish.

        c_src does NOT become 1 here. It keeps the double-mismatch factor
        (1 - |Gamma_rec|^2) -- the textbook power-transfer efficiency -- because
        a mismatched receiver still reflects |Gamma_rec|^2 straight back out.
        Asserting 1.0 with a mismatched receiver would be wrong physics, and
        the repo's own numpy reference returns 0.99 for this input.
        """
        gamma_rec = jnp.full(3, 0.1 + 0.0j)
        c = couplings(jnp.zeros(3, dtype=complex), gamma_rec)
        np.testing.assert_allclose(
            np.asarray(c.c_src), 1.0 - np.abs(np.asarray(gamma_rec)) ** 2, rtol=1e-14
        )
        np.testing.assert_allclose(np.asarray(c.k_unc), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(c.k_cos), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(c.k_sin), 0.0, atol=1e-15)

    def test_a_fully_matched_pair_transfers_everything(self):
        """Both ports matched: F = 1, c_src = 1, nothing reflected anywhere."""
        c = couplings(jnp.zeros(3, dtype=complex), jnp.zeros(3, dtype=complex))
        np.testing.assert_allclose(np.asarray(c.c_src), 1.0, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(c.stacked[:, 1:]), 0.0, atol=1e-15)

    def test_k_unc_carries_the_square_of_F(self):
        """D1: draft Eq. 4 prints |Gamma|^2 |F|, the model is |Gamma|^2 |F|^2.

        Pinned because the two differ by |F|, which is 1 only for a matched
        receiver -- exactly the case a careless test would pick.
        """
        g_src, g_rec = jnp.array([0.4 + 0.2j]), jnp.array([0.3 - 0.15j])
        f = reflection_factor(g_src, g_rec)
        c = couplings(g_src, g_rec)
        np.testing.assert_allclose(
            np.asarray(c.k_unc),
            np.abs(np.asarray(g_src)) ** 2 * np.abs(np.asarray(f)) ** 2,
            rtol=1e-14,
        )
        assert not np.allclose(np.asarray(c.k_unc), np.abs(np.asarray(g_src)) ** 2 * np.abs(np.asarray(f)))

    def test_cos_and_sin_are_the_real_and_imaginary_parts_of_gamma_F(self):
        g_src, g_rec = jnp.array([0.4 + 0.2j]), jnp.array([0.3 - 0.15j])
        prod = np.asarray(g_src) * np.asarray(reflection_factor(g_src, g_rec))
        c = couplings(g_src, g_rec)
        np.testing.assert_allclose(np.asarray(c.k_cos), prod.real, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(c.k_sin), prod.imag, rtol=1e-14)

    def test_stacked_orders_the_columns_as_documented(self):
        c = couplings(jnp.array([0.2 + 0.1j]), jnp.array([0.05 + 0.0j]))
        stacked = c.stacked
        assert stacked.shape == (1, 4)
        np.testing.assert_allclose(np.asarray(stacked[:, 0]), np.asarray(c.c_src))
        np.testing.assert_allclose(np.asarray(stacked[:, 1]), np.asarray(c.k_unc))
        np.testing.assert_allclose(np.asarray(stacked[:, 2]), np.asarray(c.k_cos))
        np.testing.assert_allclose(np.asarray(stacked[:, 3]), np.asarray(c.k_sin))

    def test_broadcasts_a_source_axis_against_a_shared_receiver(self):
        g_src = jnp.array([[0.2 + 0.1j, 0.3 + 0.0j], [0.0 + 0.0j, -0.1 + 0.2j]])  # (2 src, 2 freq)
        g_rec = jnp.array([0.05 + 0.0j, 0.06 - 0.01j])  # (2 freq)
        c = couplings(g_src, g_rec)
        assert c.c_src.shape == (2, 2)
        assert c.stacked.shape == (2, 2, 4)


class TestRejections:
    def test_a_real_gamma_is_refused(self):
        """A real Gamma silently zeroes k_sin -- finite, right-shaped, wrong."""
        with pytest.raises(ValidationError, match="complex"):
            couplings(jnp.array([0.2, 0.3]), jnp.array([0.05 + 0.0j, 0.0 + 0.0j]))

    def test_a_real_gamma_rec_is_refused(self):
        with pytest.raises(ValidationError, match="complex"):
            couplings(jnp.array([0.2 + 0.0j]), jnp.array([0.05]))

    def test_a_length_one_gamma_rec_is_refused_rather_than_broadcast(self):
        """The silent-broadcast trap: one receiver value applied to every channel.

        NumPy semantics would happily stretch a (1,) gamma_rec across three
        channels and return a finite, correctly-shaped, wrong Couplings.
        """
        with pytest.raises(ValidationError, match="channels"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j, -0.1 + 0.2j]),
                      jnp.array([0.05 + 0.0j]))

    def test_a_scalar_gamma_rec_is_refused(self):
        with pytest.raises(ValidationError, match="1D"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j]), jnp.array(0.05 + 0.0j))

    def test_a_gamma_rec_longer_than_the_band_is_refused(self):
        with pytest.raises(ValidationError, match="channels"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j]),
                      jnp.full(5, 0.05 + 0.0j))


class TestStackedRoundTrip:
    """`stacked` and `from_stacked` must be exact inverses.

    A transposed column index at a distant call site would give a Couplings
    that is finite, correctly shaped and wrong, so the order is defined once
    and this pins the round trip.
    """

    def test_from_stacked_inverts_stacked(self):
        c = couplings(jnp.array([0.25 + 0.1j, 0.2 - 0.05j]),
                      jnp.array([0.08 - 0.03j, 0.07 + 0.01j]))
        back = Couplings.from_stacked(c.stacked)
        for field in ("c_src", "k_unc", "k_cos", "k_sin"):
            np.testing.assert_array_equal(
                np.asarray(getattr(back, field)), np.asarray(getattr(c, field))
            )

    def test_from_stacked_reads_the_columns_in_the_documented_order(self):
        """Column identity, not just shape -- a transposition must be visible."""
        stacked = jnp.asarray(
            np.arange(8.0).reshape(2, 4)  # row f: [4f+0, 4f+1, 4f+2, 4f+3]
        )
        c = Couplings.from_stacked(stacked)
        np.testing.assert_array_equal(np.asarray(c.c_src), [0.0, 4.0])
        np.testing.assert_array_equal(np.asarray(c.k_unc), [1.0, 5.0])
        np.testing.assert_array_equal(np.asarray(c.k_cos), [2.0, 6.0])
        np.testing.assert_array_equal(np.asarray(c.k_sin), [3.0, 7.0])

    def test_from_stacked_refuses_a_wrong_trailing_axis(self):
        with pytest.raises(ValidationError, match="4"):
            Couplings.from_stacked(jnp.zeros((2, 3)))

    def test_mismatched_field_shapes_are_refused_at_construction(self):
        """Fail at construction naming the field, not later inside `stacked`."""
        with pytest.raises(ValidationError, match="shape"):
            Couplings(c_src=jnp.zeros(3), k_unc=jnp.zeros(5),
                      k_cos=jnp.zeros(3), k_sin=jnp.zeros(3))


class TestBoundaries:
    """Extreme reflection coefficients: failures must be loud, never finite-wrong."""

    # The source phase is OFFSET from the receiver phase, not conjugate to it.
    # With g_src = m e^{-i phi} against g_rec = m' e^{+i phi} the product
    # Gamma_src Gamma_rec = m m' e^{i(phi - phi)} is always REAL, so the whole
    # sweep would run with a real denominator and a real F -- 64 cells that
    # never touch the complex half of the function, and blind to any swapped
    # real/imaginary term inside reflection_factor. The offset makes
    # max |Im F| ~ 0.56 across the grid instead of ~1e-13.
    PHASE_OFFSET = np.pi / 4

    @pytest.mark.parametrize("mag_rec", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("mag_src", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("phase", [0.0, np.pi / 3, np.pi, -np.pi / 2])
    def test_physical_reflections_stay_finite(self, mag_rec, mag_src, phase):
        g_rec = jnp.array([mag_rec * np.exp(1j * phase)])
        g_src = jnp.array([mag_src * np.exp(1j * (phase + self.PHASE_OFFSET))])
        c = couplings(g_src, g_rec)
        assert np.all(np.isfinite(np.asarray(c.stacked)))

    @pytest.mark.parametrize("mag_rec", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("mag_src", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("phase", [0.0, np.pi / 3, np.pi, -np.pi / 2])
    def test_the_exact_identities_hold_across_the_grid(self, mag_rec, mag_src, phase):
        """Two algebraic identities the four couplings must satisfy exactly.

        ``c_src + k_unc == |F|^2`` (the |Gamma|^2 terms cancel) and
        ``k_cos^2 + k_sin^2 == k_unc`` (both are |Gamma F|^2). They are cheap,
        they hold to round-off, and either would have caught a dropped power
        like the one the draft's Eq. 4 contains -- which is exactly the class
        of mistake this module had to be checked for.
        """
        g_rec = jnp.array([mag_rec * np.exp(1j * phase)])
        g_src = jnp.array([mag_src * np.exp(1j * (phase + self.PHASE_OFFSET))])
        c = couplings(g_src, g_rec)
        abs2_f = np.abs(np.asarray(reflection_factor(g_src, g_rec))) ** 2
        np.testing.assert_allclose(
            np.asarray(c.c_src) + np.asarray(c.k_unc), abs2_f, rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            np.asarray(c.k_cos) ** 2 + np.asarray(c.k_sin) ** 2,
            np.asarray(c.k_unc), rtol=1e-13, atol=1e-15,
        )

    def test_an_overunity_receiver_gives_nan_not_a_plausible_number(self):
        """|Gamma_rec| > 1 is unphysical; sqrt of a negative must not be silently real."""
        c = couplings(jnp.array([0.2 + 0.0j]), jnp.array([1.5 + 0.0j]))
        assert np.all(np.isnan(np.asarray(c.stacked)))

    def test_the_resonance_pole_is_not_finite(self):
        """Gamma_src * Gamma_rec -> 1 is a genuine pole of Eq. 3.

        Asserted as "not finite" rather than "Inf" on purpose: complex IEEE
        arithmetic turns the pole into NaN, not Inf. At this input the numerator
        vanishes too (|Gamma_rec| = 1), giving 0/0; and even for a clean pole
        with |Gamma_rec| < 1, taking the modulus of an infinite complex value
        evaluates 0*inf and yields NaN. Plain numpy agrees. Either way it is
        loud, which is all this test needs to establish.
        """
        c = couplings(jnp.array([1.0 + 0.0j]), jnp.array([1.0 + 0.0j]))
        assert not np.all(np.isfinite(np.asarray(c.stacked)))


class TestTransforms:
    def test_gradients_flow_to_both_reflection_coefficients(self):
        def loss(re_src, im_src, re_rec, im_rec):
            c = couplings(re_src + 1j * im_src, re_rec + 1j * im_rec)
            return jnp.sum(c.stacked)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            jnp.array([0.2]), jnp.array([0.1]), jnp.array([0.05]), jnp.array([0.02])
        )
        for g in grads:
            assert np.all(np.isfinite(np.asarray(g)))
            assert not np.allclose(np.asarray(g), 0.0)

    def test_jit_and_vmap_round_trip(self):
        g_src = jnp.array([[0.2 + 0.1j], [0.3 - 0.2j]])
        g_rec = jnp.array([0.05 + 0.0j])
        direct = couplings(g_src, g_rec).stacked
        mapped = jax.jit(jax.vmap(couplings, in_axes=(0, None)))(g_src, g_rec).stacked
        np.testing.assert_allclose(np.asarray(direct), np.asarray(mapped), rtol=1e-14)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_reflection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax.reflection'`

- [ ] **Step 3: Write minimal implementation**

Create `rhino_cal_jax/reflection.py`:

```python
"""Draft Eqs. 2-6: the coupling spectra a source imposes on the receiver.

These four quantities are what multiply the temperatures in Eq. 1. They depend
on the reflection coefficients alone -- no temperature enters -- which is why
they can be built once per source and reused for every time sample, and why
the resulting array is directly the design matrix of the linear system the
GCR solves.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError


def _require_complex(name: str, value: jax.Array) -> jax.Array:
    """Reject a real reflection coefficient.

    A real ``Gamma`` makes ``Im(Gamma F)`` identically zero, so ``k_sin``
    silently vanishes and ``T_sin`` drops out of the model: a finite,
    correctly-shaped, wrong answer. Refuse it at the door.
    """
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValidationError(
            f"{name} must be complex (got dtype {value.dtype}). A real reflection "
            "coefficient silently zeroes the sine coupling; pass e.g. "
            f"{name} + 0j if it really is purely real."
        )
    return value


def _require_matching_channels(gamma_src: jax.Array, gamma_rec: jax.Array) -> None:
    """Reject a ``gamma_rec`` whose channel axis does not match ``gamma_src``.

    NumPy broadcasting would happily stretch a length-1 or scalar ``gamma_rec``
    across every channel, applying one receiver reflection to the whole band and
    returning a finite, correctly-shaped, wrong result. Shape-only, so it stays
    safe under ``jit``.
    """
    if gamma_rec.ndim != 1:
        raise ValidationError(
            f"gamma_rec must be 1D (n_freq,), got shape {gamma_rec.shape}."
        )
    n_src = None if gamma_src.ndim == 0 else gamma_src.shape[-1]
    if n_src != gamma_rec.shape[0]:
        raise ValidationError(
            f"gamma_src has {n_src} channels but gamma_rec has "
            f"{gamma_rec.shape[0]}. A mismatched gamma_rec would broadcast "
            "silently and apply one receiver reflection to every channel."
        )


def reflection_factor(gamma_src: jax.Array, gamma_rec: jax.Array) -> jax.Array:
    """``F = sqrt(1 - |Gamma_rec|^2) / (1 - Gamma_src Gamma_rec)`` (draft Eq. 3).

    Args:
        gamma_src: complex source reflection coefficient, ``(..., n_freq)``.
        gamma_rec: complex receiver reflection coefficient, ``(n_freq,)``.

    Returns:
        Complex ``F``, broadcast to the shape of ``gamma_src``.

    Note:
        ``|Gamma_rec| > 1`` (an active receiver) yields NaN, and the resonance
        ``Gamma_src Gamma_rec -> 1`` also yields NaN rather than Inf, because
        complex IEEE arithmetic turns an infinite component into NaN once the
        modulus evaluates ``0 * inf``. Both are left loud on purpose.
    """
    gamma_src = _require_complex("gamma_src", gamma_src)
    gamma_rec = _require_complex("gamma_rec", gamma_rec)
    _require_matching_channels(gamma_src, gamma_rec)
    return jnp.sqrt(1.0 - jnp.abs(gamma_rec) ** 2) / (1.0 - gamma_src * gamma_rec)


class Couplings(eqx.Module):
    """The four coupling spectra of draft Eq. 1, for one or many sources.

    Attributes:
        c_src: ``(1 - |Gamma|^2) |F|^2`` -- the source term (Eq. 2).
        k_unc: ``|Gamma|^2 |F|^2`` -- the uncorrelated noise wave (Eq. 4).
        k_cos: ``Re(Gamma F)`` -- the in-phase noise wave (Eq. 5).
        k_sin: ``Im(Gamma F)`` -- the quadrature noise wave (Eq. 6).

    All four share the shape of ``gamma_src``: ``(n_freq,)`` for a single
    source, ``(n_source, n_freq)`` for a switched set.
    """

    c_src: jax.Array = eqx.field(converter=jnp.asarray)
    k_unc: jax.Array = eqx.field(converter=jnp.asarray)
    k_cos: jax.Array = eqx.field(converter=jnp.asarray)
    k_sin: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        # Without this, mismatched fields construct happily and only fail later
        # inside `stacked`, with a message that names no field.
        shapes = {
            "c_src": self.c_src.shape, "k_unc": self.k_unc.shape,
            "k_cos": self.k_cos.shape, "k_sin": self.k_sin.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValidationError(
                f"Couplings fields must all share one shape, got {shapes}."
            )

    @property
    def stacked(self) -> jax.Array:
        """``(..., n_freq, 4)`` in the order ``(c_src, k_unc, k_cos, k_sin)``.

        This is the design matrix: contracting it with the temperature vector
        ``(T_src, T_unc, T_cos, T_sin)`` reproduces every term of Eq. 1 except
        the receiver offset ``T_rx``, which has no coupling of its own.

        Deliberately not cached: it is a pure ``jnp.stack`` over four leaves
        already in hand, so XLA folds it away on the jitted path this module is
        built for, while a cached attribute would complicate pytree flattening
        of a frozen Module for no real saving.
        """
        return jnp.stack([self.c_src, self.k_unc, self.k_cos, self.k_sin], axis=-1)

    @classmethod
    def from_stacked(cls, stacked: jax.Array) -> "Couplings":
        """Inverse of :attr:`stacked`: split a ``(..., n_freq, 4)`` array apart.

        Exists so that no call site ever hand-unpacks the column indices. A
        transposed index somewhere downstream would yield a ``Couplings`` that
        is finite, correctly shaped and wrong; defining the order in exactly one
        place is what prevents it.

        Args:
            stacked: ``(..., n_freq, 4)`` ordered as :attr:`stacked` produces.

        Returns:
            The corresponding :class:`Couplings`.

        Raises:
            ValidationError: if the trailing axis is not the four couplings.
        """
        stacked = jnp.asarray(stacked)
        if stacked.ndim < 2 or stacked.shape[-1] != 4:
            raise ValidationError(
                "from_stacked expects a trailing axis of 4 couplings, got shape "
                f"{stacked.shape}."
            )
        return cls(
            c_src=stacked[..., 0], k_unc=stacked[..., 1],
            k_cos=stacked[..., 2], k_sin=stacked[..., 3],
        )


def couplings(gamma_src: jax.Array, gamma_rec: jax.Array) -> Couplings:
    """Build the coupling spectra of draft Eqs. 2-6.

    Args:
        gamma_src: complex source reflection coefficient. ``(n_freq,)`` for one
            source, or ``(n_source, n_freq)`` for a switched set.
        gamma_rec: complex receiver reflection coefficient, ``(n_freq,)``,
            broadcast across the source axis.

    Returns:
        A :class:`Couplings` whose fields share the shape of ``gamma_src``.

    Raises:
        ValidationError: if either coefficient has a real dtype.
    """
    f = reflection_factor(gamma_src, gamma_rec)
    gamma_src = jnp.asarray(gamma_src)
    abs2_src = jnp.abs(gamma_src) ** 2
    abs2_f = jnp.abs(f) ** 2
    product = gamma_src * f
    return Couplings(
        c_src=(1.0 - abs2_src) * abs2_f,
        # D1: the draft's Eq. 4 prints a single |F|; Eq. 2 shows both squares,
        # the numpy reference squares it, and so does the transfer matrix in
        # gcr/transfer_matrix_construction.py. |F|^2 it is.
        k_unc=abs2_src * abs2_f,
        k_cos=jnp.real(product),
        k_sin=jnp.imag(product),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_reflection.py -q`
Expected: 77 passed (the boundary sweep is parametrized 4×4×4 = 64 of them).

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/reflection.py tests/test_reflection.py
git commit -m "feat: coupling spectra of the noise-wave data model (Eqs. 2-6)

Refuses a real-dtype Gamma: it would silently zero the sine coupling and
drop T_sin from the model. Pins |Gamma|^2 |F|^2 for the uncorrelated term
against the draft's Eq. 4, which prints only one square."
```

---

## Task 3: Consistency of the coupling core against the numpy reference

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_consistency_with_numpy.py`

This task exists on its own because the cross-check is the deliverable the user asked for, not a by-product. It compares against `simulation/radiometer_power.py`, which is the module the whole numpy pipeline calls.

- [ ] **Step 1: Write the failing test**

Create `tests/test_consistency_with_numpy.py`:

```python
"""Cross-check rhino_cal_jax against the numpy implementation in simulation/.

The reference is `simulation.radiometer_power.compute_radiometer_power`, the
function every numpy notebook in this repository ultimately calls. Agreement is
demanded at float64 round-off, over a grid that includes the extremes, because
the failure mode that matters here is a *finite, correctly-shaped, wrong*
number -- the kind a spot check at one nice parameter value cannot see.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from simulation.radiometer_power import compute_radiometer_power  # noqa: E402

from rhino_cal_jax.power import radiometer_power, system_temperature  # noqa: E402
from rhino_cal_jax.reflection import couplings  # noqa: E402

# Extremes on purpose: a matched source (Gamma = 0) zeroes three of the four
# couplings, and |Gamma| -> 1 is where F is most sensitive to the receiver.
MAGS = (0.0, 0.05, 0.5, 0.95)
PHASES = (0.0, np.pi / 3, np.pi, -2.0)
TEMPS = (
    (300.0, 250.0, 30.0, -40.0, 290.0),   # a realistic set
    (0.0, 0.0, 0.0, 0.0, 0.0),            # everything off
    (1200.0, 0.0, 0.0, 0.0, 0.0),         # noise-diode-only
    (10.0, 5000.0, -900.0, 900.0, 1.0),   # noise waves dominating
)


def _numpy_power(t, g_src, g_rec, gain):
    t_src, t_unc, t_cos, t_sin, t_rx = t
    return compute_radiometer_power(
        t_src=t_src, t_unc=t_unc, t_sin=t_sin, t_cos=t_cos, t_0=t_rx,
        gamma_rec=g_rec, gamma_src=g_src, gain=gain, add_noise=False,
    )


def _jax_power(t, g_src, g_rec, gain):
    t_src, t_unc, t_cos, t_sin, t_rx = t
    coup = couplings(jnp.asarray(g_src), jnp.asarray(g_rec))
    t_sys = system_temperature(
        coup, t_src=t_src, t_unc=t_unc, t_cos=t_cos, t_sin=t_sin, t_rx=t_rx
    )
    return np.asarray(radiometer_power(t_sys, gain=jnp.asarray(gain)))


@pytest.mark.parametrize("temps", TEMPS)
@pytest.mark.parametrize("mag_src,phase_src", list(itertools.product(MAGS, PHASES)))
@pytest.mark.parametrize("mag_rec", MAGS)
def test_eq1_matches_the_numpy_reference(temps, mag_src, phase_src, mag_rec):
    g_src = np.array([mag_src * np.exp(1j * phase_src)])
    g_rec = np.array([mag_rec * np.exp(-0.7j)])
    gain = np.array([1000.0])

    reference = _numpy_power(temps, g_src, g_rec, gain)
    ours = _jax_power(temps, g_src, g_rec, gain)

    assert np.all(np.isfinite(reference)) == np.all(np.isfinite(ours))
    scale = max(abs(float(reference[0])), 1.0)
    assert abs(float(ours[0]) - float(reference[0])) / scale < 1e-13


def test_agreement_holds_across_a_frequency_band():
    """A per-channel Gamma, which is how the real S11 measurements arrive."""
    rng = np.random.default_rng(0)
    n_freq = 64
    g_src = 0.3 * np.exp(1j * np.linspace(0, 6.0, n_freq)) * rng.uniform(0.5, 1.0, n_freq)
    g_rec = 0.12 * np.exp(-1j * np.linspace(0, 2.0, n_freq))
    gain = np.linspace(900.0, 1100.0, n_freq)
    temps = (300.0, 250.0, 30.0, -40.0, 290.0)

    reference = _numpy_power(temps, g_src, g_rec, gain)
    ours = _jax_power(temps, g_src, g_rec, gain)
    np.testing.assert_allclose(ours, reference, rtol=1e-13)


def test_agreement_with_per_channel_temperatures():
    """T_unc/T_cos/T_sin are smooth functions of frequency, not scalars."""
    n_freq = 32
    nu = np.linspace(-1.0, 1.0, n_freq)
    temps = (
        250.0 + 20.0 * nu,
        240.0 - 15.0 * nu**2,
        30.0 * nu,
        -40.0 + 5.0 * nu,
        290.0 + nu,
    )
    g_src = np.full(n_freq, 0.25 + 0.1j)
    g_rec = np.full(n_freq, 0.08 - 0.03j)
    gain = np.full(n_freq, 1000.0)

    np.testing.assert_allclose(
        _jax_power(temps, g_src, g_rec, gain),
        _numpy_power(temps, g_src, g_rec, gain),
        rtol=1e-13,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_consistency_with_numpy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax.power'`

- [ ] **Step 3: Implement `power.py` (minimum to satisfy this task)**

Create `rhino_cal_jax/power.py`:

```python
"""Draft Eqs. 1, 8 and 11: system temperature, recorded power, radiometer noise.

The bracket of Eq. 1 is exactly linear in the temperature vector
``(T_src, T_unc, T_cos, T_sin, T_rx)``. That is not an approximation to be
checked at run time -- it is the structure the GCR sampler is built on -- so
:func:`system_temperature` is written as a contraction against the coupling
array rather than as five hand-written products.
"""

import jax
import jax.numpy as jnp

from rhino_cal_jax.reflection import Couplings


def system_temperature(
    coup: Couplings,
    *,
    t_src: jax.Array | float,
    t_unc: jax.Array | float,
    t_cos: jax.Array | float,
    t_sin: jax.Array | float,
    t_rx: jax.Array | float,
) -> jax.Array:
    """``T_sys`` -- the bracket of draft Eq. 1 (equivalently Eq. 11).

    Args:
        coup: coupling spectra for the connected source. Fields are
            ``(n_freq,)``, or ``(n_time, n_freq)`` once gathered through a
            :class:`~rhino_cal_jax.switching.SwitchCycle`.
        t_src: source noise temperature [K].
        t_unc: uncorrelated noise-wave temperature [K].
        t_cos: in-phase noise-wave temperature [K].
        t_sin: quadrature noise-wave temperature [K].
        t_rx: receiver offset temperature [K] (the draft's ``T_rx``; the numpy
            reference calls it ``t_0``). It has no coupling -- it enters the
            bracket bare.

    Every temperature broadcasts against the coupling shape, so a scalar, a
    ``(n_freq,)`` spectrum and a ``(n_time, n_freq)`` field are all accepted.

    Returns:
        ``T_sys`` with the broadcast shape.
    """
    return (
        t_src * coup.c_src
        + t_unc * coup.k_unc
        + t_cos * coup.k_cos
        + t_sin * coup.k_sin
        + jnp.asarray(t_rx)
    )


def radiometer_power(t_sys: jax.Array, gain: jax.Array | float) -> jax.Array:
    """``d = G T_sys`` -- draft Eq. 1 without the noise term.

    Args:
        t_sys: system temperature [K], from :func:`system_temperature`.
        gain: ``G(nu, t)`` [power per kelvin]; scalar, ``(n_freq,)``, or
            ``(n_time, n_freq)``.

    Returns:
        Recorded spectral power, broadcast to the joint shape.
    """
    return jnp.asarray(gain) * t_sys
```

- [ ] **Step 4: Run the consistency suite**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_consistency_with_numpy.py -q`
Expected: 258 passed (`4 temps × 16 (mag,phase) × 4 mag_rec` = 256, plus the two band tests).

If any cell fails, do **not** loosen the tolerance. Print the offending
`(temps, gamma_src, gamma_rec)` and find which of the five terms disagrees by
evaluating each product separately — a disagreement here means one of the two
implementations has the physics wrong, which is the entire point of the check.

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/power.py tests/test_consistency_with_numpy.py
git commit -m "test: cross-check Eq. 1 against the numpy reference

256 parameter cells at float64 round-off, spanning matched to near-total
reflection on both ports and four temperature regimes, plus per-channel
Gamma and per-channel temperature bands."
```

---

## Task 4: Reflection-coefficient construction (`loads.py`)

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/loads.py`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_loads.py`

**Background the implementer needs.** The numpy repo builds `Γ` three different ways with three different cable conventions: `simulation/loads.py::Load` uses `phase = -2π L ν / c` per pass with **no** velocity factor and then squares `s21`; `simulation/receiver_simulation.py::calculate_cable_params` includes a velocity factor, `-2π L ν / (vf·c)`; and `TerminatedCable` in the same file writes the round trip in one go as `exp(-4jπ ν L ε / c)`. Implement **one** convention — the `Load` one, parametrized with a velocity factor that defaults to 1.0 so it reduces to `Load` exactly — and note the divergence in the docstring.

`Load` also contains `s21 * s21 * s11 / (1 - 0*s11)`; the divisor is identically 1. Do not reproduce the dead term.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loads.py`:

```python
"""Reflection-coefficient construction for calibration loads and the receiver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from rhino_cal_jax.errors import ValidationError  # noqa: E402
from rhino_cal_jax.loads import Load, Receiver, cable_gamma, termination_gamma  # noqa: E402

FREQ = np.linspace(60e6, 85e6, 16)


class TestTerminationGamma:
    def test_open_is_plus_one(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("open", 4)), 1.0 + 0j)

    def test_short_is_minus_one(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("short", 4)), -1.0 + 0j)

    def test_matched_is_zero(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("matched", 4)), 0.0 + 0j)

    def test_a_resistive_termination_uses_the_mismatch_formula(self):
        got = termination_gamma("resistive", 3, impedance=75.0, z0=50.0)
        np.testing.assert_allclose(np.asarray(got), (75.0 - 50.0) / (75.0 + 50.0) + 0j)

    def test_the_result_is_always_complex(self):
        """A real dtype here would be refused downstream by couplings()."""
        assert jnp.issubdtype(termination_gamma("open", 4).dtype, jnp.complexfloating)

    def test_an_unknown_termination_is_refused(self):
        with pytest.raises(ValidationError, match="termination"):
            termination_gamma("banana", 4)

    def test_resistive_without_an_impedance_is_refused(self):
        with pytest.raises(ValidationError, match="impedance"):
            termination_gamma("resistive", 4)


class TestCableGamma:
    def test_a_zero_length_cable_is_transparent(self):
        term = termination_gamma("open", FREQ.size)
        np.testing.assert_allclose(
            np.asarray(cable_gamma(term, FREQ, length=0.0)), np.asarray(term), rtol=1e-14
        )

    def test_the_round_trip_phase_matches_the_numpy_load_convention(self):
        """Reproduces simulation/loads.py: one-way phase, s21 applied twice."""
        term = termination_gamma("open", FREQ.size)
        length, loss = 3.0, 0.9
        expected = (loss * np.exp(-1j * 2 * np.pi * length * FREQ / 299792458.0)) ** 2 * np.asarray(term)
        np.testing.assert_allclose(
            np.asarray(cable_gamma(term, FREQ, length=length, loss=loss)), expected, rtol=1e-13
        )

    def test_a_lossy_cable_shrinks_the_magnitude_as_the_square_of_the_loss(self):
        term = termination_gamma("open", FREQ.size)
        lossless = np.abs(np.asarray(cable_gamma(term, FREQ, length=3.0, loss=1.0)))
        lossy = np.abs(np.asarray(cable_gamma(term, FREQ, length=3.0, loss=0.5)))
        np.testing.assert_allclose(lossy, 0.25 * lossless, rtol=1e-13)

    def test_the_velocity_factor_stretches_the_phase(self):
        term = termination_gamma("open", FREQ.size)
        slow = cable_gamma(term, FREQ, length=3.0, velocity_factor=0.66)
        fast = cable_gamma(term, FREQ, length=3.0 * 0.66, velocity_factor=1.0)
        np.testing.assert_allclose(np.asarray(slow), np.asarray(fast), rtol=1e-13)


class TestContainers:
    def test_a_load_keeps_its_label_static(self):
        load = Load(gamma_src=jnp.zeros(4, dtype=complex), t_src=jnp.array(300.0), label="ambient")
        leaves = jax.tree_util.tree_leaves(load)
        assert len(leaves) == 2  # gamma_src and t_src only; the label is static
        assert load.label == "ambient"

    def test_a_receiver_rejects_a_gamma_that_does_not_match_its_gain(self):
        with pytest.raises(ValidationError, match="n_freq"):
            Receiver(gamma_rec=jnp.zeros(4, dtype=complex), gain=jnp.ones(5))

    def test_a_receiver_accepts_a_time_dependent_gain(self):
        rx = Receiver(gamma_rec=jnp.zeros(4, dtype=complex), gain=jnp.ones((10, 4)))
        assert rx.gain.shape == (10, 4)

    def test_a_load_refuses_a_real_gamma(self):
        with pytest.raises(ValidationError, match="complex"):
            Load(gamma_src=jnp.zeros(4), t_src=jnp.array(300.0), label="ambient")

    def test_loads_are_differentiable_in_their_temperature(self):
        load = Load(gamma_src=jnp.zeros(4, dtype=complex), t_src=jnp.array(300.0), label="x")
        grad = jax.grad(lambda ld: jnp.sum(ld.t_src))(load)
        assert float(grad.t_src) == pytest.approx(1.0)


class TestConsistencyWithNumpyLoads:
    @pytest.mark.parametrize("kind", ["open", "short", "matched"])
    def test_termination_matches_the_numpy_load(self, kind):
        from simulation.loads import Load as NumpyLoad

        reference = NumpyLoad(
            physical_temperature=300.0, freqs=FREQ.copy(), termination_type=kind, label="ref"
        )
        ours = termination_gamma(kind, FREQ.size)
        np.testing.assert_allclose(
            np.asarray(ours), np.asarray(reference.gamma_src, dtype=complex), rtol=1e-14, atol=1e-16
        )

    def test_cabled_open_matches_the_numpy_load(self):
        from simulation.loads import Load as NumpyLoad

        reference = NumpyLoad(
            physical_temperature=300.0, freqs=FREQ.copy(), termination_type="open",
            effective_cable_length=2.5, cable_loss=0.95, label="ref",
        )
        ours = cable_gamma(
            termination_gamma("open", FREQ.size), FREQ, length=2.5, loss=0.95
        )
        np.testing.assert_allclose(
            np.asarray(ours), np.asarray(reference.gamma_src), rtol=1e-12, atol=1e-15
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_loads.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax.loads'`

- [ ] **Step 3: Write minimal implementation**

Create `rhino_cal_jax/loads.py`:

```python
"""Reflection coefficients for calibration loads and the receiver.

The numpy repository builds ``Gamma`` with three different cable conventions:
``simulation/loads.py::Load`` uses a one-way phase ``-2 pi L nu / c`` with no
velocity factor and squares ``s21``; ``receiver_simulation.calculate_cable_params``
divides that phase by a velocity factor; and ``TerminatedCable`` writes the
round trip directly as ``exp(-4j pi nu L eps / c)``. This module implements the
``Load`` convention with an explicit ``velocity_factor`` that defaults to 1.0,
so it reduces to ``Load`` exactly and reaches the other two by argument.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError

SPEED_OF_LIGHT: float = 299792458.0
"""Vacuum speed of light [m/s] -- matches astropy.constants.c to the metre."""

_TERMINATIONS = ("open", "short", "matched", "resistive")


def termination_gamma(
    kind: str,
    n_freq: int,
    *,
    impedance: float | None = None,
    z0: float = 50.0,
) -> jax.Array:
    """Reflection coefficient of an ideal termination, one value per channel.

    Args:
        kind: ``"open"`` (+1), ``"short"`` (-1), ``"matched"`` (0), or
            ``"resistive"`` (``(Z - Z0) / (Z + Z0)``).
        n_freq: number of channels; the value is constant across them.
        impedance: termination impedance [Ohm]; required for ``"resistive"``.
        z0: characteristic impedance [Ohm].

    Returns:
        A complex ``(n_freq,)`` array.

    Raises:
        ValidationError: on an unknown ``kind``, or ``"resistive"`` with no
            ``impedance``.
    """
    if kind not in _TERMINATIONS:
        raise ValidationError(
            f"Unknown termination {kind!r}; expected one of {_TERMINATIONS}."
        )
    if kind == "open":
        value = 1.0 + 0.0j
    elif kind == "short":
        value = -1.0 + 0.0j
    elif kind == "matched":
        value = 0.0 + 0.0j
    else:
        if impedance is None:
            raise ValidationError(
                "termination_gamma('resistive', ...) needs an impedance [Ohm]."
            )
        value = complex((impedance - z0) / (impedance + z0), 0.0)
    return jnp.full((n_freq,), value, dtype=complex)


def cable_gamma(
    gamma_termination: jax.Array,
    freq: jax.Array,
    *,
    length: float | jax.Array,
    velocity_factor: float | jax.Array = 1.0,
    loss: float | jax.Array = 1.0,
) -> jax.Array:
    """Move a termination behind a length of cable.

    The signal traverses the cable twice, so the one-way transmission
    ``s21 = loss * exp(-2j pi L nu / (vf c))`` is applied squared.

    Args:
        gamma_termination: complex ``(n_freq,)`` reflection at the far end.
        freq: channel frequencies [Hz], ``(n_freq,)``.
        length: physical cable length [m].
        velocity_factor: propagation velocity as a fraction of ``c``.
        loss: one-way amplitude transmission (1.0 = lossless).

    Returns:
        The complex ``(n_freq,)`` reflection seen at the near end.
    """
    phase = -2.0 * jnp.pi * length * jnp.asarray(freq) / (velocity_factor * SPEED_OF_LIGHT)
    s21 = loss * jnp.exp(1j * phase)
    return s21 * s21 * jnp.asarray(gamma_termination)


def _check_complex(name: str, value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValidationError(
            f"{name} must be complex (got dtype {value.dtype}); a real reflection "
            "coefficient silently zeroes the sine coupling."
        )
    return value


class Load(eqx.Module):
    """A source that can be switched to the receiver input.

    Covers the antenna and every calibration load alike -- from the model's
    point of view they differ only in ``gamma_src`` and ``t_src``.

    Attributes:
        gamma_src: complex ``(n_freq,)`` reflection coefficient.
        t_src: noise temperature [K]; scalar, ``(n_freq,)`` or ``(n_time, n_freq)``.
        label: identifier used by the switch cycle (static).
    """

    gamma_src: jax.Array = eqx.field(converter=jnp.asarray)
    t_src: jax.Array = eqx.field(converter=jnp.asarray)
    label: str = eqx.field(static=True)

    def __check_init__(self):
        _check_complex("gamma_src", self.gamma_src)
        if self.gamma_src.ndim != 1:
            raise ValidationError(
                f"gamma_src must be 1D (n_freq,), got ndim={self.gamma_src.ndim}."
            )


class Receiver(eqx.Module):
    """The receiver: its input reflection coefficient and its power gain.

    Attributes:
        gamma_rec: complex ``(n_freq,)`` receiver reflection coefficient.
        gain: ``G(nu)`` as ``(n_freq,)`` or ``G(nu, t)`` as ``(n_time, n_freq)``.
    """

    gamma_rec: jax.Array = eqx.field(converter=jnp.asarray)
    gain: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        _check_complex("gamma_rec", self.gamma_rec)
        if self.gamma_rec.ndim != 1:
            raise ValidationError(
                f"gamma_rec must be 1D (n_freq,), got ndim={self.gamma_rec.ndim}."
            )
        if self.gain.ndim not in (0, 1, 2):
            raise ValidationError(f"gain must be 0/1/2-D, got ndim={self.gain.ndim}.")
        if self.gain.ndim >= 1 and self.gain.shape[-1] != self.gamma_rec.shape[0]:
            raise ValidationError(
                f"gain has n_freq={self.gain.shape[-1]} but gamma_rec has "
                f"n_freq={self.gamma_rec.shape[0]}."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_loads.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/loads.py tests/test_loads.py
git commit -m "feat: reflection-coefficient construction for loads and receiver

One cable convention, parametrized by velocity factor so it reduces exactly
to simulation/loads.py at vf=1. Verified against that reference for open,
short, matched and cabled-open terminations."
```

---

## Task 5: The switch (draft Eqs. 11–12)

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/switching.py`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_switching.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_switching.py`:

```python
"""Draft Eqs. 11-12: which source is connected at each time sample."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from rhino_cal_jax.errors import ValidationError  # noqa: E402
from rhino_cal_jax.reflection import couplings  # noqa: E402
from rhino_cal_jax.switching import SwitchCycle  # noqa: E402


class TestConstruction:
    def test_from_labels_maps_names_to_indices(self):
        cycle = SwitchCycle.from_labels(
            ["antenna", "load", "antenna", "noise_diode"],
            labels=("antenna", "load", "noise_diode"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 1, 0, 2])

    def test_an_unknown_label_is_refused(self):
        with pytest.raises(ValidationError, match="banana"):
            SwitchCycle.from_labels(["antenna", "banana"], labels=("antenna", "load"))

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(ValidationError, match="out of range"):
            SwitchCycle(source_index=jnp.array([0, 3]), labels=("a", "b"))

    def test_a_negative_index_is_refused(self):
        with pytest.raises(ValidationError, match="out of range"):
            SwitchCycle(source_index=jnp.array([-1, 0]), labels=("a", "b"))

    def test_a_float_index_is_refused(self):
        """Float indices would round silently and mis-assign samples."""
        with pytest.raises(ValidationError, match="integer"):
            SwitchCycle(source_index=jnp.array([0.0, 1.0]), labels=("a", "b"))


class TestSchedule:
    def test_a_schedule_assigns_each_sample_to_the_last_change_before_it(self):
        cycle = SwitchCycle.from_schedule(
            times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            switch_times=np.array([0.0, 2.0, 3.5]),
            switch_labels=["antenna", "load", "antenna"],
            labels=("antenna", "load"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 0, 1, 1, 0])

    def test_samples_before_the_first_change_take_the_first_state(self):
        cycle = SwitchCycle.from_schedule(
            times=np.array([-1.0, 0.5]),
            switch_times=np.array([0.0, 1.0]),
            switch_labels=["antenna", "load"],
            labels=("antenna", "load"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 0])

    def test_it_matches_the_numpy_assign_states(self):
        from utils.utils import assign_states

        rng = np.random.default_rng(1)
        times = np.sort(rng.uniform(0.0, 100.0, 200))
        switch_times = np.arange(0.0, 100.0, 7.0)
        names = np.array(["antenna", "load", "noise_diode"])
        switch_labels = list(names[np.arange(switch_times.size) % 3])

        reference = assign_states(times, switch_times, np.array(switch_labels))
        cycle = SwitchCycle.from_schedule(
            times, switch_times, switch_labels, labels=tuple(names)
        )
        np.testing.assert_array_equal(
            names[np.asarray(cycle.source_index)], np.asarray(reference)
        )


class TestGather:
    def test_one_hot_is_a_permutation_matrix_of_the_index(self):
        cycle = SwitchCycle(source_index=jnp.array([2, 0, 1]), labels=("a", "b", "c"))
        np.testing.assert_array_equal(
            np.asarray(cycle.one_hot()),
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

    def test_gather_selects_the_connected_source_per_sample(self):
        per_source = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])  # (3 src, 2 freq)
        cycle = SwitchCycle(source_index=jnp.array([2, 0, 0, 1]), labels=("a", "b", "c"))
        np.testing.assert_array_equal(
            np.asarray(cycle.gather(per_source)),
            [[30.0, 31.0], [10.0, 11.0], [10.0, 11.0], [20.0, 21.0]],
        )

    def test_gather_rejects_a_mismatched_source_axis(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "b"))
        with pytest.raises(ValidationError, match="n_source"):
            cycle.gather(jnp.zeros((3, 4)))

    def test_gather_carries_the_trailing_column_axis_of_stacked_couplings(self):
        g_src = jnp.array([[0.2 + 0.1j, 0.1 + 0.0j], [0.0 + 0.0j, 0.3 - 0.1j]])
        g_rec = jnp.array([0.05 + 0.0j, 0.05 + 0.0j])
        stacked = couplings(g_src, g_rec).stacked  # (2 src, 2 freq, 4)
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 1]), labels=("a", "b"))
        assert cycle.gather(stacked).shape == (3, 2, 4)

    def test_gather_agrees_with_the_one_hot_contraction(self):
        """The index form and Eq. 12's theta must be the same operator."""
        per_source = jnp.arange(12.0).reshape(3, 4)
        cycle = SwitchCycle(source_index=jnp.array([1, 0, 2, 2]), labels=("a", "b", "c"))
        np.testing.assert_allclose(
            np.asarray(cycle.gather(per_source)),
            np.asarray(cycle.one_hot() @ per_source),
            rtol=1e-14,
        )


class TestTransforms:
    def test_gather_is_jittable(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "b"))
        per_source = jnp.array([[1.0], [2.0]])
        out = jax.jit(lambda c, p: c.gather(p))(cycle, per_source)
        np.testing.assert_allclose(np.asarray(out), [[1.0], [2.0]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_switching.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax.switching'`

- [ ] **Step 3: Write minimal implementation**

Create `rhino_cal_jax/switching.py`:

```python
"""Draft Eqs. 11-12: the Dicke switch that chooses the connected source.

Equation 12 defines ``theta(t_j, phi_h)`` as one-hot over sources. Storing it
densely would be an ``(n_time, n_source)`` matrix of mostly zeros, so this
module stores the index and offers ``one_hot()`` for the cases where the matrix
form is what a downstream formula wants. The two are tested against each other.

Why this class exists at all: each source has its *own* reflection coefficient,
so the couplings differ per sample -- and that difference is the only thing that
makes the noise-wave temperatures identifiable. Each switch position contributes
exactly one equation per frequency channel, so with per-channel temperatures the
design matrix has rank ``min(n_src, 3) * n_freq``: one load leaves it deficient
by a factor of three, and three loads make it square. Sharing a single ``Gamma``
across the switch collapses every source onto the same row.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rhino_cal_jax.errors import ValidationError


class SwitchCycle(eqx.Module):
    """Which source is connected at each time sample.

    Attributes:
        source_index: ``(n_time,)`` integer index into ``labels``.
        labels: source names, in the order the index refers to (static).
    """

    source_index: jax.Array = eqx.field(converter=jnp.asarray)
    labels: tuple[str, ...] = eqx.field(static=True, converter=tuple)

    def __check_init__(self):
        if not jnp.issubdtype(self.source_index.dtype, jnp.integer):
            raise ValidationError(
                f"source_index must be an integer array, got dtype "
                f"{self.source_index.dtype}. A float index would round silently "
                "and mis-assign samples to sources."
            )
        if self.source_index.ndim != 1:
            raise ValidationError(
                f"source_index must be 1D (n_time,), got ndim={self.source_index.ndim}."
            )
        # Range check needs concrete values, so it is skipped under tracing.
        # An out-of-range index would otherwise be clamped silently by JAX's
        # gather semantics and quietly assign samples to the wrong source.
        try:
            as_np = np.asarray(self.source_index)
        except (jax.errors.TracerArrayConversionError, TypeError):  # pragma: no cover
            return
        if as_np.size and (as_np.min() < 0 or as_np.max() >= len(self.labels)):
            raise ValidationError(
                f"source_index values span [{as_np.min()}, {as_np.max()}] which is "
                f"out of range for {len(self.labels)} labels {self.labels}."
            )

    @property
    def n_source(self) -> int:
        """Number of distinct sources in the cycle."""
        return len(self.labels)

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return int(self.source_index.shape[0])

    @classmethod
    def from_labels(cls, per_sample: list[str], *, labels: tuple[str, ...]) -> "SwitchCycle":
        """Build from an explicit label per time sample.

        Args:
            per_sample: ``(n_time,)`` source label for each sample.
            labels: the source ordering to index against.

        Raises:
            ValidationError: if a sample carries a label not in ``labels``.
        """
        lookup = {name: i for i, name in enumerate(labels)}
        unknown = sorted({s for s in per_sample if s not in lookup})
        if unknown:
            raise ValidationError(
                f"Sample labels {unknown} are not among the declared sources {labels}."
            )
        return cls(source_index=jnp.asarray([lookup[s] for s in per_sample]), labels=labels)

    @classmethod
    def from_schedule(
        cls,
        times,
        switch_times,
        switch_labels: list[str],
        *,
        labels: tuple[str, ...],
    ) -> "SwitchCycle":
        """Build from a switch schedule: each sample takes the last state at or before it.

        Samples earlier than the first switch take the first state, matching
        ``utils.utils.assign_states`` in the numpy pipeline.

        Args:
            times: ``(n_time,)`` sample times.
            switch_times: ``(n_change,)`` times at which the state changes.
            switch_labels: ``(n_change,)`` source label taking effect at each change.
            labels: the source ordering to index against.
        """
        idx = np.searchsorted(np.asarray(switch_times), np.asarray(times), side="right") - 1
        idx = np.clip(idx, 0, len(switch_labels) - 1)
        return cls.from_labels([switch_labels[i] for i in idx], labels=labels)

    def one_hot(self) -> jax.Array:
        """``theta(t_j, phi_h)`` of Eq. 12 as a dense ``(n_time, n_source)`` matrix."""
        return jax.nn.one_hot(self.source_index, self.n_source)

    def gather(self, per_source: jax.Array) -> jax.Array:
        """Select each sample's connected source from a per-source array.

        Args:
            per_source: ``(n_source, ...)`` -- typically ``(n_source, n_freq)``
                for one coupling, or ``(n_source, n_freq, 4)`` for
                :attr:`~rhino_cal_jax.reflection.Couplings.stacked`.

        Returns:
            ``(n_time, ...)`` with the leading axis replaced by time.

        Raises:
            ValidationError: if the leading axis is not ``n_source``.
        """
        per_source = jnp.asarray(per_source)
        if per_source.ndim == 0 or per_source.shape[0] != self.n_source:
            got = "scalar" if per_source.ndim == 0 else str(per_source.shape[0])
            raise ValidationError(
                f"gather expects a leading n_source={self.n_source} axis, got {got}."
            )
        return per_source[self.source_index]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_switching.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/switching.py tests/test_switching.py
git commit -m "feat: source switching (Eqs. 11-12)

Carries Gamma per source and gathers onto the time axis. Each switch
position contributes one equation per frequency channel, so per-channel
noise-wave temperatures need at least three distinct loads to be identified
at all; sharing one Gamma across the switch collapses every source onto the
same row. Verified against utils.assign_states."
```

---

## Task 6: Radiometer noise (draft Eq. 8) and the exact-linearity guarantee

**Files:**
- Modify: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/power.py` (append)
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_power.py`
- Modify: `/Users/zzhang/projects/rhino-cal/tests/test_consistency_with_numpy.py` (append)

- [ ] **Step 1: Write the failing test**

Create `tests/test_power.py`:

```python
"""Draft Eqs. 1 and 8: the recorded power and its radiometer noise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from rhino_cal_jax.power import (  # noqa: E402
    add_radiometer_noise,
    design_matrix,
    radiometer_power,
    system_temperature,
)
from rhino_cal_jax.reflection import couplings  # noqa: E402
from rhino_cal_jax.switching import SwitchCycle  # noqa: E402

G_SRC = jnp.array([0.25 + 0.1j, 0.2 - 0.05j])
G_REC = jnp.array([0.08 - 0.03j, 0.07 + 0.01j])
TEMPS = dict(t_src=300.0, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0)


class TestSystemTemperature:
    def test_a_matched_source_reduces_to_t_src_plus_t_rx(self):
        coup = couplings(jnp.zeros(2, dtype=complex), jnp.zeros(2, dtype=complex))
        t_sys = system_temperature(coup, **TEMPS)
        np.testing.assert_allclose(np.asarray(t_sys), 300.0 + 290.0, rtol=1e-14)

    def test_it_is_exactly_linear_in_the_temperature_vector(self):
        """Not approximately: the GCR is built on this being an identity."""
        coup = couplings(G_SRC, G_REC)
        a = dict(t_src=100.0, t_unc=50.0, t_cos=10.0, t_sin=-5.0, t_rx=200.0)
        b = dict(t_src=-30.0, t_unc=400.0, t_cos=-70.0, t_sin=60.0, t_rx=17.0)
        combined = {k: 2.5 * a[k] - 1.5 * b[k] for k in a}
        np.testing.assert_allclose(
            np.asarray(system_temperature(coup, **combined)),
            2.5 * np.asarray(system_temperature(coup, **a))
            - 1.5 * np.asarray(system_temperature(coup, **b)),
            rtol=1e-13,
        )

    def test_zero_temperatures_give_exactly_zero(self):
        coup = couplings(G_SRC, G_REC)
        zeros = dict.fromkeys(TEMPS, 0.0)
        np.testing.assert_array_equal(np.asarray(system_temperature(coup, **zeros)), 0.0)

    def test_per_channel_temperatures_broadcast(self):
        coup = couplings(G_SRC, G_REC)
        t_sys = system_temperature(
            coup, t_src=jnp.array([300.0, 310.0]), t_unc=250.0,
            t_cos=30.0, t_sin=-40.0, t_rx=290.0,
        )
        assert t_sys.shape == (2,)

    def test_gathered_couplings_give_a_time_frequency_field(self):
        g_src = jnp.stack([G_SRC, jnp.zeros(2, dtype=complex)])  # (2 src, 2 freq)
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 0]), labels=("ant", "load"))
        stacked = cycle.gather(couplings(g_src, G_REC).stacked)  # (3, 2, 4)
        t_sys = design_matrix(stacked) @ jnp.array([300.0, 250.0, 30.0, -40.0])
        assert t_sys.shape == (3 * 2,)


class TestDesignMatrix:
    def test_it_flattens_time_and_frequency_into_rows(self):
        stacked = jnp.zeros((5, 8, 4))
        assert design_matrix(stacked).shape == (40, 4)

    def test_its_product_reproduces_system_temperature(self):
        """The matrix form and the direct form are the same model."""
        coup = couplings(G_SRC, G_REC)
        vector = jnp.array([300.0, 250.0, 30.0, -40.0])
        direct = system_temperature(
            coup, t_src=300.0, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=0.0
        )
        np.testing.assert_allclose(
            np.asarray(design_matrix(coup.stacked) @ vector),
            np.asarray(direct).ravel(),
            rtol=1e-14,
        )

    def test_a_switched_pair_of_sources_gives_a_full_rank_block(self):
        """Two loads over two channels span all four coupling directions."""
        g_src = jnp.stack([
            jnp.array([0.30 + 0.10j, 0.28 - 0.02j]),
            jnp.array([0.02 + 0.00j, 0.01 + 0.03j]),
        ])
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 0, 1]), labels=("ant", "load"))
        matrix = design_matrix(cycle.gather(couplings(g_src, G_REC).stacked))
        assert np.linalg.matrix_rank(np.asarray(matrix)) == 4

    def test_one_load_cannot_span_the_four_coupling_directions(self):
        """One load contributes ONE row per channel, so rank <= n_freq.

        Two channels therefore cap the rank at 2, short of the four coupling
        directions -- no matter how many time samples are taken, because every
        sample repeats the same row. That row-counting argument, not any
        proportionality between the columns, is what switching fixes.
        """
        one_source = couplings(G_SRC, G_REC).stacked[None, ...]  # (1, 2 freq, 4)
        cycle = SwitchCycle(source_index=jnp.zeros(4, dtype=int), labels=("ant",))
        matrix = design_matrix(cycle.gather(one_source))
        assert np.linalg.matrix_rank(np.asarray(matrix)) == 2  # == n_freq, < 4


class TestRadiometerNoise:
    def test_the_fractional_scatter_matches_one_over_root_bt(self):
        """Draft Eq. 8: sigma_w = 1 / sqrt(delta_nu tau)."""
        power = jnp.full((20000,), 1000.0)
        noisy = add_radiometer_noise(
            power, jax.random.key(0), t_int=1.0, delta_nu=1e4
        )
        fractional = np.asarray(noisy) / 1000.0 - 1.0
        assert float(np.std(fractional)) == pytest.approx(1e-2, rel=0.05)
        assert abs(float(np.mean(fractional))) < 5e-4

    def test_the_noise_scales_with_the_power_itself(self):
        """It is multiplicative: doubling the power doubles the absolute scatter."""
        key = jax.random.key(1)
        small = add_radiometer_noise(jnp.full((20000,), 100.0), key, t_int=1.0, delta_nu=1e4)
        large = add_radiometer_noise(jnp.full((20000,), 200.0), key, t_int=1.0, delta_nu=1e4)
        ratio = float(np.std(np.asarray(large))) / float(np.std(np.asarray(small)))
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_a_longer_integration_narrows_the_scatter_as_the_square_root(self):
        key = jax.random.key(2)
        short = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=1.0, delta_nu=1e4)
        long = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=4.0, delta_nu=1e4)
        ratio = float(np.std(np.asarray(short))) / float(np.std(np.asarray(long)))
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_folding_is_off_by_default_and_biases_the_mean_when_on(self):
        """D3: the numpy reference takes |P + n|, which is not a noise model.

        At a deliberately low B*tau the fold is visible: it reflects the
        negative tail back and pushes the mean above the true power.
        """
        power = jnp.full((40000,), 1.0)
        kwargs = dict(t_int=1.0, delta_nu=1.0)  # sigma_w = 1, so the tail is large
        plain = add_radiometer_noise(power, jax.random.key(3), **kwargs)
        folded = add_radiometer_noise(power, jax.random.key(3), fold_negative=True, **kwargs)
        assert float(np.mean(np.asarray(plain))) == pytest.approx(1.0, abs=0.02)
        assert float(np.mean(np.asarray(folded))) > 1.05
        np.testing.assert_allclose(
            np.asarray(folded), np.abs(np.asarray(plain)), rtol=1e-14
        )

    def test_the_same_key_reproduces_the_same_draw(self):
        power = jnp.full((100,), 1000.0)
        a = add_radiometer_noise(power, jax.random.key(7), t_int=1.0, delta_nu=1e4)
        b = add_radiometer_noise(power, jax.random.key(7), t_int=1.0, delta_nu=1e4)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestTransforms:
    def test_gradients_reach_every_temperature(self):
        coup = couplings(G_SRC, G_REC)

        def loss(t_src, t_unc, t_cos, t_sin, t_rx):
            t_sys = system_temperature(
                coup, t_src=t_src, t_unc=t_unc, t_cos=t_cos, t_sin=t_sin, t_rx=t_rx
            )
            return jnp.sum(radiometer_power(t_sys, gain=1000.0))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(300.0, 250.0, 30.0, -40.0, 290.0)
        for g in grads:
            assert np.isfinite(float(g))
            assert abs(float(g)) > 0.0

    def test_the_whole_forward_model_jits(self):
        @jax.jit
        def forward(g_src, g_rec, t_src):
            coup = couplings(g_src, g_rec)
            t_sys = system_temperature(
                coup, t_src=t_src, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0
            )
            return radiometer_power(t_sys, gain=1000.0)

        out = forward(G_SRC, G_REC, 300.0)
        assert out.shape == (2,)
        assert np.all(np.isfinite(np.asarray(out)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_power.py -q`
Expected: FAIL — `ImportError: cannot import name 'add_radiometer_noise'`

- [ ] **Step 3: Append to `rhino_cal_jax/power.py`**

Add these imports at the top of the file (alongside the existing ones):

```python
from rhino_cal_jax.errors import ValidationError
```

Append:

```python
def design_matrix(stacked: jax.Array) -> jax.Array:
    """Flatten stacked couplings into the ``(n_row, 4)`` matrix of the linear model.

    Args:
        stacked: ``(..., n_freq, 4)`` from
            :attr:`~rhino_cal_jax.reflection.Couplings.stacked`, optionally
            gathered onto a time axis by
            :meth:`~rhino_cal_jax.switching.SwitchCycle.gather`.

    Returns:
        ``(prod(leading dims) * n_freq, 4)``. Multiplying by
        ``(T_src, T_unc, T_cos, T_sin)`` reproduces ``T_sys - T_rx`` flattened
        in C order, which is what the GCR of draft Eqs. 30-31 solves.

    Raises:
        ValidationError: if the trailing axis is not the four couplings.
    """
    stacked = jnp.asarray(stacked)
    if stacked.ndim < 2 or stacked.shape[-1] != 4:
        raise ValidationError(
            f"design_matrix expects a trailing axis of 4 couplings, got shape "
            f"{stacked.shape}."
        )
    return stacked.reshape(-1, 4)


def add_radiometer_noise(
    power: jax.Array,
    key: jax.Array,
    *,
    t_int: float | jax.Array,
    delta_nu: float | jax.Array,
    fold_negative: bool = False,
) -> jax.Array:
    """Apply the fractional radiometer noise of draft Eq. 8.

    ``d -> d (1 + w)`` with ``w ~ N(0, sigma_w)`` and
    ``sigma_w = 1 / sqrt(delta_nu * t_int)`` -- multiplicative, so the absolute
    scatter tracks the power.

    Args:
        power: noiseless power from :func:`radiometer_power`.
        key: a typed JAX PRNG key (``jax.random.key(seed)``).
        t_int: integration time per sample [s].
        delta_nu: channel bandwidth [Hz].
        fold_negative: reproduce the numpy reference's ``abs(P + n)``. **Off by
            default and left off in any scientific use** -- folding reflects the
            negative tail and biases the mean upward whenever
            ``delta_nu * t_int`` is not large, which silently breaks the
            Gaussian likelihood the GCR assumes. It exists so the consistency
            suite can reproduce the reference bit for bit.

    Returns:
        The noisy power, same shape as ``power``.

    Note:
        Draft Eq. 1 writes the noise as an additive ``n_w`` inside the bracket
        while Eq. 8 writes it as fractional; the two agree only for
        ``n_w = T_sys w``. Eq. 8 is what the radiometer equation means and what
        the numpy reference implements, so it is what this function does.
    """
    sigma_w = 1.0 / jnp.sqrt(jnp.asarray(delta_nu) * jnp.asarray(t_int))
    noisy = power * (1.0 + sigma_w * jax.random.normal(key, power.shape, power.dtype))
    return jnp.abs(noisy) if fold_negative else noisy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_power.py -q`
Expected: all pass.

- [ ] **Step 5: Extend the consistency suite with the noise check**

Append to `tests/test_consistency_with_numpy.py`:

```python
def test_noise_scale_matches_the_numpy_reference_statistically():
    """Different RNGs, so compare distributions rather than draws.

    Folding is enabled here purely to match the reference; see D3 in the plan
    for why it is off by default everywhere else.
    """
    import jax

    from rhino_cal_jax.power import add_radiometer_noise

    n = 40000
    clean = 1000.0
    t_int, delta_nu = 1.0, 1e4

    np.random.seed(0)
    reference = compute_radiometer_power(
        t_src=300.0, t_unc=250.0, t_sin=-40.0, t_cos=30.0, t_0=290.0,
        gamma_rec=np.full(n, 0.0 + 0j), gamma_src=np.full(n, 0.0 + 0j),
        gain=clean / (300.0 + 290.0), add_noise=True, t_int=t_int, delta_nu=delta_nu,
    )
    ours = add_radiometer_noise(
        jnp.full((n,), clean), jax.random.key(0),
        t_int=t_int, delta_nu=delta_nu, fold_negative=True,
    )

    # Both are ~N(clean, clean/sqrt(B tau)); 40k draws pins the std to ~0.5%.
    assert float(np.std(np.asarray(ours))) == pytest.approx(
        float(np.std(reference)), rel=0.03
    )
    assert float(np.mean(np.asarray(ours))) == pytest.approx(
        float(np.mean(reference)), rel=1e-3
    )
```

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/power.py tests/test_power.py tests/test_consistency_with_numpy.py
git commit -m "feat: radiometer noise (Eq. 8) and the design-matrix export

Noise is fractional, matching Eq. 8 and the numpy reference; the reference's
abs(P+n) fold is reachable behind fold_negative= but off by default, because
it biases the mean and breaks the Gaussian likelihood the GCR assumes.

design_matrix() makes the linear structure explicit, and two tests pin the
identifiability claim: a switched pair of sources spans rank 4, a single
shared Gamma does not."
```

---

## Task 7: Source sky model

**Files:**
- Create: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/sky.py`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_sky.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sky.py`:

```python
"""The synchrotron power law used as a stand-in antenna temperature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from rhino_cal_jax.sky import synchrotron_temperature  # noqa: E402

FREQ = np.linspace(60e6, 85e6, 16)


def test_it_returns_the_amplitude_at_the_reference_frequency():
    got = synchrotron_temperature(jnp.array([210e6]), t_ref=180.0, beta=-2.6)
    assert float(got[0]) == pytest.approx(180.0, rel=1e-12)


def test_it_follows_the_declared_power_law():
    got = np.asarray(synchrotron_temperature(jnp.asarray(FREQ), t_ref=180.0, beta=-2.6))
    np.testing.assert_allclose(got, 180.0 * (FREQ / 210e6) ** -2.6, rtol=1e-13)


def test_a_steeper_index_gives_more_signal_below_the_reference():
    at_70 = jnp.array([70e6])
    steep = float(synchrotron_temperature(at_70, beta=-2.8)[0])
    shallow = float(synchrotron_temperature(at_70, beta=-2.4)[0])
    assert steep > shallow


def test_it_is_differentiable_in_both_parameters():
    grads = jax.grad(
        lambda t_ref, beta: jnp.sum(synchrotron_temperature(jnp.asarray(FREQ), t_ref=t_ref, beta=beta)),
        argnums=(0, 1),
    )(180.0, -2.6)
    for g in grads:
        assert np.isfinite(float(g))
        assert abs(float(g)) > 0.0


def test_it_matches_the_numpy_reference():
    from simulation.toy_sky import synchrotron_temperatures

    reference = synchrotron_temperatures(FREQ.copy(), T_210=180.0, beta=-2.6)
    ours = synchrotron_temperature(jnp.asarray(FREQ), t_ref=180.0, beta=-2.6)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(reference), rtol=1e-13)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_sky.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rhino_cal_jax.sky'`

- [ ] **Step 3: Write minimal implementation**

Create `rhino_cal_jax/sky.py`:

```python
"""A synchrotron power law standing in for the antenna temperature.

This is not a sky model in any serious sense -- it is the smooth, bright,
spectrally simple thing you point the calibration machinery at while testing
it. Real sky models belong upstream (rheplicant's sky engines, limTOD).
"""

import jax
import jax.numpy as jnp


def synchrotron_temperature(
    freq: jax.Array,
    *,
    t_ref: float | jax.Array = 180.0,
    beta: float | jax.Array = -2.6,
    freq_ref: float | jax.Array = 210e6,
) -> jax.Array:
    """``T(nu) = t_ref (nu / freq_ref) ** beta``.

    Args:
        freq: channel frequencies [Hz].
        t_ref: brightness temperature at ``freq_ref`` [K].
        beta: spectral index (negative for synchrotron).
        freq_ref: reference frequency [Hz].

    Returns:
        Brightness temperature [K], shaped like ``freq``. Differentiable in
        ``t_ref`` and ``beta``, which is what makes it usable as a fitted
        foreground rather than a fixed backdrop.
    """
    return t_ref * jnp.power(jnp.asarray(freq) / freq_ref, beta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_sky.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/sky.py tests/test_sky.py
git commit -m "feat: differentiable synchrotron power law

Matches simulation/toy_sky.py, but differentiable in amplitude and spectral
index so it can be fitted rather than only evaluated."
```

---

## Task 8: Public API, README, and push

**Files:**
- Modify: `/Users/zzhang/projects/rhino-cal/rhino_cal_jax/__init__.py`
- Modify: `/Users/zzhang/projects/rhino-cal/README.md`
- Create: `/Users/zzhang/projects/rhino-cal/tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_public_api.py`:

```python
"""Everything a user needs is reachable from the package root."""

import rhino_cal_jax


def test_the_documented_surface_is_exported():
    expected = {
        "Couplings", "Load", "Receiver", "RhinoCalError", "SwitchCycle",
        "ValidationError", "add_radiometer_noise", "cable_gamma",
        "couplings", "design_matrix", "radiometer_power", "reflection_factor",
        "synchrotron_temperature", "system_temperature", "termination_gamma",
    }
    assert set(rhino_cal_jax.__all__) == expected
    # __version__ is public but not part of the star-import surface.
    assert rhino_cal_jax.__version__


def test_every_exported_name_actually_resolves():
    for name in rhino_cal_jax.__all__:
        assert getattr(rhino_cal_jax, name) is not None


def test_an_end_to_end_switched_observation_runs():
    """The smallest complete use: two sources, a switch, noise."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    freq = jnp.linspace(60e6, 85e6, 8)
    antenna = rhino_cal_jax.Load(
        gamma_src=rhino_cal_jax.cable_gamma(
            rhino_cal_jax.termination_gamma("open", 8), freq, length=2.0, loss=0.9
        ),
        t_src=rhino_cal_jax.synchrotron_temperature(freq),
        label="antenna",
    )
    load = rhino_cal_jax.Load(
        gamma_src=rhino_cal_jax.termination_gamma("matched", 8),
        t_src=jnp.array(300.0),
        label="load",
    )
    receiver = rhino_cal_jax.Receiver(
        gamma_rec=rhino_cal_jax.termination_gamma("resistive", 8, impedance=45.0),
        gain=jnp.full(8, 1000.0),
    )
    cycle = rhino_cal_jax.SwitchCycle.from_labels(
        ["antenna", "load"] * 6, labels=("antenna", "load")
    )

    per_source = jnp.stack([antenna.gamma_src, load.gamma_src])
    coup = rhino_cal_jax.Couplings.from_stacked(
        cycle.gather(rhino_cal_jax.couplings(per_source, receiver.gamma_rec).stacked)
    )
    t_src = cycle.gather(jnp.stack([jnp.broadcast_to(antenna.t_src, (8,)),
                                    jnp.broadcast_to(load.t_src, (8,))]))
    t_sys = rhino_cal_jax.system_temperature(
        coup, t_src=t_src, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0
    )
    power = rhino_cal_jax.add_radiometer_noise(
        rhino_cal_jax.radiometer_power(t_sys, receiver.gain),
        jax.random.key(0), t_int=1.0, delta_nu=1e4,
    )
    assert power.shape == (12, 8)
    assert np.all(np.isfinite(np.asarray(power)))
    assert np.all(np.asarray(power) > 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/test_public_api.py -q`
Expected: FAIL — `AttributeError: module 'rhino_cal_jax' has no attribute 'Couplings'`

- [ ] **Step 3: Write the public API**

Replace the import/`__all__` block of `rhino_cal_jax/__init__.py` (keep the module docstring already written in Task 1) with:

```python
from rhino_cal_jax.errors import RhinoCalError, ValidationError
from rhino_cal_jax.loads import Load, Receiver, cable_gamma, termination_gamma
from rhino_cal_jax.power import (
    add_radiometer_noise,
    design_matrix,
    radiometer_power,
    system_temperature,
)
from rhino_cal_jax.reflection import Couplings, couplings, reflection_factor
from rhino_cal_jax.sky import synchrotron_temperature
from rhino_cal_jax.switching import SwitchCycle

# Read by hatchling via [tool.hatch.version]; keep it the single source.
__version__ = "0.1.0"

__all__ = [
    "Couplings",
    "Load",
    "Receiver",
    "RhinoCalError",
    "SwitchCycle",
    "ValidationError",
    "add_radiometer_noise",
    "cable_gamma",
    "couplings",
    "design_matrix",
    "radiometer_power",
    "reflection_factor",
    "synchrotron_temperature",
    "system_temperature",
    "termination_gamma",
]
```

- [ ] **Step 4: Run the whole suite and the linter**

Run:

```bash
cd /Users/zzhang/projects/rhino-cal && python -m pytest tests/ -q && ruff check rhino_cal_jax tests
```

Expected: all tests pass, ruff reports no issues.

- [ ] **Step 5: Document it in the README**

Append to `README.md`:

```markdown
## `rhino_cal_jax` — the differentiable data model

A JAX/Equinox implementation of the noise-wave data model (Eq. 1 of the
Noise-Wave GCR note), independent of the numpy pipeline above and verified
against it channel by channel.

```bash
pip install -e .
pytest tests/
```

Where it differs from `simulation/`, deliberately:

| | `simulation/` | `rhino_cal_jax` |
|---|---|---|
| `Γ` per switched source | one shared value | one per source, gathered by the switch |
| noisy power | `abs(P + n)` | `P (1 + w)`; the fold is opt-in via `fold_negative=` |
| cable phase | three conventions across three modules | one, with an explicit `velocity_factor` |

The first row is the one that matters scientifically. Each switch position
contributes one equation per frequency channel, so with per-channel noise-wave
temperatures the design matrix has rank `min(n_src, 3) × n_freq`: one load is
never enough, and three is the minimum that makes the system square. A single
shared `Γ` collapses every source onto the same row and forfeits that entirely.
```

- [ ] **Step 6: Commit and push**

```bash
cd /Users/zzhang/projects/rhino-cal
git add rhino_cal_jax/__init__.py tests/test_public_api.py README.md
git commit -m "feat: public API for rhino_cal_jax, with an end-to-end smoke test"
git push origin main
```

> **Stop here and report to the user before pushing.** This is a shared
> repository with another author's work in it; confirm the push is wanted.

---

## Task 9: rheplicant adapter — a real `NoiseWaveOperator`

**Files:**
- Modify: `/Users/zzhang/projects/e-RHINO/src/rheplicant/radio/instrument/noise_wave.py` (full rewrite)
- Create: `/Users/zzhang/projects/e-RHINO/tests/radio/test_noise_wave.py`

**Design.** The operator sits at the `noise_wave` graph node, downstream of the
`receiver_input` selector — so `state.data` already carries the *selected*
source's `T_src`. What the selector throws away is *which* source that was, and
that is exactly what the couplings need. The operator therefore reads the same
switch array the selector used, `coords.extra["receiver_input"]`, and carries
`Γ` per source.

`Γ` is stored as separate real leaves (`gamma_src_re` / `gamma_src_im`) rather
than one complex leaf: rheplicant's `fisher_information` runs `jax.jacfwd` over
the latents, which does not accept complex parameters. The operator combines
them into a complex array before calling `rhino_cal_jax`.

- [ ] **Step 1: Write the failing test**

Create `tests/radio/test_noise_wave.py`:

```python
"""NoiseWaveOperator: draft Eq. 1 as a rheplicant operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")

from rheplicant.core.coordinates import Coordinates  # noqa: E402
from rheplicant.core.errors import StateValidationError  # noqa: E402
from rheplicant.core.state import State  # noqa: E402
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

N_TIME, N_FREQ = 12, 4

# Three physically distinct loads, in the order the switch indexes them:
# a structured antenna reflection, a near-matched ambient load, and a
# short-like reflection of opposite sign. Three is the minimum that makes the
# per-channel noise-wave system full rank -- see TestIdentifiability.
G_SRC_RE = np.array([[0.30, 0.28, 0.26, 0.24],
                     [0.02, 0.01, 0.00, -0.01],
                     [-0.60, -0.62, -0.64, -0.66]])
G_SRC_IM = np.array([[0.10, 0.05, 0.00, -0.05],
                     [0.00, 0.03, 0.02, 0.01],
                     [0.15, -0.10, 0.05, 0.20]])
N_SOURCE = G_SRC_RE.shape[0]
G_REC_RE = np.full(N_FREQ, 0.08)
G_REC_IM = np.full(N_FREQ, -0.03)

# rheplicant's suite runs float32; jax_enable_x64 is process-global and cannot
# be flipped per-module, so the tolerance is read from the active precision.
# The same file then passes under both `pytest` and `JAX_ENABLE_X64=1 pytest`.
RTOL = 1e-13 if jax.config.read("jax_enable_x64") else 2e-6


def make_operator(**overrides):
    kwargs = dict(
        t_unc=jnp.array(250.0), t_cos=jnp.array(30.0),
        t_sin=jnp.array(-40.0), t_rx=jnp.array(290.0),
        gamma_src_re=jnp.asarray(G_SRC_RE), gamma_src_im=jnp.asarray(G_SRC_IM),
        gamma_rec_re=jnp.asarray(G_REC_RE), gamma_rec_im=jnp.asarray(G_REC_IM),
    )
    kwargs.update(overrides)
    return NoiseWaveOperator(**kwargs)


def make_state(switch=None, data=None):
    extra = {} if switch is None else {"receiver_input": jnp.asarray(switch)}
    return State(
        data=jnp.full((N_TIME, N_FREQ), 300.0) if data is None else data,
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
            extra=extra,
        ),
    )


class TestForward:
    def test_it_reproduces_rhino_cal_jax_directly(self):
        import rhino_cal_jax as rcj

        switch = np.arange(N_TIME) % N_SOURCE
        out = make_operator()(make_state(switch)).data

        cycle = rcj.SwitchCycle(
            source_index=jnp.asarray(switch),
            labels=tuple(str(i) for i in range(N_SOURCE)),
        )
        stacked = cycle.gather(
            rcj.couplings(
                jnp.asarray(G_SRC_RE + 1j * G_SRC_IM), jnp.asarray(G_REC_RE + 1j * G_REC_IM)
            ).stacked
        )
        expected = (
            300.0 * stacked[..., 0] + 250.0 * stacked[..., 1]
            + 30.0 * stacked[..., 2] - 40.0 * stacked[..., 3] + 290.0
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=RTOL)

    def test_a_matched_single_source_reduces_to_t_src_plus_t_rx(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
            gamma_rec_re=jnp.zeros(N_FREQ), gamma_rec_im=jnp.zeros(N_FREQ),
        )
        out = op(make_state(np.zeros(N_TIME, dtype=int))).data
        np.testing.assert_allclose(np.asarray(out), 300.0 + 290.0, rtol=RTOL)

    def test_a_single_source_needs_no_switch_array(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
        )
        assert op(make_state()).data.shape == (N_TIME, N_FREQ)

    def test_per_channel_noise_wave_temperatures_are_accepted(self):
        op = make_operator(t_unc=jnp.linspace(240.0, 260.0, N_FREQ))
        assert op(make_state(np.arange(N_TIME) % N_SOURCE)).data.shape == (N_TIME, N_FREQ)


class TestRejections:
    def test_several_sources_without_a_switch_array_is_refused(self):
        """Silently using source 0 for every sample is finite and wrong."""
        with pytest.raises(StateValidationError, match="receiver_input"):
            make_operator()(make_state())

    def test_a_switch_longer_than_the_data_is_refused(self):
        with pytest.raises(StateValidationError, match="n_time"):
            make_operator()(make_state(np.zeros(N_TIME + 1, dtype=int)))

    def test_mismatched_gamma_real_and_imaginary_shapes_are_refused(self):
        with pytest.raises(StateValidationError, match="gamma_src"):
            make_operator(gamma_src_im=jnp.zeros((N_SOURCE + 1, N_FREQ)))

    def test_a_gamma_whose_channels_do_not_match_the_receiver_is_refused(self):
        with pytest.raises(StateValidationError, match="n_freq"):
            make_operator(gamma_rec_re=jnp.zeros(N_FREQ + 1), gamma_rec_im=jnp.zeros(N_FREQ + 1))


class TestIdentifiability:
    """What switching buys, counted per frequency channel.

    The noise-wave temperatures are functions of frequency and nothing ties
    channels together a priori, so the count that matters is per channel: each
    switch position contributes exactly ONE equation there, against the three
    unknowns living in it. Rank is therefore ``min(n_src, 3) * n_freq``.

    Note what this deliberately does NOT do: it does not use scalar,
    frequency-independent temperatures. Those are fully identified by a single
    load (frequency structure in Gamma supplies the equations, and the normal
    matrix comes out with a condition number around 6), so a scalar test cannot
    demonstrate anything about switching -- it passes with one source.
    """

    def _per_channel_jacobian(self, n_source: int) -> np.ndarray:
        """d(prediction) / d(per-channel noise-wave temperatures), (n_row, 3 * n_freq)."""
        switch = np.arange(N_TIME) % n_source

        def predict(flat):  # flat: (3, n_freq) -- t_unc, t_cos, t_sin per channel
            op = make_operator(
                t_unc=flat[0], t_cos=flat[1], t_sin=flat[2],
                gamma_src_re=jnp.asarray(G_SRC_RE[:n_source]),
                gamma_src_im=jnp.asarray(G_SRC_IM[:n_source]),
            )
            return op(make_state(switch)).data.ravel()

        jac = jax.jacobian(predict)(jnp.zeros((3, N_FREQ)))
        return np.asarray(jac).reshape(-1, 3 * N_FREQ)

    @pytest.mark.parametrize("n_source,expected_rank", [(1, N_FREQ), (2, 2 * N_FREQ)])
    def test_too_few_loads_leave_the_system_rank_deficient(self, n_source, expected_rank):
        jac = self._per_channel_jacobian(n_source)
        assert np.linalg.matrix_rank(jac) == expected_rank
        assert expected_rank < 3 * N_FREQ

    def test_three_loads_make_the_per_channel_system_full_rank(self):
        jac = self._per_channel_jacobian(3)
        assert np.linalg.matrix_rank(jac) == 3 * N_FREQ

    def test_the_operator_reproduces_its_own_truth(self):
        """Sanity: the forward model is deterministic and Gamma is wired per source."""
        switch = np.arange(N_TIME) % 2
        truth = make_operator()(make_state(switch)).data

        def predict(t_nw):
            return make_operator(t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2])(
                make_state(switch)
            ).data.ravel()

        np.testing.assert_allclose(
            np.asarray(predict(jnp.array([250.0, 30.0, -40.0]))),
            np.asarray(truth).ravel(), rtol=RTOL,
        )


class TestTransforms:
    def test_gradients_reach_every_temperature_and_gamma(self):
        switch = np.arange(N_TIME) % N_SOURCE
        op = make_operator()

        def loss(operator):
            return jnp.sum(operator(make_state(switch)).data)

        grads = jax.grad(loss)(op)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert np.all(np.isfinite(np.asarray(leaf)))

    def test_the_operator_jits(self):
        switch = np.arange(N_TIME) % N_SOURCE
        op = make_operator()
        out = jax.jit(lambda o, s: o(s).data)(op, make_state(switch))
        assert out.shape == (N_TIME, N_FREQ)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zzhang/projects/e-RHINO && python -m pytest tests/radio/test_noise_wave.py -q -p no:cacheprovider --no-cov`
Expected: FAIL — `TypeError: NoiseWaveOperator.__init__() got an unexpected keyword argument 't_rx'`

- [ ] **Step 3: Rewrite `src/rheplicant/radio/instrument/noise_wave.py`**

```python
"""NoiseWaveOperator — the receiver stage of the noise-wave data model.

Implements the bracket of the Noise-Wave GCR draft's Eq. 1::

    T_sys = T_src c_s + T_unc k_unc + T_cos k_cos + T_sin k_sin + T_rx

with the coupling spectra ``(c_s, k_unc, k_cos, k_sin)`` supplied by
``rhino_cal_jax`` (draft Eqs. 2-6). The physics lives in that package; this
module is the adapter that gives it a State -> State face and a home on the
signal graph.

Placement. This operator sits at the ``noise_wave`` node, downstream of the
``receiver_input`` selector, so ``state.data`` already carries the *selected*
source's ``T_src``. What the selector discards is which source that was — and
that is precisely what the couplings depend on. The operator therefore carries
``Gamma`` per source and re-reads the same switch array the selector used,
``coords.extra["receiver_input"]``.

That is not a convenience. Each switch position contributes exactly one equation
per frequency channel, so with per-channel noise-wave temperatures the design
matrix has rank ``min(n_src, 3) * n_freq``. One load leaves it deficient by a
factor of three; three distinct loads make it square. Sharing a single ``Gamma``
across the cycle collapses every source onto the same row and gives that up
entirely — the fit then returns a finite, correctly-shaped, wholly prior-driven
answer. (Frequency structure in ``Gamma`` *does* identify **scalar** noise-wave
temperatures from a single load; it is the per-channel case, which is the
physical one, that needs the switch.)

``Gamma`` is stored as two real leaves rather than one complex leaf because
:func:`~rheplicant.inference.uncertainty.fisher_information` runs
``jax.jacfwd``, which refuses complex parameters.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


def _rhino_cal_jax():
    """The rhino_cal_jax module, with an actionable message when it is absent."""
    try:
        import rhino_cal_jax
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "NoiseWaveOperator needs the rhino_cal_jax package: install it with "
            "pip install 'rhino-cal-jax @ "
            "git+https://github.com/RHINO-Experiment/rhino-cal.git'"
        ) from exc
    return rhino_cal_jax


class NoiseWaveOperator(AbstractOperator):
    """Apply reflection couplings and add the noise-wave temperatures.

    Attributes:
        t_unc: uncorrelated noise-wave temperature [K]; scalar or ``(n_freq,)``.
        t_cos: in-phase noise-wave temperature [K].
        t_sin: quadrature noise-wave temperature [K].
        t_rx: receiver offset temperature [K] (the draft's ``T_rx``).
        gamma_src_re: ``(n_source, n_freq)`` real part of each source's ``Gamma``.
        gamma_src_im: ``(n_source, n_freq)`` imaginary part.
        gamma_rec_re: ``(n_freq,)`` real part of the receiver's ``Gamma``.
        gamma_rec_im: ``(n_freq,)`` imaginary part.
        switch_key: key in ``coords.extra`` holding the per-sample source index.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.extra")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "noise_wave"

    t_unc: jax.Array = eqx.field(converter=jnp.asarray)
    t_cos: jax.Array = eqx.field(converter=jnp.asarray)
    t_sin: jax.Array = eqx.field(converter=jnp.asarray)
    t_rx: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_src_re: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_src_im: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_rec_re: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_rec_im: jax.Array = eqx.field(converter=jnp.asarray)
    switch_key: str = eqx.field(static=True, default="receiver_input")

    def __check_init__(self):
        if self.gamma_src_re.ndim != 2:
            raise StateValidationError(
                f"gamma_src_re must be 2D (n_source, n_freq), got "
                f"ndim={self.gamma_src_re.ndim}."
            )
        if self.gamma_src_re.shape != self.gamma_src_im.shape:
            raise StateValidationError(
                f"gamma_src real/imaginary parts disagree: {self.gamma_src_re.shape} "
                f"vs {self.gamma_src_im.shape}."
            )
        if self.gamma_rec_re.shape != self.gamma_rec_im.shape:
            raise StateValidationError(
                f"gamma_rec real/imaginary parts disagree: {self.gamma_rec_re.shape} "
                f"vs {self.gamma_rec_im.shape}."
            )
        if self.gamma_rec_re.ndim != 1:
            raise StateValidationError(
                f"gamma_rec_re must be 1D (n_freq,), got ndim={self.gamma_rec_re.ndim}."
            )
        if self.gamma_src_re.shape[1] != self.gamma_rec_re.shape[0]:
            raise StateValidationError(
                f"gamma_src has n_freq={self.gamma_src_re.shape[1]} but gamma_rec "
                f"has n_freq={self.gamma_rec_re.shape[0]}."
            )

    @property
    def n_source(self) -> int:
        """Number of switchable sources this operator carries a ``Gamma`` for."""
        return int(self.gamma_src_re.shape[0])

    def _source_index(self, state: State) -> jax.Array:
        """The per-sample source index, or an all-zeros index for a single source."""
        n_time = state.data.shape[0]
        extra = {} if state.coords is None else state.coords.extra
        if self.switch_key not in extra:
            if self.n_source == 1:
                return jnp.zeros((n_time,), dtype=int)
            raise StateValidationError(
                f"NoiseWaveOperator carries {self.n_source} sources but "
                f"coords.extra[{self.switch_key!r}] is absent, so there is no way "
                "to tell which one is connected. Defaulting to the first would "
                "return a finite, wrong answer."
            )
        index = jnp.asarray(extra[self.switch_key])
        if index.ndim != 1 or index.shape[0] != n_time:
            raise StateValidationError(
                f"coords.extra[{self.switch_key!r}] must be (n_time,) = ({n_time},), "
                f"got shape {index.shape}."
            )
        return index.astype(int)

    def __call__(self, state: State) -> State:
        rcj = _rhino_cal_jax()
        if state.data is None or jnp.asarray(state.data).ndim != 2:
            raise StateValidationError(
                "NoiseWaveOperator expects (n_time, n_freq) data; got "
                f"{None if state.data is None else jnp.asarray(state.data).shape}."
            )
        if state.data.shape[1] != self.gamma_rec_re.shape[0]:
            raise StateValidationError(
                f"data has n_freq={state.data.shape[1]} but gamma_rec has "
                f"n_freq={self.gamma_rec_re.shape[0]}."
            )

        index = self._source_index(state)
        cycle = rcj.SwitchCycle(
            source_index=index, labels=tuple(str(i) for i in range(self.n_source))
        )
        coup = rcj.Couplings.from_stacked(
            cycle.gather(
                rcj.couplings(
                    self.gamma_src_re + 1j * self.gamma_src_im,
                    self.gamma_rec_re + 1j * self.gamma_rec_im,
                ).stacked
            )
        )
        return state.with_data(
            rcj.system_temperature(
                coup, t_src=state.data, t_unc=self.t_unc,
                t_cos=self.t_cos, t_sin=self.t_sin, t_rx=self.t_rx,
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/zzhang/projects/e-RHINO && python -m pytest tests/radio/test_noise_wave.py -q --no-cov
```

Expected: all pass.

- [ ] **Step 5: Run the full rheplicant suite to catch fallout**

Run: `cd /Users/zzhang/projects/e-RHINO && python -m pytest -q`

The old placeholder had a `t_zero` field and a scalar `gamma_re`/`gamma_im`
pair. Any test or example constructing it that way now fails. Update each call
site to the new signature; do not add back-compat shims — the class was
documented as a placeholder and the package is 0.1.x.

- [ ] **Step 6: Commit**

```bash
cd /Users/zzhang/projects/e-RHINO
git add src/rheplicant/radio/instrument/noise_wave.py tests/radio/test_noise_wave.py
git commit -m "feat: real NoiseWaveOperator backed by rhino_cal_jax

Replaces the F -> 1 placeholder with draft Eq. 1 proper. Gamma is now carried
per source and gathered through the receiver_input switch. Each switch position
gives one equation per channel, so per-channel noise-wave temperatures have rank
min(n_src, 3) * n_freq: one load is deficient threefold, three loads make it
square. Tests pin the rank at one, two and three loads.

BREAKING: t_zero is now t_rx, and gamma_re/gamma_im are now per-source
(n_source, n_freq) arrays split into gamma_src_* and gamma_rec_*."
```

---

## Task 10: The payoff — noise waves as a linear block

**Files:**
- Create: `/Users/zzhang/projects/e-RHINO/examples/noise_wave_gcr.py`

This is where the design earns its keep: because Eq. 1 is exactly linear in the
noise-wave temperatures, `ParameterSpace(linear=True)` applies, `check_linearity`
verifies the claim rather than trusting it, and `wiener_solve` / `gcr_sample`
give the draft's Eqs. 30–31 with no new machinery.

- [ ] **Step 1: Write the example**

Create `examples/noise_wave_gcr.py`:

```python
"""Noise waves as a linear block: Wiener mean and exact posterior draws.

The data model (Noise-Wave GCR draft, Eq. 1) is exactly linear in the
noise-wave temperatures, so they never need a gradient sampler. This script
declares that linearity, has it CHECKED, then solves in closed form and draws
exact posterior samples -- the draft's Eqs. 30 and 31 respectively.

The switching is what makes it work. Run it with `--one-source` to see the same
solve on data taken through a single load: the couplings become proportional
and the posterior blows up.

Run:  uv run python examples/noise_wave_gcr.py
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

N_TIME, N_FREQ = 96, 16
T_RX, T_SRC, NOISE, PRIOR = 290.0, 300.0, 0.5, 100.0

parser = argparse.ArgumentParser()
parser.add_argument("--one-source", action="store_true",
                    help="use a single load, to show what switching buys")
args = parser.parse_args()

freq = jnp.linspace(60e6, 85e6, N_FREQ)
gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)

# The noise waves are inferred PER CHANNEL -- 3 x N_FREQ unknowns -- which is
# the physical case and the one that needs switching. Each switch position
# contributes one equation per channel, so three distinct loads are the minimum
# that makes the system square. (Scalar, frequency-independent noise waves
# would be identified by a single load, so they could not demonstrate this.)
TRUE_T = jnp.stack([
    250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),      # T_unc(nu)
    30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),      # T_cos(nu)
    -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,  # T_sin(nu)
])

# Three loads whose reflection coefficients differ in shape, not just in level:
# an antenna behind a cable (structured, resonant), a near-matched ambient load,
# and a short (large, opposite sign).
antenna = rcj.cable_gamma(
    rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
)
ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=52.0)
short = rcj.cable_gamma(
    rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
)

if args.one_source:
    gamma_src, n_source = jnp.stack([antenna]), 1
else:
    gamma_src, n_source = jnp.stack([antenna, ambient, short]), 3
switch = jnp.arange(N_TIME) % n_source

state = State(
    data=jnp.full((N_TIME, N_FREQ), T_SRC),
    coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=freq,
                       extra={"receiver_input": switch}),
    meta={"telescope": "RHINO"},
)


def twin(t_nw):
    return Pipeline(
        NoiseWaveOperator(
            t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=jnp.array(T_RX),
            gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
            gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
        ),
        names=("noise_wave",),
    )


truth = twin(TRUE_T)(state).data
observed = truth + NOISE * jax.random.normal(jax.random.key(0), truth.shape)
print(f"observation: {observed.shape}, {float(observed.mean()):.2f} K mean")
print(f"loads: {n_source}   unknowns: {TRUE_T.size} (3 x {N_FREQ} channels)")
print(f"equations per channel: {n_source}   -> expected rank "
      f"{min(n_source, 3) * N_FREQ}/{3 * N_FREQ}\n")

# All three spectra are ONE latent of shape (3, n_freq) feeding three leaves.
# Declaring linear=True is a claim; check_linearity is what turns it into a fact.
space = ParameterSpace(
    latents=[Latent("t_nw", init=jnp.zeros((3, N_FREQ)), linear=True)],
    bindings=[
        Bind("t_nw", into=(lambda p: p["noise_wave"].t_unc,
                           lambda p: p["noise_wave"].t_cos,
                           lambda p: p["noise_wave"].t_sin),
             fn=lambda v: (v[0], v[1], v[2])),
    ],
)
start = twin(jnp.zeros((3, N_FREQ)))
errors = check_linearity(space, start, state)
print(f"linearity check: worst relative departure {max(errors.values()):.1e}")

block = linear_operator(space, start, state)
solved, residual = wiener_solve(block, observed, noise_std=NOISE, prior_std=PRIOR)
print(f"\nWiener mean (Eq. 30), CG residual {float(residual):.1e}")
print("   band-averaged recovery, per spectrum:")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    err = jnp.sqrt(jnp.mean((solved[i] - TRUE_T[i]) ** 2))
    print(f"   {name:6s} RMS error {float(err):8.3f} K   "
          f"(truth spans {float(TRUE_T[i].min()):7.1f} .. {float(TRUE_T[i].max()):7.1f})")

# gcr_sample adds the two fluctuation terms of Eq. 31 to the same right-hand
# side, so every solve is an independent, exact posterior draw.
keys = jax.random.split(jax.random.key(9), 300)
draws = jax.vmap(
    lambda k: gcr_sample(block, observed, noise_std=NOISE, prior_std=PRIOR, key=k)[0]
)(keys)
print("\n300 exact posterior draws (Eq. 31): per-channel sigma, and how much of")
print(f"the {PRIOR:.0f} K prior width the data actually bought back:")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    sig = draws[:, i].std(axis=0)
    print(f"   {name:6s} sigma {float(sig.min()):7.3f} .. {float(sig.max()):7.3f} K"
          f"   ({100 * float(sig.mean()) / PRIOR:5.1f}% of prior)")

if args.one_source:
    print(f"\nOne load: {N_FREQ} equations per channel-triple against "
          f"{3 * N_FREQ} unknowns.")
    print("Two of every three directions are unconstrained by the data, so the")
    print("prior sets their width -- the sigmas sit near the prior, and the RMS")
    print("errors are large. Re-run without --one-source to see it close.")
else:
    print("\nThree loads with genuinely different Gamma: the per-channel system")
    print("is square, so every direction is constrained by data and the sigmas")
    print("drop far below the prior. That is what the switch buys.")
```

- [ ] **Step 2: Run it both ways**

Run:

```bash
cd /Users/zzhang/projects/e-RHINO && python examples/noise_wave_gcr.py && python examples/noise_wave_gcr.py --one-source
```

Expected: the three-load run recovers all three spectra with RMS errors small
against their own dynamic range, and per-channel `sigma` well below the 100 K
prior. The `--one-source` run should show `sigma` close to the prior for most
channels and visibly larger RMS errors.

**Do not print a claim the run does not support.** Measure first, assert second:
run it, read the numbers, and if they differ from the closing narrative, change
the narrative — not the other way round. Two specific things to check rather
than assume:
- That the one-load case really is worse. If it is *not*, the per-channel
  binding is not doing what it should (most likely the latent collapsed to a
  scalar somewhere), and that is a bug to find, not a result to report.
- That `check_linearity` passes in both configurations. The model is exactly
  linear in these temperatures, so a non-trivial departure means the operator
  is not doing what Task 9 claims.

- [ ] **Step 3: Commit**

```bash
cd /Users/zzhang/projects/e-RHINO
git add examples/noise_wave_gcr.py
git commit -m "docs: worked example -- noise waves as a checked linear block

Declares the three noise-wave temperatures linear, verifies the claim with
check_linearity, then solves Eq. 30 in closed form and draws Eq. 31 exactly.
--one-source shows the same solve without switching, where the couplings are
proportional and the prior carries the posterior."
```

---

## Task 11: Documentation and packaging

**Files:**
- Modify: `/Users/zzhang/projects/e-RHINO/pyproject.toml`
- Modify: `/Users/zzhang/projects/e-RHINO/CHANGELOG.md`
- Modify: `/Users/zzhang/projects/e-RHINO/DESIGN.md`
- Modify: `/Users/zzhang/projects/e-RHINO/docs/operators.md`

- [ ] **Step 1: Add the optional dependency**

In `pyproject.toml`, alongside the existing `limtod` extra:

```toml
# The noise-wave data model (Eq. 1 of the GCR note) lives in the rhino-cal-jax
# distribution, not here -- NoiseWaveOperator is an adapter over it. jax and
# equinox are required dependencies of that distribution, so there is no extra
# to select. Not yet on PyPI; until it is, install it directly:
#   pip install 'rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal.git'
rhinocal = ["rhino-cal-jax>=0.1"]
```

> **Check before committing:** if `rhino-cal` is still absent from PyPI,
> `pip install rheplicant[rhinocal]` will fail with a resolver error rather than
> a helpful message. If it is not published by the time this lands, delete the
> extra and keep only the comment plus the `ImportError` message in
> `_rhino_cal_jax()`, which already names the git URL.

- [ ] **Step 2: Add a decision record**

Append to the `## Decisions` section of `DESIGN.md`:

```markdown
### D15 — The noise-wave data model is an imported package, not a rheplicant module

`NoiseWaveOperator` is an adapter over `rhino_cal_jax`, the JAX/Equinox
implementation of the Noise-Wave GCR note's Eq. 1 that lives in the
`RHINO-Experiment/rhino-cal` repository beside the numpy pipeline it was
verified against. The dependency runs one way — `rhino_cal_jax` knows nothing
about `State`, `Pipeline` or operators — for the same reason `limtod_jax` does
not: the calibration model has a life outside this framework, and a physics
change should be reviewable next to the reference implementation it must keep
agreeing with, not here.

The one thing the adapter adds is placement. Reflection coefficients belong to
*sources*, and the `receiver_input` selector discards source identity before
the `noise_wave` node sees the data. The operator therefore carries `Γ` per
source and re-reads the switch array — which the previous placeholder, holding
one scalar `Γ` for every sample, could not do.

Why that placement is the whole game: count equations **per frequency channel**,
since the noise-wave temperatures are functions of frequency and nothing ties
channels together a priori. Each switch position contributes exactly one
equation per channel, so the design matrix has rank `min(n_src, 3) × n_freq` —
one load leaves it deficient threefold, and three distinct loads make it square.
That is why EDGES and REACH switch between four or five calibrators. A single
shared `Γ` collapses every source onto the same row and forfeits all of it,
returning a finite, well-shaped, wholly prior-driven answer. Draft Eqs. 11–12
say as much; the placeholder simply predated them.

The sharp edge worth recording, because the loose version of this claim is
false: frequency structure in `Γ` **does** identify *scalar*, frequency-
independent noise-wave temperatures from a single load (`cond(JᵀJ) ≈ 6`). It is
the per-channel case — the physical one — that requires switching. The bridge
between the two regimes is exactly the draft's basis matrices `U_unc`, `U_cos`,
`U_sin` (Eqs. 13–15): they tie channels together and so lower the number of
calibrators needed.
```

- [ ] **Step 3: Update the operator reference**

In `docs/operators.md`, replace the `NoiseWaveOperator` placeholder entry with
a real one: the equation, the per-source `Γ` requirement, the
`coords.extra["receiver_input"]` contract, and the `rhino-cal-jax` install
line. Cross-reference `examples/noise_wave_gcr.py`.

- [ ] **Step 4: Update the changelog**

Add under `## [Unreleased]`:

```markdown
### Added
- `examples/noise_wave_gcr.py` — the noise-wave temperatures as a checked
  linear block, solved in closed form and sampled exactly (GCR note Eqs. 30–31).
- Optional `rhinocal` extra for the `rhino_cal_jax` backend.

### Changed
- **BREAKING** `NoiseWaveOperator` now implements the full Eq. 1 through
  `rhino_cal_jax` instead of the `F -> 1` placeholder. `t_zero` is renamed
  `t_rx`; the scalar `gamma_re`/`gamma_im` pair is replaced by per-source
  `gamma_src_re`/`gamma_src_im` of shape `(n_source, n_freq)` plus
  `gamma_rec_re`/`gamma_rec_im` of shape `(n_freq,)`. The operator reads the
  connected source from `coords.extra["receiver_input"]` and raises if it
  carries several sources and that array is absent.
```

- [ ] **Step 5: Verify the docs build and the suite is green**

Run:

```bash
cd /Users/zzhang/projects/e-RHINO && python -m pytest -q && ruff check src tests examples && python -m sphinx -b html docs docs/_build -W
```

Expected: all tests pass, ruff clean, sphinx exits 0 with no warnings.

- [ ] **Step 6: Commit**

```bash
cd /Users/zzhang/projects/e-RHINO
git add pyproject.toml CHANGELOG.md DESIGN.md docs/operators.md
git commit -m "docs: record the rhino_cal_jax boundary (D15) and the Eq. 1 operator"
```

---

## Post-implementation

- [ ] Report discrepancies **D1** (Eq. 4's missing square on `|F|`), **D2** (Eq. 1's
  additive `n_w` versus Eq. 8's fractional `w`), and **D3** (the `abs(P + n)` fold)
  to the draft's author, with the pinning tests as evidence.
- [ ] Ask the user before pushing to `RHINO-Experiment/rhino-cal` — it is a
  shared repository containing another author's work.
- [ ] Consider a follow-up plan for the `ObservationHandler` layer (Chebyshev
  noise-wave fields, switch-cycle generation, mock RHINO HDF5 output), which the
  draft's GCR sections need but Eq. 1 does not.
