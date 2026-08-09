# Install

## Requirements

Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import name are
the same: `rheplicant`.

## Install

```bash
pip install rheplicant
```

`limTOD` comes with it. It is a **dependency**, not an extra — the sky engines
it carries are the forward model rather than an accessory — and since limTOD
1.10.0 reached PyPI the `>= 1.10` floor resolves with no preparatory step.

## Extras

Two of the four name a requirement that is not on PyPI, deliberately: the
package is developed alongside them and pinning a git URL in `pyproject.toml`
would make this project unpublishable. So the extra records *what is needed*,
and you install it yourself.

:::{list-table}
:header-rows: 1
:widths: 12 30 58

* - Extra
  - Gives you
  - Install
* - `numpyro`
  - NUTS, and every gradient posterior
  - `pip install "rheplicant[numpyro]"`
* - `cal`
  - `NoiseWaveOperator` — the noise-wave receiver model, reflection couplings
    and all
  - `pip install "rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax"`
    — the `@feat/rhino-cal-jax` is load-bearing: `rhino_cal_jax/` exists only on
    that branch, and the default branch has no `pyproject.toml` to build
* - `rfi`
  - `MomentRFIFlaggingOperator`, the real flagger. The threshold-based
    `FlaggingOperator` needs none of it
  - `pip install "MomentRFI @ git+https://github.com/zzhang0123/MomentRFI"`
* - `rhino`
  - `read_rhino_observation()` — the RHINO HDF5 reader (h5py). The Touchstone
    reader needs none of it, being numpy only
  - `pip install "rheplicant[rhino]"`
:::

## Development

```bash
git clone https://github.com/RHINO-Experiment/rheplicant
cd rheplicant
uv venv
uv pip install -e . --group dev
```

:::{warning}
**Neither `uv sync` nor `uv run` works in this project — with `--frozen` or
without it.** Measured on a fresh clone, not assumed.

*Without* `--frozen`, each refuses with *"your project's requirements are
unsatisfiable"*. `uv` resolves **every declared extra** when it locks, and
`rheplicant[cal]` → `rhino-cal-jax` is not on PyPI by design; `rheplicant[rfi]`
→ `MomentRFI` is the same shape. The `pyproject.toml` comment beside each says
why they name a requirement instead of resolving it.

*With* `--frozen`, each refuses with *"Unable to find lockfile at `uv.lock`"*.
This repository ships no lockfile and **cannot**: `uv lock` fails on the very
same unsatisfiable extra, so there is nothing to commit. `--frozen` applies only
where a `uv.lock` already exists in your working copy — which a fresh clone does
not have and cannot generate. Treat any `uv.lock` you find in an older checkout
as stale rather than as the missing piece.

`uv pip install` resolves only what you ask it for, never the whole declared
universe, which is why the two commands above work where the project-level ones
cannot. Nothing is removed either, so editable local checkouts of limTOD or
rhino-cal survive it — install those the usual way afterwards, in any order.
:::

## Optional local data

Two things this project compares itself against cannot be published: the RHINO
CST far-field exports, and the `rhino-cal` checkout whose numpy readers are the
reference implementation. Neither has a path that can be guessed, so neither is
guessed — you name yours, or the work that needs it stands down and says so.

:::{list-table}
:header-rows: 1
:widths: 32 68

* - Variable
  - What it unlocks
* - `RHEPLICANT_RHINO_BEAMS`
  - The directory of per-frequency CST far-field exports. Unlocks the beam tests
    in `tests/radio/test_beams.py`, the real horn in
    `examples/sky_to_noise_wave.py` (which otherwise substitutes a Gaussian and
    labels the plot as such), and the receiver figure. `--beam-dir` overrides it
    for the example.
* - `RHEPLICANT_RHINO_CAL`
  - A `rhino-cal` checkout. Unlocks
    `tests/radio/test_ingestion_vs_reference.py`, which cross-checks this
    package's Touchstone and HDF5 readers against rhino-cal's.
:::

Neither is required, and nothing fails without them — the tests that need them
skip with the variable named in the reason. Both used to be hard-coded paths
under one person's home directory, which meant those tests never ran anywhere
else and said nothing about it.

## Running the tests

There is no CI; run the suite and the linter in the project venv before pushing.
The `dev` group carries what the tests need on top of the package — `numpyro`,
because five modules import it unguarded, and `mdit-py-plugins`, because the
docs-link guard computes anchors with myst's own slugifier. Both are runtime
*extras* rather than dependencies, so nothing but the test suite pulls them in;
without them `pytest` stops at collection, or reports every documentation link
as broken.

```bash
.venv/bin/python -m pytest          # ~12 min with coverage
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/evidence   # the float64 half
.venv/bin/python -m ruff check src tests
```

The second line is **not** optional work you might skip — plain `pytest` already
runs it for you, in a subprocess, via `tests/test_evidence_session.py`. It is
written out because that is how you run those tests directly when one of them
fails.

:::{admonition} Why the suite is two sessions
:class: note
The evidence layer needs float64: a stored factor's offset scalar is the
time–bandwidth product, ~7.2e11 for one night, against a difference of ~1e5 —
which float32 annihilates rather than rounds. The rest of the suite must stay at
float32, because eighteen tests assert refusals that only float32 forces. And
`jax_enable_x64` is process-global, so the two cannot share an interpreter.

That split is also why the reported coverage is what it is: the second session
runs `--no-cov` in its own process, so its passing tests contribute nothing to
the default report, and most of the default report's uncovered statements are
the seven evidence-layer files.
:::

## Check it worked

```bash
.venv/bin/python -c "import rheplicant; print(rheplicant.__version__)"
.venv/bin/python -c "from rheplicant.radio import RADIO_GRAPH; print(len(RADIO_GRAPH.nodes), 'nodes')"
```

The interpreter is named explicitly because nothing above activates the
environment — a bare `python` here reaches whichever one is on your `PATH` and
reports `ModuleNotFoundError` for an install that is in fact fine. Drop the
prefix if you have run `source .venv/bin/activate`.

The second line is the more useful one: it proves the radio layer imported and
the default signal-path template registered. If the extras are in place, these
import too — each is the module an operator reaches for, and an absent one
raises an `ImportError` naming what to install rather than failing later:

```bash
.venv/bin/python -c "import limtod_jax, rhino_cal_jax; print('sky engines and noise waves ready')"
```

Then read [the guided tour](tour.md), or [ingestion](ingestion.md) if you have a
recording in hand.
