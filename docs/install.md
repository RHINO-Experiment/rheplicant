# Install

## Requirements

Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import name are
the same: `rheplicant`.

## Install

```bash
pip install rheplicant
```

As of 2026-10-02 that line installs 0.2.0, the latest upload on PyPI, which
predates the `rheplicant` command, the configuration layer and the workbench
these pages describe. To install 0.9.1, name its tag:

```bash
pip install "rheplicant @ git+https://github.com/RHINO-Experiment/rheplicant@v0.9.1"
```

Two packages come with it as dependencies, not extras, and both resolve
from PyPI. `limTOD` carries the sky engines, which are the forward model.
`bayesmith` carries the inference arithmetic, and brings numpyro with it.
`import rheplicant` imports neither bayesmith nor numpyro;
`import rheplicant.inference` imports both.

## Extras

Two of the integration extras name a requirement that is not on PyPI. The
package is developed alongside them, and pinning a git URL in `pyproject.toml`
would make this project unpublishable, so the extra records what is needed
and you install it yourself.

:::{list-table}
:header-rows: 1
:widths: 12 30 58

* - Extra
  - Gives you
  - Install
* - `numpyro`
  - Raises numpyro to the tested `>=0.21`. numpyro itself is already
    installed, through bayesmith
  - `pip install "rheplicant[numpyro]"`
* - `cal`
  - `NoiseWaveOperator`, the noise-wave receiver model, including the
    reflection couplings
  - `pip install "rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax"`.
    The `@feat/rhino-cal-jax` is required: `rhino_cal_jax/` exists only on
    that branch, and the default branch has no `pyproject.toml` to build
* - `rfi`
  - `MomentRFIFlaggingOperator`, the real flagger. The threshold-based
    `FlaggingOperator` needs none of it
  - `pip install "MomentEmu @ git+https://github.com/zzhang0123/MomentEmu" "MomentRFI @ git+https://github.com/zzhang0123/MomentRFI"`.
    Both go in one command: MomentRFI declares MomentEmu, neither is on PyPI,
    and MomentRFI named alone does not resolve
* - `rhino`
  - `read_rhino_observation()`, the RHINO HDF5 reader (h5py). The Touchstone
    reader needs none of it, being numpy only
  - `pip install "rheplicant[rhino]"`
* - `uvbeam`
  - Read pyuvdata `UVBeam` resources; the beam physics bridge itself comes with
    limTOD
  - `pip install "rheplicant[uvbeam]"`
* - `gui`
  - The packaged FastAPI + React configuration workbench and
    `rheplicant-gui` launcher
  - `pip install "rheplicant[gui]"`, then `rheplicant-gui`; see the
    [editor security and trust boundaries](config-gui.md)
:::

### Start the configuration workbench

The wheel contains the production assets, so an installed wheel needs no
Node.js toolchain; the `gui` extra adds the server (FastAPI and uvicorn):

```bash
pip install "rheplicant[gui]"
rheplicant-gui                    # http://127.0.0.1:8000/
```

The launcher binds to loopback by default. A non-loopback bind is refused
unless `--allow-remote` is given together with at least one
`--allowed-host NAME`, and those flags are only acknowledgement: the
application has no authentication, tenant isolation or sandbox. The server
answers only to loopback host names and the listed names, and refuses a
state-changing request whose `Origin` is not its own, which keeps out a
DNS-rebinding page under any other name. YAML may load plugins and `python:`
targets; resource/output fields are server paths; jobs use the server
account's files and compute. Read the complete
[workbench workflow and trust boundary](config-gui.md) before using remote
access or running a document from another person.

## Development

```bash
git clone https://github.com/RHINO-Experiment/rheplicant
cd rheplicant
uv venv
uv pip install -e . --group dev
```

rheplicant declares `bayesmith>=0.10,<0.11`, and bayesmith 0.10.0 is on PyPI
(2026-10-02), so the resolver takes it from the index. The fresh-environment
tests in `tests/config/` install the same way.

:::{warning}
**Neither `uv sync` nor `uv run` works in this project, with `--frozen` or
without it.** Measured on a fresh clone.

*Without* `--frozen`, each refuses with *"your project's requirements are
unsatisfiable"*. `uv` resolves every declared extra when it locks, and
`rheplicant[cal]` → `rhino-cal-jax` is not on PyPI by design; `rheplicant[rfi]`
→ `MomentRFI` is the same shape. The `pyproject.toml` comment beside each says
why they name a requirement instead of resolving it.

*With* `--frozen`, each refuses with *"Unable to find lockfile at `uv.lock`"*.
This repository ships no lockfile and cannot: `uv lock` fails on the same
unsatisfiable extra, so there is nothing to commit. `--frozen` applies only
where a `uv.lock` already exists in your working copy, which a fresh clone does
not have and cannot generate. Treat any `uv.lock` you find in an older checkout
as stale.

`uv pip install` resolves only what you ask it for, so the two commands above
work where the project-level ones cannot. It removes nothing either, so
editable local checkouts of limTOD or rhino-cal survive it; install those the
usual way afterwards, in any order.
:::

## Optional local data

Two things this project compares itself against cannot be published: the RHINO
CST far-field exports, and the `rhino-cal` checkout whose numpy readers are the
reference implementation. Neither has a default path: you name yours, or the
work that needs it stands down and says so.

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

Neither is required, and nothing fails without them: the tests that need them
skip with the variable named in the reason.

## Running the tests

`.github/workflows/test.yml` runs the suite on every push to main and on pull
requests. Run the suite and the linter in the project venv before pushing: the
runner is Linux and your machine may not be.

```bash
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e
.venv/bin/python -m pytest tests/gui/e2e -n 2
.venv/bin/python -m ruff check src tests
```

Run it in those two phases. A single `pytest -n 8` over everything has
exhausted 96 GB on a shared machine: the parent's workers, the float64
sessions' own workers and the Playwright browsers of `tests/gui/e2e` run at
once. Measured, the two phases take 337 s and 60 s against 258 s for the one
command, and memory stays flat.

`pytest-xdist` is in the `dev` group, and the counts are the same with or
without it because nothing depends on execution order. Coverage is not
measured by default; CI measures it in a separate serial job.

**What a complete test environment holds.** `uv pip install -e . --group dev`
is enough to import the package and run most of the suite. Several test
modules stand down behind `pytest.importorskip`, so a thinner environment
collects fewer tests and still passes. The reference install is the one in
`.github/workflows/test.yml`: the `rhino`, `numpyro`, `gui`, `gui-react` and
`uvbeam` extras, `pygdsm`, `rhino-cal-jax`, `MomentRFI` with `MomentEmu`, and
the Node toolchain (`npm ci` in `tools/gui/react`) for the GUI type checks and
the Playwright suite.

**The examples are part of the suite.** `tests/test_examples_run.py` runs
each script under `examples/` in its own interpreter, with its needs and its
time read from [the examples page](examples.md). The three scripts documented
at 59 s or more run only with `RHEPLICANT_ALL_EXAMPLES=1`, and on a CI runner
no script runs without it. `tests/test_global21cm_documents.py` validates the
six documents of `examples/global21cm/` where the emulator they import is
installed, which `examples/global21cm/requirements.txt` does; elsewhere those
six cases skip.

:::{admonition} Why the suite is three sessions
:class: note
Two parts of the suite need float64. `tests/evidence` does because a stored
factor's offset scalar is the time–bandwidth product, ~7.2e11 for one night,
against a difference of ~1e5, which float32 annihilates. `tests/seam` does
because it compares against a dense solve at `rtol < 1e-12`. The rest of the
suite must stay at float32, because tests there assert refusals that only
float32 forces. `jax_enable_x64` is process-global, so the three cannot share
an interpreter.

Plain `pytest` runs all three: `tests/test_evidence_session.py` and
`tests/test_seam_session.py` each run their directory as a subprocess with
`JAX_ENABLE_X64=1`, which is why those directories show as skips in the main
count. To run one directly when a test in it fails:

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/evidence
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam
```

The float64 sessions run with `--no-cov`, so the modules only they exercise
show as uncovered in the default coverage report.
:::

## Check it worked

```bash
.venv/bin/python -c "import rheplicant; print(rheplicant.__version__)"
.venv/bin/python -c "from rheplicant.radio import RADIO_GRAPH; print(len(RADIO_GRAPH.nodes), 'nodes')"
```

The interpreter is named explicitly because nothing above activates the
environment: a bare `python` here reaches whichever one is on your `PATH` and
reports `ModuleNotFoundError` for an install that is fine. Drop the
prefix if you have run `source .venv/bin/activate`.

The second line checks more: that the radio layer imported and the default
signal-path template registered. If the extras are in place, these import
too. Each is the module an operator imports, and an absent one raises an
`ImportError` naming what to install rather than failing later:

```bash
.venv/bin/python -c "import limtod_jax, rhino_cal_jax; print('sky engines and noise waves ready')"
```

Then read [the guided tour](tour.md), or [ingestion](ingestion.md) if you have a
recording in hand.
