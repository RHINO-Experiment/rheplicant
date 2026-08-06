# Install

## Requirements

Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import name are
the same: `rheplicant`.

## The one awkward step, first

`limTOD` is a **dependency**, not an extra — the sky engines it carries are the
forward model rather than an accessory — but PyPI has only ≤ 1.8.0 against a
≥ 1.10 floor, so `pip install rheplicant` cannot resolve it on its own. Install
limTOD from source first and the rest follows:

```bash
pip install "limTOD[jax] @ git+https://github.com/zzhang0123/limTOD"
pip install rheplicant
```

## Extras

Three of the four name a requirement that is not on PyPI, deliberately: the
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
  - `NoiseWaveOperator` — the noise-wave model of the GCR note's Eq. 1
  - `pip install "rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal"`
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
uv sync --frozen
```

:::{warning}
**Neither `uv sync` nor `uv run` works without `--frozen`.** Both were measured,
not assumed: each refuses with *"your project's requirements are
unsatisfiable"*, because `limTOD[jax]>=1.10` cannot be resolved against an index
carrying ≤ 1.8.0. `--frozen` works against the existing lock.

And `uv sync --frozen` **removes** anything installed outside that lock —
editable local checkouts of limTOD or rhino-cal included. If you are developing
against source checkouts of those, re-install them afterwards, or stay on
`uv run --frozen`, which does not sync.
:::

## Running the tests

There is no CI; run the suite and the linter in the project venv before pushing.

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
python -c "import rheplicant; print(rheplicant.__version__)"
python -c "from rheplicant.radio import RADIO_GRAPH; print(len(RADIO_GRAPH.nodes), 'nodes')"
```

The second line is the more useful one: it proves the radio layer imported and
the default signal-path template registered. If the extras are in place, these
import too — each is the module an operator reaches for, and an absent one
raises an `ImportError` naming what to install rather than failing later:

```bash
python -c "import limtod_jax, rhino_cal_jax; print('sky engines and noise waves ready')"
```

Then read [the guided tour](tour.md), or [ingestion](ingestion.md) if you have a
recording in hand.
