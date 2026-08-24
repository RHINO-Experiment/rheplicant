# Working notes for coding agents

Repo-specific facts that are expensive to rediscover. Everything here was
measured in this checkout, not assumed. Keep it short enough to be read.

## Running the tests

```bash
.venv/bin/python -m pytest -n 8
```

**Judge by pytest's exit code, never a pipe's.** `pytest … | tail` reports the
exit status of `tail`, so a collection error or a usage error reads as green.
Redirect to a file and read `$?`:

```bash
.venv/bin/python -m pytest -n 8 > run.log 2>&1; echo "EXIT=$?"; tail -5 run.log
```

**Partial runs no longer need `--no-cov`** — this reversed, and the old habit
is widespread enough to be worth stating. `addopts` is now just `-q`.
Coverage moved out of it into a serial CI job, because `-n 8` and
`-p no:xdist` do not measure the same suite: 88.32 % against 82.26 % on one
tree with every test passing, a six-point gap concentrated in the
bootstrap/CLI modules. A figure that depends on the worker count is not a
property of the code, so the reported one comes from the run that has no
worker count. Measured on this tree: `pytest tests/test_docs_links.py` with
no flags exits 0.

**The suite is two pytest sessions.** The evidence layer needs float64 while
other tests assert refusals only float32 forces, and `jax_enable_x64` is
process-global. `tests/test_evidence_session.py` runs the second one as a
subprocess, which is why `tests/evidence` shows as skips in the main count.

### A complete environment, and why the count depends on it

Several test modules stand down behind a module-level `pytest.importorskip`,
so a thinner virtualenv silently collects fewer tests of the same suite.
Complete means the dev group plus **`h5py`** and **`rhino-cal-jax`** (the
latter is not on PyPI; install it editable from its own checkout, and install
`editables` alongside it, which its editable hook needs).

`tests/test_readme_counts.py` pins the README's test count by equality but
**skips** where any module fails to collect, and its skip message says so
loudly. A skipping guard is not a passing one: this check stood down for
weeks on a machine missing those two packages, and three real failures sat
behind those modules the whole time. If it skips, complete the environment
rather than reading it as green.

The coverage figure in `README.md` is **truncated**, never rounded — the same
guard treats it as a floor and compares it against `[tool.coverage.report]
fail_under` in `pyproject.toml` (currently 82). That is the only place the
floor lives; the `--cov-fail-under` flag it used to read is gone.

## The config layer's boundary is textual

`tests/config/test_config_surface.py::TestTheLayerBoundaryIsMechanical` scans
the **text** of every `src/rheplicant/**/*.py` outside `config/` for
`from rheplicant.config` or `import rheplicant.config`, and allows exactly
five files:

    gui/form_catalog.py  gui/form_edits.py  gui/jobs.py
    gui/outputs.py       gui/validation.py

Because it is a text scan, even a `TYPE_CHECKING` import or a docstring
containing the phrase trips it. Any other GUI module that needs config
vocabulary imports it **from `gui/form_catalog.py`**, which re-exports through
its `__all__` — that `__all__` is load-bearing, not decoration, and removing a
name from it breaks the module that laundered it. This constraint, not file
length, is what shapes `form_catalog.py`.

## The GUI frontend: rebuild the bundle, or the e2e suite tests the last release

`tests/gui/e2e/` serves the **checked-in production bundle** under
`src/rheplicant/gui/static/assets/`, not the React source. A change to
`src/rheplicant/gui/react/**` that is not followed by

```bash
cd tools/config_gui_spike/react && npm run build:production
```

leaves the whole Playwright suite green while testing the previous release.
Measured: a component landed, all 188 e2e tests passed, and the shipped
bundle contained none of it. The build writes with `emptyOutDir`, so the old
content-hashed asset is deleted and a new one added — commit both.

TypeScript gates, all of which must pass:

```bash
npm run check:tests      # the component suite under tests/gui/react/
npm run check:e2e        # the Playwright specs
npm run test:session      -- --run   # the 400+ component tests
```

`npm test` alone runs a different, much smaller config — use `test:session`.

### Screenshot baselines

Six canonical screenshots live in `tests/gui/e2e/snapshots/`. When one
legitimately changes, produce a before/after for the human first; do not
update a baseline unasked. `--update-snapshots` may be refused by the harness;
the equivalent is to copy the `-actual.png` a normal failing run writes,
asserting each image's size and changed region before writing it, then
re-running normally to verify.

## Two habits this codebase rewards

**Derive, do not re-spell.** The widget census is built live from the config
registries; `operator_table()` walks `rheplicant.radio.__all__`; the GUI's
FAN question is asked of `sections/compose.many_shape_problem`. A second copy
of a rule is the one that goes stale, because nothing renders the two side by
side.

**Ask whether a guard can still fail.** The recurring defect here is not wrong
code but a test that has stopped being able to fail — a module that no longer
collects, an assertion whose fixtures cannot tell right from plausibly wrong,
an equality pinning an implementation detail instead of the property it
describes. When a test passes, it is worth knowing whether it would have
failed. Mutating the source and re-running is the cheap way to find out.
