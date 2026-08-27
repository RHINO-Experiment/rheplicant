# Working notes for coding agents

Repo-specific facts that are expensive to rediscover. Everything here was
measured in this checkout, not assumed. Keep it short enough to be read.

**This file exists twice.** `CLAUDE.md` and `AGENTS.md` are one document for two
tools, and `tests/test_docs_claims.py` holds them byte-identical. Edit both, or
that test goes red with the diff. They were allowed to drift once and each ended
up stale exactly where the other was current — this page's config allowlist said
five files against a real three, while `AGENTS.md` still called the coverage gap
unexplained after it was found and fixed.

## Running the tests

```bash
.venv/bin/python -m pytest -n 8
```

**On a machine you are sharing, run it in two phases instead.** That one
command has exhausted 96 GB and powered the machine off. No single test is
the cause; three kinds of parallelism stack. The `-n 8` parent, the `-n 4`
child `tests/test_evidence_session.py` spawns, and `tests/gui/e2e` shelling
out to `playwright test`, which takes CPU/2 — **14 browser workers here** —
for a peak near 27 heavy processes. Only the first is a pytest flag away.

```bash
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e
.venv/bin/python -m pytest tests/gui/e2e -n 2
```

Measured: 337 s + 60 s split, against 258 s in one `-n 8` run, with memory
flat at 95 % free throughout. About 55 % slower, and it finishes.

**Judge by pytest's exit code, never a pipe's** — and the exit code you were
handed is often not pytest's. Capture it to its own file, and read *that*:

```bash
.venv/bin/python -m pytest -n 8 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
cat run.exit; tail -5 run.log
```

Writing `…; echo "EXIT=$?"; tail -5 run.log` on one line is not enough, and
that was the old advice here. The compound command ends in `tail`, so its
status is `tail`'s — and anything reporting on the command as a whole (a
wrapper, a task notification, `&&`) says **0** however pytest ended.
Measured: a run stopped with SIGTERM at 92 % printed `EXIT=143` while the
harness announced `exited with code 0`. Zero failures so far and no
completion looks exactly like a pass.

**Non-zero is not one thing.** pytest returns **1** for tests failed, **2**
interrupted, **3** internal error, **4** usage error (a mis-split `-k`
expression does this), **5** nothing collected; a killed process gives
**143**. Only **1** means a test failed.

That distinction is load-bearing in **mutation testing**, where the rule is
"mutate, expect red": scoring any non-zero as a kill turns a typo in the
pytest invocation into a KILLED for every mutant, and the guard being
checked is never exercised. Score `1`, and treat 2–5 as "the run did not
happen".

**And clear `__pycache__` between mutants.** Restoring a source file with
`cp` inside the same second leaves bytecode whose timestamp is not older
than the source, so Python reuses it and the mutant never runs — recorded
as a **SURVIVED** that is just as false as the KILLs above, and pointing
the other way. Three shapes, three directions, all of them a green-looking
lie:

| what you see | what happened |
|---|---|
| exit 4 read as "tests failed" | a mis-split `-k`; the run never started |
| exit 143 reported as 0 | a killed run, 92 % done, zero failures so far |
| SURVIVED | stale bytecode; the mutation was never executed |

Suspect a recorded SURVIVED before you suspect the test.

And when a mutant does go red, check that the red is **your** assertion: a
guard you did not know existed can kill the mutation first, leaving the one
you just wrote unevaluated. No exit code separates those two — only the
failure's own test name does. Both halves of this were measured: a
`sorted()` mutation here was killed by an origin-shape cross-check three
functions away, and the same mistake in the sibling repository recorded
four mutants as killed by tests that never ran.

**And commit the batch before you mutate it.** The protocol restores with
`git checkout -- src/` rather than `cp`, which is right — `cp` is the stale-
bytecode trap above. But `git checkout` restores to **HEAD**, so on a tree
carrying uncommitted work it is a silent full revert of that work, not of the
mutant. Measured in the sibling repository on 2026-08-27: a mutation run was
killed on a timeout leaving a mutant in the tree, and the `git checkout --
src/` that followed took the whole unfinished feature with it. Only `tests/`
survived, because every mutation point happened to be under `src/`. `git
checkout` is the better tool precisely because HEAD is the reference — which
is also the requirement: HEAD has to already be the thing you want back.

Two smaller ones from the same run. Flush the mutation log
(`print(..., flush=True)`) or a killed run leaves a zero-byte file and no
record of how far it got. And scope the `__pycache__` sweep to the package:
`rglob("__pycache__")` from the repo root also walks `.venv`, which turns a
15-second mutant into a two-minute one for no benefit.

**Mutation testing also has one blind spot, and it is structural rather
than a matter of care.** It asks whether an assertion is sensitive to its
input. It cannot ask whether the input is the one you think it is — because
**you can only mutate what you already know is a variable**. Measured: two
cross-repository guards asserting upstream docstring text survived every
mutation of that text and were nonetheless green only because the editable
install happened to be checked out on an unmerged branch. The mutations
proved the assertions read the text; nothing in reach proved *which ref*
the text came from, since the ref was not in the variable set. Finding that
took someone working in the other repository, for whom "which checkout is
installed" was the first suspicious thing rather than a constant. When a
guard's greenness depends on something you have never varied, no amount of
mutating what you have will surface it.

**Count tests from `--junit-xml`, not from the terminal.** Counting dots
misread a run as `892 passed, 2 skipped` when it was `892 passed, 0
skipped`; the summary line is prose and the XML is the record.

**Partial runs no longer need `--no-cov`** — this reversed, and the old habit
is widespread enough to be worth stating. `addopts` is now just `-q`, so
`pytest tests/test_docs_links.py` with no flags exits 0. Coverage is measured
by its own job.

It was moved there because `-n 8` and `-p no:xdist` disagreed by six points.
**That cause is now fixed** — two tests uninstalled coverage's tracer with
`sys.settrace(None)`, blacking out 1982 consecutive tests serially, and
`tests/test_coverage_instrument.py` now refuses that shape. The two runners
agree: 89.22 % either way, one statement apart, and the remaining wobble is
run-to-run rather than runner-to-runner. Coverage stays out of `addopts` for
the second reason only, that a partial run should not trip a whole-package
gate. `pyproject.toml` carries the measurement.

**The suite is two pytest sessions.** The evidence layer needs float64 while
other tests assert refusals only float32 forces, and `jax_enable_x64` is
process-global. `tests/test_evidence_session.py` runs the second one as a
subprocess, which is why `tests/evidence` shows as skips in the main count.

### A complete environment, and why the count depends on it

Several test modules stand down behind a module-level `pytest.importorskip`,
so a thinner virtualenv silently collects fewer tests of the same suite.
Complete means the dev group plus **`h5py`**, **`rhino-cal-jax`** (not on
PyPI; install it editable from its own checkout, and install `editables`
alongside it, which its editable hook needs) and **`bayesmith`** at the
`>=0.4` surface. The floor moved from 0.2 on 2026-08-27 and the two halves
are worth telling apart: 0.2 named `first_fit` and `exact.loglinear`, which
`partition.py` and `loglinear.py` import; 0.3 names `AffinityRefused`'s
structured payload and `ComplexNormal`, which `graph_bridge.py` needs. A 0.2
install satisfies the import statements of the second pair and then fails at
the call. 0.3 named `AffinityRefused`'s payload and `ComplexNormal`; **0.4 names
`observed_mask`**, which is how the adapter presents a `FlaggedNoise`.
**0.4.0 is on PyPI as of 2026-08-27**, so
`uv pip install 'bayesmith>=0.4'` resolves; this checkout nevertheless holds
it **editable from `../bayesmith` with `--no-deps`**, because the two
repositories are developed against each other and a released version would
freeze the seam mid-programme. Its runtime deps (jax, equinox, numpy,
numpyro) are already here. Without it `rheplicant.inference` does not import at all, so this one
fails loudly rather than as silent skips.

**There are two pytest sessions that need `JAX_ENABLE_X64=1`, not one.**
`tests/evidence/` is the older; `tests/seam/` is the adapter's acceptance
tier, added 2026-08-27, whose deterministic half compares against a dense
solve at `rtol < 1e-12`. Each has its own gate conftest and its own driver
(`tests/test_evidence_session.py`, `tests/test_seam_session.py`) that runs it
as a subprocess and goes red when it does. Run either by hand with, e.g.,
`JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam`.

`tests/test_readme_counts.py` pins the README's test count by equality but
**skips** where any module fails to collect, and its skip message says so
loudly. A skipping guard is not a passing one: this check stood down for
weeks on a machine missing those two packages, and three real failures sat
behind those modules the whole time. If it skips, complete the environment
rather than reading it as green.

The coverage figure in `README.md` is **truncated**, never rounded — the same
guard treats it as a floor and compares it against `[tool.coverage.report]
fail_under` in `pyproject.toml` (currently **89**, raised from 82 with the
fix above). That is the only place the floor lives; the `--cov-fail-under`
flag it used to read is gone.

## The config layer's boundary is textual

`tests/config/test_config_surface.py::TestTheLayerBoundaryIsMechanical` scans
the **text** of every `src/rheplicant/**/*.py` outside `config/` for
`from rheplicant.config` or `import rheplicant.config`, and allows exactly
three files:

    gui/form_catalog.py  gui/form_edits.py  gui/validation.py

It used to allow five. `gui/jobs.py` and `gui/outputs.py` held the permission
and imported nothing from config -- they take the vocabulary from
`form_catalog.py`'s `__all__` like every other GUI module -- so the allowlist
now asserts in BOTH directions: an entry that does not use its permission
fails, because an unused exemption is the one file that could start reaching
into config with nothing to say so.

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
