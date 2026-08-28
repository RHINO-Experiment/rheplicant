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

**And the exit file itself can be a leftover.** Waiting on `[ -f run.exit ]`
is only a test of the *name*, and a `run.exit` in `/tmp` outlives the session
that wrote it. Measured 2026-08-28: `until [ -f /tmp/full2.exit ]` returned
immediately against a file two days old, so a run still at 23 % was read as
finished with `PYTEST_EXIT=1`, and the number was real — from a different
run. Write the exit file into this session's own scratchpad, or `rm -f` it
before starting; and wait on the **summary line in the log**, never on a
file existing. Same family as the two above: something that says "X" when
the truthful answer is "this query never happened".

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

**That trap has a mirror, and it reads as SURVIVED rather than KILLED.** The
paragraph above is "the red is not yours"; this one is "the test exists, you
just did not run it". Measured 2026-08-28: a mutation deleting a refusal was
recorded SURVIVED, and the test that kills it had been in the suite the whole
time — it simply lived in a file outside the nine the mutation script named.
Both mistakes print exactly one line and neither line says which it is. So
before running a set, answer **who tests this code** by `grep -rl` or by the
refusal census, not from memory; and treat a SURVIVED whose subject is an
obvious refusal as a target-set bug until you have shown otherwise.

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

**Rule (0) has a second half, and it is the half that bites twice.** "Commit
the batch before you mutate it" is not "commit once when the batch starts" —
it is **HEAD has to be what you want back, every time you run the set**. The
protocol's `git checkout -- src/ tests/` restores to HEAD, so a fix written
*after* the first mutation run and *before* the second is reverted by that
second run's own opening restore, silently and before any mutant is applied.
Measured 2026-08-27: two survivors were diagnosed correctly, the guards were
repaired, the set was re-run to confirm — and it reported the same two
survivors, because the repair no longer existed. Nothing in the output says
so; a reverted fix and a fix that did not work look identical.

Commit the repair, then re-run. And if a mutation script restores paths beyond
the mutants' own, narrow it: that one restored all of `tests/` to undo mutants
that were only ever in `src/`.

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

**That instance closed on 2026-08-28 and the lesson did not.** The branch
was merged; measured against the **remote** rather than a local ref
(`git ls-remote` for the tip, then `git show origin/main:<path>`), both
docstrings are on `origin/main` and `track-a-tail` no longer exists, so the
two guards now read the same text any checkout of `main` carries. The
records that named the dependency — `docs/migration/{plan,linear}.md` and
the two guards' own docstrings — were corrected in the same batch, because
a closed hazard described as open costs the next reader exactly what an
open one described as closed does. What stays true is the shape: **ask what
a green guard depends on that you have never varied**, and prefer to answer
it by measuring the remote, since a local `origin/main` is a file that was
right when it was last fetched.

**Count tests from `--junit-xml`, not from the terminal.** Counting dots
misread a run as `892 passed, 2 skipped` when it was `892 passed, 0
skipped`; the summary line is prose and the XML is the record.

**Partial runs no longer need `--no-cov`** — this reversed, and the old habit
is widespread enough to be worth stating. `addopts` is now just `-q`, so
`pytest tests/test_docs_links.py` with no flags exits 0. Coverage is measured
by its own job.

**And because `addopts` is already `-q`, do not pass another one.** A second
`-q` makes it `-qq` and **the summary line disappears entirely** — the run
still prints its dots and still exits correctly, so a passing run looks
normal and a count you wanted is simply absent. Measured here 2026-08-28:
`pytest tests/inference/test_gls.py -n 2` printed `20 passed in 14.64s`, and
the same command with `-q` printed only the progress line. This trap was on
record for the sibling repository and **not** for this one, although
`pyproject.toml` here carries the identical `addopts = "-q"`; it is written
down now because the absent line reads as "no summary was produced" rather
than as "you suppressed it".

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
`>=0.5` surface. The floor moved from 0.2 on 2026-08-27 and the halves are
worth telling apart: 0.2 named `first_fit` and `exact.loglinear`, which
`partition.py` and `loglinear.py` import; 0.3 names `AffinityRefused`'s
structured payload and `ComplexNormal`, which `graph_bridge.py` needs. A 0.2
install satisfies the import statements of the second pair and then fails at
the call. 0.4 names `observed_mask`, which is how the adapter presents a
`FlaggedNoise`. **0.5 names `local_block(..., priors=True)`** -- G15's third
block constructor, which `uncertainty.fisher_information(space=...)` now
delegates its prior curvature to instead of spelling it. A 0.4 install
imports fine and raises `TypeError: unexpected keyword argument 'priors'`
at the call, which is the shape a floor exists to turn into a resolution
error. **0.5.0 is on PyPI as of 2026-08-28**, so
`uv pip install 'bayesmith>=0.5'` resolves; this checkout nevertheless holds
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

**The floor is 89 and it gates at 88.5, and CI prints `FAIL` on every run
while passing.** Both halves measured 2026-08-28. `coverage` compares the
total **rounded to `[tool.coverage.report] precision`** digits, and `precision`
is not set here, so it defaults to **0**: `should_fail_under(88.96, 89, 0)` is
`False` because `round(88.96) == 89`. At `precision = 2` the same call is
`True`. So the effective floor is half a point below the declared one.

The confusing part is that **pytest-cov prints its own line from the
UNROUNDED number**, so every Coverage job on CI ends with

    FAIL Required test coverage of 89.0% not reached. Total coverage: 88.96%

and then exits **0** and is marked green — measured on three consecutive runs
at 88.99 %, 88.97 % and 88.96 %, all `success`, going back before this
programme. The line that prints is not the line that decides. Anyone reading
that log will conclude the gate is broken; it is doing exactly what it was
configured to do, and what is wrong is that two different numbers are
displayed by two different components.

Note also that **CI's coverage is legitimately lower than a local run's**
(88.96 % against 89.39 %) because CI skips more: `MomentRFI` cannot install
there — `momentrfi` depends on `momentemu`, which is on no registry the runner
reaches — so that step fails under `continue-on-error` and CI collects 628
skips against 566 here. The README figure is the LOCAL measurement, so it is
true where it was taken and unreachable on CI. Do not reconcile the two by
editing the README; they are measurements of different environments.

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
