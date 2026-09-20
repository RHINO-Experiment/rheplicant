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
alongside it, which its editable hook needs), **`bayesmith`** in the declared
range below, and the **`gui-react`** extra — which is in this list because
CI did not have it and nothing said so: its `httpx2` is what
`tests/gui/test_session_api.py` skips on, and its absence cost 118 statements
of GUI coverage without a single test failing.

It also means **`pyuvdata`** (the `uvbeam` extra) and **`pygdsm`**, both on
PyPI, which `tests/config` importorskips; **`panel`** (the `gui-panel` extra),
which the seven GUI spike tests in `tests/gui/test_panel_spike.py` and
`tests/gui/test_candidate_parity.py` importorskip, and which leaves with the
spike when it is removed as scheduled; **`MomentRFI`** with **`MomentEmu`**,
below; **`matplotlib`**, which rhino-cal's `gcr.data_processing` imports; and
the **Node toolchain**, `npm` on `PATH` and `npm ci` run in
`tools/config_gui_spike/react`. Without `node_modules`,
`tests/gui/test_typescript_gates.py` skips its three gates, and the closure
case in each of `tests/gui/test_e2e_typecheck.py` and
`tests/gui/test_react_test_typecheck.py` fails with `FileNotFoundError:
'node_modules/.bin/tsc'` (measured 2026-09-19), so a missing toolchain shows
as two red tests rather than as skips. Two sets of tests are opt-in by
environment variable because their data is not redistributable:
`RHEPLICANT_RHINO_BEAMS` names a directory of RHINO CST beam exports
(`tests/radio/test_beams.py`), and `RHEPLICANT_RHINO_CAL` names a rhino-cal
checkout (`tests/radio/test_ingestion_vs_reference.py`).

**MomentRFI and MomentEmu install from git, together.** Neither is on PyPI,
and MomentRFI declares MomentEmu, so naming MomentRFI alone does not resolve;
one command with both does:

```bash
uv pip install --python .venv/bin/python "MomentEmu @ git+https://github.com/zzhang0123/MomentEmu" "MomentRFI @ git+https://github.com/zzhang0123/MomentRFI"
```

With both installed, `tests/radio/test_flagging_momentrfi.py` runs all 13 of
its tests; without MomentRFI ten of them skip.

**What CI requires.** The Suite and Coverage jobs fail when any of seven
import names is absent: `h5py`, `rhino_cal_jax`, `limtod_jax`,
`numpyro`, `pyuvdata`, `pygdsm` and `MomentRFI`, the workflow-level
`REQUIRED_IMPORTS` in `.github/workflows/test.yml`. They also install the
`gui-react` extra and the Node toolchain without checking either by name.
They install neither `panel` nor `matplotlib`, so on CI the seven spike tests
skip, and neither opt-in variable is set.

The three `DataHandler` comparisons in
`tests/radio/test_ingestion_vs_reference.py` need both MomentRFI and
matplotlib, because rhino-cal's `gcr` imports each, as well as
`RHEPLICANT_RHINO_CAL`. Measured 2026-09-19 in a scratch venv holding all
three: the file's five tests pass; with MomentRFI absent the three skip on
`No module named 'MomentRFI'`.

**bayesmith is declared `>=0.10,<0.11`, and the range holds two numbers.** The
capability floor is 0.6, the highest release whose surface this package uses:
0.2 `first_fit` and `exact.loglinear`; 0.3 `AffinityRefused`'s structured
payload and `ComplexNormal`; 0.4 `observe(..., mask=)` and the node field
`Probabilistic.observed_mask`; 0.5 `local_block(..., priors=True)`; 0.6
`marginal.chain.smooth` assembled as a square root, whose 0.5 spelling returns
`nan` on a stiff chain. Below 0.5, `rheplicant.inference` fails at import,
because `bayesmith.marginal` first ships in 0.5. A 0.5 install imports and
fails only in behaviour, and the 0.4 and 0.5 keyword arguments are each a
`TypeError` at the call on the release below.
`tests/test_bayesmith_floor.py` asserts each level by capability, not by
version. The range starts at 0.10 because the stable baseline relies on
bayesmith 0.10's stability contract and is tested only against it, and it is
closed at 0.11 because a pre-1.0 minor may move the deep module paths this
package imports. 0.10 moved one: `bayesmith.optimize` became a package, so
`from bayesmith.optimize import minimize` still resolves while the module file
that name used to live in is gone.

**0.10.0 is a local release and not on PyPI** (built 2026-09-20; 0.9.0 was not
published either, and the index stopped at 0.8.0 when it was checked on
2026-09-19). Check both artefacts' sha256 against
`../bayesmith/runs/t004/release-manifest.json`, which sits one level above the
artefacts rather than beside them, then

```bash
uv pip install --python .venv/bin/python --no-deps ../bayesmith/runs/t004/dist/bayesmith-0.10.0-py3-none-any.whl
```

Any install that resolves this package's dependencies needs
`--find-links ../bayesmith/runs/t004/dist`. The wheel was built at bayesmith's
`8aefb3e` and that repository's HEAD has moved on; the manifest's shas describe
the build, so an install checked against them is reproducible even though the
wheel is byte-stale against HEAD. The wheel replaces
the editable install from `../bayesmith` this checkout used while the two
repositories were developed against each other: an editable install runs
whatever the sibling working tree holds and reports the version its metadata
was written with (0.2.0 against 0.9.0 source, measured 2026-09-19), so a
seam result could not say which bayesmith it tested. Record the wheel's
sha256 beside seam results. bayesmith's runtime dependencies, including its
`numpyro>=0.15,<0.22`, are already here. Without it `rheplicant.inference`
does not import at all, so this one fails loudly rather than as silent skips.

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

**The floor gates where it says it does, and that took a setting.** `coverage`
compares the total **rounded to `[tool.coverage.report] precision`** digits.
Unset, `precision` defaults to **0**, and an 89 floor then gated at 88.5:
`should_fail_under(88.96, 89, 0)` is `False` because `round(88.96) == 89`.
Worse, pytest-cov prints its own line from the UNROUNDED total, so every
Coverage job ended with

    FAIL Required test coverage of 89.0% not reached. Total coverage: 88.96%

and then exited **0** and was marked green — measured on three consecutive
runs at 88.99 %, 88.97 % and 88.96 %, all `success`. The line that printed was
not the line that decided.

`precision = 2` has been set since `a7bbd4f` (2026-08-28), so the number that
prints and the number that decides are now the same one, and the floor is the
declared 89. The paragraph above is kept because the failure is invisible when
it happens: a green job whose own log says FAIL.

Note also that **CI's coverage was lower than a local run's** (88.96 % against
89.39 %), and the reason first written here — that `MomentRFI` cannot install
on the runner — was **wrong**. `MomentRFI` was absent in BOTH environments
then, so it explained no difference at all; that sentence was copied from the handover
rather than measured, which is exactly the tax this file keeps recording.

**Measured 2026-08-28, per file.** The whole gap is 132 statements and it has
two parts. `platform_darwin.py` (67) and `platform_linux.py` (57) are the
irreducible half — no single machine covers both, netting 10 against CI. The
other 118 are the GUI modules, `gui/api.py` alone falling from **95.02 % to
64 %**, and they trace to one missing package: `tests/gui/test_session_api.py`
opens with `pytest.importorskip("httpx2")`, and **`httpx2` lives in the
`gui-react` extra while CI installed `gui`**. `gui` deliberately excludes it —
its own comment says `httpx2` is "solely for Starlette's test client" and must
not become a runtime requirement — so the fix is for CI to install the test
dependency, not to move it into the shipped extra. Both CI jobs now install
`gui-react` alongside, and `httpx2` is named in the complete-environment list
above.

The README figure is the LOCAL measurement. Do not reconcile the two by
editing the README; check first whether they are measuring the same
environment, because the last time they diverged the cause was an install
line and not the code.

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

<!-- agent-workspace:begin -->
## Cross-agent task handoff
- Keep shared project rules here. Keep portable skills in `.agents/skills`.
- When a user supplies a tracked task ID, read `.agents/skills/task-handoff/SKILL.md`
  and run `python3 .agents/skills/task-handoff/scripts/workspace.py context TASK_ID` before editing, even in a resumed conversation.
- Confirm the project/worktree, current code, task revision and latest handoff.
  Prior chat history is not the source of truth for the current workspace.
- At a milestone or before switching tools, checkpoint the goal, changes,
  decisions, actual test commands/results, blockers and next action. Use the
  task-handoff CLI and its expected-revision check; do not edit task.json directly.
- A recorded test result is a report, not independent verification. Never claim
  tests passed unless actually run. Do not auto-commit, push or reset for a handoff.
- Work sequentially in one worktree. Use separate worktrees for concurrent edits.
- Do not merge, rewrite or symlink native Codex/Claude/DSH session databases.
- Keep credentials and private transcripts out of handoffs. Local `.agent-state`
  is ignored by Git but is not encrypted or automatically redacted.
<!-- agent-workspace:end -->
