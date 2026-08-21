# Configuration workbench

The browser workbench is another view of a rheplicant YAML document, not
another configuration format. Its Model, Config, Execute and Results
workspaces, diagnostics, previews, output designer and jobs are projections of
the exact accepted YAML bytes. Every accepted control edit goes through a
revision-checked Python transformation and returns the complete YAML string.
Navigation, selection and an unsubmitted raw draft remain browser view state.

## Install and start

Install the selected FastAPI + React editor separately from the core package:

```bash
pip install "rheplicant[gui]"
rheplicant-gui
```

Open `http://127.0.0.1:8000/`. The launcher serves the packaged production
frontend and the API from one origin; Node.js is not needed in an installed
wheel. Use another loopback address or port when required:

```bash
rheplicant-gui --host ::1 --port 8765
```

The process owns an in-memory session registry. Restarting it discards editor
history and job records. Load and Save are explicit browser actions: loading a
file replaces a session only after the server accepts it, and saving downloads
the current YAML before the server marks that revision clean. The application
does not watch or silently rewrite a file on disk.

## The four workspaces

The persistent header shows the accepted revision, saved/dirty or active-draft
state, validation-current/stale state and the trusted-execution boundary.
Quick and Full detail appears in Diagnostics and Execute. The four workspace
tabs replace the main panel without changing YAML:

- **Model** shows one server-rendered signal-path graph at a time. Full path
  and Processing are interactive; Compare fits read-only base and variant
  diagrams into their containers. Select a node by mouse or keyboard and use
  the adjacent inspector for supported edits. For a node that holds exactly
  one operator instance the inspector also offers the same settings as typed
  fields -- one control per field, carrying the field's own dimension, its
  accepted unit spellings and its enum members. A unit select rewrites the
  unit and never the number, because `celsius` is affine and a silent
  conversion would produce a plausible wrong answer. Fields written in a form
  no control can represent, and nodes configured through `python:`, `from:`,
  `at:` or `compose:`, say so and keep the raw settings JSON, which is
  rendered for every editable node regardless.
- **Config** shows one of the 12 server-projected sections at a time. Missing
  required choices come first, followed by current and optional values. Safe
  select, toggle, integer and text controls are applied by Python; complex
  values link to the YAML drawer.
- **Execute** puts Quick checks and the latest Full validation beside bounded
  previews, output readiness, product/report configuration and the explicit
  Validate, Preview forward, Run, Compare and Benchmark actions that the
  document declares.
- **Results** retains terminal evidence grouped into current and stale
  revisions. It shows bounded result summaries, publication state and
  identity-checked audit links without changing or discarding a draft.

Diagnostics and YAML are global drawers. Active jobs are visible from every
workspace; full history and result detail live in Results.

## First preview

The starter is a bounded, file-free YAML document with a simple model and one
Forward declaration. The first-use checklist reports that the starter was
accepted, required choices are complete, Quick checks are clean, and Forward
preview has not yet run. Open Execute and choose **Preview forward**. The first
explicit job asks you to acknowledge that trusted code, server paths and
compute run as the server account. A current successful preview completes the
checklist; the guide can be reopened from Help.

This onboarding is guidance, not a wizard and not scientific metadata. It does
not rewrite the document or submit a job automatically.

## YAML and field drafts

The workbench permits one unsubmitted draft at a time: complete raw YAML, or a
raw field/graph control value tied to its base revision. The header names the
active draft and its blocking reason; the owning editor supplies its actions.
The YAML drawer offers **Apply YAML edit** and **Discard draft**. A field row
offers **Apply field** and **Discard field**, while the node inspector offers
**Apply configuration to _node_** and **Discard graph draft**. Load, Save,
Undo, Redo, other accepted mutations and job submission are disabled with a
visible reason; workspace navigation remains available.

Applying through the owning editor sends the draft and expected revision to
Python. Success replaces the accepted session with the complete returned YAML
and projections. Invalid YAML stays in the drawer beside one inline diagnostic
while the last-good projections remain visible. A YAML revision conflict
retains the raw YAML draft and offers **Copy draft**, **Refresh accepted YAML**
and **Discard draft**; it is never silently retried or overwritten.

A catalog `file` widget is labelled **Server path** and is text, not a browser
upload. The server account reads that path. Generic mappings, lists, resources,
nodes, groups and products remain YAML-first rather than being guessed into
browser-owned types.

## Checks, outputs and actions

**Quick** checks update from accepted text without running the priced model.
**Full** validation is an explicit job and is reported as not run, queued,
running, current, stale, refused or error. Findings can take you to the owning
Config section or open YAML at the exact path.

Execute keeps the workflow progressive: choose a target, resolve its safety
state, enable only the products needed, and expand their controls. All 22
product selectors remain searchable; report design appears only after
**Write report** is enabled. Existing-owned, foreign, ambiguous-recovery,
unsafe and unavailable targets keep the server's exact explanation. Run is
available only when the document and target satisfy the existing backend
contract, and an identical queued/running kind, revision and digest cannot be
submitted twice.

Jobs poll automatically while queued or running. The jobs-only response must
match the accepted session id, revision and YAML digest; a late or stale
response is ignored and never replaces YAML or a draft. Polling pauses while
the page is hidden, backs off after failures, offers Retry/Refresh, and stops
when all jobs are terminal.

Results distinguishes current success, stale evidence, scientific refusal,
internal error and unsafe output. A stale result says which revision produced
it and offers **Re-run**. Audit links remain bound to the selected full job
identity and open separately.

## Keyboard and responsive use

- Move among wide vertical workspace tabs with Up/Down, and among compact
  horizontal tabs with Left/Right. Normal Tab navigation also works.
- Model nodes use graph order with arrows, Home and End; Enter or Space selects
  a node. Full path and Processing retain Fit, 100% and zoom controls. Compare
  is fitted and read-only; Reset view returns to Full path.
- Config section buttons remain in normal Tab order and identify the active
  section. Drawers trap focus while open and return it to their opener when
  closed.

The workbench is tested at 1440x1000, 1024x768, 768x900 and 640x900. At wide
sizes the workspace rail and inspector flank the main surface; at smaller
sizes tabs become horizontal and inspectors, YAML and Diagnostics become
drawers. The page itself does not scroll horizontally; only an editable graph
may use its labelled horizontal scroll area.

Release screenshots cover six representative states: first use, graph
selection, required fields, invalid YAML, output setup, and completed
job/results evidence. The same semantic journeys run against the source
launcher and a fresh installed wheel in all four viewports; the canonical
screenshots are wide captures, with light/dark, forced-colour, keyboard,
overflow, accessibility, console and network checks around them.

## What the workbench exposes

- the complete 33-node canonical signal path, with graph-order keyboard
  navigation, base/variant diagrams, many-node ordering, composition and
  placement controls;
- a closed form projection of the live config registries, with missing and
  conditional fields visible without replacing the YAML source of truth;
- the public pre-flight finding ledger, section badges and preset diff on every
  valid edit; invalid YAML remains in the textarea beside its diagnostic;
- continuous graph/axis/shape previews and explicit priced Validate and
  forward-preview jobs;
- explicit Run, Compare and Benchmark jobs through the same command and
  orchestration surface as the CLI;
- exact requested and labelled preset-merged output views, all 22 product
  selectors, report design, predicted paths, clobber/recovery state and
  identity-checked terminal audit links.

Run remains disabled while text-level refusals exist. Validation, preview and
result evidence is bound to the YAML revision and digest that produced it and
is marked stale after a relevant edit.

## Security and trust boundaries

`rheplicant-gui` binds to `127.0.0.1` by default. It has **no authentication,
authorization, TLS termination, CSRF boundary or multi-user isolation**. A
non-loopback host is refused unless `--allow-remote` is supplied:

```bash
rheplicant-gui --host 0.0.0.0 --allow-remote
```

That flag is an acknowledgement, not a security feature. Put an authenticating
reverse proxy and process/filesystem isolation in front of the application if
remote access is genuinely required. Do not expose it directly to an untrusted
network.

The bounded loader rejects duplicate keys, unsafe YAML tags and non-mapping
documents. React renders ordinary YAML-derived values as text. The one raw-HTML
boundary is the SVG generated by rheplicant's own graph renderer, whose labels
are escaped before they enter the document.

Those properties do **not** make an arbitrary configuration untrusted code:

- `python:` and named plugin targets import and may execute code in the server
  process. They are an explicit trusted-code boundary, as in the CLI.
- Resource paths are read by the server and execution jobs can spend CPU,
  accelerator memory and wall time.
- Output paths name server-side destinations. Plan 4's descriptor-safe
  preflight, transaction and recovery rules apply, but the editor does not
  turn an otherwise unauthorized filesystem path into an authorized one.
- Sessions and jobs share one process and one account. They are not tenant
  sandboxes, quotas or durable queues.

Treat a YAML file like a program submitted to the account running the editor.
Review its plugins, paths, runs and outputs before pressing an explicit job
button.

## Development build

The TypeScript source lives in `src/rheplicant/gui/react/`; the reproducible
spike toolchain remains under `tools/config_gui_spike/react/`. Rebuild the
tracked production assets with:

```bash
npm --prefix tools/config_gui_spike/react run build:production
```

The build emits only `index.html` plus hashed JavaScript and CSS under
`src/rheplicant/gui/static/`. Both a direct wheel and a wheel rebuilt from the
sdist must contain the identical closed file list. The installed-wheel test
starts `rheplicant-gui`, creates a session through the API and fetches the
actual hashed asset, so a source-only frontend cannot pass release verification.
