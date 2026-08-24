:orphan:

# Agentic UI design

Status: **draft for review** — P0, P1, and the host side of P2 are implemented and
verified (typecheck-clean plus four integration tests); the independent-harness
distribution boots a real `dsh --profile rheplicant` without any API key, and the
`ui-analysis` client bundle builds and is served by the web runtime. Not yet
linked from the documentation index.

An **agentic surface** over rheplicant, built on DeepSeek Harness (`dsh`). A chat
session in which the agent drafts, edits, and reads back one analysis, and a
human form edits the same artifact. The config document is the single artifact;
running it is a capability seam with swappable transports.

## Requirements

Four decisions are fixed for this design:

1. **Both local and remote compute.** Run on the operator's own machine, or over
   SSH to a compute cluster, chosen per session without a restart.
2. **Both tool granularities.** A whole-document run, and per-exit shortcut
   tools for fast iteration.
3. **Hybrid build.** A GUI form and a chat agent edit the *same* config
   document, neither being a side channel.
4. **Keep the developer capability.** The code-agent tools (shell, filesystem,
   editor) survive, behind a separate agent preset rather than in the analysis
   preset.

## What this adds

Three layers exist already: the core library (`core` / `radio` / `inference`),
the config API (a declarative document executed by `run_document`), and a GUI
under construction. This design adds a fourth: an AI-analysis layer.

The load-bearing observation: **the config document is already the analysis
loop.** `runs:` is an ordered, named, dependent step list (`reuse:` only looks
backwards), and rheplicant's own checks — identifiability rank, joint χ²
convergence, r_hat — are the gates between steps. The agent never re-implements
analysis; it authors one document and reads the results and the diagnostics
back. "Chat" is the process; the document is the durable, auditable product.

## Architecture

```
browser (dsh Web GUI: brand, form editor, analysis node)
   │  Typert RPC / SSE
dsh Node runtime (TypeScript): plugins, tools, session log, approval, model selection
   │  compute seam  ctx.rheplicant  (routes by `transport`)
   ├── transport local: spawn a Python service, JSON-RPC over stdio
   ├── transport ssh:   ssh <host> <cmd>, JSON-RPC over the SSH channel
   └── transport http:  dial a long-lived service on the cluster
         │
rheplicant compute service (Python): validate / gates / run / schema
         │
rheplicant library (JAX + Equinox): unchanged
```

rheplicant's compute stays in Python (JAX, possibly a GPU). `dsh` stays in
TypeScript. The two meet at one thin, versioned contract: the four methods
below, whose payload is always the config document.

## Repository and distribution

Three repositories, one dependency direction. The AI layer is a separate
distribution that depends on both rheplicant and the harness; neither depends
back.

| repo | holds | depends on |
|---|---|---|
| `rheplicant` | core / radio / inference / config, unchanged | nothing |
| `deepseek-harness` | the harness, consumed as `@deepseek-ai/dsh-*` packages (published, or packed locally via `release:pack`), never forked | nothing |
| `rheplicant-agent` | the Python compute service and the `packages/rheplicant/` plugin group plus a bundle/profile | `rheplicant` (pip), `@deepseek-ai/dsh-*` (npm) |

The harness side runs as its own installation, independent of any dsh
development checkout: `release:pack` turns the `deepseek-harness` families into
tarballs, those tarballs plus the `rheplicant-agent` packages install into a
fresh harness home (its own `DSH_HOME`, profile, and `node_modules`), and a
`rheplicant` profile composes `dsh-base` + `dsh-headless` + the rheplicant plugin
group. Model choice is the user's settings document (`llm-pi-ai`), never a
DeepSeek default baked into the profile; the harness boots without any API key
and needs one only when a model is actually called.

The published `@deepseek-ai/*` npm packages are stale (`0.0.1-rc.1`, baseline
only), so the tarball path is the current consumption route; publishing current
versions replaces the tarballs with `npm install` and removes the checkout
dependency entirely.

A user who does not want the AI layer never installs `rheplicant-agent`.
rheplicant carries no server, no network, and no harness reference. The compute
service is a consumer of `rheplicant.config.run_document`, not an optional extra
of rheplicant: giving rheplicant a runtime server would widen its scope for one
optional layer.

The dependency is one-way, not zero-coupling: the compute service imports
rheplicant. The version anchor is the config document's `schema_version`, already
returned by `schema` and refused by `validate`/`run` on mismatch. When the schema
changes, `rheplicant` and `rheplicant-agent` release together — a rule of the
contract, not a convention.

## The compute seam

The seam copies `ctx.llm`'s multi-adapter registration model. Several providers
register on `ctx.rheplicant`, each under one or more transport names; a request
names a transport and the service routes to the provider that owns it. Swapping
local for cluster is a request field, not a restart, exactly as swapping models
is a `provider` field on `ctx.llm`.

TypeScript-shaped sketch of the DSH-side service:

```ts
type Transport = 'local' | 'ssh' | 'http'

interface ComputeService {
  registerProvider(transports: Transport[], provider: ComputeProvider): ComputeProviderHandle
  listTransports(): Transport[]
  validate(document: ComputeDocument, opts: ComputeOpts): Promise<ValidationReport>
  gates(document: ComputeDocument, opts: ComputeOpts): Promise<GatesReport>
  run(document: ComputeDocument, opts: RunOpts): Promise<RunOutcome>
  schema(opts: ComputeOpts): Promise<SchemaDocument>
}

interface ComputeOpts { transport: Transport; signal?: AbortSignal }
interface RunOpts extends ComputeOpts { runs?: string[] }   // subset by run name; default all
```

`registerProvider` is all-or-nothing per call and disposed with the calling
fiber, mirroring `ctx.llm.registerAdapter`; a transport name is owned by exactly
one provider, and registering a duplicate fails. A provider may own several
transports.

Consumers — the tools and the UI — depend on `ctx.rheplicant` only, never on a
provider. Replacing a provider changes one composition row and touches no tool.

## The wire contract

This is the reviewable core. The four methods are transport-agnostic: the
params/result shapes below are the JSON payloads over any channel, and no method
carries a `transport` field (that is a DSH-side routing key). `ComputeDocument`
is the JSON form of the existing config document; its grammar is owned by
rheplicant's schema and exposed by `schema`, and is **not restated here** — the
wire owns only the envelope around it.

### `validate`

| | |
|---|---|
| params | `{ document: ComputeDocument }` |
| result | `ValidationReport` |

`ValidationReport`:

| field | type | meaning |
|---|---|---|
| `valid` | `boolean` | whether the document passes grammar and preflight |
| `errors` | `ValidationError[]` | refusals, empty when `valid` |

`ValidationError`:

| field | type | meaning |
|---|---|---|
| `path` | `string` | JSON path to the offending node, e.g. `inference.parameters.g` |
| `code` | `string` | stable refusal code |
| `message` | `string` | rheplicant's own refusal text, verbatim |
| `check` | `string?` | owning check id when one applies |

The package's refusal messages are the authority; the wire carries them verbatim
and adds a machine-readable `path` and `code`.

### `gates`

| | |
|---|---|
| params | `{ document: ComputeDocument }` |
| result | `GatesReport` |

`GatesReport` — a structured projection of the existing `gates(...)` answer:

| field | type | meaning |
|---|---|---|
| `checks` | `CheckCost[]` | each check the document will run, and its mode and cost |
| `runs` | `RunCost[]` | each run, and what it needs (noise model, numpyro, x64) |
| `warnings` | `string[]` | text the projection cannot structure |

`CheckCost`:

| field | type | meaning |
|---|---|---|
| `check` | `'linearity' \| 'identifiability' \| 'prior_sensitivity'` | the check |
| `mode` | `'refuse' \| 'warn' \| 'report' \| 'skip'` | what a failure costs |
| `cost` | `string` | the resource it spends, in the package's own terms |

Field names mirror `gates(...)`; finalize them against the current implementation
before P0 lands, and keep this table generated from it rather than hand-edited.

### `run`

| | |
|---|---|
| params | `{ document: ComputeDocument, runs?: string[] }` |
| result | `RunOutcome` |

`RunOutcome`:

| field | type | meaning |
|---|---|---|
| `runs` | `RunEntry[]` | one per executed run, in declaration order |
| `tookMs` | `number?` | wall clock of the whole document |

`RunEntry`:

| field | type | meaning |
|---|---|---|
| `name` | `string` | run name |
| `kind` | `string` | exit kind, e.g. `plan.sample` |
| `status` | `'ok' \| 'failed'` | |
| `product` | `RunProduct?` | kind-specific result; absent when failed |
| `diagnostics` | `RunDiagnostics?` | always present for a sampled/fit run |
| `error` | `ComputeError?` | present when failed |

`RunProduct` is discriminated by `kind`, mirroring rheplicant's returned objects
with a stable JSON projection:

| product | for kinds | carries |
|---|---|---|
| `EstimateProduct` | `plan.estimate`, `conjugate.wiener` | `values: Record<string, ArrayOrScalar>` |
| `DrawsProduct` | `plan.sample`, `conjugate.gcr`, `nuts`, `npe` | `samples: Record<string, Array>`, `n_draw`, `mean?`, `std?` |
| `ReportProduct` | `fisher`, `identifiability`, `condition`, `score_directions`, `gradient`, `mmodes`, `predict`, `forward`, `optimize` | `fields: Record<string, unknown>` |

The complete exit catalog is owned by `schema`; this table lists the current
kinds for illustration.

`RunDiagnostics` — first-class, and the agent must read them, not only the
numbers:

| field | type | meaning |
|---|---|---|
| `converged` | `boolean?` | whether the solve/sampler settled |
| `rhat` | `number?` | split r_hat on the joint χ² |
| `rank` | `number?` | identifiability rank |
| `nullity` | `number?` | blind directions, when the rank test ran |
| `chi2` | `number \| number[]?` | joint χ², or its sweep trace |
| `notes` | `string[]` | provenance, seed, and limits the package already reports |

The UI renders diagnostics separately from the model's prose, so the authoritative
numbers never hide inside a generated explanation.

### `schema`

| | |
|---|---|
| params | `{ sections?: string[] }` |
| result | `SchemaDocument` |

`SchemaDocument`:

| field | type | meaning |
|---|---|---|
| `schemaVersion` | `string` | rheplicant config schema version |
| `jsonSchema` | `object` | the config grammar as JSON Schema |
| `exits` | `ExitDescriptor[]` | the run kinds and their keys |
| `operators` | `OperatorDescriptor[]` | the operator catalog |
| `transforms` | `TransformDescriptor[]` | the transform registry |

One schema, two projections: the GUI form and the tool descriptions both render
from `SchemaDocument`. No hand-maintained copy of the grammar may exist in the
UI or in tool prompts.

### `ComputeError`

Stable error codes, in the harness's own style:

| code | meaning |
|---|---|
| `TRANSPORT` | the channel (spawn / ssh / dial) failed |
| `INVALID_DOCUMENT` | grammar or preflight refused; carries the `ValidationReport` |
| `BUILD_FAILED` | document valid, build or postflight check refused |
| `RUN_FAILED` | a run raised |
| `TIMEOUT` | a run exceeded its budget |
| `NOT_FOUND` | `runs:` named a run that does not exist |
| `INTERNAL` | anything else |

`ComputeError` fields: `code`, `message`, and an optional `detail` carrying the
applicable report.

## Plugin inventory

All in a new workspace group `packages/rheplicant/`, each package one plugin,
each row one role. Roles follow the capability-seam split: Service Definition
(SD), Provider (P), Consumer (C), UI.

| package | role | owns |
|---|---|---|
| `dsh-rheplicant` | SD | the `Compute` interface, `ctx.rheplicant` key, the wire vocabulary types and `ComputeError`. No transport, no UI. |
| `dsh-rheplicant-transport` | library | JSON-RPC codec shared by the providers. Not a plugin. |
| `dsh-rheplicant-local` | P | `transport: local` |
| `dsh-rheplicant-ssh` | P | `transport: ssh` |
| `dsh-rheplicant-http` | P | `transport: http` |
| `dsh-rheplicant-tool-validate` | C | `rheplicant_validate` tool |
| `dsh-rheplicant-tool-gates` | C | `rheplicant_gates` tool |
| `dsh-rheplicant-tool-run` | C | `rheplicant_run` tool (whole document) |
| `dsh-rheplicant-tool-exits` | C | per-exit shortcut tools (`forward`, `fisher`, `plan.sample`, …) |
| `dsh-rheplicant-tool-schema` | C | `rheplicant_schema` tool |
| `dsh-rheplicant-ui-brand` | UI | replaces the official brand slot |
| `dsh-rheplicant-ui-document` | UI | the config-document form editor |
| `dsh-rheplicant-ui-analysis` | UI | a chat node rendering a run's steps and diagnostics |
| `dsh-rheplicant-ui-compute` | UI | the transport/endpoint settings card |

The SD plus any one provider plus the tools is the minimal runnable set; every
other package is droppable without touching the rest.

## The hybrid builder

One artifact, two editors. The config document lives as a file in the session
workspace, and:

- the form (`ui-document`) edits it section by section, producing only valid
  fragments;
- the chat agent edits it through `rheplicant_run` with an inline document, or by
  rewriting the file;
- `validate` gates both, so neither editor has a path that skips the schema.

The form lowers the agent's failure surface on what it can model; the chat
covers the freedom the form does not express. Both write the same document, so a
session is auditable and each round resumes from the last document rather than
from a fresh description.

## Agent presets

Two per-session presets over the same web profile, not a code branch:

| preset | mounts |
|---|---|
| `analysis` (default) | `rheplicant_*` tools, read-only filesystem if desired, plan mode. No shell, no editor — the agent designs and reads analyses, it does not modify code. |
| `developer` | everything in `analysis`, plus `tool-bash`, `tool-fs`, `tool-str-replace-editor`, and the web tool, for people who also change rheplicant code or scripts |

Switching presets is a per-session choice; the brand and persona are uniform
across both.

## Branding and cuts

The brand slot (`ui-brand-official` in the shipped web bundle) is replaced by
`ui-brand-rheplicant`; the `system-prompt` persona becomes an analysis assistant,
not a coding agent. The base bundle's code-agent rows are dropped from the
`analysis` composition and restored only inside the `developer` preset. The
model default (`agent-default-model`) is kept neutral, and `llm-pi-ai` remains
the multi-provider surface, so no single API is assumed.

## Long tasks and JAX compilation

JAX JIT compilation is expensive and amortized only by a long-lived process, so
the interactive transports (`http` against a daemon; a managed local daemon)
reuse the compiled cache across sessions while a per-call spawn does not. The
provider owns the daemon's lifecycle — one asynchronous operation, one lifecycle
controller — rather than leaking it.

NUTS, `npe`, and large GCR runs are minute-scale. Asynchrony is a DSH-side
concern, not a wire concern: `rheplicant_run` stays synchronous on the wire, and
long runs execute inside a background job (`dsh-jobs` + `job_*`), with the result
delivered into the analysis chat node. `gates` exists precisely to tell the user
a step will take minutes before it is committed to a background job.

## Reproducibility and the session log

Model-visible means logged. `rheplicant_run` records a durable `rheplicant/run`
session event carrying the document, the transport, and the `RunOutcome`, so
fork, replay, and audit all derive from the log. The document already carries
`runtime.seed`; provenance and one-seed-reproduces-one-run hold across the
boundary unchanged.

## Phasing

| phase | delivers | accepts |
|---|---|---|
| P0 | `dsh-rheplicant` (SD), `dsh-rheplicant-local` (P), `dsh-rheplicant-tool-run` (C) | a chat turn runs a `forward` and reads the result back |
| P1 | `tool-validate`, `tool-gates`, `tool-schema`, `ui-analysis` | "check identifiability, then estimate the gain" becomes a step list with diagnostics, approvable before expensive runs |
| P2 | `dsh-rheplicant-ssh`, `dsh-rheplicant-http`, `ui-compute`, `ui-document` | local ↔ cluster switches per session; form and chat edit one document |
| P3 | `ui-brand`, `analysis`/`developer` presets, jobs-backed long runs, daemon lifecycle | a cold start shows the analysis console, long runs do not block |

P0, P1, and the host side of P2 are implemented and verified. The verification
runs inside a dsh checkout: a client test folds a synthetic `rheplicant/run`
event into a node, a host test runs a real `forward` and asserts the emitted
event, and an assembled-turn test drives a mock-LLM tool call to the same end.
The `ui-analysis` client bundle also builds and is served by the web runtime at
`/plugins/@deepseek-ai/dsh-client-rheplicant-ui-analysis/client.js`; rendering it
in a live transcript still needs a session that carries a `rheplicant/run` event
(host tools in a web profile plus a model turn or replay). `ui-compute` and
`ui-document` remain.

## Open decisions

1. Whether the compute provider set is deployment-fixed or every transport is
   offered to every session (recommended: offer all registered transports, pick
   per session like a model).
2. The exact `GatesReport` field names, finalized against the current `gates(...)`
   implementation.
3. How the document file in the workspace is rendered and versioned per turn.
