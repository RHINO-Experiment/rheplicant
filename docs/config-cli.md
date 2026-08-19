# Configuration command line

RHEPLICANT has one installed command and three forms:

```text
rheplicant validate CONFIG [--base-dir DIR]
rheplicant run CONFIG [--base-dir DIR]
rheplicant script CONFIG [--base-dir DIR] [-o OUTPUT.py]
```

`CONFIG` is a YAML file or `-` for exact bytes read from standard input. With
`-`, `--base-dir` is required so relative data and resource paths have a
defined anchor. A file input derives that anchor from the lexical file name;
supplying a different `--base-dir` is refused. A symlink therefore keeps the
directory in which the symlink was named while provenance also records its
resolved target.

## Validate, run, and exit status

`validate` safely parses the source, applies package presets, establishes the
declared JAX runtime, imports named plugins, and validates the base plus every
variant. Every run kind parses its options before any run executes. Validation
does not create an output parent, lock, journal, result, or failure directory.

`run` performs the same preparation, executes the base schedule in declaration
order, and publishes the audit tree plus any requested scientific products. A file named `config.yaml`
defaults to `config.results/`; an explicit relative `outputs.dir` is resolved
against the config file's directory. `run -` needs an explicit `outputs.dir`.

| Status | Meaning |
|---|---|
| `0` | validation, execution, or script generation succeeded |
| `2` | usage, YAML, or configuration refusal |
| `1` | unexpected package or internal failure; a traceback is printed |

`outputs.stdout` is `none`, `summary`, or `verbose` and defaults to `summary`.
It controls success/progress text only. Warnings and errors always use standard
error.

## Minimal workflow

```bash
rheplicant validate observation.yaml
rheplicant run observation.yaml
```

A package preset is an exact, hashed YAML layer rather than a replacement for
the user document. `rhino_v1` intentionally leaves instrument-specific facts
for the user:

```yaml
schema_version: 1
defaults: [rhino_v1]
runtime: {jax_enable_x64: true, platform: cpu, seed: 20260817}
outputs:
  dir: results/rhino-night-1
  clobber: false
runs:
  - {name: simulate, kind: forward}
```

The remaining observation, beam-orientation, and model facts still have to be
declared; `defaults: [rhino_v1]` alone is not a runnable observation.

## Audit and scientific output trees

A successful run publishes:

```text
config.results/
├── .rheplicant-results.json
├── config.input.yaml
├── config.resolved.yaml
├── variants/<encoded-name>/config.resolved.yaml
├── products.json                         # when a product/report is requested
├── runs/<encoded-run>/arrays.npz         # example run product
├── layers/base/assembly.json             # example layer product
├── report.txt                            # optional report
├── provenance.json
└── diagnostics.json
```

A refusal or internal error after publication trust is established uses a
non-clobbering sibling such as `config.results.refused-<stamp>-<pid>/` or
`config.results.error-<stamp>-<pid>/`, with the same mandatory metadata and
every resolved layer that actually completed. No later boundary or file is
claimed. An ambiguity during recovery preserves every named path and starts no
second transaction.

`outputs.write.config`, `provenance`, and `diagnostics` are mandatory and
cannot be false. Product files and `products.json` join those bytes inside the
same recoverable transaction; materialization finishes before staging and
never writes directly to the destination. Product refusal therefore leaves no
partial success directory. `validate` parses the same requests but executes
and materializes none of them.

## Scientific products

Each scientific key under `outputs.write` is either `true` (its default format,
all compatible executed runs) or a closed mapping. `false` is an error rather
than a silent no-op. Every mapping accepts `format:` and `runs:`; `aux` and
`taps` additionally accept `keys:`, while `signal_paths` accepts `themes:`
(`light` or `dark`). Run and variant names are UTF-8 encoded before they become
path components; raw names never enter the output path.

```yaml
outputs:
  dir: results/night-1
  write:
    arrays: true
    parameters: {runs: [fit]}
    taps: {runs: [simulate], keys: [before_adc, after_adc]}
    signal_paths: {format: svg, themes: [light, dark]}
    chains: {format: netcdf, runs: [posterior]}
  report:
    rows: [fit, posterior]
    columns: [mean, std, seconds]
    reference: posterior
    relative: [mean_sigma, width_ratio]
    format: [json, text]
```

The selectors and default formats are:

| Default | Selectors |
|---|---|
| NPZ | `arrays`, `aux`, `taps`, `estimates`, `parameters`, `draws`, `losses`, `gradients`, `covariance`, `prediction_bands`, `posterior_predictives`, `scores`, `training_history`, `chains` |
| JSON | `assembly`, `identifiability`, `recovery`, `timings`, `compare`, `benchmark` |
| text | `refusals` |
| SVG | `signal_paths` (also `html` or `mermaid`) |

`chains` may instead request `netcdf`; that spelling is refused if the optional
writer is unavailable and never falls back to NPZ. An explicitly filtered run
that cannot produce a selector is refused. Without `runs:`, compatible runs
are emitted, incompatible ones appear as omissions in `products.json`, and the
request is refused if no run can produce it. `assembly` and `signal_paths` are
per prepared layer and do not accept run filters. `timings` reads recorded wall
times, `refusals` reads captured `expect: refuse` outcomes, and neither reruns a
run. Likewise, a report reads only prior results and timings.

`products.json` is canonical JSON validated against the packaged strict
`products-v1.schema.json`. It records each request, every truthful omission,
and every file's relative path, selector, run kind, format, byte count, SHA-256
digest, and metadata. Numeric archives are deterministic NPZ without object
dtype or pickle; JSON records reject non-finite numbers.

## Trusted executable code

Named `plugins:` and `python:` targets are explicitly trusted executable code.
For an invocation that reaches either boundary, the CLI prints exactly:

```text
warning: trusted plugin/python code may perform unobserved filesystem I/O
```

The validate zero-mutation guarantee covers RHEPLICANT, its bootstrap, and its
output manager. It cannot cover filesystem side effects performed privately by
named plugin or Python code. Provenance records each discovered executable
boundary with `unobserved_io: true`; it never claims that the whole host
process was side-effect-free.

## Clobber, recovery, and filesystem safety

`outputs.clobber` defaults to false. With `true`, RHEPLICANT replaces only a
mode-0700 directory owned by the effective user whose canonical mode-0600
ownership marker proves it was created by RHEPLICANT.

Before A34 clobber authorization, `run` acquires the persistent per-target
lock and recovers any proved prior transaction. The same platform adapter then
proves access/default ACLs and atomic no-replace support without a write,
revalidates root-to-parent ancestor identity and rename protection, and budgets
every target, failure, journal, staging, backup, run, and variant component
against the leased filesystem `NAME_MAX` before the first directory or lock
write. No replacing-rename fallback exists.

An ordinary terminal metadata or publication failure is recovered under that
same lease before an error sibling is attempted. The retry needs a fresh
post-recovery publication view. If recovery is ambiguous, all paths are listed
and preserved and no second transaction begins. The lock file persists after
close: unlinking it would permit two processes to hold different lock inodes
for the same target.

## Generated programs

Generate to standard output or publish an absent mode-0600 file:

```bash
rheplicant script observation.yaml > run-observation.py
rheplicant script observation.yaml -o run-observation.py
python run-observation.py
```

The program base64-embeds the exact source and selected preset bytes, hashes,
and source path facts, then calls the shared
`_rheplicant_bootstrap.run_embedded_config` entry. It contains no copied model
builder or executor. `-o` uses a same-directory fsynced temporary and an atomic
no-replace link; an existing destination is always refused, independently of
`outputs.clobber`.
