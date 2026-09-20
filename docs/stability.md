# Stability and capabilities

What this package promises, what it does not, and how you can tell which is
which without reading the source.

Every number on this page is checked against the code by
`tests/test_stability_page.py`. If one of them is wrong, that test is red —
the page cannot drift quietly, which is the only reason it is worth writing
numbers down at all.

## The layers, and which may depend on which

```text
_rheplicant_bootstrap      depends on nothing in this project
rheplicant.core            -> bootstrap
rheplicant.radio           -> core
rheplicant.inference       -> core
rheplicant.config          -> bootstrap, core, radio, inference
rheplicant.gui             -> bootstrap, core, config, radio
```

Two properties matter more than the rest:

`radio` and `inference` do not import each other. A forward model and a
likelihood layer are siblings, so you can use the instrument model without the
Bayesian half and point the Bayesian half at something else.

`gui` does not import `inference`. It reaches `config`, and `config` reaches
`inference`.

There is one cycle and it is deliberate: `config` imports the bootstrap at
module scope, and the bootstrap's command half imports `config.orchestration`
at call time. Importing it any earlier would put JAX behind
`rheplicant --help`.

**The bootstrap is two layers in one package**, and which module is in which
is now written down rather than implied. Seven modules are the **command
half** — `__init__`, `__main__`, `cli`, `entry`, `execution_environment`,
`gui_worker` and `script` — and the other forty-seven are the **foundation**.
The seam is what a module costs at import: the foundation is read before
`rheplicant` is importable and must stay that way, while a command module
drives the package once it is. No foundation module imports a command one.

The package root is in the command half by role and not by cost, and it is the
one member the foundation may import — it has to be, because importing any
submodule runs it first. Both of its entry points defer into function bodies,
and `tests/test_import_direction.py` asserts that they stay deferred, so the
exemption proves itself instead of being taken on trust.

`tests/test_import_direction.py` pins all of this, in both directions — a
dependency that is allowed and unused is deleted, so the table is the whole
truth rather than a ceiling.

## What counts as public

The importable surface is `__all__`, in six namespaces, **338 names**:

| Namespace | Names |
|---|---|
| `rheplicant` | 25 |
| `rheplicant.core` | 31 |
| `rheplicant.radio` | 61 |
| `rheplicant.inference` | 106 |
| `rheplicant.config` | 41 |
| `rheplicant.gui` | 74 |

Every one is pinned by name in `tests/test_public_surface.py`. Adding a name
is a promise; removing one breaks whoever believed the last promise. Both show
up as a diff in review.

A leading underscore means private, and a private name crosses a package
boundary only through a route recorded in
`tests/test_cross_package_privates.py` — eight routes today. If you are
importing an underscore name from another package and it is not on that list,
it will not stay importable.

## The four capability levels

Every shipped capability — an operator, a sky model or a sky projector —
declares one, as a `maturity` class variable beside `requires` and `provides`.
There is no default: a class that does not declare a level raises rather than
inheriting a claim about its physics.

| Level | What you may conclude |
|---|---|
| Maintained | Documented contract, regression-tested within its declared domain. |
| Experimental | Usable, with stated limits and no general validity guarantee. Anything built on an upstream Experimental surface inherits this. |
| Placeholder | Correct plumbing and shapes, stand-in physics. The contract is real and tested; the numbers are not predictions. |
| Unavailable | Named by the schema or API and refused with a typed error. |

Today, of **35** shipped capabilities:

| Level | Count |
|---|---|
| Maintained | 14 |
| Experimental | 2 |
| Placeholder | 19 |

Read them at runtime rather than from this table:

```python
from rheplicant.radio import at_level, capabilities
from rheplicant import Maturity

capabilities()                      # every capability -> its level
at_level(Maturity.PLACEHOLDER)      # the names whose numbers are stand-ins
```

**A level belongs to a capability on a surface, not to a capability alone.**
`NeuralOperator` is a working operator from Python and is refused from a
configuration document with a typed error naming its capability — one class,
two answers. So the class variable records how far the *implementation* has
been taken, and `Unavailable` describes a *surface's* answer. No class is ever
`Unavailable`; the eight document keys that are live in
`_rheplicant_bootstrap.capability.REGISTRY`.

A placeholder is not a bug and not a warning. Placing one is a reasonable
thing to do — the contract, the ordering and the differentiability are real,
which is what makes the pipeline worth assembling before the physics arrives.
What you must not do is read its numbers as a prediction.

## The plugin registration protocol is not public yet

`config/` has five registration hooks — `register_kind`, `register_reader`,
`register_form`, `register_derivation` and `register_formula_checked`. **None
of them is on any `__all__`.**

That is the honest status rather than an oversight being papered over. A
protocol with no public surface has nothing to version, so there is no
`PLUGIN_API_VERSION`: publishing one would declare a contract that no
supported import reaches. Which registries should become public is an open
question for after this baseline.

If you are writing something that calls one of these, you are reaching into
internals, and they may move in a minor release. `docs/config-resources.md`
mentions `register_reader(..., array=False)` while describing how the shipped
readers are built; that is a description of internals, not an invitation.

`tests/config/test_plugin_protocol.py` pins the five as private, so making one
public becomes a decision someone takes rather than a line that slips through.

## The GUI's HTTP API is internal

Twenty routes under `/api`, consumed by the React client bundled beside
them in `src/rheplicant/gui/static/`. It is versioned **with that client**, not
on its own, and carries no compatibility promise to any other consumer.

The OpenAPI document reports the package's version rather than FastAPI's
default `0.1.0`, which reads as a declared API version and was a placeholder,
and its description says the API is internal so a reader of the generated
documentation does not have to guess.

The route table is snapshotted in `tests/gui/golden/http_routes.json`. An
internal API is pinned for almost the opposite reason a public one is: not so
it cannot change, but so a change is VISIBLE — the only supported client is
checked in and rebuilt by hand, and a route removed without rebuilding the
bundle is a 404 nothing else would catch.

## External contract versions

| Contract | Version | Where |
|---|---|---|
| Configuration document schema | `"1"` | `json_schema()["schemaVersion"]` |
| Audit bundle integrity manifest | `1` | `_rheplicant_bootstrap.audit.integrity.INTEGRITY_FORMAT_VERSION` |
| Audit provenance document | `1` | `provenance-v1.schema.json`, `format_version.const` |
| Audit diagnostics document | `1` | `diagnostics-v1.schema.json`, `format_version.const` |
| Scientific product manifest | `1` | `products-v1.schema.json`, `format_version.const` |
| Capability record | `1` | `capabilities-v1.schema.json`, `format_version.const` |
| Inference archive | `3` | `rheplicant.inference.archive` |
| Generated script | `1` | `_rheplicant_bootstrap.script.SCRIPT_FORMAT_VERSION` |

A published script carries its format version in the call it makes, and a
script written before versions existed is refused with the command that
regenerates it — the embedded source bytes in the old file are unchanged and
still the author's.

`capabilities.json` names, per resolved layer, every node the document places
and the maturity of the class it resolves to, so an archived run says which of
its physics was a stand-in without anyone reading prose. Check A53 says the
same thing in a report-severity finding, which is for a person; this is the
same resolution as a record. It is written on the success path only -- it is a
view of the resolved layers, and a document refused before it resolves has
none.

It is a separate file rather than a field in `provenance.json` for the reason
`integrity.json` and the published preset sources are: `provenance-v1` is
closed and requires every property it declares, so a new field there is a
`format_version` bump on a schema that ships in the wheel.

`products.json` names every scientific product a run published, the requests
that asked for them and the omissions, so a reader of an archived tree can see
what was asked for as well as what arrived. It was absent from this table until
2026-09-20 -- the table was a hand-written list, and a hand-written list of
published formats is one that goes one row short. It is now derived: every
schema in `src/rheplicant/config/schemas/` that declares a `format_version`
must appear here by file name and version, which is what
`tests/test_stability_page.py` checks.

The two audit documents are **closed**: every object in them sets
`additionalProperties: false` and requires every property it declares, so a
field cannot be added or removed without raising the version — and the schemas
ship in the wheel, where an out-of-repo consumer reads them. The exceptions are
three map definitions (`jsonObject`, `intMap`, `stringMap`), which take
arbitrary keys because arbitrary keys are what they capture.

`tests/config/test_audit_schemas.py` holds both halves against the goldens in
`tests/config/golden/`. Closure is checked at every object path a golden
reaches, and every array the schemas declare must carry an item in at least one
golden — an empty array validates against any item type, so a corpus with all
arrays empty checks no item schema at all. `provenance-populated.json` and
`diagnostics-populated.json` are the goldens that answer for the item shapes;
the three status goldens are byte-identical whatever an item becomes.

Each is compared **type-exactly** where it is read. `3.0 != 3` is False in
Python and so is `True != 1`, so a manifest storing either would otherwise be
accepted as a version it is not — and the archive's version guards a byte
layout, where being wrong returns numbers rather than an error.

## Deliberate reference implementations

Some arithmetic exists on both sides of the bayesmith seam. That is a
decision, not a leftover, and the copies are kept rather than deleted for the
stable baseline: an independent second implementation is the strongest oracle
either package has, and bayesmith's `tests/crosscheck/` runs the same inputs
through both and compares the outputs, so "they agree" is a measurement on
every run rather than an intention.

| Here | Held in agreement by |
|---|---|
| `rheplicant.inference.sqrtinfo` — `SqrtInfo`, `marginalise` | `tests/crosscheck/test_sqrtinfo_agrees.py` |
| `rheplicant.inference.linear` — the affinity criterion and the linear solve | `tests/crosscheck/test_linear.py` |
| `rheplicant.inference.chain._zeta_joint` — the joint covariance | `tests/crosscheck/test_provenance.py` |

Those paths are in **bayesmith's** repository, because the comparison belongs
to whichever side is checking the other and running it here would be this
package against itself.

The cost of the arrangement is real and worth stating: a copy can drift, and
one had. `linear.py`'s `_worse` had lost a case its far side kept (A4-5). What
makes the decision defensible is not that drift cannot happen, but that a
cross-check makes it a failing test somewhere rather than two answers nobody
compares.

`rheplicant.inference.engines._adam` is a separate case and is **not**
delegated. bayesmith 0.10's `minimize` accepts every keyword the delegation
would need, and upstream's own stability page calls the descent engine inside
it *Experimental (reference implementation)* — so delegating would put this
package's gradient engine on a surface upstream does not promise to keep.
`tests/test_bayesmith_floor.py` watches both halves of that reason: it asserts
the keywords still exist, and it fails if upstream raises the level, which is
the moment to reopen the question.

## Compatibility policy

This package is pre-1.0 and says so: `Development Status :: 3 - Alpha`.

- A **patch** release changes no public name and no contract version.
- A **minor** release may add public names and may raise a contract version.
  It may remove a name only where the removal is stated in the changelog.
- Placeholder physics may be replaced in any release. Its *contract* —
  shapes, ordering, purity, PRNG consumption — is what is stable, and
  replacing the body raises the capability's level rather than changing its
  interface.

## The bayesmith range

`bayesmith>=0.10,<0.11`.

The floor and the ceiling are both load-bearing, and for different reasons.
The floor is a capability floor: `rheplicant.inference.plan` imports
`bayesmith.optimize.certify`, which 0.10 added, so an earlier bayesmith fails
at import rather than at a call site. The ceiling is closed at the next minor
because a pre-1.0 bayesmith minor may move the deep module paths this package
imports — 0.10 did exactly that, turning `bayesmith.optimize` from a module
into a package.

`tests/test_bayesmith_floor.py` asserts each level by capability rather than
by version number, because an editable install reports whatever version its
metadata was written with.
