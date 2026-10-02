# bayesmith: what this package uses it for, and which versions it accepts

[bayesmith](https://pypi.org/project/bayesmith/) does Bayesian inference over
an explicit graph, with no radio astronomy in it. It is a required dependency
of this package: `pyproject.toml` declares `bayesmith>=0.10,<0.11`, and
`rheplicant.inference` imports it. `import rheplicant` on its own does not, so
the forward model loads without it.

`rheplicant.inference` is the layer between a twin and bayesmith. It holds
what depends on the instrument: the parameter space and its bindings into the
twin, the noise models, the plan, and the accumulation of a campaign. It
builds a bayesmith graph from those and calls bayesmith for the arithmetic
that has no instrument in it.

If you have a RHINO twin, use `rheplicant.inference`. If you have a model that
is not an instrument twin, use bayesmith directly.

## What is delegated

Each row is a module of `rheplicant.inference` and the bayesmith module it
imports.

| What | Module here | bayesmith module |
|---|---|---|
| the graph a twin is read as | `graph_bridge`, `numpyro_bridge`, `priors` | `bayesmith` (`trace`, `sample`, `det`, `observe`, `to_numpyro`) |
| the block partition | `partition` | `bayesmith.dispatch.factor` |
| the exact linear-Gaussian solve and draw, and the condition bound | `linear_solve` | `bayesmith.exact.solve`, `bayesmith.exact.block`, `bayesmith.exact.precision` |
| the iterative GLS solve | `gls` | `bayesmith.exact.gls` |
| the tolerances of the affinity check | `linear`, `linear_probe` | `bayesmith.exact.linearity` |
| the log-space transform | `loglinear` | `bayesmith.exact.loglinear` |
| the reduced basis | `reduced_basis` | `bayesmith.exact.reduced_basis` |
| the Fisher matrix and covariance propagation | `uncertainty` | `bayesmith.exact.fisher`, `bayesmith.exact.gaussian`, `bayesmith.diagnose.local` |
| identifiability | `identifiability` | `bayesmith.diagnose.identifiability` |
| prior sensitivity | `sensitivity` | `bayesmith.diagnose.sensitivity` |
| the chain marginal and the smoother | `chain_recursion` | `bayesmith.marginal.chain` |
| the square-root information marginalisation | `sqrtinfo` | `bayesmith.marginal.sqrtinfo` |
| the shrinkage and held-out diagnostics | `diagnostics` | `bayesmith.marginal.diagnostics`, `bayesmith.marginal.compress` |
| training an amortized posterior | `npe` | `bayesmith.amortize` |
| the calibrator's minimiser | `calibrate` | `bayesmith.optimize` |
| the convergence certificate | `engines`, `plan_estimate`, `plan_settings` | `bayesmith.optimize.certify` |

`tests/test_bayesmith_floor.py` reads the imports out of `src/` and fails if
one is missing from this table.

NumPyro supplies the chain; bayesmith decides the partition. A message about
which latents share a block comes from bayesmith's dispatch, not from the
sampler. The seams in `calibrate`, `chain_recursion`, `graph_bridge`,
`identifiability` and `npe` catch a bayesmith refusal and re-raise it as one
of this package's error classes.

## What is kept here

The parameter space (`Latent`, `Bind`), the noise models, `SamplingPlan`, and
the accumulation, compression and archive layers are this package's.

Three pieces of arithmetic exist on both sides. [The stability
page](stability.md) lists each copy and the test in this repository's
`tests/crosscheck/` or `tests/evidence/` that holds it in agreement. A copy
can drift, and one had; the comparison makes drift a failing test.

The gradient engine in `rheplicant.inference.engines` is not delegated.
bayesmith 0.10's `minimize` accepts every keyword it would need, and
upstream's stability page calls the descent engine inside `minimize`
*Experimental (reference implementation)*. Every non-conjugate block's point
estimate runs through the gradient engine, so delegating would put it on a
surface upstream does not promise to keep. `tests/test_bayesmith_floor.py`
checks that the keywords still exist and fails if upstream raises the level.

## Which versions are accepted

The declared range is `bayesmith>=0.10,<0.11`. 0.10.0 is on PyPI as of
2026-10-02, so an install resolves the range from the index; [the install
section](https://github.com/RHINO-Experiment/rheplicant#install) has the
commands.

**The range is closed at 0.11 because bayesmith is pre-1.0**, where a minor
release may move the deep module paths this package imports. 0.10 moved one:
`bayesmith.optimize` became a package, so `from bayesmith.optimize import
minimize` still resolves while the module file that name used to live in is
gone. A range open at the top would make such a move a runtime failure in an
installed environment; closed, it is a resolver error at install time.

**The floor is 0.10 by capability.** Each earlier release added a surface
this package uses, and `tests/test_bayesmith_floor.py` asserts each one by
capability, so the floor can be re-measured:

| Release | What this package needs from it |
|---|---|
| 0.2 | `first_fit`, `exact.loglinear` |
| 0.3 | `AffinityRefused`'s structured payload, `ComplexNormal` |
| 0.4 | `observe(..., mask=)`, `Probabilistic.observed_mask` |
| 0.5 | `local_block(..., priors=True)`; `marginal.chain.smooth` as a square root, whose 0.5 spelling returns `nan` on a stiff chain |
| 0.10 | `optimize.certify`, the convergence certificate an estimate stops on |

Below 0.5, `rheplicant.inference` does not import at all, because
`bayesmith.marginal` first ships there. Below 0.10 it fails the same way on
`optimize.certify`. A 0.5 install imports and fails only in behaviour, which
is the case the capability tests exist for.

## What the documentation cannot link to

bayesmith's own documentation is a hand-built HTML site and publishes no
`objects.inv`, so intersphinx has nothing to resolve against. References to
bayesmith names in these pages are plain literals, not links, and
`docs/conf.py` says so where it silences them.
