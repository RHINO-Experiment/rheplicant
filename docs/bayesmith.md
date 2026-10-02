# bayesmith: what this package uses it for, and which versions it accepts

[bayesmith](https://pypi.org/project/bayesmith/) does Bayesian inference over
an explicit graph, with no radio astronomy in it. `rheplicant.inference` uses
it. That sentence is the whole relationship and it is worth stating plainly,
because the two packages share arithmetic and the obvious readings are both
wrong: this is not a fork, and it is not a migration.

**Nothing here is moving and nothing here is deprecated.**
`rheplicant.inference` is the implementation of the inference pages. If you
have a RHINO twin you are in the right place; if you have a model that is not
this instrument, bayesmith is the general one.

## Which parts are bayesmith's

`rheplicant.inference` delegates rather than reimplements wherever the
arithmetic has no instrument in it:

| Here | Delegates to |
|---|---|
| the block partition, the per-block engine and its tolerance | `bayesmith.dispatch` |
| the iterative GLS solve | `bayesmith.exact.gls` |
| the chain marginal and the smoother | `bayesmith.marginal.chain` |
| the square-root information marginalisation | `bayesmith.marginal.sqrtinfo` |
| the shrinkage and held-out diagnostics | `bayesmith.marginal.diagnostics` |
| the convergence certificate | `bayesmith.optimize.certify` |

NumPyro supplies the chain; bayesmith decides the partition. The distinction
matters when reading a refusal: a message about which latents share a block
comes from bayesmith's dispatch, not from the sampler.

## Which parts are deliberately still here

Some arithmetic exists on both sides, and that is a decision rather than a
leftover. [The stability page](stability.md) lists each copy and the
comparison in this repository's `tests/crosscheck/` that holds it in
agreement, and says the cost out loud: a copy can drift, and one had. What makes it defensible is that
a cross-check turns drift into a failing test rather than into two answers
nobody compares.

One case is not a copy but a refusal to delegate.
`rheplicant.inference.engines`'s gradient engine could use bayesmith 0.10's
`minimize`, which accepts every keyword it would need — and upstream's own
stability page calls the descent engine inside `minimize` *Experimental
(reference implementation)*. Delegating would put this package's gradient
engine, which every non-conjugate block's point estimate runs through, on a
surface upstream does not promise to keep. `tests/test_bayesmith_floor.py`
watches both halves of that reason and fails if either changes.

## Which versions are accepted

The declared range is `bayesmith>=0.10,<0.11`.

**bayesmith has settled at 0.10.** It is not moving to 0.11 while this
baseline is being cut, so the range describes a version that has stopped
rather than one still in flight. 0.10.0 is on PyPI as of 2026-10-02, so an
install resolves the range from the index; [the install section](https://github.com/RHINO-Experiment/rheplicant#install)
has the commands.

**It is closed at 0.11 because bayesmith is pre-1.0**, where a minor release
may move the deep module paths this package imports — and 0.10 moved one:
`bayesmith.optimize` became a package, so `from bayesmith.optimize import
minimize` still resolves while the module file that name used to live in is
gone. A range open at the top would have made that a runtime failure in
somebody's environment rather than a resolver error at install time.

**The floor is 0.10 by capability, not by preference.** Each earlier release
added a surface this package uses, and `tests/test_bayesmith_floor.py` asserts
each one by capability rather than by version number — so the floor is a
measurement that can be re-run, not a number somebody chose:

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

bayesmith's own documentation is a hand-built HTML site rather than a Sphinx
one, so it publishes no `objects.inv` and there is nothing for intersphinx to
resolve against. References to bayesmith names in these pages are therefore
plain literals, not links, and `docs/conf.py` says so where it silences them.
That ends the day upstream publishes an inventory.
