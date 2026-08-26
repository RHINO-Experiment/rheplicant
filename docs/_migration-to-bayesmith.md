:::{admonition} There is a sibling package, and this layer is not moving to it
:class: note

`bayesmith <https://pypi.org/project/bayesmith/>`_ does Bayesian inference over
an explicit graph, with no radio astronomy in it. Several capabilities on these
pages have a counterpart there: the linear-Gaussian exact solves, the iterative
GLS, the Fisher matrix, the square-root information layer, and the graph
diagnostics.

**Nothing here is moving, and nothing here is deprecated — and since
2026-08-26 the two packages are no longer fully disjoint.**
`rheplicant.inference` is the implementation of everything on these pages, it
is fully supported, and the historically shared capabilities stay separate,
held in agreement by a cross-check suite rather than by shared code — 123
comparisons that run on every change to bayesmith. The NEW capabilities are
the exception, by the owner's decision: the auto-partition grouping rule and
the log-space transform arithmetic were implemented in bayesmith
(`dispatch.factor`, `exact.loglinear`) and `rheplicant.inference` imports
them (`partition.py`, `loglinear.py`) rather than carrying a second copy —
so bayesmith is now a runtime dependency, and for those two seams there is
one statement of the algorithm with two paradigms' probes feeding it. The
cross-check discipline stays for what was born twice; what was born once is
simply shared.

The reason is worth one sentence, because "keep both" can read as indecision.
The two are different paradigms — this layer reads a `Pipeline` and a
`ParameterSpace`, bayesmith reads a `Graph` — so of the 106 names published here
only 24 exist anywhere in bayesmith, and only 12 in the 42 it publishes at top
level, and even those take different arguments. And the
three that do match exactly are the ones the cross-check compares; making one a
re-export of the other would leave that test comparing an object with itself,
unable to fail. The easier half of the merge was the half that would have cost
the most.

**Which to reach for.** If you have a RHINO twin, you are in the right place.
If you have a model that is not this instrument, bayesmith is the general one.
The two agreeing is the point, and it is checked rather than asserted.
:::

<!--
One source, included by docs/inference.md, docs/inference-spaces.md,
docs/inference-linear.md, docs/inference-plans.md and docs/evidence.md.

Kept as a snippet rather than written onto each page because a claim spelled
five times is five things to update and one thing that gets updated -- which
is not a hypothetical here: the float32-refusal measurement was spelled six
times across this repository and all six went stale on a day none of them was
edited (see tests/test_evidence_session.py). This note itself went stale
within hours of being written, when the decision it described was made; one
source is why fixing that was one edit.

No headings in this file, deliberately. myst registers no heading anchors for
included content, so a heading here would be unlinkable from anywhere -- the
reasoning is in tests/test_docs_links.py. Listed in conf.py's
`exclude_patterns` so Sphinx does not also try to build it as a page of its
own and warn that it is in no toctree.
-->
