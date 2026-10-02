:::{admonition} This layer is built on bayesmith
:class: note

[bayesmith](https://pypi.org/project/bayesmith/) does Bayesian inference over
an explicit graph, with no radio astronomy in it. It is a required dependency
of this package, and `rheplicant.inference` imports it.

These pages describe the part that knows about the instrument: the parameter
space and its bindings into a twin, the noise models, the plan, and the
accumulation of a campaign. `rheplicant.inference` builds a bayesmith graph
from those and calls bayesmith for the block partition, the exact
linear-Gaussian solves, the iterative GLS, the Fisher matrix, the
identifiability and sensitivity diagnostics, the chain marginal, the training
of an amortized posterior and the convergence certificate.

If you have a RHINO twin, use these pages. If you have a model that is not an
instrument twin, use bayesmith directly. [The bayesmith page](bayesmith.md)
lists each delegation, what is kept here, and which versions are accepted.
:::

<!--
One source, included by docs/inference.md, docs/inference-spaces.md,
docs/inference-linear.md, docs/inference-plans.md and docs/evidence.md, so
the statement is edited in one place.

No headings in this file. myst registers no heading anchors for included
content, so a heading here could not be linked to; the reasoning is in
tests/test_docs_links.py. The file is listed in conf.py's `exclude_patterns`
so Sphinx does not build it as a page of its own.
-->
