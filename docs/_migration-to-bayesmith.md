:::{admonition} This layer is being separated into its own package
:class: note

The Bayesian inference layer, and the streaming evidence built on it, are
moving to **bayesmith** — a package that does inference over a graph without
knowing anything about radio astronomy.

**Nothing has moved yet, and nothing you write today breaks.**
`rheplicant.inference` is still the implementation, it is still fully
supported, and it is what these pages describe. bayesmith is not released, so
there is nothing to install and nothing to migrate to.

Worth saying what the separation is *for*, because it is not a rewrite. The
twin stays here. bayesmith reads a whole `Pipeline` as a single deterministic
node, which means the property that makes inference on a twin worth doing at
all — that the model you fit is the model you simulate with — is the thing the
boundary is designed to preserve, not the thing being traded away for it.

When the move does happen, the intent is that imports keep working through
this package rather than breaking. Neither the mechanism nor the timing is
settled yet; this note will say so when they are.
:::

<!--
One source, included by docs/inference.md, docs/inference-spaces.md,
docs/inference-linear.md, docs/inference-plans.md and docs/evidence.md.

Kept as a snippet rather than written onto each page because a claim spelled
five times is five things to update and one thing that gets updated -- which
is not a hypothetical here: the float32-refusal measurement was spelled six
times across this repository and all six went stale on a day none of them was
edited (see tests/test_evidence_session.py).

No headings in this file, deliberately. myst registers no heading anchors for
included content, so a heading here would be unlinkable from anywhere -- the
reasoning is in tests/test_docs_links.py. Listed in conf.py's
`exclude_patterns` so Sphinx does not also try to build it as a page of its
own and warn that it is in no toctree.
-->
