"""The resolved-YAML bytes are stable, and stable in a specific way.

A10 asked for this: ``ResolvedYamlEncoder._mapping_keys`` returns
``tuple(value)`` -- insertion order, with no ``sort_keys`` guard -- and the
bytes it produces are hashed into ``integrity.json`` and quoted in
provenance. "Deterministic today" was the review's assessment; a test is
what keeps it true.

Two properties, and the second is the one worth having:

1. **Byte-stability.** The same document encodes to the same bytes, in the
   same process and in a FRESH interpreter under a different
   ``PYTHONHASHSEED``. That is what "deterministic" has to mean for a value
   that anchors a published tree -- a digest recomputed tomorrow on the same
   inputs must match.

2. **Key order is part of the document, deliberately.** Two mappings with
   identical content in different textual order encode DIFFERENTLY, and that
   is not a defect to be repaired by sorting. Sorting would be a one-line
   change that silently altered every digest ever recorded, so the property
   is pinned in the direction that refuses it rather than left as an absence.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from _rheplicant_bootstrap.audit.yaml import dump_resolved_yaml
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import Origin
from tests.config.test_resolved_yaml import (
    NESTED_ORIGINS,
    NESTED_TREE,
    origin_tree,
)


def _origins(tree: dict) -> OriginNode:
    return OriginNode(
        None, {key: origin_tree(value, Origin("user")) for key, value in tree.items()}
    )


def test_the_same_document_encodes_to_the_same_bytes_every_time():
    first = dump_resolved_yaml(NESTED_TREE, NESTED_ORIGINS)
    for _ in range(4):
        assert dump_resolved_yaml(NESTED_TREE, NESTED_ORIGINS) == first


def test_the_bytes_do_not_move_with_PYTHONHASHSEED():
    """The real content of "deterministic", and the only way to see it.

    String hashing is randomised per process, so anything that walked a
    ``set`` -- or a ``dict`` built from one -- would produce a different
    order per run while looking perfectly stable inside any single one. Two
    fresh interpreters with different seeds are what separates "stable" from
    "stable because we only ever looked once".
    """
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        from _rheplicant_bootstrap.audit.yaml import dump_resolved_yaml
        from _rheplicant_bootstrap.layering import OriginNode
        from _rheplicant_bootstrap.types import Origin

        def origin_tree(value, origin):
            if isinstance(value, dict):
                return OriginNode(origin, {k: origin_tree(v, origin) for k, v in value.items()})
            if isinstance(value, (list, tuple)):
                return OriginNode(origin, {i: origin_tree(v, origin) for i, v in enumerate(value)})
            return OriginNode(origin, {})

        tree = {"model": {"answer": 42, "items": ["x", {"snow": True}]}, "z": 1, "a": 2}
        origins = OriginNode(None, {k: origin_tree(v, Origin("user")) for k, v in tree.items()})
        sys.stdout.write(dump_resolved_yaml(tree, origins).hex())
        """
    )
    seeds = ("0", "1", "12345")
    outputs = []
    for seed in seeds:
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=False,
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout)
    assert outputs[0], "the subprocess must actually have encoded something"
    assert len(set(outputs)) == 1, dict(zip(seeds, outputs, strict=True))


def test_key_order_is_part_of_the_document_and_sorting_would_change_every_digest():
    """The anti-repair clause.

    ``_mapping_keys`` preserves insertion order on purpose: the resolved YAML
    renders the document, and a document's key order is part of its bytes.
    Adding ``sorted()`` there is a one-line change that would look like
    tidying and would alter every digest in every published tree.

    **What this catches, measured, because the first answer was wrong.**
    A bare ``sorted()`` in ``_mapping_keys`` does NOT reach this assertion:
    the encoder already cross-checks its key order against the origins tree,
    and ``_origin_children`` raises "resolved YAML origin shape differs"
    first (``yaml.py:120``). That guard was there before this test and is the
    better one, because it fires on the spot.

    What it does not see is a CONSISTENT sort -- keys sorted and the origins
    comparison relaxed to match, which is what someone tidying both halves
    would write. Mutated that way, this test fails and the origins check does
    not. So the two are not redundant: one catches the careless change and
    this one catches the careful change.
    """
    forward = {"z": 1, "a": 2}
    backward = {"a": 2, "z": 1}
    assert forward == backward, "same content, and Python agrees"

    encoded_forward = dump_resolved_yaml(forward, _origins(forward))
    encoded_backward = dump_resolved_yaml(backward, _origins(backward))

    assert encoded_forward != encoded_backward, (
        "the encoder started sorting keys. That changes every digest ever "
        "recorded in integrity.json and provenance.json; if it is wanted, it "
        "is a format decision, not a cleanup."
    )
    # ... and each is still stable in itself, so the difference above is the
    # ORDER and not nondeterminism in either.
    assert dump_resolved_yaml(forward, _origins(forward)) == encoded_forward
    assert dump_resolved_yaml(backward, _origins(backward)) == encoded_backward
