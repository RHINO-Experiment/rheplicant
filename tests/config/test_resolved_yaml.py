from __future__ import annotations

import decimal
import math
from pathlib import Path

import pytest
import yaml

from _rheplicant_bootstrap.audit.yaml import dump_resolved_yaml
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import Origin

GOLDEN_DIR = Path(__file__).with_name("golden")

SCALAR_TREE = {
    "null": None,
    "true": True,
    "false": False,
    "positive": 12,
    "negative": -3,
    "zero": 0.0,
    "negative_zero": -0.0,
    "ordinary": 1.25,
    "exponent": 1e100,
    "unicode": "雪",
    "escaped": 'line\n"x"',
    "empty_mapping": {},
    "empty_sequence": [],
}

NESTED_TREE = {
    "model": {"answer": 42, "items": ["x", [], {}, {"雪": True}]},
    "_rheplicant_resolved": {"format_version": 1},
}


def origin_tree(value, origin):
    if isinstance(value, dict):
        return OriginNode(origin, {key: origin_tree(child, origin) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return OriginNode(
            origin,
            {index: origin_tree(child, origin) for index, child in enumerate(value)},
        )
    return OriginNode(origin, {})


SCALAR_ORIGINS = OriginNode(
    None,
    {key: origin_tree(value, Origin("user")) for key, value in SCALAR_TREE.items()},
)

NESTED_ORIGINS = OriginNode(
    None,
    {
        "model": OriginNode(
            Origin("preset", "base"),
            {
                "answer": OriginNode(Origin("preset", "base"), {}),
                "items": OriginNode(
                    Origin("user"),
                    {
                        0: OriginNode(Origin("user"), {}),
                        1: OriginNode(Origin("rheplicant-default"), {}),
                        2: OriginNode(Origin("variant", "v"), {}),
                        3: OriginNode(
                            Origin("preset", "extra"),
                            {"雪": OriginNode(Origin("user"), {})},
                        ),
                    },
                ),
            },
        ),
        "_rheplicant_resolved": OriginNode(
            Origin("rheplicant-default"),
            {"format_version": OriginNode(Origin("rheplicant-default"), {})},
        ),
    },
)


def test_scalar_golden_is_exact_bytes():
    actual = dump_resolved_yaml(SCALAR_TREE, SCALAR_ORIGINS)
    assert actual == (GOLDEN_DIR / "resolved-yaml-scalars.yaml").read_bytes()
    assert actual.endswith(b"\n") and not actual.endswith(b"\n\n")
    assert b"\r" not in actual and b"\t" not in actual
    assert tuple(yaml.safe_load(actual)) == tuple(SCALAR_TREE)


def test_nested_golden_is_exact_bytes_and_valid_yaml():
    actual = dump_resolved_yaml(NESTED_TREE, NESTED_ORIGINS)
    assert actual == (GOLDEN_DIR / "resolved-yaml-nested.yaml").read_bytes()
    assert yaml.safe_load(actual) == NESTED_TREE


@pytest.mark.parametrize(
    "bad",
    [
        math.nan,
        math.inf,
        -math.inf,
        decimal.Decimal("1"),
        object(),
    ],
)
def test_non_contract_scalars_are_refused(bad):
    origins = OriginNode(None, {"bad": OriginNode(Origin("user"), {})})
    with pytest.raises(ConfigError, match="resolved YAML"):
        dump_resolved_yaml({"bad": bad}, origins)


@pytest.mark.parametrize(
    "origins",
    [
        OriginNode(None, {}),
        OriginNode(
            None,
            {
                "x": OriginNode(Origin("user"), {}),
                "extra": OriginNode(Origin("user"), {}),
            },
        ),
        OriginNode(None, {0: OriginNode(Origin("user"), {})}),
    ],
)
def test_origin_shape_mismatch_is_refused_at_root(origins):
    with pytest.raises(ConfigError, match="origin shape differs at <root>"):
        dump_resolved_yaml({"x": 1}, origins)


def test_reserved_namespace_must_be_last():
    document = {"_rheplicant_resolved": {}, "model": {}}
    origins = origin_tree(document, Origin("rheplicant-default"))
    origins = OriginNode(None, origins.children)
    with pytest.raises(ConfigError, match="final key"):
        dump_resolved_yaml(document, origins)
