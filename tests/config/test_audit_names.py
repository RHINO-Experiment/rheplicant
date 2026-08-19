from __future__ import annotations

import os

import pytest

from _rheplicant_bootstrap.audit import encode_name, validate_encoded_names
from _rheplicant_bootstrap.audit.types import PathEncoding
from _rheplicant_bootstrap.errors import ConfigError


@pytest.mark.parametrize(
    ("name", "encoded"),
    [
        ("", "n-"),
        (".", "n-2e"),
        ("..", "n-2e2e"),
        ("A", "n-41"),
        ("a", "n-61"),
        ("a/b", "n-612f62"),
        ("雪", "n-e99baa"),
    ],
)
def test_utf8_hex_codec(name, encoded):
    assert encode_name(name) == encoded


@pytest.mark.parametrize("kind", ["run", "variant"])
def test_validator_preserves_declaration_order(kind):
    assert validate_encoded_names(kind, ["b", "a"], component_limit=255) == (
        PathEncoding(kind, "b", "n-62"),
        PathEncoding(kind, "a", "n-61"),
    )


def test_validator_rejects_empty_names_and_uses_leased_limit():
    with pytest.raises(ConfigError, match="non-empty"):
        validate_encoded_names("run", [""], component_limit=255)
    encoded = encode_name("雪")
    with pytest.raises(ConfigError, match="filesystem limit"):
        validate_encoded_names("variant", ["雪"], component_limit=len(os.fsencode(encoded)) - 1)


def test_validator_defends_against_casefold_collisions(monkeypatch):
    values = iter(("n-A", "n-a"))
    monkeypatch.setattr("_rheplicant_bootstrap.audit.names.encode_name", lambda _name: next(values))
    with pytest.raises(ConfigError, match="collide"):
        validate_encoded_names("run", ["one", "two"], component_limit=255)
