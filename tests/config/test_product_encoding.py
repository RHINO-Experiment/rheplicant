from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.config.products.encoding import (
    canonical_product_json,
    deterministic_npz,
    validate_relative_product_path,
)


def test_npz_bytes_are_deterministic_sorted_and_pickle_free():
    values = {
        "z/complex": np.array([1 + 2j], dtype=np.complex64),
        "a": np.array([[1, 2]], dtype=np.int16),
    }
    first, metadata = deterministic_npz(values)
    second, again = deterministic_npz(dict(reversed(tuple(values.items()))))
    assert first == second
    assert metadata == again == {
        "a": {"dtype": "int16", "shape": [1, 2]},
        "z/complex": {"dtype": "complex64", "shape": [1]},
    }
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a.npy", "z/complex.npy"]
        assert {entry.date_time for entry in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }
    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["a"], values["a"])
        np.testing.assert_array_equal(loaded["z/complex"], values["z/complex"])


@pytest.mark.parametrize(
    "values",
    (
        {},
        {"": np.array([1])},
        {"../escape": np.array([1])},
        {"a.npy": np.array([1])},
        {"bad": np.array([object()], dtype=object)},
    ),
)
def test_npz_refuses_empty_hostile_or_object_payloads(values):
    with pytest.raises(ConfigError):
        deterministic_npz(values)


def test_product_json_is_canonical_utf8_and_finite():
    assert canonical_product_json({"雪": [2], "a": True}) == (
        b'{\n  "a": true,\n  "\xe9\x9b\xaa": [\n    2\n  ]\n}\n'
    )
    with pytest.raises(ConfigError, match="non-finite"):
        canonical_product_json({"bad": float("nan")})


@pytest.mark.parametrize(
    "path",
    ("", "/absolute.json", "../escape.json", "runs//x.json", "runs/./x.json", "a\\b"),
)
def test_relative_product_paths_refuse_ambiguous_or_hostile_spellings(path):
    with pytest.raises(ConfigError, match="product path"):
        validate_relative_product_path(path, component_limit=255)


def test_relative_product_path_enforces_the_real_component_budget():
    validate_relative_product_path("runs/n-61/arrays.npz", component_limit=10)
    with pytest.raises(ConfigError, match="filesystem limit"):
        validate_relative_product_path("runs/n-61/arrays.npz", component_limit=9)


def test_npz_is_a_real_zip_and_not_a_json_disguise():
    payload, _metadata = deterministic_npz({"x": np.arange(3)})
    assert zipfile.is_zipfile(io.BytesIO(payload))
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(payload.decode("utf-8"))
