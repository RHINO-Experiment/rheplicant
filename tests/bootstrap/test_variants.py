"""One canonical Config Plan 4A layer enumeration and attribution seam."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

import pytest

from _rheplicant_bootstrap import frozen as frozen_module
from _rheplicant_bootstrap import layering as layering_module
from _rheplicant_bootstrap import variants as variants_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.layering import initial_merge, origins_at
from _rheplicant_bootstrap.path_syntax import is_legal_path
from _rheplicant_bootstrap.types import Origin
from _rheplicant_bootstrap.variants import (
    LayerAttributor,
    LayerEnumeration,
    LayerRef,
    enumerate_layers,
    enumerate_layers_once,
)


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    where: str
    message: str


def _refuse(where: str, message: str) -> Finding:
    return Finding("A1", "refuse", where, message)


def _enumerate(document: dict):
    merged = initial_merge(document, origin=Origin("user"))
    return enumerate_layers_once(
        merged.document, merged.origins, merged.deletions
    )


def _canonical_wrapper(parent, name="x"):
    parent = layering_module._canonical_variant_parent(parent)
    return layering_module._canonical_variant_document(
        parent, name, parent.document["variants"][name]
    )


def _overlay_type():
    implementation = getattr(layering_module, "_OverlayMapping", None)
    assert implementation is not None, "persistent mapping overlays are missing"
    return implementation


def test_enumerate_layers_is_the_exact_once_function():
    assert enumerate_layers is enumerate_layers_once


def test_layer_attributor_matches_the_old_whole_finding_rule():
    inherited = _refuse("runtime", "same")
    variant_only = _refuse("runtime", "different")
    enumeration = _enumerate(
        {
            "runtime": {},
            "runs": [],
            "variants": {"x": {}, "y": {}},
        }
    )
    base, x, y = enumeration.layers
    attributor = LayerAttributor()

    assert attributor.attribute(base, [inherited]) == (inherited,)
    assert attributor.attribute(x, [inherited, variant_only]) == (
        Finding(
            "A1",
            "refuse",
            "variants.x.runtime",
            "variants.x: different",
        ),
    )
    assert attributor.attribute(y, [inherited, variant_only]) == (
        Finding(
            "A1",
            "refuse",
            "variants.y.runtime",
            "variants.y: different",
        ),
    )


def test_layer_attributor_preserves_duplicate_occurrences_within_one_layer():
    own = _refuse("model", "one")
    enumeration = _enumerate(
        {"runtime": {}, "runs": [], "variants": {"x": {}}}
    )
    attributor = LayerAttributor()
    attributor.attribute(enumeration.layers[0], ())

    assert attributor.attribute(enumeration.layers[1], (own, own)) == (
        Finding("A1", "refuse", "variants.x.model", "variants.x: one"),
        Finding("A1", "refuse", "variants.x.model", "variants.x: one"),
    )


def test_layer_attributor_requires_one_base_first_and_never_learns_siblings():
    shared = _refuse("model", "sibling fault")
    enumeration = _enumerate(
        {"runtime": {}, "runs": [], "variants": {"x": {}, "y": {}}}
    )
    base, x, y = enumeration.layers

    with pytest.raises(RuntimeError, match="base layer must be attributed first"):
        LayerAttributor().attribute(x, (shared,))

    attributor = LayerAttributor()
    attributor.attribute(base, ())
    assert len(attributor.attribute(x, (shared,))) == 1
    assert len(attributor.attribute(y, (shared,))) == 1


@pytest.mark.parametrize(
    "name", ["unity-gain", ".", "..", "A", "a", "slash/name", "雪"]
)
def test_all_nonempty_string_variant_names_are_preserved(name):
    enumeration = _enumerate(
        {
            "runtime": {"seed": 1},
            "runs": [],
            "variants": {name: {"runtime": {"seed": 2}}},
        }
    )
    layer = enumeration.layers[1]

    assert layer.name == name
    assert layer.prefix == f"variants.{name}"
    assert layer.attribute(_refuse("runtime", "fault")).message == (
        f"variants.{name}: fault"
    )
    full_where = f"variants.{name}.runtime"
    expected_where = full_where if is_legal_path(full_where) else "variants"
    assert layer.attribute(_refuse("runtime", "fault")).where == expected_where


def test_layer_documents_and_builder_copies_have_the_closed_shapes():
    enumeration = _enumerate(
        {
            "schema_version": 1,
            "defaults": ["one"],
            "plugins": ["plug"],
            "runtime": {"seed": 1},
            "runs": [{"name": "base"}],
            "outputs": {"dir": "out"},
            "variants": {
                "x": {"runtime": {"seed": 2}, "runs": [{"name": "x"}]}
            },
        }
    )
    base, variant = enumeration.layers

    assert "variants" not in base.document
    assert "variants" not in variant.document
    assert base.declared_runs == ({"name": "base"},)
    assert variant.declared_runs == ({"name": "x"},)
    assert set(base.mutable_document()).isdisjoint(
        {"defaults", "plugins", "outputs", "variants"}
    )
    mutable = variant.mutable_document()
    mutable["runtime"]["seed"] = 99
    assert variant.document["runtime"]["seed"] == 2


def test_layer_ref_canonicalizes_string_subclasses_before_invariant_checks():
    class HostileText(str):
        def __bool__(self):
            raise RuntimeError("hostile truth")

        def __eq__(self, other):
            raise RuntimeError("hostile equality")

        def __ne__(self, other):
            raise RuntimeError("hostile inequality")

    base = LayerRef(
        HostileText("base"),
        None,
        HostileText(""),
        {"runtime": {}},
        (),
    )
    variant = LayerRef(
        HostileText("variant"),
        HostileText("x"),
        HostileText("variants.x"),
        {"runtime": {}},
        (),
    )

    assert type(base.kind) is str
    assert type(base.prefix) is str
    assert type(variant.kind) is str
    assert type(variant.name) is str
    assert type(variant.prefix) is str


def test_each_variant_is_applied_once_and_origins_are_not_replayed(monkeypatch):
    calls: list[str] = []
    origin_merges: list[str | None] = []
    real = variants_module.apply_variant
    real_merge = layering_module.merge_with_origins

    def counting(document, name):
        calls.append(name)
        return real(document, name)

    def counting_merge(parent, patch, *, origin):
        origin_merges.append(origin.name)
        return real_merge(parent, patch, origin=origin)

    monkeypatch.setattr(variants_module, "apply_variant", counting)
    monkeypatch.setattr(
        layering_module, "merge_with_origins", counting_merge
    )
    enumeration = _enumerate(
        {
            "runtime": {"seed": 1},
            "runs": [],
            "variants": {
                "x": {"runtime": {"seed": 2}},
                "y": {"runtime": {"seed": 3}},
            },
        }
    )

    assert calls == ["x", "y"]
    assert origin_merges == ["x", "y"]
    x = enumeration.layers[1]
    assert origins_at(enumeration.origins[x.identity], ("runtime", "seed")) == Origin(
        "variant", "x"
    )
    assert thaw(x.document)["runtime"]["seed"] == 2


def test_public_apply_variant_keeps_cross_root_deepcopy_alias_semantics():
    shared = {"values": [1]}
    document = {
        "kept": shared,
        "variants": {"x": {"replaced": shared}},
    }

    applied = layering_module.apply_variant(document, "x")

    assert set(applied) == {"kept", "replaced", "variants"}
    assert applied["kept"] is applied["replaced"]
    assert applied["kept"] is applied["variants"]["x"]["replaced"]
    assert applied["kept"] is not shared


def test_overlay_mapping_has_final_dict_order_lookup_and_length():
    overlay_type = _overlay_type()
    base = MappingProxyType({"a": 1, "b": 2, "c": 3})
    delta = MappingProxyType({"a": 10, "b": 20, "d": 4})
    overlay = overlay_type(
        base,
        delta,
        frozenset({"a"}),
        ("a", "d"),
    )

    assert tuple(overlay) == ("b", "c", "a", "d")
    assert tuple(overlay.items()) == (
        ("b", 20),
        ("c", 3),
        ("a", 10),
        ("d", 4),
    )
    assert len(overlay) == 4
    assert overlay["b"] == 20
    assert overlay["c"] == 3
    assert overlay["a"] == 10
    assert overlay["d"] == 4
    with pytest.raises(KeyError):
        overlay["missing"]
    with pytest.raises(AttributeError):
        overlay._delta = MappingProxyType({})
    with pytest.raises(TypeError):
        overlay["a"] = 1
    with pytest.raises(TypeError):
        hash(overlay)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"~a": None, "a": 10}, ("b", "c", "a")),
        ({"a": 10, "~a": None}, ("b", "c")),
        ({"d": 4, "~d": None}, ("a", "b", "c")),
        ({"~missing": None}, ("a", "b", "c")),
        ({"b": 20, "d": 4}, ("a", "b", "c", "d")),
    ],
)
def test_trusted_overlay_preserves_delete_readd_and_replacement_order(
    patch, expected
):
    enumeration = _enumerate(
        {
            "runtime": {},
            "runs": [],
            "a": 1,
            "b": 2,
            "c": 3,
            "variants": {"x": patch},
        }
    )
    variant = enumeration.layers[1]
    origin = enumeration.origins[variant.identity]

    assert tuple(
        key for key in variant.document if key not in {"runtime", "runs"}
    ) == expected
    assert tuple(origin.children) == tuple(variant.document)
    assert thaw(variant.document) == {
        key: thaw(variant.document)[key] for key in variant.document
    }


def test_overlay_builder_removes_new_keys_in_constant_time_and_keeps_order():
    root = layering_module._trusted_overlay_root(MappingProxyType({}))
    builder = layering_module._OverlayBuilder(root)
    for index in range(1_000):
        builder[f"new_{index}"] = index
    for index in reversed(range(0, 1_000, 2)):
        builder.pop(f"new_{index}")
    for index in range(0, 1_000, 4):
        builder[f"new_{index}"] = index

    assert type(builder._end_keys) is dict
    published = builder.publish()
    expected = tuple(
        [f"new_{index}" for index in range(1, 1_000, 2)]
        + [f"new_{index}" for index in range(0, 1_000, 4)]
    )
    assert tuple(published) == expected

    def best_elapsed(size: int) -> float:
        patch = {
            **{f"x{index}": index for index in range(size)},
            **{
                f"~x{index}": None
                for index in reversed(range(size))
            },
        }
        document = {"runtime": {}, "variants": {"x": patch}}
        samples: list[float] = []
        for _ in range(2):
            started = perf_counter()
            _enumerate(document)
            samples.append(perf_counter() - started)
        return min(samples)

    small = best_elapsed(1_500)
    large = best_elapsed(6_000)
    assert large <= max(0.12, small * 7), (
        "removing newly appended overlay keys scaled superlinearly: "
        f"small={small * 1000:.1f}ms large={large * 1000:.1f}ms"
    )


def test_overlay_builder_rejects_hostile_lookup_keys_without_hooks():
    calls: list[str] = []

    class HostileKey:
        def __hash__(self):
            calls.append("hash")
            raise AssertionError("overlay lookup must validate first")

        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("overlay lookup must validate first")

    builder = layering_module._OverlayBuilder(
        layering_module._trusted_overlay_root(
            MappingProxyType({"kept": 1})
        )
    )
    key = HostileKey()
    marker = object()

    with pytest.raises(KeyError, match="exact strings or integers"):
        builder[key]
    assert builder.get(key, marker) is marker
    assert builder.pop(key, marker) is marker
    assert calls == []


def test_overlay_mapping_rejects_noncanonical_storage_without_hooks():
    overlay_type = _overlay_type()
    base = MappingProxyType({"a": 1})

    class MappingSubclass(dict):
        def __iter__(self):
            raise AssertionError("mapping iteration hook must not run")

    class TupleSubclass(tuple):
        pass

    class FrozenSetSubclass(frozenset):
        pass

    class HostileText(str):
        armed = False

        def __hash__(self):
            if type(self).armed:
                raise AssertionError("hash hook must not run")
            return str.__hash__(self)

        def __eq__(self, other):
            if type(self).armed:
                raise AssertionError("equality hook must not run")
            return str.__eq__(self, other)

        def __repr__(self):
            raise AssertionError("repr hook must not run")

    with pytest.raises(ConfigError, match="base"):
        overlay_type(MappingSubclass(a=1), MappingProxyType({}), frozenset(), ())
    with pytest.raises(ConfigError, match="delta"):
        overlay_type(base, MappingSubclass(), frozenset(), ())
    with pytest.raises(ConfigError, match="hidden"):
        overlay_type(base, MappingProxyType({}), FrozenSetSubclass(), ())
    with pytest.raises(ConfigError, match="end"):
        overlay_type(base, MappingProxyType({}), frozenset(), TupleSubclass())

    hostile_key = HostileText("a")
    hostile_dict = {hostile_key: 1}
    HostileText.armed = True
    with pytest.raises(ConfigError, match="exact string"):
        overlay_type(
            MappingProxyType(hostile_dict),
            MappingProxyType({}),
            frozenset(),
            (),
        )


class _BrokenOverlayBacking(Mapping):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, key):
        raise self.failure

    def __iter__(self):
        return iter(("a",))

    def __len__(self):
        return 1


class _AbortOverlayBacking(BaseException):
    pass


def test_overlay_mapping_staticizes_ordinary_proxy_failures_only():
    overlay_type = _overlay_type()

    with pytest.raises(ConfigError, match="overlay base protocol failed"):
        overlay_type(
            MappingProxyType(_BrokenOverlayBacking(RuntimeError("boom"))),
            MappingProxyType({}),
            frozenset(),
            (),
        )
    with pytest.raises(_AbortOverlayBacking, match="stop"):
        overlay_type(
            MappingProxyType(
                _BrokenOverlayBacking(_AbortOverlayBacking("stop"))
            ),
            MappingProxyType({}),
            frozenset(),
            (),
        )


def test_overlay_mapping_snapshots_each_proxy_once_before_validation():
    overlay_type = _overlay_type()

    class ChangingBacking(Mapping):
        def __init__(self):
            self.iterations = 0

        def __getitem__(self, key):
            if key == "late":
                return 1
            raise KeyError(key)

        def __iter__(self):
            self.iterations += 1
            return iter(()) if self.iterations == 1 else iter(("late",))

        def __len__(self):
            return 0

    changing_base = ChangingBacking()
    base_overlay = overlay_type(
        MappingProxyType(changing_base),
        MappingProxyType({}),
        frozenset(),
        (),
    )
    assert changing_base.iterations == 1
    assert len(base_overlay) == 0
    assert tuple(base_overlay) == ()
    with pytest.raises(KeyError):
        base_overlay["late"]

    changing_delta = ChangingBacking()
    delta_overlay = overlay_type(
        MappingProxyType({}),
        MappingProxyType(changing_delta),
        frozenset(),
        (),
    )
    assert changing_delta.iterations == 1
    assert len(delta_overlay) == 0
    assert tuple(delta_overlay) == ()
    with pytest.raises(KeyError):
        delta_overlay["late"]


def test_overlay_subclasses_and_lookalikes_never_enter_trusted_merge_results():
    overlay_type = _overlay_type()

    class OverlaySubclass(overlay_type):
        pass

    subclass = OverlaySubclass(
        MappingProxyType({}), MappingProxyType({}), frozenset(), ()
    )
    ledger = layering_module._DeletionLedger((), ())
    origins = layering_module.OriginNode(None, {})

    with pytest.raises(ConfigError, match="trusted merge document"):
        layering_module._trusted_merge_result(subclass, origins, ledger)


def test_merge_rejects_a_forged_ledger_parent_without_overlay_roots():
    public = initial_merge({"runtime": {}}, origin=Origin("user"))
    forged = object.__new__(layering_module.MergeResult)
    object.__setattr__(forged, "document", public.document)
    object.__setattr__(forged, "origins", public.origins)
    object.__setattr__(
        forged,
        "deletions",
        layering_module._DeletionLedger((), ()),
    )

    with pytest.raises(ConfigError, match="trusted merge parent"):
        layering_module.merge_with_origins(
            forged, {}, origin=Origin("variant", "x")
        )


def test_trusted_overlay_requires_the_exact_private_root_marker_without_hooks():
    calls: list[str] = []

    class HostileBacking(Mapping):
        def __getitem__(self, key):
            calls.append("getitem")
            raise AssertionError("trusted storage must not read this mapping")

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("trusted storage must not read this mapping")

        def __len__(self):
            calls.append("len")
            raise AssertionError("trusted storage must not read this mapping")

    with pytest.raises(ConfigError, match="trusted overlay base"):
        layering_module._trusted_overlay(
            MappingProxyType(HostileBacking()),
            MappingProxyType({"x": 1}),
            frozenset(),
            ("x",),
        )

    assert calls == []


def test_trusted_overlay_noop_requires_exact_storage_before_truth_testing():
    root = layering_module._trusted_overlay_root(MappingProxyType({}))
    with pytest.raises(ConfigError, match="overlay delta"):
        layering_module._trusted_overlay(root, {}, frozenset(), ())
    with pytest.raises(ConfigError, match="overlay hidden"):
        layering_module._trusted_overlay(
            root, MappingProxyType({}), set(), ()
        )
    with pytest.raises(ConfigError, match="overlay end"):
        layering_module._trusted_overlay(
            root, MappingProxyType({}), frozenset(), []
        )

    calls: list[str] = []

    class HostileTruth(dict):
        def __bool__(self):
            calls.append("bool")
            raise _AbortOverlayBacking("stop")

    with pytest.raises(ConfigError, match="overlay delta"):
        layering_module._trusted_overlay(
            root, HostileTruth(), frozenset(), ()
        )
    assert calls == []

    class LyingDelta(Mapping):
        def __getitem__(self, key):
            return 1

        def __iter__(self):
            return iter(("late",))

        def __len__(self):
            return 0

    with pytest.raises(ConfigError, match="overlay delta replacements"):
        layering_module._trusted_overlay(
            root, MappingProxyType(LyingDelta()), frozenset(), ()
        )

    class BrokenIteration(Mapping):
        def __init__(self, failure):
            self.failure = failure

        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            raise self.failure

        def __len__(self):
            raise AssertionError("trusted snapshot must not ask for a length")

    with pytest.raises(ConfigError, match="overlay delta protocol"):
        layering_module._trusted_overlay(
            root,
            MappingProxyType(BrokenIteration(RuntimeError("boom"))),
            frozenset(),
            (),
        )
    with pytest.raises(_AbortOverlayBacking, match="stop"):
        layering_module._trusted_overlay(
            root,
            MappingProxyType(
                BrokenIteration(_AbortOverlayBacking("stop"))
            ),
            frozenset(),
            (),
        )


def test_canonical_parent_marks_public_roots_once_before_trusted_merges():
    public = initial_merge(
        {
            "runtime": {},
            "runs": [],
            "variants": {"x": {}},
        },
        origin=Origin("user"),
    )

    canonical = layering_module._canonical_variant_parent(public)

    assert type(canonical.document) is _overlay_type()
    assert canonical.document._base is public.document
    assert canonical.document._delta == {}
    assert canonical.document._hidden == frozenset()
    assert canonical.document._end_keys == ()
    assert type(canonical.origins.children) is _overlay_type()
    assert canonical.origins.children._base is public.origins.children


def test_trusted_layer_records_require_overlay_roots_before_any_lookup():
    calls: list[str] = []

    class HostileBacking(Mapping):
        def __getitem__(self, key):
            calls.append("getitem")
            raise AssertionError("trusted records must reject before lookup")

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("trusted records must reject before traversal")

        def __len__(self):
            calls.append("len")
            raise AssertionError("trusted records must reject before length")

    with pytest.raises(ConfigError, match="trusted layer document"):
        variants_module._trusted_layer_ref(
            kind="base",
            name=None,
            prefix="",
            document=MappingProxyType(HostileBacking()),
        )

    assert calls == []

    public = LayerRef("base", None, "", {"runtime": {}}, None)
    public_origin = layering_module.initial_merge(
        {"runtime": {}}, origin=Origin("user")
    ).origins
    ledger = layering_module._DeletionLedger((), ())
    with pytest.raises(
        ConfigError, match="trusted layer enumeration documents"
    ):
        variants_module._trusted_layer_enumeration(
            (public,),
            {public.identity: public_origin},
            {public.identity: ledger},
        )


def test_trusted_layer_ref_rejects_hostile_identity_scalars_without_hooks():
    calls: list[str] = []

    class PretendKind:
        def __eq__(self, other):
            calls.append("eq")
            return other == "base"

    class AbortingKind:
        def __eq__(self, other):
            calls.append("abort-eq")
            raise _AbortOverlayBacking("stop")

    document = layering_module._trusted_overlay_root(
        MappingProxyType({"runtime": MappingProxyType({})})
    )
    for kind in (PretendKind(), AbortingKind()):
        with pytest.raises(ConfigError, match="trusted layer identity"):
            variants_module._trusted_layer_ref(
                kind=kind,
                name=None,
                prefix="",
                document=document,
            )

    forged = object.__new__(LayerRef)
    object.__setattr__(forged, "kind", AbortingKind())
    object.__setattr__(forged, "name", None)
    object.__setattr__(forged, "prefix", "")
    object.__setattr__(forged, "document", document)
    object.__setattr__(forged, "declared_runs", None)
    with pytest.raises(ConfigError, match="trusted layer enumeration"):
        variants_module._trusted_layer_enumeration((forged,), {}, {})
    assert calls == []


def test_overlay_validation_survives_optimized_python():
    code = r'''
from types import MappingProxyType
from _rheplicant_bootstrap.layering import _OverlayMapping

def outcome(call):
    try:
        call()
    except Exception as exc:
        print(type(exc).__name__)
    else:
        print("accepted")

class TupleSubclass(tuple):
    pass

class FrozenSetSubclass(frozenset):
    pass

outcome(lambda: _OverlayMapping({}, MappingProxyType({}), frozenset(), ()))
outcome(lambda: _OverlayMapping(MappingProxyType({}), {}, frozenset(), ()))
outcome(lambda: _OverlayMapping(
    MappingProxyType({}), MappingProxyType({}), FrozenSetSubclass(), ()))
outcome(lambda: _OverlayMapping(
    MappingProxyType({}), MappingProxyType({}), frozenset(), TupleSubclass()))
'''
    done = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.splitlines() == [
        "ConfigError",
        "ConfigError",
        "ConfigError",
        "ConfigError",
    ]


def test_trusted_overlay_and_layer_gates_survive_optimized_python():
    code = r'''
from types import MappingProxyType
from _rheplicant_bootstrap.layering import (
    MergeResult, OriginNode, _DeletionLedger, _OverlayBuilder,
    _trusted_overlay, _trusted_overlay_root, merge_with_origins,
)
from _rheplicant_bootstrap.types import Origin
from _rheplicant_bootstrap.variants import (
    LayerRef, _trusted_layer_enumeration, _trusted_layer_ref,
)

def outcome(call):
    try:
        call()
    except Exception as exc:
        print(type(exc).__name__)
    else:
        print("accepted")

root = _trusted_overlay_root(MappingProxyType({}))
outcome(lambda: _trusted_overlay(root, {}, frozenset(), ()))
outcome(lambda: _trusted_overlay(root, MappingProxyType({}), set(), ()))
outcome(lambda: _trusted_overlay(root, MappingProxyType({}), frozenset(), []))
outcome(lambda: _OverlayBuilder(root)[object()])
outcome(lambda: _trusted_layer_ref(
    kind=object(), name=None, prefix="", document=root))
forged = object.__new__(LayerRef)
object.__setattr__(forged, "kind", object())
object.__setattr__(forged, "name", None)
object.__setattr__(forged, "prefix", "")
object.__setattr__(forged, "document", root)
object.__setattr__(forged, "declared_runs", None)
outcome(lambda: _trusted_layer_enumeration((forged,), {}, {}))
parent = object.__new__(MergeResult)
object.__setattr__(parent, "document", MappingProxyType({}))
object.__setattr__(parent, "origins", OriginNode(None, {}))
object.__setattr__(parent, "deletions", _DeletionLedger((), ()))
outcome(lambda: merge_with_origins(
    parent, {}, origin=Origin("variant", "x")))
'''
    done = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.splitlines() == [
        "ConfigError",
        "ConfigError",
        "ConfigError",
        "KeyError",
        "ConfigError",
        "ConfigError",
        "ConfigError",
    ]


def test_enumerator_scans_the_variant_index_once_not_once_per_variant(
    monkeypatch,
):
    document = {
        "runtime": {},
        "runs": [],
        "variants": {
            f"v{index}": {"runtime": {"seed": index}}
            for index in range(400)
        },
    }
    parent = initial_merge(document, origin=Origin("user"))
    item_passes = 0
    variant_index_pair_passes = 0
    real_items = variants_module._variant_items
    real_pairs = layering_module._mapping_pairs

    def counting_items(layered_document):
        nonlocal item_passes
        item_passes += 1
        return real_items(layered_document)

    def counting_pairs(mapping, *, failure):
        nonlocal variant_index_pair_passes
        if len(mapping) == 400:
            variant_index_pair_passes += 1
        yield from real_pairs(mapping, failure=failure)

    monkeypatch.setattr(variants_module, "_variant_items", counting_items)
    monkeypatch.setattr(layering_module, "_mapping_pairs", counting_pairs)

    enumeration = enumerate_layers_once(
        parent.document, parent.origins, parent.deletions
    )

    assert len(enumeration.layers) == 401
    assert item_passes == 1
    # The parent MergeResult canonicalizes the full input once.  Trusted
    # apply must not add one more full-index traversal per variant.
    assert variant_index_pair_passes == 1


def test_empty_variant_deletion_suffixes_reuse_one_base_ledger():
    count = 400
    merged = initial_merge(
        {
            "runtime": {},
            "runs": [],
            "variants": {f"v{index}": {} for index in range(count)},
        },
        origin=Origin("user"),
    )
    base_deletions = tuple(
        layering_module.DeletionRecord(
            (f"already-gone-{index}",), Origin("preset", "base")
        )
        for index in range(count)
    )

    enumeration = enumerate_layers_once(
        merged.document, merged.origins, base_deletions
    )
    rows = tuple(
        enumeration.deletions[layer.identity]
        for layer in enumeration.layers
    )

    assert len(rows) == count + 1
    assert len({id(value) for value in rows}) == 1
    assert type(rows[0]) is layering_module._DeletionLedger
    assert len(rows[0]) == count


def test_variant_deletion_ledgers_share_the_base_and_retain_only_tiny_suffixes(
    monkeypatch,
):
    base_count = 10_000
    variant_count = 100
    merged = initial_merge(
        {
            "runtime": {},
            "runs": [],
            **{f"kept{index}": index for index in range(variant_count)},
            "variants": {
                f"v{index}": {f"~kept{index}": None}
                for index in range(variant_count)
            },
        },
        origin=Origin("user"),
    )
    base_deletions = tuple(
        layering_module.DeletionRecord(
            (f"already-gone-{index}",), Origin("preset", "base")
        )
        for index in range(base_count)
    )
    visits = {"parent": 0, "suffix": 0}
    real_validate = layering_module._validate_deletion_ledger_chunk

    def counting_validate(rows, *, where):
        visits[where] += len(rows)
        return real_validate(rows, where=where)

    monkeypatch.setattr(
        layering_module,
        "_validate_deletion_ledger_chunk",
        counting_validate,
    )

    enumeration = enumerate_layers_once(
        merged.document, merged.origins, base_deletions
    )
    root = enumeration.deletions[enumeration.layers[0].identity]
    variants = tuple(
        enumeration.deletions[layer.identity]
        for layer in enumeration.layers[1:]
    )

    assert type(root) is layering_module._DeletionLedger
    assert type(root._parent) is tuple
    assert len(root._parent) == base_count
    assert root._suffix == ()
    assert all(type(rows) is layering_module._DeletionLedger for rows in variants)
    assert all(rows._parent is root for rows in variants)
    assert sum(len(rows._suffix) for rows in variants) == variant_count
    assert (
        len(root._parent)
        + len(root._suffix)
        + sum(len(rows._suffix) for rows in variants)
        == base_count + variant_count
    )
    assert all(len(rows) == base_count + 1 for rows in variants)
    assert visits == {"parent": base_count, "suffix": variant_count}


def test_persistent_deletion_ledger_matches_tuple_sequence_semantics():
    inherited = (
        layering_module.DeletionRecord(("one",), Origin("preset", "base")),
        layering_module.DeletionRecord(("two", 0), Origin("user")),
    )
    merged = initial_merge(
        {
            "runtime": {},
            "runs": [],
            "gone": True,
            "variants": {"x": {"~gone": None}},
        },
        origin=Origin("user"),
    )
    enumeration = enumerate_layers_once(
        merged.document, merged.origins, inherited
    )
    rows = enumeration.deletions[enumeration.layers[1].identity]
    expected = (
        *inherited,
        layering_module.DeletionRecord(
            ("gone",), Origin("variant", "x")
        ),
    )

    assert type(rows) is layering_module._DeletionLedger
    assert tuple(rows) == expected
    assert rows == expected
    assert expected == rows
    assert rows[0] == expected[0]
    assert rows[-1] == expected[-1]
    assert rows[1:] == expected[1:]
    assert rows[::-1] == expected[::-1]
    with pytest.raises(AttributeError):
        rows._suffix = ()
    with pytest.raises(TypeError):
        rows[0] = expected[0]

    class HostileIndex:
        def __index__(self):
            raise AssertionError("index hook must not run")

    with pytest.raises(TypeError, match="exact int or slice"):
        rows[HostileIndex()]
    with pytest.raises(TypeError, match="exact int or slice"):
        rows[True]
    with pytest.raises(TypeError, match="exact int or null"):
        rows[slice(HostileIndex(), None)]
    with pytest.raises(TypeError, match="exact int or null"):
        rows[slice(True, None)]
    with pytest.raises(TypeError):
        hash(rows)


def test_persistent_deletion_ledger_iteration_does_not_recurse():
    rows = layering_module._DeletionLedger((), ())
    expected = []
    for index in range(2_000):
        record = layering_module.DeletionRecord(
            (f"gone-{index}",), Origin("user")
        )
        expected.append(record)
        rows = rows.extend((record,))

    assert len(rows) == len(expected)
    assert rows[0] == expected[0]
    assert rows[-1] == expected[-1]
    assert tuple(rows) == tuple(expected)


def test_persistent_deletion_ledger_subclasses_are_not_trusted_parents():
    class LedgerSubclass(layering_module._DeletionLedger):
        pass

    class TupleSubclass(tuple):
        pass

    subclass = LedgerSubclass((), ())
    with pytest.raises(ConfigError, match="parent"):
        layering_module._DeletionLedger(subclass, ())
    with pytest.raises(ConfigError, match="parent"):
        layering_module._DeletionLedger(TupleSubclass(), ())
    with pytest.raises(ConfigError, match="suffix"):
        layering_module._DeletionLedger((), TupleSubclass())


def test_public_origin_merges_keep_exact_tuple_deletion_results():
    parent = initial_merge({"gone": True}, origin=Origin("user"))
    result = layering_module.merge_with_origins(
        parent, {"~gone": None}, origin=Origin("user")
    )

    assert type(parent.deletions) is tuple
    assert type(result.deletions) is tuple
    assert result.deletions == (
        layering_module.DeletionRecord(("gone",), Origin("user")),
    )


def test_public_layer_enumeration_materializes_private_ledgers_to_tuples():
    trusted = _enumerate(
        {
            "runtime": {},
            "runs": [],
            "variants": {"x": {}},
        }
    )

    public = LayerEnumeration(
        trusted.layers,
        trusted.origins,
        trusted.deletions,
    )

    assert all(type(layer.document) is MappingProxyType for layer in public.layers)
    assert all(
        type(root.children) is MappingProxyType for root in public.origins.values()
    )
    assert all(type(rows) is tuple for rows in public.deletions.values())


def test_public_layer_records_materialize_private_documents_and_sequences():
    trusted = _enumerate(
        {
            "runtime": {},
            "runs": [{"kind": "forward"}],
            "variants": {
                "x": {
                    "runtime": {"seed": 2},
                    "runs": {"append": [{"kind": "forward"}]},
                }
            },
        }
    )

    public_layers = tuple(
        LayerRef(
            layer.kind,
            layer.name,
            layer.prefix,
            layer.document,
            layer.declared_runs,
        )
        for layer in trusted.layers
    )
    public = LayerEnumeration(
        public_layers,
        trusted.origins,
        trusted.deletions,
    )

    assert all(type(layer.document) is MappingProxyType for layer in public.layers)
    assert all(type(layer.document["runs"]) is tuple for layer in public.layers)
    assert all(
        type(root.children) is MappingProxyType
        for root in public.origins.values()
    )
    assert all(type(rows) is tuple for rows in public.deletions.values())
    assert all(type(layer.mutable_document()) is dict for layer in public.layers)
    assert all(
        type(layer.mutable_document()["runs"]) is list for layer in public.layers
    )


def test_without_variants_is_identity_when_absent_and_constant_size_when_present():
    absent = initial_merge(
        {"runtime": {}, "runs": []}, origin=Origin("user")
    )
    absent_parent = layering_module._canonical_variant_parent(absent)

    document, origins, deletions = variants_module._without_variants(
        absent_parent
    )

    assert document is absent_parent.document
    assert origins is absent_parent.origins
    assert deletions is absent_parent.deletions

    present = initial_merge(
        {
            "runtime": {},
            "runs": [],
            "variants": {"x": {}},
        },
        origin=Origin("user"),
    )
    present_parent = layering_module._canonical_variant_parent(present)

    document, origins, deletions = variants_module._without_variants(
        present_parent
    )

    overlay_type = _overlay_type()
    assert type(document) is overlay_type
    assert type(origins.children) is overlay_type
    assert document._base is present_parent.document
    assert origins.children._base is present_parent.origins.children
    assert document._delta == origins.children._delta == {}
    assert document._hidden == origins.children._hidden == frozenset({"variants"})
    assert document._end_keys == origins.children._end_keys == ()
    assert deletions is present_parent.deletions


def test_private_deletion_ledger_validation_survives_optimized_python():
    code = r'''
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import _DeletionLedger

def outcome(call):
    try:
        call()
    except Exception as exc:
        print(type(exc).__name__)
    else:
        print("accepted")

outcome(lambda: _DeletionLedger([], ()))
outcome(lambda: _DeletionLedger((object(),), ()))
outcome(lambda: _DeletionLedger((), [object()]))
'''
    done = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert done.stdout.splitlines() == [
        "ConfigError",
        "ConfigError",
        "ConfigError",
    ]


def test_trusted_evidence_keeps_same_origin_aliases_and_splits_changed_origin():
    untouched = {"values": [1]}
    patch_shared = {"values": [2]}
    cross_origin = {"values": [3]}
    enumeration = _enumerate(
        {
            "runtime": {},
            "untouched_one": untouched,
            "untouched_two": untouched,
            "cross_user": cross_origin,
            "items": [],
            "variants": {
                "x": {
                    "patch_one": patch_shared,
                    "patch_two": patch_shared,
                    "cross_variant": cross_origin,
                    "items": {
                        "append": [patch_shared, patch_shared, cross_origin]
                    },
                }
            },
        }
    )
    variant = enumeration.layers[1]
    document = variant.document

    assert document["untouched_one"] is document["untouched_two"]
    assert document["patch_one"] is document["patch_two"]
    assert document["patch_one"] is document["items"][0]
    assert document["items"][0] is document["items"][1]
    assert document["cross_variant"] is document["items"][2]
    assert document["cross_user"] is not document["cross_variant"]

    root = enumeration.origins[variant.identity]
    assert origins_at(root, ("untouched_one",)) == Origin("user")
    assert origins_at(root, ("patch_one",)) == Origin("variant", "x")
    assert origins_at(root, ("cross_user",)) == Origin("user")
    assert origins_at(root, ("cross_variant",)) == Origin("variant", "x")
    assert origins_at(root, ("items", 0)) == Origin("variant", "x")
    assert origins_at(root, ("items", 2)) == Origin("variant", "x")


def test_trusted_overlay_cache_preserves_document_and_origin_aliases():
    shared_document = {"kept": {"value": 1}, "changed": 0}
    shared_patch = {"changed": 2}
    shared_items = [{"value": 3}]
    shared_append = {"append": [{"value": 4}]}
    enumeration = _enumerate(
        {
            "runtime": {},
            "runs": [],
            "left": shared_document,
            "right": shared_document,
            "items_left": shared_items,
            "items_right": shared_items,
            "variants": {
                "x": {
                    "left": shared_patch,
                    "right": shared_patch,
                    "items_left": shared_append,
                    "items_right": shared_append,
                }
            },
        }
    )
    variant = enumeration.layers[1]
    document = variant.document
    origins = enumeration.origins[variant.identity]

    assert document["left"] is document["right"]
    assert origins.children["left"] is origins.children["right"]
    assert document["items_left"] is document["items_right"]
    assert origins.children["items_left"] is origins.children["items_right"]
    assert type(document["left"]) is _overlay_type()
    assert type(document["items_left"]) is frozen_module._FrozenConcat
    assert thaw(document["left"]) == {"kept": {"value": 1}, "changed": 2}
    assert thaw(document["items_left"]) == [
        {"value": 3},
        {"value": 4},
    ]


def test_variant_append_uses_a_mutable_compatibility_copy_and_keeps_evidence(
    monkeypatch,
):
    calls: list[str] = []
    real = variants_module.apply_variant

    def counting(document, name):
        calls.append(name)
        return real(document, name)

    monkeypatch.setattr(variants_module, "apply_variant", counting)
    enumeration = _enumerate(
        {
            "runtime": {},
            "items": [1],
            "gone": True,
            "variants": {
                "x": {"items": {"append": [2]}, "~gone": None},
            },
        }
    )
    variant = enumeration.layers[1]

    assert calls == ["x"]
    assert variant.document["items"] == (1, 2)
    assert origins_at(enumeration.origins[variant.identity], ("items", 0)) == Origin(
        "user"
    )
    assert origins_at(enumeration.origins[variant.identity], ("items", 1)) == Origin(
        "variant", "x"
    )
    assert tuple(record.path for record in enumeration.deletions[variant.identity]) == (
        ("gone",),
    )


def test_variant_root_deletion_events_remain_in_per_layer_evidence():
    parent = initial_merge(
        {
            "runtime": {},
            "variants": {
                "x": {"runtime": {"seed": 2}},
            },
        },
        origin=Origin("preset", "base"),
    )
    merged = layering_module.merge_with_origins(
        parent,
        {"~variants": None},
        origin=Origin("user"),
    )
    enumeration = enumerate_layers_once(
        merged.document,
        merged.origins,
        merged.deletions,
    )
    base = enumeration.layers[0]

    assert tuple(
        record.path for record in enumeration.deletions[base.identity]
    ) == (("variants",),)


def test_enumerator_consumes_the_single_origin_authoritative_result(monkeypatch):
    real = variants_module.apply_variant

    def disagree(document, name):
        result = real(document, name)
        result["spoofed_document"] = {"runtime": {"seed": 999}}
        return result

    monkeypatch.setattr(variants_module, "apply_variant", disagree)
    enumeration = _enumerate(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        }
    )

    assert enumeration.layers[1].document["runtime"]["seed"] == 2


def test_enumerator_refuses_an_apply_that_does_not_arm_the_one_shot_result(
    monkeypatch,
):
    monkeypatch.setattr(
        variants_module,
        "apply_variant",
        lambda document, name: {"runtime": {"seed": 999}},
    )

    with pytest.raises(ConfigError, match="canonical variant result was not produced"):
        _enumerate(
            {
                "runtime": {"seed": 1},
                "variants": {"x": {"runtime": {"seed": 2}}},
            }
        )


def test_enumerator_refuses_a_different_return_after_apply_arms_the_result(
    monkeypatch,
):
    real = variants_module.apply_variant

    def substitute(document, name):
        real(document, name)
        return {"runtime": {"seed": 999}}

    monkeypatch.setattr(variants_module, "apply_variant", substitute)

    with pytest.raises(ConfigError, match="canonical variant return identity"):
        _enumerate(
            {
                "runtime": {"seed": 1},
                "variants": {"x": {"runtime": {"seed": 2}}},
            }
        )


def test_private_canonical_wrapper_is_exact_type_and_one_shot():
    parent = initial_merge(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        },
        origin=Origin("user"),
    )
    wrapped = _canonical_wrapper(parent)

    with pytest.raises(ConfigError, match="canonical variant result was not produced"):
        layering_module._take_canonical_variant_result(wrapped, object())

    returned = layering_module.apply_variant(wrapped, "x")
    result = layering_module._take_canonical_variant_result(wrapped, returned)
    assert result.document["runtime"]["seed"] == 2

    with pytest.raises(ConfigError, match="canonical variant result was already taken"):
        layering_module._take_canonical_variant_result(wrapped, returned)
    with pytest.raises(ConfigError, match="canonical variant document was already used"):
        layering_module.apply_variant(wrapped, "x")


def test_private_canonical_wrapper_rejects_a_nonidentical_return():
    parent = initial_merge(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        },
        origin=Origin("user"),
    )
    wrapped = _canonical_wrapper(parent)
    returned = layering_module.apply_variant(wrapped, "x")

    with pytest.raises(ConfigError, match="canonical variant return identity"):
        layering_module._take_canonical_variant_result(
            wrapped, dict(returned)
        )
    with pytest.raises(ConfigError, match="canonical variant result was not produced"):
        layering_module._take_canonical_variant_result(wrapped, returned)


class _AbortCanonicalVariant(BaseException):
    pass


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("ordinary failure"),
        ConfigError("config failure"),
        _AbortCanonicalVariant("base failure"),
    ],
    ids=["ordinary", "config", "base-exception"],
)
def test_private_canonical_failures_leave_no_consumable_result(
    monkeypatch, failure
):
    parent = initial_merge(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        },
        origin=Origin("user"),
    )
    wrapped = _canonical_wrapper(parent)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(layering_module, "merge_with_origins", fail)
    with pytest.raises(type(failure), match=str(failure)):
        layering_module.apply_variant(wrapped, "x")
    with pytest.raises(ConfigError, match="canonical variant result was not produced"):
        layering_module._take_canonical_variant_result(wrapped, object())
    with pytest.raises(ConfigError, match="canonical variant document was already used"):
        layering_module.apply_variant(wrapped, "x")


def test_private_canonical_apply_returns_an_opaque_token_without_thawing(
    monkeypatch,
):
    parent = initial_merge(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        },
        origin=Origin("user"),
    )
    wrapped = _canonical_wrapper(parent)

    def fail_thaw(*args, **kwargs):
        raise AssertionError("trusted apply must not thaw the full document")

    monkeypatch.setattr(layering_module, "thaw", fail_thaw, raising=False)
    returned = layering_module.apply_variant(wrapped, "x")
    assert type(returned) is dict
    assert returned == {}
    returned["spoofed_document"] = {"runtime": {"seed": 999}}

    result = layering_module._take_canonical_variant_result(wrapped, returned)
    assert result.document["runtime"]["seed"] == 2


def test_private_canonical_wrapper_rejects_a_different_expected_name():
    parent = initial_merge(
        {
            "runtime": {"seed": 1},
            "variants": {"x": {"runtime": {"seed": 2}}},
        },
        origin=Origin("user"),
    )
    wrapped = _canonical_wrapper(parent)

    with pytest.raises(ConfigError, match="canonical variant name did not match"):
        layering_module.apply_variant(wrapped, "y")
    with pytest.raises(ConfigError, match="canonical variant result was not produced"):
        layering_module._take_canonical_variant_result(wrapped, object())
    with pytest.raises(ConfigError, match="canonical variant document was already used"):
        layering_module.apply_variant(wrapped, "x")


def test_wrapper_subclasses_never_enter_the_trusted_branch(monkeypatch):
    parent = initial_merge(
        {"runtime": {}, "variants": {"x": {"runtime": {"seed": 2}}}},
        origin=Origin("user"),
    )
    canonical_parent = layering_module._canonical_variant_parent(parent)
    wrapper_type = type(_canonical_wrapper(parent))

    class WrapperSubclass(wrapper_type):
        pass

    origin_calls = 0
    compatibility_calls = 0
    real = layering_module.merge_with_origins

    def counting(*args, **kwargs):
        nonlocal origin_calls
        origin_calls += 1
        return real(*args, **kwargs)

    def compatibility(document, patch):
        nonlocal compatibility_calls
        compatibility_calls += 1
        return {"legacy": True}

    monkeypatch.setattr(layering_module, "merge_with_origins", counting)
    monkeypatch.setattr(layering_module, "recursive_update", compatibility)
    subclass = WrapperSubclass(
        canonical_parent,
        "x",
        canonical_parent.document["variants"]["x"],
    )
    assert layering_module.apply_variant(subclass, "x") == {"legacy": True}
    assert compatibility_calls == 1
    assert origin_calls == 0


def test_a_nonmapping_patch_stays_controlled_if_apply_accepts_it(monkeypatch):
    monkeypatch.setattr(
        variants_module,
        "apply_variant",
        lambda document, name: document,
    )
    merged = initial_merge(
        {"runtime": {}, "variants": {"x": []}}, origin=Origin("user")
    )

    with pytest.raises(ConfigError, match="accepted a non-mapping patch"):
        enumerate_layers_once(
            merged.document, merged.origins, merged.deletions
        )


def test_process_section_changes_are_refused_with_the_full_variant_name():
    merged = initial_merge(
        {
            "runtime": {},
            "runs": [],
            "outputs": {"dir": "one"},
            "variants": {"slash/name": {"outputs": {"dir": "two"}}},
        },
        origin=Origin("user"),
    )

    with pytest.raises(ConfigError) as caught:
        enumerate_layers_once(merged.document, merged.origins, merged.deletions)
    assert str(caught.value) == (
        "variants.slash/name.outputs: variants may not change process-entry "
        "section 'outputs'; edit the base outputs section."
    )


def test_layer_enumeration_direct_construction_validates_evidence_types():
    enumeration = _enumerate({"runtime": {}, "runs": []})
    base = enumeration.layers[0]
    with pytest.raises(ConfigError):
        LayerEnumeration(
            layers=(base,),
            origins={base.identity: object()},
            deletions={base.identity: ()},
        )
    with pytest.raises(ConfigError):
        LayerEnumeration(
            layers=(base,),
            origins={base.identity: enumeration.origins[base.identity]},
            deletions={base.identity: (object(),)},
        )


class _AbortEvidenceMapping(BaseException):
    pass


class _BrokenEvidenceMapping(Mapping):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, key):
        raise self.failure

    def __iter__(self):
        raise self.failure

    def __len__(self):
        return 1


class _BrokenDeletionSequence(Sequence):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, index):
        raise self.failure

    def __iter__(self):
        raise self.failure

    def __len__(self):
        return 1


def test_layer_enumeration_validates_parallel_origins_and_mapping_protocols():
    enumeration = _enumerate({"runtime": {}, "runs": []})
    base = enumeration.layers[0]
    valid_origin = enumeration.origins[base.identity]
    valid_deletions = enumeration.deletions

    with pytest.raises(ConfigError, match="origin tree"):
        LayerEnumeration(
            layers=(base,),
            origins={base.identity: layering_module.OriginNode(None, {})},
            deletions=valid_deletions,
        )

    children = dict(valid_origin.children)
    runtime = children["runtime"]
    children["runtime"] = layering_module.OriginNode(
        None, runtime.children
    )
    with pytest.raises(ConfigError, match="concrete origins"):
        LayerEnumeration(
            layers=(base,),
            origins={
                base.identity: layering_module.OriginNode(None, children)
            },
            deletions=valid_deletions,
        )

    with pytest.raises(ConfigError, match="mapping traversal"):
        LayerEnumeration(
            layers=(base,),
            origins=_BrokenEvidenceMapping(RuntimeError("boom")),
            deletions=valid_deletions,
        )
    with pytest.raises(_AbortEvidenceMapping, match="stop"):
        LayerEnumeration(
            layers=(base,),
            origins=_BrokenEvidenceMapping(_AbortEvidenceMapping("stop")),
            deletions=valid_deletions,
        )

    with pytest.raises(ConfigError, match="deletion sequence traversal"):
        LayerEnumeration(
            layers=(base,),
            origins=enumeration.origins,
            deletions={
                base.identity: _BrokenDeletionSequence(RuntimeError("boom"))
            },
        )
    with pytest.raises(_AbortEvidenceMapping, match="stop"):
        LayerEnumeration(
            layers=(base,),
            origins=enumeration.origins,
            deletions={
                base.identity: _BrokenDeletionSequence(
                    _AbortEvidenceMapping("stop")
                )
            },
        )


def test_private_trusted_enumeration_rejects_identity_mismatch_explicitly():
    enumeration = _enumerate({"runtime": {}, "runs": []})
    base = enumeration.layers[0]

    with pytest.raises(ConfigError, match="trusted layer enumeration evidence"):
        variants_module._trusted_layer_enumeration(
            (base,),
            {},
            enumeration.deletions,
        )
