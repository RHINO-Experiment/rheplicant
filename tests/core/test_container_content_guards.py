"""Containers refusing malformed contents: the key-type family, and immutability.

Two ideas, both copy-pasted, both mostly untested:

**"keys must be strings"** appears four times in ``rheplicant.core`` -- once per
container that holds a user-supplied dict. Three are ``eqx.Module``
``__check_init__`` guards over a traced dict (``coords.extra``, ``env.extra``,
``State.aux``) and raise ``StateValidationError``; the fourth is
``FrozenMapping``, which is not a Module and raises ``TypeError`` because it is
a mapping constructor and that is what mappings raise. A coverage run found the
first two never executed. As with the lookup family, the population is derived
from the source rather than listed, so the next container to grow a ``dict``
field cannot ship the guard untested.

Deliberately NOT folded into that family: ``"Operator names must be strings."``
in ``resolve_names``. It reads similarly and it was grouped with these in the
brief, but it guards a *sequence of labels* rather than the keys of a mapping,
and it is one site reached from three composites -- so the interesting question
about it is which callers reach it, not how many copies exist. It gets its own
section below.

**"FrozenMapping is immutable"** is the other shape. The line the coverage run
flagged is ``__delattr__``; ``__setattr__`` carries the identical sentence, and
one of the two being exercised is exactly the situation where a guard looks
tested and is not. Which entry points reach it is derived too, and the verbs
that do NOT reach it -- there are more of those than one expects -- are pinned,
because the answer determines what a caller who writes ``meta["k"] = v``
actually sees.
"""

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import pytest

import rheplicant.core
from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.environment import Environment
from rheplicant.core.errors import PipelineError, StateValidationError
from rheplicant.core.frozen import FrozenMapping
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State

# ---------------------------------------------------------------------------
# the "keys must be strings" family
# ---------------------------------------------------------------------------

_KEY_GUARD = re.compile(r"keys must be strings")

#: A key no container should take, and one every container should.
BAD_KEY = 1
GOOD_KEY = "one"


class _KeyGuarded:
    """How to build one member of the family with a given key."""

    def __init__(
        self,
        build: Callable[[Any], Any],
        error: type[Exception],
        contents: Callable[[Any], dict],
        expected_in_message: str,
    ):
        self.build = build
        self.error = error
        self.contents = contents
        self.expected_in_message = expected_in_message


#: Class name -> how to exercise its key guard. Values are distinct per member
#: so that a "the good key round-tripped" assertion cannot be satisfied by some
#: other container's contents.
KEY_GUARDED: dict[str, _KeyGuarded] = {
    "Coordinates": _KeyGuarded(
        build=lambda key: Coordinates(time=jnp.arange(3.0), extra={key: jnp.full(3, 2.0)}),
        error=StateValidationError,
        contents=lambda c: c.extra,
        expected_in_message="coords.extra",
    ),
    "Environment": _KeyGuarded(
        build=lambda key: Environment(temperature=jnp.asarray(290.0),
                                      extra={key: jnp.full(3, 3.0)}),
        error=StateValidationError,
        contents=lambda c: c.extra,
        expected_in_message="env.extra",
    ),
    "State": _KeyGuarded(
        build=lambda key: State(data=jnp.zeros(3), aux={key: jnp.full(3, 5.0)}),
        error=StateValidationError,
        contents=lambda c: c.aux,
        expected_in_message="State.aux",
    ),
    "FrozenMapping": _KeyGuarded(
        build=lambda key: FrozenMapping({key: "seven"}),
        error=TypeError,
        contents=dict,
        expected_in_message="FrozenMapping",
    ),
}

KEY_GUARDED_IDS = sorted(KEY_GUARDED)


def _classes_in_core():
    for info in pkgutil.iter_modules(rheplicant.core.__path__):
        module = importlib.import_module(f"rheplicant.core.{info.name}")
        for klass in vars(module).values():
            if inspect.isclass(klass) and klass.__module__ == module.__name__:
                yield klass


def _source_of(func) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        # Synthesised by `dataclasses`, so no source file. Such a method
        # cannot carry a hand-written guard.
        return ""


def _carrying_the_key_guard() -> set[str]:
    """Every class in ``rheplicant.core`` whose own methods refuse non-string keys."""
    found = set()
    for klass in _classes_in_core():
        for member in vars(klass).values():
            if inspect.isfunction(member) and _KEY_GUARD.search(_source_of(member)):
                found.add(klass.__name__)
    return found


def test_the_table_is_the_family_and_the_family_is_the_table():
    """Derived, so the fifth container to hold a user dict fails here.

    Two of the four copies had no test at all when this was written, and they
    were the two whose files nobody had opened recently -- the same selection
    rule that left six of the nine coords guards uncovered.
    """
    carried = _carrying_the_key_guard()
    assert carried == set(KEY_GUARDED), {
        "carry the guard but are untested": sorted(carried - set(KEY_GUARDED)),
        "listed but no longer carry it": sorted(set(KEY_GUARDED) - carried),
    }


@pytest.mark.parametrize("name", KEY_GUARDED_IDS)
def test_a_non_string_key_is_refused(name):
    member = KEY_GUARDED[name]
    with pytest.raises(member.error):
        member.build(BAD_KEY)


@pytest.mark.parametrize("name", KEY_GUARDED_IDS)
def test_the_refusal_names_the_container_it_came_from(name):
    """Four containers, four messages, each saying which field it is about.

    ``pytest.raises(StateValidationError, match="keys must be strings")``
    passes for three of these at once and would keep passing if
    ``Environment`` reported ``coords.extra`` -- which is the copy-paste this
    family invites, since ``env.extra``'s guard is a two-line transcription of
    ``coords.extra``'s.
    """
    member = KEY_GUARDED[name]
    with pytest.raises(member.error) as excinfo:
        member.build(BAD_KEY)
    message = str(excinfo.value)
    assert member.expected_in_message in message, message
    assert "keys must be strings" in message, message


def test_the_four_refusals_differ_pairwise():
    """The check a shared substring cannot make.

    Every message here contains "keys must be strings"; if that were all any of
    them said, all four ``match=`` assertions above would still pass and a user
    handed one of them would not know which container to look at.
    """
    messages = {}
    for name in KEY_GUARDED_IDS:
        member = KEY_GUARDED[name]
        with pytest.raises(member.error) as excinfo:
            member.build(BAD_KEY)
        messages[name] = str(excinfo.value)
    assert len(set(messages.values())) == len(messages), messages


@pytest.mark.parametrize("name", KEY_GUARDED_IDS)
def test_a_string_key_gets_past_the_guard(name):
    """The other branch.

    A container whose ``__check_init__`` raised unconditionally would satisfy
    every test above. The guard is a one-character mutation away from that:
    ``all(...)`` -> ``any(...)`` inverts it, and an empty ``extra`` -- the
    default -- makes ``all`` vacuously true, so nothing else in the suite
    notices.
    """
    member = KEY_GUARDED[name]
    container = member.build(GOOD_KEY)
    assert GOOD_KEY in member.contents(container)


@pytest.mark.parametrize("name", KEY_GUARDED_IDS)
def test_a_mixed_dict_is_refused_for_the_one_bad_key(name):
    """One bad key among good ones still fails: the guard is ``all``, not ``any``.

    Distinct from the test above in the mutation it catches -- ``all`` -> ``any``
    passes "every key is bad" and "every key is good" alike, and only shows up
    on a mixture.
    """
    member = KEY_GUARDED[name]
    with pytest.raises(member.error):
        member.build(BAD_KEY)
    assert GOOD_KEY in member.contents(member.build(GOOD_KEY))


# ---------------------------------------------------------------------------
# "Operator names must be strings." -- one site, three callers
# ---------------------------------------------------------------------------

class _Source(AbstractOperator):
    """Distinct value per instance, so a composite's members stay tellable apart."""

    value: Any

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


#: The three composites that route their ``names=`` through ``resolve_names``.
#: The guard has one copy; what needs covering is that every entry point still
#: reaches it, which is the same failure mode as ``FrozenMapping`` below.
NAME_TYPED_COMPOSITES = {
    "Pipeline": Pipeline,
    "SumOperator": SumOperator,
    "SelectOperator": SelectOperator,
}


@pytest.mark.parametrize("name", sorted(NAME_TYPED_COMPOSITES))
def test_non_string_operator_names_are_refused(name):
    composite = NAME_TYPED_COMPOSITES[name]
    with pytest.raises(PipelineError, match="Operator names must be strings"):
        composite(_Source(value=2.0), _Source(value=3.0), names=("first", 2))


@pytest.mark.parametrize("name", sorted(NAME_TYPED_COMPOSITES))
def test_string_operator_names_are_accepted(name):
    """The other branch, and the reason the count check cannot stand in for it.

    ``resolve_names`` checks the name COUNT before the name types, so a
    two-name tuple gets past the first guard either way; only the type check
    separates these two tests.
    """
    composite = NAME_TYPED_COMPOSITES[name]
    built = composite(_Source(value=2.0), _Source(value=3.0), names=("first", "second"))
    assert built.names == ("first", "second")
    assert float(built["second"].value) == 3.0


# ---------------------------------------------------------------------------
# "FrozenMapping is immutable" -- which entry points reach it, and which do not
# ---------------------------------------------------------------------------

_IMMUTABLE_SENTENCE = re.compile(r"FrozenMapping is immutable")

#: Attribute-level mutations, one per method carrying the sentence.
IMMUTABLE_ENTRY_POINTS = {
    "__setattr__": lambda m: setattr(m, "telescope", "other"),
    "__delattr__": lambda m: delattr(m, "_items"),
}

#: Mapping-level mutation verbs a user would reasonably try. FrozenMapping
#: subclasses ``Mapping``, not ``MutableMapping``, so it does not define any of
#: them -- none reaches the branded sentence. Pinned rather than assumed: the
#: guard would look fully covered from ``__setattr__`` alone, and a reader
#: deciding whether ``meta["k"] = v`` is safely refused needs to know it is
#: refused by Python rather than by this class.
ABSENT_MUTATION_VERBS = (
    "__setitem__",
    "__delitem__",
    "update",
    "pop",
    "popitem",
    "clear",
    "setdefault",
)


def test_the_entry_points_carrying_the_sentence_are_the_ones_tested():
    """Derived, so a third mutation hook cannot be added without a test.

    This is the check the brief's concern asks for: with two copies of one
    sentence, covering either makes the *line* look tested while the other
    entry point may not reach it at all. Here it turned out both do -- but the
    assertion is what keeps that true.
    """
    carried = {
        method_name
        for method_name, member in vars(FrozenMapping).items()
        if inspect.isfunction(member) and _IMMUTABLE_SENTENCE.search(_source_of(member))
    }
    assert carried == set(IMMUTABLE_ENTRY_POINTS), {
        "carry the sentence but are untested": sorted(carried - set(IMMUTABLE_ENTRY_POINTS)),
        "listed but no longer carry it": sorted(set(IMMUTABLE_ENTRY_POINTS) - carried),
    }


@pytest.mark.parametrize("entry_point", sorted(IMMUTABLE_ENTRY_POINTS))
def test_each_entry_point_refuses_with_the_branded_sentence(entry_point):
    meta = FrozenMapping(telescope="RHINO", obs_id="demo-001")
    with pytest.raises(AttributeError, match="FrozenMapping is immutable"):
        IMMUTABLE_ENTRY_POINTS[entry_point](meta)


@pytest.mark.parametrize("entry_point", sorted(IMMUTABLE_ENTRY_POINTS))
def test_a_refused_mutation_changes_nothing(entry_point):
    """The refusal has to be a refusal, not a report.

    ``__delattr__`` targets ``_items`` -- the dict the whole object is -- so a
    guard that raised after doing the work would leave a FrozenMapping that
    hashes, compares and then fails on first read.
    """
    meta = FrozenMapping(telescope="RHINO", obs_id="demo-001")
    before, before_hash = dict(meta), hash(meta)
    with pytest.raises(AttributeError):
        IMMUTABLE_ENTRY_POINTS[entry_point](meta)
    assert dict(meta) == before
    assert hash(meta) == before_hash
    assert meta["telescope"] == "RHINO"


@pytest.mark.parametrize("verb", ABSENT_MUTATION_VERBS)
def test_the_mapping_mutation_verbs_do_not_exist(verb):
    assert not hasattr(FrozenMapping, verb)


def test_item_assignment_is_refused_by_python_without_the_brand():
    """What a user who writes ``meta["k"] = v`` actually sees.

    Not the sentence this class wrote: ``Mapping`` defines no ``__setitem__``,
    so the subscript never reaches ``__setattr__`` and the interpreter answers
    with a generic ``TypeError``. Recorded because it is the most likely way to
    try to mutate a mapping, and because "FrozenMapping is immutable" being
    covered says nothing about it.
    """
    meta = FrozenMapping(telescope="RHINO")
    with pytest.raises(TypeError, match="does not support item assignment"):
        meta["band"] = "low"
    with pytest.raises(TypeError, match="does not support item deletion"):
        del meta["telescope"]
    assert dict(meta) == {"telescope": "RHINO"}


def test_the_functional_updates_still_work():
    """The other branch: an immutability guard that broke construction would pass above.

    ``__init__`` writes through ``object.__setattr__`` precisely to get past
    the guard, so "does the guard still let the object be built and updated"
    is a live question and not a formality.
    """
    meta = FrozenMapping(telescope="RHINO")
    added = meta.set(obs_id="demo-001")
    merged = added | {"band": "low"}
    removed = merged.remove("obs_id")

    assert dict(meta) == {"telescope": "RHINO"}, "the original was mutated"
    assert dict(added) == {"telescope": "RHINO", "obs_id": "demo-001"}
    assert dict(merged) == {"telescope": "RHINO", "obs_id": "demo-001", "band": "low"}
    assert dict(removed) == {"telescope": "RHINO", "band": "low"}
    assert hash(meta) != hash(added)
