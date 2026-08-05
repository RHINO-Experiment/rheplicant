"""One name-lookup guard, copy-pasted five times, tested once.

Three composite operators -- ``Pipeline``, ``SumOperator``, ``SelectOperator``
-- let you address a member by name, and each one that does opens with the same
refusal: *"No <noun> named 'x'; available: (...)"*. There are five copies of it
across ``__getitem__`` and the ``replace_*`` methods, and a full-suite coverage
run found four of the five never executed. The one that ran is
``Pipeline.__getitem__``, which is the copy the README uses.

The brief that started this file listed four copies. There are five. That is
the argument for deriving the population rather than listing it: a hand list of
this family had already gone stale before anyone wrote a test against it, and a
fourth composite type -- or a second accessor on ``SelectOperator``, which today
has ``__getitem__`` and no ``replace_branch`` -- would arrive in exactly the
same position.

What this file asserts, in order:

1. every copy of the guard in ``rheplicant.core`` is in ``LOOKUP_GUARDED``,
   derived from the source so a sixth copy fails here;
2. each one raises ``KeyError`` on a name the container does not have;
3. each message quotes the missing name AND that container's own names, checked
   against the container rather than by substring -- five ``match=`` patterns
   can all be satisfied by one over-broad message, and this is what that would
   look like if it happened;
4. the five messages differ pairwise, given five containers with distinct
   names, plus the one pin recording what does NOT distinguish them;
5. the successful lookup resolves to the right member -- every fixture carries
   a distinct value, so a guard that fired on a name it should have found, or
   an accessor off by one position, cannot pass.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import rheplicant.core
from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline

#: The sentence that defines the family, matched against each method's own
#: source. Captures the noun so the "branch"/"stage" split is derived too.
_GUARD_SENTENCE = re.compile(r"No (\w+) named \{index!r\}; available: \{self\.names\}")


class _Value(AbstractOperator):
    """A source carrying one distinct number, so members stay tellable apart."""

    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


def _op(value: float) -> _Value:
    return _Value(value=jnp.asarray(value))


#: Swapped in by the ``replace_*`` sites to find out which position a name
#: resolved to. Identity is what matters, not the number.
SENTINEL = _op(-1.0)


@dataclasses.dataclass(frozen=True)
class _Site:
    """One copy of the guard: how to build its container and drive its lookup."""

    noun: str
    names: tuple[str, str]
    values: tuple[float, float]
    build: Callable[[tuple[AbstractOperator, ...], tuple[str, str]], Any]
    members: Callable[[Any], tuple[AbstractOperator, ...]]
    #: (container, key) -> the value of the member the key resolved to.
    resolve: Callable[[Any, Any], jax.Array]


def _getitem_resolve(container, key):
    return container[key].value


def _replace_resolve(attr: str):
    """The value at the position ``replace_*`` decided ``key`` meant.

    Reading the answer off the REPLACED position, rather than off the returned
    container, is what makes the asymmetric fixture values bite: an accessor
    that resolved a name to the wrong member returns a perfectly good container
    and is only visible in *which* member it overwrote.
    """

    def resolve(container, key):
        site = LOOKUP_GUARDED[(type(container).__name__, attr)]
        before = site.members(container)
        after = site.members(getattr(container, attr)(key, SENTINEL))
        replaced = [i for i, member in enumerate(after) if member is SENTINEL]
        assert len(replaced) == 1, f"{attr} replaced {len(replaced)} members, expected 1"
        return before[replaced[0]].value

    return resolve


#: (class name, method name) -> how to exercise that copy. Distinct names and
#: distinct values per site: the names so the five refusals are distinguishable
#: at all (test 4), the values so a lookup that lands on the wrong member is
#: not numerically indistinguishable from one that lands on the right one.
LOOKUP_GUARDED: dict[tuple[str, str], _Site] = {
    ("SumOperator", "__getitem__"): _Site(
        noun="branch",
        names=("sum_get_left", "sum_get_right"),
        values=(2.0, 3.0),
        build=lambda ops, names: SumOperator(*ops, names=names),
        members=lambda c: c.branches,
        resolve=_getitem_resolve,
    ),
    ("SumOperator", "replace_branch"): _Site(
        noun="branch",
        names=("sum_put_left", "sum_put_right"),
        values=(5.0, 7.0),
        build=lambda ops, names: SumOperator(*ops, names=names),
        members=lambda c: c.branches,
        resolve=_replace_resolve("replace_branch"),
    ),
    ("SelectOperator", "__getitem__"): _Site(
        noun="branch",
        names=("select_get_left", "select_get_right"),
        values=(11.0, 13.0),
        build=lambda ops, names: SelectOperator(*ops, names=names),
        members=lambda c: c.branches,
        resolve=_getitem_resolve,
    ),
    ("Pipeline", "__getitem__"): _Site(
        noun="stage",
        names=("pipe_get_first", "pipe_get_second"),
        values=(17.0, 19.0),
        build=lambda ops, names: Pipeline(*ops, names=names),
        members=lambda c: c.stages,
        resolve=_getitem_resolve,
    ),
    ("Pipeline", "replace_stage"): _Site(
        noun="stage",
        names=("pipe_put_first", "pipe_put_second"),
        values=(23.0, 29.0),
        build=lambda ops, names: Pipeline(*ops, names=names),
        members=lambda c: c.stages,
        resolve=_replace_resolve("replace_stage"),
    ),
}

SITE_IDS = sorted(LOOKUP_GUARDED)


def _container(site_id: tuple[str, str]):
    site = LOOKUP_GUARDED[site_id]
    return site.build(tuple(_op(v) for v in site.values), site.names)


def _carrying_the_guard() -> dict[tuple[str, str], str]:
    """{(class, method): noun} for every copy of the sentence in ``rheplicant.core``.

    Scoped to ``core`` because that is where composite operators live and where
    ``self.names`` means "the members of this composite". The near-miss
    sentences elsewhere in the package (``uncertainty`` says "There is no
    parameter named ...", ``numpyro_bridge`` says "samples is missing site ...")
    are deliberately outside the regex: they address a parameter vector and a
    sample dict, not a composite's members, and folding them in would make this
    family a grep result rather than an idea.
    """
    found: dict[tuple[str, str], str] = {}
    for info in pkgutil.iter_modules(rheplicant.core.__path__):
        module = importlib.import_module(f"rheplicant.core.{info.name}")
        for klass in vars(module).values():
            if not inspect.isclass(klass) or klass.__module__ != module.__name__:
                continue
            for method_name, member in vars(klass).items():
                if not inspect.isfunction(member):
                    continue
                try:
                    source = inspect.getsource(member)
                except (OSError, TypeError):
                    # Synthesised, so it has no source file: the methods
                    # `dataclasses` writes for NodeSpec and At. They cannot
                    # carry the guard, and skipping them is not a hole.
                    continue
                match = _GUARD_SENTENCE.search(source)
                if match:
                    found[(klass.__name__, method_name)] = match.group(1)
    return found


def test_the_table_is_the_family_and_the_family_is_the_table():
    """The assertion that makes the rest of this file self-maintaining.

    Derived from the source, so a sixth copy of the guard -- a fifth composite
    type, or the ``replace_branch`` ``SelectOperator`` does not have yet --
    fails here naming itself, instead of shipping with no test the way four of
    the current five did.
    """
    carried = _carrying_the_guard()
    assert set(carried) == set(LOOKUP_GUARDED), {
        "carry the guard but are untested": sorted(set(carried) - set(LOOKUP_GUARDED)),
        "listed but no longer carry it": sorted(set(LOOKUP_GUARDED) - set(carried)),
    }


@pytest.mark.parametrize("site_id", SITE_IDS)
def test_the_derived_noun_matches_the_table(site_id):
    """``branch`` for the combinators, ``stage`` for Pipeline -- read off the source.

    The noun is the only word that varies between the five copies, so it is the
    only thing standing between them and a single indistinguishable sentence.
    """
    assert _carrying_the_guard()[site_id] == LOOKUP_GUARDED[site_id].noun


@pytest.mark.parametrize("site_id", SITE_IDS)
def test_an_unknown_name_is_refused(site_id):
    site = LOOKUP_GUARDED[site_id]
    with pytest.raises(KeyError):
        site.resolve(_container(site_id), "no_such_member")


@pytest.mark.parametrize("site_id", SITE_IDS)
def test_the_refusal_quotes_the_name_and_this_containers_own_names(site_id):
    """Checked against the container, not by substring.

    ``pytest.raises(KeyError, match="No branch named")`` passes for all three
    combinator copies at once, and would keep passing if one of them reported
    some other container's names -- which is the failure this shape of guard
    actually has, since the sentence never says which object refused. Asserting
    the rendered names come from THIS container is the part a substring cannot
    do.
    """
    site = LOOKUP_GUARDED[site_id]
    container = _container(site_id)
    with pytest.raises(KeyError) as excinfo:
        site.resolve(container, "no_such_member")
    message = str(excinfo.value)
    assert "no_such_member" in message, message
    assert repr(container.names) in message, message
    assert f"No {site.noun} named" in message, message


def test_the_five_refusals_differ_pairwise():
    """Five distinct containers, five distinct sentences.

    Directly the "one over-broad message satisfies every substring match"
    check: if a future edit collapsed the noun or dropped ``available:``, some
    pair here collides and this fails, while every ``match=`` in the file above
    would still pass.
    """
    messages = {}
    for site_id in SITE_IDS:
        site = LOOKUP_GUARDED[site_id]
        with pytest.raises(KeyError) as excinfo:
            site.resolve(_container(site_id), "no_such_member")
        messages[site_id] = str(excinfo.value)
    collisions = [
        (a, b)
        for i, a in enumerate(SITE_IDS)
        for b in SITE_IDS[i + 1:]
        if messages[a] == messages[b]
    ]
    assert not collisions, {"identical messages": collisions, "messages": messages}


def test_what_the_message_does_not_distinguish():
    """The honest limit of the sentence above, pinned rather than implied.

    The five messages differ only because the five fixtures were given
    different names. Two accessors on the SAME container produce a
    byte-identical refusal: the sentence identifies the container by its
    members, and never says whether ``__getitem__`` or ``replace_branch``
    turned the name down. That is tolerable -- both are lookups of the same
    name in the same tuple -- but it is the reason the family cannot be tested
    the way the coords-guard family is, where each copy names the operator that
    raised it.
    """
    container = SumOperator(_op(2.0), _op(3.0), names=("left", "right"))
    with pytest.raises(KeyError) as from_getitem:
        container["absent"]
    with pytest.raises(KeyError) as from_replace:
        container.replace_branch("absent", SENTINEL)
    assert str(from_getitem.value) == str(from_replace.value)


@pytest.mark.parametrize("site_id", SITE_IDS)
@pytest.mark.parametrize("position", [0, 1])
def test_a_known_name_resolves_to_that_member(site_id, position):
    """The other branch.

    Without it, an accessor whose lookup always raised would pass every test
    above, and so would one that resolved every name to member zero -- which is
    why the two fixture members carry different values and both positions are
    asked for.
    """
    site = LOOKUP_GUARDED[site_id]
    resolved = site.resolve(_container(site_id), site.names[position])
    assert float(resolved) == site.values[position]


@pytest.mark.parametrize("site_id", SITE_IDS)
@pytest.mark.parametrize("position", [0, 1, -1])
def test_an_integer_index_bypasses_the_name_guard(site_id, position):
    """Integers never enter the guard: it is behind ``isinstance(index, str)``.

    Pinned because it is the other half of the dispatch, and because it decides
    what an out-of-range integer does -- see the test below.
    """
    site = LOOKUP_GUARDED[site_id]
    resolved = site.resolve(_container(site_id), position)
    assert float(resolved) == site.values[position]


@pytest.mark.parametrize("site_id", SITE_IDS)
def test_an_out_of_range_integer_is_an_indexerror_not_the_branded_keyerror(site_id):
    """A number too large is a tuple's refusal, not this family's.

    Worth pinning: a caller catching ``KeyError`` around a lookup does not
    catch the integer route, and the message they get names no container and
    lists no available members.
    """
    site = LOOKUP_GUARDED[site_id]
    with pytest.raises(IndexError):
        site.resolve(_container(site_id), 99)


@pytest.mark.parametrize("site_id", [s for s in SITE_IDS if s[1] != "__getitem__"])
def test_replacing_leaves_the_original_container_untouched(site_id):
    """``replace_*`` returns a new composite; the guard sits on a read, not a write."""
    site = LOOKUP_GUARDED[site_id]
    container = _container(site_id)
    before = tuple(float(m.value) for m in site.members(container))
    getattr(container, site_id[1])(site.names[0], SENTINEL)
    after = tuple(float(m.value) for m in site.members(container))
    assert before == after == site.values
