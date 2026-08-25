"""One comparator, and the disagreement that made it necessary.

``gui/document.py::_same_value`` and ``gui/validation.py::_same`` were two
answers to "is this the same YAML value?" and they differed. These tests pin
that there is now one implementation, that it is the STRICT one, and -- the
part that matters most -- they pin the exact inputs the two used to disagree
on, so a future re-spelling has something to fail against.

The review that filed this said the two "answer differently for key-reordered
mappings". Measured, they do not: both walk mappings by lookup and both are
order-independent. That is asserted below too, because a corrected finding is
worth keeping in a form that stays corrected.
"""

import pytest

from rheplicant.gui.document import _same_value
from rheplicant.gui.validation import _same
from rheplicant.gui.yaml_values import same_value


def test_all_three_names_are_one_function():
    """Identity, not equality of behaviour: two implementations that agree on
    every case anyone thought to test is exactly the state this replaced."""
    assert _same_value is same_value
    assert _same is same_value


#: ``(label, left, right)`` where a dict-lookup comparator says "same" and a
#: value-matching one says "different". Measured against the two shipped
#: implementations before they were unified.
COLLAPSING = [
    ("bool against int key", {1: "x"}, {True: "x"}),
    ("float against int key", {1: "x"}, {1.0: "x"}),
    ("bool key nested one level", {"o": {0: "x"}}, {"o": {False: "x"}}),
    ("numeric key inside a sequence", [{1: "x"}], [{True: "x"}]),
]


@pytest.mark.parametrize(("label", "left", "right"), COLLAPSING)
def test_numeric_keys_are_kept_apart(label, left, right):
    """``1``, ``True`` and ``1.0`` are one key to a dict and three scalars to
    YAML. The document that writes ``true:`` where ``1:`` stood HAS changed,
    and the comparator that says otherwise reports a real difference as none.
    """
    assert same_value(left, right) is False, label
    # Anti-vacuity: the fixture differs ONLY in the key's type, so a
    # comparator that returned False for everything would not be evidence.
    assert same_value(left, left) is True, label


@pytest.mark.parametrize(("label", "left", "right"), COLLAPSING)
def test_a_dict_lookup_comparator_would_have_said_these_are_the_same(
    label, left, right
):
    """The counterfactual, computed rather than asserted in prose.

    This is the lax comparator that used to live in ``validation.py``,
    rewritten here in four lines. It exists so the cases above are known to
    be DISCRIMINATING: if Python ever stopped collapsing these keys, the
    tests above would still pass while testing nothing, and this one would go
    red and say why.
    """

    def lax(left_value, right_value):
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            return len(left_value) == len(right_value) and all(
                key in right_value and lax(value, right_value[key])
                for key, value in left_value.items()
            )
        if isinstance(left_value, list) and isinstance(right_value, list):
            return len(left_value) == len(right_value) and all(
                lax(a, b) for a, b in zip(left_value, right_value, strict=True)
            )
        return type(left_value) is type(right_value) and left_value == right_value

    assert lax(left, right) is True, label


def test_key_order_was_never_the_difference():
    """The review's stated symptom, measured false and kept measured."""
    assert same_value({"a": 1, "b": 2}, {"b": 2, "a": 1}) is True


@pytest.mark.parametrize(
    ("label", "left", "right", "expected"),
    [
        ("bool against int VALUE", {"a": 1}, {"a": True}, False),
        ("none against false", {"a": None}, {"a": False}, False),
        ("str against int key", {"1": "x"}, {1: "x"}, False),
        ("list against tuple", {"a": [1, 2]}, {"a": (1, 2)}, True),
        ("extra key", {"a": 1}, {"a": 1, "b": 2}, False),
        ("equal documents", {"a": [1, {"b": "c"}]}, {"a": [1, {"b": "c"}]}, True),
    ],
)
def test_the_cases_both_implementations_always_agreed_on(label, left, right, expected):
    """Values were never the disagreement -- both compared ``type`` before
    ``==``. Pinned so the unification is known to have changed only what it
    meant to: a sequence compared across list and tuple stays the same value,
    because YAML has one sequence type and the difference is Python's.
    """
    assert same_value(left, right) is expected, label
