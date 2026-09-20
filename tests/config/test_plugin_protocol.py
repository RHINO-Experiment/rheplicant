"""The five registration hooks are private, and that is the current answer.

A3-5 recorded the plugin protocol as undeclared and unversioned, with an open
question: which registries should be public. Measured 2026-09-20, the question
has a simpler answer than it looks -- **none of the five hooks is on any
``__all__``**, so there is no public protocol to version, and publishing a
``PLUGIN_API_VERSION`` would declare a contract that no supported import
reaches.

So this pins the status quo rather than inventing a version. Making one of
these public becomes a decision someone takes, in a commit that also has to
answer what the protocol promises -- instead of a name slipping onto a surface
and becoming a promise by accident.

The direction is the one the rest of this baseline uses: the code is the
truth, and ``docs/stability.md`` says what the code does.
"""

from __future__ import annotations

import importlib

import pytest

#: hook -> the module that defines it.
HOOKS = {
    "register_kind": "rheplicant.config.resources",
    "register_reader": "rheplicant.config.files",
    "register_form": "rheplicant.config.values",
    "register_derivation": "rheplicant.config.derive",
    "register_formula_checked": "rheplicant.config.dimensions",
}

NAMESPACES = (
    "rheplicant",
    "rheplicant.core",
    "rheplicant.radio",
    "rheplicant.inference",
    "rheplicant.config",
    "rheplicant.gui",
)


@pytest.mark.parametrize(("hook", "module"), sorted(HOOKS.items()), ids=sorted(HOOKS))
def test_the_hook_exists_where_this_file_says_it_does(hook, module):
    """Anti-vacuity. A table naming five functions that have moved would assert
    their absence from every ``__all__`` and pass for the wrong reason."""
    assert hasattr(importlib.import_module(module), hook), (
        f"{module} no longer defines {hook}; this table is stale and the "
        "privacy assertion below has stopped being about anything"
    )


@pytest.mark.parametrize("hook", sorted(HOOKS), ids=sorted(HOOKS))
def test_the_hook_is_on_no_public_surface(hook):
    published = [
        namespace
        for namespace in NAMESPACES
        if hook in getattr(importlib.import_module(namespace), "__all__", ())
    ]
    assert not published, (
        f"{hook} is now published from {published}. That makes the plugin "
        "registration protocol public, which is a decision rather than an "
        "export: it needs a stated contract, a version, and an update to "
        "docs/stability.md, which currently tells readers the protocol is "
        "not public"
    )
