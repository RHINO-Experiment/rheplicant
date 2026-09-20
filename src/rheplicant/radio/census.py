"""The capability census: which classes this package ships, and at what level.

One walk, in ``src/``, so that everything needing the answer is a VIEW of it
rather than another copy of the walk. Before this there were two: a test that
walked ``operator_table()`` and a census test that walked
``radio.__all__`` with its own abstract-class filter, and they disagreed about
two operators for as long as nobody compared them.

The level itself is never computed here. It is read off the ``maturity``
ClassVar each class declares, which is the registry; this module only decides
WHICH classes are asked. That division is the point: adding a class cannot
change a level, and changing a level cannot change the population.

The walk is lazy for the same reason ``config.sections.model.operator_table``
is -- ``rheplicant.radio.__init__`` is what defines ``__all__``, so a
module-scope import of it from inside the package would read a
half-initialised module.
"""

from __future__ import annotations

import inspect

from rheplicant.core.capability import Maturity


def _capability_bases() -> tuple[type, ...]:
    """The three bases that make a public class a capability.

    An operator, a sky model or a sky projector is something this package
    offers to do. ``Touchstone`` and ``RhinoObservation`` are parsed-data
    RECORDS and fall out here without being named: the capability is the
    reader that returns one, and readers are functions.
    """
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.radio.sky.model import AbstractSkyModel
    from rheplicant.radio.sky.projection import AbstractSkyProjector

    return (AbstractOperator, AbstractSkyModel, AbstractSkyProjector)


def capability_classes() -> dict[str, type]:
    """Every CONCRETE capability class on the public radio surface, by name.

    Abstract classes are excluded, and they are excluded by
    ``inspect.isabstract`` plus identity against the bases rather than by a
    list of names. A base must not carry a level at all -- it has nothing to
    claim -- and ``tests/core/test_capability_registry.py`` asserts that in
    the other direction.
    """
    import rheplicant.radio as radio

    bases = _capability_bases()
    found: dict[str, type] = {}
    for name in radio.__all__:
        obj = getattr(radio, name, None)
        if not inspect.isclass(obj) or not issubclass(obj, bases):
            continue
        if obj in bases or inspect.isabstract(obj):
            continue
        found[name] = obj
    return found


def capabilities() -> dict[str, Maturity]:
    """Name -> declared level, for every concrete capability this package ships.

    The answer generated documentation, the GUI's disabled/reason and the
    placeholder census are all views of. A class with no ``maturity`` raises
    ``AttributeError`` here rather than being reported at some default, which
    is why the base classes deliberately carry no value.
    """
    return {name: cls.maturity for name, cls in capability_classes().items()}


def at_level(level: Maturity) -> frozenset[str]:
    """The names declared at one level."""
    return frozenset(name for name, declared in capabilities().items() if declared is level)
