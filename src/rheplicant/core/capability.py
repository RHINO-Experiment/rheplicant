"""Capability levels, re-exported into the layer that declares them.

The enum itself is defined in the bootstrap package, which is the JAX-free
side of this checkout -- see :mod:`_rheplicant_bootstrap.capability` for why
the GUI needs it there. This module is the same arrangement
:mod:`rheplicant.core.errors` already uses for ``AssemblyError``: the
definition sits in the bootstrap, and the name is documented, imported and
pickled from here, so the layers above import it the ordinary way
(``from rheplicant.core.capability import Maturity``) and gain no edge of their
own into the bootstrap.

That matters more than it looks. Measured 2026-09-20, ``rheplicant.config`` has
80 imports of the bootstrap and ``rheplicant.gui`` has 34, but ``radio`` and
``inference`` have **none**. Twenty-eight operator classes importing the enum
directly would have opened that edge for a vocabulary word. This file keeps the
count where it was and makes ``core`` the one place that reaches across.
"""

from __future__ import annotations

from _rheplicant_bootstrap.capability import Maturity

__all__ = ["Maturity"]

# The enum is documented and pickled from this module rather than the
# bootstrap, matching `core.errors`. Without this, `repr` and Sphinx would
# name a private package in the public API reference.
Maturity.__module__ = __name__
