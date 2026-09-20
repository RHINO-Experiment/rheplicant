"""Neutral exceptions shared across bootstrap and public rheplicant modules."""

from __future__ import annotations


class DirtError(Exception):
    """Base class for all public rheplicant errors."""


class ConfigError(DirtError, ValueError):
    """A configuration document was refused."""

    def __init__(self, *args: object, report: object | None = None) -> None:
        super().__init__(*args)
        self.report = report


class AssemblyError(DirtError, ValueError):
    """A provided operator set cannot be assembled on the signal graph."""


#: What the command line reports as a REFUSAL (exit 2, a refused audit)
#: rather than as an internal failure (exit 1, a traceback). An
#: ``AssemblyError`` is the fold refusing an operator set, and on the command
#: line every operator set comes from the user's document, so it is a refusal
#: of that document wherever in a run it surfaces. It is defined here, beside
#: ``ConfigError``, because the bootstrap may not import ``rheplicant`` to
#: name it.
REFUSALS: tuple[type[DirtError], ...] = (ConfigError, AssemblyError)


DirtError.__module__ = "rheplicant.core.errors"
ConfigError.__module__ = "rheplicant.config.errors"
AssemblyError.__module__ = "rheplicant.core.errors"
