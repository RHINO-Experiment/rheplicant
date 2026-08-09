"""rheplicant.config: turn a declarative document into a running twin.

This layer sits ABOVE ``rheplicant.core`` and ``rheplicant.radio``: it imports
both and neither imports it. That direction is what keeps ``core``
extractable, and ``tests/core/test_layering.py`` enforces it mechanically.

Plan 1A ships the value grammar only -- the eight forms of a value node, the
eight modifiers, and the rule that decides whether a resolved value reaches a
field as a Python scalar or as a traced array.
"""

from rheplicant.config.errors import ConfigError

__all__ = ["ConfigError"]
