"""Document layering: named patches over one base document.

``variants:`` entries (and, when Plan 4 lands, ``defaults:`` presets) are deep
merges with one shared rule set -- mappings merge, lists replace,
``{append: [...]}`` extends, ``~key: null`` deletes. The algorithm itself is
:func:`rheplicant.config.resources.merge_extends`, shipped by Plan 1B for
``extends:``; this module applies it at document level rather than writing a
second merge that could drift from the first.

Layering is ONE level deep by design (schema §4.6): there is no ordering
between variants, no variant references another, and a variant may not declare
variants of its own. That is what keeps a comparison's two halves from
disagreeing in exactly the keys the comparison is about.
"""

from __future__ import annotations

from collections.abc import Mapping

from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import merge_extends

__all__ = ["apply_variant", "recursive_update"]


def recursive_update(base: Mapping, patch: Mapping) -> dict:
    """A NEW mapping: ``patch`` deep-merged over ``base``; neither is mutated.

    Mappings merge, lists replace, ``{append: [...]}`` extends an inherited
    list, ``~key: null`` deletes an inherited key.
    """
    for label, given in (("base", base), ("patch", patch)):
        if not isinstance(given, Mapping):
            raise ConfigError(
                f"recursive_update: {label} is a mapping; got "
                f"{type(given).__name__} ({given!r})."
            )
    return merge_extends(dict(patch), dict(base))


def apply_variant(document: Mapping, name: str) -> dict:
    """The base document with the named variant's patch merged over it.

    Raises:
        ConfigError: when the document declares no variants, the name is not
            among them, the patch is not a mapping, or the patch tries to
            declare ``variants`` / rewrite ``schema_version``.
    """
    variants = document.get("variants") or {}
    if not isinstance(variants, Mapping):
        raise ConfigError(
            f"variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__} ({variants!r})."
        )
    if not variants:
        raise ConfigError(
            f"variant {name!r} was requested but this document declares no variants."
        )
    if name not in variants:
        raise ConfigError(
            f"variant {name!r} is not declared; this document declares "
            f"{sorted(variants)}."
        )
    patch = variants[name]
    if not isinstance(patch, Mapping):
        raise ConfigError(
            f"variant {name!r}: the patch is a mapping of sections; got "
            f"{type(patch).__name__} ({patch!r})."
        )
    for key in ("variants", "~variants"):
        if key in patch:
            raise ConfigError(
                f"variant {name!r} declares {key!r}. Layering is one level "
                "deep by design: there is no ordering between variants and "
                "no variant builds on another, so a comparison's halves "
                "cannot drift apart through a chain."
            )
    for key in ("schema_version", "~schema_version"):
        if key in patch:
            raise ConfigError(
                f"variant {name!r} touches {key!r}. The version belongs to "
                "the document; a patch that changes -- or deletes -- how the "
                "document is read is not a patch."
            )
    return recursive_update(document, patch)
