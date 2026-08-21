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

from _rheplicant_bootstrap.layering import apply_variant, recursive_update

__all__ = ["apply_variant", "recursive_update"]
