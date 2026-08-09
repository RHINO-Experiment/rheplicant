"""Importing this package is how every resource-kind builder gets registered.

``arrays`` today; Tasks 4-8 add the remaining five, one import each, until the
set of six is complete.

One import rather than six: a kind that is defined but never imported is a
kind the registry does not have, and the failure is an "unknown kind" refusal
that lists a set which is silently short.
"""

# `_arrays_kind`, not `_arrays`: `rheplicant.config.arrays` is form 2 of the
# value grammar (the array constructors), an unrelated module one `grep
# arrays` away from this one. The alias names which of the two this import is.
from rheplicant.config.kinds import arrays as _arrays_kind  # noqa: F401  (registers 'arrays')
