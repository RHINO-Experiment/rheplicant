"""ConfigError: one refusal type for the whole config layer.

The package's other error classes each name a *stage* -- a State was built
wrong, a pipeline was misconfigured, a file contradicted its declaration. A
config document can be wrong in all of those ways at once and in one place, so
splitting the refusal by stage would ask the reader to guess which stage a key
belongs to before they can catch it. One class, and the message carries the
distinction -- which is where this package puts distinctions anyway.

Not :class:`~rheplicant.core.errors.DataIngestionError`: that one is scoped by
its own docstring to "a data file could not be read, or its contents
contradict what the caller declared about them", and it is confined to
``radio/touchstone.py`` and ``radio/rhino.py``. A config refusal is about what
a document *meant*.
"""

from _rheplicant_bootstrap.errors import ConfigError  # noqa: F401
