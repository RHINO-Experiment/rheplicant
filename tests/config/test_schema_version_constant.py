"""The configuration grammar's version is spelled once and derived everywhere.

A3-2 measured it in about eight places: this module's default and constructor,
three refusal messages, the equality that accepts a document, the published
``json_schema()["schemaVersion"]`` and the GUI's starter document. A version
living in eight places is a version that will be bumped in seven.

``_rheplicant_bootstrap.process.SCHEMA_VERSION`` is now the one. What this
file does is hold the places that CANNOT derive it to the ones that do.

**Two places deliberately still write the number as text**, and that is the
interesting part rather than an exception to apologise for.

The three refusal sentences are pinned verbatim by
``test_config_preflight.py::TestNoMovedMessageWasReworded``, which
reconstructs each from the source literal. Interpolating a constant splits the
literal across two f-strings and the guard stops finding it -- measured, not
predicted: doing so turned that test red with "1 message(s) this layer shipped
at be2027b are gone". So the constant governs behaviour and the sentences stay
prose, and the test below is what stops them drifting apart: bump the constant
and the messages go red rather than quietly going on saying 1.

The starter document substitutes rather than interpolating, because the
template carries literal braces for the config grammar's own syntax and an
f-string turns those into fields. Also measured -- ruff reported an undefined
name.
"""

from __future__ import annotations

from _rheplicant_bootstrap.process import SCHEMA_VERSION, schema_version_problem
from rheplicant.config.schema import json_schema
from rheplicant.gui.starter import STARTER_YAML


def test_the_constant_is_the_version_a_document_may_declare():
    """Anti-vacuity for everything below: the number must actually be accepted."""
    assert schema_version_problem(SCHEMA_VERSION) is None, (
        f"SCHEMA_VERSION is {SCHEMA_VERSION} and a document declaring it is "
        f"refused: {schema_version_problem(SCHEMA_VERSION)}"
    )


def test_the_neighbours_of_the_constant_are_refused():
    """So the acceptance above is about this number, not about integers."""
    for neighbour in (SCHEMA_VERSION - 1, SCHEMA_VERSION + 1):
        assert schema_version_problem(neighbour) is not None, (
            f"schema_version {neighbour} is accepted, so the check is not "
            f"about {SCHEMA_VERSION} at all"
        )


def test_the_published_schema_renders_the_constant():
    """``json_schema()`` is a string by contract; the NUMBER is the same one."""
    published = json_schema()["schemaVersion"]
    assert published == str(SCHEMA_VERSION), (
        f"json_schema() publishes schemaVersion {published!r} while the "
        f"grammar is at {SCHEMA_VERSION}. A consumer branches on that string"
    )


def test_the_starter_document_declares_the_constant():
    """The one document in the package guaranteed to be loaded by a new user.

    If it declared a version the loader refuses, the workbench would open on a
    document that cannot load -- the worst possible first impression, and one
    no other test would notice because the starter is a string constant.
    """
    first = STARTER_YAML.splitlines()[0]
    assert first == f"schema_version: {SCHEMA_VERSION}", (
        f"the starter document opens with {first!r} and the grammar is at {SCHEMA_VERSION}"
    )
    assert "{SCHEMA_VERSION}" not in STARTER_YAML, (
        "the starter still carries the substitution placeholder, so the "
        "replace() at the foot of starter.py did not run"
    )


def test_the_starter_document_actually_loads():
    """The claim above is about a line; this is about the document.

    Cheap, and it is the only thing that says the starter is a document rather
    than a string that begins correctly.
    """
    from _rheplicant_bootstrap.yaml import safe_load_document
    from rheplicant.config import load_document
    from rheplicant.config.errors import ConfigError

    # `load_document` takes a MAPPING, not YAML text: handed a string it
    # answers "A document is a mapping of sections; got str", which is a
    # refusal about the argument and says nothing about the document.
    # `safe_load_document` takes BYTES and returns a `LoadedYaml`, whose
    # `.value` is the mapping -- three signatures read rather than guessed,
    # after guessing each one wrong first.
    try:
        parsed = safe_load_document(STARTER_YAML.encode(), source_name="starter")
        load_document(parsed.value)
    except ConfigError as refusal:
        raise AssertionError(f"the GUI's starter document does not load: {refusal}") from refusal


def test_the_refusal_sentences_still_name_the_current_version():
    """Prose and constant, held together.

    The three sentences write the number as text because a pinned-message
    guard reconstructs them from the source literal. That is a reasonable
    trade and it has a cost: nothing in the sentence knows the constant. This
    is what pays that cost.
    """
    for probe in ("not an integer", SCHEMA_VERSION + 1, SCHEMA_VERSION - 1):
        message = schema_version_problem(probe)
        assert message is not None, f"{probe!r} was accepted"
        assert f"schema_version {SCHEMA_VERSION}" in message or (
            f"schema_version: {SCHEMA_VERSION} is required" in message
        ), (
            f"the refusal for {probe!r} does not name version "
            f"{SCHEMA_VERSION}: {message!r}. The constant moved and the "
            "sentences did not -- they are prose and cannot follow it, which "
            "is why this test exists"
        )
