"""inference.npe: the section (schema §4.7.10), and the exit that consumes it.

One feature, one file.  ``parse_npe`` is called from ``build_inference``;
Tasks 7 and 8 append the ``kind: npe`` executor below it, so a reader who
wants to know what ``inference.npe.create.width:`` does finds the grammar and
the call that receives it without changing modules.

**Every optional key here is a real parameter of the call it feeds**, and
that is checked mechanically rather than by eye
(``tests/config/test_config_section_npe.py::TestTheGrammarMatchesTheSignatures``).
Two keys are NOT their parameter's name and the executor translates them:
``seed:`` becomes ``key=`` and ``n_draws:`` becomes ``n_samples=``.  Nothing
else is renamed -- in particular the key is ``width:``, because
``NeuralPosterior.create`` takes ``width=`` and passes it to equinox as
``width_size=`` itself (``npe.py:216``); a grammar that spelled the key
``width_size:`` would ``TypeError`` on the first document that used it.

**Four independent named seeds.**  ``simulate_pairs``, ``create``,
``train_posterior`` and ``sample`` each take a PRNG key of their own, which is
why the seeds live in these subsections rather than on the run: a run carries
one ``seed:`` and this exit needs four.  They are CHECKED here and resolved
nowhere -- a key is a draw and belongs to the run, not to the document read.

**``embed:`` resolves to a callable at parse time**, so a
``{python: "mod:fn"}`` that cannot be imported, or cannot be called with the
one argument ``jax.vmap(embed)(data)`` passes it (``npe.py:209``), is refused
when the document is READ rather than after the bank has been simulated.

Nothing in this module may import ``rheplicant.inference`` at module scope.
``inference.py`` imports this file and ``rheplicant.config`` imports that, so
this module is loaded by every process that reads a document; measured after
``import rheplicant.config``, ``rheplicant.inference`` is absent from
``sys.modules`` and ``numpyro`` with it.  (``equinox`` is already present --
``config/paths.py:30`` imports it and so does much of ``core`` -- so it is
not what the invariant is about.)  The executor's own imports go inside its
body, which is what ``predict``'s samples route already does
(``diagnostics.py:745``).

**THREE TASKS APPEND TO THIS ONE MODULE, so every module-level name is
owned.**  Task 3 (the parser) binds, and no later task may rebind: every
``_*_KEYS``/``_*_OPTIONS`` table, ``_TRANSLATED``, ``_BANK_HINTS``,
``_CREATE_HINTS``, ``_real``, ``_positive``, ``_fraction``, ``_subsection``,
``_seeded``, ``_count``, ``_bank``, ``_sample``, ``_create``, ``_train``,
``_embed``, ``NpeSpec`` and ``parse_npe``.  ``_bank`` and ``_sample`` are the
two most likely to be taken twice: they name the SUBSECTION PARSERS, and the
executor's own simulator and draw are ``_simulate_bank`` and Task 8's, never
these.  A second ``def _bank`` here rebinds the parser and every document
declaring ``inference.npe:`` dies with ``TypeError: _bank() missing 2
required positional arguments``.  ``__all__`` is the one module-level name a
later task may touch, and it is EXTENDED rather than rebound.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from rheplicant.config.draws import _seed_name
from rheplicant.config.errors import ConfigError
from rheplicant.config.hatch import import_target
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.exit_support import _PROBE, _binds
from rheplicant.config.sections.transforms import _whole

__all__ = ["NpeSpec", "parse_npe"]

_NPE_KEYS = frozenset({"bank", "embed", "create", "train", "sample"})

#: The optional knobs of each call, IN THE PACKAGE'S OWN PARAMETER NAMES and
#: as tuples, because that is what ``_passthrough(options, keys)`` takes and
#: Tasks 7 and 8 forward them through it.  A key that is not a parameter of
#: its call is a TypeError at the first run; the mechanical test in
#: ``test_config_section_npe.py`` is what makes that impossible to ship.
_CREATE_OPTIONS = ("n_components", "width", "depth", "min_scale")
_TRAIN_OPTIONS = ("n_steps", "batch_size", "learning_rate",
                  "validation_fraction", "beta1", "beta2", "eps")

_BANK_KEYS = frozenset({"n_simulations", "seed"})
_CREATE_KEYS = frozenset(_CREATE_OPTIONS) | frozenset({"seed"})
_TRAIN_KEYS = frozenset(_TRAIN_OPTIONS) | frozenset({"seed"})
_SAMPLE_KEYS = frozenset({"n_draws", "seed"})

#: The only two config keys whose name is not the parameter's.  A THIRD entry
#: here is a claim that the layer renames something else, and the test asserts
#: this mapping equals its own literal so that adding one cannot pass.
_TRANSLATED = {"seed": "key", "n_draws": "n_samples"}

#: Schema §4.7.10 shows ``bank: {..., cache: {file: {...}}}``.  It does not
#: ship: see the task's Executor's note.  Refused BY NAME rather than as a
#: stray key, because the schema is the document a reader trusts.
_BANK_HINTS = {
    "cache": ("cache: names a file the bank is written to and read back "
              "from; file outputs are Plan 4's (outputs, provenance, the "
              "CLI) and nothing in this layer reads a cache, so a document "
              "that declared one would name a file no run ever writes."),
}

_CREATE_HINTS = {
    "width_size": ("the config key is width:, because create() takes width= "
                   "and passes it to equinox as width_size= itself "
                   "(npe.py:216)."),
}


class NpeSpec(NamedTuple):
    """``inference.npe:`` parsed (schema §4.7.10).

    One entry per subsection rather than five loose dicts, because four of
    the five carry their own seed and a flat mapping cannot say which seed
    belongs to which call.  Each dict holds ``seed`` as the RAW
    ``{from: runtime.seeds.<name>}`` declaration -- the executor resolves it
    to a key at run time through ``draws.seed_for`` (``config/draws.py:46``)
    and needs the declaration, not a key -- beside only the keys the
    document actually declared, so the package's own default applies to
    every key it did not.  (The shared helper that will wrap that resolution,
    ``_draw_key``, arrives with the ``nuts`` executor; it does not exist yet,
    so do not go looking for it.)

    ``embed`` is a CALLABLE, resolved at parse time, ``jnp.ravel`` when the
    document is silent.  That is not a restated default: it is the very
    object ``NeuralPosterior.create``'s signature carries, so passing it and
    omitting it are the same call.
    """

    bank: dict[str, Any]
    embed: Any
    create: dict[str, Any]
    train: dict[str, Any]
    sample: dict[str, Any]


def _real(where: str, value: Any) -> float:
    """A configuration float, with ``bool`` refused: ``True`` is not a rate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: is a number; got {value!r}.")
    return float(value)


def _positive(where: str, value: Any) -> float:
    """``min_scale``, ``learning_rate``, ``eps`` -- strictly above zero.

    ``min_scale: 0`` is the collapse ``MIN_SCALE`` exists to prevent (a
    mixture component on one training point takes the log-density to
    infinity, ``npe.py:59-61``); ``learning_rate: 0`` is a training run that
    returns the untrained estimator; ``eps: 0`` divides by the square root of
    a zero second moment.
    """
    number = _real(where, value)
    if not number > 0.0:
        raise ConfigError(f"{where}: is greater than zero; got {value!r}.")
    return number


def _fraction(where: str, value: Any) -> float:
    """``validation_fraction`` and the two Adam betas -- ``[0, 1)``."""
    number = _real(where, value)
    if not 0.0 <= number < 1.0:
        raise ConfigError(
            f"{where}: is in [0, 1); got {value!r}. The package says the "
            "same of validation_fraction (npe.py:356); of the two Adam "
            "betas it says nothing, and beta1: 1.0 makes 1 - beta1**t "
            "exactly zero -- measured, every training loss after the first "
            "step is NaN and the draws come back non-finite."
        )
    return number


def _subsection(section: Mapping, name: str) -> dict:
    """One of the four required subsections, as a plain dict."""
    where = f"inference.npe.{name}"
    if name not in section:
        raise ConfigError(
            f"{where}: is required. Each of npe's four package calls draws "
            "from a PRNG key of its own and check A29 makes every one of "
            "them a named entry of runtime.seeds, so there is no subsection "
            "this section can do without."
        )
    spec = section[name]
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: is a mapping; got {spec!r}.")
    return dict(spec)


def _seeded(where: str, spec: dict, keys: frozenset[str], label: str,
            hints: dict | None = None) -> dict:
    """Sweep the subsection's keys and check its seed DECLARATION.

    ``_seed_name`` is called for its refusals and its answer is thrown away:
    the name resolves to an integer, and the integer to a key, in the run --
    ``parse_npe`` reads a document and draws nothing.  Its three messages are
    prefixed by the ``form`` argument, which is why every call here passes the
    subsection's dotted name: the four are otherwise word-for-word identical.
    """
    check_unknown_keys(where, spec, keys, label=label, hints=hints)
    _seed_name(spec, where)
    return {"seed": spec["seed"]}


def _count(where: str, spec: dict, key: str, because: str) -> int:
    """A required count, refused when absent and when not a whole number."""
    if key not in spec:
        raise ConfigError(f"{where}.{key}: is required -- {because}")
    return _whole(f"{where}.{key}", spec[key], 1)


def _bank(spec: dict) -> dict:
    where = "inference.npe.bank"
    parsed = _seeded(where, spec, _BANK_KEYS, "the bank", _BANK_HINTS)
    parsed["n_simulations"] = _count(
        where, spec, "n_simulations",
        "simulate_pairs takes it keyword-only with no default (npe.py:72), "
        "and the size of the bank is very nearly the whole cost of an npe "
        "run, so there is no number worth guessing on a document's behalf.")
    return parsed


def _sample(spec: dict) -> dict:
    where = "inference.npe.sample"
    parsed = _seeded(where, spec, _SAMPLE_KEYS, "the draw")
    parsed["n_draws"] = _count(
        where, spec, "n_draws",
        "NeuralPosterior.sample takes n_samples positionally with no "
        "default (npe.py:257). The config key is n_draws, the word "
        "conjugate.gcr already uses for the same quantity, and the "
        "executor is what translates it.")
    return parsed


def _create(spec: dict) -> dict:
    where = "inference.npe.create"
    parsed = _seeded(where, spec, _CREATE_KEYS, "the estimator", _CREATE_HINTS)
    for key in ("n_components", "width"):
        if key in spec:
            # width >= 1, not >= 0: equinox ACCEPTS width_size=0 (measured),
            # and the MLP it builds has no path from input to output -- two
            # inputs 100 apart return the identical vector.  An estimator
            # that cannot see the data still trains, still samples, and
            # reports a posterior that is the prior.
            parsed[key] = _whole(f"{where}.{key}", spec[key], 1)
    if "depth" in spec:
        # depth: 0 IS legal and means one linear layer (measured).
        parsed["depth"] = _whole(f"{where}.depth", spec["depth"], 0)
    if "min_scale" in spec:
        parsed["min_scale"] = _positive(f"{where}.min_scale",
                                        spec["min_scale"])
    return parsed


def _train(spec: dict) -> dict:
    where = "inference.npe.train"
    parsed = _seeded(where, spec, _TRAIN_KEYS, "training")
    for key in ("n_steps", "batch_size"):
        if key in spec:
            parsed[key] = _whole(f"{where}.{key}", spec[key], 1)
    for key in ("learning_rate", "eps"):
        if key in spec:
            parsed[key] = _positive(f"{where}.{key}", spec[key])
    for key in ("validation_fraction", "beta1", "beta2"):
        if key in spec:
            parsed[key] = _fraction(f"{where}.{key}", spec[key])
    return parsed


def _embed(node: Any) -> Any:
    """``embed:`` -> the callable ``NeuralPosterior.create`` receives.

    Resolved HERE, at document-read time, rather than in the executor.  The
    alternative is a run that simulates the bank -- the expensive half -- and
    then dies on an import the layer could have refused before it started.
    """
    where = "inference.npe.embed"
    if node is None or node == "ravel":
        return jnp.ravel
    if isinstance(node, Mapping) and "python" in node:
        if set(node) != {"python"}:
            raise ConfigError(
                f"{where}: {sorted(set(node) - {'python'})} rides beside "
                "python: in the value grammar and not here. embed: hands "
                "the estimator a CALLABLE, and args:/literal: is how the "
                "hatch spells CALLING one (hatch.py's presence-of-the-key "
                "rule), so a factory that must be called to produce an "
                "embedding has no spelling in this key."
            )
        target = node["python"]
        embedding = import_target(target)
        if not callable(embedding):
            raise ConfigError(
                f"{where}: {target!r} is a "
                f"{type(embedding).__name__} and embed: takes a callable. "
                "The hatch hands over the attribute itself when no args: is "
                "written (hatch.py's presence-of-the-key rule), so a target "
                "naming a constant resolves cleanly and dies inside "
                "jax.vmap. _binds cannot see this one: inspect describes no "
                "signature for it, and _binds passes an indescribable "
                "callable through by design."
            )
        binds, signature = _binds(embedding, _PROBE)
        if not binds:
            raise ConfigError(
                f"{where}: {target!r} cannot be called as (datum) -- its "
                f"signature is {signature}. The estimator embeds ONE "
                "simulated datum at a time (jax.vmap(embed)(data), "
                "npe.py:209), so an embedding takes the datum and nothing "
                "else."
            )
        return embedding
    raise ConfigError(
        f"{where}: is 'ravel' or {{python: 'mod:fn'}}; got {node!r}."
    )


def parse_npe(section: Any, context: Any) -> NpeSpec:
    """``inference.npe:`` -> a :class:`NpeSpec`.  Called by ``build_inference``.

    ``context`` is pinned by the plan's §3.1 and unused: nothing in this
    section is a value node, so nothing resolves.  It is kept rather than
    dropped because the first key that does resolve one -- Plan 4's bank
    ``cache: {file: ...}`` is the obvious candidate -- would otherwise change
    this signature and its call site together.

    Raises:
        ConfigError: on anything the grammar does not accept, naming the
            subsection as ``inference.npe.<sub>:``.
    """
    if not isinstance(section, Mapping):
        raise ConfigError(
            "inference.npe: is a mapping with bank:, embed:, create:, "
            f"train: and sample:; got {section!r}."
        )
    check_unknown_keys("inference.npe", dict(section), _NPE_KEYS,
                       label="the npe section")
    return NpeSpec(bank=_bank(_subsection(section, "bank")),
                   embed=_embed(section.get("embed")),
                   create=_create(_subsection(section, "create")),
                   train=_train(_subsection(section, "train")),
                   sample=_sample(_subsection(section, "sample")))
