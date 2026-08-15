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
(``diagnostics.py:771``).

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

*Task 7 binds three more*, plus the ``_decided_model``/``_passthrough``
imports it added to the ``exit_support`` list and the ``_draw_key``/
``_sampled_space`` imports it added from ``posterior_support``:
:func:`_npe_spec`, :func:`_simulate_bank` and :func:`_estimator`.  **Task 8
bound ``NpeProduct`` and ``_run_npe`` and nothing else**, plus the
``_observed``/``_sweep``/``register`` imports it added to the
``exit_support`` list and the ``_unravel`` import it added from
``posterior_support``; it also EXTENDED ``__all__``, which is the one
module-level name a later task may touch.  It was the last task of Plan 2D to
write this file: the draw is ``_run_npe``'s own business and never
``_sample``.  Twenty-four names from Task 3, three from Task 7, two from Task
8 -- and the reason the inventory is kept here rather than left to plan §3.1
is that §3.1 lists seven of the twenty-four, and a drafter who reads an
authoritative list and does not find the name they need concludes the name is
free.  That is how three names in this file came to be bound twice before the
plan was executed at all.

*Plan 3A's Task 8 binds one more*, and it is a SPLIT rather than a new
feature: :func:`_a29_npe_takes_no_run_seed` is the refusal that used to sit
inline at the head of :func:`_run_npe`, lifted to module level so that
``config/preflight/fitting.py`` can call the same object from the raw
document before any beam is read (plan §2.2: one name, one binding, two call
sites).  The three counts above are Plan 2D's and are left as they were --
they are the per-task lists a later drafter actually needs, and re-totalling
them here would put a number in this docstring that no test defends.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from rheplicant.config.draws import _seed_name
from rheplicant.config.errors import ConfigError
from rheplicant.config.hatch import import_target
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.exit_support import (
    _PROBE,
    _binds,
    _decided_model,
    _observed,
    _passthrough,
    _sweep,
    register,
)
from rheplicant.config.sections.posterior_support import (
    _draw_key,
    _sampled_space,
    _unravel,
)
from rheplicant.config.sections.transforms import _whole

__all__ = ["NpeProduct", "NpeSpec", "parse_npe"]

_NPE_KEYS = frozenset({"bank", "embed", "create", "train", "sample"})

#: What ``kind: npe`` tells ``_decided_model`` it wants the noise RULE for,
#: and what it offers a document that decided its sigma into an array
#: (check A28).  **Neither clause existed until Plan 3A**: the accessor wrote
#: ONE sentence for both its callers, ``conjugate.gls``'s, so a ``kind: npe``
#: run was told it "solves for the covariance a PREDICTION-DEPENDENT sigma
#: implies" and was offered ``kind: conjugate.wiener``.  Measured on
#: ``posterior_helpers.npe_document(noise=FROZEN)``, both were false: this
#: exit hands ``noise=`` to ``simulate_pairs``, which DRAWS from the rule
#: (``:475-480``), and no conjugate exit produces an amortized posterior.
#: ``_decided_model`` takes both keyword-only and REQUIRED, so a third caller
#: has no default left to inherit -- which is exactly how this defect
#: arrived.
_A28_NPE_CLAUSES: dict[str, str] = {
    "wants": ("SIMULATES a bank of (theta, data) pairs and draws the noise "
              "for each one"),
    "instead": ("Declare inference.noise.kind: radiometer or homoscedastic "
                "-- either is a rule simulate_pairs can draw from. There is "
                "no amortized-posterior exit that takes a decided array, so "
                "the sigma is what has to change."),
}

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


# --- The kind: npe exit (Tasks 7 and 8) -------------------------------------
#
# The parser above and the executor below share a file on purpose: one
# feature, one place.  ``inference.npe:`` exists only to be read by
# ``kind: npe``, and a grammar whose consumer lives in another module is a
# grammar that can drift from it silently.
#
# NOTHING here imports ``rheplicant.inference`` at module scope -- see the
# module docstring's import rule.  Every package import in this section sits
# inside a function body.
#
# What ``create`` forwards is ``_CREATE_OPTIONS``, ALREADY DEFINED AT THE HEAD
# OF THIS MODULE -- exactly ``NeuralPosterior.create``'s keyword-only knobs
# minus ``key`` and ``embed``, which travel on their own, and the same tuple
# the grammar sweeps and ``test_config_section_npe.py`` checks against
# ``inspect.signature``.  DO NOT RE-TYPE IT AT A CALL SITE.  A local
# ``_CREATE_KEYS`` tuple here would collide with the frozenset of that name
# above and turn ``check_unknown_keys``' ``set(spec) - allowed`` into a raw
# ``TypeError: unsupported operand type(s) for -: 'set' and 'tuple'`` on every
# document that declared ``inference.npe.create:``.


def _npe_spec(run: Any, built: Any) -> Any:
    """The parsed ``inference.npe:``, or a refusal saying where it goes.

    ``kind: npe`` takes no run-level keys at all: the bank size, the
    embedding, the estimator's shape, the training schedule, the draw count
    and FOUR named seeds all live in the section, because a run carries one
    seed and this exit draws four times.
    """
    spec = built.inference.npe
    if spec is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: npe trains an amortized posterior and "
            "reads every knob from inference.npe: -- bank:, embed:, create:, "
            "train: and sample:, four of them with a seed: "
            "{from: runtime.seeds.<name>} of their own (check A29) -- and "
            "this document declares no inference.npe:."
        )
    return spec


def _simulate_bank(run: Any, built: Any, spec: Any) -> tuple:
    """``(space, thetas, data)`` -- the simulation bank, and its layout.

    NOT ``_bank``: Task 3 already binds that name in this module to the
    ``bank:`` SUBSECTION PARSER that :func:`parse_npe` calls.  A second
    module-level ``def _bank`` here rebinds it, and every document declaring
    ``inference.npe:`` then dies with ``TypeError: _bank() missing 2 required
    positional arguments``.

    The space is returned ALONGSIDE the pairs rather than fetched again by the
    caller, because ``thetas``' columns are laid out in THAT space's ``names``
    order (``inference/npe.py:100-102``) and Task 8 unravels the draws back
    through the same object.  Two separate lookups are two chances to check
    one space and unravel against another.

    ``noise=`` is keyword-only and takes the NoiseModel WHOLE -- the rule, not
    a decided array -- so this reads :func:`_decided_model` and must never
    read ``_decided_sigma``.  Measured, a decided array here does not raise
    the ``ParameterSpaceError`` the plan predicted: it dies inside
    ``simulate_pairs``' vmap as ``jax.errors.ConcretizationTypeError: ... The
    axis argument must be known statically``, which names no run and no
    document key.

    The pipeline is the FIT twin and not ``built.twin``.  That one is loud
    rather than silent -- measured, banking from it raises
    ``ParameterSpaceError``, because the model's ``NoiseOperator`` declares
    ``key`` in ``requires`` and a space refuses a forward model that draws --
    so every test that simulates a bank kills the mutation.
    """
    from rheplicant.inference import simulate_pairs

    space = _sampled_space(run, built, route="npe")
    thetas, data = simulate_pairs(
        built.inference.fit_twin, built.state, space,
        noise=_decided_model(run, built, **_A28_NPE_CLAUSES),
        key=_draw_key(run, "inference.npe.bank", built, spec.bank),
        n_simulations=spec.bank["n_simulations"],
    )
    return space, thetas, data


def _estimator(run: Any, built: Any, spec: Any, thetas: Any, data: Any) -> Any:
    """An untrained ``NeuralPosterior``, sized and standardized to the bank.

    ``embed`` is already a callable -- :func:`parse_npe` resolves it, so a bad
    ``{python:}`` is refused when the document is READ rather than after the
    bank has been simulated -- and it is passed unconditionally because the
    parser's own silent default IS ``jnp.ravel``, which is the package's.

    Everything else travels through ``_passthrough``, so an undeclared key
    gets the package's default rather than one restated here.  That matters
    most for ``n_components``: the package's default is 4 and its own tuning
    table says 4 over-fits (``tests/inference/test_npe.py:145`` and the
    shipped example both pass 1).  The layer documents that and defaults
    nothing -- a warning is Plan 3's, on its ledger.

    ``width=``, never ``width_size=``.  ``create`` takes ``width`` and passes
    it to equinox as ``width_size`` itself (``inference/npe.py:216``); 2C's
    carry-forward note reads that fact as an instruction to the caller and is
    wrong about the caller, and the wrong spelling is a ``TypeError`` on the
    first call.

    The ``where`` strings the two ``_draw_key`` calls carry are DEFENSIVE and
    no test in this plan can reach them: they prefix ``draws._seed_name``'s
    refusals, and ``parse_npe`` has already run ``_seed_name`` over every
    subsection by the time an executor sees the spec, so a seed that could
    trigger one never reaches here.  They are written correctly rather than
    left blank because Task 8 adds two more of them and a blank one would
    read as the pattern to copy.
    """
    from rheplicant.inference import NeuralPosterior

    return NeuralPosterior.create(
        thetas, data,
        key=_draw_key(run, "inference.npe.create", built, spec.create),
        embed=spec.embed,
        **_passthrough(spec.create, _CREATE_OPTIONS),
    )


# What ``train_posterior`` forwards is ``_TRAIN_OPTIONS``, ALREADY DEFINED BY
# TASK 3 at the head of this module.  **Do not re-type it here.**  A draft of
# this step declared a local ``_TRAIN_KEYS`` tuple, which collided with Task
# 3's ``_TRAIN_KEYS`` frozenset -- the one the grammar sweeps and Task 3's own
# test module imports -- and turned ``check_unknown_keys``' ``set(spec) -
# allowed`` into a raw ``TypeError`` on every document declaring
# ``inference.npe.train:``.  ``key`` travels on its own, and there is no
# ``validation_fraction`` default anywhere: the package's is 0.1, and a
# document that declares 0.0 gets an EMPTY validation curve, which this
# executor carries as it is and never indexes.


class NpeProduct(NamedTuple):
    """What a ``kind: npe`` run returns.

    ``samples`` and ``n_draw`` are not free choices and carry the same
    contract :class:`~rheplicant.config.sections.nuts.NutsProduct` does, for
    the same reason: 2C's shipped ``predict`` reads a samples product as
    ``product.n_draw`` (an int) and ``product.samples`` (a mapping of latent
    name -> stack with a leading draw axis), ``diagnostics.py:774`` and
    ``:791``, and Task 9 makes ``npe`` one of its sources.
    ``NeuralPosterior.sample`` returns a FLAT ``(n_draws, n_params)`` array,
    so :func:`~rheplicant.config.sections.posterior_support._unravel` is what
    gets it to a mapping, in ``space.names`` DECLARATION order.

    ``posterior`` is the trained estimator, so a caller can ``log_prob``
    against it -- the amortized half of what NPE is for.

    ``best_step`` is an ``int``: ``train_posterior`` returns a traced
    ``ArrayImpl``, it is 1-based, and it is 50 after 50 steps even when
    ``validation_loss`` is empty, so it is no signal that validation happened.

    ``validation_loss`` is ``(n_steps,)``, or ``(0,)`` when the document
    declares ``validation_fraction: 0.0``.  **Nothing in this layer indexes
    it** -- ``history.validation[-1]`` on the empty one raises IndexError, and
    the honest product is the empty curve rather than an invented number.  A
    consumer that plots it must check ``.size`` first, which is recorded to
    the plan's ledger for Plan 3.
    """

    samples: dict[str, Any]
    n_draw: int
    posterior: Any
    best_step: int
    train_loss: Any
    validation_loss: Any


def _a29_npe_takes_no_run_seed(where: str, options: Mapping[str, Any]) -> None:
    """``kind: npe`` draws four times, so its seeds are per subsection.

    Module-level and taking plain data for plan §2.2's reason: the pre-flight
    pass calls this same object from the raw document, so the refusal before
    the beam and the one the executor raises cannot drift apart.  The four
    subsections' own seeds are :func:`_seeded`'s, through ``_seed_name``.
    """
    if "seed" in options:
        raise ConfigError(
            f"{where}: kind: npe needs FOUR seeds -- the bank draws theta "
            "from the priors, create initialises the network's weights, "
            "train shuffles the minibatches and sample draws -- so they are "
            "declared per subsection in inference.npe: as "
            "seed: {from: runtime.seeds.<name>}, not once on the run "
            "(check A29). A run carries one seed and this exit draws four "
            "times."
        )


@register("npe")
def _run_npe(run: Any, built: Any, *, results: Any = None) -> Any:
    """One ``kind: npe`` run -> an :class:`NpeProduct`."""
    from rheplicant.inference import train_posterior

    where = f"runs[{run.name!r}]"
    _a29_npe_takes_no_run_seed(where, run.options)
    # Everything else this exit could take lives in inference.npe:, so the
    # allowed set is empty -- the same shape `kind: forward` has.
    _sweep(run, frozenset())
    spec = _npe_spec(run, built)
    space, thetas, data = _simulate_bank(run, built, spec)
    trained, history = train_posterior(
        _estimator(run, built, spec, thetas, data), thetas, data,
        key=_draw_key(run, "inference.npe.train", built, spec.train),
        **_passthrough(spec.train, _TRAIN_OPTIONS),
    )
    # `sample(datum, key, n_samples)`: the data FIRST, which is the reverse of
    # `log_prob(theta, datum)` eighteen lines above it in inference/npe.py.
    # `sample` is the one entry point on that class with no keyword-only
    # marker, so both orders bind; the wrong one dies inside embed as
    # `TypeError: subtract does not accept dtypes key<fry>, float32`.
    flat = trained.sample(
        _observed(run, built),
        _draw_key(run, "inference.npe.sample", built, spec.sample),
        spec.sample["n_draws"],
    )
    return NpeProduct(
        # `space` is the one `_simulate_bank` checked and banked against, not
        # a second lookup: `flat`'s columns are laid out in ITS names order,
        # and checking one space while unravelling against another is a bug no
        # shape assertion can see.  NO TEST DEFENDS THAT CHOICE and none can
        # today -- measured, replacing this with `built.inference.space` left
        # every test in tests/config green, because on every document in the
        # suite the two ARE the same object (Task 7's first bank test asserts
        # exactly that identity).  It is discipline, not a guard, and it is
        # written down so the next reader does not mistake it for one.
        samples=_unravel(space, flat, where=where),
        # From the RETURNED stack, not from `spec.sample["n_draws"]`.  The two
        # are equal by construction on this route -- `sample` draws exactly
        # what it is asked for and nothing thins -- so the mutation between
        # them survives the whole suite (measured), and this is the spelling
        # that stays right the day the package grows a thinning knob.
        n_draw=int(flat.shape[0]),
        posterior=trained,
        # int(), because train_posterior's best_step is a traced ArrayImpl and
        # a product field that is sometimes an array is a field no caller can
        # format.
        best_step=int(history.best_step),
        train_loss=history.train,
        validation_loss=history.validation,
    )
