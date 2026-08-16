"""A35: every optional dependency, named by the feature that asks for it.

**What arrives today.**  Measured: ``resources.beams.<n>.format: healpix``
raises a bare ``ImportError`` from ``import healpy`` while its sibling
``format: gaussian`` -- three lines away, in the same module -- gets a
``ConfigError`` naming healpy and the extra it lives in.  ``horizon.mode:
truncate_map`` on a limTOD without ``horizon_truncated_beam`` raises after the
maps have been read, normalised and shape-checked.  ``observation.from_file``
without h5py arrives as a ``ConfigError`` **telling the wrong story**:
``files.py``'s blanket ``except Exception`` re-labels a missing distribution as
a file-parsing problem and then advises about delimiters and skiprows.
``model.noise_wave`` and ``flagging: MomentRFIFlaggingOperator`` raise at
forward-evaluation time, the second from inside a ``jax.pure_callback``.

Five gates already say the right sentence -- ``_require_pyuvdata``,
``_require_healpy``, ``_require_cal``, ``kinds/sky_models.py``'s gdsm gate and
``sections/parameters.py::_require_numpyro``.  They are the model for the
wording here, and they all keep working: this check runs **before** them and
says the same thing one phase earlier, so nothing is moved and nothing is
copied.  **A35 is an INVENTION, not a hoist** (§0.3 E.2): no message leaves
``kinds/`` or ``radio/`` and the one-binding walker gets no A35 row.

**This is the one check whose subject makes it likely to break §0's import
invariant.**  So:

* **it never imports an optional module**, at module scope or at call time;
* presence is :func:`importlib.util.find_spec` on a **top-level** name only.
  Measured for this plan: one ``find_spec("limTOD.uvbeam")`` in a registered
  check took the cold-budget guard from 26 ms to **1180.8 ms** against a 50 ms
  bound, because a dotted ``find_spec`` imports the parent package.  Nine
  top-level calls together cost **0.00019 s**, measured in this worktree.
  A submodule requirement is therefore probed by its top-level distribution
  only, **and the message says so**;
* the version leg reads ``sys.modules`` and never populates it (below).

**The version leg, and the false negative it carries -- stated rather than
hidden.**  ``import limtod_jax`` SUCCEEDS on limTOD 1.6 and the failure is a
missing attribute, so presence alone closes four routes and leaves the measured
one open.  ``radio/beams.py::_require_limtod_jax`` answers that with
``hasattr(module, feature)``, and this check mirrors it **only for a module the
process already holds**: asking ``hasattr`` of a module that is not imported
would mean importing it, which is exactly the thing this pass may not do.  A
fresh process has not imported ``limtod_jax`` when a document is loaded, so on
that path the attribute leg says nothing and the shipped gates
(``_require_limtod_jax``, ``radio/sky/driftscan.py::_limtod_jax``) remain the
backstop.  In a session that has already built one driftscan projector -- the
second ``load_document`` in a notebook, a campaign, a test process -- the leg
does fire, one phase before the build.  **That is the whole of what it claims.**

``hasattr`` is legal for ``limtod_jax`` and for nothing else here.  Measured on
a complete limTOD 1.10.0, ``hasattr(limTOD, "uvbeam")`` and ``hasattr(limTOD,
"cstbeam")`` are both **False** -- they are submodules, not attributes -- so a
``hasattr`` gate there would refuse every install in existence.  A version
comparison is not used either, and that is ``_require_cstbeam``'s own reasoning
rather than a new opinion: an editable install reports whatever its dist
metadata was written with, and this repository's venv sat at a recorded 1.8.0
while running 1.10.0 source.  (``importlib.metadata.version("limtod-jax")``
raises ``PackageNotFoundError`` in any case; the distribution is spelled
``limTOD``, which is why :data:`_FEATURES` carries the distribution and the
module separately.)

**What ``find_spec`` claims, and what it does not.**  It answers *findable*,
not *importable*.  A distribution that is installed but BROKEN -- a failing
loader, a missing shared object -- is found, so A35 stands down on it and the
shipped gate reports the real ``ImportError``.  That is deliberate: this pass
must not execute a module to find out, and a check that refused on "I could not
import it" would be running the import it exists to avoid.
``test_preflight_depends.py`` pins both directions.

**The routes it walks** (§0.3 E.2 adds four the task body omitted:
``format: cst``, ``engine: driftscan``, ``engine: general_pointing`` and
``inference.parameters.<n>.prior``).  Ten section-tokens, each one literal
word, each paired with the value the document wrote:

    format · horizon · engine · s_params · sky_model · file · node ·
    transform · run · prior

**``inference.twin.replace`` IS walked** (§0.3 E.10).  ``model.<n>.type`` is
one of the ten tokens and ``twin.py:69`` sends ``replace.<node>`` down the same
``build_node_operator`` path, so a ``replace: {noise_wave: {type:
NoiseWaveOperator}}`` earns the same finding with ``where =
inference.twin.replace.noise_wave.type``.  The message is BUILT from the token
rather than hoisted, so it names the section it actually found -- unlike A13's
shipped literal, which hardcodes ``f"model.{node_id}: "``.

**Every layer, not just the base** (§0.3 F.5(1)).  ``variants:`` can introduce
``format: uvbeam`` into a document whose base has no beam at all, so the walk
goes through ``preflight/document.py::_task3_over_layers`` and reads each
layer's ``resources:`` through ``config/resources.py::resolved_specs`` -- which
is TOTAL and drops a malformed entry rather than raising.  An entry that does
not resolve is one this check stands down on; ``build_resources`` says the
right sentence for it at the right phase.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.resources import resolved_specs

__all__ = ["Requirement"]

#: ``(distribution, top-level module, the attribute or None)``.
#:
#: The distribution and the module are separate members because they disagree
#: for the one that matters: ``limtod_jax`` ships in the **limTOD**
#: distribution, and ``importlib.metadata.version("limtod-jax")`` raises
#: ``PackageNotFoundError``.  The third member is ``None`` wherever the
#: requirement is the module itself; where it is a name inside the module it is
#: that name, and it is read with ``hasattr`` -- never with a dotted
#: ``find_spec``, which imports the parent.
Requirement = tuple[str, str, str | None]

#: ``(section-token, the value the document wrote)`` -> what that route needs.
#:
#: A bare token cannot carry this contract, which is why the key is a pair
#: (§0.3 E.2): ``format`` alone has **eight** values and **four** distinct
#: requirement sets, two routes need **two** distributions each
#: (``format: uvbeam`` -> pyuvdata + limTOD; ``kind: gdsm`` -> pygdsm + limTOD),
#: and one requirement depends on a SIBLING key rather than on the value at all
#: (:data:`_CONDITIONAL`).
#:
#: **The empty rows are load-bearing.**  ``format: npy``, ``kind: touchstone``,
#: ``engine: matrix`` and ``kind: npe`` need nothing optional, and writing that
#: down is what lets ``test_the_table_covers_every_value_the_source_declares``
#: compare this table against ``BEAM_FORMATS``, ``ENGINES``, ``S_PARAM_KINDS``
#: and ``SKY_KINDS`` -- so a ninth beam format cannot be added without a
#: decision being taken here.  A value absent from this table is a value A35
#: stands down on, which is also what happens to a typo: ``format: helpix`` is
#: ``build_beam``'s refusal to make, not this one's.
_FEATURES: dict[tuple[str, str], tuple[Requirement, ...]] = {
    # resources.beams.<name>.format -- BEAM_FORMATS, all eight
    ("format", "cst"): (("limTOD", "limTOD", None),),
    ("format", "uvbeam"): (("pyuvdata", "pyuvdata", None),
                           ("limTOD", "limTOD", None)),
    ("format", "healpix"): (("healpy", "healpy", None),),
    ("format", "gaussian"): (("healpy", "healpy", None),),
    ("format", "npy"): (),
    ("format", "npz"): (),
    ("format", "inline"): (),
    ("format", "python"): (),
    # resources.beams.<name>.horizon.mode -- build_beam's own three
    ("horizon", "truncate_map"): (("limTOD", "limtod_jax", "horizon_truncated_beam"),),
    ("horizon", "projector_mask"): (),
    ("horizon", "none"): (),
    # resources.projectors.<name>.engine -- ENGINES, all three.
    # NEITHER engine carries a healpy row, and that is a decision rather than
    # an omission.  `DriftScanProjector.from_beam_maps` analyses the beam with
    # `limtod_jax.map2alm_iter` and reaches healpy nowhere (measured: `grep -n
    # healpy radio/sky/driftscan.py` finds six docstring mentions and no call).
    # `general_pointing` DOES, through `kinds/projectors.py::_analyse` -- but
    # only when `beam_alms:` is absent, which is a requirement conditional on
    # a sibling being ABSENT and :data:`_CONDITIONAL` cannot say that.  A row
    # here would refuse a document that builds, so the stand-down is taken and
    # `_require_healpy` -- already a ConfigError naming healpy -- is the
    # backstop for it.
    ("engine", "driftscan"): (("limTOD", "limtod_jax", "driftscan"),),
    ("engine", "general_pointing"): (("limTOD", "limtod_jax", None),),
    ("engine", "matrix"): (),
    # resources.s_params.<name>.kind -- S_PARAM_KINDS, all three
    ("s_params", "touchstone"): (),
    ("s_params", "termination"): (("rhino-cal-jax", "rhino_cal_jax", None),),
    ("s_params", "cable"): (("rhino-cal-jax", "rhino_cal_jax", None),),
    # resources.sky_models.<name>.kind -- SKY_KINDS, all five
    ("sky_model", "uniform"): (),
    ("sky_model", "power_law"): (),
    ("sky_model", "maps"): (),
    ("sky_model", "gdsm"): (("pygdsm", "pygdsm", None),
                            ("limTOD", "limTOD", None)),
    ("sky_model", "python"): (),
    # observation.from_file, and every {file: {format: ...}} value node
    ("file", "rhino_hdf5"): (("h5py", "h5py", None),),
    # model.<node>.type, and inference.twin.replace.<node>.type
    ("node", "NoiseWaveOperator"): (("rhino-cal-jax", "rhino_cal_jax", None),),
    ("node", "MomentRFIFlaggingOperator"): (("MomentRFI", "MomentRFI", None),),
    # a transform block, wherever one is written
    ("transform", "beam_analysis"): (("limTOD", "limtod_jax", "map2alm_iter"),),
    # runs[].kind -- only the two this layer has a verdict about
    ("run", "nuts"): (("numpyro", "numpyro", None),),
    ("run", "npe"): (),
    # inference.parameters.<name>.prior
    ("prior", "declared"): (("numpyro", "numpyro", None),),
}

#: ``(section-token, value)`` -> ``(the sibling key, the requirement it adds)``.
#:
#: One route needs a requirement that its own value cannot decide:
#: ``DriftScanProjector.from_beam_maps`` calls ``_limtod_jax(uniform_sampling)``
#: (``radio/sky/driftscan.py:299``) and the ``uniform=True`` branch demands
#: ``limtod_jax.check_uniform_grid``, the FFT fast path added in limTOD 1.7.
#: A document without ``uniform_sampling:`` never reaches it, so folding this
#: into :data:`_FEATURES` would refuse installs that run the document fine.
_CONDITIONAL: dict[tuple[str, str], tuple[str, Requirement]] = {
    ("engine", "driftscan"): ("uniform_sampling",
                              ("limTOD", "limtod_jax", "check_uniform_grid")),
}

#: The section-token -> the phrase that names the route in the user's own
#: words, appended to the ENTRY's path exactly as ``_require_pyuvdata`` writes
#: ``f"{name}: format: uvbeam needs pyuvdata"``.  Ten rows, one per token, and
#: the test that pins them against :data:`_FEATURES` is what stops a token
#: being added with no sentence.
#:
#: ``node`` names the CLASS and no key, because the same class arrives under
#: two spellings -- ``type: NoiseWaveOperator`` and ``python:
#: 'rheplicant.radio:NoiseWaveOperator'`` -- and a message naming ``type:`` on
#: the second would quote a key its reader did not write.
_TRIGGER: dict[str, str] = {
    "format": "format: {value}",
    "horizon": "horizon.mode: {value}",
    "engine": "engine: {value}",
    "s_params": "kind: {value}",
    "sky_model": "kind: {value}",
    "file": "format: {value}",
    "node": "{value}",
    "transform": "transform: {value}",
    "run": "kind: {value}",
    "prior": "a declared prior",
}

#: distribution -> how a reader gets it.  **Every one of these is followable**
#: (R4): each names an extra this ``pyproject.toml`` actually declares, or says
#: plainly that the requirement is not resolvable from an index.  In
#: particular there is **no ``rheplicant[gdsm]``** -- the shipped gate advises
#: ``limTOD[gdsm]`` and so does this one.
_INSTALL: dict[str, str] = {
    "limTOD": ('limTOD is a hard dependency of this package rather than an extra, so a '
               'missing one means the install is broken or limTOD was removed: pip '
               'install "limTOD[jax]>=1.10".'),
    "healpy": ("healpy arrives with limTOD's own dependencies, so a missing one means "
               'the install is incomplete: pip install "limTOD[jax]>=1.10".'),
    "pygdsm": ("pygdsm is optional and arrives through limTOD's extra rather than "
               'through this package\'s: pip install "limTOD[gdsm]". There is no '
               "rheplicant[gdsm]."),
    "pyuvdata": ("pyuvdata is the 'uvbeam' extra; the limTOD bridge itself ships with "
                 'limTOD, so only the file reader is missing: uv pip install -e ".[uvbeam]".'),
    "h5py": ("h5py is the 'rhino' extra and it does resolve from an index: uv pip "
             'install -e ".[rhino]".'),
    "numpyro": ("numpyro is the 'numpyro' extra: pip install 'rheplicant[numpyro]'."),
    "rhino-cal-jax": ("rhino-cal-jax is the 'cal' extra and is not on PyPI, so the extra "
                      "names the requirement rather than resolving it, and the branch "
                      'matters: uv pip install "rhino-cal-jax @ '
                      'git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax".'),
    "MomentRFI": ("MomentRFI is the 'rfi' extra and is not on PyPI, so the extra names "
                  'the requirement rather than resolving it: uv pip install "MomentRFI @ '
                  'git+https://github.com/zzhang0123/MomentRFI".'),
}

#: A requirement -> the sentence that says what this check did NOT probe.
#: Ruling §0.3 E.2(1): a submodule requirement is probed by its top-level
#: distribution only, **and the message says so** -- so a reader who has limTOD
#: installed and still meets an ``ImportError`` from ``limTOD.cstbeam`` is not
#: left thinking this check had already cleared it.
_PROBE_NOTE: dict[Requirement, str] = {
    ("limTOD", "limTOD", None): (
        "Only the top-level limTOD is probed here: the submodules this layer reaches "
        "(limTOD.cstbeam, limTOD.uvbeam, limTOD.sky_model) are settled by their own "
        "gates when the resource is built, because probing one at this phase would "
        "import limTOD -- measured at 1180.8 ms against this pass's 50 ms budget."),
}

#: ``(section-token, value)`` -> the route that needs nothing optional, where
#: one exists.  Quoted from the package's own gates rather than invented, so
#: applying the advice reaches a document this layer accepts (R4).  A route
#: with no honest alternative gets no sentence, which is the other half of the
#: same rule.
_ALTERNATIVE: dict[tuple[str, str], str] = {
    ("node", "MomentRFIFlaggingOperator"):
        "The threshold-based FlaggingOperator needs none of it.",
    ("engine", "driftscan"):
        "engine: matrix takes a precomputed sky->TOD matrix and needs no optional "
        "dependency (fixed pointing and beam only).",
    ("engine", "general_pointing"):
        "engine: matrix takes a precomputed sky->TOD matrix and needs no optional "
        "dependency (fixed pointing and beam only).",
}

#: The tail every A35 finding carries: why the sentence is said HERE.
_TAIL = ("Said from the document's text, so that a missing dependency arrives before the "
         "run rather than as an ImportError in the middle of one (check A35).")


def _requirements(token: str, value: str,
                  siblings: Any) -> tuple[Requirement, ...]:
    """Everything ``(token, value)`` needs, the sibling-conditional one included."""
    required = _FEATURES.get((token, value))
    if required is None:
        return ()
    conditional = _CONDITIONAL.get((token, value))
    if conditional is not None:
        sibling, extra = conditional
        if isinstance(siblings, Mapping) and siblings.get(sibling):
            return (*required, extra)
    return required


def _verdict(requirement: Requirement) -> str | None:
    """``"absent"``, ``"outdated"`` or ``None`` -- **importing nothing, ever**.

    ``sys.modules`` first, for three reasons rather than for speed.  A module
    the process already holds answers the attribute question, which is the only
    way the version leg can be asked without an import.  A ``sys.modules``
    entry that is ``None`` is CPython's own way of blocking an import, so it is
    what a caller (and a test) uses to say "absent" -- and ``find_spec`` agrees,
    returning ``None`` for it.  And a module with no ``__spec__`` makes
    ``find_spec`` raise ``ValueError``, which inside the pass would become
    *"pre-flight check 'A35' RAISED"* and hide every later finding; reading
    ``sys.modules`` first means a live module never reaches that path.

    **``except (ImportError, ValueError)`` is a DOCUMENTED EQUIVALENT MUTANT**
    -- recorded here rather than defended by a test, which is §0.3 F.5(10)'s
    own resolution for this shape.  Narrowing it to ``except ImportError``
    changes no answer and no test goes red, and that is correct: for a
    TOP-LEVEL name ``find_spec`` raises ``ValueError`` only when the name is in
    ``sys.modules`` with ``__spec__ is None`` -- a state the branch above has
    already consumed -- and ``ImportError`` only through a missing parent
    package, which a top-level name does not have.  Both clauses are
    unreachable defensive code kept because the table's dot-free invariant is
    what makes them so, and that invariant is a test
    (``test_no_requirement_probes_a_dotted_module``) rather than a property of
    this function.
    """
    _, module, attribute = requirement
    if module in sys.modules:
        held = sys.modules[module]
        if held is None:
            return "absent"
        if attribute is not None and not hasattr(held, attribute):
            return "outdated"
        return None
    try:
        found = importlib.util.find_spec(module)
    except (ImportError, ValueError):  # a missing parent, or a __spec__ of None
        return "absent"
    return None if found is not None else "absent"


def _message(where: str, token: str, value: str, requirement: Requirement,
             verdict: str) -> str:
    """The whole sentence, built from the token rather than hoisted."""
    distribution, module, attribute = requirement
    trigger = _TRIGGER[token].format(value=value)
    if verdict == "absent":
        head = (f"{where}: {trigger} needs the {distribution} distribution, and {module} "
                f"is not importable in this environment.")
    else:
        head = (f"{where}: {trigger} needs {module}.{attribute}, and the {module} this "
                f"process holds does not carry it -- the module imports and the symbol is "
                f"missing, which is what a {distribution} older than this route looks like.")
    clauses = [head, _INSTALL[distribution]]
    note = _PROBE_NOTE.get(requirement)
    if note is not None:
        clauses.append(note)
    alternative = _ALTERNATIVE.get((token, value))
    if alternative is not None:
        clauses.append(alternative)
    clauses.append(_TAIL)
    return " ".join(clauses)


def _file_nodes(value: Any, path: str) -> Iterable[tuple[str, Mapping]]:
    """``(path, body)`` for every ``{file: {...}}`` value node under ``value``.

    The walk is what closes A35's own twin: ``observation.from_file`` is not
    the only h5py route -- ``parse_from_file`` itself resolves
    ``{"file": dict(spec)}`` (``sections/ingest.py:113``), and any value node
    anywhere may write ``{file: {format: rhino_hdf5}}`` and reach the same
    reader.  A walk that read ``observation.from_file`` alone would guard one
    of them.

    **An explicit stack, concrete type tests, and a path built only when
    something is found -- all three are measurements rather than preferences.**
    This runs once per LAYER, and 3A's cold-budget guard drives a document with
    forty runs and twenty variants against a 50 ms bound, so a walk that costs
    a millisecond per layer is a fifth of that budget spent on nothing.
    Measured on that document, over its twenty-one layers: the obvious
    spelling -- ``yield from`` recursion, ``isinstance(x, Mapping)`` on every
    node, an f-string path for every node visited -- cost **2.0 ms** a pass;
    this one costs **1.1 ms** (52 us a layer), and the two were compared
    element for element while the change was made.

    ``dict`` and ``list`` before ``Mapping``: the abstract test goes through
    ``abc.__instancecheck__``, which was 261 300 calls of one profile.  A
    document is dicts and lists -- ``yaml.safe_load``'s output, the test
    builders' literals, and ``resolved_specs``' own shallow copies -- so the
    abstract test is kept only for the node this is ENTERED on, where a caller
    could reasonably hand over something else.
    """
    if not isinstance(value, (dict, list)):
        if not isinstance(value, Mapping):
            return
        value = dict(value)
    # (node, its key, its parent's link) -- a linked list, walked backwards
    # only when something is found, so a document with no `file:` node in it
    # builds no path strings at all.
    stack: list[tuple[Any, Any]] = [(value, None)]
    while stack:
        current, link = stack.pop()
        if isinstance(current, dict):
            body = current.get("file")
            if isinstance(body, dict):
                yield (_joined(path, link), body)
            for key, item in current.items():
                if isinstance(item, (dict, list)):
                    stack.append((item, (key, link)))
        else:
            for index, item in enumerate(current):
                if isinstance(item, (dict, list)):
                    stack.append((item, (f"[{index}]", link)))


def _joined(root: str, link: Any) -> str:
    """The dotted path a :func:`_file_nodes` link stands for."""
    pieces: list[str] = []
    while link is not None:
        key, link = link
        pieces.append(str(key))
    pieces.reverse()
    out = root
    for piece in pieces:
        out = f"{out}{piece}" if piece.startswith("[") else f"{out}.{piece}"
    return out


def _typed_entries(layer: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """``(where, spec)`` for every entry that reaches ``build_node_operator``.

    Both halves of §0.3 E.10's ruling: ``model:``'s four routes through
    ``preflight/model.py::_t4_entries`` (single, ``compose: stages``, a ``many``
    node's list and its FAN entries), **and** ``inference.twin.replace``, which
    ``_nodes()`` cannot see and which reaches the same builder.
    """
    from rheplicant.config.preflight.model import _nodes, _t4_entries, _t4_graph

    graph = _t4_graph()
    for node_id, spec in _nodes(layer).items():
        node = graph.nodes.get(node_id)
        many = bool(node.many) if node is not None else False
        for where, entry in _t4_entries(node_id, spec, many=many):
            yield (f"model.{where}", entry)
    inference = layer.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        for node_id, entry in replace.items():
            yield (f"inference.twin.replace.{node_id}", entry)


def _routes(layer: Mapping[str, Any]) -> Iterable[tuple[str, str, Any, Any]]:
    """``(where, token, value, siblings)`` for every A35 route in ONE layer.

    ``siblings`` is the mapping the value was read out of, which is what
    :data:`_CONDITIONAL` asks about.
    """
    from rheplicant.config.preflight.model import _t5_radio_class

    for key, spec in resolved_specs(layer.get("resources")).items():
        kind = key.split(".", 2)[1]
        if kind == "beams":
            yield (key, "format", spec.get("format"), spec)
            horizon = spec.get("horizon")
            if isinstance(horizon, Mapping):
                yield (key, "horizon", horizon.get("mode"), horizon)
        elif kind == "projectors":
            yield (key, "engine", spec.get("engine"), spec)
        elif kind == "s_params":
            yield (key, "s_params", spec.get("kind"), spec)
        elif kind == "sky_models":
            yield (key, "sky_model", spec.get("kind"), spec)

    observation = layer.get("observation")
    if isinstance(observation, Mapping):
        from_file = observation.get("from_file")
        if isinstance(from_file, Mapping):
            yield ("observation.from_file", "file", from_file.get("format"), from_file)
    for section, block in layer.items():
        # `variants:` is skipped because `_task3_over_layers` has already
        # applied it: walking it here would read every variant once per layer.
        if section == "variants":
            continue
        for path, body in _file_nodes(block, str(section)):
            yield (path, "file", body.get("format"), body)

    for where, entry in _typed_entries(layer):
        if not isinstance(entry, Mapping):
            continue
        named = entry.get("type")
        if isinstance(named, str):
            yield (where, "node", named, entry)
            continue
        shipped = _t5_radio_class(entry)
        if shipped is not None:
            yield (where, "node", shipped.__name__, entry)

    inference = layer.get("inference")
    if isinstance(inference, Mapping):
        parameters = inference.get("parameters")
        if isinstance(parameters, Mapping):
            for name, spec in parameters.items():
                if not isinstance(spec, Mapping):
                    continue
                if "prior" in spec:
                    yield (f"inference.parameters.{name}", "prior", "declared", spec)
                yield from _transforms(f"inference.parameters.{name}", spec)
        bindings = inference.get("bindings")
        if isinstance(bindings, (list, tuple)):
            for index, entry in enumerate(bindings):
                if isinstance(entry, Mapping):
                    yield from _transforms(f"inference.bindings[{index}]", entry)

    runs = layer.get("runs")
    if isinstance(runs, Mapping):
        runs = [runs]
    if isinstance(runs, (list, tuple)):
        for index, entry in enumerate(runs):
            if isinstance(entry, Mapping):
                yield (f"runs[{index}]", "run", entry.get("kind"), entry)


def _transforms(where: str, spec: Mapping) -> Iterable[tuple[str, str, Any, Any]]:
    """The ``transform:`` block's own words -- both places one may be written.

    ``parse_transform`` is called from two sites (``sections/transforms.py:354``
    for a latent's own ``transform:`` and ``:393`` for a binding entry's), and
    ``{beam_analysis: ...}`` is the only spelling of it that imports anything.
    """
    block = spec.get("transform")
    if isinstance(block, Mapping):
        for word in block:
            yield (where, "transform", word, block)


def _in_layer(layer: Mapping[str, Any],
              seen: dict[Requirement, str | None]) -> Iterable[Finding]:
    """A35 over one layer of the document.

    ``seen`` is one pass's answers, and it is a correctness-preserving cache
    rather than an optimisation with a caveat: the environment cannot change
    between two layers of one document, so asking ``find_spec`` again can only
    return what it returned a moment ago.  It is created fresh per call, never
    at module scope, so a test that blocks a module between two passes gets two
    different answers -- which is what
    ``test_the_same_route_is_silent_when_the_distribution_is_there`` rests on.

    It matters: measured on 3A's cold-budget document -- forty runs, twenty
    variants -- one ``find_spec("numpyro")`` costs nine ``stat`` calls and the
    uncached walk paid it twenty-one times, once per layer.

    **One sentence per place, however many requirements reach it.**  A
    driftscan projector with ``uniform_sampling: true`` needs ``limtod_jax``
    twice -- once for the engine and once for the FFT fast path
    (:data:`_CONDITIONAL`) -- and when the module is absent outright those two
    requirements produce the *same* sentence, because an absent module has no
    attribute to name.  Emitting it twice would say nothing new and would break
    ``preflight_helpers.only``, which is the accessor every test about this
    check uses.  The de-duplication is on ``(where, message)`` rather than on
    the module, so a projector missing limtod_jax AND a second distribution
    still hears about both.
    """
    from rheplicant.config.preflight.document import _task3_where

    said: set[tuple[str, str]] = set()
    for where, token, value, siblings in _routes(layer):
        if not isinstance(value, str):
            continue
        for requirement in _requirements(token, value, siblings):
            if requirement in seen:
                verdict = seen[requirement]
            else:
                verdict = seen[requirement] = _verdict(requirement)
            if verdict is None:
                continue
            message = _message(where, token, value, requirement, verdict)
            if (where, message) in said:
                continue
            said.add((where, message))
            yield refuse("A35", _task3_where(where), message)


@register("A35")
def _extras(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A35: the optional distributions this document's own tokens ask for.

    **On run order.**  §0.3 C.5 records a decision that A35 should lead the
    pass -- ``CHECKS`` insertion order is run order, ``raise_if_refused`` shows
    the first refusal verbatim, and a missing distribution is the one fault a
    reader cannot work around by fixing the other one -- and it attributes that
    to the alphabetical foot block, ``depends`` sorting ahead of ``document``.
    **The attribution is wrong, measured.**  What decides the order is which
    module is imported first, and a foot-imported module that HEAD-imports a
    sibling registers that sibling's checks ahead of its own; a sibling task
    measured a module sorting first in the foot block whose own check landed at
    index 29 of 40.

    Measured in a fresh process at this commit, ``list(CHECKS)[:4]`` is
    ``['A35', 'A1.runs', 'A1.horizon', 'A1.variants']`` -- and A35 leads
    because every helper this module borrows from ``preflight/document.py`` and
    ``preflight/model.py`` is imported inside a function.  That property is
    what ``test_this_module_head_imports_no_sibling_under_preflight`` asserts;
    the position itself is not asserted anywhere, because a later module
    sorting before ``depends`` would move it.

    **Cost.**  This calls ``_task3_over_layers`` exactly once per pass.  On
    3A's cold-cost document (forty runs, twenty variants) that walk used to
    cost **3.65 ms** of A35's **5.1 ms**, because ``_task3_layers`` built every
    layer eagerly through ``apply_variant``, which deep-copies the document --
    once per declared variant for THIS caller and again for each of the others.

    **Both halves of that sentence are now out of date, and the correction is
    the point.**  ``_task3_layers`` IS memoised as of the wave-boundary fix:
    one document's layers are built once per pass and handed to every caller,
    of which there were ELEVEN by the end of wave 1, not four -- ten
    ``_task3_over_layers`` call sites plus ``_variant_text``'s own merge.
    Measured on the cold guard's own document, where ten of the eleven run
    (``noise``'s is gated off there): 210 ``apply_variant`` calls before and **21**
    after -- one per declared variant, for the whole pass.  What this walk still
    pays is its own per-layer read; the deep merge is no longer its to pay, and
    ``test_preflight_depends_cost.py`` times the read rather than the merge for
    exactly that reason.
    """
    from rheplicant.config.preflight.document import _task3_over_layers

    seen: dict[Requirement, str | None] = {}
    return _task3_over_layers(document, lambda layer: _in_layer(layer, seen))
