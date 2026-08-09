"""The path grammar: a dotted string, the selector it compiles to, and refusal.

``Bind.into`` holds **callables**, not strings (``inference/parameters.py:338``),
and ``ParameterSpace._resolve_targets`` *invokes* them against a copy of the
twin whose every leaf has been replaced by its own key path. So a path here
compiles to a callable that walks the object's own accessors. Synthesising key
paths directly would not survive ``Pipeline.__getitem__``: ``p["gain"]``
resolves through ``self.names.index("gain")`` into a positional stage, and the
string ``"gain"`` never appears in the path that comes back
(``.stages[0].gain`` does).

That mismatch is also why every refusal below names **both** spellings. The
package's own messages quote ``keystr(path)`` -- ``.stages[0].gain`` -- and a
reader who wrote ``gain.gain`` has never seen that string. Naming only the
structural path would be reusing wording at the cost of the reader.

Resolution happens eagerly, against a tagged twin, before the forward
function is built and anything is traced. That twin must already exist by
this point, built from already-constructed resources -- so the claim is not
"before any file is read"; it is "before assembly proceeds to building and
tracing the model", which is where schema §6's "before any expensive work"
promise is aimed. The alternative is the same refusal arriving from
``ParameterSpace`` once that build is already underway.
"""

import re
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import equinox as eqx
import jax

from rheplicant.config.errors import ConfigError

_STEP = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z_0-9]*)?(?:\[(?P<index>0|[1-9][0-9]*)\])?$")


def parse_path(path: str) -> tuple[str | int, ...]:
    """Split a path into its head and steps.

    ``head ( "." step )*`` where a step is an identifier, an identifier with a
    non-negative integer subscript, or a bare subscript. Returns a tuple whose
    entries are ``str`` for an attribute or key and ``int`` for an index.
    """
    if not isinstance(path, str):
        raise ConfigError(
            f"A path must be a string; got {type(path).__name__} ({path!r}). A YAML "
            "'into:' entry with no value after the colon parses to None, not to an "
            "empty path -- write the dotted path as a quoted string."
        )
    text = path
    if not text or text != text.strip():
        raise ConfigError(
            f"Path {path!r} is empty or padded with whitespace. A path is "
            "'head' or 'head.step.step', where head names a graph node and each step "
            "is an attribute, optionally with a non-negative index."
        )
    parts: list[Any] = []
    for piece in text.split("."):
        match = _STEP.match(piece)
        if match is None or (match.group("name") is None and match.group("index") is None):
            raise ConfigError(
                f"Path {path!r} has an unusable segment {piece!r}. Each segment is an "
                "identifier ('t_unc'), an identifier with a non-negative index "
                "('stages[0]'), or a bare index ('[1]'). There is no slicing, no "
                "negative index, no leading zero and no wildcard: a path names one "
                "leaf, and anything that could name several would silently bind the "
                "first."
            )
        if match.group("name"):
            parts.append(match.group("name"))
        if match.group("index") is not None:
            parts.append(int(match.group("index")))
    return tuple(parts)


def compile_path(path: str) -> Callable[[Any], Any]:
    """Compile a path into the selector ``Bind(into=...)`` takes.

    The returned callable walks with ``__getitem__`` for the head (which is how
    a node id is addressed) and ``getattr`` / ``__getitem__`` for the steps.
    """
    parts = parse_path(path)
    head, steps = parts[0], parts[1:]

    def selector(twin: Any) -> Any:
        current = twin[head]
        for step in steps:
            current = current[step] if isinstance(step, int) else getattr(current, step)
        return current

    selector.__doc__ = f"selector for config path {path!r}"
    selector.declared_path = path
    return selector


class ResolvedPath(NamedTuple):
    """A path that reached a real array leaf.

    Attributes:
        declared: the string the document wrote.
        key_path: the JAX key path, as ``_resolve_targets`` produces it.
        keystr: that path in ``jax.tree_util.keystr`` form, which is what the
            package's own refusals quote.
        leaf: the current value of the leaf.
        selector: the callable, ready for ``Bind(into=...)``.
    """

    declared: str
    key_path: tuple
    keystr: str
    leaf: Any
    selector: Callable[[Any], Any]


_KEY_ENTRY = (
    jax.tree_util.SequenceKey,
    jax.tree_util.GetAttrKey,
    jax.tree_util.DictKey,
    jax.tree_util.FlattenedIndexKey,
)


def _is_tagged_subtree(value: Any) -> bool:
    """True if ``value`` is still a live piece of the TAGGED twin -- an
    operator or other pytree container whose descendants are all path tags
    left behind by ``tree_map_with_path`` -- rather than a static field's own
    raw value.

    A static field is invisible to ``tree_map_with_path`` (it is treedef, not
    a traced child), so whatever it holds passes through completely
    untouched: ``None``, ``()``, a multi-element tuple, a bare scalar, or a
    container of its own -- none of which contain path-tag objects, so this
    correctly returns False for all of them regardless of whether the raw
    value happens to look "leaf-like" or "container-like" on its own terms.
    An earlier check asked "does this look like a single leaf", which got
    this backwards: ``None`` and ``()`` flatten to *zero* leaves, which
    failed a single-leaf test even though there is nothing further to walk
    into -- misrouting every static field shaped like one of those to the
    "stops on an operator, go one step deeper" message instead of "static
    configuration". Demonstrated on shipped
    ``MomentRFIFlaggingOperator.kernel_shapes`` (``static=True,
    default=()``) and several ``DriftScanProjector`` fields (``static=True,
    default=None``).
    """
    leaves = jax.tree_util.tree_leaves(value)
    return bool(leaves) and all(isinstance(x, _KEY_ENTRY) for x in leaves)


def _describe(value: Any, limit: int = 120) -> str:
    """The type of ``value``, plus its repr when the repr is short AND
    single-line enough to help rather than flood the message.

    An array's repr can run to thousands of characters, and an eqx.Module's
    pretty-printed repr can run to several lines even when short in total
    length; a scalar's or an empty container's repr does neither. Below the
    character limit and on one line, both are shown; otherwise only the type
    name is, so the refusal stays readable instead of dumping a multi-line
    object into one sentence.
    """
    text = repr(value)
    name = type(value).__name__
    if "\n" in text or len(text) > limit:
        return name
    return f"{name} ({text})"


def resolve_path_on(path: str, twin: Any) -> ResolvedPath:
    """Resolve ``path`` against ``twin`` and refuse if it reaches no array leaf.

    Raises:
        ConfigError: on any of three conditions found here -- independent of
            the schema's own refusal numbering, which none of the three maps
            to one-for-one. (1) The walk itself raises: a bad head, an
            ambiguous ``many`` node (``AmbiguousNodeError``), or an unknown
            attribute. (2) The walk lands short of a leaf: either on an
            operator or other pytree container with fields still below it, or
            on a genuine static field. (3) The walk reaches a real pytree
            leaf that is not an array. The whole-document checks -- two paths
            sharing a leaf, a path into an aliased node, and a region's
            config key not equal to its last covered node -- are
            ``refuse_duplicate_targets``, ``refuse_aliased_target`` and
            ``refuse_misaddressed_region`` below; a fifth whole-document
            check (a ``twin.replace`` target colliding with a binding's) is
            Plan 2's, once ``inference.parameters``' replace targets exist
            for it to fold in.
    """
    tagged = jax.tree_util.tree_map_with_path(lambda key, _: key, twin)
    leaves = dict(jax.tree_util.tree_flatten_with_path(twin)[0])
    selector = compile_path(path)
    try:
        key_path = selector(tagged)
    except Exception as exc:
        raise ConfigError(
            f"Path {path!r} could not be walked against this twin: {exc}. A path walks "
            "attributes and indices only. Check the head against the graph's own node "
            "ids -- a region is addressed by its LAST covered node, and a nested "
            "Assembly is opaque to the search."
        ) from exc

    found = False
    if isinstance(key_path, tuple):
        try:
            found = key_path in leaves
        except TypeError:
            # A static field whose value happens to be a tuple holding
            # something unhashable (e.g. a list). It is still not a usable
            # leaf key, so it falls through to the same refusal below.
            found = False

    if not found:
        try:
            # Describe the value as it really is on the untagged twin -- the
            # tagged one holds key-path objects at this position, not
            # anything a reader typed a config document to produce.
            what_found = _describe(selector(twin))
        except Exception:
            # The walk succeeded against the tagged twin (every leaf is
            # present there, even a static one) but could, in principle,
            # behave differently against the real twin -- fall back to
            # naming the tagged value's own type rather than crashing here.
            what_found = type(key_path).__name__

        if _is_tagged_subtree(key_path):
            raise ConfigError(
                f"Path {path!r} stops on {what_found}, which is not a leaf: it "
                "is an operator (or other pytree container) with fields still below it "
                "on the walk. A path must name exactly one leaf -- go one step deeper, "
                f"e.g. {path!r} plus '.<field>', naming the field on it you mean."
            )
        raise ConfigError(
            f"Path {path!r} does not reach an array leaf of the twin. It landed on "
            f"static configuration -- {what_found} -- which inference cannot "
            "touch: a static field lives in the treedef, not among the traced leaves, "
            "so it is not merely unreachable by this path, it is unreachable by any "
            "path. Measured: on a path-tagged twin a static field still holds its own "
            "value rather than a path, because tree_map_with_path never visited it. "
            "Bind a traced field, or set this one as configuration in model:."
        )

    leaf = leaves[key_path]
    if not eqx.is_array(leaf):
        keystr = jax.tree_util.keystr(key_path)
        raise ConfigError(
            f"Path {path!r} as written -- {keystr!r} as the twin sees it -- reaches "
            f"{_describe(leaf)}, not an array leaf. It IS a genuine pytree leaf, unlike "
            "a static field, but inference can only bind arrays: there is no gradient "
            "to take through a plain Python value. Give the field a jnp.ndarray type, "
            "or mark it eqx.field(static=True) if it really is configuration rather "
            "than something to infer."
        )
    return ResolvedPath(path, key_path, jax.tree_util.keystr(key_path), leaf, selector)


def refuse_duplicate_targets(paths: Iterable[str], twin: Any) -> None:
    """Refusal 4: two declared paths that reach one leaf.

    Both spellings are named. ``keystr`` alone would tell the reader that
    ``.stages[0].gain`` is written twice without saying which two keys of
    their document did it.
    """
    seen: dict[tuple, str] = {}
    for path in paths:
        resolved = resolve_path_on(path, twin)
        earlier = seen.get(resolved.key_path)
        if earlier is not None:
            raise ConfigError(
                f"Two declared paths reach one leaf: {earlier!r} and {path!r} as "
                f"written, both {resolved.keystr!r} as the twin sees it. One would "
                "silently win, and which one is an implementation detail of the order "
                "the document happened to be read in. Give each leaf a single "
                "binding; if two quantities really are the same number, declare one "
                "latent and give its Bind both targets with fan: broadcast."
            )
        seen[resolved.key_path] = path


def refuse_aliased_target(path: str, twin: Any) -> None:
    """Refusal 3: the path's head names a node folded in at more than one place.

    ``Assembly.aliased`` is empty for every shipped graph and is the documented
    hazard for user-defined ones. Binding would rewrite the one branch this
    path reaches and leave the others in the forward model, which would then
    answer as if the latent were frozen everywhere but that branch -- finite,
    correctly shaped, wrong. The package's own guard
    (``inference/parameters.py:704-738``) is path-based rather than
    spelling-based, so naming the second copy by index is refused identically;
    this one is a pre-flight on the head and does not replace it.
    """
    aliased = tuple(getattr(twin, "aliased", ()) or ())
    if not aliased:
        return None
    head = parse_path(path)[0]
    if head in aliased:
        raise ConfigError(
            f"Path {path!r} writes into node {head!r}, which is folded into this "
            "assembly at more than one place: its contribution reaches the sink by "
            "several paths, so the operator sits in several branches. Binding would "
            "rewrite the one branch this path reaches and silently leave the others in "
            "the forward model, which would then answer as if the parameter were "
            "frozen everywhere but that branch -- finite, correctly shaped, wrong. "
            "Bind at a node downstream of the fork instead, or restructure the graph "
            f"so that {head!r} reaches the sink once."
        )
    return None


def refuse_misaddressed_region(config_key: str, region: Iterable[str]) -> None:
    """Refusal 6 / check A47: a region's config key must equal ``at[-1]``.

    ``At`` with a tuple of node ids covers a contiguous region, and the fold
    labels the covering operator with the LAST node id
    (``core/graph.py:131-134``, implemented at ``core/fold.py:409-414`` and
    ``core/graph.py:1071``). A config key naming any other covered node
    resolves to nothing, and the failure is a bare ``KeyError`` rather than
    the refusal the schema promised.
    """
    nodes = tuple(region)
    if len(nodes) < 2:
        return None
    if config_key != nodes[-1]:
        raise ConfigError(
            f"A multi-node at: region covering {list(nodes)} is written under the key "
            f"{config_key!r}, but a region is addressed in the assembly by its LAST "
            f"covered node id -- here {nodes[-1]!r}. assembly[{config_key!r}] would "
            "raise KeyError, and any into: path with that head would fail as a name "
            "error rather than as the 'landed on static configuration' refusal. Name "
            f"the entry {nodes[-1]!r}. (A region is *entered* at its first node, which "
            "is where slot kinds are screened -- that is a different node and a "
            "different check.)"
        )
    return None
