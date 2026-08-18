"""JAX-free ownership of process-global runtime establishment and facts."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    freeze_evidence,
    static_isinstance,
    static_type_name,
)
from _rheplicant_bootstrap.process import RuntimeSpec
from _rheplicant_bootstrap.types import JsonValue

RUNTIME_ROW_KEYS = ("requested", "actual", "prior_environment")
RUNTIME_REQUESTED_ROW_KEYS = ("jax_enable_x64", "platform")
RUNTIME_ACTUAL_ROW_KEYS = ("jax_enable_x64", "backend")
RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS = ("jax_enable_x64", "jax_platforms")
_FOREIGN_EXCEPTION_DETAIL_LIMIT = 1024


def _validate_utf8_text(value: str, *, where: str) -> None:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain only valid UTF-8 text.") from None


def _is_utf8_text(value: str) -> bool:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class PriorEnvironment:
    jax_enable_x64: str | None
    jax_platforms: str | None

    def __post_init__(self) -> None:
        for field_name in ("jax_enable_x64", "jax_platforms"):
            value = object.__getattribute__(self, field_name)
            if value is None:
                continue
            if not static_isinstance(value, str):
                raise ConfigError(
                    f"runtime prior_environment.{field_name} must be a "
                    "string or null."
                )
            canonical = str.__str__(value)
            _validate_utf8_text(
                canonical,
                where=f"runtime prior_environment.{field_name}",
            )
            object.__setattr__(self, field_name, canonical)
        _validate_prior_environment(self, where="runtime")


@dataclass(frozen=True, slots=True)
class RuntimeActual:
    jax_enable_x64: bool
    backend: str

    def __post_init__(self) -> None:
        if not static_isinstance(self.jax_enable_x64, bool):
            raise ConfigError("runtime actual jax_enable_x64 must be a bool.")
        if not static_isinstance(self.backend, str) or not str.__str__(self.backend):
            raise ConfigError("runtime actual backend must be a non-empty string.")
        backend = str.__str__(self.backend)
        _validate_utf8_text(backend, where="runtime actual backend")
        object.__setattr__(self, "backend", backend)
        _validate_actual(self, where="runtime")


@dataclass(slots=True)
class RuntimeSession:
    requested: RuntimeSpec
    prior_environment: PriorEnvironment
    actual: RuntimeActual | None = None

    def __post_init__(self) -> None:
        try:
            raw_requested = object.__getattribute__(self, "requested")
            raw_prior = object.__getattribute__(self, "prior_environment")
            raw_actual = object.__getattribute__(self, "actual")
        except Exception:
            raise ConfigError("runtime session is malformed.") from None
        requested = _copy_requested(raw_requested, where="runtime")
        prior = _copy_prior_environment(raw_prior, where="runtime")
        actual = (
            None
            if raw_actual is None
            else _copy_actual(raw_actual, where="runtime")
        )
        self.requested = requested
        self.prior_environment = prior
        self.actual = actual

    def verify(self, *, boundary: str) -> RuntimeActual:
        """Inspect JAX and refuse any drift from the requested global state."""
        try:
            raw_requested = object.__getattribute__(self, "requested")
            raw_prior = object.__getattribute__(self, "prior_environment")
            raw_actual = object.__getattribute__(self, "actual")
        except Exception:
            raise ConfigError("runtime session is malformed.") from None
        requested = _copy_requested(raw_requested, where="runtime")
        prior = _copy_prior_environment(raw_prior, where="runtime")
        previous_actual = (
            None
            if raw_actual is None
            else _copy_actual(raw_actual, where="runtime")
        )
        if not static_isinstance(boundary, str) or not str.__str__(boundary):
            raise ConfigError("runtime: verification boundary must be non-empty.")
        exact_boundary = str.__str__(boundary)
        _validate_utf8_text(
            exact_boundary,
            where="runtime verification boundary",
        )
        try:
            try:
                from jax import config, default_backend

                raw_x64 = config.read("jax_enable_x64")
                if not static_isinstance(raw_x64, bool):
                    raise TypeError(
                        "jax_enable_x64 inspection did not return a bool"
                    )
                raw_backend = default_backend()
                if not static_isinstance(raw_backend, str) or not str.__str__(raw_backend):
                    raise TypeError("backend inspection did not return a name")
                actual = RuntimeActual(
                    jax_enable_x64=bool(raw_x64),
                    backend=str.__str__(raw_backend),
                )
            except Exception as exc:
                raise ConfigError(
                    f"runtime: {exact_boundary} could not select/inspect the requested "
                    f"JAX backend ({_safe_exception_text(exc)})."
                ) from None
        finally:
            self.requested = requested
            self.prior_environment = prior
            self.actual = previous_actual
        if actual.jax_enable_x64 is not requested.jax_enable_x64:
            if (
                exact_boundary == "current process"
                and requested.jax_enable_x64 is True
                and actual.jax_enable_x64 is False
            ):
                raise ConfigError(
                    "runtime declares jax_enable_x64: true but this process is "
                    "running float32. The flag is process-global and must be set "
                    "before any array exists: "
                    "jax.config.update('jax_enable_x64', True) at the top of your "
                    "session (Plan 4's CLI applies it from the document "
                    "automatically). Refused here rather than at the first traced "
                    "delivery, where the message could not name the section."
                )
            raise ConfigError(
                "runtime.jax_enable_x64: requested "
                f"{requested.jax_enable_x64!r}, but {exact_boundary} has "
                f"{actual.jax_enable_x64!r}."
            )
        if (
            requested.platform != "auto"
            and actual.backend != requested.platform
        ):
            raise ConfigError(
                f"runtime.platform: requested {requested.platform!r}, "
                f"but {exact_boundary} selected {actual.backend!r}."
            )
        self.actual = actual
        return actual


def establish_runtime(
    requested: RuntimeSpec,
    *,
    import_main: Callable[[], object],
) -> tuple[RuntimeSession, object]:
    """Establish environment state before the sole main-package import."""
    requested_snapshot = _copy_requested(requested, where="runtime")
    prior = PriorEnvironment(
        jax_enable_x64=os.environ.get("JAX_ENABLE_X64"),
        jax_platforms=os.environ.get("JAX_PLATFORMS"),
    )
    session = RuntimeSession(requested_snapshot, prior)
    stable_requested = session.requested
    if "jax" in sys.modules:
        session.verify(boundary="existing process")
    else:
        os.environ["JAX_ENABLE_X64"] = (
            "true" if stable_requested.jax_enable_x64 else "false"
        )
        if stable_requested.platform != "auto":
            os.environ["JAX_PLATFORMS"] = stable_requested.platform
    main = import_main()
    session.verify(boundary="main import")
    return session, main


def runtime_audit_row(session: RuntimeSession) -> Mapping[str, JsonValue]:
    """Validate and close the only JSON projection of runtime facts."""
    if type(session) is not RuntimeSession:
        raise ConfigError("runtime audit row requires a RuntimeSession.")
    _validate_requested(session.requested, where="runtime audit")
    _validate_prior_environment(
        session.prior_environment, where="runtime audit"
    )
    if session.actual is None:
        raise ConfigError("runtime session has not been verified.")
    _validate_actual(session.actual, where="runtime audit")

    prior = session.prior_environment
    actual = session.actual
    if actual.jax_enable_x64 is not session.requested.jax_enable_x64:
        raise ConfigError(
            "runtime audit actual jax_enable_x64 contradicts the requested state."
        )
    if (
        session.requested.platform != "auto"
        and actual.backend != session.requested.platform
    ):
        raise ConfigError(
            "runtime audit actual backend contradicts the requested platform."
        )

    projected = {
        "requested": {
            "jax_enable_x64": session.requested.jax_enable_x64,
            "platform": session.requested.platform,
        },
        "actual": {
            "jax_enable_x64": bool(actual.jax_enable_x64),
            "backend": str.__str__(actual.backend),
        },
        "prior_environment": {
            "jax_enable_x64": (
                None
                if prior.jax_enable_x64 is None
                else str.__str__(prior.jax_enable_x64)
            ),
            "jax_platforms": (
                None
                if prior.jax_platforms is None
                else str.__str__(prior.jax_platforms)
            ),
        },
    }
    frozen = freeze_evidence(projected, where="runtime audit row")
    if not static_isinstance(frozen, Mapping):
        raise ConfigError("runtime audit row must be a mapping.")
    return frozen  # type: ignore[return-value]


def _safe_exception_text(exc: Exception) -> str:
    """Render ordinary foreign failures without invoking override hooks."""
    name = static_type_name(exc)
    args = BaseException.args.__get__(exc, BaseException)
    if type(args) is tuple and len(args) == 1:
        detail = args[0]
        if (
            type(detail) is str
            and str.__len__(detail) <= _FOREIGN_EXCEPTION_DETAIL_LIMIT
            and _is_utf8_text(detail)
        ):
            return f"{name}: {detail}"
    return f"{name}: details unavailable"


def _validate_requested(requested: object, *, where: str) -> None:
    if type(requested) is not RuntimeSpec:
        raise ConfigError(f"{where}: requested state must be a RuntimeSpec.")
    x64 = object.__getattribute__(requested, "jax_enable_x64")
    platform = object.__getattribute__(requested, "platform")
    if not static_isinstance(x64, bool):
        raise ConfigError(f"{where}: requested jax_enable_x64 must be a bool.")
    if (
        not static_isinstance(platform, str)
        or type(platform) is not str
        or platform not in ("auto", "cpu", "gpu", "tpu")
    ):
        raise ConfigError(f"{where}: requested platform is invalid.")


def _copy_requested(requested: object, *, where: str) -> RuntimeSpec:
    _validate_requested(requested, where=where)
    try:
        x64 = object.__getattribute__(requested, "jax_enable_x64")
        platform = object.__getattribute__(requested, "platform")
        seed = object.__getattribute__(requested, "seed")
        seeds = object.__getattribute__(requested, "seeds")
    except Exception:
        raise ConfigError(f"{where}: requested state is malformed.") from None
    return RuntimeSpec(x64, platform, seed, seeds)


def _copy_prior_environment(prior: object, *, where: str) -> PriorEnvironment:
    _validate_prior_environment(prior, where=where)
    try:
        x64 = object.__getattribute__(prior, "jax_enable_x64")
        platforms = object.__getattribute__(prior, "jax_platforms")
    except Exception:
        raise ConfigError(f"{where}: prior environment is malformed.") from None
    return PriorEnvironment(x64, platforms)


def _copy_actual(actual: object, *, where: str) -> RuntimeActual:
    _validate_actual(actual, where=where)
    try:
        x64 = object.__getattribute__(actual, "jax_enable_x64")
        backend = object.__getattribute__(actual, "backend")
    except Exception:
        raise ConfigError(f"{where}: actual state is malformed.") from None
    return RuntimeActual(x64, backend)


def _validate_prior_environment(prior: object, *, where: str) -> None:
    if type(prior) is not PriorEnvironment:
        raise ConfigError(
            f"{where}: prior environment must be a PriorEnvironment."
        )
    for field_name in ("jax_enable_x64", "jax_platforms"):
        value = object.__getattribute__(prior, field_name)
        if value is not None and type(value) is not str:
            raise ConfigError(
                f"{where} prior_environment.{field_name} must be a string or null."
            )
        if value is not None:
            _validate_utf8_text(
                value,
                where=f"{where} prior_environment.{field_name}",
            )


def _validate_actual(actual: object, *, where: str) -> None:
    if type(actual) is not RuntimeActual:
        raise ConfigError(f"{where}: actual state must be a RuntimeActual.")
    x64 = object.__getattribute__(actual, "jax_enable_x64")
    backend = object.__getattribute__(actual, "backend")
    if type(x64) is not bool:
        raise ConfigError(f"{where} actual jax_enable_x64 must be a bool.")
    if type(backend) is not str or not backend:
        raise ConfigError(
            f"{where} actual backend must be a non-empty string."
        )
    _validate_utf8_text(backend, where=f"{where} actual backend")


__all__ = [
    "PriorEnvironment",
    "RUNTIME_ACTUAL_ROW_KEYS",
    "RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS",
    "RUNTIME_REQUESTED_ROW_KEYS",
    "RUNTIME_ROW_KEYS",
    "RuntimeActual",
    "RuntimeSession",
    "establish_runtime",
    "runtime_audit_row",
]
