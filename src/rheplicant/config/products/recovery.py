"""Truthful recovery tables from declared truth and recorded estimates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from _rheplicant_bootstrap.errors import ConfigError


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} is not a latent mapping.")
    return value


def _moments(kind: str, product: object) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if kind == "plan.estimate":
        return _mapping(product.values, where="Estimate.values"), {}
    if kind == "plan.sample" or kind in ("nuts", "npe"):
        samples = _mapping(product.samples, where=f"{kind} samples")
        centre = {name: np.mean(np.asarray(stack), axis=0) for name, stack in samples.items()}
        sigma = {name: np.std(np.asarray(stack), axis=0) for name, stack in samples.items()}
        return centre, sigma
    if kind == "conjugate.wiener":
        mapping = _mapping(product, where="conjugate.wiener product")
        centre = _mapping(mapping["mean"], where="conjugate.wiener mean")
        if "covariance" not in mapping:
            return centre, {}
        covariance = np.asarray(mapping["covariance"])
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ConfigError("conjugate.wiener covariance must be a square matrix.")
        flat_sigma = np.sqrt(np.diag(covariance))
        sigma: dict[str, object] = {}
        offset = 0
        for name, value in centre.items():
            shape = np.asarray(value).shape
            size = int(np.asarray(value).size)
            sigma[name] = flat_sigma[offset:offset + size].reshape(shape)
            offset += size
        if offset != flat_sigma.size:
            raise ConfigError("conjugate.wiener covariance does not match its mean.")
        return centre, sigma
    if kind == "conjugate.gls":
        return _mapping(product.solution, where="GLS solution"), {}
    raise ConfigError(f"kind: {kind} has no recovery product.")


def _json_number_or_array(value: object) -> object:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ConfigError("recovery values must be real numeric arrays.")
    if not np.all(np.isfinite(array)):
        raise ConfigError("recovery values must be finite.")
    return array.item() if array.ndim == 0 else array.tolist()


def recovery_record(kind: str, product: object, inference: object) -> dict[str, object]:
    """Build latent recovery rows, recording absent truth and undefined pulls."""
    centres, sigmas = _moments(kind, product)
    truth = _mapping(inference.truth, where="InferenceBuild.truth")
    truth_omitted = _mapping(inference.truth_omitted, where="InferenceBuild.truth_omitted")
    rows: list[dict[str, object]] = []
    omissions: list[dict[str, str]] = []
    for name, centre_value in centres.items():
        if name not in truth:
            omissions.append(
                {"name": name, "reason": str(truth_omitted.get(name, "truth is unavailable"))}
            )
            continue
        centre = np.asarray(centre_value)
        actual = np.asarray(truth[name])
        if centre.shape != actual.shape:
            raise ConfigError(f"recovery truth and centre shapes disagree for latent {name!r}.")
        error = centre - actual
        row: dict[str, object] = {
            "name": name,
            "truth": _json_number_or_array(actual),
            "centre": _json_number_or_array(centre),
            "absolute_error": _json_number_or_array(np.abs(error)),
        }
        if name in sigmas:
            sigma = np.asarray(sigmas[name])
            if sigma.shape != centre.shape:
                raise ConfigError(f"recovery sigma shape disagrees for latent {name!r}.")
            row["sigma"] = _json_number_or_array(sigma)
            if np.any(sigma == 0):
                omissions.append(
                    {"name": name, "reason": "posterior sigma is zero; pull is undefined"}
                )
            else:
                row["pull"] = _json_number_or_array(error / sigma)
        rows.append(row)
    return {"latents": rows, "omissions": omissions}


__all__ = ["recovery_record"]
