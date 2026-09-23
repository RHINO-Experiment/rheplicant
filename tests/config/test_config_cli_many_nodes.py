"""Every spelling of a ``many`` node and of ``compose: stages`` on the command line.

The command line audits every delivered value against the origins tree, which
addresses a list entry by its index (``model.foregrounds[0].amplitude``), a
FAN entry by its label (``model.cal_loads.hot.t_load``) and a composed stage
by its position (``model.gain.stages[0].gain``). The builder used to name all
three by the bare node (``model.foregrounds.amplitude``), so every such
document was refused with ``audit: no origin for ...`` although the same
mapping ran from Python, where no origins tree exists. These tests go
through ``main()`` for that reason: a test that calls ``build_model`` or
``run_document`` directly never reaches the audit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.config.test_config_cli import document, write_document

FOREGROUND = {
    "amplitude": {"value": 2500.0, "unit": "K"},
    "spectral_index": 2.55,
    "ref_freq": {"value": 70.0, "unit": "MHz"},
}
BAND = {"type": "FourierBandFilter", "axis": 0, "low": 0.02, "high": 0.5}
GAIN_STAGES = {
    "compose": "cascade",
    "stages": [
        {
            "name": "lna",
            "type": "GainOperator",
            "gain": {"value": 1.0, "unit": "dimensionless"},
        },
        {
            "name": "backend",
            "type": "GainOperator",
            "gain": {"value": 1.1, "unit": "dimensionless"},
        },
    ],
}


BASIS = {"time": {"kind": "legendre", "n_basis": 2}, "freq": {"kind": "legendre", "n_basis": 3}}


def _with_model(nodes: dict, *, observation: dict | None = None, output: Path | None = None):
    value = document(output=output)
    value["model"] = {**value["model"], **nodes}
    if observation is not None:
        value["observation"] = {**value["observation"], **observation}
    if "t_sys_extra" in nodes:
        value["resources"] = {**value.get("resources", {}), "bases": {"b": BASIS}}
    return value


CASES = {
    "one_foreground": ({"foregrounds": [FOREGROUND]}, None),
    "two_foregrounds": ({"foregrounds": [FOREGROUND, FOREGROUND]}, None),
    "filter_chain": ({"filters": [BAND]}, None),
    "cal_loads_by_label": (
        {
            "cal_loads": {
                "ambient": {"t_load": {"value": 300.0, "unit": "K"}},
                "hot": {"t_load": {"value": 400.0, "unit": "K"}},
            }
        },
        {"switching": {"mode": "cycle", "order": ["antenna", "ambient", "hot"]}},
    ),
    "compose_stages": ({"gain": GAIN_STAGES}, None),
    # from: basis builds its own coeff destination rather than via _construct.
    "t_sys_extra_from_basis": (
        {
            "t_sys_extra": [
                {
                    "from": "basis",
                    "basis": {"ref": "resources.bases.b"},
                    "coeff": {"zeros": [2, 3], "unit": "K"},
                }
            ]
        },
        None,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_validate_accepts_every_many_and_compose_spelling(tmp_path, capsys, case):
    from _rheplicant_bootstrap.cli import main

    nodes, observation = CASES[case]
    config = tmp_path / "config.yaml"
    write_document(config, _with_model(nodes, observation=observation))
    status = main(["validate", str(config)])
    streams = capsys.readouterr()
    assert status == 0, streams.err
    assert "audit: no origin" not in streams.err


#: The audit's numeric delivery rows name each many/compose field by the document path
#: the user wrote, index and label included.
EXPECTED_PATHS = {
    "two_foregrounds": {
        "model.foregrounds[0].amplitude",
        "model.foregrounds[1].amplitude",
    },
    "cal_loads_by_label": {
        "model.cal_loads.ambient.t_load",
        "model.cal_loads.hot.t_load",
    },
    "compose_stages": {
        "model.gain.stages[0].gain",
        "model.gain.stages[1].gain",
    },
    "t_sys_extra_from_basis": {"model.t_sys_extra[0].coeff"},
}


def _resolved(target: Path) -> dict:
    return yaml.safe_load((target / "config.resolved.yaml").read_text())


def _delivered_paths(resolved: dict) -> set[str]:
    """The document paths the audit recorded a numeric delivery at."""
    return set(resolved["_rheplicant_resolved"]["numeric"])


@pytest.mark.parametrize("case", sorted(EXPECTED_PATHS))
def test_run_records_each_entry_at_its_own_document_path(tmp_path, capsys, case):
    from _rheplicant_bootstrap.cli import main

    nodes, observation = CASES[case]
    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, _with_model(nodes, observation=observation, output=target))
    status = main(["run", str(config)])
    assert status == 0, capsys.readouterr().err
    recorded = _delivered_paths(_resolved(target))
    assert EXPECTED_PATHS[case] <= recorded, sorted(recorded)


def test_an_entry_default_lands_in_that_entry_of_the_resolved_document(tmp_path, capsys):
    """A default recorded at ``model.filters[1].mode`` is written into entry 1.

    Before indexed default paths existed, a list entry's defaults were named by
    the bare node and ``audit/resolved.py`` dropped them without a word, so the
    resolved document could not say which mode a filter ran with.
    """
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    chain = {"filters": [{**BAND, "mode": "extract"}, BAND]}
    write_document(config, _with_model(chain, output=target))
    assert main(["run", str(config)]) == 0, capsys.readouterr().err
    filters = _resolved(target)["model"]["filters"]
    assert [entry["mode"] for entry in filters] == ["extract", "remove"]
    assert "mode" not in _resolved(target)["model"]  # nothing at the bare node


def test_an_indexed_default_path_reaches_only_its_own_entry():
    """``[<n>]`` names one list entry; ``[]`` still names every entry."""
    from _rheplicant_bootstrap.audit.resolved import _apply_default, _copy_origins
    from _rheplicant_bootstrap.layering import OriginNode
    from _rheplicant_bootstrap.types import Origin

    def origins_of(value: object) -> OriginNode:
        if isinstance(value, dict):
            return OriginNode(Origin("user"), {k: origins_of(v) for k, v in value.items()})
        if isinstance(value, list):
            return OriginNode(Origin("user"), {i: origins_of(v) for i, v in enumerate(value)})
        return OriginNode(Origin("user"), {})

    def applied(path: str) -> list:
        document = {"model": {"xs": [{"a": 1}, {"a": 2}, {"a": 3}]}}
        origins = _copy_origins(document, origins_of(document), path="")
        _apply_default(document, origins, path=path, value=0)
        return document["model"]["xs"]

    assert applied("model.xs[1].k") == [{"a": 1}, {"a": 2, "k": 0}, {"a": 3}]
    assert applied("model.xs[7].k") == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert applied("model.xs[].k") == [{"a": i, "k": 0} for i in (1, 2, 3)]


#: One refusal per phase: the mapping form is pre-flight check A6; a list entry
#: missing a required field is refused while the model is built, after the
#: output lease is taken, which is where a ``*.refused-*`` directory appears.
REFUSED = {
    "preflight_mapping_form": {"foregrounds": FOREGROUND},
    "build_missing_field": {
        "foregrounds": [{k: v for k, v in FOREGROUND.items() if k != "ref_freq"}]
    },
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_refused_run_exits_2_through_the_installed_entry_point(tmp_path, case):
    """docs/config-cli.md: a refusal exits 2, and the process status says so.

    Asked of a subprocess rather than of ``main()``'s return value, because
    the status a shell sees is ``sys.exit(main())``'s, and it is the one a
    report of "exited 0" is about.
    """
    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, _with_model(REFUSED[case], output=target))
    completed = subprocess.run(
        [sys.executable, "-m", "_rheplicant_bootstrap", "run", str(config)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 2, (completed.returncode, completed.stderr[-2000:])
    assert not target.exists()
