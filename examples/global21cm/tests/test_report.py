"""The tables ``analyse.py`` prints, from the ``fom.json`` it keeps beside the figures."""

import json

import pytest

from global21cm import report, scenario

FOM = scenario.RESULTS / "analysis" / "fom.json"
POSTERIORS = ("oracle", "beamconv", "physical")


def test_absent_values_print_as_a_dash():
    assert report._f(None, ".3g") == "-"
    assert report._f(0.123456, ".3g") == "0.123"
    assert report._f(True, "") == "True"


@pytest.fixture(scope="module")
def tables() -> dict:
    """The per-scenario reports of the shipped fom.json."""
    return {k: v for k, v in json.loads(FOM.read_text()).items() if k != "input_sha256"}


def _printed(reports, capsys) -> list[list[list[str]]]:
    """The cells of every row of every printed table, header and rule dropped."""
    report.print_tables(reports)
    blocks = capsys.readouterr().out.strip().split("\n\n")
    return [[[c.strip() for c in line.strip().strip("|").split("|")] for line in block.splitlines()[2:]]
            for block in blocks]  # fmt: skip


def test_every_table_has_a_row_per_scenario_and_posterior(tables, capsys):
    printed = _printed(tables, capsys)
    assert len(printed) == 5
    assert [(r[0], r[1]) for r in printed[0]] == [(s, k) for s in tables for k in ("prior", *POSTERIORS)]
    for rows in printed[1:]:
        assert [(r[0], r[1]) for r in rows] == [(s, k) for s in tables for k in POSTERIORS]


def test_headline_columns_read_their_own_fields(tables, capsys):
    headline = _printed(tables, capsys)[0]
    for cells in headline:
        entry = tables[cells[0]][cells[1]]
        assert cells[2] == format(entry["ser"], ".3g")
        assert cells[10] == ("-" if entry.get("calibrated") is None else str(entry["calibrated"]))
        assert cells[11] == ("-" if entry.get("eta_smc") is None else format(entry["eta_smc"], ".3g"))
