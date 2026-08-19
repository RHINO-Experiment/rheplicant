from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rheplicant.gui.previews import project_previews

DOCUMENT = {
    "schema_version": 1,
    "observation": {
        "freq": {
            "grid": {
                "linspace": {
                    "start": 60.0,
                    "stop": 84.0,
                    "num": 4,
                    "endpoint": True,
                },
                "unit": "MHz",
            }
        },
        "time": {
            "grid": {
                "arange": {"start": 0.0, "step": 2.0, "num": 8},
                "unit": "s",
            }
        },
        "switching": {"mode": "cycle", "order": ["antenna", "hot", "cold"]},
    },
    "resources": {
        "beams": {"horn": {"nside": 64}},
        "projectors": {"drift": {"lmax": 191, "optimizations": []}},
    },
    "model": {},
    "runs": [{"name": "forward", "kind": "forward"}],
}


def test_exactly_four_preview_classes_have_the_schema_cost_boundaries():
    found = project_previews(DOCUMENT)

    assert tuple(row.preview_id for row in found.classes) == (
        "graph",
        "axes_shapes",
        "validate",
        "forward",
    )
    assert tuple((row.cadence, row.priced) for row in found.classes) == (
        ("continuous", False),
        ("continuous", False),
        ("explicit", True),
        ("explicit", True),
    )
    with pytest.raises(FrozenInstanceError):
        found.classes[0].priced = True  # type: ignore[misc]


def test_axis_strips_and_shape_symbols_are_derived_without_building_a_twin():
    found = project_previews(DOCUMENT)
    by_axis = {row.axis: row for row in found.axes}
    shapes = {row.symbol: row.value for row in found.shapes}

    assert by_axis["freq"].first == (60.0, 68.0, 76.0)
    assert by_axis["freq"].last == (68.0, 76.0, 84.0)
    assert by_axis["freq"].count == 4
    assert by_axis["freq"].spacing == pytest.approx(8.0)
    assert by_axis["freq"].unit == "MHz"
    assert by_axis["time"].first == (0.0, 2.0, 4.0)
    assert by_axis["time"].last == (10.0, 12.0, 14.0)
    assert by_axis["time"].precision_ratio is not None
    assert by_axis["time"].precision_ok is True
    assert shapes == {
        "n_time": 8,
        "n_freq": 4,
        "n_source": 3,
        "n_load": 2,
        "resources.beams.horn.n_pix": 49_152,
        "resources.projectors.drift.n_alm": 18_528,
    }


def test_linspace_endpoint_and_time_precision_are_not_guessed():
    document = {
        **DOCUMENT,
        "observation": {
            **DOCUMENT["observation"],
            "freq": {
                "grid": {
                    "linspace": {
                        "start": 60.0,
                        "stop": 84.0,
                        "num": 4,
                        "endpoint": False,
                    },
                    "unit": "MHz",
                }
            },
            "time": {
                "grid": {
                    "arange": {"start": 1.0e16, "step": 1.0, "num": 4},
                    "unit": "s",
                }
            },
        },
    }
    found = project_previews(document)
    by_axis = {row.axis: row for row in found.axes}

    assert by_axis["freq"].last[-1] == 78.0
    assert by_axis["time"].precision_ok is False
    assert by_axis["time"].precision_ratio is not None
    assert by_axis["time"].precision_ratio < 100.0


def test_time_precision_gauge_uses_the_declared_storage_dtype():
    def with_runtime(*, x64):
        return {
            **DOCUMENT,
            "runtime": {"jax_enable_x64": x64},
            "observation": {
                **DOCUMENT["observation"],
                "time": {
                    "grid": {
                        "arange": {"start": 1.0e7, "step": 99.0, "num": 4},
                        "unit": "s",
                    }
                },
            },
        }

    float32 = {row.axis: row for row in project_previews(with_runtime(x64=False)).axes}
    float64 = {row.axis: row for row in project_previews(with_runtime(x64=True)).axes}

    assert float32["time"].precision_ratio == pytest.approx(99.0)
    assert float32["time"].precision_ok is False
    assert float64["time"].precision_ratio is not None
    assert float64["time"].precision_ratio > 100.0
    assert float64["time"].precision_ok is True


def test_forward_cost_label_responds_to_declared_scientific_size_and_optimisation():
    baseline = project_previews(DOCUMENT).forward_cost
    larger = {
        **DOCUMENT,
        "observation": {
            **DOCUMENT["observation"],
            "freq": {
                "grid": {
                    "linspace": {
                        "start": 60.0,
                        "stop": 85.0,
                        "num": 8,
                        "endpoint": True,
                    },
                    "unit": "MHz",
                }
            },
        },
    }
    cached = {
        **larger,
        "resources": {
            **larger["resources"],
            "projectors": {
                "drift": {
                    "lmax": 191,
                    "optimizations": ["cache_beam_rotation"],
                }
            },
        },
    }
    wider_band = {
        **larger,
        "resources": {
            **larger["resources"],
            "projectors": {"drift": {"lmax": 383, "optimizations": []}},
        },
    }

    larger_cost = project_previews(larger).forward_cost
    cached_cost = project_previews(cached).forward_cost
    wider_cost = project_previews(wider_band).forward_cost
    assert baseline.estimated_milliseconds is not None
    assert larger_cost.estimated_milliseconds == pytest.approx(
        2.0 * baseline.estimated_milliseconds
    )
    assert cached_cost.estimated_milliseconds < larger_cost.estimated_milliseconds
    assert wider_cost.estimated_milliseconds > larger_cost.estimated_milliseconds
    assert wider_cost.estimated_peak_megabytes > larger_cost.estimated_peak_megabytes
    assert "8 channels" in larger_cost.label
    assert "estimated" in larger_cost.label


def test_unresolved_or_file_axes_are_reported_as_unavailable_not_read():
    document = {
        **DOCUMENT,
        "observation": {
            **DOCUMENT["observation"],
            "freq": {"grid": {"file": "do-not-read.npy", "unit": "MHz"}},
        },
    }
    found = project_previews(document)

    assert tuple(row.axis for row in found.axes) == ("time",)
    assert "n_freq" not in {row.symbol for row in found.shapes}
    assert found.forward_cost.estimated_milliseconds is None
    assert "complete axes" in found.forward_cost.label


def test_generated_axis_preview_is_bounded_even_for_a_huge_declared_extent():
    document = {
        **DOCUMENT,
        "observation": {
            **DOCUMENT["observation"],
            "freq": {
                "grid": {
                    "arange": {"start": 10.0, "step": 0.5, "num": 1_000_000_000},
                    "unit": "MHz",
                }
            },
        },
    }

    freq = {row.axis: row for row in project_previews(document).axes}["freq"]

    assert freq.count == 1_000_000_000
    assert freq.first == (10.0, 10.5, 11.0)
    assert freq.last == (500_000_008.5, 500_000_009.0, 500_000_009.5)
