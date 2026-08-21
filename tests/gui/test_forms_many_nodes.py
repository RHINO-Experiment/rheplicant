"""A ``many`` node's fields live at one path per instance, not at one path.

``foregrounds``, ``t_sys_extra`` and ``filters`` are lists in a document and
``cal_loads`` is a label-keyed mapping, but the widget census spelled all four
flat -- ``model.filters.axis``. A flat segment reads a list with
``Mapping.get``, which never finds anything, so every field of every ``many``
node was permanently absent and, being required, permanently *must decide*.
The onboarding checklist showed ten of those on a fully valid document and
none of the real gaps on a document that had two.
"""

from __future__ import annotations

import pytest
import yaml

from rheplicant.gui.forms import (
    ProjectedWidget,
    assert_catalog_closed,
    project_forms,
    widget_catalog,
)
from rheplicant.gui.starter import STARTER_YAML

#: One valid ``foregrounds`` SUM entry and one valid ``filters`` CHAIN entry,
#: spliced into the shipped starter document -- which
#: ``test_previews.py::test_starter_is_valid_bounded_and_file_free`` already
#: pins at zero missing-required entries. Adding two fully specified nodes to
#: a document that needs nothing cannot make it need something, so the correct
#: answer here is zero by construction rather than by our own judgement.
D1_NODES = """\
  foregrounds:
    - amplitude: {value: 1000.0, unit: K}
      spectral_index: -2.5
      ref_freq: {value: 75.0, unit: MHz}
  filters:
    - type: SiderealFilter
      n_days: 3
"""

#: The same starter with exactly TWO real gaps: the ``cold`` load has no
#: ``t_load``, and the second filter is a ``FourierBandFilter`` with no
#: ``high``. Every other field of every other instance is supplied, so a
#: correct projection reports those two paths and nothing else.
D2_NODES = """\
  cal_loads:
    hot: {t_load: {value: 350.0, unit: K}}
    cold: {}
  filters:
    - type: SiderealFilter
      n_days: 3
    - type: FourierBandFilter
      axis: time
      low: {value: 0.0, unit: Hz}
"""

D2_SWITCHING = """\
  switching:
    mode: cycle
    order: [antenna, hot, cold]
"""


def _document(nodes: str, *, switching: bool = False) -> dict:
    """The starter document with ``nodes`` spliced under ``model:``."""
    text = STARTER_YAML
    if switching:
        text = text.replace("  environment:\n", D2_SWITCHING + "  environment:\n")
        assert D2_SWITCHING in text, "the switching splice found no anchor"
    spliced = text.replace("runs:\n", nodes + "runs:\n")
    assert spliced != text, "the model splice found no anchor in the starter"
    loaded = yaml.safe_load(spliced)
    assert isinstance(loaded, dict)
    return loaded


def _widgets(document: dict) -> dict[str, ProjectedWidget]:
    projected = project_forms(document)
    return {
        widget.path: widget
        for section in projected.sections
        for widget in section.widgets
    }


def _catalog_paths() -> set[str]:
    return {widget.path for widget in widget_catalog().widgets}


class TestManyNodeInstancePaths:
    def test_a_valid_document_with_a_list_node_needs_nothing(self):
        """D1. Ten phantom entries before the fix; the fields are supplied,
        so a projection that can see them reports nothing at all.

        The visibility assertions are not decoration: a projection that made
        the fields *invisible* rather than *present* would also report zero,
        and would be just as wrong."""
        document = _document(D1_NODES)
        projected = project_forms(document)

        assert projected.missing_required == ()

        widgets = _widgets(document)
        for path in (
            "model.foregrounds[0].amplitude",
            "model.foregrounds[0].spectral_index",
            "model.foregrounds[0].ref_freq",
            "model.filters[0].n_days",
        ):
            assert widgets[path].visible, f"{path} must be visible"
            assert widgets[path].present, f"{path} must read as present"
            assert not widgets[path].must_decide

    def test_the_two_real_gaps_are_the_only_gaps(self):
        """D2. Eight entries before the fix, none of them a real gap, and one
        of them (``model.cal_loads.t_load``) not even a writable path."""
        projected = project_forms(_document(D2_NODES, switching=True))

        assert sorted(projected.missing_required) == [
            "model.cal_loads.cold.t_load",
            "model.filters[1].high",
        ]

    def test_each_instance_answers_for_itself(self):
        """Instance 0 is a ``SiderealFilter`` and instance 1 a
        ``FourierBandFilter``; each one's ``type`` governs its own fields and
        neither governs the other's."""
        widgets = _widgets(_document(D2_NODES, switching=True))

        assert widgets["model.filters[0].n_days"].visible
        assert not widgets["model.filters[0].axis"].visible
        assert not widgets["model.filters[1].n_days"].visible
        assert widgets["model.filters[1].axis"].visible
        assert widgets["model.filters[1].axis"].value == "time"
        assert widgets["model.filters[1].high"].visible
        assert widgets["model.filters[1].high"].must_decide

    def test_a_fan_node_expands_to_the_documents_own_labels(self):
        """``cal_loads`` is keyed by ``observation.switching.order[1:]``, so
        its instances are named rather than numbered. The pattern the widget
        was declared with survives on the projection for the client to group
        by."""
        widgets = _widgets(_document(D2_NODES, switching=True))

        assert widgets["model.cal_loads.hot.t_load"].present
        assert not widgets["model.cal_loads.cold.t_load"].present
        assert widgets["model.cal_loads.cold.t_load"].must_decide
        for label in ("hot", "cold"):
            widget = widgets[f"model.cal_loads.{label}.t_load"]
            assert widget.path_pattern == "model.cal_loads.*.t_load"
            assert widget.label == "t load"
            assert "*" not in widget.label and "[]" not in widget.label

    @pytest.mark.parametrize(
        "path",
        [
            "model.global_signal.depth",
            "model.gain.gain",
            "model.noise.sigma",
            "model.noise_wave.switch_key",
            "model.cw_tone.line_width",
        ],
    )
    def test_a_single_slot_node_keeps_its_flat_path(self, path):
        """The prefix is for ``many`` nodes only. Twenty-nine of the thirty-
        three nodes hold one instance, and their fields are exactly where they
        were -- including in the four existing suites that name them."""
        assert path in _catalog_paths()

    def test_the_instance_prefix_does_not_move_a_node_between_sections(self):
        """``_section_for`` reads ``path.split(".")[1]`` and looks it up in the
        graph. ``"filters[]"`` is not a node id, so without a matching trim it
        falls through to the Instrument default -- silently moving every
        filter field out of Backend, where its ``processing`` segment puts
        it."""
        sections = {
            widget.path: widget.section
            for widget in widget_catalog().widgets
        }

        assert sections["model.filters"] == "backend"
        assert sections["model.filters[].axis"] == "backend"
        assert sections["model.foregrounds[].ref_freq"] == "instrument"
        assert sections["model.cal_loads.*.t_load"] == "instrument"

    def test_every_many_node_field_is_declared_once_at_its_instance_path(self):
        """The census half. A flat path left behind anywhere -- by a stale
        ``builder.source`` destination, say -- reappears as a second widget
        for the same field that can never read a value, which is the defect
        this whole change exists to remove."""
        paths = _catalog_paths()
        expected = {
            "model.foregrounds[].amplitude",
            "model.foregrounds[].spectral_index",
            "model.foregrounds[].ref_freq",
            "model.foregrounds[].type",
            "model.foregrounds[].eqx_leaves",
            "model.t_sys_extra[].coeff",
            "model.t_sys_extra[].time_basis",
            "model.t_sys_extra[].freq_basis",
            "model.t_sys_extra[].type",
            "model.t_sys_extra[].eqx_leaves",
            "model.cal_loads.*.t_load",
            "model.cal_loads.*.type",
            "model.cal_loads.*.eqx_leaves",
            "model.filters[].axis",
            "model.filters[].low",
            "model.filters[].high",
            "model.filters[].mode",
            "model.filters[].n_days",
            "model.filters[].projector",
            "model.filters[].regularization",
            "model.filters[].cg_tol",
            "model.filters[].cg_maxiter",
            "model.filters[].type",
            "model.filters[].eqx_leaves",
        }
        assert expected <= paths

        flat = {
            path
            for path in paths
            for node in ("foregrounds", "t_sys_extra", "cal_loads", "filters")
            if path.startswith(f"model.{node}.") and "*" not in path
        }
        assert flat == set(), f"flat many-node widget paths survive: {sorted(flat)}"

        # The node widget itself is the one flat path each of them keeps: it
        # names the whole list or mapping, not one instance of it.
        for node in ("foregrounds", "t_sys_extra", "cal_loads", "filters"):
            assert f"model.{node}" in paths

    def test_the_catalog_is_still_closed_over_its_live_registries(self):
        """Every dimension row still reaches a widget, and every widget path
        is still unique -- the two drift questions this change could have
        answered wrongly without any test noticing."""
        assert_catalog_closed()
