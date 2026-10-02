"""The YAML documents agree with ``scenario.py``, with their strategies, and with each other.

Text-level checks only: no document is built here, so these run without
the simulation's arrays on disk.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from global21cm import collapse, physical_inputs, scenario, signal21

HERE = Path(__file__).resolve().parents[1]
FULL = ("oracle", "beamconv", "physical")
STRATEGY = {"oracle": "known_foreground", "beamconv": "per_lst_moments", "physical": "gaussian_sky"}
OWN = {"collapsed": {"ref": "resources.arrays.foreground"}}


def _load(name: str) -> dict:
    return yaml.safe_load((HERE / f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", [*FULL, *(f"{n}_quick" for n in FULL)])
class TestEachDocument:
    def test_grids_match_the_two_scenarios(self, name):
        doc = _load(name)
        base = doc["observation"]["freq"]["grid"]["linspace"]
        stress = doc["variants"]["stress"]["observation"]["freq"]["grid"]["linspace"]
        for grid, case in ((base, scenario.MAIN), (stress, scenario.STRESS)):
            assert (grid["start"], grid["stop"], grid["num"]) == (
                case.freq_start_mhz, case.freq_stop_mhz, case.n_freq,
            )  # fmt: skip
        assert doc["observation"]["time"]["grid"]["arange"]["num"] == 1

    def test_the_strategy_is_in_the_document(self, name):
        key = name.removesuffix("_quick")
        arrays = _load(name)["resources"]["arrays"]
        assert arrays["foreground"]["python"] == f"global21cm.strategies:{STRATEGY[key]}"
        for path in (a["file"]["path"] for a in arrays["foreground"]["args"].values()):
            assert path.startswith("results/sim/")

    def test_design_and_start_use_this_documents_own_foreground(self, name):
        key = name.removesuffix("_quick")
        doc = _load(name)
        arrays = doc["resources"]["arrays"]
        assert arrays["design"]["python"] == "global21cm.strategies:design_checked"
        assert arrays["start"]["python"] == "global21cm.strategies:start_point"
        for entry in ("design", "start"):
            assert arrays[entry]["args"]["collapsed"] == OWN["collapsed"]
        assert arrays["design"]["args"]["observed"]["file"]["path"] == f"results/sim/{key}_data.npy"
        assert doc["inference"]["observed"]["file"]["path"] == f"results/sim/{key}_data.npy"
        assert doc["model"]["global_signal"]["design"] == {"ref": "resources.arrays.design"}

    def test_stress_variant_moves_every_file(self, name):
        doc = _load(name)
        base = doc["resources"]["arrays"]
        stress = doc["variants"]["stress"]["resources"]["arrays"]
        assert set(stress["foreground"]["args"]) == set(base["foreground"]["args"])
        paths = [a["file"]["path"] for a in stress["foreground"]["args"].values()]
        paths += [stress["design"]["args"]["observed"]["file"]["path"],
                  stress["start"]["args"]["freqs_mhz"]["file"]["path"],
                  doc["variants"]["stress"]["inference"]["observed"]["file"]["path"]]  # fmt: skip
        assert all(p.startswith("results/sim_stress/") for p in paths)

    def test_noise_is_the_compressed_scale(self, name):
        noise = _load(name)["inference"]["noise"]
        assert noise["kind"] == "homoscedastic"
        assert noise["sigma"]["value"] == collapse.SIGMA0 == scenario.NOISE_SIGMA_K

    def test_the_21cm_latents_are_the_seven_unit_normals_started_from_the_bank(self, name):
        params = _load(name)["inference"]["parameters"]
        assert tuple(params) == signal21.THETA_NAMES
        for index, latent in enumerate(signal21.THETA_NAMES):
            assert params[latent]["prior"] == {"normal": {"loc": 0.0, "scale": 1.0}}
            init = params[latent]["init"]
            assert init["args"] == {"start": {"ref": "resources.arrays.start"}}
            assert init["literal"] == {"index": index}

    def test_model_theta_is_the_box_centre(self, name):
        theta = _load(name)["model"]["global_signal"]["theta"]["list"]
        low, high = (np.asarray(v) for v in signal21.prior_box())
        np.testing.assert_allclose(theta, (low + high) / 2, atol=1e-6)

    def test_both_scenarios_run_with_four_chains(self, name):
        runs = _load(name)["runs"]
        assert [(r["name"], r.get("variant")) for r in runs] == [("posterior", None), ("stress", "stress")]
        assert all(r["kind"] == "nuts" and r["num_chains"] == 4 for r in runs)


def test_physical_prior_structure_is_on_the_declared_grid():
    for name in ("physical", "physical_quick"):
        literal = _load(name)["resources"]["arrays"]["foreground"]["literal"]
        structure = (literal["amplitude_frac"], literal["global_amplitude_sigma"], literal["global_index_sigma"])
        assert structure in physical_inputs.PRIOR_GRID
        assert literal["template"] == "gsm2008"
        assert (literal["nside"], literal["lmax"]) == (scenario.NSIDE_FIT, scenario.LMAX_FIT)


def test_beamconv_literal_matches_the_scenario():
    for name in ("beamconv", "beamconv_quick"):
        literal = _load(name)["resources"]["arrays"]["foreground"]["literal"]
        assert literal["beta0"] == scenario.BETA0_MOMENT
        assert literal["nu_ref_mhz"] == scenario.NU_REF_MOMENT_MHZ


#: The only keys a quick document may change: each run's chain lengths, and outputs.dir.
RUN_LENGTHS = ("num_warmup", "num_samples")


def _without(mapping: dict, keys) -> dict:
    return {k: v for k, v in mapping.items() if k not in keys}


@pytest.mark.parametrize("name", FULL)
def test_quick_differs_only_in_chain_lengths_and_output_dir(name):
    full, quick = _load(name), _load(f"{name}_quick")
    assert set(full) == set(quick)
    for section in full:
        if section not in ("runs", "outputs"):
            assert full[section] == quick[section], section
    assert [_without(r, RUN_LENGTHS) for r in full["runs"]] == [_without(r, RUN_LENGTHS) for r in quick["runs"]]
    assert _without(full["outputs"], ("dir",)) == _without(quick["outputs"], ("dir",))
    assert quick["outputs"]["dir"] == full["outputs"]["dir"] + "_quick"
    # And the quick runs are the shorter ones, whatever their lengths are.
    for f, q in zip(full["runs"], quick["runs"]):
        assert all(q[key] <= f[key] for key in RUN_LENGTHS), f["name"]
