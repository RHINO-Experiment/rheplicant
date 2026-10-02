"""The documents' likelihood and the one smc.py samples are the same function.

Each document is built through rheplicant (``load_document``, its strategy
hooks included); at several latent points its Gaussian log-likelihood,
from the fit twin's prediction and the declared noise model, is compared
with ``smc.log_likelihood`` on the compressed statistic ``prepare.py``
saved. They must agree up to one constant per document. Needs the arrays
``simulate.py`` and ``prepare.py`` write; skips without them.
"""

import numpy as np
import pytest
import yaml

import global21cm.plugin  # noqa: F401  (the registrations the documents' plugins: runs)
from global21cm import collapse, scenario, signal21, smc

HERE = scenario.HERE
CASES = [(model, case) for model in ("oracle", "beamconv", "physical") for case in ("main", "stress")]
N_POINTS = 5


def _needed(case):
    directory = scenario.SCENARIOS[case].sim_dir
    return [directory / f"{m}_collapsed.npz" for m in ("oracle", "beamconv", "physical")]


def _document_loglik(model: str, case: str):
    from rheplicant.config import load_document

    document = yaml.safe_load((HERE / f"{model}.yaml").read_text())
    for key in ("plugins", "outputs"):
        document.pop(key)
    run = load_document(document, variant=None if case == "main" else case, base_dir=str(HERE))
    inference = run.inference
    observed = np.asarray(inference.observed.entries["primary"])
    space, twin = inference.space, inference.fit_twin

    def loglik(u_row):
        values = dict(zip(signal21.THETA_NAMES, (np.float64(v) for v in u_row)))
        prediction = space.bind(twin, values)(run.state).data
        std = np.asarray(inference.noise.model.std(prediction))
        return float(-0.5 * np.sum(((observed - np.asarray(prediction)) / std) ** 2))

    return loglik


@pytest.mark.parametrize("model, case", CASES)
def test_document_and_smc_likelihoods_agree(model, case):
    if not all(p.is_file() for p in _needed(case)):
        pytest.skip("run simulate.py and prepare.py first")
    points = np.random.default_rng(9).normal(size=(N_POINTS, len(signal21.THETA_NAMES)))
    document = [_document_loglik(model, case)(u) for u in points]
    item = collapse.load(scenario.SCENARIOS[case].sim_dir, model)
    sampled = np.asarray(smc.log_likelihood(item.design, item.data, scenario.SCENARIOS[case].freqs_mhz())(points))
    difference = np.asarray(document) - sampled
    spread = np.ptp(sampled)
    assert spread > 1.0  # the points are far enough apart for the check to mean something
    assert np.ptp(difference) < 1e-8 * max(spread, 1.0)
