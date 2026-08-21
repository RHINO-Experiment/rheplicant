from _rheplicant_bootstrap.capture import capture_routes

# Import the owners that extend the live reader/capture registries.
from rheplicant.config import values  # noqa: F401
from rheplicant.config.files import regular_reader_names
from rheplicant.config.kinds import beams, s_params  # noqa: F401
from rheplicant.config.sections import ingest, model  # noqa: F401

EXPECTED_CAPTURE_ROUTES = {
    "npy",
    "npz",
    "txt",
    "csv",
    "touchstone",
    "rhino_hdf5",
    "eqx_leaves",
    "cst",
    "uvbeam",
    "healpix",
}


def test_every_live_file_opening_route_has_one_capture_adapter():
    assert set(capture_routes()) == EXPECTED_CAPTURE_ROUTES
    assert set(regular_reader_names()) <= set(capture_routes())


def test_external_routes_have_explicit_distinct_owners():
    routes = capture_routes()
    assert routes["cst"] == "resources.beams:cst"
    assert routes["uvbeam"] == "resources.beams:uvbeam"
    assert routes["healpix"] == "resources.beams:healpix"
