"""Shared fixtures, and the guard that every worker's coverage was combined."""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, Environment, State

N_TIME = 8
N_FREQ = 4


@pytest.fixture
def coords():
    return Coordinates(
        time=jnp.linspace(0.0, 7.0, N_TIME),
        freq=jnp.linspace(60e6, 85e6, N_FREQ),
    )


@pytest.fixture
def template_state(coords):
    """A seeded, data-less state ready to be pushed through a forward pipeline."""
    return State(
        coords=coords,
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO", "obs_id": "test-000"},
    )


# --- every xdist worker's coverage must reach the combine step --------------
#
# A data file that is written and then not combined is invisible: no banner,
# no warning, and the only symptom is a coverage figure that is quietly too
# low. Reproduced on demand by withholding one entry from `combinable_files`
# -- 595 statements and two points of coverage, in silence. The reasoning,
# the measurements and why pytest-cov's own `failed workers` banner does NOT
# cover this are in `tests/test_coverage_accounting.py`.
#
# The count is only knowable at the master, which is why this lives in a hook
# rather than in a test.

_combined_files: list[int] = []


def pytest_configure(config):
    """Record how many data files the combine step was given.

    Master only -- `combine()` never runs in a worker, and patching there
    would count nothing while making every worker pay for the import.
    """
    if hasattr(config, "workerinput"):
        return
    try:
        from coverage import data as coverage_data
    except ImportError:  # coverage is optional; a run without it has nothing to check
        return

    original = coverage_data.combinable_files

    def combinable_files(data_file, data_paths=None):
        found = original(data_file, data_paths)
        _combined_files.append(len(found))
        return found

    coverage_data.combinable_files = combinable_files


def _verdict(config):
    """The message for this run, or ``None`` -- across BOTH mechanisms.

    Two of them, and each is invisible to the other's check:

    * a data file written and then not combined -- the file count is short;
    * a worker that DIED before writing one -- xdist clones it, the
      replacement writes a file, and the count comes back to normal.

    The second is the likelier trigger for the recorded fault and the reason
    the first check alone is not enough.
    """
    from tests.coverage_accounting import crashed_worker, missing_worker_data

    if getattr(config.option, "numprocesses", None) is None:
        return None
    shortfall = missing_worker_data(
        combined=_combined_files[-1] if _combined_files else None,
        numprocesses=config.option.numprocesses,
    )
    if shortfall is not None:
        return shortfall
    if not _combined_files:  # no coverage on this run; nothing to undercount
        return None
    session = config.pluginmanager.getplugin("dsession")
    cov = config.pluginmanager.get_plugin("_cov")
    controller = getattr(cov, "cov_controller", None)
    return crashed_worker(
        failed_workers=len(getattr(controller, "failed_workers", ()) or ()),
        restarts=getattr(session, "_failed_nodes_count", 0) or 0,
    )


def pytest_sessionfinish(session, exitstatus):
    """FAIL the run, and in the one hook that still can.

    `session.testsfailed += 1` is pytest-cov's own idiom for this, and it
    works there because it happens inside `pytest_runtestloop` -- before the
    exit status is computed. From `pytest_terminal_summary` it is far too
    late: measured on a one-test session, bumping `testsfailed` there left
    the process exiting 0 while printing that it had failed. Setting
    `session.exitstatus` here exits 1.

    That is the same defect this guard exists to catch, one level up -- a
    report nobody is obliged to act on -- so it is worth the extra hook.

    Only when the run was otherwise clean: a session that is already failing,
    interrupted or internally errored has a more urgent status than this one
    and must not be downgraded to a plain 1.
    """
    if exitstatus == 0 and _verdict(session.config) is not None:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say which count went missing, where a reader will see it."""
    message = _verdict(config)
    if message is None:
        return
    terminalreporter.write_sep("=", "coverage: worker data lost", red=True, bold=True)
    terminalreporter.write_line(message)
