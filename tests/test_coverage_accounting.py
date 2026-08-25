"""Every xdist worker's coverage must reach the combine step.

A worker's data file that is written but never combined produces **no
diagnostic of any kind**. Measured by injecting exactly that fault -- one
entry removed from ``coverage.data.combinable_files`` on the full suite:

    missed    3272 -> 3867   (595 statements)
    coverage  89.22 % -> 87.26 %
    banner    none
    warning   none
    verdict   FAIL, but only via `fail_under`

That is the recorded intermittent dropout (``PROPOSAL_MERGED.md`` A0.3, "one
xdist worker's data never being combined") reproduced on demand, and it
matches its description: the tests all passed and only the coverage vanished.

**pytest-cov's own ``failed workers`` banner does not cover this**, which was
worth finding out before trusting it. The banner fires when a worker returns
no ``cov_worker_node_id``; the DATA travels by disk, not by that channel.
Measured both ways: dropping the node id raises the banner and loses nothing
(17128 missed either way), and withholding the file loses 595 statements and
raises no banner. Neither implies the other, so a guard built on the banner
would have watched the wrong thing.

Nothing else can see it either. Coverage's only no-data warning tests
``not self._data`` -- *no* data at all -- and ``combine_parallel_data``
reports its ``combined/skipped/errored`` counts through ``message``, which is
``Coverage(messages=False)`` by default and pytest-cov never enables. The
count of files merged is knowable only at the master, and nothing was asking.

Today the floor catches every such loss, and by a wide margin rather than by
luck: headroom is **66.5 statements** (89.2191 % against a floor of 89), and
the smallest per-worker exclusive share measured on this suite is **348**
statements -- 5.2x the headroom. This guard exists anyway, because the floor
says "the number is too low" and this says WHICH WORKER went missing.
"""

from __future__ import annotations

import pytest

from tests.coverage_accounting import (
    EXPECTED_MASTER_FILES,
    crashed_worker,
    missing_worker_data,
)


class TestThePredicate:
    def test_a_complete_accounting_says_nothing(self):
        assert missing_worker_data(combined=9, numprocesses=8) is None

    def test_more_files_than_expected_is_not_this_guard_s_business(self):
        """A restarted worker leaves an extra file. Not a loss, not reported."""
        assert missing_worker_data(combined=10, numprocesses=8) is None

    def test_one_file_short_is_reported_and_says_how_many(self):
        message = missing_worker_data(combined=8, numprocesses=8)
        assert message is not None
        assert "8" in message and "9" in message

    def test_a_serial_run_is_exempt(self):
        """``numprocesses`` is None without xdist; there are no workers to lose."""
        assert missing_worker_data(combined=1, numprocesses=None) is None

    def test_a_run_without_coverage_is_exempt(self):
        """``combined`` is None when ``combine()`` never ran."""
        assert missing_worker_data(combined=None, numprocesses=8) is None

    @pytest.mark.parametrize("workers", [1, 2, 4, 8, 28])
    def test_the_expected_count_is_workers_plus_the_master_s_own_file(self, workers):
        """Measured, not assumed: ``-n 4`` returns FIVE files to combine.

        The master measures itself too (``DistMaster.start`` calls
        ``self.cov.start()``), so its own data file is in the glob. The
        ``.combine`` file is NOT -- it is written after combining, so it is
        never in the list this counts.
        """
        assert missing_worker_data(
            combined=workers + EXPECTED_MASTER_FILES, numprocesses=workers
        ) is None
        assert missing_worker_data(
            combined=workers + EXPECTED_MASTER_FILES - 1, numprocesses=workers
        ) is not None

    def test_the_message_names_the_consequence_not_just_the_count(self):
        """A count alone reads as a curiosity; this has to read as a defect."""
        message = missing_worker_data(combined=8, numprocesses=8)
        assert "coverage" in message.lower()
        assert "undercount" in message.lower()


class TestTheHookCanActuallyFailTheRun:
    """The property this guard nearly shipped without.

    The first wiring bumped ``session.testsfailed`` from
    ``pytest_terminal_summary`` -- pytest-cov's own idiom, copied from a
    place where it works. It works THERE because pytest-cov does it inside
    ``pytest_runtestloop``, before the exit status is computed. From
    ``pytest_terminal_summary`` the status is already decided: measured on a
    one-test session, the bump printed and the process still exited 0.

    A guard that reports and does not fail is precisely the defect this
    module exists to catch, so the difference is pinned here as an
    executable fact rather than left as a comment that could quietly stop
    being true after a pytest upgrade.

    Run as subprocesses because an exit code is the thing under test, and
    the current process has already decided its own.
    """

    HARNESS = "def test_passes():\n    assert True\n"

    def _run(self, tmp_path, conftest):
        import subprocess
        import sys

        (tmp_path / "conftest.py").write_text(conftest)
        (tmp_path / "test_ok.py").write_text(self.HARNESS)
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

    def test_setting_exitstatus_in_sessionfinish_fails_the_run(self, tmp_path):
        result = self._run(
            tmp_path,
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    if exitstatus == 0:\n"
            "        session.exitstatus = 1\n",
        )
        assert result.returncode == 1, result.stdout
        assert "1 passed" in result.stdout

    def test_bumping_testsfailed_in_terminal_summary_does_NOT(self, tmp_path):
        """The anti-regression clause: if this ever starts failing, the
        comment in ``tests/conftest.py`` explaining the hook choice is wrong
        and the simpler wiring should be preferred."""
        result = self._run(
            tmp_path,
            "def pytest_terminal_summary(terminalreporter, exitstatus, config):\n"
            "    session = getattr(terminalreporter, '_session', None)\n"
            "    if session is not None:\n"
            "        session.testsfailed += 1\n",
        )
        assert result.returncode == 0, result.stdout

    def test_a_failing_run_is_not_downgraded(self, tmp_path):
        """``exitstatus == 0`` guards the override, so a worse status stands.

        Without it a run that was INTERRUPTED (2) or internally errored (3)
        would be reported as an ordinary test failure, which is a less
        urgent thing than what actually happened.
        """
        result = self._run(
            tmp_path,
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    if exitstatus == 0:\n"
            "        session.exitstatus = 1\n"
            "\n"
            "def pytest_collection_modifyitems(session, config, items):\n"
            "    session.shouldstop = 'stopping on purpose'\n",
        )
        assert result.returncode != 1, result.stdout


class TestTheCrashedWorkerPredicate:
    """The second mechanism, and the reason the file count is not enough.

    A worker that DIES never reaches ``DistWorker.finish``, so it writes no
    data file. Whether the file count notices depends on what xdist does
    next, and both outcomes were measured rather than predicted:

    * killing a worker with ``os._exit`` partway through a ``-n 4`` run
      produced NO replacement line, the count fell 5 -> 3, and
      :func:`missing_worker_data` caught it;
    * where xdist does clone the node -- ``replacing crashed worker gwN``,
      allowed ``numprocesses * 4`` times by default, 32 at ``-n 8`` -- the
      replacement writes a file and the count returns to normal, which is
      the case this predicate exists for.

    Only the item the worker died on is re-run. Everything it had already
    finished stays reported as passed, and that work's coverage is gone.
    "Those tests all passed; only their coverage vanished" is the recorded
    fault's own wording, and this is a mechanism that produces it exactly --
    including why the loss clusters on the subprocess-driven modules, which
    are the tests that can take a worker down.

    NOT observed on this machine: 32 runs logged during the investigation
    contain no ``replacing crashed worker`` line at all. So this is a
    consistent explanation rather than a demonstrated one, and the guard is
    what would turn the next occurrence into evidence.
    """

    def test_a_clean_run_says_nothing(self):
        assert crashed_worker(failed_workers=0, restarts=0) is None

    def test_a_worker_with_no_coverage_data_is_reported(self):
        message = crashed_worker(failed_workers=1, restarts=0)
        assert message is not None
        assert "no coverage data" in message

    def test_a_replaced_worker_is_reported_even_with_no_failed_workers(self):
        """The count check is blind here, so this must not be."""
        message = crashed_worker(failed_workers=0, restarts=1)
        assert message is not None
        assert "replaced" in message

    def test_both_together_are_reported_together(self):
        message = crashed_worker(failed_workers=2, restarts=1)
        assert "2 worker(s) returned no coverage data" in message
        assert "replaced 1 crashed worker(s)" in message

    def test_the_message_says_why_the_file_count_missed_it(self):
        """Otherwise the next reader deletes one of the two checks."""
        message = crashed_worker(failed_workers=1, restarts=1)
        assert "file count cannot see this" in message
        assert "UNDERCOUNT" in message
