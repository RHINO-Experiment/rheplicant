"""How many coverage data files the combine step should have found.

The invariant is small and was measured rather than reasoned about: under
``-n N``, ``coverage.data.combinable_files`` returns exactly ``N + 1`` paths
-- one per worker, plus the master's own, because ``DistMaster.start`` starts
coverage on the master too. The ``.combine`` file is not among them; it is
written after the merge, so it is never in the list.

Fewer than that means a worker's contribution is missing from the reported
figure. See ``tests/test_coverage_accounting.py`` for the measurements, the
injected reproduction, and why pytest-cov's own ``failed workers`` banner
does not cover this case.
"""

from __future__ import annotations

#: The master measures itself, so its data file is in the glob beside the
#: workers'. Named rather than written as a bare ``+ 1`` so the reason for
#: the offset survives the next reader.
EXPECTED_MASTER_FILES = 1


def crashed_worker(*, failed_workers: int, restarts: int) -> str | None:
    """The sentence for a worker that DIED, or ``None``.

    A second mechanism, and one :func:`missing_worker_data` can miss. A
    worker that crashes never reaches ``DistWorker.finish``, so it writes no
    data file at all. What happens next decides whether the file count
    notices, and BOTH outcomes are real:

    * if xdist CLONES it -- ``replacing crashed worker gwN``, allowed up to
      ``numprocesses * 4`` times by default, which is 32 at ``-n 8`` -- the
      replacement writes a file of its own and the count comes back to
      ``workers + 1``, where the file-count check sees nothing wrong;
    * if it is not replaced, the count drops and the file-count check fires.

    Measured, by killing a worker with ``os._exit`` partway through a ``-n 4``
    run: no replacement line appeared, the count fell from 5 to 3, and the
    file-count branch caught it. So this predicate is not the only thing
    standing between a dead worker and a green run -- it is what covers the
    case where the replacement hides the arithmetic, which is the case the
    first draft of this docstring asserted was the ONLY one.

    Meanwhile the tests the dead worker had already completed are not re-run
    -- only the item it died on is -- so they stay green and their coverage
    is simply gone. That is the recorded fault's description exactly: "those
    tests all passed; only their coverage vanished", clustered on the
    subprocess-driven modules, which are the ones that can take a worker
    down in the first place.

    pytest-cov already notices (``DistMaster.testnodedown`` files the node
    under ``failed_workers`` and prints a banner) and does nothing about it:
    measured, the only branch in ``pytest_cov/plugin.py`` that touches the
    exit status is the coverage floor.

    :param failed_workers: how many nodes pytest-cov recorded as returning no
        coverage data.
    :param restarts: how many workers xdist replaced.
    """
    if not failed_workers and not restarts:
        return None
    parts = []
    if failed_workers:
        parts.append(f"{failed_workers} worker(s) returned no coverage data")
    if restarts:
        parts.append(f"xdist replaced {restarts} crashed worker(s)")
    return (
        f"coverage: {' and '.join(parts)}. A worker that dies never writes its "
        f"data file, and only the item it died on is re-run -- so the tests it "
        f"had already finished stay green and their coverage is an UNDERCOUNT. "
        f"The file count cannot see this: the replacement worker writes a file "
        f"and the total comes back to normal."
    )


def missing_worker_data(*, combined: int | None, numprocesses: int | None) -> str | None:
    """The sentence to print when a worker's coverage never arrived, else ``None``.

    A pure predicate, so it can be tested without an xdist session and
    without a coverage run -- the hook that feeds it is three lines and the
    judgement is here.

    :param combined: how many files ``combinable_files`` returned, or ``None``
        when combining never happened (no coverage on this run).
    :param numprocesses: xdist's resolved worker count, or ``None`` for a
        serial run, where there are no workers to lose.
    :returns: a message naming the shortfall and what it means for the
        figure, or ``None`` when the accounting is complete.
    """
    if combined is None or numprocesses is None:
        return None
    expected = numprocesses + EXPECTED_MASTER_FILES
    if combined >= expected:
        # MORE than expected is not this guard's business: a worker that
        # crashed and was restarted leaves an extra file behind, and that is
        # a surplus of data rather than a loss of it.
        return None
    return (
        f"coverage: {expected - combined} of {expected} data files never reached "
        f"the combine step ({combined} found, {numprocesses} workers + the "
        f"master's own). Every test may have passed; the reported coverage is "
        f"an UNDERCOUNT missing those workers' whole contribution. Measured on "
        f"this suite, one worker's exclusive share runs 348-1334 statements, so "
        f"the figure from this run should not be believed or recorded."
    )
