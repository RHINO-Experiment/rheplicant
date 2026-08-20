import type { JobPollingState } from "./useJobPolling";
import type { JobProjection } from "./types";

interface JobsDrawerProps extends JobPollingState {
  jobs: JobProjection[];
  disabled?: boolean;
  disabledReasonId?: string;
}

function isActive(job: JobProjection) {
  return job.status === "queued" || job.status === "running";
}

function isTerminal(job: JobProjection) {
  return !isActive(job);
}

export function JobsDrawer({
  jobs,
  status,
  error,
  nextRetryMs,
  refreshNow,
  disabled = false,
  disabledReasonId,
}: JobsDrawerProps) {
  const active = jobs.filter(isActive);
  const latestTerminal = jobs.filter(isTerminal).at(-1);
  const visible = latestTerminal === undefined ? active : [...active, latestTerminal];
  const retrySeconds = nextRetryMs === null ? null : nextRetryMs / 1000;

  return (
    <section aria-label="Jobs">
      <h2>Jobs</h2>
      <p aria-live="polite">
        {status === "polling" && "Polling jobs"}
        {status === "idle" && "Job polling idle"}
        {status === "retrying" && retrySeconds !== null
          && `Retrying in ${retrySeconds} ${retrySeconds === 1 ? "second" : "seconds"}`}
      </p>
      {error && <p role="alert">{error}</p>}
      <button
        type="button"
        disabled={disabled}
        aria-describedby={disabledReasonId}
        onClick={refreshNow}
      >
        Refresh jobs
      </button>
      {visible.length === 0 ? <p>No jobs submitted.</p> : (
        <ul>
          {visible.map((job) => (
            <li key={job.job_id}>
              <code>{job.job_id}</code> · {job.kind} · {job.status}
              {job.message && <> · {job.message}</>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
