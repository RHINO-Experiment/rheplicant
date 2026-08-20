import { useEffect, useRef, useState } from "react";

import { StatusChip, type StatusTone } from "./StatusChip";
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

function jobState(job: JobProjection): {
  label: string;
  tone: StatusTone;
} {
  const suffix = job.message ? `: ${job.message}` : "";
  if (job.status === "queued") return {
    label: `Queued ${job.kind} at revision ${job.revision}`,
    tone: "neutral",
  };
  if (job.status === "running") return {
    label: `Running ${job.kind} at revision ${job.revision}`,
    tone: "neutral",
  };
  if (job.status === "refused") return {
    label: `Refused ${job.kind} at revision ${job.revision}${suffix}`,
    tone: "danger",
  };
  if (job.status === "error") return {
    label: `Internal error in ${job.kind} at revision ${job.revision}${suffix}`,
    tone: "danger",
  };
  if (job.stale) return {
    label: `Succeeded ${job.kind} from revision ${job.revision}`,
    tone: "stale",
  };
  return {
    label: `Current ${job.kind} succeeded at revision ${job.revision}`,
    tone: "success",
  };
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
  const knownStatuses = useRef(new Map(jobs.map((job) => [job.job_id, job.status])));
  const [urgentJob, setUrgentJob] = useState<string | null>(null);
  const active = jobs.filter(isActive);
  const latestTerminal = jobs.filter(isTerminal).at(-1);
  const urgentTerminal = urgentJob === null
    ? undefined
    : jobs.find((job) => (
      isTerminal(job) && `${job.job_id}:${job.status}` === urgentJob
    ));
  const visibleTerminal = urgentTerminal ?? latestTerminal;
  const visible = visibleTerminal === undefined ? active : [...active, visibleTerminal];
  const retrySeconds = nextRetryMs === null ? null : nextRetryMs / 1000;

  useEffect(() => {
    let terminalTransition: JobProjection | undefined;
    let dangerTransition: JobProjection | undefined;
    for (const job of jobs) {
      if (
        isTerminal(job)
        && knownStatuses.current.get(job.job_id) !== job.status
      ) {
        terminalTransition = job;
        if (job.status === "refused" || job.status === "error") dangerTransition = job;
      }
    }
    knownStatuses.current = new Map(jobs.map((job) => [job.job_id, job.status]));
    setUrgentJob((current) => {
      if (dangerTransition !== undefined) {
        return `${dangerTransition.job_id}:${dangerTransition.status}`;
      }
      if (terminalTransition !== undefined) {
        return null;
      }
      if (current === null) return null;
      return jobs.some((job) => `${job.job_id}:${job.status}` === current) ? current : null;
    });
  }, [jobs]);

  return (
    <section aria-label="Jobs">
      <h2>Jobs</h2>
      {error ? (
        <StatusChip
          tone="danger"
          label={`Polling failure: ${error}${retrySeconds === null
            ? ""
            : `. Retrying in ${retrySeconds} ${retrySeconds === 1 ? "second" : "seconds"}`}`}
          urgent
        />
      ) : (
        <StatusChip
          tone="neutral"
          label={status === "polling" ? "Polling jobs" : "Job polling idle"}
        />
      )}
      <button
        type="button"
        disabled={disabled}
        aria-describedby={disabledReasonId}
        onClick={refreshNow}
      >
        Refresh jobs
      </button>
      {visible.length === 0 ? (
        <StatusChip tone="neutral" label="No jobs submitted." />
      ) : (
        <ul>
          {visible.map((job) => {
            const presentation = jobState(job);
            return (
              <li key={job.job_id}>
                <code>{job.job_id}</code> · {job.kind} · {job.status}{" "}
                <StatusChip
                  tone={presentation.tone}
                  label={presentation.label}
                  urgent={urgentJob === `${job.job_id}:${job.status}`}
                />
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
