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

function identity(job: JobProjection) {
  return `${job.job_id}:${job.status}`;
}

// The transition the drawer is currently reporting. `urgent` separates "this is the news"
// from "this alerts": §7.4 shows the most recent terminal transition whatever it was, while
// only a refusal or an internal error interrupts with role="alert".
interface TerminalEvidence {
  key: string;
  urgent: boolean;
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
  // Initialised once, lazily: useRef keeps only the first argument, so building the Map inline
  // rebuilt and discarded one on every render for the whole life of the drawer.
  const knownStatuses = useRef<Map<string, JobProjection["status"]> | null>(null);
  if (knownStatuses.current === null) {
    knownStatuses.current = new Map(jobs.map((job) => [job.job_id, job.status]));
  }
  const [evidence, setEvidence] = useState<TerminalEvidence | null>(null);
  const active = jobs.filter(isActive);
  // Only a fallback: the backend orders jobs by submission, so array position names the most
  // recent completion only until an actual transition has been observed to retain.
  const arrayLatestTerminal = jobs.filter(isTerminal).at(-1);
  // No isTerminal test here, and none is possible: `identity` carries the status, and an evidence
  // key is only ever minted from a terminal job, so a job whose identity equals the key IS that
  // terminal job at that terminal status. The `isTerminal(job) &&` this line used to carry was a
  // second copy of an invariant the key already holds — unreachable by construction, which is why
  // no fixture could kill its removal. What keeps the key honest is the test that the identity
  // must not lose its status ("stands down when a job_id comes back at a status it never alerted
  // on"), plus the row census that no non-terminal row is ever reported as the latest completion.
  const transitionedTerminal = evidence === null
    ? undefined
    : jobs.find((job) => identity(job) === evidence.key);
  const visibleTerminal = transitionedTerminal ?? arrayLatestTerminal;
  const visible = visibleTerminal === undefined ? active : [...active, visibleTerminal];
  const retrySeconds = nextRetryMs === null ? null : nextRetryMs / 1000;

  useEffect(() => {
    let terminalTransition: JobProjection | undefined;
    let dangerTransition: JobProjection | undefined;
    for (const job of jobs) {
      if (
        isTerminal(job)
        && knownStatuses.current?.get(job.job_id) !== job.status
      ) {
        terminalTransition = job;
        if (job.status === "refused" || job.status === "error") dangerTransition = job;
      }
    }
    knownStatuses.current = new Map(jobs.map((job) => [job.job_id, job.status]));
    setEvidence((current) => {
      if (dangerTransition !== undefined) {
        return { key: identity(dangerTransition), urgent: true };
      }
      if (terminalTransition !== undefined) {
        return { key: identity(terminalTransition), urgent: false };
      }
      if (current === null) return null;
      return jobs.some((job) => identity(job) === current.key) ? current : null;
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
                  urgent={evidence !== null && evidence.urgent && evidence.key === identity(job)}
                />
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
