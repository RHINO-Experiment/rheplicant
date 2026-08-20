import { useState } from "react";

import { resultLabel } from "./ResultSummary";
import type { JobProjection } from "./types";

interface JobsPanelProps {
  active: JobProjection[];
  current: JobProjection[];
  stale: JobProjection[];
  selected: string | null;
  onSelect(jobId: string): void;
}

export function shortJobId(jobId: string) {
  return jobId.length > 6 ? `${jobId.slice(0, 6)}…` : jobId;
}

function JobGroup({
  label,
  empty,
  jobs,
  selected,
  onSelect,
  onCopy,
}: {
  label: string;
  empty: string;
  jobs: JobProjection[];
  selected: string | null;
  onSelect(jobId: string): void;
  onCopy(jobId: string): void;
}) {
  return (
    <section role="group" aria-label={label}>
      <h3>{label}</h3>
      {jobs.length === 0 ? <p>{empty}</p> : (
        <ol>{jobs.map((job) => {
          const visibleId = shortJobId(job.job_id);
          return (
            <li key={job.job_id}>
              <button
                type="button"
                aria-pressed={selected === job.job_id}
                aria-label={`View job ${visibleId}`}
                onClick={() => onSelect(job.job_id)}
              >
                View
              </button>
              {" "}<code>{visibleId}</code>{" · "}<strong>{resultLabel(job)}</strong>
              {" · "}{job.kind}
              {job.stale && <p>From revision {job.revision}</p>}
              <button
                type="button"
                aria-label="Copy full job id"
                onClick={() => onCopy(job.job_id)}
              >
                Copy full job id
              </button>
            </li>
          );
        })}</ol>
      )}
    </section>
  );
}

export function JobsPanel({ active, current, stale, selected, onSelect }: JobsPanelProps) {
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  async function copy(jobId: string) {
    try {
      if (typeof navigator.clipboard?.writeText !== "function") throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(jobId);
      setCopyStatus("Full job id copied.");
    } catch {
      setCopyStatus("Could not copy the job id. Copy it from the selected result summary.");
    }
  }
  return (
    <section aria-label="Job results and history">
      <h2>Results</h2>
      <JobGroup label="Active jobs" empty="No active jobs." jobs={active} selected={selected} onSelect={onSelect} onCopy={(jobId) => void copy(jobId)} />
      <JobGroup label="Current history" empty="No current terminal jobs." jobs={current} selected={selected} onSelect={onSelect} onCopy={(jobId) => void copy(jobId)} />
      <JobGroup label="Stale history" empty="No stale terminal jobs." jobs={stale} selected={selected} onSelect={onSelect} onCopy={(jobId) => void copy(jobId)} />
      {copyStatus !== null && <p role="status" aria-live="polite">{copyStatus}</p>}
    </section>
  );
}
