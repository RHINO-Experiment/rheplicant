import { useState } from "react";

import { AuditBundles } from "./AuditBundles";
import { JobsPanel } from "./JobsPanel";
import type { WorkspaceSurface } from "./ModelWorkspace";
import { ResultSummary } from "./ResultSummary";
import type { JobKind, JobProjection } from "./types";

export interface ResultsWorkspaceProps {
  jobs: JobProjection[];
  revision: number;
  yamlDigest: string;
  disabledReason: string | null;
  onSubmit(kind: JobKind): void;
}

function isActive(job: JobProjection) {
  return job.status === "queued" || job.status === "running";
}

export function hasIdenticalActiveJob(
  jobs: JobProjection[],
  kind: JobKind,
  revision: number,
  yamlDigest: string,
) {
  return jobs.some((job) => job.kind === kind
    && job.revision === revision
    && job.yaml_digest === yamlDigest
    && isActive(job));
}

function jobLabel(kind: JobKind) {
  return {
    validate: "Validate",
    preview_forward: "Preview forward",
    run: "Run",
    compare: "Compare",
    benchmark: "Benchmark",
  }[kind];
}

export function useResultsWorkspace({
  jobs,
  revision,
  yamlDigest,
  disabledReason,
  onSubmit,
}: ResultsWorkspaceProps): WorkspaceSurface {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const selected = jobs.find((job) => job.job_id === selectedJobId) ?? null;
  const active = jobs.filter(isActive);
  const current = jobs.filter((job) => !job.stale && !isActive(job));
  const stale = jobs.filter((job) => job.stale && !isActive(job));
  const duplicate = selected !== null
    && hasIdenticalActiveJob(jobs, selected.kind, revision, yamlDigest);
  const rerunReason = disabledReason ?? (duplicate
    ? `An identical ${jobLabel(selected.kind)} job is already active for revision ${revision}.`
    : null);
  const rerunReasonId = rerunReason === null ? undefined : "results-rerun-disabled-reason";

  return {
    main: (
      <JobsPanel
        active={active}
        current={current}
        stale={stale}
        selected={selectedJobId}
        onSelect={setSelectedJobId}
      />
    ),
    inspector: selected === null ? (
      <aside aria-label="Context inspector"><p>Select a job</p></aside>
    ) : (
      <aside aria-label="Context inspector">
        <ResultSummary job={selected} />
        <AuditBundles job={selected} />
        {selected.stale && (
          <>
            {rerunReason !== null && <p id={rerunReasonId}>{rerunReason}</p>}
            <button
              type="button"
              disabled={rerunReason !== null}
              aria-describedby={rerunReasonId}
              onClick={() => onSubmit(selected.kind)}
            >
              Re-run {jobLabel(selected.kind)}
            </button>
          </>
        )}
      </aside>
    ),
  };
}
