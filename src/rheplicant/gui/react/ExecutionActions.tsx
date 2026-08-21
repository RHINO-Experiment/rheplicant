import { StatusChip } from "./StatusChip";
import type { JobKind, JobProjection } from "./types";

export interface ExecutionActionsProps {
  jobs: JobProjection[];
  revision: number;
  yamlDigest: string;
  previewCurrent: boolean;
  runDeclared: boolean;
  targetRunnable: boolean;
  declaredKinds: JobKind[];
  disabledReason: string | null;
  advanced: boolean;
  onAdvanced(next: boolean): void;
  onSubmit(kind: JobKind): void;
}

function identicalActiveJob(jobs: JobProjection[], kind: JobKind, revision: number, digest: string) {
  return jobs.some((job) => job.kind === kind && job.revision === revision && job.yaml_digest === digest && (job.status === "queued" || job.status === "running"));
}

function activeJob(jobs: JobProjection[], kind: JobKind, revision: number, digest: string) {
  return jobs.find((job) => job.kind === kind
    && job.revision === revision
    && job.yaml_digest === digest
    && (job.status === "queued" || job.status === "running"));
}

function actionLabel(kind: JobKind) {
  return kind === "preview_forward"
    ? "Preview forward"
    : `${kind[0].toUpperCase()}${kind.slice(1)}`;
}

export function ExecutionActions({ jobs, revision, yamlDigest, previewCurrent, runDeclared, targetRunnable, declaredKinds, disabledReason, advanced, onAdvanced, onSubmit }: ExecutionActionsProps) {
  const blocked = disabledReason !== null;
  const unavailable = (kind: JobKind) => blocked || identicalActiveJob(jobs, kind, revision, yamlDigest);
  const runUnavailable = unavailable("run") || !runDeclared || !targetRunnable;
  const primary = previewCurrent && !runUnavailable ? "run" : "preview_forward";
  // Deduplicated: a kind declared twice would render two buttons keyed alike and two spans sharing
  // one `execution-<kind>-active` id, so aria-describedby would name an ambiguous element.
  const advancedKinds = [...new Set(declaredKinds.filter((kind): kind is "compare" | "benchmark" => kind === "compare" || kind === "benchmark"))];
  const kinds: JobKind[] = ["validate", "preview_forward", "run", ...advancedKinds];
  const active = kinds.flatMap((kind) => {
    const job = activeJob(jobs, kind, revision, yamlDigest);
    return job === undefined ? [] : [job];
  });
  const description = (kind: JobKind) => {
    if (disabledReason !== null) return disabledReason;
    if (identicalActiveJob(jobs, kind, revision, yamlDigest)) return `execution-${kind}-active`;
    if (kind === "run" && (!runDeclared || !targetRunnable)) return "execution-run-disabled";
    return undefined;
  };
  return (
    <section aria-label="Execution actions">
      <h2>Actions</h2>
      <button type="button" disabled={unavailable("validate")} aria-describedby={description("validate")} onClick={() => onSubmit("validate")}>Validate</button>
      <button type="button" className={primary === "preview_forward" ? "primary-action" : undefined} disabled={unavailable("preview_forward")} aria-describedby={description("preview_forward")} onClick={() => onSubmit("preview_forward")}>Preview forward</button>
      <button type="button" className={primary === "run" ? "primary-action" : undefined} disabled={runUnavailable} aria-describedby={description("run")} onClick={() => onSubmit("run")}>Run</button>
      {advancedKinds.length > 0 && <button type="button" onClick={() => onAdvanced(!advanced)} aria-expanded={advanced}>Advanced actions</button>}
      {advanced && advancedKinds.map((kind) => <button key={kind} type="button" disabled={unavailable(kind)} aria-describedby={description(kind)} onClick={() => onSubmit(kind)}>{actionLabel(kind)}</button>)}
      {active.map((job) => (
        <span key={job.kind} id={`execution-${job.kind}-active`}>
          <StatusChip
            tone="neutral"
            label={`${job.status === "queued" ? "Queued" : "Running"} ${actionLabel(job.kind)} at revision ${job.revision}`}
          />
        </span>
      ))}
      {!runDeclared && (
        <span id="execution-run-disabled">
          <StatusChip tone="disabled" label="Run disabled: no run is declared." />
        </span>
      )}
      {runDeclared && !targetRunnable && (
        <span id="execution-run-disabled">
          <StatusChip tone="disabled" label="Run disabled: repair the output target." />
        </span>
      )}
    </section>
  );
}
