import { useState } from "react";

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
  onSubmit(kind: JobKind): void;
}

function identicalActiveJob(jobs: JobProjection[], kind: JobKind, revision: number, digest: string) {
  return jobs.some((job) => job.kind === kind && job.revision === revision && job.yaml_digest === digest && (job.status === "queued" || job.status === "running"));
}

export function ExecutionActions({ jobs, revision, yamlDigest, previewCurrent, runDeclared, targetRunnable, declaredKinds, disabledReason, onSubmit }: ExecutionActionsProps) {
  const [advanced, setAdvanced] = useState(false);
  const blocked = disabledReason !== null;
  const unavailable = (kind: JobKind) => blocked || identicalActiveJob(jobs, kind, revision, yamlDigest);
  const runUnavailable = unavailable("run") || !runDeclared || !targetRunnable;
  const primary = previewCurrent && !runUnavailable ? "run" : "preview_forward";
  const advancedKinds = declaredKinds.filter((kind): kind is "compare" | "benchmark" => kind === "compare" || kind === "benchmark");
  return (
    <section aria-label="Execution actions">
      <h2>Actions</h2>
      <button type="button" disabled={unavailable("validate")} aria-describedby={disabledReason ?? undefined} onClick={() => onSubmit("validate")}>Validate</button>
      <button type="button" className={primary === "preview_forward" ? "primary-action" : undefined} disabled={unavailable("preview_forward")} aria-describedby={disabledReason ?? undefined} onClick={() => onSubmit("preview_forward")}>Preview forward</button>
      <button type="button" className={primary === "run" ? "primary-action" : undefined} disabled={runUnavailable} aria-describedby={disabledReason ?? undefined} onClick={() => onSubmit("run")}>Run</button>
      {advancedKinds.length > 0 && <button type="button" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}>Advanced actions</button>}
      {advanced && advancedKinds.map((kind) => <button key={kind} type="button" disabled={unavailable(kind)} aria-describedby={disabledReason ?? undefined} onClick={() => onSubmit(kind)}>{kind[0].toUpperCase()}{kind.slice(1)}</button>)}
    </section>
  );
}
