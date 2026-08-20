import { ExecutionActions } from "./ExecutionActions";
import { type WorkspaceSurface } from "./ModelWorkspace";
import { OutputWorkflow } from "./OutputWorkflow";
import { ForwardPreviewSummary } from "./PreviewPanel";
import type { DraftCoordinator } from "./drafts";
import type { EditorSession, JobKind, JobProjection, SessionTransport } from "./types";

export interface ExecuteWorkspaceProps {
  session: EditorSession;
  jobs: JobProjection[];
  transport: SessionTransport;
  drafts: DraftCoordinator;
  disabledReason: string | null;
  onAccept(next: EditorSession, message: string): void;
  onSubmit(kind: JobKind): void;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

function currentPreview(jobs: JobProjection[], revision: number, digest: string) {
  return jobs.some((job) => job.kind === "preview_forward" && job.revision === revision && job.yaml_digest === digest && job.status === "succeeded" && !job.stale);
}

export function useExecuteWorkspace({ session, jobs, transport, disabledReason, onSubmit, onRun }: ExecuteWorkspaceProps): WorkspaceSurface {
  const targetRunnable = session.outputs.state === "ready_new" || session.outputs.state === "replace_owned";
  const runDeclared = session.outputs.declared_runs.length > 0;
  const runBlocked = session.document.validation.run_blocked;
  const actionDisabledReason = runBlocked ? "run-blocked-reason" : disabledReason;
  return {
    main: (
      <section aria-label="Execute workspace">
        <section aria-label="Validation readiness"><h2>Validation</h2><p id={runBlocked ? "run-blocked-reason" : undefined}>{runBlocked ? "Run is blocked until validation is repaired." : "Quick and full validation are ready."}</p></section>
        <ForwardPreviewSummary previews={session.document.previews} />
        <OutputWorkflow session={session} transport={transport} disabled={disabledReason !== null} disabledReasonId={disabledReason ?? undefined} onRun={onRun} />
        <ExecutionActions
          jobs={jobs}
          revision={session.revision}
          yamlDigest={session.yaml_digest}
          previewCurrent={currentPreview(jobs, session.revision, session.yaml_digest)}
          runDeclared={runDeclared}
          targetRunnable={targetRunnable}
          declaredKinds={session.document.previews.declared_run_kinds.filter((kind): kind is JobKind => kind === "run" || kind === "compare" || kind === "benchmark")}
          disabledReason={actionDisabledReason}
          onSubmit={onSubmit}
        />
      </section>
    ),
    inspector: <aside aria-label="Context inspector"><p>Review validation, previews, target, products, and report before submitting work.</p></aside>,
  };
}
