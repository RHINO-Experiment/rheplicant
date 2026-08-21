import { useState } from "react";

import { ExecutionActions } from "./ExecutionActions";
import { type WorkspaceSurface } from "./ModelWorkspace";
import { OutputWorkflow } from "./OutputWorkflow";
import { ForwardPreviewSummary } from "./PreviewPanel";
import { CLOSED_PRODUCT_VIEW, type ProductView } from "./ProductSelector";
import { describedBy } from "./ReportDesigner";
import { StatusChip, type StatusTone } from "./StatusChip";
import { FULL_VALIDATION_TONE, NO_DETAIL, NO_REASON, deriveFullValidation } from "./fullValidation";
import type { EditorSession, JobKind, JobProjection, SessionTransport } from "./types";

// Execute neither reads a draft nor accepts a session of its own: every mutation it offers goes
// through onRun. `drafts` and `onAccept` were required props the hook never read, and every caller
// and harness had to construct them to satisfy a demand that was not real.
export interface ExecuteWorkspaceProps {
  session: EditorSession;
  jobs: JobProjection[];
  transport: SessionTransport;
  disabledReason: string | null;
  onSubmit(kind: JobKind): void;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

const RUN_BLOCKED_REASON_ID = "run-blocked-reason";

// One identity for "does this verdict still describe the document on screen", shared with
// fullValidation.ts and with OnboardingChecklist: the YAML digest, never the revision. The server
// defines staleness the same way, and `mark_saved` bumps the revision while leaving `yaml_text`
// — and so the digest — untouched, so `Save YAML` alone used to retire a preview that still
// described this exact document. It also made the three readers of that one question disagree:
// this one went false while the checklist still called the same job current.
function currentPreview(jobs: JobProjection[], digest: string) {
  return jobs.some((job) => job.kind === "preview_forward" && job.yaml_digest === digest && job.status === "succeeded" && !job.stale);
}

// Execute renders its own words; the state behind them comes from the one shared derivation.
// The switch is exhaustive on purpose: an eighth state must be a compile error HERE too, not a
// silent fall-through to the green "current" label an unknown state has no right to.
function fullValidation(jobs: JobProjection[], digest: string): { label: string; tone: StatusTone } {
  const derived = deriveFullValidation(jobs, digest);
  const tone = FULL_VALIDATION_TONE[derived.state];
  switch (derived.state) {
    case "not-run": return { label: "Full validation not run", tone };
    case "stale": return { label: `Full validation stale from revision ${derived.job.revision}`, tone };
    case "queued": return { label: `Full validation queued for revision ${derived.job.revision}`, tone };
    case "running": return { label: `Full validation running for revision ${derived.job.revision}`, tone };
    // The same words for an absent message as Diagnostics: one surface saying "refused" with no
    // reason while the other names the absence is the divergence this vocabulary exists to stop.
    // `||`, never `??`: the server builds this as `bounded_text(error)`, which is "" for an
    // exception raised with no args, and the frame validator only asks whether it is a string.
    // `??` let that empty string through and rendered a bare "Full validation refused: ".
    case "refused": return { label: `Full validation refused: ${derived.job.message || NO_REASON}`, tone };
    case "error": return { label: `Full validation internal error: ${derived.job.message || NO_DETAIL}`, tone };
    case "current": return { label: `Full validation current for revision ${derived.job.revision}`, tone };
    default: {
      const unreachable: never = derived;
      return unreachable;
    }
  }
}

export function useExecuteWorkspace({ session, jobs, transport, disabledReason, onSubmit, onRun }: ExecuteWorkspaceProps): WorkspaceSurface {
  const targetRunnable = session.outputs.state === "ready_new" || session.outputs.state === "replace_owned";
  const runDeclared = session.outputs.declared_runs.length > 0;
  const runBlocked = session.document.validation.run_blocked;
  // Both causes, joined, rather than the first one that happens to hold: a dirty draft and a
  // refused quick check can be true at once, and naming only the validation reason left the
  // blocker the user can actually clear undescribed on every action button.
  const actionDisabledReason = describedBy(
    disabledReason ?? undefined,
    runBlocked ? RUN_BLOCKED_REASON_ID : undefined,
  ) ?? null;
  // Execute owns its own progressive-disclosure state; the hook outlives every workspace switch,
  // so picker disclosure, its query, the active option and the expanded product all survive one.
  const [productView, setProductView] = useState<ProductView>(CLOSED_PRODUCT_VIEW);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const full = fullValidation(jobs, session.yaml_digest);
  return {
    main: (
      <section aria-label="Execute workspace">
        <section aria-label="Validation readiness">
          <h2>Validation</h2>
          <StatusChip tone={runBlocked ? "danger" : "success"} label={runBlocked ? "Quick checks need attention" : "Quick checks clean"} />
          <StatusChip tone={full.tone} label={full.label} />
          {runBlocked && <p id={RUN_BLOCKED_REASON_ID}>Run is blocked until validation is repaired.</p>}
        </section>
        <ForwardPreviewSummary previews={session.document.previews} />
        <OutputWorkflow session={session} transport={transport} disabled={disabledReason !== null} disabledReasonId={disabledReason ?? undefined} productView={productView} onProductView={setProductView} comparisonOpen={comparisonOpen} onComparisonOpen={setComparisonOpen} onRun={onRun} />
        <ExecutionActions
          jobs={jobs}
          revision={session.revision}
          yamlDigest={session.yaml_digest}
          previewCurrent={currentPreview(jobs, session.yaml_digest)}
          runDeclared={runDeclared}
          targetRunnable={targetRunnable}
          declaredKinds={session.document.previews.declared_run_kinds.filter((kind): kind is JobKind => kind === "run" || kind === "compare" || kind === "benchmark")}
          disabledReason={actionDisabledReason}
          advanced={advanced}
          onAdvanced={setAdvanced}
          onSubmit={onSubmit}
        />
      </section>
    ),
    inspector: <aside aria-label="Context inspector"><p>Review validation, previews, target, products, and report before submitting work.</p></aside>,
  };
}
