import { ValidationLedger } from "./ValidationLedger";
import { StatusChip, type StatusTone } from "./StatusChip";
import { useWorkbenchModal } from "./WorkbenchShell";
import type {
  EditorSession,
  JobProjection,
  LedgerFinding,
  ValidationProjection,
} from "./types";

interface DiagnosticsDrawerProps {
  session: EditorSession;
  onOpenConfigPath(path: string): void;
  onOpenYamlPath(path: string): void;
  onClose(): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function fullFindings(job: JobProjection | null): LedgerFinding[] {
  if (!job || !isRecord(job.result) || !Array.isArray(job.result.findings)) return [];
  return job.result.findings.flatMap((candidate): LedgerFinding[] => {
    if (!isRecord(candidate)) return [];
    const severity = candidate.severity;
    if (severity !== "refuse" && severity !== "warn" && severity !== "report") return [];
    if (typeof candidate.where !== "string" || typeof candidate.message !== "string") return [];
    const attribution = typeof candidate.layer === "string"
      ? candidate.layer
      : typeof candidate.attribution === "string" ? candidate.attribution : "base";
    return [{
      check: typeof candidate.check === "string" ? candidate.check : "",
      severity,
      where: candidate.where,
      message: candidate.message,
      attribution,
    }];
  });
}

function latestValidation(session: EditorSession): JobProjection | null {
  const validations = session.jobs.filter((job) => job.kind === "validate");
  const matching = validations.filter((job) => !job.stale);
  return matching.at(-1) ?? validations.at(-1) ?? null;
}

function validationState(job: JobProjection | null): {
  label: string;
  tone: StatusTone;
} {
  if (!job) return { label: "Not run for this YAML", tone: "neutral" };
  if (job.stale) return { label: "Stale for this YAML", tone: "stale" };
  if (job.status === "queued") return { label: "Queued", tone: "neutral" };
  if (job.status === "running") return { label: "Running", tone: "neutral" };
  if (job.status === "refused") return {
    label: `Refused${job.message ? ` · ${job.message}` : ""}`,
    tone: "danger",
  };
  if (job.status === "error") return {
    label: `Internal error${job.message ? ` · ${job.message}` : ""}`,
    tone: "danger",
  };
  return { label: `Current for revision ${job.revision}`, tone: "success" };
}

export function DiagnosticsDrawer({
  session,
  onOpenConfigPath,
  onOpenYamlPath,
  onClose,
}: DiagnosticsDrawerProps) {
  const { dialogRef, closeModal, handleModalKeyDown } = useWorkbenchModal(onClose);

  const validationJob = latestValidation(session);
  const projectedPaths = new Set(session.document.forms.sections.flatMap(
    (section) => section.widgets.map((widget) => widget.path),
  ));
  const followFinding = (finding: LedgerFinding) => {
    if (projectedPaths.has(finding.where)) onOpenConfigPath(finding.where);
    else onOpenYamlPath(finding.where);
  };
  const fullProjection: ValidationProjection = {
    findings: fullFindings(validationJob),
    section_badges: [],
    selected_presets: [],
    preset_changes: [],
    run_blocked: false,
  };
  const fullState = validationState(validationJob);

  return (
    <aside
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label="Diagnostics"
      onKeyDown={handleModalKeyDown}
    >
      <header>
        <h2>Diagnostics</h2>
        <button type="button" onClick={closeModal}>Close diagnostics</button>
      </header>
      <section aria-label="Quick checks">
        <h2>Quick checks</h2>
        <p>Accepted schema projection for revision {session.revision}.</p>
        <ValidationLedger
          validation={session.document.validation}
          showPresetDiff={false}
          onFinding={followFinding}
        />
      </section>
      <section aria-label="Full validation">
        <h2>Full validation</h2>
        <StatusChip
          tone={fullState.tone}
          label={fullState.label}
        />
        {fullProjection.findings.length > 0 ? (
          <ValidationLedger
            validation={fullProjection}
            ledgerLabel="Full finding ledger"
            heading="Full findings"
            showPresetDiff={false}
            onFinding={followFinding}
          />
        ) : <p>No full-validation findings available.</p>}
      </section>
      <section aria-label="Diagnostics preset diff">
        <ValidationLedger
          validation={{ ...session.document.validation, findings: [] }}
          showFindings={false}
          onFinding={followFinding}
        />
      </section>
    </aside>
  );
}
