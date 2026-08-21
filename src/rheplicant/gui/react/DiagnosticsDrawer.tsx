import { ValidationLedger } from "./ValidationLedger";
import { StatusChip, type StatusTone } from "./StatusChip";
import { useWorkbenchModal } from "./WorkbenchShell";
import { FULL_VALIDATION_TONE, NO_DETAIL, NO_REASON, deriveFullValidation, type FullValidation } from "./fullValidation";
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

// Diagnostics renders its own words; the state behind them comes from the one shared derivation.
// The switch is exhaustive on purpose: an eighth state must be a compile error HERE too, not a
// silent fall-through to the green "current" label an unknown state has no right to.
function validationState(full: FullValidation): {
  label: string;
  tone: StatusTone;
} {
  const tone = FULL_VALIDATION_TONE[full.state];
  switch (full.state) {
    case "not-run": return { label: "Not run for this YAML", tone };
    case "stale": return { label: "Stale for this YAML", tone };
    case "queued": return { label: "Queued", tone };
    case "running": return { label: "Running", tone };
    // The same words for an absent message as Execute: a bare "Refused" left the user unable to
    // tell a reason withheld from a reason never reported.
    // `||`, never `??`: `bounded_text(error)` is "" for an exception raised with no args and the
    // frame validator only asks whether it is a string, so `??` rendered a bare "Refused · ".
    case "refused": return { label: `Refused · ${full.job.message || NO_REASON}`, tone };
    case "error": return { label: `Internal error · ${full.job.message || NO_DETAIL}`, tone };
    case "current": return { label: `Current for revision ${full.job.revision}`, tone };
    default: {
      const unreachable: never = full;
      return unreachable;
    }
  }
}

export function DiagnosticsDrawer({
  session,
  onOpenConfigPath,
  onOpenYamlPath,
  onClose,
}: DiagnosticsDrawerProps) {
  const { dialogRef, closeModal, handleModalKeyDown } = useWorkbenchModal(onClose);

  const fullValidation = deriveFullValidation(session.jobs, session.yaml_digest);
  const projectedPaths = new Set(session.document.forms.sections.flatMap(
    (section) => section.widgets.map((widget) => widget.path),
  ));
  const followFinding = (finding: LedgerFinding) => {
    if (projectedPaths.has(finding.where)) onOpenConfigPath(finding.where);
    else onOpenYamlPath(finding.where);
  };
  const fullProjection: ValidationProjection = {
    findings: fullFindings(fullValidation.job),
    section_badges: [],
    selected_presets: [],
    preset_changes: [],
    run_blocked: false,
  };
  const fullState = validationState(fullValidation);

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
