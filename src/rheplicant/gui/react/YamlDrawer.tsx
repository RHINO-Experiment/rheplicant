import { useEffect, useRef, useState } from "react";

import type { DraftEnvelope } from "./drafts";
import { StatusChip } from "./StatusChip";
import { useWorkbenchModal } from "./WorkbenchShell";

// Every copy outcome is a fixed sentence. Nothing from the rejection reaches the surface, so the
// status is bounded by construction and no error text can flood the conflict section.
type CopyOutcome = "copied" | "failed" | "unavailable";

const COPY_MESSAGE: Record<CopyOutcome, string> = {
  copied: "Copied the raw draft to the clipboard.",
  failed: "Copy failed: the clipboard refused the draft.",
  unavailable: "Copy unavailable: this browser exposes no clipboard.",
};

const COPY_STATUS_ID = "yaml-copy-status";

// The outcome is bound to the exact draft it describes. The clipboard holds one specific text, so
// a status that outlives it would claim a copy that never happened for the text now on screen —
// and §9 offers Copy beside Discard, where a trusted stale claim costs the user the newer edit.
interface CopyAttempt {
  outcome: CopyOutcome;
  text: string;
}

interface YamlDrawerProps {
  acceptedYaml: string;
  revision: number;
  draft: DraftEnvelope;
  diagnostic: string | null;
  conflict?: string | null;
  busy?: boolean;
  onChange(text: string): void;
  onApply(): void;
  onDiscard(): void;
  onClose(): void;
  onRefresh?(): void;
}

function useNewEvidence(value: string | null) {
  const previous = useRef(value);
  const [urgentValue, setUrgentValue] = useState<string | null>(null);

  useEffect(() => {
    if (previous.current !== value && value !== null) setUrgentValue(value);
    else if (value === null) setUrgentValue(null);
    previous.current = value;
  }, [value]);

  return value !== null && urgentValue === value;
}

export function YamlDrawer({
  acceptedYaml,
  revision,
  draft,
  diagnostic,
  conflict = null,
  busy = false,
  onChange,
  onApply,
  onDiscard,
  onClose,
  onRefresh,
}: YamlDrawerProps) {
  const { dialogRef, closeModal, handleModalKeyDown } = useWorkbenchModal(onClose);
  const urgentDiagnostic = useNewEvidence(diagnostic);
  const urgentConflict = useNewEvidence(conflict);
  const [copyAttempt, setCopyAttempt] = useState<CopyAttempt | null>(null);

  const yamlDraft = draft.kind === "yaml" ? draft : null;
  const text = yamlDraft?.text ?? acceptedYaml;
  // Only an attempt on the draft still on screen has anything to say about it. An edit, a discard
  // or a refreshed accepted revision retires the status without an effect having to notice.
  const copyOutcome = copyAttempt !== null && copyAttempt.text === text ? copyAttempt.outcome : null;

  // `text` is the exact value the textarea renders: never trimmed, normalised or re-serialised.
  async function copyDraft(): Promise<void> {
    const attempted = text;
    const clipboard = navigator.clipboard;
    if (!clipboard) {
      setCopyAttempt({ outcome: "unavailable", text: attempted });
      return;
    }
    try {
      await clipboard.writeText(attempted);
      setCopyAttempt({ outcome: "copied", text: attempted });
    } catch {
      setCopyAttempt({ outcome: "failed", text: attempted });
    }
  }

  return (
    <aside
      ref={dialogRef}
      aria-label="YAML drawer"
      role="dialog"
      aria-modal="true"
      onKeyDown={handleModalKeyDown}
    >
      <header>
        <h2>YAML source of truth</h2>
        <button type="button" aria-label="Close YAML drawer" onClick={closeModal}>Close</button>
      </header>
      <p>Accepted revision {revision}</p>
      <textarea
        aria-label="YAML source of truth"
        aria-invalid={diagnostic !== null}
        aria-describedby={diagnostic ? "yaml-diagnostic" : undefined}
        value={text}
        disabled={busy || (draft.kind !== "none" && draft.kind !== "yaml")}
        onChange={(event) => onChange(event.target.value)}
      />
      <button
        type="button"
        disabled={busy || yamlDraft === null}
        onClick={onApply}
      >
        Apply YAML edit
      </button>
      {yamlDraft && <button type="button" disabled={busy} onClick={onDiscard}>Discard draft</button>}
      {diagnostic && (
        <span id="yaml-diagnostic">
          <StatusChip
            key={urgentDiagnostic ? `urgent:${diagnostic}` : "diagnostic"}
            tone="danger"
            label={`Invalid YAML: ${diagnostic}`}
            urgent={urgentDiagnostic}
          />
        </span>
      )}
      {conflict && (
        <section aria-label="YAML revision conflict" className="error-surface">
          <StatusChip
            key={urgentConflict ? `urgent:${conflict}` : "conflict"}
            tone="danger"
            label={`Revision conflict: ${conflict}`}
            urgent={urgentConflict}
          />
          <p>Draft base revision {yamlDraft?.baseRevision}; accepted revision {revision}.</p>
          <button type="button" aria-describedby={copyOutcome === null ? undefined : COPY_STATUS_ID} onClick={() => { void copyDraft(); }}>Copy draft</button>
          {onRefresh && <button type="button" disabled={busy} onClick={onRefresh}>Refresh accepted YAML</button>}
          {copyOutcome !== null && (
            <span id={COPY_STATUS_ID}>
              <StatusChip
                tone={copyOutcome === "copied" ? "success" : "warning"}
                label={COPY_MESSAGE[copyOutcome]}
              />
            </span>
          )}
        </section>
      )}
    </aside>
  );
}
