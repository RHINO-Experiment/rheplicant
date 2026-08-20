import { useEffect, useRef, useState } from "react";

import type { DraftEnvelope } from "./drafts";
import { StatusChip } from "./StatusChip";
import { useWorkbenchModal } from "./WorkbenchShell";

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

  const yamlDraft = draft.kind === "yaml" ? draft : null;
  const text = yamlDraft?.text ?? acceptedYaml;

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
          <button type="button" onClick={() => void navigator.clipboard?.writeText(text)}>Copy draft</button>
          {onRefresh && <button type="button" disabled={busy} onClick={onRefresh}>Refresh accepted YAML</button>}
        </section>
      )}
    </aside>
  );
}
