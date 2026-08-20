import type { DraftEnvelope } from "./drafts";

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
  const yamlDraft = draft.kind === "yaml" ? draft : null;
  const text = yamlDraft?.text ?? acceptedYaml;

  return (
    <aside aria-label="YAML drawer" role="dialog" aria-modal="true">
      <header>
        <h2>YAML source of truth</h2>
        <button type="button" aria-label="Close YAML drawer" onClick={onClose}>Close</button>
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
        <p id="yaml-diagnostic" role="alert" aria-label="YAML parse diagnostic" className="error-surface">
          {diagnostic}
        </p>
      )}
      {conflict && (
        <section aria-label="YAML revision conflict" className="error-surface">
          <p role="alert">{conflict}</p>
          <p>Draft base revision {yamlDraft?.baseRevision}; accepted revision {revision}.</p>
          <button type="button" onClick={() => void navigator.clipboard?.writeText(text)}>Copy draft</button>
          {onRefresh && <button type="button" disabled={busy} onClick={onRefresh}>Refresh accepted YAML</button>}
        </section>
      )}
    </aside>
  );
}
