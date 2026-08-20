import { useState } from "react";

import type { DraftCoordinator } from "./drafts";
import type { EditorSession, ProjectedWidget, SessionTransport } from "./types";

const primitiveWidgets = new Set(["select", "toggle", "integer", "text", "file"]);

export interface FieldEditorProps {
  widget: ProjectedWidget;
  session: EditorSession;
  transport: SessionTransport;
  drafts: DraftCoordinator;
  disabled: boolean;
  disabledReason: string | null;
  onAccept(next: EditorSession, message: string): void;
  onEditYaml(path: string): void;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

function acceptedRawValue(widget: ProjectedWidget): string {
  if (!widget.present || widget.value === null || widget.value === undefined) return "";
  if (typeof widget.value === "boolean") return widget.value ? "true" : "false";
  return String(widget.value);
}

export function primitiveValue(widget: ProjectedWidget, rawValue: string): unknown {
  if (widget.widget === "select") {
    if (!widget.choices.includes(rawValue)) throw new Error("Choose a declared value");
    return rawValue;
  }
  if (widget.widget === "toggle") {
    if (rawValue !== "true" && rawValue !== "false") throw new Error("Choose true or false");
    return rawValue === "true";
  }
  if (widget.widget === "integer") {
    if (!/^-?(0|[1-9]\d*)$/.test(rawValue)) throw new Error("Enter a whole number");
    const parsed = Number(rawValue);
    if (!Number.isSafeInteger(parsed)) throw new Error("Whole number is outside the safe range");
    return parsed;
  }
  return rawValue;
}

export function FieldEditor({
  widget,
  session,
  transport,
  drafts,
  disabled,
  disabledReason,
  onEditYaml,
  onRun,
}: FieldEditorProps) {
  const [error, setError] = useState<string | null>(null);
  const owner = drafts.draft.kind === "field" && drafts.draft.path === widget.path
    ? drafts.draft
    : null;
  const blockedByDraft = drafts.draft.kind !== "none" && owner === null;
  const readOnly = disabled || widget.disabled || blockedByDraft;
  const rawValue = owner?.rawValue ?? acceptedRawValue(widget);

  function update(raw: string) {
    if (readOnly) return;
    setError(null);
    if (owner) {
      drafts.update({ ...owner, rawValue: raw });
      return;
    }
    if (raw === acceptedRawValue(widget)) return;
    drafts.begin({
      kind: "field",
      baseRevision: session.revision,
      path: widget.path,
      rawValue: raw,
    });
  }

  function apply() {
    if (!owner || readOnly) return;
    let value: unknown;
    try {
      value = primitiveValue(widget, owner.rawValue);
    } catch (candidate) {
      setError(candidate instanceof Error ? candidate.message : String(candidate));
      return;
    }
    setError(null);
    onRun(
      () => transport.setField(
        session.session_id,
        widget.path,
        value,
        false,
        owner.baseRevision,
      ),
      `Updated ${widget.path}`,
    );
  }

  if (!primitiveWidgets.has(widget.widget)) {
    return (
      <button
        type="button"
        disabled={readOnly}
        aria-describedby={disabledReason ?? undefined}
        onClick={() => onEditYaml(widget.path)}
      >
        Edit in YAML
      </button>
    );
  }

  let control;
  if (widget.widget === "select") {
    control = (
      <label>
        {widget.label}
        <select
          value={rawValue}
          disabled={readOnly}
          aria-describedby={disabledReason ?? undefined}
          onChange={(event) => update(event.target.value)}
        >
          {!widget.choices.includes(rawValue) && <option value="">Choose…</option>}
          {widget.choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
        </select>
      </label>
    );
  } else if (widget.widget === "toggle") {
    control = (
      <label>
        <input
          type="checkbox"
          checked={rawValue === "true"}
          disabled={readOnly}
          aria-describedby={disabledReason ?? undefined}
          onChange={(event) => update(event.target.checked ? "true" : "false")}
        />
        {widget.label}
      </label>
    );
  } else {
    const label = widget.widget === "file" ? "Server path" : widget.label;
    control = (
      <label>
        {label}
        <input
          type="text"
          inputMode={widget.widget === "integer" ? "numeric" : undefined}
          value={rawValue}
          disabled={readOnly}
          aria-describedby={disabledReason ?? undefined}
          onChange={(event) => update(event.target.value)}
        />
      </label>
    );
  }

  return (
    <section aria-label={`${widget.label} editor`}>
      {control}
      {error && <p role="alert">{error}</p>}
      {owner && (
        <p>
          <button type="button" disabled={readOnly} onClick={apply}>Apply field</button>
          <button
            type="button"
            disabled={disabled}
            aria-describedby={disabledReason ?? undefined}
            onClick={() => { setError(null); drafts.clear(); }}
          >
            Discard field
          </button>
        </p>
      )}
    </section>
  );
}
