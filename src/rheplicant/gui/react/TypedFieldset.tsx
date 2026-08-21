import type { NodeField, TypedFields } from "./types";

export interface TypedFieldsetProps {
  /** One operator's settings: a single-slot node's card, or one instance of
   *  a `many` node. The node above a `many` one has none of its own. */
  owner: TypedFields;
  /** What the controls are named after -- the node id for a card, the
   *  instance label for an instance. Two instances of one node would
   *  otherwise give their controls the same accessible name. */
  subject: string;
  disabled: boolean;
  shownNumber: (field: NodeField) => string;
  shownUnit: (field: NodeField) => string;
  shownForm: (field: NodeField) => string;
  shownReference: (field: NodeField) => string;
  /** Whether the value-spelling switchers are open. Off by default: a select
   *  under every field doubles the height of the set to offer something most
   *  edits never need. */
  showForms: boolean;
  onToggleForms: () => void;
  onNumber: (field: NodeField, text: string) => void;
  onUnit: (field: NodeField, unit: string) => void;
  onForm: (field: NodeField, form: string) => void;
  onReference: (field: NodeField, dotted: string) => void;
  onText: (field: NodeField, value: string) => void;
  /** A class chosen but not yet confirmed, or null. */
  pendingType: string | null;
  onChooseType: (next: string) => void;
  onConfirmType: () => void;
  onCancelType: () => void;
}

/** One operator's settings as labelled controls, or the reason there are none.
 *
 *  Purely presentational: every value it shows and every edit it reports goes
 *  through the props above, so the one draft the raw JSON textarea edits stays
 *  the only place a node's settings live.
 */
export function TypedFieldset({
  owner,
  subject,
  disabled,
  shownNumber,
  shownUnit,
  shownForm,
  shownReference,
  showForms,
  onToggleForms,
  onNumber,
  onUnit,
  onForm,
  onReference,
  onText,
  pendingType,
  onChooseType,
  onConfirmType,
  onCancelType,
}: TypedFieldsetProps) {
  const removed = pendingType === null ? [] : owner.removed_by_type[pendingType] ?? [];
  const respellable = owner.fields.some((field) => field.forms.length > 1);
  return (
    <fieldset aria-label={`${subject} typed fields`}>
      <legend>Typed fields</legend>
      {!owner.typed_form && <p>{owner.typed_form_reason}</p>}
      {owner.typed_form && owner.type_choices.length > 1 && (
        <>
          <label>
            type
            <select
              aria-label={`${subject} type`}
              value={pendingType ?? owner.selected_type ?? ""}
              disabled={disabled}
              onChange={(event) => onChooseType(event.target.value)}
            >
              {owner.selected_type === null && <option value="">not chosen</option>}
              {owner.type_choices.map((choice) => (
                <option key={choice} value={choice}>{choice}</option>
              ))}
            </select>
          </label>
          {pendingType !== null && (
            <div role="group" aria-label={`${subject} type change`}>
              <p>
                {pendingType} has no field for {removed.join(", ")}. Changing the
                type removes {removed.length > 1 ? "those values" : "that value"} from
                the draft.
              </p>
              <button type="button" onClick={onConfirmType}>
                Change {subject} to {pendingType}
              </button>
              <button type="button" onClick={onCancelType}>
                Keep {owner.selected_type ?? "the current type"}
              </button>
            </div>
          )}
        </>
      )}
      {owner.typed_form && owner.fields.map((field) => (
        <p key={field.name}>
          <label>
            {field.label}{field.required ? " (required)" : ""}
            {field.control === "resource" ? (
              <select
                aria-label={field.label}
                value={shownReference(field)}
                disabled={disabled || !field.typed || field.choices.length === 0}
                onChange={(event) => onReference(field, event.target.value)}
              >
                <option value="">not set</option>
                {field.choices.map((choice) => (
                  <option key={choice} value={choice}>{choice}</option>
                ))}
              </select>
            ) : field.control === "select" ? (
              <select
                aria-label={field.label}
                value={shownNumber(field)}
                disabled={disabled || !field.typed}
                onChange={(event) => onText(field, event.target.value)}
              >
                <option value="">not set</option>
                {field.choices.map((choice) => (
                  <option key={choice} value={choice}>{choice}</option>
                ))}
              </select>
            ) : (
              <input
                aria-label={field.label}
                type={field.control === "quantity" || field.control === "integer"
                  ? "number"
                  : "text"}
                value={shownNumber(field)}
                disabled={disabled || !field.typed}
                onChange={(event) => (
                  field.control === "quantity" || field.control === "integer"
                    ? onNumber(field, event.target.value)
                    : onText(field, event.target.value)
                )}
              />
            )}
          </label>
          {field.units.length > 1 && field.unit_policy !== "forbidden" && (
            <select
              aria-label={`${field.label} unit`}
              value={shownUnit(field)}
              disabled={disabled || !field.typed}
              onChange={(event) => onUnit(field, event.target.value)}
            >
              {field.units.map((unit) => (
                <option key={unit} value={unit}>{unit}</option>
              ))}
            </select>
          )}
          {field.control === "resource" && field.choices.length === 0 && (
            <span> no {field.resource_kind} declared: add one under resources:</span>
          )}
          {showForms && field.forms.length > 1 && (
            <select
              aria-label={`${field.label} form`}
              value={shownForm(field)}
              disabled={disabled || !field.typed}
              onChange={(event) => onForm(field, event.target.value)}
            >
              {field.forms.map((form) => (
                <option key={form} value={form}>{form}</option>
              ))}
            </select>
          )}
          {!field.typed && <span> written as {field.form}: edit it in the JSON below</span>}
          {field.help !== "" && <span className="field-help"> {field.help}</span>}
        </p>
      ))}
      {owner.typed_form && respellable && (
        <label>
          <input
            type="checkbox"
            aria-label={`${subject} value spellings`}
            checked={showForms}
            disabled={disabled}
            onChange={onToggleForms}
          />
          value spellings
        </label>
      )}
      {owner.typed_form && owner.extra_keys.length > 0 && (
        <p>Also written: {owner.extra_keys.join(", ")}. Edit those in the JSON below.</p>
      )}
    </fieldset>
  );
}
