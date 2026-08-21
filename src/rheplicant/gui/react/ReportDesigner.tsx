import { StatusChip } from "./StatusChip";
import type { EditorSession, SessionTransport } from "./types";

const REPORT_COLUMNS = ["mean", "std", "seconds"];
const RELATIVE_COLUMNS = ["mean_sigma", "width_ratio"];
const REPORT_FORMATS = ["text", "json"];

// Stable ids: one shared reason element per disabled cause, referenced by every control it
// explains, so §9 ("readable control plus adjacent reason and aria-describedby") holds without
// duplicating ids across the checkboxes that share a cause.
const LAST_ROW_REASON_ID = "report-last-row-reason";
const LAST_COLUMN_REASON_ID = "report-last-column-reason";
const LAST_FORMAT_REASON_ID = "report-last-format-reason";
const NO_REFERENCE_REASON_ID = "report-no-reference-reason";

export function describedBy(...ids: (string | undefined)[]) {
  const named = ids.filter((id): id is string => id !== undefined && id !== "");
  return named.length === 0 ? undefined : named.join(" ");
}

function toggled(values: string[], value: string, enabled: boolean) {
  return enabled ? [...values, value].filter((item, index, all) => all.indexOf(item) === index) : values.filter((item) => item !== value);
}

// The one value a selector still holds when unchecking it would empty the selection, or null when
// unchecking is legal. Identical in effect to `values.length === 1 && values.includes(candidate)`
// evaluated per rendered candidate, so no control changes its enabled state.
function heldValue(values: string[], candidates: string[]) {
  return values.length === 1 && candidates.includes(values[0]) ? values[0] : null;
}

interface Props {
  session: EditorSession;
  transport: SessionTransport;
  onRun(action: () => Promise<EditorSession>, message: string): void;
  disabled: boolean;
  disabledReasonId?: string;
}

export function ReportDesigner({ session, transport, onRun, disabled, disabledReasonId }: Props) {
  const report = session.outputs.report;
  const blockedReason = disabled ? disabledReasonId : undefined;
  const heldRow = heldValue(report.rows, session.outputs.declared_runs);
  const heldColumn = heldValue(report.columns, REPORT_COLUMNS);
  const heldFormat = heldValue(report.formats, REPORT_FORMATS);
  const noReference = report.reference === null;
  function update(values: Partial<typeof report>) {
    onRun(
      () => transport.setOutputReport(
        session.session_id, values.enabled ?? report.enabled, values.rows ?? report.rows,
        values.columns ?? report.columns, values.reference === undefined ? report.reference : values.reference,
        values.relative ?? report.relative, values.formats ?? report.formats, session.revision,
      ),
      "Updated report output",
    );
  }
  return (
    <fieldset aria-label="Report selector">
      <legend>Report table</legend>
      <fieldset>
        <legend>Rows</legend>
        {session.outputs.declared_runs.map((name) => <label key={name}><input type="checkbox" aria-label={`Report row ${name}`} checked={report.rows.includes(name)} disabled={disabled || heldRow === name} aria-describedby={describedBy(blockedReason, heldRow === name ? LAST_ROW_REASON_ID : undefined)} onChange={(event) => update({ rows: toggled(report.rows, name, event.target.checked), reference: !event.target.checked && report.reference === name ? null : report.reference, relative: !event.target.checked && report.reference === name ? [] : report.relative })} />{name}</label>)}
        {heldRow !== null && <span id={LAST_ROW_REASON_ID}><StatusChip tone="disabled" label="Report row disabled: the report keeps at least one row." /></span>}
      </fieldset>
      <fieldset aria-label="Report columns">
        <legend>Columns</legend>
        {REPORT_COLUMNS.map((name) => <label key={name}><input type="checkbox" checked={report.columns.includes(name)} disabled={disabled || heldColumn === name} aria-describedby={describedBy(blockedReason, heldColumn === name ? LAST_COLUMN_REASON_ID : undefined)} onChange={(event) => update({ columns: toggled(report.columns, name, event.target.checked) })} />{name}</label>)}
        {heldColumn !== null && <span id={LAST_COLUMN_REASON_ID}><StatusChip tone="disabled" label="Report column disabled: the report keeps at least one column." /></span>}
      </fieldset>
      <label>Reference row<select value={report.reference ?? ""} disabled={disabled} aria-describedby={blockedReason} onChange={(event) => update({ reference: event.target.value || null, relative: event.target.value ? report.relative : [] })}><option value="">none</option>{report.rows.map((name) => <option key={name}>{name}</option>)}</select></label>
      <fieldset><legend>Relative columns</legend>{RELATIVE_COLUMNS.map((name) => <label key={name}><input type="checkbox" checked={report.relative.includes(name)} disabled={disabled || noReference} aria-describedby={describedBy(blockedReason, noReference ? NO_REFERENCE_REASON_ID : undefined)} onChange={(event) => update({ relative: toggled(report.relative, name, event.target.checked) })} />{name}</label>)}
        {noReference && <span id={NO_REFERENCE_REASON_ID}><StatusChip tone="disabled" label="Relative columns disabled: choose a reference row first." /></span>}
      </fieldset>
      <fieldset><legend>Formats</legend>{REPORT_FORMATS.map((name) => <label key={name}><input type="checkbox" checked={report.formats.includes(name)} disabled={disabled || heldFormat === name} aria-describedby={describedBy(blockedReason, heldFormat === name ? LAST_FORMAT_REASON_ID : undefined)} onChange={(event) => update({ formats: toggled(report.formats, name, event.target.checked) })} />{name}</label>)}
        {heldFormat !== null && <span id={LAST_FORMAT_REASON_ID}><StatusChip tone="disabled" label="Report format disabled: the report keeps at least one format." /></span>}
      </fieldset>
      {report.expected_paths.length > 0 && <ul aria-label="Report expected paths">{report.expected_paths.map((path) => <li key={path}><code>{path}</code></li>)}</ul>}
    </fieldset>
  );
}
