import type { EditorSession, SessionTransport } from "./types";

function toggled(values: string[], value: string, enabled: boolean) {
  return enabled ? [...values, value].filter((item, index, all) => all.indexOf(item) === index) : values.filter((item) => item !== value);
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
        {session.outputs.declared_runs.map((name) => <label key={name}><input type="checkbox" aria-label={`Report row ${name}`} checked={report.rows.includes(name)} disabled={disabled || (report.rows.length === 1 && report.rows.includes(name))} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update({ rows: toggled(report.rows, name, event.target.checked), reference: !event.target.checked && report.reference === name ? null : report.reference, relative: !event.target.checked && report.reference === name ? [] : report.relative })} />{name}</label>)}
      </fieldset>
      <fieldset aria-label="Report columns">
        <legend>Columns</legend>
        {["mean", "std", "seconds"].map((name) => <label key={name}><input type="checkbox" checked={report.columns.includes(name)} disabled={disabled || (report.columns.length === 1 && report.columns.includes(name))} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update({ columns: toggled(report.columns, name, event.target.checked) })} />{name}</label>)}
      </fieldset>
      <label>Reference row<select value={report.reference ?? ""} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update({ reference: event.target.value || null, relative: event.target.value ? report.relative : [] })}><option value="">none</option>{report.rows.map((name) => <option key={name}>{name}</option>)}</select></label>
      <fieldset><legend>Relative columns</legend>{["mean_sigma", "width_ratio"].map((name) => <label key={name}><input type="checkbox" checked={report.relative.includes(name)} disabled={disabled || report.reference === null} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update({ relative: toggled(report.relative, name, event.target.checked) })} />{name}</label>)}</fieldset>
      <fieldset><legend>Formats</legend>{["text", "json"].map((name) => <label key={name}><input type="checkbox" checked={report.formats.includes(name)} disabled={disabled || (report.formats.length === 1 && report.formats.includes(name))} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update({ formats: toggled(report.formats, name, event.target.checked) })} />{name}</label>)}</fieldset>
      {report.expected_paths.length > 0 && <ul aria-label="Report expected paths">{report.expected_paths.map((path) => <li key={path}><code>{path}</code></li>)}</ul>}
    </fieldset>
  );
}
