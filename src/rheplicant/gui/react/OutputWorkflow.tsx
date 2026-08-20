import { useRef, useState } from "react";

import type {
  EditorSession,
  JobProjection,
  OutputProductProjection,
  SessionTransport,
} from "./types";

interface Props {
  session: EditorSession;
  transport: SessionTransport;
  onAccept: (next: EditorSession, message: string) => void;
  disabled?: boolean;
  disabledReasonId?: string;
  onRun?: (action: () => Promise<EditorSession>, message: string) => void;
}

function toggled(values: string[], value: string, enabled: boolean) {
  return enabled
    ? [...values, value].filter((item, index, all) => all.indexOf(item) === index)
    : values.filter((item) => item !== value);
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

interface AuditBundle {
  job: JobProjection;
  files: string[];
}

function auditBundles(jobs: JobProjection[]): AuditBundle[] {
  return jobs.flatMap((job) => {
    if (job.status === "queued" || job.status === "running") return [];
    const output = record(record(job.result)?.output);
    const files = Array.isArray(output?.audit_files)
      ? output.audit_files.filter((item): item is string => typeof item === "string")
      : [];
    return files.length > 0 ? [{ job, files }] : [];
  });
}

function ProductSelector({
  row,
  runs,
  disabled,
  apply,
  disabledReasonId,
}: {
  row: OutputProductProjection;
  runs: string[];
  disabled: boolean;
  disabledReasonId?: string;
  apply: (
    enabled: boolean,
    format: string,
    selectedRuns: string[],
    keys: string[],
    themes: string[],
  ) => void;
}) {
  function submit(values: Partial<OutputProductProjection>) {
    apply(
      values.enabled ?? row.enabled,
      values.format ?? row.format,
      values.runs ?? row.runs,
      values.keys ?? row.keys,
      values.themes ?? row.themes,
    );
  }

  return (
    <fieldset className="output-product">
      <legend>{row.name}</legend>
      <label>
        <input
          type="checkbox"
          aria-label={`Write ${row.name}`}
          checked={row.enabled}
          disabled={disabled}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onChange={(event) => submit({ enabled: event.target.checked })}
        />
        Write {row.name}
      </label>
      <label>
        Format
        <select
          aria-label={`${row.name} format`}
          value={row.format}
          disabled={disabled}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onChange={(event) => submit({ format: event.target.value })}
        >
          {row.formats.map((format) => (
            <option key={format} value={format}>{format}</option>
          ))}
        </select>
      </label>
      {runs.length > 0 && row.name !== "assembly" && row.name !== "signal_paths" && (
        <fieldset>
          <legend>Runs</legend>
          {runs.map((run) => (
            <label key={run}>
              <input
                type="checkbox"
                aria-label={`${row.name} run ${run}`}
                checked={row.runs.includes(run)}
                disabled={disabled}
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => submit({
                  runs: toggled(row.runs, run, event.target.checked),
                })}
              />
              {run}
            </label>
          ))}
        </fieldset>
      )}
      {(row.name === "aux" || row.name === "taps") && (
        <label>
          Keys, comma-separated
          <input
            aria-label={`${row.name} keys`}
            value={row.keys.join(", ")}
            disabled={disabled}
            aria-describedby={disabled ? disabledReasonId : undefined}
            onChange={(event) => submit({
              keys: event.target.value.split(",").map((item) => item.trim()).filter(Boolean),
            })}
          />
        </label>
      )}
      {row.name === "signal_paths" && (
        <fieldset>
          <legend>Themes</legend>
          {["light", "dark"].map((theme) => (
            <label key={theme}>
              <input
                type="checkbox"
                aria-label={`signal_paths theme ${theme}`}
                checked={row.themes.includes(theme)}
                disabled={disabled}
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => submit({
                  themes: toggled(row.themes, theme, event.target.checked),
                })}
              />
              {theme}
            </label>
          ))}
        </fieldset>
      )}
      {row.expected_paths.length > 0 && (
        <ul aria-label={`${row.name} expected paths`}>
          {row.expected_paths.map((path) => <li key={path}><code>{path}</code></li>)}
        </ul>
      )}
    </fieldset>
  );
}

export function OutputWorkflow({
  session,
  transport,
  onAccept,
  disabled = false,
  disabledReasonId,
  onRun,
}: Props) {
  const [tab, setTab] = useState<"requested" | "resolved">("requested");
  const [error, setError] = useState<string | null>(null);
  const requestedTab = useRef<HTMLButtonElement>(null);
  const resolvedTab = useRef<HTMLButtonElement>(null);
  const output = session.outputs;
  const report = output.report;
  const bundles = auditBundles(session.jobs);

  async function run(action: () => Promise<EditorSession>, message: string) {
    if (onRun) {
      onRun(action, message);
      return;
    }
    try {
      const next = await action();
      setError(null);
      onAccept(next, message);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  function product(
    row: OutputProductProjection,
    enabled: boolean,
    format: string,
    runs: string[],
    keys: string[],
    themes: string[],
  ) {
    void run(
      () => transport.setOutputProduct(
        session.session_id,
        row.name,
        enabled,
        format,
        runs,
        keys,
        themes,
        session.revision,
      ),
      `Updated ${row.name} output`,
    );
  }

  function updateReport(values: {
    enabled?: boolean;
    rows?: string[];
    columns?: string[];
    reference?: string | null;
    relative?: string[];
    formats?: string[];
  }) {
    void run(
      () => transport.setOutputReport(
        session.session_id,
        values.enabled ?? report.enabled,
        values.rows ?? report.rows,
        values.columns ?? report.columns,
        values.reference === undefined ? report.reference : values.reference,
        values.relative ?? report.relative,
        values.formats ?? report.formats,
        session.revision,
      ),
      "Updated report output",
    );
  }

  const targetRole = output.state === "ready_new" || output.state === "replace_owned"
    ? undefined
    : "alert";

  function moveTab(event: React.KeyboardEvent<HTMLButtonElement>) {
    const next = event.key === "ArrowRight" || event.key === "ArrowDown" || event.key === "End"
      ? "resolved"
      : event.key === "ArrowLeft" || event.key === "ArrowUp" || event.key === "Home"
        ? "requested"
        : null;
    if (next === null) return;
    event.preventDefault();
    setTab(next);
    (next === "requested" ? requestedTab : resolvedTab).current?.focus();
  }

  return (
    <section aria-label="Output workflow" className="output-workflow">
      <h2>Outputs and audit trail</h2>
      <div role="tablist" aria-label="Configuration artefacts">
        <button
          ref={requestedTab}
          id="requested-tab"
          role="tab"
          aria-selected={tab === "requested"}
          aria-controls="config-artefact-panel"
          tabIndex={tab === "requested" ? 0 : -1}
          onClick={() => setTab("requested")}
          onKeyDown={moveTab}
        >
          What I asked for
        </button>
        <button
          ref={resolvedTab}
          id="resolved-tab"
          role="tab"
          aria-selected={tab === "resolved"}
          aria-controls="config-artefact-panel"
          tabIndex={tab === "resolved" ? 0 : -1}
          onClick={() => setTab("resolved")}
          onKeyDown={moveTab}
        >
          What will run
        </button>
      </div>
      <div
        id="config-artefact-panel"
        role="tabpanel"
        aria-labelledby={tab === "requested" ? "requested-tab" : "resolved-tab"}
      >
        <pre><code>{tab === "requested" ? output.requested_yaml : output.resolved_yaml}</code></pre>
        {tab === "resolved" && <p>{output.resolution_note}</p>}
      </div>

      <section aria-label="Output target">
        <h3>Target and recovery</h3>
        <p><code>{output.target_path ?? "No run target"}</code></p>
        <p
          role={targetRole}
          aria-label="Output target state"
          aria-live={targetRole === undefined ? "polite" : "assertive"}
          className={targetRole === "alert" ? "error-surface" : undefined}
        >
          {output.state_message}
        </p>
      </section>

      <section aria-label="Scientific product selectors">
        <h3>Products</h3>
        <div className="product-grid">
          {output.products.map((row) => (
            <ProductSelector
              key={row.name}
              row={row}
              runs={output.declared_runs}
              disabled={disabled}
              disabledReasonId={disabledReasonId}
              apply={(...values) => product(row, ...values)}
            />
          ))}
        </div>
      </section>

      <fieldset aria-label="Report selector">
        <legend>Report table</legend>
        <label>
          <input
          type="checkbox"
          checked={report.enabled}
          disabled={disabled || (!report.enabled && output.declared_runs.length === 0)}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onChange={(event) => updateReport({
            enabled: event.target.checked,
            rows: event.target.checked && report.rows.length === 0
              ? output.declared_runs.slice(0, 1)
              : report.rows,
          })}
          />
          Write report
        </label>
        <fieldset>
          <legend>Rows</legend>
          {output.declared_runs.map((name) => (
            <label key={name}>
              <input
                type="checkbox"
                aria-label={`Report row ${name}`}
                checked={report.rows.includes(name)}
                disabled={
                  disabled
                  || !report.enabled
                  || (report.rows.length === 1 && report.rows.includes(name))
                }
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => updateReport({
                  rows: toggled(report.rows, name, event.target.checked),
                  reference: !event.target.checked && report.reference === name
                    ? null
                    : report.reference,
                  relative: !event.target.checked && report.reference === name
                    ? []
                    : report.relative,
                })}
              />
              {name}
            </label>
          ))}
        </fieldset>
        <fieldset>
          <legend>Columns</legend>
          {["mean", "std", "seconds"].map((name) => (
            <label key={name}>
              <input
                type="checkbox"
                checked={report.columns.includes(name)}
                disabled={
                  disabled
                  || !report.enabled
                  || (report.columns.length === 1 && report.columns.includes(name))
                }
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => updateReport({
                  columns: toggled(report.columns, name, event.target.checked),
                })}
              />
              {name}
            </label>
          ))}
        </fieldset>
        <label>
          Reference row
          <select
            value={report.reference ?? ""}
            disabled={disabled || !report.enabled}
            aria-describedby={disabled ? disabledReasonId : undefined}
            onChange={(event) => updateReport({
              reference: event.target.value || null,
              relative: event.target.value ? report.relative : [],
            })}
          >
            <option value="">none</option>
            {report.rows.map((name) => <option key={name}>{name}</option>)}
          </select>
        </label>
        <fieldset>
          <legend>Relative columns</legend>
          {["mean_sigma", "width_ratio"].map((name) => (
            <label key={name}>
              <input
                type="checkbox"
                checked={report.relative.includes(name)}
                disabled={disabled || !report.enabled || report.reference === null}
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => updateReport({
                  relative: toggled(report.relative, name, event.target.checked),
                })}
              />
              {name}
            </label>
          ))}
        </fieldset>
        <fieldset>
          <legend>Formats</legend>
          {["text", "json"].map((name) => (
            <label key={name}>
              <input
                type="checkbox"
                checked={report.formats.includes(name)}
                disabled={
                  disabled
                  || !report.enabled
                  || (report.formats.length === 1 && report.formats.includes(name))
                }
                aria-describedby={disabled ? disabledReasonId : undefined}
                onChange={(event) => updateReport({
                  formats: toggled(report.formats, name, event.target.checked),
                })}
              />
              {name}
            </label>
          ))}
        </fieldset>
        {report.expected_paths.length > 0 && (
          <ul aria-label="Report expected paths">
            {report.expected_paths.map((path) => <li key={path}><code>{path}</code></li>)}
          </ul>
        )}
      </fieldset>

      <section aria-label="Completed audit bundles">
        <h3>Completed audit bundles</h3>
        {bundles.length === 0 ? <p>No completed audit bundle is available.</p> : (
          <ul>
            {bundles.map(({ job, files }) => (
              <li key={job.job_id}>
                <strong>{job.job_id}</strong>
                <ul>
                  {files.map((file) => (
                    <li key={file}>
                      <a
                        href={`/api/sessions/${encodeURIComponent(session.session_id)}/jobs/${encodeURIComponent(job.job_id)}/artifacts/${encodeURIComponent(file)}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {file}
                      </a>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </section>
      {error && <p role="alert" className="error-surface">{error}</p>}
    </section>
  );
}
