import type {
  JobKind,
  JobProjection,
  PreviewProjection,
} from "./types";

interface Props {
  previews: PreviewProjection;
  jobs: JobProjection[];
  disabled: boolean;
  blocked: boolean;
  disabledReasonId?: string;
  onSubmit: (kind: JobKind) => void;
}

function values(items: number[], unit: string | null) {
  return `${items.join(", ")}${unit ? ` ${unit}` : ""}`;
}

function jobLabel(kind: JobKind) {
  return {
    validate: "Validate",
    preview_forward: "Preview forward",
    run: "Run",
    compare: "Compare",
    benchmark: "Benchmark",
  }[kind];
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function shape(value: unknown) {
  return Array.isArray(value) && value.every((item) => typeof item === "number")
    ? value.join(" × ")
    : "unknown";
}

function ValidationResult({ result }: { result: unknown }) {
  const found = record(result);
  const findings = Array.isArray(found?.findings) ? found.findings : null;
  if (findings === null) return null;
  return findings.length === 0 ? <p>Validation ledger: no findings.</p> : (
    <ul aria-label="Priced validation ledger">
      {findings.map((value, index) => {
        const finding = record(value) ?? {};
        return (
          <li key={`${String(finding.check)}-${index}`}>
            {[finding.check, finding.severity, finding.where, finding.layer, finding.message]
              .map(String).join(" · ")}
          </li>
        );
      })}
    </ul>
  );
}

function ForwardResult({ result }: { result: unknown }) {
  const found = record(result);
  const waterfall = record(found?.waterfall);
  if (waterfall === null) return null;
  const values = Array.isArray(waterfall.values) ? waterfall.values : [];
  const taps = record(found?.taps) ?? {};
  const uniform = record(found?.uniform_sky_mean) ?? {};
  const saturation = typeof found?.saturated_fraction === "number"
    ? `${(100 * found.saturated_fraction).toFixed(2)}%`
    : "unavailable";
  return (
    <div>
      <p>
        waterfall {shape(waterfall.shape)} · {String(waterfall.dtype)} · range {String(waterfall.minimum)}…{String(waterfall.maximum)} · mean {String(waterfall.mean)}
      </p>
      {values.length > 0 && (
        <table aria-label="Predicted waterfall">
          <tbody>
            {values.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {(Array.isArray(row) ? row : []).map((cell, columnIndex) => (
                  <td key={columnIndex}>{String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p>saturation {saturation}</p>
      {Object.keys(taps).length > 0 && (
        <ul aria-label="Forward taps">
          {Object.entries(taps).map(([name, value]) => {
            const tap = record(value) ?? {};
            return <li key={name}>{name}: {shape(tap.shape)} {String(tap.dtype)}</li>;
          })}
        </ul>
      )}
      {Object.keys(uniform).length > 0 && (
        <ul aria-label="Uniform-sky probes">
          {Object.entries(uniform).map(([name, value]) => (
            <li key={name}>{name}: {String(value)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function PreviewPanel({
  previews,
  jobs,
  disabled,
  blocked,
  disabledReasonId,
  onSubmit,
}: Props) {
  const unavailable = disabled || blocked;
  const declared = new Set(previews.declared_run_kinds);

  return (
    <section aria-label="Preview and execution controls">
      <section aria-label="Continuous axis and shape previews">
        <h2>Continuous previews</h2>
        <p>The signal-path graph, axes, and shapes update without evaluating the model.</p>
        {previews.axes.length === 0 ? (
          <p>No text-declared axes are available.</p>
        ) : (
          <dl>
            {previews.axes.map((axis) => (
              <div key={axis.axis}>
                <dt>{axis.axis}</dt>
                <dd>
                  {values(axis.first, axis.unit)} … {values(axis.last, axis.unit)}
                  {` · count ${axis.count}`}
                  {axis.spacing !== null && ` · spacing ${axis.spacing}`}
                  {axis.precision_ok !== null && (
                    <span>
                      {` · precision ${axis.precision_ok ? "safe" : "unsafe"}`}
                    </span>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        )}
        <ul>
          {previews.shapes.map((shape) => (
            <li key={shape.symbol}><code>{shape.symbol}</code> = {shape.value}</li>
          ))}
        </ul>
      </section>

      <section aria-label="Explicit preview and run actions">
        <h2>Explicit, priced work</h2>
        <button disabled={unavailable} aria-describedby={disabled ? disabledReasonId : undefined} onClick={() => onSubmit("validate")}>
          Validate
        </button>
        <button
          aria-label={`Preview forward · ${previews.forward_cost.label}`}
          disabled={unavailable}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onClick={() => onSubmit("preview_forward")}
        >
          Preview forward · {previews.forward_cost.label}
        </button>
        <button disabled={unavailable} aria-describedby={disabled ? disabledReasonId : undefined} onClick={() => onSubmit("run")}>Run</button>
        <button
          disabled={unavailable || !declared.has("compare")}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onClick={() => onSubmit("compare")}
        >
          Compare
        </button>
        <button
          disabled={unavailable || !declared.has("benchmark")}
          aria-describedby={disabled ? disabledReasonId : undefined}
          onClick={() => onSubmit("benchmark")}
        >
          Benchmark
        </button>
        <p>Fitting is always a Run job; it is never executed as a preview.</p>
      </section>

      <section aria-label="Explicit jobs">
        <h2>Jobs</h2>
        {jobs.length === 0 ? (
          <p>No jobs submitted.</p>
        ) : (
          <ol>
            {jobs.map((job) => (
              <li key={job.job_id}>
                <strong>{jobLabel(job.kind)}</strong>
                {` · ${job.job_id} · ${job.status}`}
                {job.stale && " · stale"}
                {job.message && <p role="alert">{job.message}</p>}
                {job.kind === "validate" && <ValidationResult result={job.result} />}
                {job.kind === "preview_forward" && <ForwardResult result={job.result} />}
              </li>
            ))}
          </ol>
        )}
      </section>
    </section>
  );
}
