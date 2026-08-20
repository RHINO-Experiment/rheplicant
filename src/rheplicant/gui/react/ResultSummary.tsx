import type { JobProjection } from "./types";

type Finding = {
  check: string;
  severity: "refuse" | "warn" | "report";
  where: string;
  message: string;
  layer: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function output(job: JobProjection) {
  if (!isRecord(job.result) || !isRecord(job.result.output)) return null;
  return job.result.output;
}

function findings(job: JobProjection): Finding[] {
  if (!isRecord(job.result) || !Array.isArray(job.result.findings)) return [];
  return job.result.findings.flatMap((candidate): Finding[] => {
    if (
      !isRecord(candidate)
      || typeof candidate.check !== "string"
      || (candidate.severity !== "refuse"
        && candidate.severity !== "warn"
        && candidate.severity !== "report")
      || typeof candidate.where !== "string"
      || typeof candidate.message !== "string"
      || typeof candidate.layer !== "string"
    ) return [];
    return [{
      check: candidate.check,
      severity: candidate.severity,
      where: candidate.where,
      message: candidate.message,
      layer: candidate.layer,
    }];
  });
}

function hasWarning(job: JobProjection) {
  return findings(job).some((finding) => finding.severity === "warn");
}

export function resultLabel(job: JobProjection) {
  if (output(job)?.state === "blocked_unsafe") return "Unsafe target";
  if (job.status === "error") return "Internal error";
  if (job.status === "refused") return "Refused";
  if (hasWarning(job)) return "Warning";
  return {
    queued: "Queued",
    running: "Running",
    succeeded: "Succeeded",
  }[job.status];
}

function recovery(job: JobProjection) {
  const label = resultLabel(job);
  if (label === "Unsafe target") return "Choose a safe server output target before re-running.";
  if (label === "Internal error") return "Retry the job or report this internal failure.";
  if (label === "Refused") return "Correct the accepted configuration before re-running.";
  if (label === "Warning") return "Review the warning before continuing.";
  return null;
}

type ArraySummary = {
  shape: number[] | null;
  dtype: string | null;
  statistic: string | null;
  minimum: number | null;
  maximum: number | null;
  mean: number | null;
  values: number[][];
};

const MAX_PREVIEW_DIMENSION = 64;

function arraySummary(value: unknown, includeValues: boolean): ArraySummary | null {
  if (!isRecord(value)) return null;
  const shape = Array.isArray(value.shape)
    && value.shape.length <= 4
    && value.shape.every((item) => typeof item === "number")
    ? value.shape
    : null;
  const dtype = typeof value.dtype === "string" ? value.dtype : null;
  const statistic = typeof value.statistic === "string" ? value.statistic : null;
  const number = (key: "minimum" | "maximum" | "mean") => (
    typeof value[key] === "number" ? value[key] : null
  );
  const rawValues = includeValues && Array.isArray(value.values)
    && value.values.length <= MAX_PREVIEW_DIMENSION
    && value.values.every((row) => Array.isArray(row)
      && row.length <= MAX_PREVIEW_DIMENSION
      && row.every((cell) => typeof cell === "number"))
    ? value.values as number[][]
    : [];
  const summary = {
    shape,
    dtype,
    statistic,
    minimum: number("minimum"),
    maximum: number("maximum"),
    mean: number("mean"),
    values: rawValues,
  };
  return Object.values(summary).every((item) => item === null || Array.isArray(item) && item.length === 0)
    ? null
    : summary;
}

function WaterfallSummary({ result }: { result: unknown }) {
  if (!isRecord(result)) return null;
  const waterfall = arraySummary(result.waterfall, true);
  if (waterfall === null) return null;
  const numbers = (["minimum", "maximum", "mean"] as const).flatMap((key) => (
    waterfall[key] === null ? [] : [[key, waterfall[key]] as const]
  ));
  const taps = isRecord(result.taps)
    ? Object.entries(result.taps).slice(0, MAX_PREVIEW_DIMENSION).flatMap(([name, value]) => {
      const summary = arraySummary(value, false);
      return summary === null ? [] : [[name, summary] as const];
    })
    : [];
  const uniform = isRecord(result.uniform_sky_mean)
    ? Object.entries(result.uniform_sky_mean).slice(0, MAX_PREVIEW_DIMENSION)
      .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    : [];
  const saturation = typeof result.saturated_fraction === "number"
    ? `${(100 * result.saturated_fraction).toFixed(2)}%`
    : "unavailable";
  return (
    <section aria-label="Waterfall summary">
      <h3>Waterfall</h3>
      {waterfall.shape !== null && <p>Shape {waterfall.shape.join(" × ")}</p>}
      {waterfall.dtype !== null && <p>Data type {waterfall.dtype}</p>}
      {waterfall.statistic !== null && <p>Statistic {waterfall.statistic}</p>}
      {numbers.length > 0 && <dl>{numbers.map(([key, value]) => (
        <div key={key}><dt>{key}</dt><dd>{` ${value}`}</dd></div>
      ))}</dl>}
      {waterfall.values.length > 0 && (
        <table aria-label="Predicted waterfall">
          <tbody>{waterfall.values.map((row, rowIndex) => (
            <tr key={rowIndex}>{row.map((cell, columnIndex) => (
              <td key={columnIndex}>{cell}</td>
            ))}</tr>
          ))}</tbody>
        </table>
      )}
      <p>saturation {saturation}</p>
      {taps.length > 0 && <ul aria-label="Forward taps">{taps.map(([name, tap]) => (
        <li key={name}>{name}: {tap.shape?.join(" × ") ?? "unknown"} {tap.dtype ?? "unknown"}</li>
      ))}</ul>}
      {uniform.length > 0 && <ul aria-label="Uniform-sky probes">{uniform.map(([name, value]) => (
        <li key={name}>{name}: {value}</li>
      ))}</ul>}
    </section>
  );
}

const outputStates = new Set([
  "ready_new",
  "blocked_existing",
  "blocked_foreign",
  "replace_owned",
  "ambiguous_recovery",
  "blocked_unsafe",
  "unavailable",
]);

function OutputSummary({ job }: { job: JobProjection }) {
  const found = output(job);
  if (found === null) return null;
  const state = typeof found.state === "string" && outputStates.has(found.state)
    ? found.state
    : null;
  const stateMessage = typeof found.state_message === "string" ? found.state_message : null;
  const target = typeof found.target_path === "string" ? found.target_path : null;
  if (state === null && stateMessage === null && target === null) return null;
  return (
    <section aria-label="Published output summary">
      <h3>Output publication</h3>
      {state !== null && <p>State {state}</p>}
      {stateMessage !== null && <p>{stateMessage}</p>}
      {target !== null && <p><code>{target}</code></p>}
    </section>
  );
}

export function ResultSummary({ job }: { job: JobProjection }) {
  const knownFindings = findings(job);
  const result = isRecord(job.result) ? job.result : null;
  const exitCode = result !== null && typeof result.exit_code === "number"
    ? result.exit_code
    : null;
  const help = recovery(job);
  return (
    <section aria-label="Result summary">
      <h2>{resultLabel(job)}</h2>
      <p>Job <code>{job.job_id}</code></p>
      <p>{job.kind} · revision {job.revision}</p>
      {job.stale && <p>From revision {job.revision}</p>}
      {job.message !== null && <p role={job.status === "succeeded" ? undefined : "alert"}>{job.message}</p>}
      {help !== null && <p>{help}</p>}
      {exitCode !== null && <p>Exit code {exitCode}</p>}
      {knownFindings.length > 0 && (
        <section aria-label="Job findings">
          <h3>Findings</h3>
          <ol>{knownFindings.map((finding, index) => (
            <li key={`${finding.layer}:${finding.where}:${finding.check}:${index}`}>
              <strong>{finding.severity}</strong> {finding.check}
              <span> · {finding.layer} · {finding.where}</span>
              <p>{finding.message}</p>
            </li>
          ))}</ol>
        </section>
      )}
      <WaterfallSummary result={job.result} />
      <OutputSummary job={job} />
    </section>
  );
}
