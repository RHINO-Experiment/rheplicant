import { StatusChip, type StatusTone } from "./StatusChip";
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

// Defence in depth, never a substitute: the backend bounds the same things by
// the same numbers in _rheplicant_bootstrap/gui_limits.py. These mirror it so
// one unbounded payload cannot reach the DOM even if it reaches the browser.
//
// Nearly every server string this module renders passes through boundedText: the
// job message, the four finding fields, the output state message and target
// path, the array dtype and statistic, and the tap and uniform-sky probe names.
// The output and waterfall paths were the exception once, and one megabyte in
// state_message plus one in target_path put 2,000,140 characters in the DOM.
//
// Two server strings still reach the DOM unbounded: `job_id` and `kind`. Both
// come from the job row rather than from a result payload — the id is a uuid the
// server mints and the kind is one of a closed set — so neither is a channel an
// oversized payload can travel down. Everything else left unbounded is a number
// or a member of the fixed `outputStates` set.
const TRUNCATION_MARKER = "…[truncated]";
const MAX_RENDERED_CHARACTERS = 4000;
const MAX_RENDERED_FINDINGS = 100;

// Counted in code points, as Python's len() counts them, so the JS bound and the
// backend bound in gui_limits.py mean the same thing for non-BMP text — and so a
// cut never lands between the halves of a surrogate pair and emits a lone
// surrogate. UTF-16 length is an upper bound on the code-point count, which is
// what makes the cheap first test sound.
function boundedText(value: string) {
  if (value.length <= MAX_RENDERED_CHARACTERS) return value;
  const points = [...value];
  if (points.length <= MAX_RENDERED_CHARACTERS) return value;
  return points.slice(0, MAX_RENDERED_CHARACTERS - TRUNCATION_MARKER.length).join("")
    + TRUNCATION_MARKER;
}

function rawFindings(job: JobProjection): unknown[] {
  if (!isRecord(job.result) || !Array.isArray(job.result.findings)) return [];
  return job.result.findings;
}

/**
 * Everything the payload carried that the DOM does not show: the rows past the cap AND the rows
 * inside it too malformed to render. Counting the overflow alone under-reports exactly the rows
 * the user has no other way of learning about.
 */
function droppedFindings(job: JobProjection, rendered: number) {
  return Math.max(0, rawFindings(job).length - rendered);
}

/**
 * One predicate for "is this row a finding at all", so what the list renders and what the
 * headline label reports can never disagree about which rows count.
 */
function isFindingRow(candidate: unknown): candidate is Finding {
  return isRecord(candidate)
    && typeof candidate.check === "string"
    && (candidate.severity === "refuse"
      || candidate.severity === "warn"
      || candidate.severity === "report")
    && typeof candidate.where === "string"
    && typeof candidate.message === "string"
    && typeof candidate.layer === "string";
}

function findings(job: JobProjection): Finding[] {
  // Sliced before the map: mapping the whole array is the unbounded step.
  return rawFindings(job).slice(0, MAX_RENDERED_FINDINGS).flatMap((candidate): Finding[] => {
    if (!isFindingRow(candidate)) return [];
    return [{
      check: boundedText(candidate.check),
      severity: candidate.severity,
      where: boundedText(candidate.where),
      message: boundedText(candidate.message),
      layer: boundedText(candidate.layer),
    }];
  });
}

/**
 * Asked of the WHOLE payload, never of the rendered slice. MAX_RENDERED_FINDINGS bounds what the
 * DOM holds; it does not bound what the job means. Reading the slice labelled a succeeded job
 * whose only `warn` sat at index 150 a green "Succeeded" and withheld §9's corrective action,
 * "Review the warning before continuing." — the one instruction that state exists to carry.
 * Scanning costs a severity test per row and puts nothing extra in the DOM.
 */
function hasWarning(job: JobProjection) {
  return rawFindings(job).some((candidate) => isFindingRow(candidate) && candidate.severity === "warn");
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

export function resultTone(job: JobProjection): StatusTone {
  const label = resultLabel(job);
  if (label === "Unsafe target" || label === "Internal error" || label === "Refused") {
    return "danger";
  }
  if (label === "Warning") return "warning";
  if (job.status === "succeeded") return "success";
  return "neutral";
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
  const dtype = typeof value.dtype === "string" ? boundedText(value.dtype) : null;
  const statistic = typeof value.statistic === "string" ? boundedText(value.statistic) : null;
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
  const rawTaps = isRecord(result.taps) ? Object.entries(result.taps) : [];
  const taps = rawTaps.slice(0, MAX_PREVIEW_DIMENSION).flatMap(([name, value]) => {
    const summary = arraySummary(value, false);
    return summary === null ? [] : [[name, summary] as const];
  });
  const rawUniform = isRecord(result.uniform_sky_mean) ? Object.entries(result.uniform_sky_mean) : [];
  const uniform = rawUniform.slice(0, MAX_PREVIEW_DIMENSION)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number");
  // The same evidence the finding list already gives: MAX_PREVIEW_DIMENSION is 64 while the
  // backend forwards up to 256, so a preview with 200 taps dropped 136 of them and said nothing
  // at all — silence a reader has no way to tell apart from "the model declared 64 taps".
  // Counted like droppedFindings: rows past the cap AND rows inside it too malformed to render.
  const droppedTaps = Math.max(0, rawTaps.length - taps.length);
  const droppedUniform = Math.max(0, rawUniform.length - uniform.length);
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
      {/* The key stays the raw name: two names that truncate alike must not collide as one row. */}
      {taps.length > 0 && <ul aria-label="Forward taps">{taps.map(([name, tap]) => (
        <li key={name}>{boundedText(name)}: {tap.shape?.join(" × ") ?? "unknown"} {tap.dtype ?? "unknown"}</li>
      ))}</ul>}
      {droppedTaps > 0 && (
        <p>{`${droppedTaps} further forward taps were not rendered ${TRUNCATION_MARKER}`}</p>
      )}
      {uniform.length > 0 && <ul aria-label="Uniform-sky probes">{uniform.map(([name, value]) => (
        <li key={name}>{boundedText(name)}: {value}</li>
      ))}</ul>}
      {droppedUniform > 0 && (
        <p>{`${droppedUniform} further uniform-sky probes were not rendered ${TRUNCATION_MARKER}`}</p>
      )}
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
  const stateMessage = typeof found.state_message === "string" ? boundedText(found.state_message) : null;
  const target = typeof found.target_path === "string" ? boundedText(found.target_path) : null;
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
  const dropped = droppedFindings(job, knownFindings.length);
  const result = isRecord(job.result) ? job.result : null;
  const exitCode = result !== null && typeof result.exit_code === "number"
    ? result.exit_code
    : null;
  const help = recovery(job);
  const label = resultLabel(job);
  const tone = resultTone(job);
  return (
    <section aria-label="Result summary">
      <h2>Job result</h2>
      <StatusChip tone={tone} label={label} />
      <p>Job <code>{job.job_id}</code></p>
      <p>{job.kind} · revision {job.revision}</p>
      {job.stale && (
        <StatusChip tone="stale" label={`From revision ${job.revision}`} />
      )}
      {job.message !== null && <p>{boundedText(job.message)}</p>}
      {help !== null && <p>{help}</p>}
      {exitCode !== null && <p>Exit code {exitCode}</p>}
      {/* The count is evidence in its own right: a payload whose every row was malformed renders
          no list at all, and silence there reads as "nothing was found". */}
      {(knownFindings.length > 0 || dropped > 0) && (
        <section aria-label="Job findings">
          <h3>Findings</h3>
          {knownFindings.length > 0 && <ol>{knownFindings.map((finding, index) => (
            <li key={`${finding.layer}:${finding.where}:${finding.check}:${index}`}>
              <strong>{finding.severity}</strong> {finding.check}
              <span> · {finding.layer} · {finding.where}</span>
              <p>{finding.message}</p>
            </li>
          ))}</ol>}
          {dropped > 0 && (
            <p>{`${dropped} further findings were not rendered ${TRUNCATION_MARKER}`}</p>
          )}
        </section>
      )}
      <WaterfallSummary result={job.result} />
      <OutputSummary job={job} />
    </section>
  );
}
