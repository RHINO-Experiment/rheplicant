import type { StatusTone } from "./StatusChip";
import type { JobProjection } from "./types";

/**
 * The one derivation of the §7.5 Full-validation vocabulary.
 *
 * Execute and Diagnostics each render their own words for these states, but both read the state
 * from here: two derivations of the same seven states drifted apart once already and let one
 * surface call a document refused while the other called the same job, at the same instant, stale.
 */
export type FullValidationState =
  | "not-run"
  | "queued"
  | "running"
  | "current"
  | "stale"
  | "refused"
  | "error";

export const FULL_VALIDATION_TONE: Record<FullValidationState, StatusTone> = {
  "not-run": "neutral",
  queued: "neutral",
  running: "neutral",
  current: "success",
  stale: "stale",
  refused: "danger",
  error: "danger",
};

/**
 * The words for a refusal or an internal error the server reported without one. Shared for the
 * same reason the states are: Execute invented these while Diagnostics rendered a bare "Refused",
 * so one surface named the absence and the other left the user to guess at it.
 */
export const NO_REASON = "no reason reported";
export const NO_DETAIL = "no detail reported";

/**
 * The state and the job it was read from, as ONE discriminated union rather than two independent
 * fields: "not-run" is the only state without a job, so every other state carries a job the
 * compiler can dereference with no null check at all.
 *
 * Both surfaces used to open with `if (job === null) return <the not-run label>` ahead of a switch
 * that already handled "not-run" — the same invariant written twice in prose, in two places, where
 * nothing could keep the two copies honest. The type carries it now.
 */
export type FullValidation =
  | { state: "not-run"; job: null }
  | { state: Exclude<FullValidationState, "not-run">; job: JobProjection };

/**
 * A validate job is stale when the server says so OR when its own YAML identity no longer matches
 * the document on screen. The flag alone is not trusted: a projection built before the document
 * moved carries a clear flag and a digest that has already been left behind.
 *
 * The identity is the YAML digest, not the revision: the server defines staleness the same way
 * (`stale=row.yaml_digest != current_digest` in gui/jobs.py), and a revision that advances without
 * changing the YAML — an undo back to the same text, an output-only edit — leaves a verdict that
 * still describes this document.
 */
function isStale(job: JobProjection, yamlDigest: string): boolean {
  return job.stale || job.yaml_digest !== yamlDigest;
}

/**
 * Select the last non-stale validate job, falling back to the last validate job overall, and
 * report its state. Staleness is answered first: a superseded job carries no news about this
 * document, whatever status it reached before the document moved.
 */
export function deriveFullValidation(jobs: JobProjection[], yamlDigest: string): FullValidation {
  const validations = jobs.filter((candidate) => candidate.kind === "validate");
  const job = validations.filter((candidate) => !isStale(candidate, yamlDigest)).at(-1)
    ?? validations.at(-1)
    ?? null;
  if (job === null) return { state: "not-run", job: null };
  if (isStale(job, yamlDigest)) return { state: "stale", job };
  if (job.status === "queued") return { state: "queued", job };
  if (job.status === "running") return { state: "running", job };
  if (job.status === "refused") return { state: "refused", job };
  if (job.status === "error") return { state: "error", job };
  return { state: "current", job };
}
