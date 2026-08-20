import type { OutputProjection } from "./types";

const targetLabels: Record<OutputProjection["state"], string> = {
  ready_new: "New target ready",
  blocked_existing: "Existing target blocked",
  blocked_foreign: "Foreign target blocked",
  replace_owned: "Owned target can be replaced",
  ambiguous_recovery: "Recovery needs review",
  blocked_unsafe: "Unsafe target blocked",
  unavailable: "Target unavailable",
};

export function OutputTargetCard({ output }: { output: OutputProjection }) {
  const runnable = output.state === "ready_new" || output.state === "replace_owned";
  return (
    <section aria-label="Output target">
      <h3>Target and recovery</h3>
      <p><code>{output.target_path ?? "No run target"}</code></p>
      <p>{targetLabels[output.state]}</p>
      <p
        role={runnable ? "status" : "alert"}
        aria-label="Output target state"
        aria-live={runnable ? "polite" : "assertive"}
        className={runnable ? undefined : "error-surface"}
      >
        {output.state_message}
      </p>
    </section>
  );
}
