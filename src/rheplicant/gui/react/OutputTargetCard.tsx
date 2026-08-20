import { useEffect, useRef, useState } from "react";

import { StatusChip, type StatusTone } from "./StatusChip";
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

const outputTones: Record<OutputProjection["state"], StatusTone> = {
  ready_new: "success",
  blocked_existing: "warning",
  replace_owned: "warning",
  blocked_foreign: "danger",
  ambiguous_recovery: "danger",
  blocked_unsafe: "danger",
  unavailable: "danger",
};

export function outputTone(state: OutputProjection["state"]): StatusTone {
  return outputTones[state];
}

export function OutputTargetCard({ output }: { output: OutputProjection }) {
  const tone = outputTone(output.state);
  const targetPath = output.target_path ?? "No run target";
  const evidenceIdentity = JSON.stringify([
    output.state,
    output.state_message,
    output.target_path,
  ]);
  const previousEvidence = useRef(evidenceIdentity);
  const [urgentEvidence, setUrgentEvidence] = useState<string | null>(null);

  useEffect(() => {
    if (
      tone === "danger"
      && previousEvidence.current !== evidenceIdentity
    ) setUrgentEvidence(evidenceIdentity);
    else if (tone !== "danger") setUrgentEvidence(null);
    previousEvidence.current = evidenceIdentity;
  }, [evidenceIdentity, tone]);

  const urgent = tone === "danger" && urgentEvidence === evidenceIdentity;

  return (
    <section aria-label="Output target">
      <h3>Target and recovery</h3>
      <StatusChip
        key={urgent ? evidenceIdentity : "output-state"}
        tone={tone}
        label={<>
          <span>{targetLabels[output.state]}</span>
          {" — "}
          <span>{output.state_message}</span>
          {" Target: "}<code>{targetPath}</code>
        </>}
        urgent={urgent}
      />
    </section>
  );
}
