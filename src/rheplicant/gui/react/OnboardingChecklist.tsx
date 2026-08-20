import { useEffect, useState } from "react";

import type { JobProjection } from "./types";

interface Props {
  missingRequired: string[];
  runBlocked: boolean;
  jobs: JobProjection[];
}

export function OnboardingChecklist({ missingRequired, runBlocked, jobs }: Props) {
  const currentForward = jobs.some((job) => (
    job.kind === "preview_forward" && job.status === "succeeded" && !job.stale
  ));
  const staleForward = jobs.some((job) => (
    job.kind === "preview_forward" && job.status === "succeeded" && job.stale
  ));
  const [open, setOpen] = useState(() => !currentForward);

  useEffect(() => {
    if (currentForward) setOpen(false);
  }, [currentForward]);

  if (!open) {
    return <button type="button" onClick={() => setOpen(true)}>Help</button>;
  }

  const requiredLabel = missingRequired.length === 0
    ? "Required choices complete"
    : `${missingRequired.length} required choices remain`;
  const validationLabel = runBlocked ? "Quick checks need attention" : "Quick checks clean";
  const forwardLabel = currentForward
    ? "Forward preview complete"
    : staleForward
      ? "Forward preview stale"
      : "Forward preview not run";
  const nextAction = missingRequired.length > 0
    ? "Next: complete required choices."
    : runBlocked
      ? "Next: resolve quick checks."
      : currentForward
        ? "Next: continue in the workbench."
        : "Next: run Preview forward.";

  return (
    <section aria-label="First preview checklist">
      <h2>First preview</h2>
      <ul>
        <li>{requiredLabel}</li>
        <li>{validationLabel}</li>
        <li>{forwardLabel}</li>
      </ul>
      <p>{nextAction}</p>
      <button type="button" onClick={() => setOpen(false)}>Dismiss setup guide</button>
    </section>
  );
}
