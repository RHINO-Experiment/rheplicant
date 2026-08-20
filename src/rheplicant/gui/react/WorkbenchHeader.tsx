import type { ReactNode } from "react";

interface WorkbenchHeaderProps {
  dirty: boolean;
  validationStale: boolean;
  revision: number;
  mutationBlocked: boolean;
  mutationReason: string | null;
  onOpenYaml(): void;
  actions: ReactNode;
}

export function WorkbenchHeader({
  dirty,
  validationStale,
  revision,
  mutationBlocked,
  mutationReason,
  onOpenYaml,
  actions,
}: WorkbenchHeaderProps) {
  return (
    <>
      <h1>Rheplicant configuration workbench</h1>
      <p>YAML is the sole scientific state; controls are projections.</p>
      <div aria-label="Editor session state">
        <strong>{dirty ? "Unsaved changes" : "Saved"}</strong>
        <span>{validationStale ? "Validation stale" : "Validation current"}</span>
        <span>Revision {revision}</span>
      </div>
      {mutationBlocked && mutationReason && <p id="mutation-blocked-reason">{mutationReason}</p>}
      <nav aria-label="History and file actions">
        <button type="button" onClick={onOpenYaml}>YAML</button>
        {actions}
      </nav>
    </>
  );
}
