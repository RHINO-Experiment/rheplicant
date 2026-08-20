import type { ReactNode } from "react";

import { SecurityBoundaryNotice } from "./SecurityBoundaryNotice";

interface WorkbenchHeaderProps {
  dirty: boolean;
  validationStale: boolean;
  revision: number;
  mutationBlocked: boolean;
  mutationReason: string | null;
  yamlBlocked: boolean;
  onOpenYaml(): void;
  actions: ReactNode;
}

export function WorkbenchHeader({
  dirty,
  validationStale,
  revision,
  mutationBlocked,
  mutationReason,
  yamlBlocked,
  onOpenYaml,
  actions,
}: WorkbenchHeaderProps) {
  return (
    <>
      <h1>Rheplicant configuration workbench</h1>
      <p>YAML is the sole scientific state; controls are projections.</p>
      <SecurityBoundaryNotice />
      <div aria-label="Editor session state">
        <strong>{dirty ? "Unsaved changes" : "Saved"}</strong>
        <span>{validationStale ? "Validation stale" : "Validation current"}</span>
        <span>Revision {revision}</span>
      </div>
      {mutationBlocked && mutationReason && <p id="mutation-blocked-reason">{mutationReason}</p>}
      <nav aria-label="History and file actions">
        <button type="button" disabled={yamlBlocked} onClick={onOpenYaml}>YAML</button>
        {actions}
      </nav>
    </>
  );
}
