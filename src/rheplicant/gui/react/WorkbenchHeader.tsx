import type { ReactNode } from "react";

import { SecurityBoundaryNotice } from "./SecurityBoundaryNotice";
import { StatusChip } from "./StatusChip";

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
        <StatusChip
          tone={dirty ? "warning" : "success"}
          label={dirty ? "Unsaved changes" : "Saved"}
        />
        <StatusChip
          tone={validationStale ? "stale" : "success"}
          label={validationStale ? "Validation stale" : "Validation current"}
        />
        <span>Revision {revision}</span>
      </div>
      {mutationBlocked && mutationReason && (
        <span id="mutation-blocked-reason">
          <StatusChip tone="disabled" label={mutationReason} />
        </span>
      )}
      <nav aria-label="History and file actions">
        <button type="button" disabled={yamlBlocked} onClick={onOpenYaml}>YAML</button>
        {actions}
      </nav>
    </>
  );
}
