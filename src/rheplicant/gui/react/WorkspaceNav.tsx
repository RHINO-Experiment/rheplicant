import type { KeyboardEvent } from "react";

export type WorkspaceId = "model" | "config" | "execute" | "results";

export interface WorkspaceNavProps {
  active: WorkspaceId;
  orientation?: "horizontal" | "vertical";
  onChange(next: WorkspaceId): void;
}

const workspaces: readonly { id: WorkspaceId; label: string }[] = [
  { id: "model", label: "Model" },
  { id: "config", label: "Config" },
  { id: "execute", label: "Execute" },
  { id: "results", label: "Results" },
];

export function WorkspaceNav({
  active,
  orientation = "vertical",
  onChange,
}: WorkspaceNavProps) {
  function move(current: WorkspaceId, direction: 1 | -1) {
    const index = workspaces.findIndex((workspace) => workspace.id === current);
    const next = workspaces[(index + direction + workspaces.length) % workspaces.length];
    document.getElementById(`workspace-tab-${next.id}`)?.focus();
    onChange(next.id);
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, current: WorkspaceId) {
    const forward = orientation === "vertical" ? "ArrowDown" : "ArrowRight";
    const backward = orientation === "vertical" ? "ArrowUp" : "ArrowLeft";
    if (event.key === forward) {
      event.preventDefault();
      move(current, 1);
    } else if (event.key === backward) {
      event.preventDefault();
      move(current, -1);
    }
  }

  return (
    <nav aria-label="Workbench workspaces">
      <div role="tablist" aria-orientation={orientation}>
        {workspaces.map((workspace) => (
          <button
            key={workspace.id}
            id={`workspace-tab-${workspace.id}`}
            type="button"
            role="tab"
            aria-selected={active === workspace.id}
            aria-controls={`workspace-panel-${workspace.id}`}
            tabIndex={active === workspace.id ? 0 : -1}
            onClick={() => onChange(workspace.id)}
            onKeyDown={(event) => onKeyDown(event, workspace.id)}
          >
            {workspace.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
