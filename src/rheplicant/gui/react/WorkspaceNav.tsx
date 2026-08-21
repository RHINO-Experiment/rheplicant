import { useEffect, useState, type KeyboardEvent } from "react";

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

const compactQuery = "(max-width: 959px)";

function compactMatches() {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(compactQuery).matches
    : false;
}

export function WorkspaceNav({
  active,
  orientation,
  onChange,
}: WorkspaceNavProps) {
  const [compact, setCompact] = useState(compactMatches);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia(compactQuery);
    const changed = () => setCompact(media.matches);
    media.addEventListener("change", changed);
    changed();
    return () => media.removeEventListener("change", changed);
  }, []);

  const resolvedOrientation = orientation ?? (compact ? "horizontal" : "vertical");

  function move(current: WorkspaceId, direction: 1 | -1) {
    const index = workspaces.findIndex((workspace) => workspace.id === current);
    const next = workspaces[(index + direction + workspaces.length) % workspaces.length];
    document.getElementById(`workspace-tab-${next.id}`)?.focus();
    onChange(next.id);
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, current: WorkspaceId) {
    const forward = resolvedOrientation === "vertical" ? "ArrowDown" : "ArrowRight";
    const backward = resolvedOrientation === "vertical" ? "ArrowUp" : "ArrowLeft";
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
      <div className="workspace-nav" role="tablist" aria-orientation={resolvedOrientation}>
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
