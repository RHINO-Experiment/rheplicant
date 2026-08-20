import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { DraftCoordinator } from "./drafts";
import { GraphCanvas } from "./GraphCanvas";
import { NodeInspector } from "./NodeInspector";
import type { EditorSession, GraphDiagram, SessionTransport } from "./types";

export interface WorkspaceSurface {
  main: ReactNode;
  inspector: ReactNode;
}

export interface ModelWorkspaceProps {
  session: EditorSession;
  transport: SessionTransport;
  drafts?: DraftCoordinator;
  disabled?: boolean;
  disabledReason?: string | null;
  onAccept: (next: EditorSession, message: string) => void;
  onRun?: (action: () => Promise<EditorSession>, message: string) => void;
}

type GraphView = "full" | "processing" | "compare";

function countLine(diagram: GraphDiagram) {
  const { lit, skipped, reserved, instances } = diagram.counts;
  return `lit ${lit} · skipped ${skipped} · reserved ${reserved} · instances ${instances}`;
}

export function useModelWorkspace({
  session,
  transport,
  drafts,
  disabled = false,
  disabledReason = null,
  onAccept,
  onRun,
}: ModelWorkspaceProps): WorkspaceSurface {
  const [activeVariant, setActiveVariant] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(
    session.document.walk_order[0] ?? null,
  );
  const [view, setView] = useState<GraphView>("full");
  const [zoom, setZoom] = useState(1);
  const [status, setStatus] = useState("Graph ready");
  const [statusError, setStatusError] = useState(false);

  const activeVariantExists = activeVariant === null
    || session.document.variant_diagrams.some((item) => item.name === activeVariant);
  const activeVariantDraftOwned = activeVariant !== null
    && drafts?.draft.kind === "graph"
    && drafts.draft.path.startsWith(`${activeVariant}:`);
  const resolvedActiveVariant = activeVariant !== null
    && (activeVariantExists || activeVariantDraftOwned)
    ? activeVariant
    : null;
  const fullDiagram = resolvedActiveVariant === null
    ? session.document.base_diagram
    : session.document.variant_diagrams.find((item) => item.name === resolvedActiveVariant)
      ?? session.document.base_diagram;
  const editableDiagram = view === "processing"
    ? session.document.backend_diagram
    : fullDiagram;
  const selected = useMemo(
    () => editableDiagram.nodes.find((node) => node.node_id === selectedNode)
      ?? editableDiagram.nodes[0],
    [editableDiagram.nodes, selectedNode],
  );
  const comparisonVariant = session.document.variant_diagrams.find(
    (item) => item.name === resolvedActiveVariant,
  ) ?? session.document.variant_diagrams[0];

  useEffect(() => {
    if (activeVariant !== null && resolvedActiveVariant === null) setActiveVariant(null);
  }, [activeVariant, resolvedActiveVariant]);

  useEffect(() => {
    if (view === "compare") return;
    if (selected && selected.node_id !== selectedNode) setSelectedNode(selected.node_id);
  }, [selected, selectedNode, view]);

  const select = useCallback((nodeId: string) => setSelectedNode(nodeId), []);
  const updateStatus = useCallback((message: string, error: boolean) => {
    setStatus(message);
    setStatusError(error);
  }, []);

  const main = (
    <section aria-label="Model graph workspace">
      <header>
        <h2>Instrument signal path</h2>
        <p>Choose a node to configure it in the adjacent inspector.</p>
        <div aria-label="Graph views">
          <button type="button" aria-pressed={view === "full"} onClick={() => setView("full")}>Full path</button>
          <button type="button" aria-pressed={view === "processing"} onClick={() => setView("processing")}>Processing</button>
          <button type="button" aria-pressed={view === "compare"} onClick={() => setView("compare")}>Compare</button>
        </div>
        <label>
          Editing layer
          <select
            value={resolvedActiveVariant ?? ""}
            onChange={(event) => setActiveVariant(event.target.value || null)}
          >
            <option value="">base</option>
            {session.document.variant_diagrams.map((item) => (
              <option key={item.name} value={item.name}>{item.name}</option>
            ))}
          </select>
        </label>
        <div aria-label="Graph zoom">
          <button type="button" onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))}>Zoom out</button>
          <button type="button" onClick={() => setZoom(0.8)}>Fit</button>
          <button type="button" onClick={() => setZoom(1)}>100%</button>
          <button type="button" onClick={() => setZoom((value) => Math.min(2, value + 0.25))}>Zoom in</button>
          <button type="button" onClick={() => { setView("full"); setZoom(1); }}>Reset view</button>
        </div>
      </header>

      {view === "compare" ? (
        <section aria-label="Base versus selected variant">
          <article>
            <h3>Base</h3>
            <GraphCanvas
              diagram={session.document.base_diagram}
              editable={false}
              selectedNode={selectedNode}
              zoom={zoom}
              onSelect={select}
            />
          </article>
          {comparisonVariant && (
            <article>
              <h3>{comparisonVariant.name}</h3>
              <p>Changed nodes: {comparisonVariant.changed_nodes.join(", ") || "none"}</p>
              <GraphCanvas
                diagram={comparisonVariant}
                editable={false}
                selectedNode={selectedNode}
                zoom={zoom}
                onSelect={select}
              />
            </article>
          )}
        </section>
      ) : (
        <>
          {view === "processing" && (
            <p>Processing-only stages overwrite data; add a snapshot to keep the raw waterfall.</p>
          )}
          <GraphCanvas
            diagram={editableDiagram}
            editable
            selectedNode={selected?.node_id ?? null}
            zoom={zoom}
            onSelect={select}
          />
          <p>{countLine(editableDiagram)}</p>
        </>
      )}
      <p
        role={statusError ? "alert" : undefined}
        aria-live={statusError ? "assertive" : "polite"}
        className={statusError ? "error-surface" : undefined}
      >
        {status}
      </p>
    </section>
  );

  const inspector = (
    <NodeInspector
      session={session}
      transport={transport}
      drafts={drafts}
      selected={selected}
      activeVariant={resolvedActiveVariant}
      disabled={disabled}
      disabledReason={disabledReason}
      onAccept={onAccept}
      onRun={onRun}
      onStatus={updateStatus}
    />
  );

  return { main, inspector };
}
