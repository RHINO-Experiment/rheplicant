import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  EditorSession,
  GraphDiagram,
  NodeCard,
  SessionTransport,
} from "./types";
import type { DraftCoordinator } from "./drafts";

interface Props {
  session: EditorSession;
  transport: SessionTransport;
  onAccept: (next: EditorSession, message: string) => void;
  disabled?: boolean;
  disabledReasonId?: string;
  coordinator?: DraftCoordinator;
  onRun?: (action: () => Promise<EditorSession>, message: string) => void;
}

function nodeElement(target: EventTarget | null) {
  return target instanceof Element ? target.closest("[data-node-id]") : null;
}

interface CanvasProps {
  diagram: GraphDiagram;
  label: string;
  onSelect?: (nodeId: string) => void;
}

const SignalCanvas = memo(function SignalCanvas({
  diagram,
  label,
  onSelect,
}: CanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const interactive = onSelect !== undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const buttons = Array.from(
      canvas.querySelectorAll<SVGElement>('[data-node-id][role="button"]'),
    );
    if (!interactive) {
      buttons.forEach((button) => {
        button.setAttribute("tabindex", "-1");
        button.setAttribute("aria-disabled", "true");
      });
      return;
    }
    const active = buttons.find((button) => button.getAttribute("tabindex") === "0")
      ?? buttons[0];
    buttons.forEach((button) => {
      const selected = button === active;
      button.setAttribute("tabindex", selected ? "0" : "-1");
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
  }, [diagram.svg, interactive]);

  function select(target: EventTarget | null) {
    const selected = nodeElement(target);
    const nodeId = selected?.getAttribute("data-node-id");
    const canvas = canvasRef.current;
    if (selected && canvas && interactive) {
      canvas.querySelectorAll('[data-node-id][role="button"]').forEach((button) => {
        const active = button === selected;
        button.setAttribute("tabindex", active ? "0" : "-1");
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }
    if (nodeId) onSelect?.(nodeId);
  }

  function keyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!interactive) return;
    const current = nodeElement(event.target);
    if (!current) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select(event.target);
      return;
    }
    const movement = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!movement.includes(event.key)) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const byId = new Map(
      Array.from(canvas.querySelectorAll<SVGElement>('[data-node-id][role="button"]'))
        .map((button) => [button.getAttribute("data-node-id"), button]),
    );
    const ordered = diagram.walk_order.flatMap((nodeId) => {
      const button = byId.get(nodeId);
      return button ? [button] : [];
    });
    if (ordered.length === 0) return;
    const currentIndex = Math.max(0, ordered.indexOf(current as SVGElement));
    const nextIndex = event.key === "Home"
      ? 0
      : event.key === "End"
        ? ordered.length - 1
        : event.key === "ArrowRight" || event.key === "ArrowDown"
          ? (currentIndex + 1) % ordered.length
          : (currentIndex - 1 + ordered.length) % ordered.length;
    event.preventDefault();
    const next = ordered[nextIndex];
    ordered.forEach((button) => button.setAttribute("tabindex", button === next ? "0" : "-1"));
    next.focus();
    select(next);
  }

  return (
    <div
      ref={canvasRef}
      aria-label={label}
      onClick={interactive ? (event) => select(event.target) : undefined}
      onKeyDown={interactive ? keyDown : undefined}
      dangerouslySetInnerHTML={{ __html: diagram.svg }}
    />
  );
});

function jsonText(value: unknown, fallback: unknown) {
  return JSON.stringify(value ?? fallback, null, 2);
}

function parseObject(text: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(text);
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("Settings JSON must be an object.");
  }
  return parsed as Record<string, unknown>;
}

function parseSettings(text: string): unknown {
  const parsed: unknown = JSON.parse(text);
  if (parsed === null || typeof parsed !== "object") {
    throw new Error("Node settings JSON must be an object or list.");
  }
  return parsed;
}

function countLine(diagram: GraphDiagram) {
  const { lit, skipped, reserved, instances } = diagram.counts;
  return `lit ${lit} · skipped ${skipped} · reserved ${reserved} · instances ${instances}`;
}

export function GraphEditor({ session, transport, onAccept, disabled = false, disabledReasonId, coordinator, onRun }: Props) {
  const [selectedId, setSelectedId] = useState(session.document.walk_order[0] ?? "");
  const [activeVariant, setActiveVariant] = useState<string | null>(null);
  const [settingsText, setSettingsText] = useState("{}");
  const [stagesText, setStagesText] = useState("[]");
  const [regionText, setRegionText] = useState("");
  const [snapshotName, setSnapshotName] = useState("raw");
  const [status, setStatus] = useState("Graph ready");
  const [statusError, setStatusError] = useState(false);

  const diagram = activeVariant === null
    ? session.document.base_diagram
    : session.document.variant_diagrams.find((item) => item.name === activeVariant)
      ?? session.document.base_diagram;
  const byId = useMemo(
    () => new Map(diagram.nodes.map((node) => [node.node_id, node])),
    [diagram.nodes],
  );
  const selected = byId.get(selectedId);
  const selectedAt = selected?.settings && typeof selected.settings === "object"
    ? (selected.settings as { at?: unknown }).at
    : undefined;
  const selectedSettings = selected
    ? jsonText(selected.settings, selected.configuration === "fan" ? {} : selected.many ? [] : {})
    : "{}";
  const selectedStages = selected?.settings && typeof selected.settings === "object"
    ? jsonText((selected.settings as { stages?: unknown }).stages, [])
    : "[]";
  const selectedRegion = Array.isArray(selectedAt) ? selectedAt.join(", ") : typeof selectedAt === "string" ? selectedAt : "";
  const selectedSnapshot = selected?.settings && typeof selected.settings === "object"
    && typeof (selected.settings as { snapshot_before?: unknown }).snapshot_before === "string"
    ? (selected.settings as { snapshot_before: string }).snapshot_before
    : "";

  useEffect(() => {
    if (coordinator) return;
    if (!selected) return;
    const fallback = selected.configuration === "fan" ? {} : selected.many ? [] : {};
    setSettingsText(jsonText(selected.settings, fallback));
    const stages = selected.settings && typeof selected.settings === "object"
      ? (selected.settings as { stages?: unknown }).stages
      : undefined;
    setStagesText(jsonText(stages, [
      { name: "stage_1" },
      { name: "stage_2" },
    ]));
  }, [selected, coordinator]);

  function rawGraphValue(path: string, fallback: string) {
    const draft = coordinator?.draft;
    return draft?.kind === "graph" && draft.path === path ? draft.rawValue : fallback;
  }

  function ownsGraphDraft(path: string) {
    return coordinator?.draft.kind === "graph" && coordinator.draft.path === path;
  }

  function graphControlDisabled(path: string) {
    return disabled || (coordinator !== undefined && coordinator.draft.kind !== "none" && !ownsGraphDraft(path));
  }

  function graphRevision(path: string) {
    const current = coordinator?.draft;
    return current?.kind === "graph" && current.path === path
      ? current.baseRevision
      : session.revision;
  }

  function placementValues(path: string) {
    const draft = coordinator?.draft;
    const fallback = { settings: selectedSettings, region: selectedRegion };
    if (draft?.kind !== "graph" || draft.path !== path) return fallback;
    try {
      const parsed = JSON.parse(draft.rawValue) as { settings?: unknown; region?: unknown };
      return {
        settings: typeof parsed.settings === "string" ? parsed.settings : fallback.settings,
        region: typeof parsed.region === "string" ? parsed.region : fallback.region,
      };
    } catch { return fallback; }
  }

  function updatePlacement(path: string, key: "settings" | "region", value: string) {
    if (!coordinator) return;
    const values = placementValues(path);
    const current = coordinator.draft;
    const next = {
      kind: "graph" as const,
      path,
      rawValue: JSON.stringify({ ...values, [key]: value }),
      baseRevision: current.kind === "graph" && current.path === path ? current.baseRevision : session.revision,
    };
    if (current.kind === "none") coordinator.begin(next);
    else if (current.kind === "graph" && current.path === path) coordinator.update(next);
  }

  function updateRawGraph(path: string, rawValue: string, setLocal: (value: string) => void) {
    if (!coordinator) {
      setLocal(rawValue);
      return;
    }
    const previous = coordinator.draft;
    const baseRevision = previous.kind === "graph" && previous.path === path
      ? previous.baseRevision
      : session.revision;
    const next = { kind: "graph" as const, path, rawValue, baseRevision };
    if (coordinator.draft.kind === "none") coordinator.begin(next);
    else if (coordinator.draft.kind === "graph" && coordinator.draft.path === path) coordinator.update(next);
  }

  async function run(action: () => Promise<EditorSession>, message: string) {
    if (onRun) {
      onRun(action, message);
      return;
    }
    try {
      const next = await action();
      onAccept(next, message);
      setStatus(message);
      setStatusError(false);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setStatusError(true);
    }
  }

  const select = useCallback((nodeId: string) => {
    setSelectedId(nodeId);
  }, []);

  function applyNode() {
    if (!selected) return;
    if (graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:settings`)) return;
    let settings: unknown;
    try {
      settings = parseSettings(rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:settings`, coordinator ? selectedSettings : settingsText));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setStatusError(true);
      return;
    }
    void run(
      () => transport.editNode(
        session.session_id,
        selected.node_id,
        true,
        settings,
        graphRevision(`${activeVariant ?? "base"}:${selected.node_id}:settings`),
        activeVariant,
      ),
      `Configured ${selected.node_id}`,
    );
  }

  function disableNode() {
    if (!selected) return;
    if (graphControlDisabled("node-action")) return;
    void run(
      () => transport.editNode(
        session.session_id,
        selected.node_id,
        false,
        null,
        session.revision,
        activeVariant,
      ),
      `Disabled ${selected.node_id}`,
    );
  }

  function moveInstance(index: number) {
    if (!selected || graphControlDisabled("node-action")) return;
    void run(
      () => transport.moveNodeInstance(session.session_id, selected.node_id, index, index - 1, session.revision, activeVariant),
      `Moved ${selected.instances[index].label}`,
    );
  }

  function applyComposition() {
    if (!selected) return;
    if (graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:stages`)) return;
    try {
      const stages = JSON.parse(rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:stages`, coordinator ? selectedStages : stagesText)) as unknown;
      if (!Array.isArray(stages) || stages.some(
        (stage) => stage === null || Array.isArray(stage) || typeof stage !== "object",
      )) {
        throw new Error("Composition stages JSON must be a list of objects.");
      }
      const compose = selected.kind === "source" ? "sum" : "cascade";
      void run(
        () => transport.composeNode(
          session.session_id,
          selected.node_id,
          compose,
          stages as Record<string, unknown>[],
          graphRevision(`${activeVariant ?? "base"}:${selected.node_id}:stages`),
          activeVariant,
        ),
        `Composed ${selected.node_id}`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setStatusError(true);
    }
  }

  function applyPlacement() {
    if (!selected) return;
    const path = `${activeVariant ?? "base"}:${selected.node_id}:placement`;
    if (graphControlDisabled(path)) return;
    try {
      const values = coordinator ? placementValues(path) : { settings: settingsText, region: regionText };
      const settings = parseObject(values.settings);
      const nodes = values.region.split(",").map((item) => item.trim()).filter(Boolean);
      if (nodes.length === 0) throw new Error("A placement needs at least one node.");
      void run(
        () => transport.placeNode(
          session.session_id,
          selected.node_id,
          nodes.length === 1 ? nodes[0] : nodes,
          settings,
          graphRevision(path),
          activeVariant,
        ),
        `Placed ${selected.node_id}`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setStatusError(true);
    }
  }

  return (
    <section aria-label="Graph-guided instrument editor">
      <header>
        <h2>Instrument signal path</h2>
        <p>Click a box to light and configure it in one YAML transformation.</p>
        <label>
          Editing layer
          <select
            value={activeVariant ?? ""}
            onChange={(event) => setActiveVariant(event.target.value || null)}
          >
            <option value="">base</option>
            {session.document.variant_diagrams.map((item) => (
              <option key={item.name} value={item.name}>{item.name}</option>
            ))}
          </select>
        </label>
      </header>

      <SignalCanvas
        diagram={diagram}
        label="Signal path diagram"
        onSelect={select}
      />
      <p>{countLine(diagram)}</p>

      {selected && (
        <aside aria-label="Selected graph node">
          <h3>{selected.label}</h3>
          <p>{selected.explanation}</p>
          {selected.reserved && <strong>reserved — python: only</strong>}
          {selected.many && <strong>{selected.configuration.toUpperCase()} · {selected.count} instances</strong>}
          {selected.stage_names.length > 0 && (
            <ol aria-label={`${selected.node_id} composed stages`}>
              {selected.stage_names.map((name) => <li key={name}>{name}</li>)}
            </ol>
          )}
          {Array.isArray(selectedAt) && selectedAt.length > 1 && (
            <p>
              Region: {selectedAt.join(" → ")} · addressed as {selectedAt[selectedAt.length - 1]}
            </p>
          )}

          {selected.instances.length > 0 && (
            <ul aria-label={`${selected.node_id} instances`}>
              {selected.instances.map((instance, index) => (
                <li key={instance.instance_id}>
                  <span>{instance.label}</span>
                  {selected.configuration === "chain" && index > 0 && (
                    <button
                      type="button"
                      disabled={graphControlDisabled("node-action")}
                      aria-describedby={graphControlDisabled("node-action") ? disabledReasonId : undefined}
                      aria-label={`Move ${instance.label} up`}
                      onClick={() => moveInstance(index)}
                    >
                      Move up
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}

          {selected.editable && (
            <>
              <label>
                Node settings JSON
                <textarea
                  aria-label="Node settings JSON"
                  value={rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:settings`, coordinator ? selectedSettings : settingsText)}
                  disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:settings`)}
                  aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:settings`) ? disabledReasonId : undefined}
                  onChange={(event) => updateRawGraph(`${activeVariant ?? "base"}:${selected.node_id}:settings`, event.target.value, setSettingsText)}
                />
              </label>
              <button
                type="button"
                disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:settings`)}
                aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:settings`) ? disabledReasonId : undefined}
                onClick={applyNode}
              >
                {selected.lit ? `Apply configuration to ${selected.node_id}` : `Light and configure ${selected.node_id}`}
              </button>
              {selected.lit && (
                <button type="button" disabled={graphControlDisabled("node-action")} aria-describedby={graphControlDisabled("node-action") ? disabledReasonId : undefined} onClick={disableNode}>
                  Disable {selected.node_id}
                </button>
              )}
              {coordinator?.draft.kind === "graph" && (
                <button type="button" disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onClick={() => coordinator.clear()}>Discard graph draft</button>
              )}

              {!selected.many && !selected.reserved && (
                <details>
                  <summary>Compose stages</summary>
                  <textarea
                    aria-label="Composition stages JSON"
                    value={rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:stages`, coordinator ? selectedStages : stagesText)}
                    disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:stages`)}
                    aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:stages`) ? disabledReasonId : undefined}
                    onChange={(event) => updateRawGraph(`${activeVariant ?? "base"}:${selected.node_id}:stages`, event.target.value, setStagesText)}
                  />
                  <button type="button" disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:stages`)} aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:stages`) ? disabledReasonId : undefined} onClick={applyComposition}>
                    Apply {selected.kind === "source" ? "sum" : "cascade"}
                  </button>
                </details>
              )}

              <details>
                <summary>Place python operator with at:</summary>
                <label>
                  Placement settings JSON
                  <textarea
                    aria-label="Placement settings JSON"
                    value={coordinator ? placementValues(`${activeVariant ?? "base"}:${selected.node_id}:placement`).settings : settingsText}
                    disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`)}
                    aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`) ? disabledReasonId : undefined}
                    onChange={(event) => coordinator ? updatePlacement(`${activeVariant ?? "base"}:${selected.node_id}:placement`, "settings", event.target.value) : setSettingsText(event.target.value)}
                  />
                </label>
                <label>
                  Covered nodes in signal order
                  <input
                    value={coordinator ? placementValues(`${activeVariant ?? "base"}:${selected.node_id}:placement`).region : regionText}
                    disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`)}
                    aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`) ? disabledReasonId : undefined}
                    placeholder="noise, emi"
                    onChange={(event) => coordinator ? updatePlacement(`${activeVariant ?? "base"}:${selected.node_id}:placement`, "region", event.target.value) : setRegionText(event.target.value)}
                  />
                </label>
                <button type="button" disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`)} aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:placement`) ? disabledReasonId : undefined} onClick={applyPlacement}>
                  Apply placement
                </button>
              </details>

              {selected.segment === "processing" && !selected.many && (
                <label>
                  Snapshot name
                  <input
                    aria-label="Snapshot name"
                    value={rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`, coordinator ? selectedSnapshot : snapshotName)}
                    disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`)}
                    aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`) ? disabledReasonId : undefined}
                    onChange={(event) => updateRawGraph(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`, event.target.value, setSnapshotName)}
                  />
                  <button
                    type="button"
                    disabled={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`) || !rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`, coordinator ? selectedSnapshot : snapshotName)}
                    aria-describedby={graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`) ? disabledReasonId : undefined}
                    onClick={() => !graphControlDisabled(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`) && void run(
                      () => transport.setSnapshotBefore(
                        session.session_id,
                        selected.node_id,
                        rawGraphValue(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`, coordinator ? selectedSnapshot : snapshotName),
                        graphRevision(`${activeVariant ?? "base"}:${selected.node_id}:snapshot`),
                        activeVariant,
                      ),
                      `Snapshot before ${selected.node_id}`,
                    )}
                  >
                    Keep raw data before {selected.node_id}
                  </button>
                </label>
              )}
            </>
          )}
        </aside>
      )}

      <section aria-label="Processing backend">
        <h3>Backend — processing only</h3>
        <p>These stages overwrite data; add a snapshot to keep the raw waterfall.</p>
        <SignalCanvas
          diagram={session.document.backend_diagram}
          label="Processing-only signal path"
          onSelect={select}
        />
      </section>

      <section aria-label="Base versus variants">
        <h3>Base versus variants</h3>
        <article>
          <h4>Base</h4>
          <SignalCanvas diagram={session.document.base_diagram} label="Base signal path" />
        </article>
        {session.document.variant_diagrams.map((variant) => (
          <article key={variant.name}>
            <h4>{variant.name}</h4>
            <p>Changed nodes: {variant.changed_nodes.join(", ") || "none"}</p>
            <SignalCanvas diagram={variant} label={`${variant.name} signal path`} />
          </article>
        ))}
      </section>
      <p
        role={statusError ? "alert" : undefined}
        aria-live={statusError ? "assertive" : "polite"}
        className={statusError ? "error-surface" : undefined}
      >
        {status}
      </p>
    </section>
  );
}
