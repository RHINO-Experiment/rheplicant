import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  EditorSession,
  GraphDiagram,
  NodeCard,
  SessionTransport,
} from "./types";

interface Props {
  session: EditorSession;
  transport: SessionTransport;
  onAccept: (next: EditorSession, message: string) => void;
  disabled?: boolean;
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

export function GraphEditor({ session, transport, onAccept, disabled = false }: Props) {
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

  useEffect(() => {
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
  }, [selected]);

  async function run(action: () => Promise<EditorSession>, message: string) {
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
    let settings: unknown;
    try {
      settings = parseSettings(settingsText);
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
        session.revision,
        activeVariant,
      ),
      `Configured ${selected.node_id}`,
    );
  }

  function disableNode() {
    if (!selected) return;
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

  function applyComposition() {
    if (!selected) return;
    try {
      const stages = JSON.parse(stagesText) as unknown;
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
          session.revision,
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
    try {
      const settings = parseObject(settingsText);
      const nodes = regionText.split(",").map((item) => item.trim()).filter(Boolean);
      if (nodes.length === 0) throw new Error("A placement needs at least one node.");
      void run(
        () => transport.placeNode(
          session.session_id,
          selected.node_id,
          nodes.length === 1 ? nodes[0] : nodes,
          settings,
          session.revision,
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
                      disabled={disabled}
                      aria-label={`Move ${instance.label} up`}
                      onClick={() => void run(
                        () => transport.moveNodeInstance(
                          session.session_id,
                          selected.node_id,
                          index,
                          index - 1,
                          session.revision,
                          activeVariant,
                        ),
                        `Moved ${instance.label}`,
                      )}
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
                  value={settingsText}
                  disabled={disabled}
                  onChange={(event) => setSettingsText(event.target.value)}
                />
              </label>
              <button
                type="button"
                disabled={disabled}
                onClick={applyNode}
              >
                {selected.lit ? `Apply configuration to ${selected.node_id}` : `Light and configure ${selected.node_id}`}
              </button>
              {selected.lit && (
                <button type="button" disabled={disabled} onClick={disableNode}>
                  Disable {selected.node_id}
                </button>
              )}

              {!selected.many && !selected.reserved && (
                <details>
                  <summary>Compose stages</summary>
                  <textarea
                    aria-label="Composition stages JSON"
                    value={stagesText}
                    disabled={disabled}
                    onChange={(event) => setStagesText(event.target.value)}
                  />
                  <button type="button" disabled={disabled} onClick={applyComposition}>
                    Apply {selected.kind === "source" ? "sum" : "cascade"}
                  </button>
                </details>
              )}

              <details>
                <summary>Place python operator with at:</summary>
                <label>
                  Covered nodes in signal order
                  <input
                    value={regionText}
                    disabled={disabled}
                    placeholder="noise, emi"
                    onChange={(event) => setRegionText(event.target.value)}
                  />
                </label>
                <button type="button" disabled={disabled} onClick={applyPlacement}>
                  Apply placement
                </button>
              </details>

              {selected.segment === "processing" && !selected.many && (
                <label>
                  Snapshot name
                  <input
                    aria-label="Snapshot name"
                    value={snapshotName}
                    disabled={disabled}
                    onChange={(event) => setSnapshotName(event.target.value)}
                  />
                  <button
                    type="button"
                    disabled={disabled || !snapshotName}
                    onClick={() => void run(
                      () => transport.setSnapshotBefore(
                        session.session_id,
                        selected.node_id,
                        snapshotName,
                        session.revision,
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
