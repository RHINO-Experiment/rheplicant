import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { EditorSnapshot, EditorTransport, NodeCard } from "./types";

interface Props {
  initial: EditorSnapshot;
  transport: EditorTransport;
}

function nodeElement(target: EventTarget | null): Element | null {
  return target instanceof Element ? target.closest("[data-node-id]") : null;
}

function editableNode(target: EventTarget | null): Element | null {
  const node = nodeElement(target);
  return node?.getAttribute("role") === "button" ? node : null;
}

interface CanvasProps {
  svg: string;
  onSelect: (nodeId: string) => void;
  onHover: (nodeId: string) => void;
}

const SignalCanvas = memo(function SignalCanvas({ svg, onSelect, onHover }: CanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const hover = (event: MouseEvent) => {
      onHover(nodeElement(event.target)?.getAttribute("data-node-id") ?? "");
    };
    canvas.addEventListener("mouseover", hover);
    return () => canvas.removeEventListener("mouseover", hover);
  }, [onHover]);

  function click(event: React.MouseEvent<HTMLDivElement>) {
    const node = editableNode(event.target);
    if (node) onSelect(node.getAttribute("data-node-id") ?? "");
  }

  function keyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Enter" && event.key !== " ") return;
    const node = editableNode(event.target);
    if (!node) return;
    event.preventDefault();
    onSelect(node.getAttribute("data-node-id") ?? "");
  }

  return (
    <div
      ref={canvasRef}
      className="signal-canvas"
      onClick={click}
      onKeyDown={keyDown}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
});

export function ConfigEditor({ initial, transport }: Props) {
  const [document, setDocument] = useState(initial);
  const [selectedId, setSelectedId] = useState("gain");
  const [hoveredId, setHoveredId] = useState("");
  const [yamlText, setYamlText] = useState(initial.yaml_text);
  const [enabled, setEnabled] = useState(
    initial.nodes.find((node) => node.node_id === "gain")?.lit ?? false,
  );
  const [operatorType, setOperatorType] = useState("GainOperator");
  const [gain, setGain] = useState("1.0");
  const [status, setStatus] = useState("Ready");

  const byId = useMemo(
    () => new Map(document.nodes.map((node) => [node.node_id, node])),
    [document.nodes],
  );
  const selected = byId.get(selectedId);
  const thinSliceEditable = selectedId === "gain";

  const choose = useCallback((nodeId: string) => {
    const node = byId.get(nodeId);
    if (!node?.editable) return;
    setSelectedId(nodeId);
    setEnabled(node.lit);
  }, [byId]);

  const showHover = useCallback((nodeId: string) => setHoveredId(nodeId), []);

  function walk(step: number) {
    const editable = document.walk_order.filter((nodeId) => byId.get(nodeId)?.editable);
    const current = Math.max(0, editable.indexOf(selectedId));
    choose(editable[(current + step + editable.length) % editable.length]);
  }

  async function applyNode() {
    try {
      const next = await transport.editNode(selectedId, {
        yaml_text: yamlText,
        enabled,
        settings: enabled ? { type: operatorType, gain: Number(gain) } : null,
      });
      setDocument(next);
      setYamlText(next.yaml_text);
      setEnabled(next.nodes.find((node) => node.node_id === selectedId)?.lit ?? false);
      setStatus("YAML transformed by rheplicant.gui");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadYaml() {
    try {
      const next = await transport.snapshot(yamlText);
      setDocument(next);
      setYamlText(next.yaml_text);
      setEnabled(next.nodes.find((node) => node.node_id === selectedId)?.lit ?? false);
      setStatus("YAML mirror loaded");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">Config Plan 5 stack spike</p>
        <h1>Rheplicant config editor — React spike</h1>
        <p>YAML is the scientific state. The canvas and settings are projections.</p>
      </header>
      <div className="workspace">
        <section className="canvas-panel" aria-label="Signal path editor">
          <div className="canvas-toolbar">
            <button onClick={() => walk(-1)}>Previous node</button>
            <strong>Selected: {selectedId}</strong>
            <button onClick={() => walk(1)}>Next node</button>
            <span>Hovered: {hoveredId || "—"}</span>
          </div>
          <SignalCanvas svg={document.svg} onSelect={choose} onHover={showHover} />
        </section>
        <aside>
          <section className="settings" aria-label="Node settings">
            <h2>{selected?.label ?? selectedId}</h2>
            <p>{selected?.description}</p>
            <label>
              <input
                type="checkbox"
                checked={enabled}
                disabled={!thinSliceEditable}
                onChange={(event) => setEnabled(event.target.checked)}
              />
              Node enabled
            </label>
            <label>
              type
              <input
                value={operatorType}
                disabled={!thinSliceEditable}
                onChange={(event) => setOperatorType(event.target.value)}
              />
            </label>
            <label>
              gain
              <input
                value={gain}
                disabled={!thinSliceEditable}
                onChange={(event) => setGain(event.target.value)}
                inputMode="decimal"
              />
            </label>
            {!thinSliceEditable && <p>Task 1 edits gain only; this node proves canvas navigation.</p>}
            <button className="primary" disabled={!thinSliceEditable} onClick={applyNode}>
              Apply node edit
            </button>
          </section>
          <section className="yaml-mirror" aria-label="YAML mirror">
            <h2>YAML — source of truth</h2>
            <textarea
              aria-label="YAML source of truth"
              value={yamlText}
              onChange={(event) => setYamlText(event.target.value)}
            />
            <button onClick={loadYaml}>Load YAML mirror</button>
          </section>
          <p role="status">{status}</p>
        </aside>
      </div>
    </main>
  );
}
