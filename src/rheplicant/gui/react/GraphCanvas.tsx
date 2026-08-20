import { memo, useEffect, useRef } from "react";

import type { GraphDiagram } from "./types";

export interface GraphCanvasProps {
  diagram: GraphDiagram;
  editable: boolean;
  selectedNode: string | null;
  zoom: number;
  onSelect: (nodeId: string) => void;
}

function injectedNode(target: EventTarget | null) {
  return target instanceof Element
    ? target.closest<SVGElement>("[data-node-id]")
    : null;
}

const InjectedGraphMarkup = memo(function InjectedGraphMarkup({ svg }: { svg: string }) {
  return <div className="graph-markup" dangerouslySetInnerHTML={{ __html: svg }} />;
});

export const GraphCanvas = memo(function GraphCanvas({
  diagram,
  editable,
  selectedNode,
  zoom,
  onSelect,
}: GraphCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const nodes = Array.from(
      canvas.querySelectorAll<SVGElement>('[data-node-id][role="button"]'),
    );
    if (!editable) {
      nodes.forEach((node) => {
        node.setAttribute("tabindex", "-1");
        node.setAttribute("aria-disabled", "true");
        node.removeAttribute("aria-pressed");
      });
      return;
    }
    const active = nodes.find((node) => node.dataset.nodeId === selectedNode)
      ?? nodes.find((node) => node.getAttribute("tabindex") === "0")
      ?? nodes[0];
    nodes.forEach((node) => {
      const selected = node === active;
      node.setAttribute("tabindex", selected ? "0" : "-1");
      node.setAttribute("aria-pressed", selected ? "true" : "false");
      node.removeAttribute("aria-disabled");
    });
  }, [diagram.svg, editable, selectedNode]);

  function select(target: EventTarget | null) {
    if (!editable) return;
    const node = injectedNode(target);
    const nodeId = node?.dataset.nodeId;
    if (!nodeId || !canvasRef.current) return;
    if (node.getAttribute("role") === "button") {
      canvasRef.current.querySelectorAll<SVGElement>('[data-node-id][role="button"]')
        .forEach((candidate) => {
          const selected = candidate === node;
          candidate.setAttribute("tabindex", selected ? "0" : "-1");
          candidate.setAttribute("aria-pressed", selected ? "true" : "false");
        });
    }
    onSelect(nodeId);
  }

  function keyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!editable) return;
    const current = injectedNode(event.target);
    if (!current || current.getAttribute("role") !== "button") return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select(current);
      return;
    }
    if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) {
      return;
    }
    const byId = new Map(
      Array.from(
        event.currentTarget.querySelectorAll<SVGElement>('[data-node-id][role="button"]'),
      ).map((node) => [node.dataset.nodeId, node]),
    );
    const nodes = diagram.walk_order.flatMap((nodeId) => {
      const node = byId.get(nodeId);
      return node ? [node] : [];
    });
    if (nodes.length === 0) return;
    const at = Math.max(0, nodes.indexOf(current));
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? nodes.length - 1
        : event.key === "ArrowRight" || event.key === "ArrowDown"
          ? (at + 1) % nodes.length
          : (at - 1 + nodes.length) % nodes.length;
    event.preventDefault();
    nodes.forEach((node, index) => {
      node.setAttribute("tabindex", index === next ? "0" : "-1");
      node.setAttribute("aria-pressed", index === next ? "true" : "false");
    });
    nodes[next].focus();
    onSelect(nodes[next].dataset.nodeId!);
  }

  return (
    <div
      ref={canvasRef}
      className="graph-viewport"
      role={editable ? "group" : "img"}
      aria-label={editable ? "Signal path" : `${diagram.name} comparison graph`}
      style={{ maxWidth: "100%", overflow: "auto" }}
    >
      <div
        className="graph-scale"
        style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
        data-selected-node={selectedNode ?? undefined}
        onClick={editable ? (event) => select(event.target) : undefined}
        onKeyDown={editable ? keyDown : undefined}
      >
        <InjectedGraphMarkup svg={diagram.svg} />
      </div>
    </div>
  );
}, (previous, next) => (
  previous.diagram === next.diagram
  && previous.editable === next.editable
  && previous.selectedNode === next.selectedNode
  && previous.zoom === next.zoom
  && previous.onSelect === next.onSelect
));
