import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GraphEditor } from "../../../src/rheplicant/gui/react/GraphEditor";
import type {
  EditorSession,
  GraphDiagram,
  NodeCard,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const GAIN: NodeCard = {
  node_id: "gain",
  label: "gain",
  kind: "transform",
  description: "time-dependent gain",
  explanation: "Configure the time-dependent gain.",
  editable: true,
  reserved: false,
  many: false,
  segment: "forward",
  lit: false,
  count: 0,
  configuration: "single",
  settings: null,
  instances: [],
  stage_names: [],
};

const FILTERS: NodeCard = {
  ...GAIN,
  node_id: "filters",
  label: "filters",
  description: "ordered processing filters",
  explanation: "CHAIN list; order is execution order.",
  many: true,
  segment: "processing",
  lit: true,
  count: 2,
  configuration: "chain",
  settings: [{ name: "first" }, { name: "second" }],
  instances: [
    { instance_id: "filters_1", label: "filters 1", settings: { name: "first" } },
    { instance_id: "filters_2", label: "filters 2", settings: { name: "second" } },
  ],
};

const FLAGGING: NodeCard = {
  ...GAIN,
  node_id: "flagging",
  label: "flagging",
  description: "RFI flags",
  explanation: "Destructive processing stage.",
  segment: "processing",
  lit: true,
  count: 1,
  settings: { type: "FlaggingOperator", threshold: 4.0 },
};

const JUNCTION: NodeCard = {
  ...GAIN,
  node_id: "astro_sum",
  label: "astro sum",
  kind: "junction",
  description: "astrophysical sum",
  explanation: "Automatic junction: it adds live branches and is not an operator slot.",
  editable: false,
  configuration: "junction",
};

function diagram(
  name: string,
  nodes: NodeCard[],
  svg = '<svg><g data-node-id="gain" role="button" tabindex="0"></g><g data-node-id="flagging" role="button" tabindex="0"></g><g data-node-id="filters" role="button" tabindex="0"></g><g data-node-id="astro_sum" aria-disabled="true"></g></svg>',
): GraphDiagram {
  return {
    name,
    svg,
    nodes,
    walk_order: nodes.map((node) => node.node_id),
    counts: { lit: 1, skipped: 2, reserved: 3, instances: 2, materialized: 1 },
    changed_nodes: [],
  };
}

function state(): EditorSession {
  const base = diagram("base", [GAIN, FLAGGING, FILTERS, JUNCTION]);
  const backend = diagram(
    "backend",
    [FLAGGING, FILTERS],
    '<svg><g data-node-id="flagging" role="button" tabindex="0"></g><g data-node-id="filters" role="button" tabindex="0"></g></svg>',
  );
  const variant = { ...diagram("low_gain", [{ ...GAIN, lit: true, count: 1 }]), changed_nodes: ["gain"] };
  return {
    session_id: "session-1",
    revision: 4,
    dirty: false,
    validation_stale: true,
    can_undo: false,
    can_redo: false,
    document: {
      yaml_text: "model: {}\n",
      svg: base.svg,
      nodes: base.nodes,
      walk_order: base.walk_order,
      forms: { sections: [], missing_required: [] },
      base_diagram: base,
      backend_diagram: backend,
      variant_diagrams: [variant],
    },
  };
}

function transport(): SessionTransport {
  const unchanged = async () => state();
  return {
    replaceYaml: vi.fn(unchanged),
    undo: vi.fn(unchanged),
    redo: vi.fn(unchanged),
    load: vi.fn(unchanged),
    save: vi.fn(unchanged),
    editNode: vi.fn(unchanged),
    moveNodeInstance: vi.fn(unchanged),
    composeNode: vi.fn(unchanged),
    placeNode: vi.fn(unchanged),
    setSnapshotBefore: vi.fn(unchanged),
  };
}

describe("graph-guided instrument editor", () => {
  it("lights and configures a selected node in one revision-checked act", async () => {
    const initial = state();
    const api = transport();
    const accept = vi.fn();
    render(<GraphEditor session={initial} transport={api} onAccept={accept} />);

    fireEvent.click(document.querySelector('[data-node-id="gain"]')!);
    expect(screen.getByRole("heading", { name: "gain" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Node settings JSON" }), {
      target: { value: '{"type":"GainOperator","gain":1.25}' },
    });
    fireEvent.click(screen.getByRole("button", { name: "Light and configure gain" }));

    expect(api.editNode).toHaveBeenCalledWith(
      "session-1",
      "gain",
      true,
      { type: "GainOperator", gain: 1.25 },
      4,
      null,
    );
  });

  it("shows composition-node explanations without offering an operator edit", () => {
    const initial = state();
    render(<GraphEditor session={initial} transport={transport()} onAccept={vi.fn()} />);

    fireEvent.click(document.querySelector('[data-node-id="astro_sum"]')!);
    expect(screen.getByText(/adds live branches/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /configure astro_sum/ })).not.toBeInTheDocument();
  });

  it("renders CHAIN instances in order and moves only the requested entry", () => {
    const initial = state();
    const api = transport();
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} />);

    fireEvent.click(document.querySelector('[data-node-id="filters"]')!);
    const list = screen.getByRole("list", { name: "filters instances" });
    expect(within(list).getAllByRole("listitem").map((item) => item.textContent)).toEqual([
      expect.stringContaining("filters 1"),
      expect.stringContaining("filters 2"),
    ]);
    fireEvent.click(screen.getByRole("button", { name: "Move filters 2 up" }));
    expect(api.moveNodeInstance).toHaveBeenCalledWith("session-1", "filters", 1, 0, 4, null);
  });

  it("shows the processing-only backend, snapshot action, counts, and variant comparison", () => {
    const initial = state();
    const api = transport();
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} />);

    expect(screen.getByText("lit 1 · skipped 2 · reserved 3 · instances 2")).toBeInTheDocument();
    const backend = screen.getByRole("region", { name: "Processing backend" });
    expect(within(backend).getByText(/overwrite data/)).toBeInTheDocument();
    expect(backend.querySelector('[data-node-id="filters"]')).not.toBeNull();
    expect(backend.querySelector('[data-node-id="gain"]')).toBeNull();

    fireEvent.click(document.querySelector('[aria-label="Signal path diagram"] [data-node-id="flagging"]')!);
    fireEvent.change(screen.getByRole("textbox", { name: "Snapshot name" }), {
      target: { value: "raw" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Keep raw data before flagging" }));
    expect(api.setSnapshotBefore).toHaveBeenCalledWith(
      "session-1",
      "flagging",
      "raw",
      4,
      null,
    );

    const comparison = screen.getByRole("region", { name: "Base versus variants" });
    expect(within(comparison).getByText("Base")).toBeInTheDocument();
    expect(within(comparison).getByText("low_gain")).toBeInTheDocument();
    expect(within(comparison).getByText("Changed nodes: gain")).toBeInTheDocument();
  });

  it("authors ordered compose and at-region edits through their document routes", () => {
    const initial = state();
    const api = transport();
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} />);

    fireEvent.click(document.querySelector('[aria-label="Signal path diagram"] [data-node-id="gain"]')!);
    fireEvent.change(screen.getByRole("textbox", { name: "Composition stages JSON" }), {
      target: { value: '[{"name":"lna"},{"name":"post"}]' },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply cascade" }));
    expect(api.composeNode).toHaveBeenCalledWith(
      "session-1",
      "gain",
      "cascade",
      [{ name: "lna" }, { name: "post" }],
      4,
      null,
    );

    fireEvent.change(screen.getByRole("textbox", { name: "Node settings JSON" }), {
      target: { value: '{"python":"pkg:Gain"}' },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Covered nodes in signal order" }), {
      target: { value: "noise, emi" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply placement" }));
    expect(api.placeNode).toHaveBeenCalledWith(
      "session-1",
      "gain",
      ["noise", "emi"],
      { python: "pkg:Gain" },
      4,
      null,
    );
  });

  it("sends edits to the selected variant without mutating the base route", () => {
    const initial = state();
    const api = transport();
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} />);

    fireEvent.change(screen.getByRole("combobox", { name: "Editing layer" }), {
      target: { value: "low_gain" },
    });
    fireEvent.click(document.querySelector('[aria-label="Signal path diagram"] [data-node-id="gain"]')!);
    fireEvent.click(screen.getByRole("button", { name: "Apply configuration to gain" }));
    expect(api.editNode).toHaveBeenCalledWith(
      "session-1",
      "gain",
      true,
      {},
      4,
      "low_gain",
    );
  });
});
