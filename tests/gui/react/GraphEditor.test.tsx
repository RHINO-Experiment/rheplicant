import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GraphEditor } from "../../../src/rheplicant/gui/react/GraphEditor";
import { GraphCanvas } from "../../../src/rheplicant/gui/react/GraphCanvas";
import {
  useModelWorkspace,
  type ModelWorkspaceProps,
} from "../../../src/rheplicant/gui/react/ModelWorkspace";
import {
  canUpdateDraft,
  draftLabel,
  NO_DRAFT,
  type DraftCoordinator,
  type DraftEnvelope,
} from "../../../src/rheplicant/gui/react/drafts";
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
  const secondVariant = {
    ...diagram("high_gain", [{ ...GAIN, lit: true, count: 1, settings: { gain: 2 } }]),
    changed_nodes: ["gain"],
  };
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
      validation: {
        findings: [],
        section_badges: [],
        selected_presets: [],
        preset_changes: [],
        run_blocked: false,
      },
      base_diagram: base,
      backend_diagram: backend,
      variant_diagrams: [variant, secondVariant],
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

const BASE_OPERATION_SETTINGS = {
  type: "BaseAcceptedOperator",
  stages: [{ name: "base-stage", scale: 1 }],
  at: ["base-left", "base-right"],
  snapshot_before: "base-raw",
};

const VARIANT_OPERATION_SETTINGS = {
  type: "VariantAcceptedOperator",
  stages: [{ name: "variant-stage-old", scale: 2 }],
  at: ["variant-left", "variant-right"],
  snapshot_before: "variant-raw",
};

function operationState(
  revision: number,
  variantSettings: Record<string, unknown>,
): EditorSession {
  const initial = state();
  const baseNode = { ...FLAGGING, settings: BASE_OPERATION_SETTINGS };
  const variantNode = { ...FLAGGING, settings: variantSettings };
  return {
    ...initial,
    revision,
    document: {
      ...initial.document,
      walk_order: ["flagging"],
      base_diagram: diagram("base", [baseNode]),
      backend_diagram: diagram("backend", [baseNode]),
      variant_diagrams: [{ ...diagram("low_gain", [variantNode]), changed_nodes: ["flagging"] }],
    },
  };
}

function CoordinatedGraph({
  initial,
  refreshed,
  api,
}: {
  initial: EditorSession;
  refreshed: EditorSession;
  api: SessionTransport;
}) {
  const [accepted, setAccepted] = useState(initial);
  const [draft, setDraft] = useState<DraftEnvelope>(NO_DRAFT);
  const coordinator: DraftCoordinator = {
    draft,
    begin(next) {
      if (draft.kind !== "none") return false;
      setDraft(next);
      return true;
    },
    update(next) {
      if (!canUpdateDraft(draft, next)) {
        throw new Error("Cannot replace a different editor draft");
      }
      setDraft(next);
    },
    clear() { setDraft(NO_DRAFT); },
  };
  const reason = draftLabel(draft);
  return (
    <>
      <button type="button" onClick={() => setAccepted(refreshed)}>Refresh accepted graph</button>
      {reason && <p id="graph-draft-reason">{reason}</p>}
      <GraphEditor
        session={accepted}
        transport={api}
        onAccept={(next) => {
          setAccepted(next);
          coordinator.clear();
        }}
        drafts={coordinator}
        disabledReason={reason ? "graph-draft-reason" : null}
      />
    </>
  );
}

function selectVariant() {
  fireEvent.change(screen.getByRole("combobox", { name: "Editing layer" }), {
    target: { value: "low_gain" },
  });
}

function openAdvanced() {
  if (!screen.queryByRole("textbox", { name: "Placement settings JSON" })) {
    fireEvent.click(screen.getByText("Advanced node controls"));
  }
}

function expectBlocked(control: HTMLElement, reason: string) {
  expect(control).toBeDisabled();
  expect(control).toHaveAccessibleDescription(reason);
}

function modelProps(overrides: Partial<ModelWorkspaceProps> = {}): ModelWorkspaceProps {
  const drafts: DraftCoordinator = {
    draft: NO_DRAFT,
    begin: vi.fn(() => true),
    update: vi.fn(),
    clear: vi.fn(),
  };
  return {
    session: state(),
    transport: transport(),
    drafts,
    disabled: false,
    disabledReason: null,
    onAccept: vi.fn(),
    ...overrides,
  };
}

function ModelHarness(props: ModelWorkspaceProps) {
  const surface = useModelWorkspace(props);
  return <>{surface.main}{surface.inspector}</>;
}

describe("model workspace ownership", () => {
  it("updates the roving stop when only the controlled canvas selection changes", () => {
    const graph = state().document.base_diagram;
    const onSelect = vi.fn();
    const { container, rerender } = render(
      <GraphCanvas
        diagram={graph}
        editable
        selectedNode="gain"
        zoom={1}
        onSelect={onSelect}
      />,
    );
    const gain = container.querySelector('[data-node-id="gain"]');
    const flagging = container.querySelector('[data-node-id="flagging"]');

    expect(gain).toHaveAttribute("tabindex", "0");
    expect(gain).toHaveAttribute("aria-pressed", "true");
    rerender(
      <GraphCanvas
        diagram={graph}
        editable
        selectedNode="flagging"
        zoom={1}
        onSelect={onSelect}
      />,
    );

    expect(gain).toHaveAttribute("tabindex", "-1");
    expect(gain).toHaveAttribute("aria-pressed", "false");
    expect(flagging).toHaveAttribute("tabindex", "0");
    expect(flagging).toHaveAttribute("aria-pressed", "true");
  });

  it("renders one interactive server canvas and the selected node inspector as siblings", () => {
    const { container } = render(<ModelHarness {...modelProps()} />);

    expect(screen.getAllByRole("group", { name: "Signal path" })).toHaveLength(1);
    expect(container.querySelectorAll(".graph-viewport svg")).toHaveLength(1);
    fireEvent.click(container.querySelector('[data-node-id="gain"]')!);
    expect(screen.getByRole("complementary", { name: "gain settings" })).toBeVisible();
  });

  it("switches the one editable canvas between full and processing projections", () => {
    const { container } = render(<ModelHarness {...modelProps()} />);

    fireEvent.click(screen.getByRole("button", { name: "Processing" }));
    expect(container.querySelectorAll(".graph-viewport svg")).toHaveLength(1);
    expect(container.querySelector('[data-node-id="filters"]')).not.toBeNull();
    expect(container.querySelector('[data-node-id="gain"]')).toBeNull();
    expect(screen.getAllByRole("group", { name: "Signal path" })).toHaveLength(1);
  });

  it("contains zoomed graph overflow inside its labelled viewport", () => {
    render(<ModelHarness {...modelProps()} />);

    expect(screen.getByRole("group", { name: "Signal path" })).toHaveStyle({
      maxWidth: "100%",
      overflow: "auto",
    });
  });

  it("mounts only base and the selected variant in Compare and removes every node from the tab order", () => {
    render(<ModelHarness {...modelProps()} />);

    fireEvent.change(screen.getByRole("combobox", { name: "Editing layer" }), {
      target: { value: "high_gain" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    const diagrams = screen.getAllByRole("img", { name: /comparison graph/i });
    expect(diagrams).toHaveLength(2);
    expect(screen.getByRole("img", { name: "base comparison graph" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "high_gain comparison graph" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "low_gain comparison graph" })).not.toBeInTheDocument();
    for (const diagram of diagrams) {
      for (const node of within(diagram).getAllByRole("button")) {
        expect(node).toHaveAttribute("tabindex", "-1");
        expect(node).toHaveAttribute("aria-disabled", "true");
        expect(node).not.toHaveAttribute("aria-pressed");
      }
    }
  });

  it("preserves one roving tab stop when a composition node is selected", () => {
    const { container } = render(<ModelHarness {...modelProps()} />);
    const canvas = screen.getByRole("group", { name: "Signal path" });

    fireEvent.click(container.querySelector('[data-node-id="astro_sum"]')!);
    expect(screen.getByRole("complementary", { name: "astro_sum settings" })).toBeVisible();
    expect(canvas.querySelectorAll('[data-node-id][role="button"][tabindex="0"]')).toHaveLength(1);
    expect(canvas.querySelector('[data-node-id="gain"]')).toHaveAttribute("tabindex", "0");
  });

  it("stops routing edits to a selected variant after a complete session removes it", () => {
    const initial = state();
    const api = transport();
    const props = modelProps({ session: initial, transport: api });
    const { rerender } = render(<ModelHarness {...props} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Editing layer" }), {
      target: { value: "high_gain" },
    });

    const withoutSelectedVariant = {
      ...initial,
      revision: 5,
      document: {
        ...initial.document,
        variant_diagrams: initial.document.variant_diagrams.filter(
          (variant) => variant.name !== "high_gain",
        ),
      },
    };
    rerender(<ModelHarness {...props} session={withoutSelectedVariant} />);

    expect(screen.getByRole("combobox", { name: "Editing layer" })).toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: "Light and configure gain" }));
    expect(api.editNode).toHaveBeenLastCalledWith(
      "session-1",
      "gain",
      true,
      {},
      5,
      null,
    );
  });

  it("preserves a removed variant while its graph draft still owns that route", () => {
    const initial = state();
    const api = transport();
    const drafts: DraftCoordinator = {
      draft: {
        kind: "graph",
        baseRevision: 4,
        path: "high_gain:gain:settings",
        rawValue: '{"gain":9}',
      },
      begin: vi.fn(() => false),
      update: vi.fn(),
      clear: vi.fn(),
    };
    const props = modelProps({ session: initial, transport: api, drafts });
    const { rerender } = render(<ModelHarness {...props} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Editing layer" }), {
      target: { value: "high_gain" },
    });

    const withoutSelectedVariant = {
      ...initial,
      revision: 5,
      document: {
        ...initial.document,
        variant_diagrams: initial.document.variant_diagrams.filter(
          (variant) => variant.name !== "high_gain",
        ),
      },
    };
    rerender(<ModelHarness {...props} session={withoutSelectedVariant} />);

    expect(screen.getByRole("textbox", { name: "Node settings JSON" })).toHaveValue(
      '{"gain":9}',
    );
    fireEvent.click(screen.getByRole("button", { name: "Light and configure gain" }));
    expect(api.editNode).toHaveBeenLastCalledWith(
      "session-1",
      "gain",
      true,
      { gain: 9 },
      4,
      "high_gain",
    );
  });

  it("keeps selection while changing views and owns zoom without changing the session", () => {
    const props = modelProps();
    const { container } = render(<ModelHarness {...props} />);
    fireEvent.click(container.querySelector('[data-node-id="flagging"]')!);

    fireEvent.click(screen.getByRole("button", { name: "Processing" }));
    expect(screen.getByRole("complementary", { name: "flagging settings" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(container.querySelector(".graph-scale")).toHaveStyle({ transform: "scale(1.25)" });
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    expect(container.querySelector(".graph-scale")).toHaveStyle({ transform: "scale(1)" });
    expect(props.transport.editNode).not.toHaveBeenCalled();
    expect(props.onAccept).not.toHaveBeenCalled();
  });

  it("does not mount advanced inspector controls until their disclosure opens", () => {
    const { container } = render(<ModelHarness {...modelProps()} />);
    fireEvent.click(container.querySelector('[data-node-id="flagging"]')!);

    expect(screen.queryByRole("textbox", { name: "Composition stages JSON" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Placement settings JSON" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Snapshot name" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Advanced node controls"));
    expect(screen.getByRole("textbox", { name: "Composition stages JSON" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Placement settings JSON" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Snapshot name" })).toBeInTheDocument();
  });
});

describe("graph-guided instrument editor", () => {
  it("keeps the owning graph draft editable while preserving accepted values and offering Apply and Discard", () => {
    const initial = state();
    const projected = {
      ...initial,
      document: {
        ...initial.document,
        base_diagram: diagram("base", [{ ...GAIN, lit: true, settings: { gain: 1.25 } }]),
        walk_order: ["gain"],
      },
    };
    const api = transport();
    function CoordinatedGraph() {
      const [draft, setDraft] = useState<DraftEnvelope>(NO_DRAFT);
      const coordinator: DraftCoordinator = {
        draft,
        begin(next) { if (draft.kind !== "none") return false; setDraft(next); return true; },
        update(next) { setDraft(next); },
        clear() { setDraft(NO_DRAFT); },
      };
      return <GraphEditor session={projected} transport={api} onAccept={vi.fn()} drafts={coordinator} />;
    }
    render(<CoordinatedGraph />);
    fireEvent.click(document.querySelector('[data-node-id="gain"]')!);
    const settings = screen.getByRole("textbox", { name: "Node settings JSON" });
    expect(settings).toHaveValue('{\n  "gain": 1.25\n}');
    fireEvent.change(settings, { target: { value: '{"gain":2}' } });
    expect(settings).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply configuration to gain" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard graph draft" })).toBeEnabled();
  });

  it("blocks every non-owning graph mutation while a settings draft owns the coordinator", () => {
    const initial = state();
    const api = transport();
    const coordinator: DraftCoordinator = {
      draft: { kind: "graph", baseRevision: 4, path: "base:gain:settings", rawValue: '{"gain":2}' },
      begin: vi.fn(() => false),
      update: vi.fn(),
      clear: vi.fn(),
    };
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} drafts={coordinator} />);
    fireEvent.click(document.querySelector('[data-node-id="gain"]')!);
    openAdvanced();
    expect(screen.getByRole("textbox", { name: "Node settings JSON" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Light and configure gain" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply cascade" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Apply placement" })).toBeDisabled();
  });

  it("keeps a variant composition draft through accepted refresh, applies from its original base, and discards independently", async () => {
    const initial = operationState(4, VARIANT_OPERATION_SETTINGS);
    const refreshed = operationState(9, {
      ...VARIANT_OPERATION_SETTINGS,
      stages: [{ name: "variant-stage-from-refresh", scale: 9 }],
    });
    const accepted = operationState(10, {
      ...VARIANT_OPERATION_SETTINGS,
      stages: [{ name: "variant-stage-after-apply", scale: 10 }],
    });
    const api = transport();
    const composeNode = vi.fn(async () => accepted);
    api.composeNode = composeNode;
    render(<CoordinatedGraph initial={initial} refreshed={refreshed} api={api} />);
    openAdvanced();

    expect(screen.getByRole("textbox", { name: "Composition stages JSON" })).toHaveValue(
      '[\n  {\n    "name": "base-stage",\n    "scale": 1\n  }\n]',
    );
    selectVariant();
    const stages = screen.getByRole("textbox", { name: "Composition stages JSON" });
    expect(stages).toHaveValue(
      '[\n  {\n    "name": "variant-stage-old",\n    "scale": 2\n  }\n]',
    );
    const raw = '[\n{"name":"draft-stage","scale":7}\n]';
    fireEvent.change(stages, { target: { value: raw } });

    const reason = "Unsaved graph: low_gain:flagging:stages";
    expect(stages).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply cascade" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard graph draft" })).toBeEnabled();
    for (const control of [
      screen.getByRole("textbox", { name: "Node settings JSON" }),
      screen.getByRole("button", { name: "Apply configuration to flagging" }),
      screen.getByRole("button", { name: "Disable flagging" }),
      screen.getByRole("textbox", { name: "Placement settings JSON" }),
      screen.getByRole("textbox", { name: "Covered nodes in signal order" }),
      screen.getByRole("button", { name: "Apply placement" }),
      screen.getByRole("textbox", { name: "Snapshot name" }),
      screen.getByRole("button", { name: "Keep raw data before flagging" }),
    ]) expectBlocked(control, reason);

    fireEvent.click(screen.getByRole("button", { name: "Refresh accepted graph" }));
    expect(stages).toHaveValue(raw);
    fireEvent.click(screen.getByRole("button", { name: "Apply cascade" }));
    await waitFor(() => expect(composeNode).toHaveBeenCalledWith(
      "session-1",
      "flagging",
      "cascade",
      [{ name: "draft-stage", scale: 7 }],
      4,
      "low_gain",
    ));
    expect(screen.queryByRole("button", { name: "Discard graph draft" })).not.toBeInTheDocument();
    expect(stages).toHaveValue(
      '[\n  {\n    "name": "variant-stage-after-apply",\n    "scale": 10\n  }\n]',
    );

    composeNode.mockClear();
    fireEvent.change(stages, { target: { value: '[{"name":"discard-me"}]' } });
    fireEvent.click(screen.getByRole("button", { name: "Discard graph draft" }));
    expect(stages).toHaveValue(
      '[\n  {\n    "name": "variant-stage-after-apply",\n    "scale": 10\n  }\n]',
    );
    expect(composeNode).not.toHaveBeenCalled();
  });

  it("preserves both exact placement strings through refresh and applies one variant envelope from its original base", async () => {
    const initial = operationState(4, VARIANT_OPERATION_SETTINGS);
    const refreshed = operationState(9, {
      type: "VariantRefreshedOperator",
      stages: [{ name: "refresh-stage" }],
      at: ["refresh-left", "refresh-right"],
      snapshot_before: "refresh-raw",
    });
    const accepted = operationState(10, {
      type: "VariantAppliedOperator",
      gain: 11,
      at: ["success-left", "success-right"],
      snapshot_before: "success-raw",
    });
    const api = transport();
    const placeNode = vi.fn(async () => accepted);
    api.placeNode = placeNode;
    render(<CoordinatedGraph initial={initial} refreshed={refreshed} api={api} />);
    openAdvanced();

    expect(screen.getByRole("textbox", { name: "Placement settings JSON" })).toHaveValue(
      '{\n  "type": "BaseAcceptedOperator",\n  "stages": [\n    {\n      "name": "base-stage",\n      "scale": 1\n    }\n  ],\n  "at": [\n    "base-left",\n    "base-right"\n  ],\n  "snapshot_before": "base-raw"\n}',
    );
    expect(screen.getByRole("textbox", { name: "Covered nodes in signal order" }))
      .toHaveValue("base-left, base-right");
    selectVariant();
    const settings = screen.getByRole("textbox", { name: "Placement settings JSON" });
    const region = screen.getByRole("textbox", { name: "Covered nodes in signal order" });
    expect(settings).toHaveValue(
      '{\n  "type": "VariantAcceptedOperator",\n  "stages": [\n    {\n      "name": "variant-stage-old",\n      "scale": 2\n    }\n  ],\n  "at": [\n    "variant-left",\n    "variant-right"\n  ],\n  "snapshot_before": "variant-raw"\n}',
    );
    expect(region).toHaveValue("variant-left, variant-right");
    const rawSettings = '{ "python" : "pkg:Draft" , "gain": 7 }';
    const rawRegion = "draft-left, draft-middle, draft-right";
    fireEvent.change(settings, { target: { value: rawSettings } });
    fireEvent.change(region, { target: { value: rawRegion } });

    const reason = "Unsaved graph: low_gain:flagging:placement";
    expect(settings).toBeEnabled();
    expect(region).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply placement" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard graph draft" })).toBeEnabled();
    for (const control of [
      screen.getByRole("textbox", { name: "Node settings JSON" }),
      screen.getByRole("button", { name: "Apply configuration to flagging" }),
      screen.getByRole("button", { name: "Disable flagging" }),
      screen.getByRole("textbox", { name: "Composition stages JSON" }),
      screen.getByRole("button", { name: "Apply cascade" }),
      screen.getByRole("textbox", { name: "Snapshot name" }),
      screen.getByRole("button", { name: "Keep raw data before flagging" }),
    ]) expectBlocked(control, reason);

    fireEvent.click(screen.getByRole("button", { name: "Refresh accepted graph" }));
    expect(settings).toHaveValue(rawSettings);
    expect(region).toHaveValue(rawRegion);
    fireEvent.click(screen.getByRole("button", { name: "Apply placement" }));
    await waitFor(() => expect(placeNode).toHaveBeenCalledWith(
      "session-1",
      "flagging",
      ["draft-left", "draft-middle", "draft-right"],
      { python: "pkg:Draft", gain: 7 },
      4,
      "low_gain",
    ));
    expect(screen.queryByRole("button", { name: "Discard graph draft" })).not.toBeInTheDocument();
    expect(settings).toHaveValue(
      '{\n  "type": "VariantAppliedOperator",\n  "gain": 11,\n  "at": [\n    "success-left",\n    "success-right"\n  ],\n  "snapshot_before": "success-raw"\n}',
    );
    expect(region).toHaveValue("success-left, success-right");

    placeNode.mockClear();
    fireEvent.change(settings, { target: { value: '{"python":"discard"}' } });
    fireEvent.change(region, { target: { value: "discard-left, discard-right" } });
    fireEvent.click(screen.getByRole("button", { name: "Discard graph draft" }));
    expect(settings).toHaveValue(
      '{\n  "type": "VariantAppliedOperator",\n  "gain": 11,\n  "at": [\n    "success-left",\n    "success-right"\n  ],\n  "snapshot_before": "success-raw"\n}',
    );
    expect(region).toHaveValue("success-left, success-right");
    expect(placeNode).not.toHaveBeenCalled();
  });

  it("keeps a variant snapshot draft through accepted refresh, applies from its original base, and discards independently", async () => {
    const initial = operationState(4, VARIANT_OPERATION_SETTINGS);
    const refreshed = operationState(9, { ...VARIANT_OPERATION_SETTINGS, snapshot_before: "refresh-raw" });
    const accepted = operationState(10, { ...VARIANT_OPERATION_SETTINGS, snapshot_before: "success-raw" });
    const api = transport();
    const setSnapshotBefore = vi.fn(async () => accepted);
    api.setSnapshotBefore = setSnapshotBefore;
    render(<CoordinatedGraph initial={initial} refreshed={refreshed} api={api} />);
    openAdvanced();

    expect(screen.getByRole("textbox", { name: "Snapshot name" })).toHaveValue("base-raw");
    selectVariant();
    const snapshot = screen.getByRole("textbox", { name: "Snapshot name" });
    expect(snapshot).toHaveValue("variant-raw");
    fireEvent.change(snapshot, { target: { value: "draft-snapshot-byte-exact" } });

    const reason = "Unsaved graph: low_gain:flagging:snapshot";
    expect(snapshot).toBeEnabled();
    expect(screen.getByRole("button", { name: "Keep raw data before flagging" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard graph draft" })).toBeEnabled();
    for (const control of [
      screen.getByRole("textbox", { name: "Node settings JSON" }),
      screen.getByRole("button", { name: "Apply configuration to flagging" }),
      screen.getByRole("button", { name: "Disable flagging" }),
      screen.getByRole("textbox", { name: "Composition stages JSON" }),
      screen.getByRole("button", { name: "Apply cascade" }),
      screen.getByRole("textbox", { name: "Placement settings JSON" }),
      screen.getByRole("textbox", { name: "Covered nodes in signal order" }),
      screen.getByRole("button", { name: "Apply placement" }),
    ]) expectBlocked(control, reason);

    fireEvent.click(screen.getByRole("button", { name: "Refresh accepted graph" }));
    expect(snapshot).toHaveValue("draft-snapshot-byte-exact");
    fireEvent.click(screen.getByRole("button", { name: "Keep raw data before flagging" }));
    await waitFor(() => expect(setSnapshotBefore).toHaveBeenCalledWith(
      "session-1",
      "flagging",
      "draft-snapshot-byte-exact",
      4,
      "low_gain",
    ));
    expect(screen.queryByRole("button", { name: "Discard graph draft" })).not.toBeInTheDocument();
    expect(snapshot).toHaveValue("success-raw");

    setSnapshotBefore.mockClear();
    fireEvent.change(snapshot, { target: { value: "discard-snapshot" } });
    fireEvent.click(screen.getByRole("button", { name: "Discard graph draft" }));
    expect(snapshot).toHaveValue("success-raw");
    expect(setSnapshotBefore).not.toHaveBeenCalled();
  });

  it("blocks Disable, CHAIN move, and snapshot handlers while a settings draft owns the coordinator", () => {
    const initial = state();
    const filters = { ...FILTERS, settings: [{ name: "accepted-first" }, { name: "accepted-second" }] };
    const flagging = { ...FLAGGING, settings: { snapshot_before: "accepted-raw" } };
    const guarded = {
      ...initial,
      document: {
        ...initial.document,
        walk_order: ["filters", "flagging"],
        base_diagram: diagram("base", [filters, flagging]),
        backend_diagram: diagram("backend", [filters, flagging]),
        variant_diagrams: [],
      },
    };
    const api = transport();
    render(<CoordinatedGraph initial={guarded} refreshed={guarded} api={api} />);
    openAdvanced();

    fireEvent.change(screen.getByRole("textbox", { name: "Node settings JSON" }), {
      target: { value: '[{"name":"draft-first"},{"name":"draft-second"}]' },
    });
    const reason = "Unsaved graph: base:filters:settings";
    const disable = screen.getByRole("button", { name: "Disable filters" });
    const move = screen.getByRole("button", { name: "Move filters 2 up" });
    expectBlocked(disable, reason);
    expectBlocked(move, reason);
    disable.removeAttribute("disabled");
    move.removeAttribute("disabled");
    fireEvent.click(disable);
    fireEvent.click(move);
    expect(api.editNode).not.toHaveBeenCalled();
    expect(api.moveNodeInstance).not.toHaveBeenCalled();

    fireEvent.click(document.querySelector('[aria-label="Signal path"] [data-node-id="flagging"]')!);
    openAdvanced();
    const snapshot = screen.getByRole("button", { name: "Keep raw data before flagging" });
    expectBlocked(snapshot, reason);
    snapshot.removeAttribute("disabled");
    fireEvent.click(snapshot);
    expect(api.setSnapshotBefore).not.toHaveBeenCalled();
  });

  it("rechecks coordinator ownership inside enabled Disable, move, and snapshot handlers", () => {
    const initial = state();
    const filters = { ...FILTERS, settings: [{ name: "accepted-first" }, { name: "accepted-second" }] };
    const flagging = { ...FLAGGING, settings: { snapshot_before: "accepted-raw" } };
    const guarded = {
      ...initial,
      document: {
        ...initial.document,
        walk_order: ["filters", "flagging"],
        base_diagram: diagram("base", [filters, flagging]),
        backend_diagram: diagram("backend", [filters, flagging]),
        variant_diagrams: [],
      },
    };
    const coordinator: DraftCoordinator = {
      draft: NO_DRAFT,
      begin: vi.fn(() => true),
      update: vi.fn(),
      clear: vi.fn(),
    };
    const api = transport();
    render(
      <GraphEditor
        session={guarded}
        transport={api}
        onAccept={vi.fn()}
        drafts={coordinator}
      />,
    );
    openAdvanced();

    const disable = screen.getByRole("button", { name: "Disable filters" });
    const move = screen.getByRole("button", { name: "Move filters 2 up" });
    expect(disable).toBeEnabled();
    expect(move).toBeEnabled();

    coordinator.draft = {
      kind: "graph",
      baseRevision: 4,
      path: "base:filters:settings",
      rawValue: '[{"name":"draft-first"},{"name":"draft-second"}]',
    };
    expect(disable).toBeEnabled();
    expect(move).toBeEnabled();
    fireEvent.click(disable);
    fireEvent.click(move);
    expect(api.editNode).not.toHaveBeenCalled();
    expect(api.moveNodeInstance).not.toHaveBeenCalled();

    coordinator.draft = NO_DRAFT;
    fireEvent.click(document.querySelector('[aria-label="Signal path"] [data-node-id="flagging"]')!);
    openAdvanced();
    const snapshot = screen.getByRole("button", { name: "Keep raw data before flagging" });
    expect(snapshot).toBeEnabled();
    coordinator.draft = {
      kind: "graph",
      baseRevision: 4,
      path: "base:flagging:settings",
      rawValue: '{"snapshot_before":"draft-raw"}',
    };
    expect(snapshot).toBeEnabled();
    fireEvent.click(snapshot);
    expect(api.setSnapshotBefore).not.toHaveBeenCalled();
  });

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
    expect(screen.queryByRole("list", { name: "filters instances" })).not.toBeInTheDocument();
    openAdvanced();
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
    fireEvent.click(screen.getByRole("button", { name: "Processing" }));
    const backend = screen.getByRole("region", { name: "Model graph workspace" });
    expect(within(backend).getByText(/overwrite data/)).toBeInTheDocument();
    expect(backend.querySelector('[data-node-id="filters"]')).not.toBeNull();
    expect(backend.querySelector('[data-node-id="gain"]')).toBeNull();

    fireEvent.click(document.querySelector('[aria-label="Signal path"] [data-node-id="flagging"]')!);
    openAdvanced();
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

    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    const comparison = screen.getByRole("region", { name: "Base versus selected variant" });
    expect(within(comparison).getByText("Base")).toBeInTheDocument();
    expect(within(comparison).getByText("low_gain")).toBeInTheDocument();
    expect(within(comparison).getByText("Changed nodes: gain")).toBeInTheDocument();
    const comparisonNode = comparison.querySelector('[data-node-id="gain"]')!;
    expect(comparisonNode).toHaveAttribute("tabindex", "-1");
    fireEvent.click(comparisonNode);
    expect(comparisonNode).toHaveAttribute("tabindex", "-1");
  });

  it("authors ordered compose and at-region edits through their document routes", () => {
    const initial = state();
    const api = transport();
    render(<GraphEditor session={initial} transport={api} onAccept={vi.fn()} />);

    fireEvent.click(document.querySelector('[aria-label="Signal path"] [data-node-id="gain"]')!);
    openAdvanced();
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
    fireEvent.click(document.querySelector('[aria-label="Signal path"] [data-node-id="gain"]')!);
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

  it("uses a roving focus stop and traverses editable nodes in graph order", () => {
    const initial = state();
    render(<GraphEditor session={initial} transport={transport()} onAccept={vi.fn()} />);
    const canvas = screen.getByLabelText("Signal path");
    const gain = canvas.querySelector('[data-node-id="gain"]') as SVGElement;
    const flagging = canvas.querySelector('[data-node-id="flagging"]') as SVGElement;
    const filters = canvas.querySelector('[data-node-id="filters"]') as SVGElement;

    expect(gain).toHaveAttribute("tabindex", "0");
    expect(flagging).toHaveAttribute("tabindex", "-1");
    gain.focus();
    fireEvent.keyDown(gain, { key: "ArrowRight" });
    expect(flagging).toHaveAttribute("tabindex", "0");
    expect(gain).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("heading", { name: "flagging" })).toBeInTheDocument();
    fireEvent.keyDown(flagging, { key: "End" });
    expect(filters).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(filters, { key: "Home" });
    expect(gain).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(gain, { key: "ArrowLeft" });
    expect(filters).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(filters, { key: "ArrowUp" });
    expect(flagging).toHaveAttribute("tabindex", "0");
  });

  it("selects editable nodes with Enter and Space", () => {
    render(<GraphEditor session={state()} transport={transport()} onAccept={vi.fn()} />);
    const canvas = screen.getByLabelText("Signal path");
    const gain = canvas.querySelector('[data-node-id="gain"]') as SVGElement;
    const flagging = canvas.querySelector('[data-node-id="flagging"]') as SVGElement;

    expect(gain).toHaveAttribute("tabindex", "0");
    gain.focus();
    expect(gain).toHaveFocus();
    expect(fireEvent.keyDown(gain, { key: "Enter" })).toBe(false);
    expect(screen.getByRole("complementary", { name: "gain settings" })).toBeVisible();

    fireEvent.keyDown(gain, { key: "ArrowRight" });
    expect(flagging).toHaveAttribute("tabindex", "0");
    expect(flagging).toHaveFocus();
    expect(screen.getByRole("complementary", { name: "flagging settings" })).toBeVisible();
    expect(fireEvent.keyDown(flagging, { key: " " })).toBe(false);
    expect(screen.getByRole("complementary", { name: "flagging settings" })).toBeVisible();
  });
});
