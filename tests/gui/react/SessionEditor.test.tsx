import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import { RequestError } from "../../../src/rheplicant/gui/react/api";
import type {
  EditorSession,
  GraphDiagram,
  NodeCard,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const YAML = "model:\n  gain:\n    type: GainOperator\n    gain: 1.0\n";
const EDITED = YAML.replace("1.0", "1.25");
const SVG = '<svg><g data-node-id="gain" role="button"><text>gain</text></g></svg>';
const FORMS = { sections: [], missing_required: [] };
const VALIDATION = {
  findings: [],
  section_badges: [],
  selected_presets: [],
  preset_changes: [],
  run_blocked: false,
};
const PREVIEWS = {
  classes: [
    { preview_id: "graph" as const, label: "Signal path", cadence: "continuous" as const, priced: false, description: "free" },
    { preview_id: "axes_shapes" as const, label: "Axes and shapes", cadence: "continuous" as const, priced: false, description: "free" },
    { preview_id: "validate" as const, label: "Validate", cadence: "explicit" as const, priced: true, description: "priced" },
    { preview_id: "forward" as const, label: "Preview forward", cadence: "explicit" as const, priced: true, description: "priced" },
  ],
  axes: [],
  shapes: [],
  forward_cost: {
    label: "Cost unavailable until the document declares complete axes",
    estimated_milliseconds: null,
    estimated_peak_megabytes: null,
    n_freq: null,
    nside: null,
    lmax: null,
    optimizations: [],
  },
  declared_run_kinds: ["forward"],
};
const OUTPUTS = {
  requested_yaml: YAML,
  resolved_yaml: YAML,
  resolution_note: "Preset-merged preview.",
  target_path: "/rheplicant-gui/session.results",
  state: "ready_new" as const,
  state_message: "The target is absent and ready for a new run.",
  clobber: false,
  declared_runs: ["forward"],
  products: [{
    name: "assembly",
    enabled: true,
    format: "fits",
    formats: ["fits"],
    runs: [],
    keys: [],
    themes: [],
    expected_paths: [],
  }],
  report: {
    enabled: false,
    rows: [],
    columns: ["mean", "std", "seconds"],
    reference: null,
    relative: [],
    formats: ["text"],
    expected_paths: [],
  },
  audit_paths: [
    "config.input.yaml",
    "config.resolved.yaml",
    "provenance.json",
    "diagnostics.json",
  ],
};
const DIAGRAM: GraphDiagram = {
  name: "base",
  svg: SVG,
  nodes: [],
  walk_order: [],
  counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
  changed_nodes: [],
};
const GAIN: NodeCard = {
  node_id: "gain",
  label: "gain",
  kind: "transform",
  description: "gain",
  explanation: "Configure gain.",
  editable: true,
  reserved: false,
  many: false,
  segment: "forward",
  lit: true,
  count: 1,
  configuration: "single",
  settings: {},
  instances: [],
  stage_names: [],
};

function documentState(yamlText = YAML) {
  return {
    yaml_text: yamlText,
    svg: SVG,
    nodes: [],
    walk_order: [],
    forms: FORMS,
    previews: PREVIEWS,
    validation: VALIDATION,
    base_diagram: DIAGRAM,
    backend_diagram: { ...DIAGRAM, name: "backend" },
    variant_diagrams: [],
  };
}

function state(overrides: Partial<EditorSession> = {}): EditorSession {
  return {
    session_id: "session-1",
    revision: 0,
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    outputs: OUTPUTS,
    jobs: [],
    document: documentState(),
    ...overrides,
  };
}

function candidate(initial = state()) {
  const replaceYaml = vi.fn(async () => state({
    revision: 1,
    dirty: true,
    can_undo: true,
    document: { ...initial.document, yaml_text: EDITED },
  }));
  const undo = vi.fn(async () => state({ revision: 2, can_redo: true }));
  const redo = vi.fn(async () => state({
    revision: 3,
    dirty: true,
    can_undo: true,
    document: { ...initial.document, yaml_text: EDITED },
  }));
  const load = vi.fn(async (_id: string, yamlText: string) => state({
    revision: 1,
    document: { ...initial.document, yaml_text: yamlText },
  }));
  const save = vi.fn(async () => state({ revision: initial.revision + 1 }));
  const refresh = vi.fn(async () => initial);
  const unchanged = vi.fn(async () => initial);
  const setOutputProduct = vi.fn(async () => initial);
  const setOutputReport = vi.fn(async () => initial);
  const submitJob = vi.fn(async () => initial);
  const transport: SessionTransport = {
    refresh,
    replaceYaml,
    undo,
    redo,
    load,
    save,
    editNode: unchanged,
    moveNodeInstance: unchanged,
    composeNode: unchanged,
    placeNode: unchanged,
    setSnapshotBefore: unchanged,
    setOutputProduct,
    setOutputReport,
    submitJob,
  };
  return {
    transport,
    replaceYaml,
    undo,
    redo,
    load,
    save,
    refresh,
    setOutputProduct,
    setOutputReport,
    submitJob,
  };
}

function interactiveState(revision = 4): EditorSession {
  const graph = {
    ...DIAGRAM,
    nodes: [{ ...GAIN, settings: { gain: 1 } }],
    walk_order: ["gain"],
  };
  return state({
    revision,
    can_undo: true,
    can_redo: true,
    document: {
      ...documentState(),
      nodes: graph.nodes,
      walk_order: graph.walk_order,
      base_diagram: graph,
      backend_diagram: { ...graph, name: "backend" },
    },
  });
}

function expectRunningDescription(control: HTMLElement) {
  expect(control).toBeDisabled();
  expect(control).toHaveAccessibleDescription("Another action is running");
}

const FULL_ACCEPTED_YAML = "accepted:\n  exact: 17\n  source: server-response\n";

function fullyDistinctSession(): EditorSession {
  const acceptedGraphNode: NodeCard = {
    ...GAIN,
    node_id: "accepted_gain",
    label: "accepted graph node",
    explanation: "Accepted graph explanation.",
    settings: { gain: 17 },
    stage_names: ["accepted-stage"],
  };
  const acceptedBackendNode: NodeCard = {
    ...GAIN,
    node_id: "accepted_backend",
    label: "accepted backend node",
    explanation: "Accepted backend explanation.",
    segment: "processing",
    settings: { snapshot_before: "accepted-raw" },
  };
  const baseDiagram: GraphDiagram = {
    name: "base",
    svg: '<svg><g data-node-id="accepted_gain" role="button" tabindex="0"><text>accepted graph SVG</text></g></svg>',
    nodes: [acceptedGraphNode],
    walk_order: ["accepted_gain"],
    counts: { lit: 1, skipped: 2, reserved: 3, instances: 4, materialized: 5 },
    changed_nodes: [],
  };
  const backendDiagram: GraphDiagram = {
    name: "backend",
    svg: '<svg><g data-node-id="accepted_backend" role="button" tabindex="0"><text>accepted backend SVG</text></g></svg>',
    nodes: [acceptedBackendNode],
    walk_order: ["accepted_backend"],
    counts: { lit: 6, skipped: 7, reserved: 8, instances: 9, materialized: 10 },
    changed_nodes: [],
  };
  const variantDiagram: GraphDiagram = {
    ...baseDiagram,
    name: "accepted_variant",
    svg: '<svg><g data-node-id="accepted_gain" role="button" tabindex="0"><text>accepted variant SVG</text></g></svg>',
    changed_nodes: ["accepted_gain"],
  };
  return {
    session_id: "session-1",
    revision: 17,
    dirty: true,
    validation_stale: true,
    can_undo: true,
    can_redo: true,
    document: {
      yaml_text: FULL_ACCEPTED_YAML,
      svg: baseDiagram.svg,
      nodes: [acceptedGraphNode],
      walk_order: ["accepted_gain"],
      forms: {
        sections: [{
          section_id: "accepted-runtime",
          label: "Accepted runtime form",
          disabled: false,
          reason: "Accepted section reason",
          widgets: [{
            path: "runtime.accepted_value",
            path_pattern: "runtime.accepted_value",
            label: "Accepted scalar widget",
            widget: "number",
            choices: ["17", "19"],
            visible: true,
            present: true,
            must_decide: true,
            value: 17,
            dimension: "time",
            unit_policy: "seconds",
            delivery: "static",
            disabled: false,
            reason: "Accepted widget reason",
          }],
        }],
        missing_required: ["runtime.another_required_value"],
      },
      previews: {
        classes: [
          { preview_id: "graph", label: "Accepted graph preview", cadence: "continuous", priced: false, description: "Accepted free graph" },
          { preview_id: "axes_shapes", label: "Accepted axes preview", cadence: "continuous", priced: false, description: "Accepted free axes" },
          { preview_id: "validate", label: "Accepted validation preview", cadence: "explicit", priced: true, description: "Accepted priced validation" },
          { preview_id: "forward", label: "Accepted forward preview", cadence: "explicit", priced: true, description: "Accepted priced forward" },
        ],
        axes: [{
          axis: "time",
          first: [1, 2],
          last: [8, 9],
          count: 17,
          spacing: 0.5,
          unit: "s",
          precision_ratio: 0.25,
          precision_ok: true,
        }],
        shapes: [{ symbol: "Naccepted", value: 17 }],
        forward_cost: {
          label: "Accepted forward cost",
          estimated_milliseconds: 170,
          estimated_peak_megabytes: 71,
          n_freq: 19,
          nside: 8,
          lmax: 23,
          optimizations: ["accepted-optimization"],
        },
        declared_run_kinds: ["forward", "compare", "benchmark"],
      },
      validation: {
        findings: [{
          check: "ACCEPTED-CHECK",
          severity: "warn",
          where: "accepted.where",
          message: "Accepted validation finding",
          attribution: "accepted:server",
        }],
        section_badges: [{
          section_id: "accepted-runtime",
          incomplete: 1,
          refuse: 0,
          warn: 1,
          report: 0,
          preset_changes: 1,
        }],
        selected_presets: ["accepted_preset"],
        preset_changes: [{
          path: "runtime.accepted_value",
          kind: "changed",
          preset_value: 11,
          document_value: 17,
        }],
        run_blocked: false,
      },
      base_diagram: baseDiagram,
      backend_diagram: backendDiagram,
      variant_diagrams: [variantDiagram],
    },
    outputs: {
      requested_yaml: "accepted-requested: true\n",
      resolved_yaml: "accepted-resolved: true\n",
      resolution_note: "Accepted resolution note",
      target_path: "/accepted/session.results",
      state: "ready_new",
      state_message: "Accepted output target is ready.",
      clobber: true,
      declared_runs: ["accepted_run"],
      products: [{
        name: "accepted_product",
        enabled: false,
        format: "fits",
        formats: ["fits", "zarr"],
        runs: ["accepted_run"],
        keys: ["accepted-key"],
        themes: ["dark"],
        expected_paths: ["accepted/product.fits"],
      }],
      report: {
        enabled: true,
        rows: ["accepted_run"],
        columns: ["mean"],
        reference: "accepted_run",
        relative: ["mean_sigma"],
        formats: ["json"],
        expected_paths: ["accepted/report.json"],
      },
      audit_paths: ["accepted-audit.json"],
    },
    jobs: [{
      job_id: "accepted-job-17",
      session_id: "session-1",
      kind: "run",
      revision: 17,
      yaml_digest: "accepted-digest",
      status: "succeeded",
      result: { output: { audit_files: ["accepted-audit.json"] } },
      message: null,
      stale: true,
    }],
  };
}

describe("durable React editor session", () => {
  it("retains a YAML draft across workspaces and explains blocked mutations", async () => {
    const user = userEvent.setup();
    const initial = state({
      document: {
        ...documentState(),
        walk_order: ["gain"],
        base_diagram: { ...DIAGRAM, nodes: [GAIN], walk_order: ["gain"] },
      },
    });
    const { transport, submitJob } = candidate(initial);
    render(<SessionEditor initial={initial} transport={transport} />);
    await user.click(screen.getByRole("button", { name: "YAML" }));
    await user.clear(screen.getByRole("textbox", { name: "YAML source of truth" }));
    fireEvent.change(screen.getByRole("textbox", { name: "YAML source of truth" }), {
      target: { value: "model: [" },
    });
    await user.click(screen.getByRole("tab", { name: "Model" }));
    expect(screen.getByText("Unsaved YAML draft")).toBeVisible();
    expect(screen.getByRole("button", { name: /Apply configuration/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Apply configuration/ }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Write assembly" }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    expect(screen.getByRole("checkbox", { name: "Write report" }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    expect(submitJob).not.toHaveBeenCalled();
  });

  it("switches four workspaces without changing YAML or revision", async () => {
    const user = userEvent.setup();
    const session = state({ revision: 7 });
    const {
      transport,
      replaceYaml,
      submitJob,
    } = candidate(session);
    const refresh = vi.fn(async () => session);
    transport.refresh = refresh;
    render(<SessionEditor initial={session} transport={transport} />);
    await user.click(screen.getByRole("button", { name: "YAML" }));

    for (const workspace of ["Config", "Execute", "Results", "Model"] as const) {
      await user.click(screen.getByRole("tab", { name: workspace }));
      expect(screen.getByRole("tabpanel", { name: workspace })).toBeVisible();
      expect(document.querySelectorAll('[role="tabpanel"][id^="workspace-panel-"]'))
        .toHaveLength(1);
      expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(YAML);
      expect(screen.getByText(`Revision ${session.revision}`)).toBeVisible();
    }
    expect(replaceYaml).not.toHaveBeenCalled();
    expect(submitJob).not.toHaveBeenCalled();
    expect(refresh).not.toHaveBeenCalled();
  });

  it("applies YAML with the visible revision and projects dirty/stale state", async () => {
    const { transport, replaceYaml } = candidate();
    render(<SessionEditor initial={state()} transport={transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    expect(screen.getByText("Saved")).toBeInTheDocument();
    expect(screen.getByText("Validation current")).toBeInTheDocument();
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    await waitFor(() => expect(replaceYaml).toHaveBeenCalledWith("session-1", EDITED, 0));
    expect(await screen.findByText("Unsaved changes")).toBeInTheDocument();
    expect(screen.getByText("Revision 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
  });

  it("installs every branch of the complete session returned by YAML Apply", async () => {
    const candidateApi = candidate();
    candidateApi.replaceYaml.mockResolvedValueOnce(fullyDistinctSession());
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const source = screen.getByRole("textbox", { name: "YAML source of truth" });
    const rawDraft = "client-draft:\n  exact: preserve-before-server\n";
    fireEvent.change(source, { target: { value: rawDraft } });
    expect(screen.getByText("Unsaved YAML draft")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    await waitFor(() => expect(candidateApi.replaceYaml).toHaveBeenCalledWith(
      "session-1",
      rawDraft,
      0,
    ));
    expect(await screen.findByText("Revision 17")).toBeInTheDocument();
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(screen.getByText("Validation stale")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Redo" })).toBeEnabled();
    expect(source).toHaveValue(FULL_ACCEPTED_YAML);
    expect(screen.getByText("Accepted revision 17")).toBeInTheDocument();
    expect(screen.queryByText("Unsaved YAML draft")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Discard draft" })).not.toBeInTheDocument();

    const model = screen.getByRole("tabpanel", { name: "Model" });
    expect(within(model).getAllByText("accepted graph SVG").length).toBeGreaterThan(0);
    expect(within(model).getByText("accepted backend SVG")).toBeInTheDocument();
    expect(within(model).getByText("Changed nodes: accepted_gain")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Config" }));
    const form = screen.getByRole("region", { name: "Schema-projected forms" });
    expect(within(form).getByRole("button", { name: /Accepted runtime form/ }))
      .toHaveTextContent("1 incomplete · 1 warn · 1 preset changes");
    expect(within(form).getByRole("article", { name: "runtime.accepted_value" }))
      .toHaveTextContent("Accepted scalar widget");
    expect(within(form).getByText("Accepted widget reason")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    const execute = screen.getByRole("tabpanel", { name: "Execute" });
    expect(within(execute).getByText("/accepted/session.results")).toBeInTheDocument();
    expect(within(execute).getByText("Accepted output target is ready.")).toBeInTheDocument();
    expect(within(execute).getByRole("checkbox", { name: "Write accepted_product" }))
      .not.toBeChecked();
    expect(within(execute).getByText("accepted/product.fits")).toBeInTheDocument();
    expect(within(execute).getByText(/count 17/)).toBeInTheDocument();
    expect(within(execute).getByText("Naccepted").closest("li")).toHaveTextContent("17");
    expect(within(execute).getByRole("button", { name: /Accepted forward cost/ })).toBeEnabled();
    expect(within(execute).getByRole("region", { name: "Explicit jobs" }))
      .toHaveTextContent("accepted-job-17 · succeeded · stale");
    expect(within(execute).getByRole("link", { name: "accepted-audit.json" }))
      .toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    const results = screen.getByRole("tabpanel", { name: "Results" });
    expect(within(results).getByText("Accepted validation finding")).toBeInTheDocument();
    expect(within(results).getByText("Compared with accepted_preset")).toBeInTheDocument();
    expect(within(results).getByText("runtime.accepted_value")).toBeInTheDocument();
    expect(source).toHaveValue(FULL_ACCEPTED_YAML);
  });

  it("does not discard a pending YAML Apply while keeping drawer Close available", async () => {
    let resolveApply: ((next: EditorSession) => void) | undefined;
    const candidateApi = candidate();
    candidateApi.replaceYaml.mockImplementationOnce(() => new Promise<EditorSession>((resolve) => {
      resolveApply = resolve;
    }));
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const source = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(source, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    expect(source).toHaveValue(EDITED);
    resolveApply?.(state({ revision: 1, document: documentState(EDITED) }));
    await waitFor(() => expect(screen.getByText("Revision 1")).toBeInTheDocument());
  });

  it("retains a pending YAML draft for a delayed 422 failure", async () => {
    let rejectApply: ((error: Error) => void) | undefined;
    const candidateApi = candidate();
    candidateApi.replaceYaml.mockImplementationOnce(() => new Promise<EditorSession>((_, reject) => {
      rejectApply = reject;
    }));
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const source = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(source, { target: { value: "model: [" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
    rejectApply?.(new RequestError(422, "expected node"));
    expect(await screen.findByRole("alert", { name: "YAML parse diagnostic" })).toHaveTextContent("expected node");
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(source).toHaveValue("model: [");
  });

  it("retains exact raw YAML and its original base through a delayed 409 conflict", async () => {
    let rejectApply: ((error: Error) => void) | undefined;
    const initial = state({ revision: 3 });
    const candidateApi = candidate(initial);
    candidateApi.replaceYaml.mockImplementationOnce(() => new Promise<EditorSession>((_, reject) => {
      rejectApply = reject;
    }));
    candidateApi.refresh.mockResolvedValueOnce(state({ revision: 4 }));
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const source = screen.getByRole("textbox", { name: "YAML source of truth" });
    const raw = "model: {\n  exact-conflict: [\n";
    fireEvent.change(source, { target: { value: raw } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    const discard = screen.getByRole("button", { name: "Discard draft" });
    expect(discard).toBeDisabled();
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
    fireEvent.click(discard);
    expect(source).toHaveValue(raw);
    fireEvent.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    expect(screen.queryByRole("dialog", { name: "YAML drawer" })).not.toBeInTheDocument();

    rejectApply?.(new RequestError(
      409,
      "Editor command expected revision 3, but the current revision is 4.",
    ));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("YAML revision conflict"));
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(raw);
    const conflict = screen.getByRole("region", { name: "YAML revision conflict" });
    expect(within(conflict).getByRole("alert"))
      .toHaveTextContent("expected revision 3, but the current revision is 4");
    expect(within(conflict).getByText("Draft base revision 3; accepted revision 3."))
      .toBeInTheDocument();
    expect(within(conflict).getByRole("button", { name: "Copy draft" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeEnabled();
    expect(within(conflict).getByRole("button", { name: "Refresh accepted YAML" })).toBeEnabled();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });

  it("undoes and redoes through optimistic revisions", async () => {
    const initial = state({
      revision: 1,
      dirty: true,
      can_undo: true,
      document: documentState(EDITED),
    });
    const { transport, undo, redo } = candidate(initial);
    render(<SessionEditor initial={initial} transport={transport} />);

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(undo).toHaveBeenCalledWith("session-1", 1));
    fireEvent.click(screen.getByRole("button", { name: "Redo" }));
    await waitFor(() => expect(redo).toHaveBeenCalledWith("session-1", 2));
  });

  it("keeps a graph draft base revision across refresh and routes Apply through the parent guard", async () => {
    const initial = state({
      document: {
        ...documentState(),
        walk_order: ["gain"],
        base_diagram: { ...DIAGRAM, nodes: [GAIN], walk_order: ["gain"] },
      },
    });
    const candidateApi = candidate(initial);
    const edited = vi.fn(async () => state({ revision: 2, document: initial.document }));
    const refresh = vi.fn(async () => state({ revision: 1, document: initial.document }));
    candidateApi.transport.editNode = edited;
    candidateApi.transport.refresh = refresh;
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);

    fireEvent.click(document.querySelector('[data-node-id="gain"]')!);
    const settings = screen.getByRole("textbox", { name: "Node settings JSON" });
    fireEvent.change(settings, { target: { value: '{"gain":2}' } });
    expect(settings).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledWith("session-1"));
    fireEvent.change(settings, { target: { value: '{"gain":3}' } });
    fireEvent.click(screen.getByRole("button", { name: "Apply configuration to gain" }));
    await waitFor(() => expect(edited).toHaveBeenCalledWith(
      "session-1", "gain", true, { gain: 3 }, 0, null,
    ));
  });

  it("coordinates a deferred graph action through the parent runner and suppresses overlap", async () => {
    let resolveGraph: ((next: EditorSession) => void) | undefined;
    const initial = interactiveState();
    const candidateApi = candidate(initial);
    const editNode = vi.fn(() => new Promise<EditorSession>((resolve) => {
      resolveGraph = resolve;
    }));
    candidateApi.transport.editNode = editNode;
    const saveFile = vi.fn(async () => undefined);
    render(<SessionEditor initial={initial} transport={candidateApi.transport} saveFile={saveFile} />);

    const settings = screen.getByRole("textbox", { name: "Node settings JSON" });
    fireEvent.change(settings, { target: { value: '{"gain":7}' } });
    const apply = screen.getByRole("button", { name: "Apply configuration to gain" });
    const undo = screen.getByRole("button", { name: "Undo" });
    act(() => {
      apply.click();
      undo.click();
    });
    expect(editNode).toHaveBeenCalledOnce();
    expect(candidateApi.undo).not.toHaveBeenCalled();

    for (const control of [
      screen.getByLabelText("Load YAML file"),
      screen.getByRole("button", { name: "Save YAML" }),
      screen.getByRole("button", { name: "Undo" }),
      screen.getByRole("button", { name: "Redo" }),
      screen.getByRole("button", { name: "Refresh jobs" }),
      screen.getByRole("button", { name: "Apply configuration to gain" }),
    ]) expectRunningDescription(control);

    const execute = screen.getByRole("tab", { name: "Execute" });
    expect(execute).toBeEnabled();
    fireEvent.click(execute);
    const assembly = screen.getByRole("checkbox", { name: "Write assembly" });
    const report = screen.getByRole("checkbox", { name: "Write report" });
    const run = screen.getByRole("button", { name: "Run" });
    expectRunningDescription(assembly);
    expectRunningDescription(report);
    expectRunningDescription(run);
    fireEvent.click(assembly);
    expect(candidateApi.setOutputProduct).not.toHaveBeenCalled();
    expect(candidateApi.submitJob).not.toHaveBeenCalled();

    const yaml = screen.getByRole("button", { name: "YAML" });
    expect(yaml).toBeEnabled();
    fireEvent.click(yaml);
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Close YAML drawer" }));

    resolveGraph?.(interactiveState(5));
    await waitFor(() => expect(screen.getByText("Revision 5")).toBeInTheDocument());
    expect(screen.getByLabelText("Load YAML file")).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save YAML" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Redo" })).toBeEnabled();
    expect(assembly).toBeEnabled();
    expect(run).toBeEnabled();
    fireEvent.click(screen.getByRole("tab", { name: "Model" }));
    expect(screen.getByRole("button", { name: "Apply configuration to gain" })).toBeEnabled();
  });

  it("coordinates a deferred output action through the parent runner and suppresses overlap", async () => {
    let resolveOutput: ((next: EditorSession) => void) | undefined;
    const initial = interactiveState();
    const candidateApi = candidate(initial);
    const setOutputProduct = vi.fn(() => new Promise<EditorSession>((resolve) => {
      resolveOutput = resolve;
    }));
    candidateApi.transport.setOutputProduct = setOutputProduct;
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));

    const assembly = screen.getByRole("checkbox", { name: "Write assembly" });
    const report = screen.getByRole("checkbox", { name: "Write report" });
    act(() => {
      assembly.click();
      report.click();
    });
    expect(setOutputProduct).toHaveBeenCalledOnce();
    expect(candidateApi.setOutputReport).not.toHaveBeenCalled();

    for (const control of [
      screen.getByLabelText("Load YAML file"),
      screen.getByRole("button", { name: "Save YAML" }),
      screen.getByRole("button", { name: "Undo" }),
      screen.getByRole("button", { name: "Redo" }),
      screen.getByRole("button", { name: "Refresh jobs" }),
      screen.getByRole("checkbox", { name: "Write assembly" }),
      screen.getByRole("checkbox", { name: "Write report" }),
      screen.getByRole("button", { name: "Run" }),
    ]) expectRunningDescription(control);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(candidateApi.submitJob).not.toHaveBeenCalled();

    const model = screen.getByRole("tab", { name: "Model" });
    expect(model).toBeEnabled();
    fireEvent.click(model);
    const graphApply = screen.getByRole("button", { name: "Apply configuration to gain" });
    expectRunningDescription(graphApply);
    fireEvent.click(graphApply);
    expect(candidateApi.transport.editNode).not.toHaveBeenCalled();

    expect(screen.getByRole("button", { name: "YAML" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Close YAML drawer" }));

    resolveOutput?.(interactiveState(6));
    await waitFor(() => expect(screen.getByText("Revision 6")).toBeInTheDocument());
    expect(screen.getByLabelText("Load YAML file")).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save YAML" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Redo" })).toBeEnabled();
    expect(graphApply).toBeEnabled();
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("checkbox", { name: "Write assembly" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("loads a file only after user selection and replaces the whole projection", async () => {
    const { transport, load } = candidate();
    const readFile = vi.fn(async () => EDITED);
    render(
      <SessionEditor initial={state()} transport={transport} readFile={readFile} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    const file = new File(["unread by the component"], "experiment.yaml");
    fireEvent.change(screen.getByLabelText("Load YAML file"), {
      target: { files: [file] },
    });

    await waitFor(() => expect(readFile).toHaveBeenCalledWith(file));
    expect(load).toHaveBeenCalledWith("session-1", EDITED, 0);
    expect(await screen.findByRole("status")).toHaveTextContent("Loaded experiment.yaml");
    expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(EDITED);
  });

  it("marks clean only after the explicit save boundary succeeds", async () => {
    const initial = state({ revision: 4, dirty: true });
    const { transport, save } = candidate(initial);
    save.mockResolvedValueOnce(state({ revision: 5, dirty: false }));
    const saveFile = vi.fn(async () => undefined);
    render(
      <SessionEditor initial={initial} transport={transport} saveFile={saveFile} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Save YAML" }));
    await waitFor(() => expect(saveFile).toHaveBeenCalledWith(YAML));
    expect(save).toHaveBeenCalledWith("session-1", 4);
    expect(await screen.findByText("Saved")).toBeInTheDocument();
  });

  it("does not mark the server session saved when the file boundary fails", async () => {
    const initial = state({ revision: 4, dirty: true });
    const { transport, save } = candidate(initial);
    const saveFile = vi.fn(async () => { throw new Error("download refused"); });
    render(
      <SessionEditor initial={initial} transport={transport} saveFile={saveFile} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Save YAML" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("download refused");
    expect(save).not.toHaveBeenCalled();
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
  });

  it("surfaces a revision conflict without overwriting the current YAML", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    const { transport, replaceYaml } = candidate();
    replaceYaml.mockRejectedValueOnce(
      new RequestError(
        409,
        "Editor command expected revision 0, but the current revision is 1.",
      ),
    );
    const refreshed = state({ revision: 1, document: documentState("model: {}\n") });
    const refresh = vi.fn(async () => refreshed);
    transport.refresh = refresh;
    render(<SessionEditor initial={state()} transport={transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("current revision is 1");
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
    expect(screen.queryByRole("alert", { name: "YAML parse diagnostic" }))
      .not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Copy draft" }));
    expect(writeText).toHaveBeenCalledOnce();
    expect(writeText).toHaveBeenCalledWith(EDITED);
    fireEvent.click(screen.getByRole("button", { name: "Refresh accepted YAML" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledWith("session-1"));
    expect(screen.getByText("Revision 1")).toBeInTheDocument();
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Draft base revision 0; accepted revision 1.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    expect(mirror).toHaveValue("model: {}\n");
  });

  it("keeps invalid YAML editable, reports the parse failure, and retains the last good projection", async () => {
    const { transport, replaceYaml } = candidate();
    replaceYaml.mockRejectedValueOnce(
      new RequestError(422, "GUI document: expected the node content"),
    );
    render(<SessionEditor initial={state()} transport={transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });

    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
    fireEvent.change(mirror, { target: { value: "model: [" } });
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    expect(await screen.findByRole("alert", { name: "YAML parse diagnostic" }))
      .toHaveTextContent("expected the node content");
    expect(mirror).toHaveValue("model: [");
    fireEvent.click(screen.getByRole("tab", { name: "Model" }));
    expect(screen.getAllByText("gain").length).toBeGreaterThan(0);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
    fireEvent.change(mirror, { target: { value: "model: [still editing" } });
    expect(mirror).toHaveValue("model: [still editing");
  });

  it("renders every finding, preset difference, section badge, and refusal Run gate", () => {
    const validation = {
      findings: [
        {
          check: "A2",
          severity: "refuse" as const,
          where: "variants.bad.model",
          message: "unknown model node",
          attribution: "variant:bad",
        },
        {
          check: "A7",
          severity: "warn" as const,
          where: "runs[0]",
          message: "suspicious run",
          attribution: "base",
        },
        {
          check: "C20",
          severity: "report" as const,
          where: "inference",
          message: "recorded fact",
          attribution: "base",
        },
      ],
      section_badges: [
        {
          section_id: "variants",
          incomplete: 0,
          refuse: 1,
          warn: 0,
          report: 0,
          preset_changes: 0,
        },
      ],
      selected_presets: ["rhino_v1"],
      preset_changes: [
        {
          path: "runtime.jax_enable_x64",
          kind: "changed" as const,
          preset_value: true,
          document_value: false,
        },
      ],
      run_blocked: true,
    };
    render(<SessionEditor initial={state({
      document: { ...documentState(), validation },
    })} transport={candidate().transport} />);

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    const ledger = screen.getByRole("region", { name: "Pre-flight finding ledger" });
    expect(ledger).toHaveTextContent("unknown model node");
    expect(ledger).toHaveTextContent("suspicious run");
    expect(ledger).toHaveTextContent("recorded fact");
    expect(ledger).toHaveTextContent("variant:bad");
    expect(screen.getByRole("region", { name: "Diff against preset" }))
      .toHaveTextContent("runtime.jax_enable_x64");
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("refreshes terminal jobs without erasing an in-progress YAML draft", async () => {
    const initial = state({
      jobs: [{
        job_id: "job-1",
        session_id: "session-1",
        kind: "run",
        revision: 0,
        yaml_digest: "abc",
        status: "running",
        result: null,
        message: null,
        stale: false,
      }],
    });
    const refreshed = state({
      jobs: [{ ...initial.jobs[0], status: "succeeded", result: { exit_code: 0 } }],
    });
    const candidateApi = candidate(initial);
    const refresh = vi.fn(async () => refreshed);
    candidateApi.transport.refresh = refresh;
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));

    await waitFor(() => expect(refresh).toHaveBeenCalledWith("session-1"));
    expect(mirror).toHaveValue(EDITED);
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("region", { name: "Explicit jobs" }))
      .toHaveTextContent("succeeded");
  });
});
