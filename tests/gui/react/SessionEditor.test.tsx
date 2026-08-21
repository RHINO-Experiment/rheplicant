import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import { JobsDrawer } from "../../../src/rheplicant/gui/react/JobsDrawer";
import { RequestError } from "../../../src/rheplicant/gui/react/api";
import { BootstrapShell } from "../../../src/rheplicant/gui/react/main";
import { YamlDrawer } from "../../../src/rheplicant/gui/react/YamlDrawer";
import type {
  EditorSession,
  GraphDiagram,
  JobPollProjection,
  JobProjection,
  NodeCard,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

type ViewportListener = (event: { matches: boolean }) => void;

function installViewport(initialWidth: number) {
  let width = initialWidth;
  const media = new Map<string, {
    matches: boolean;
    listeners: Set<ViewportListener>;
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
  }>();
  function matches(query: string) {
    const maximum = /max-width:\s*(\d+)px/.exec(query)?.[1];
    return maximum === undefined ? false : width <= Number(maximum);
  }
  vi.stubGlobal("matchMedia", vi.fn((query: string) => {
    let entry = media.get(query);
    if (!entry) {
      const listeners = new Set<ViewportListener>();
      entry = {
        matches: matches(query),
        listeners,
        addEventListener: vi.fn((_type: string, listener: ViewportListener) => {
          listeners.add(listener);
        }),
        removeEventListener: vi.fn((_type: string, listener: ViewportListener) => {
          listeners.delete(listener);
        }),
      };
      media.set(query, entry);
    }
    return entry;
  }));
  return {
    resize(nextWidth: number) {
      width = nextWidth;
      for (const [query, entry] of media) {
        const next = matches(query);
        if (entry.matches === next) continue;
        entry.matches = next;
        for (const listener of entry.listeners) listener({ matches: next });
      }
    },
  };
}

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
  typed_form: false,
  typed_form_reason: null,
  type_choices: [],
  selected_type: null,
  fields: [],
  extra_keys: [],
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
  const revision = overrides.revision ?? 0;
  return {
    session_id: "session-1",
    revision,
    yaml_digest: `digest-${revision}`,
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
  const refreshJobs = vi.fn(
    (_sessionId: string, _signal: AbortSignal) => new Promise<JobPollProjection>(() => undefined),
  );
  const unchanged = vi.fn(async () => initial);
  const setOutputProduct = vi.fn(async () => initial);
  const setOutputReport = vi.fn(async () => initial);
  const submitJob = vi.fn(async () => initial);
  const transport: SessionTransport = {
    refresh,
    refreshJobs,
    replaceYaml,
    setField: unchanged,
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
    refreshJobs,
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
    yaml_digest: "accepted-digest",
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

function visibleEvidence(
  label: string | RegExp,
  role: "status" | "alert",
  tone?: "neutral" | "success" | "warning" | "danger" | "stale" | "disabled",
) {
  const visibleLabel = screen.getByText(label);
  const evidence = visibleLabel.closest("[role]");
  expect(visibleLabel).toBeVisible();
  expect(evidence).toHaveAttribute("role", role);
  if (tone) expect(evidence).toHaveClass(`status-${tone}`);
  return evidence as HTMLElement;
}

async function findVisibleEvidence(
  label: string | RegExp,
  role: "status" | "alert",
  tone?: "neutral" | "success" | "warning" | "danger" | "stale" | "disabled",
) {
  await screen.findByText(label);
  const evidence = await waitFor(() => {
    const found = screen.getByText(label).closest("[role]");
    expect(found).toHaveAttribute("role", role);
    if (tone) expect(found).toHaveClass(`status-${tone}`);
    return found;
  });
  // A runtime check rather than a cast: `closest` returns null when nothing in
  // the ancestry carries a role, and casting that away is how a helper starts
  // handing `null` to every caller that believes it received an element.
  if (!(evidence instanceof HTMLElement)) {
    throw new Error(`no element with role ${role} above ${String(label)}`);
  }
  return evidence;
}

describe("durable React editor session", () => {
  it("keeps the named projection visible while bootstrap is loading", () => {
    render(<BootstrapShell error="" />);

    const loading = screen.getByRole("status", { name: "Workbench startup" });
    expect(loading).toHaveTextContent("canonical starter");
    expect(loading).toHaveTextContent("editor session");
    expect(loading).toHaveAttribute("aria-busy", "true");
    expect(loading).toHaveAttribute("aria-live", "polite");
  });

  it("renders the empty jobs state as quiet evidence with one refresh action", () => {
    const initial = state();
    render(<SessionEditor initial={initial} transport={candidate(initial).transport} />);

    const jobs = screen.getByRole("region", { name: "Jobs" });
    const empty = visibleEvidence("No jobs submitted.", "status", "neutral");
    expect(empty).toHaveTextContent("No jobs submitted");
    expect(empty).toHaveAttribute("aria-live", "polite");
    expect(within(jobs).getAllByRole("button")).toHaveLength(1);
    expect(within(jobs).getByRole("button", { name: "Refresh jobs" })).toBeEnabled();
  });

  it("alerts once for each new refusal or internal-error transition, not historical evidence or rerenders", async () => {
    const running: JobProjection = {
      job_id: "transition-job",
      session_id: "session-1",
      kind: "run",
      revision: 4,
      yaml_digest: "digest-4",
      status: "running",
      result: null,
      message: null,
      stale: false,
    };
    const polling = {
      status: "polling" as const,
      error: null,
      nextRetryMs: null,
      refreshNow: vi.fn(),
    };
    const { rerender } = render(<JobsDrawer jobs={[running]} {...polling} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    visibleEvidence("Running run at revision 4", "status", "neutral");

    const refused = { ...running, status: "refused" as const, message: "repair the source" };
    rerender(<JobsDrawer jobs={[refused]} {...polling} />);
    const refusalAlert = await findVisibleEvidence(
      "Refused run at revision 4: repair the source",
      "alert",
      "danger",
    );
    expect(refusalAlert).toHaveTextContent("Refused");

    rerender(<JobsDrawer jobs={[refused]} {...polling} />);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByRole("alert")).toBe(refusalAlert);

    const failed = { ...refused, status: "error" as const, message: "bounded failure" };
    rerender(<JobsDrawer jobs={[failed]} {...polling} />);
    const errorAlert = await findVisibleEvidence(
      "Internal error in run at revision 4: bounded failure",
      "alert",
      "danger",
    );
    expect(errorAlert).toBe(refusalAlert);
    expect(errorAlert).toHaveTextContent("Internal error");
  });

  it("keeps a newly refused job visible when another terminal row is array-latest", async () => {
    const running: JobProjection = {
      job_id: "job-a",
      session_id: "session-1",
      kind: "run",
      revision: 4,
      yaml_digest: "digest-4",
      status: "running",
      result: null,
      message: null,
      stale: false,
    };
    const succeeded: JobProjection = {
      ...running,
      job_id: "job-b",
      kind: "validate",
      status: "succeeded",
    };
    const polling = {
      status: "idle" as const,
      error: null,
      nextRetryMs: null,
      refreshNow: vi.fn(),
    };
    const view = render(<JobsDrawer jobs={[running, succeeded]} {...polling} />);
    visibleEvidence("Running run at revision 4", "status", "neutral");
    visibleEvidence("Current validate succeeded at revision 4", "status", "success");

    view.rerender(<JobsDrawer
      jobs={[{ ...running, status: "refused", message: "new refusal" }, succeeded]}
      {...polling}
    />);
    const alert = await findVisibleEvidence(
      "Refused run at revision 4: new refusal",
      "alert",
      "danger",
    );
    expect(alert).toBeVisible();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.queryByText("Current validate succeeded at revision 4"))
      .not.toBeInTheDocument();

    const nextRunning: JobProjection = {
      ...running,
      job_id: "job-c",
      kind: "benchmark",
    };
    view.rerender(<JobsDrawer
      jobs={[
        { ...running, status: "refused", message: "new refusal" },
        succeeded,
        nextRunning,
      ]}
      {...polling}
    />);
    expect(visibleEvidence("Refused run at revision 4: new refusal", "alert", "danger"))
      .toBe(alert);
    visibleEvidence("Running benchmark at revision 4", "status", "neutral");

    view.rerender(<JobsDrawer
      jobs={[
        { ...running, status: "refused", message: "new refusal" },
        succeeded,
        { ...nextRunning, status: "succeeded" },
      ]}
      {...polling}
    />);
    await findVisibleEvidence(
      "Current benchmark succeeded at revision 4",
      "status",
      "success",
    );
    expect(screen.queryByText("Refused run at revision 4: new refusal"))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("prioritizes a same-batch danger before a genuinely later success", async () => {
    const runningA: JobProjection = {
      job_id: "batch-a",
      session_id: "session-1",
      kind: "run",
      revision: 4,
      yaml_digest: "digest-4",
      status: "running",
      result: null,
      message: null,
      stale: false,
    };
    const runningB: JobProjection = {
      ...runningA,
      job_id: "batch-b",
      kind: "validate",
    };
    const polling = {
      status: "idle" as const,
      error: null,
      nextRetryMs: null,
      refreshNow: vi.fn(),
    };
    const view = render(<JobsDrawer jobs={[runningA, runningB]} {...polling} />);
    visibleEvidence("Running run at revision 4", "status", "neutral");
    visibleEvidence("Running validate at revision 4", "status", "neutral");

    const refusedA = { ...runningA, status: "refused" as const, message: "same batch" };
    const succeededB = { ...runningB, status: "succeeded" as const };
    view.rerender(<JobsDrawer jobs={[refusedA, succeededB]} {...polling} />);
    await findVisibleEvidence(
      "Refused run at revision 4: same batch",
      "alert",
      "danger",
    );
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.queryByText("Current validate succeeded at revision 4"))
      .not.toBeInTheDocument();

    const runningC: JobProjection = {
      ...runningA,
      job_id: "batch-c",
      kind: "benchmark",
    };
    view.rerender(<JobsDrawer jobs={[refusedA, succeededB, runningC]} {...polling} />);
    visibleEvidence("Running benchmark at revision 4", "status", "neutral");
    visibleEvidence("Refused run at revision 4: same batch", "alert", "danger");

    view.rerender(<JobsDrawer
      jobs={[refusedA, succeededB, { ...runningC, status: "succeeded" }]}
      {...polling}
    />);
    await findVisibleEvidence(
      "Current benchmark succeeded at revision 4",
      "status",
      "success",
    );
    expect(screen.queryByText("Refused run at revision 4: same batch"))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("re-alerts a refusal after the same job returns through running", async () => {
    const refused: JobProjection = {
      job_id: "cycle-job",
      session_id: "session-1",
      kind: "run",
      revision: 4,
      yaml_digest: "digest-4",
      status: "refused",
      result: null,
      message: "cycle refusal",
      stale: false,
    };
    const polling = {
      status: "idle" as const,
      error: null,
      nextRetryMs: null,
      refreshNow: vi.fn(),
    };
    const view = render(<JobsDrawer jobs={[refused]} {...polling} />);
    visibleEvidence("Refused run at revision 4: cycle refusal", "status", "danger");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    view.rerender(<JobsDrawer jobs={[{ ...refused, status: "running", message: null }]} {...polling} />);
    visibleEvidence("Running run at revision 4", "status", "neutral");
    view.rerender(<JobsDrawer jobs={[refused]} {...polling} />);
    await findVisibleEvidence("Refused run at revision 4: cycle refusal", "alert", "danger");
  });

  it("alerts only newly reached YAML evidence and keeps reopened history quiet", async () => {
    const base = {
      acceptedYaml: YAML,
      revision: 4,
      draft: { kind: "yaml" as const, baseRevision: 4, text: "model: [" },
      busy: false,
      onChange: vi.fn(),
      onApply: vi.fn(),
      onDiscard: vi.fn(),
      onClose: vi.fn(),
      onRefresh: vi.fn(),
    };
    const view = render(<YamlDrawer {...base} diagnostic={null} conflict={null} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    view.rerender(<YamlDrawer {...base} diagnostic="expected node" conflict={null} />);
    const diagnostic = await findVisibleEvidence("Invalid YAML: expected node", "alert", "danger");
    view.rerender(<YamlDrawer {...base} diagnostic="expected node" conflict={null} />);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(visibleEvidence("Invalid YAML: expected node", "alert", "danger")).toBe(diagnostic);

    view.rerender(<YamlDrawer {...base} diagnostic="expected mapping" conflict={null} />);
    const nextDiagnostic = await findVisibleEvidence(
      "Invalid YAML: expected mapping",
      "alert",
      "danger",
    );
    expect(nextDiagnostic).not.toBe(diagnostic);
    view.rerender(<YamlDrawer {...base} diagnostic="expected mapping" conflict={null} />);
    expect(visibleEvidence("Invalid YAML: expected mapping", "alert", "danger"))
      .toBe(nextDiagnostic);

    view.unmount();
    const diagnosticHistory = render(
      <YamlDrawer {...base} diagnostic="expected mapping" conflict={null} />,
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    visibleEvidence("Invalid YAML: expected mapping", "status", "danger");
    diagnosticHistory.unmount();

    const conflictView = render(<YamlDrawer {...base} diagnostic={null} conflict={null} />);
    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="expected revision 4, current 5" />,
    );
    const conflict = await findVisibleEvidence(
      "Revision conflict: expected revision 4, current 5",
      "alert",
      "danger",
    );
    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="expected revision 4, current 5" />,
    );
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(visibleEvidence(
      "Revision conflict: expected revision 4, current 5",
      "alert",
      "danger",
    ))
      .toBe(conflict);

    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="expected revision 5, current 6" />,
    );
    const nextConflict = await findVisibleEvidence(
      "Revision conflict: expected revision 5, current 6",
      "alert",
      "danger",
    );
    expect(nextConflict).not.toBe(conflict);
    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="expected revision 5, current 6" />,
    );
    expect(visibleEvidence(
      "Revision conflict: expected revision 5, current 6",
      "alert",
      "danger",
    ))
      .toBe(nextConflict);

    conflictView.unmount();
    render(<YamlDrawer
      {...base}
      diagnostic={null}
      conflict="expected revision 5, current 6"
    />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    visibleEvidence("Revision conflict: expected revision 5, current 6", "status", "danger");
  });

  it("re-alerts a historical YAML diagnostic identity after an A-B-A cycle", async () => {
    const base = {
      acceptedYaml: YAML,
      revision: 4,
      draft: { kind: "yaml" as const, baseRevision: 4, text: "model: [" },
      busy: false,
      onChange: vi.fn(),
      onApply: vi.fn(),
      onDiscard: vi.fn(),
      onClose: vi.fn(),
      onRefresh: vi.fn(),
    };
    const diagnosticView = render(
      <YamlDrawer {...base} diagnostic="diagnostic A" conflict={null} />,
    );
    visibleEvidence("Invalid YAML: diagnostic A", "status", "danger");
    diagnosticView.rerender(
      <YamlDrawer {...base} diagnostic="diagnostic B" conflict={null} />,
    );
    await findVisibleEvidence("Invalid YAML: diagnostic B", "alert", "danger");
    diagnosticView.rerender(
      <YamlDrawer {...base} diagnostic="diagnostic A" conflict={null} />,
    );
    await findVisibleEvidence("Invalid YAML: diagnostic A", "alert", "danger");
  });

  it("re-alerts a historical YAML conflict identity after an A-B-A cycle", async () => {
    const base = {
      acceptedYaml: YAML,
      revision: 4,
      draft: { kind: "yaml" as const, baseRevision: 4, text: "model: [" },
      busy: false,
      onChange: vi.fn(),
      onApply: vi.fn(),
      onDiscard: vi.fn(),
      onClose: vi.fn(),
      onRefresh: vi.fn(),
    };
    const conflictView = render(
      <YamlDrawer {...base} diagnostic={null} conflict="conflict A" />,
    );
    visibleEvidence("Revision conflict: conflict A", "status", "danger");
    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="conflict B" />,
    );
    await findVisibleEvidence("Revision conflict: conflict B", "alert", "danger");
    conflictView.rerender(
      <YamlDrawer {...base} diagnostic={null} conflict="conflict A" />,
    );
    await findVisibleEvidence("Revision conflict: conflict A", "alert", "danger");
  });

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
    const draftState = screen.getByText("Unsaved YAML draft").closest("[role]");
    expect(draftState).toHaveAttribute("role", "status");
    expect(draftState).toHaveAttribute("aria-live", "polite");
    expect(draftState).toHaveClass("status-disabled");
    expect(screen.getByRole("button", { name: /Apply configuration/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Apply configuration/ }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Expand assembly product settings" }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    // Kills a regression that lets a pending draft bypass the lazy report-output gate.
    expect(screen.getByRole("button", { name: "Write report" }))
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

    expect(screen.getByText("Saved").closest("[role]"))
      .toHaveAttribute("role", "status");
    expect(screen.getByText("Validation current").closest("[role]"))
      .toHaveAttribute("role", "status");
    expect(screen.getByText("Validation current").closest("[role]"))
      .toHaveClass("status-success");
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    await waitFor(() => expect(replaceYaml).toHaveBeenCalledWith("session-1", EDITED, 0));
    const unsaved = await screen.findByText("Unsaved changes");
    expect(unsaved.closest("[role]")).toHaveAttribute("role", "status");
    expect(unsaved.closest("[role]")).toHaveClass("status-warning");
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
    fireEvent.click(within(model).getByRole("button", { name: "Processing" }));
    expect(within(model).getByText("accepted backend SVG")).toBeInTheDocument();
    fireEvent.click(within(model).getByRole("button", { name: "Compare" }));
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
    // Kills a regression that loses accepted product configuration when its lazy control is opened.
    fireEvent.click(within(execute).getByRole("button", { name: "Add product" }));
    fireEvent.click(within(execute).getByRole("option", { name: "accepted_product" }));
    expect(within(execute).getByRole("checkbox", { name: "Write accepted_product" }))
      .not.toBeChecked();
    expect(within(execute).getByText("accepted/product.fits")).toBeInTheDocument();
    expect(within(execute).getByText(/count 17/)).toBeInTheDocument();
    expect(within(execute).getByText("Naccepted").closest("li")).toHaveTextContent("17");
    expect(within(execute).getByText("Accepted forward cost")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("accepted-job-17 · run · succeeded");
    // Terminal artefacts belong to Task 3 Results, never the progressive Execute workspace.
    expect(within(execute).queryByRole("link", { name: "accepted-audit.json" }))
      .not.toBeInTheDocument();

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
    const diagnostic = await findVisibleEvidence("Invalid YAML: expected node", "alert", "danger");
    expect(diagnostic).toHaveTextContent("Invalid YAML");
    expect(diagnostic).toHaveTextContent("expected node");
    expect(diagnostic).toHaveAttribute("aria-live", "assertive");
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
    await waitFor(() => expect(screen.getByText("YAML revision conflict").closest("[role='status']"))
      .toHaveTextContent("YAML revision conflict"));
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(raw);
    const conflict = screen.getByRole("region", { name: "YAML revision conflict" });
    const historicalConflict = within(conflict)
      .getByText(/Revision conflict: Editor command expected revision 3/)
      .closest("[role]");
    expect(historicalConflict).toHaveAttribute("role", "status");
    expect(historicalConflict).toHaveClass("status-danger");
    expect(historicalConflict).toHaveTextContent("Revision conflict");
    expect(historicalConflict)
      .toHaveTextContent("expected revision 3, but the current revision is 4");
    expect(historicalConflict).toHaveAttribute("aria-live", "polite");
    expect(within(conflict).getByText("Draft base revision 3; accepted revision 3."))
      .toBeInTheDocument();
    expect(within(conflict).getByRole("button", { name: "Copy draft" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeEnabled();
    expect(within(conflict).getByRole("button", { name: "Refresh accepted YAML" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
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

  it("keeps a graph draft base revision across jobs-only refresh and routes Apply through the parent guard", async () => {
    const initial = state({
      document: {
        ...documentState(),
        walk_order: ["gain"],
        base_diagram: { ...DIAGRAM, nodes: [GAIN], walk_order: ["gain"] },
      },
    });
    const candidateApi = candidate(initial);
    const edited = vi.fn(async () => state({ revision: 2, document: initial.document }));
    candidateApi.transport.editNode = edited;
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);

    fireEvent.click(document.querySelector('[data-node-id="gain"]')!);
    const settings = screen.getByRole("textbox", { name: "Node settings JSON" });
    fireEvent.change(settings, { target: { value: '{"gain":2}' } });
    expect(settings).toBeEnabled();
    await waitFor(() => expect(candidateApi.refreshJobs).toHaveBeenCalledOnce());
    candidateApi.refreshJobs.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    await waitFor(() => expect(candidateApi.refreshJobs).toHaveBeenCalledWith(
      "session-1",
      expect.any(AbortSignal),
    ));
    expect(candidateApi.refresh).not.toHaveBeenCalled();
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
    const assembly = screen.getByRole("button", { name: "Expand assembly product settings" });
    // Kills a regression that omits the lazy report trigger from the shared busy gate.
    const report = screen.getByRole("button", { name: "Write report" });
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

    fireEvent.click(screen.getByRole("button", { name: "Expand assembly product settings" }));
    const assembly = screen.getByRole("checkbox", { name: "Write assembly" });
    // Kills a regression that lets a second output mutation bypass the parent runner while the first is pending.
    const report = screen.getByRole("button", { name: "Write report" });
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
      screen.getByRole("button", { name: "Write report" }),
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
    expect(screen.getByRole("button", { name: "Expand assembly product settings" })).toBeEnabled();
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
    expect((await screen.findByText("Loaded experiment.yaml")).closest("[role='status']"))
      .toHaveTextContent("Loaded experiment.yaml");
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

    expect(await findVisibleEvidence(
      /Revision conflict: Editor command expected revision 0/,
      "alert",
      "danger",
    )).toHaveTextContent("current revision is 1");
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
    expect(screen.queryByText(/^Invalid YAML:/)).not.toBeInTheDocument();
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

    expect(await findVisibleEvidence(
      "Invalid YAML: GUI document: expected the node content",
      "alert",
      "danger",
    ))
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

  it("installs polled jobs without erasing an in-progress YAML draft or accepted session", async () => {
    const initial = state({
      jobs: [{
        job_id: "job-1",
        session_id: "session-1",
        kind: "run",
        revision: 0,
        yaml_digest: "digest-0",
        status: "running",
        result: null,
        message: null,
        stale: false,
      }],
    });
    const terminal: JobProjection = {
      ...initial.jobs[0],
      status: "succeeded",
      result: { exit_code: 0 },
    };
    let resolveJobs: ((next: JobPollProjection) => void) | undefined;
    const candidateApi = candidate(initial);
    candidateApi.transport.refreshJobs = vi.fn(() => new Promise<JobPollProjection>((resolve) => {
      resolveJobs = resolve;
    }));
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));

    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    resolveJobs?.({
      session_id: "session-1",
      revision: 0,
      yaml_digest: "digest-0",
      jobs: [terminal],
    });

    await waitFor(() => expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("job-1"));
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
    expect(candidateApi.refresh).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    // Kills a regression that renders polling state from an old accepted session rather than displayJobs.
    expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("succeeded");
  });

  it("syncs display jobs from every complete accepted mutation response", async () => {
    const acceptedJob: JobProjection = {
      job_id: "accepted-queued",
      session_id: "session-1",
      kind: "validate",
      revision: 1,
      yaml_digest: "digest-1",
      status: "queued",
      result: null,
      message: null,
      stale: false,
    };
    const candidateApi = candidate();
    candidateApi.replaceYaml.mockResolvedValueOnce(state({
      revision: 1,
      dirty: true,
      jobs: [acceptedJob],
      document: documentState(EDITED),
    }));
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    fireEvent.change(screen.getByRole("textbox", { name: "YAML source of truth" }), {
      target: { value: EDITED },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    await waitFor(() => expect(screen.getByText("Revision 1")).toBeInTheDocument());
    expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("accepted-queued");
  });

  it("rejects an old same-identity poll resolved after an accepted job response in the same batch", async () => {
    const acceptedJob: JobProjection = {
      job_id: "accepted-job",
      session_id: "session-1",
      kind: "validate",
      revision: 0,
      yaml_digest: "digest-0",
      status: "queued",
      result: null,
      message: null,
      stale: false,
    };
    const accepted = state({
      jobs: [acceptedJob],
    });
    let resolveSubmit: ((next: EditorSession) => void) | undefined;
    let resolveOldPoll: ((next: JobPollProjection) => void) | undefined;
    const candidateApi = candidate();
    candidateApi.submitJob.mockImplementationOnce(() => new Promise<EditorSession>((resolve) => {
      resolveSubmit = resolve;
    }));
    candidateApi.transport.refreshJobs = vi.fn()
      .mockImplementationOnce(() => new Promise<JobPollProjection>((resolve) => {
        resolveOldPoll = resolve;
      }))
      .mockImplementation(() => new Promise<JobPollProjection>(() => undefined));
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    await waitFor(() => expect(candidateApi.transport.refreshJobs).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("tab", { name: "Execute" }));
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    fireEvent.click(screen.getByRole("button", { name: "I understand, continue" }));
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledOnce());

    await act(async () => {
      resolveSubmit?.(accepted);
      resolveOldPoll?.({
        session_id: "session-1",
        revision: 0,
        yaml_digest: "digest-0",
        jobs: [],
      });
      await Promise.resolve();
    });

    await waitFor(() => expect(screen.getByText("Revision 0")).toBeInTheDocument());
    const jobs = screen.getByRole("region", { name: "Jobs" });
    expect(jobs).toHaveTextContent("accepted-job");
  });

  it("bounds the Jobs drawer to active jobs and the most recent terminal transition", () => {
    const jobs: JobProjection[] = [
      {
        job_id: "old-terminal",
        session_id: "session-1",
        kind: "validate",
        revision: 0,
        yaml_digest: "digest-0",
        status: "succeeded",
        result: null,
        message: null,
        stale: false,
      },
      {
        job_id: "queued-active",
        session_id: "session-1",
        kind: "run",
        revision: 0,
        yaml_digest: "digest-0",
        status: "queued",
        result: null,
        message: null,
        stale: false,
      },
      {
        job_id: "new-terminal",
        session_id: "session-1",
        kind: "compare",
        revision: 0,
        yaml_digest: "digest-0",
        status: "refused",
        result: null,
        message: "bounded refusal",
        stale: false,
      },
      {
        job_id: "running-active",
        session_id: "session-1",
        kind: "benchmark",
        revision: 0,
        yaml_digest: "digest-0",
        status: "running",
        result: null,
        message: null,
        stale: false,
      },
    ];
    render(<SessionEditor initial={state({ jobs })} transport={candidate(state({ jobs })).transport} />);

    const drawer = screen.getByRole("region", { name: "Jobs" });
    expect(drawer).toHaveTextContent("queued-active");
    expect(drawer).toHaveTextContent("running-active");
    expect(drawer).toHaveTextContent("new-terminal");
    expect(drawer).toHaveTextContent("bounded refusal");
    expect(drawer).not.toHaveTextContent("old-terminal");
    expect(visibleEvidence("Queued run at revision 0", "status", "neutral"))
      .toHaveTextContent("Queued run at revision 0");
    expect(visibleEvidence("Running benchmark at revision 0", "status", "neutral"))
      .toHaveTextContent("Running benchmark at revision 0");
    const refused = visibleEvidence(/Refused compare at revision 0/, "status", "danger");
    expect(refused).toHaveTextContent("bounded refusal");
    expect(refused).toHaveAttribute("aria-live", "polite");
  });

  it("shows polling failures and manually refreshes without the full-session transport", async () => {
    const candidateApi = candidate();
    candidateApi.transport.refreshJobs = vi.fn(async () => {
      throw new Error("jobs endpoint offline");
    });
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);

    const drawer = screen.getByRole("region", { name: "Jobs" });
    await waitFor(() => expect(drawer).toHaveTextContent("jobs endpoint offline"));
    expect(drawer).toHaveTextContent("Retrying in 1 second");
    const failure = visibleEvidence(
      "Polling failure: jobs endpoint offline. Retrying in 1 second",
      "alert",
      "danger",
    );
    expect(failure).toHaveTextContent("jobs endpoint offline");
    expect(failure).toHaveTextContent("Retrying in 1 second");
    expect(failure).toHaveAttribute("aria-live", "assertive");
    candidateApi.refreshJobs.mockClear();
    fireEvent.click(within(drawer).getByRole("button", { name: "Refresh jobs" }));

    await waitFor(() => expect(candidateApi.transport.refreshJobs).toHaveBeenCalledWith(
      "session-1",
      expect.any(AbortSignal),
    ));
    expect(candidateApi.refresh).not.toHaveBeenCalled();
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
  });

  it("keeps the memoized refresh stable for fresh arrays with the same active identity", async () => {
    const active: JobProjection = {
      job_id: "stable-active",
      session_id: "session-1",
      kind: "run",
      revision: 0,
      yaml_digest: "digest-0",
      status: "running",
      result: null,
      message: null,
      stale: false,
    };
    const initial = state({ jobs: [active] });
    const candidateApi = candidate(initial);
    candidateApi.transport.refreshJobs = vi.fn(async () => ({
      session_id: "session-1",
      revision: 0,
      yaml_digest: "digest-0",
      jobs: [{ ...active }],
    }));

    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);

    await waitFor(() => expect(candidateApi.transport.refreshJobs).toHaveBeenCalledOnce());
    await act(async () => Promise.resolve());
    expect(candidateApi.transport.refreshJobs).toHaveBeenCalledOnce();
  });

  it("holds the first explicit job, submits its stored kind and current revision once, and acknowledges only this browser editor", async () => {
    const user = userEvent.setup();
    const initial = state({ revision: 6 });
    const currentForward = state({
      revision: 6,
      jobs: [{
        job_id: "forward-current",
        session_id: "session-1",
        kind: "preview_forward",
        revision: 6,
        yaml_digest: "digest-6",
        status: "succeeded",
        result: { waterfall: { values: [] } },
        message: null,
        stale: false,
      }],
    });
    const candidateApi = candidate(initial);
    candidateApi.submitJob.mockResolvedValue(currentForward);
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));

    const preview = screen.getByRole("button", { name: /^Preview forward/ });
    await user.click(preview);
    const dialog = screen.getByRole("dialog", { name: "Trusted execution" });
    expect(dialog).toBeVisible();
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
    expect(dialog).toHaveTextContent("plugins and python targets");
    expect(dialog).toHaveTextContent("server filesystem");
    expect(dialog).toHaveTextContent("CPU, accelerator time and wall time");
    expect(dialog).toHaveTextContent("shared server process and account");

    await user.click(within(dialog).getByRole("button", { name: "I understand, continue" }));
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1",
      "preview_forward",
      6,
    ));
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    await waitFor(() => expect(preview).toHaveFocus());
    expect(screen.getByRole("button", { name: "Help" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Validate" }));
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1",
      "validate",
      6,
    ));
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expect(screen.getByText(
      "Trusted YAML: plugins, python targets, paths and jobs run as the server account.",
    )).toBeVisible();
    expect(candidateApi.replaceYaml).not.toHaveBeenCalled();
  });

  it("cancels with the button or Escape without submitting and restores each exact opener", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate(state({ revision: 3 }));
    render(<SessionEditor initial={state({ revision: 3 })} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));

    const run = screen.getByRole("button", { name: "Run" });
    await user.click(run);
    await user.click(screen.getByRole("button", { name: "Cancel trusted execution" }));
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
    await waitFor(() => expect(run).toHaveFocus());

    const validate = screen.getByRole("button", { name: "Validate" });
    await user.click(validate);
    expect(screen.getByRole("dialog", { name: "Trusted execution" })).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
    await waitFor(() => expect(validate).toHaveFocus());
  });

  it("restores a pointer-invoked job button even when it was not the active element", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));

    const yaml = screen.getByRole("button", { name: "YAML" });
    const run = screen.getByRole("button", { name: "Run" });
    yaml.focus();
    expect(yaml).toHaveFocus();
    fireEvent.click(run);
    expect(screen.getByRole("dialog", { name: "Trusted execution" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Cancel trusted execution" }));

    await waitFor(() => expect(run).toHaveFocus());
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
  });

  it("wraps focus between the first and last confirmation controls", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    await user.click(screen.getByRole("button", { name: "Run" }));

    const cancel = screen.getByRole("button", { name: "Cancel trusted execution" });
    const confirm = screen.getByRole("button", { name: "I understand, continue" });
    expect(cancel).toHaveFocus();
    await user.tab({ shift: true });
    expect(confirm).toHaveFocus();
    await user.tab();
    expect(cancel).toHaveFocus();
    screen.getByRole("tab", { name: "Execute" }).focus();
    expect(cancel).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
  });

  it("keeps the YAML drawer and trusted confirmation mutually exclusive", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const run = screen.getByRole("button", { name: "Run" });
    const yaml = screen.getByRole("button", { name: "YAML" });

    await user.click(yaml);
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(screen.getByRole("dialog", { name: "YAML drawer" })).toBeVisible();
    fireEvent.click(run);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(candidateApi.submitJob).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    fireEvent.click(run);
    const confirmation = screen.getByRole("dialog", { name: "Trusted execution" });
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    const cancel = within(confirmation).getByRole("button", { name: "Cancel trusted execution" });
    expect(cancel).toHaveFocus();
    expect(yaml).toBeDisabled();
    fireEvent.click(yaml);
    expect(screen.queryByRole("dialog", { name: "YAML drawer" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(cancel).toHaveFocus();
    await user.click(cancel);
    await waitFor(() => expect(run).toHaveFocus());
  });

  it("restores the global YAML and Diagnostics controls after their drawers close", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    const yaml = screen.getByRole("button", { name: "YAML" });
    const diagnostics = screen.getByRole("button", { name: "Diagnostics" });

    diagnostics.focus();
    fireEvent.click(yaml);
    expect(screen.getByRole("dialog", { name: "YAML drawer" }).parentElement)
      .toHaveClass("workbench-drawer");
    await user.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    await waitFor(() => expect(yaml).toHaveFocus());

    yaml.focus();
    fireEvent.click(diagnostics);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Diagnostics" })).not.toBeInTheDocument();
    await waitFor(() => expect(diagnostics).toHaveFocus());
    expect(candidateApi.replaceYaml).not.toHaveBeenCalled();
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
  });

  it("moves modal focus inside, inerts every background region, and preserves first-job focus", async () => {
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    const background = () => [
      ".workbench-header",
      ".workbench-navigation",
      ".workbench-main",
      ".workbench-inspector",
      ".workbench-jobs",
    ].map((selector) => document.querySelector(selector) as HTMLElement);
    const expectInert = (inert: boolean) => {
      for (const region of background()) {
        expect(region).not.toBeNull();
        if (inert) expect(region).toHaveAttribute("inert");
        else expect(region).not.toHaveAttribute("inert");
      }
    };

    await user.click(screen.getByRole("button", { name: "YAML" }));
    const yamlClose = screen.getByRole("button", { name: "Close YAML drawer" });
    await waitFor(() => expect(yamlClose).toHaveFocus());
    expectInert(true);
    screen.getByRole("tab", { name: "Model" }).focus();
    expect(yamlClose).toHaveFocus();
    await user.click(yamlClose);
    expectInert(false);

    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    const diagnosticsClose = screen.getByRole("button", { name: "Close diagnostics" });
    await waitFor(() => expect(diagnosticsClose).toHaveFocus());
    expectInert(true);
    await user.click(diagnosticsClose);
    expectInert(false);

    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const run = screen.getByRole("button", { name: "Run" });
    await user.click(run);
    const cancel = screen.getByRole("button", { name: "Cancel trusted execution" });
    expect(cancel).toHaveFocus();
    expectInert(true);
    await user.click(cancel);
    await waitFor(() => expect(run).toHaveFocus());
    expectInert(false);
  });

  it.each([1024, 768, 640])(
    "starts the context inspector collapsed at %ipx",
    (width) => {
      installViewport(width);
      render(<SessionEditor initial={state()} transport={candidate().transport} />);
      expect(document.querySelector("details.workbench-inspector"))
        .not.toHaveAttribute("open");
    },
  );

  it("exposes the desktop inspector and resets it safely across compact breakpoints", async () => {
    const viewport = installViewport(1440);
    const user = userEvent.setup();
    const candidateApi = candidate();
    render(<SessionEditor initial={state()} transport={candidateApi.transport} />);
    const inspector = document.querySelector<HTMLDetailsElement>(
      "details.workbench-inspector",
    );

    expect(inspector).not.toBeNull();
    expect(inspector).toHaveAttribute("open");
    act(() => viewport.resize(1024));
    await waitFor(() => expect(inspector).not.toHaveAttribute("open"));
    await user.click(within(inspector as HTMLDetailsElement).getByText("Context inspector"));
    expect(inspector).toHaveAttribute("open");
    await user.click(within(inspector as HTMLDetailsElement).getByText("Context inspector"));
    expect(inspector).not.toHaveAttribute("open");
    act(() => viewport.resize(1440));
    await waitFor(() => expect(inspector).toHaveAttribute("open"));
    act(() => viewport.resize(640));
    await waitFor(() => expect(inspector).not.toHaveAttribute("open"));
    expect(candidateApi.replaceYaml).not.toHaveBeenCalled();
    expect(candidateApi.submitJob).not.toHaveBeenCalled();
  });

  it("does not open or submit trusted execution while a draft or accepted operation blocks mutations", async () => {
    const user = userEvent.setup();
    const draftCandidate = candidate();
    const draftEditor = render(
      <SessionEditor initial={state()} transport={draftCandidate.transport} />,
    );
    await user.click(screen.getByRole("button", { name: "YAML" }));
    fireEvent.change(screen.getByRole("textbox", { name: "YAML source of truth" }), {
      target: { value: "model: [" },
    });
    await user.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const dirtyPreview = screen.getByRole("button", { name: /^Preview forward/ });
    expect(dirtyPreview).toBeDisabled();
    fireEvent.click(dirtyPreview);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expect(draftCandidate.submitJob).not.toHaveBeenCalled();
    draftEditor.unmount();

    let resolveOutput: ((next: EditorSession) => void) | undefined;
    const initial = interactiveState(9);
    const busyCandidate = candidate(initial);
    busyCandidate.transport.setOutputProduct = vi.fn(() => new Promise<EditorSession>((resolve) => {
      resolveOutput = resolve;
    }));
    render(<SessionEditor initial={initial} transport={busyCandidate.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    await user.click(screen.getByRole("button", { name: "Expand assembly product settings" }));
    await user.click(screen.getByRole("checkbox", { name: "Write assembly" }));
    const busyRun = screen.getByRole("button", { name: "Run" });
    expectRunningDescription(busyRun);
    fireEvent.click(busyRun);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expect(busyCandidate.submitJob).not.toHaveBeenCalled();
    resolveOutput?.(interactiveState(10));
    await waitFor(() => expect(screen.getByText("Revision 10")).toBeInTheDocument());
  });

  it("keeps confirmation bound to accepted state across a jobs-only refresh", async () => {
    const user = userEvent.setup();
    const initial = state({ revision: 4 });
    const candidateApi = candidate(initial);
    candidateApi.transport.refreshJobs = vi.fn(async () => ({
      session_id: "session-1",
      revision: 4,
      yaml_digest: "digest-4",
      jobs: [],
    }));
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const preview = screen.getByRole("button", { name: /^Preview forward/ });
    await user.click(preview);

    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    const confirm = screen.getByRole("button", { name: "I understand, continue" });
    await waitFor(() => expect(candidateApi.transport.refreshJobs).toHaveBeenCalled());
    expect(screen.getByText("Revision 4")).toBeInTheDocument();
    expect(confirm).toBeEnabled();

    await user.click(confirm);
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1",
      "preview_forward",
      4,
    ));
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(preview).toHaveFocus());
  });

  it.each([
    ["validate", "Validate"],
    ["preview_forward", /^Preview forward/],
    ["run", "Run"],
    ["compare", "Compare"],
    ["benchmark", "Benchmark"],
  ] as const)("confirms the %s kind with the current accepted revision", async (kind, name) => {
    const user = userEvent.setup();
    const initial = state({
      revision: 12,
      document: {
        ...documentState(),
        previews: {
          ...PREVIEWS,
          declared_run_kinds: ["forward", "compare", "benchmark"],
        },
      },
    });
    const candidateApi = candidate(initial);
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));

    if (kind === "compare" || kind === "benchmark") {
      // Kills a regression that submits a declared advanced job without explicit disclosure.
      await user.click(screen.getByRole("button", { name: "Advanced actions" }));
    }
    await user.click(screen.getByRole("button", { name }));
    await user.click(screen.getByRole("button", { name: "I understand, continue" }));
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1",
      kind,
      12,
    ));
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);
  });

  it("suppresses rapid double confirmation until the one deferred job settles", async () => {
    let resolveSubmit: ((next: EditorSession) => void) | undefined;
    const user = userEvent.setup();
    const initial = state({ revision: 12 });
    const candidateApi = candidate(initial);
    candidateApi.submitJob.mockImplementationOnce(() => new Promise<EditorSession>((resolve) => {
      resolveSubmit = resolve;
    }));
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const run = screen.getByRole("button", { name: "Run" });
    await user.click(run);
    const confirm = screen.getByRole("button", { name: "I understand, continue" });

    act(() => {
      confirm.click();
      confirm.click();
    });
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);
    expect(candidateApi.submitJob).toHaveBeenCalledWith("session-1", "run", 12);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
    expectRunningDescription(run);
    expectRunningDescription(screen.getByRole("button", { name: "Save YAML" }));
    expectRunningDescription(screen.getByRole("button", { name: "Refresh jobs" }));
    expect(run).not.toHaveFocus();
    confirm.click();
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);

    resolveSubmit?.(state({ revision: 13 }));
    await waitFor(() => expect(screen.getByText("Revision 13")).toBeInTheDocument());
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(run).toHaveFocus());
    expect(candidateApi.replaceYaml).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Validate" }));
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1",
      "validate",
      13,
    ));
    expect(candidateApi.submitJob).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("dialog", { name: "Trusted execution" }))
      .not.toBeInTheDocument();
  });

  it("does not let jobs-only refresh overwrite accepted validation", async () => {
    const user = userEvent.setup();
    const initial = state({ revision: 4 });
    const candidateApi = candidate(initial);
    candidateApi.transport.refreshJobs = vi.fn(async () => ({
      session_id: "session-1",
      revision: 4,
      yaml_digest: "digest-4",
      jobs: [],
    }));
    render(<SessionEditor initial={initial} transport={candidateApi.transport} />);
    await user.click(screen.getByRole("tab", { name: "Execute" }));
    const run = screen.getByRole("button", { name: "Run" });
    await user.click(run);

    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    await waitFor(() => expect(candidateApi.transport.refreshJobs).toHaveBeenCalled());
    expect(screen.getByText("Revision 4")).toBeInTheDocument();
    const confirm = screen.getByRole("button", { name: "I understand, continue" });
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    await waitFor(() => expect(candidateApi.submitJob).toHaveBeenCalledWith(
      "session-1", "run", 4,
    ));
  });

  it("derives onboarding counts and stale/current Forward state without scientific mutation", async () => {
    const user = userEvent.setup();
    const stale = state({
      document: {
        ...documentState(),
        forms: { sections: [], missing_required: ["runtime.mode", "outputs.path"] },
        validation: { ...VALIDATION, run_blocked: true },
      },
      jobs: [{
        job_id: "forward-stale",
        session_id: "session-1",
        kind: "preview_forward",
        revision: 1,
        yaml_digest: "old-digest",
        status: "succeeded",
        result: null,
        message: null,
        stale: true,
      }],
    });
    const staleCandidate = candidate(stale);
    const staleEditor = render(
      <SessionEditor initial={stale} transport={staleCandidate.transport} />,
    );
    const staleChecklist = screen.getByRole("region", { name: "First preview checklist" });
    expect(within(staleChecklist).getByText("2 required choices remain")).toBeVisible();
    expect(within(staleChecklist).getByText("Quick checks need attention")).toBeVisible();
    expect(within(staleChecklist).getByText("Forward preview stale")).toBeVisible();
    expect(staleCandidate.replaceYaml).not.toHaveBeenCalled();
    expect(staleCandidate.submitJob).not.toHaveBeenCalled();
    expect(staleCandidate.refresh).not.toHaveBeenCalled();
    staleEditor.unmount();

    const current = state({
      jobs: [{
        job_id: "forward-current",
        session_id: "session-1",
        kind: "preview_forward",
        revision: 0,
        yaml_digest: "current-digest",
        status: "succeeded",
        result: null,
        message: null,
        stale: false,
      }],
    });
    const currentCandidate = candidate(current);
    render(<SessionEditor initial={current} transport={currentCandidate.transport} />);
    expect(screen.queryByRole("region", { name: "First preview checklist" }))
      .not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Help" }));
    const currentChecklist = screen.getByRole("region", { name: "First preview checklist" });
    expect(within(currentChecklist).getByText("Required choices complete")).toBeVisible();
    expect(within(currentChecklist).getByText("Quick checks clean")).toBeVisible();
    expect(within(currentChecklist).getByText("Forward preview complete")).toBeVisible();
    expect(currentCandidate.replaceYaml).not.toHaveBeenCalled();
    expect(currentCandidate.submitJob).not.toHaveBeenCalled();
    expect(currentCandidate.refresh).not.toHaveBeenCalled();
    await user.click(within(currentChecklist).getByRole("button", { name: "Dismiss setup guide" }));
    expect(screen.queryByRole("region", { name: "First preview checklist" }))
      .not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Help" })).toBeVisible();
  });
});
