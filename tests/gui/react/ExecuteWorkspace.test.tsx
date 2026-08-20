import { act, cleanup, render, screen } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ExecutionActions, type ExecutionActionsProps } from "../../../src/rheplicant/gui/react/ExecutionActions";
import { useExecuteWorkspace, type ExecuteWorkspaceProps } from "../../../src/rheplicant/gui/react/ExecuteWorkspace";
import { OutputTargetCard } from "../../../src/rheplicant/gui/react/OutputTargetCard";
import type { EditorSession, JobProjection, OutputProductProjection, SessionTransport } from "../../../src/rheplicant/gui/react/types";
import { NO_DRAFT } from "../../../src/rheplicant/gui/react/drafts";

afterEach(cleanup);

const productNames = [
  "arrays", "aux", "taps", "assembly", "estimates", "parameters", "draws", "losses",
  "gradients", "covariance", "prediction_bands", "posterior_predictives", "identifiability",
  "scores", "recovery", "training_history", "timings", "refusals", "signal_paths", "compare",
  "benchmark", "chains",
];

function product(name: string, enabled = false): OutputProductProjection {
  return { name, enabled, format: "npz", formats: ["npz"], runs: [], keys: [], themes: [], expected_paths: [] };
}

function state(overrides: Partial<EditorSession> = {}): EditorSession {
  return {
    session_id: "session", revision: 4, yaml_digest: "current", dirty: false,
    validation_stale: false, can_undo: false, can_redo: false, jobs: [],
    outputs: {
      requested_yaml: "outputs: {}", resolved_yaml: "outputs: {}", resolution_note: "resolved",
      target_path: "/tmp/results", state: "ready_new", state_message: "ready", clobber: false,
      declared_runs: ["run"], products: productNames.map((name) => product(name)),
      report: { enabled: false, rows: [], columns: ["mean"], reference: null, relative: [], formats: ["json"], expected_paths: [] },
      audit_paths: [],
    },
    document: {
      yaml_text: "", svg: "", nodes: [], walk_order: [], forms: { sections: [], missing_required: [] },
      previews: {
        classes: [], axes: [], shapes: [], forward_cost: { label: "cost", estimated_milliseconds: 1, estimated_peak_megabytes: 1, n_freq: 1, nside: 1, lmax: 1, optimizations: [] },
        declared_run_kinds: ["run", "compare", "benchmark"],
      },
      validation: { findings: [], section_badges: [], selected_presets: [], preset_changes: [], run_blocked: false },
      base_diagram: { name: "base", svg: "", nodes: [], walk_order: [], counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 }, changed_nodes: [] },
      backend_diagram: { name: "backend", svg: "", nodes: [], walk_order: [], counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 }, changed_nodes: [] },
      variant_diagrams: [],
    },
    ...overrides,
  };
}

function transport(): SessionTransport {
  const unchanged = async () => state();
  return {
    refresh: vi.fn(unchanged), refreshJobs: vi.fn(async () => ({ session_id: "session", revision: 4, yaml_digest: "current", jobs: [] })),
    replaceYaml: vi.fn(unchanged), setField: vi.fn(unchanged), undo: vi.fn(unchanged), redo: vi.fn(unchanged), load: vi.fn(unchanged), save: vi.fn(unchanged),
    editNode: vi.fn(unchanged), moveNodeInstance: vi.fn(unchanged), composeNode: vi.fn(unchanged), placeNode: vi.fn(unchanged), setSnapshotBefore: vi.fn(unchanged),
    setOutputProduct: vi.fn(unchanged), setOutputReport: vi.fn(unchanged), submitJob: vi.fn(unchanged),
  };
}

const executeProps: ExecuteWorkspaceProps = {
  session: state(), jobs: [], transport: transport(), drafts: { draft: NO_DRAFT, begin: () => true, update: () => undefined, clear: () => undefined },
  disabledReason: null, onAccept: vi.fn(), onSubmit: vi.fn(), onRun: vi.fn(),
};

function renderExecute(overrides: Partial<ExecuteWorkspaceProps> = {}) {
  function Harness() {
    const surface = useExecuteWorkspace({ ...executeProps, ...overrides });
    return <>{surface.main}{surface.inspector}</>;
  }
  return render(<Harness />);
}

describe("progressive execute workspace", () => {
  it("discovers every product but mounts settings only for the selected product", async () => {
    // Kills a regression that eagerly mounts all product controls and makes a 22-product session unwieldy.
    const user = userEvent.setup();
    renderExecute();
    expect(screen.queryAllByRole("group", { name: /product settings/i })).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Add product" }));
    expect(screen.getAllByRole("option")).toHaveLength(22);
    await user.click(screen.getByRole("option", { name: "arrays" }));
    expect(screen.getAllByRole("group", { name: /product settings/i })).toHaveLength(1);
  });

  it("keeps terminal audit detail and legacy YAML artefacts out of the ordered execute request", () => {
    // Kills a regression that lets legacy details or a reordered section break Execute's Validation→Preview→Target→Products→Report→Actions journey.
    const session = state({ jobs: [queuedJob({ job_id: "terminal-job", status: "succeeded", result: { output: { audit_files: ["audit.json"] } } })] });
    renderExecute({ session });
    const validation = screen.getByRole("region", { name: "Validation readiness" });
    const preview = screen.getByRole("region", { name: "Forward preview summary" });
    const target = screen.getByRole("region", { name: "Output target" });
    const products = screen.getByRole("region", { name: "Scientific product selectors" });
    const report = screen.getByRole("button", { name: "Write report" });
    const actions = screen.getByRole("region", { name: "Execution actions" });
    expect(screen.queryByRole("tablist", { name: "Configuration artefacts" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Completed audit bundles" })).not.toBeInTheDocument();
    expect(screen.queryByText("terminal-job")).not.toBeInTheDocument();
    expect(validation.compareDocumentPosition(preview) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(preview.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(target.compareDocumentPosition(products) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(products.compareDocumentPosition(report) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(report.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("does not mount report controls until Write report is enabled", () => {
    // Kills a regression that exposes report configuration while report output is disabled.
    renderExecute();
    expect(screen.queryByLabelText("Report columns")).not.toBeInTheDocument();
  });

  it("gives the Run gate a stable blocking description when validation refuses execution", () => {
    // Kills a regression that leaves aria-describedby pointing to a missing run-blocked explanation.
    const session = state({ document: { ...state().document, validation: { ...state().document.validation, run_blocked: true } } });
    renderExecute({ session });
    expect(screen.getByRole("button", { name: "Run" })).toHaveAccessibleDescription("Run is blocked until validation is repaired.");
  });

  it("keeps a 22-product listbox on one keyboard tab stop and selects from its roving focus", async () => {
    // Kills a regression that exposes listbox options without Arrow/Home/End navigation or a single roving focus stop.
    const user = userEvent.setup();
    renderExecute();
    await user.click(screen.getByRole("button", { name: "Add product" }));
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(22);
    expect(options[0]).toHaveAttribute("tabindex", "0");
    for (const option of options.slice(1)) expect(option).toHaveAttribute("tabindex", "-1");
    expect(options[0]).toHaveFocus();
    await user.keyboard("{End}");
    expect(options[21]).toHaveFocus();
    expect(options[21]).toHaveAttribute("tabindex", "0");
    await user.keyboard("{Home}{ArrowDown}{Enter}");
    expect(screen.getByRole("group", { name: "aux product settings" })).toBeInTheDocument();
  });

  it("keeps enabled product summaries compact until exactly one is expanded", async () => {
    // Kills a regression that mounts a full settings group for every enabled output product.
    const user = userEvent.setup();
    const session = state({ outputs: { ...state().outputs, products: productNames.map((name) => product(name, name === "arrays" || name === "chains")) } });
    renderExecute({ session });
    expect(screen.queryAllByRole("group", { name: /product settings/i })).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Expand arrays product settings" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand chains product settings" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand arrays product settings" }));
    expect(screen.getAllByRole("group", { name: /product settings/i })).toHaveLength(1);
  });

  it("uses display jobs, rather than accepted session jobs, for live duplicate blocking", () => {
    // Kills a regression that reads session.jobs and leaves a completed accept response blocking a fresh action.
    renderExecute({ session: state({ jobs: [queuedJob({})] }), jobs: [] });
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it.each([
    ["ready_new", true], ["replace_owned", true], ["blocked_existing", false], ["blocked_foreign", false],
    ["ambiguous_recovery", false], ["blocked_unsafe", false], ["unavailable", false],
  ] as const)("gates Run from the %s target state", (targetState, runnable) => {
    // Kills a regression that hard-codes targetRunnable and permits blocked output publication.
    renderExecute({ session: state({ outputs: { ...state().outputs, state: targetState } }) });
    expect(screen.getByRole("button", { name: "Run" })).toHaveProperty("disabled", !runnable);
  });

  it("derives primary and advanced action disclosure from real workspace jobs and accepted previews", () => {
    // Kills a regression that forces current previews or advanced kinds instead of reading the supplied workspace projections.
    const undeclaredAdvanced = state({ document: { ...state().document, previews: { ...state().document.previews, declared_run_kinds: ["run"] } } });
    renderExecute({ session: undeclaredAdvanced, jobs: [] });
    expect(screen.getByRole("button", { name: "Preview forward" })).toHaveClass("primary-action");
    expect(screen.queryByRole("button", { name: "Advanced actions" })).not.toBeInTheDocument();
    cleanup();
    const preview = queuedJob({ kind: "preview_forward", status: "succeeded", stale: false });
    renderExecute({ session: undeclaredAdvanced, jobs: [preview] });
    expect(screen.getByRole("button", { name: "Run" })).toHaveClass("primary-action");
    expect(screen.queryByRole("button", { name: "Advanced actions" })).not.toBeInTheDocument();
  });

  it("runs the lazy report action through the parent and mounts its designer only after the accepted response", async () => {
    // Kills a regression that sends the wrong report payload/message or mounts ReportDesigner before the parent accepts the full session.
    const api = transport();
    const initial = state({ outputs: { ...state().outputs, declared_runs: ["first", "second"], report: { enabled: false, rows: [], columns: ["mean", "seconds"], reference: "second", relative: ["mean_sigma"], formats: ["json"], expected_paths: [] } } });
    const accepted = state({ outputs: { ...initial.outputs, report: { ...initial.outputs.report, enabled: true, rows: ["first"] } } });
    api.setOutputReport = vi.fn(async () => accepted);
    let accept: ((next: EditorSession) => void) | undefined;
    let captured: (() => Promise<EditorSession>) | undefined;
    const onRun = vi.fn((action: () => Promise<EditorSession>) => { captured = action; });
    function Harness() {
      const [session, setSession] = useState(initial);
      accept = setSession;
      const surface = useExecuteWorkspace({ ...executeProps, session, transport: api, jobs: [], onRun, onAccept: vi.fn() });
      return <>{surface.main}{surface.inspector}</>;
    }
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("button", { name: "Write report" }));
    expect(screen.queryByLabelText("Report columns")).not.toBeInTheDocument();
    expect(api.setOutputReport).not.toHaveBeenCalled();
    expect(onRun).toHaveBeenCalledOnce();
    expect(onRun.mock.calls[0][1]).toBe("Enabled report output");
    const next = await captured?.();
    expect(api.setOutputReport).toHaveBeenCalledWith("session", true, ["first"], ["mean", "seconds"], "second", ["mean_sigma"], ["json"], 4);
    act(() => accept?.(next as EditorSession));
    expect(screen.getByLabelText("Report columns")).toBeInTheDocument();
  });

  it("names every closed output target state and alerts only blocked states", () => {
    // Kills a regression that treats a blocked, unsafe, or unavailable target as runnable.
    const labels = {
      ready_new: "New target ready", blocked_existing: "Existing target blocked", blocked_foreign: "Foreign target blocked",
      replace_owned: "Owned target can be replaced", ambiguous_recovery: "Recovery needs review", blocked_unsafe: "Unsafe target blocked", unavailable: "Target unavailable",
    } as const;
    for (const [stateName, label] of Object.entries(labels)) {
      const session = state({ outputs: { ...state().outputs, state: stateName as EditorSession["outputs"]["state"] } });
      const { unmount } = render(<OutputTargetCard output={session.outputs} />);
      expect(screen.getByText(label)).toBeInTheDocument();
      const visibleLabel = screen.getByText(label);
      const target = visibleLabel.closest("[role]");
      expect(visibleLabel).toBeVisible();
      expect(target).toHaveAttribute("role", "status");
      expect(target).toHaveTextContent(label);
      expect(screen.getByText("ready")).toBeVisible();
      unmount();
    }
  });
});

const actionDefaults: ExecutionActionsProps = {
  jobs: [], revision: 4, yamlDigest: "current", previewCurrent: false, runDeclared: true,
  targetRunnable: true, declaredKinds: ["run"], disabledReason: null, onSubmit: vi.fn(),
};
const executionActions = (overrides: Partial<ExecutionActionsProps>) => (
  <ExecutionActions {...actionDefaults} {...overrides} />
);
const renderExecutionActions = (overrides: Partial<ExecutionActionsProps>) => render(executionActions(overrides));
const queuedJob = (overrides: Partial<JobProjection>): JobProjection => ({
  job_id: "queued-job", session_id: "session", kind: "run", revision: 4, yaml_digest: "current",
  status: "queued", result: null, message: null, stale: false, ...overrides,
});

describe("execution action priority", () => {
  it.each([
    [false, false, "Run disabled: no run is declared."],
    [true, false, "Run disabled: repair the output target."],
  ])("gives an unavailable Run one adjacent semantic reason", (
    runDeclared,
    targetRunnable,
    reason,
  ) => {
    const { container } = render(executionActions({ runDeclared, targetRunnable }));

    const run = screen.getByRole("button", { name: "Run" });
    expect(run).toBeDisabled();
    expect(run).toHaveAccessibleDescription(reason);
    expect(container.querySelectorAll("#execution-run-disabled")).toHaveLength(1);
    const reasonChip = screen.getByText(reason).closest("[role]");
    expect(reasonChip).toHaveAttribute("role", "status");
    expect(reasonChip).toHaveClass("status-disabled");
    expect(reasonChip).toHaveTextContent(reason);
  });

  it("promotes preview before readiness and run after a current preview", () => {
    // Kills a regression that marks Run primary before its required preview is current.
    const { rerender } = renderExecutionActions({ previewCurrent: false, jobs: [] });
    expect(screen.getByRole("button", { name: "Preview forward" })).toHaveClass("primary-action");
    rerender(executionActions({ previewCurrent: true, targetRunnable: true, jobs: [] }));
    expect(screen.getByRole("button", { name: "Run" })).toHaveClass("primary-action");
  });

  it("only disables the exact active duplicate and discloses declared advanced actions", async () => {
    // Kills a regression that blocks a new-revision/digest run or invents undeclared advanced jobs.
    const user = userEvent.setup();
    renderExecutionActions({
      jobs: [queuedJob({ kind: "run", revision: 4, yaml_digest: "other" })],
      declaredKinds: ["run", "compare", "benchmark"],
    });
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Compare" })).toBeNull();
    await user.click(screen.getByText("Advanced actions"));
    expect(screen.getByRole("button", { name: "Compare" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Benchmark" })).toBeEnabled();
  });

  it("disables an active duplicate only when its kind, revision, and digest all match", () => {
    // Kills a regression that omits active-job suppression after matching the complete job identity.
    renderExecutionActions({ jobs: [queuedJob({})] });
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("does not disclose advanced controls for undeclared jobs", () => {
    // Kills a regression that invents compare or benchmark actions absent from the accepted declaration.
    renderExecutionActions({ declaredKinds: ["run"] });
    expect(screen.queryByRole("button", { name: "Advanced actions" })).toBeNull();
  });
});
