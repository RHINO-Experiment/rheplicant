import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ExecutionActions as ExecutionActionsSurface, type ExecutionActionsProps } from "../../../src/rheplicant/gui/react/ExecutionActions";
import { useExecuteWorkspace, type ExecuteWorkspaceProps } from "../../../src/rheplicant/gui/react/ExecuteWorkspace";
import { OutputTargetCard } from "../../../src/rheplicant/gui/react/OutputTargetCard";
import type { EditorSession, JobProjection, OutputProductProjection, SessionTransport } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const productNames = [
  "arrays", "aux", "taps", "assembly", "estimates", "parameters", "draws", "losses",
  "gradients", "covariance", "prediction_bands", "posterior_predictives", "identifiability",
  "scores", "recovery", "training_history", "timings", "refusals", "signal_paths", "compare",
  "benchmark", "chains", "run_diagnostics",
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
  session: state(), jobs: [], transport: transport(),
  disabledReason: null, onSubmit: vi.fn(), onRun: vi.fn(),
};

function renderExecute(overrides: Partial<ExecuteWorkspaceProps> = {}) {
  function Harness() {
    const surface = useExecuteWorkspace({ ...executeProps, ...overrides });
    return <>{surface.main}{surface.inspector}</>;
  }
  return render(<Harness />);
}

describe("progressive execute workspace", () => {
  it.each([
    [[], "Full validation not run", "neutral"],
    [[queuedJob({ kind: "validate", status: "queued" })], "Full validation queued for revision 4", "neutral"],
    [[queuedJob({ kind: "validate", status: "running" })], "Full validation running for revision 4", "neutral"],
    [[queuedJob({ kind: "validate", status: "succeeded" })], "Full validation current for revision 4", "success"],
    [[queuedJob({ kind: "validate", status: "succeeded", revision: 3, yaml_digest: "old", stale: false })], "Full validation stale from revision 3", "stale"],
    [[queuedJob({ kind: "validate", status: "refused", message: "repair validation" })], "Full validation refused: repair validation", "danger"],
    [[queuedJob({ kind: "validate", status: "error", message: "bounded failure" })], "Full validation internal error: bounded failure", "danger"],
  ] as const)("shows the latest content-bound Full validation state %#", (jobs, label, tone) => {
    // Kills collapsing Full validation into the Quick gate or ignoring revision/digest identity.
    renderExecute({ jobs: [...jobs] });
    const evidence = screen.getByText(label).closest("[role]");
    expect(evidence).toHaveAttribute("role", "status");
    expect(evidence).toHaveClass(`status-${tone}`);
  });

  it("reads the newest validate job rather than the first", () => {
    // Kills a regression that reports the first validate job and hides a newer refusal behind an older success.
    renderExecute({ jobs: [
      queuedJob({ job_id: "older-validate", kind: "validate", status: "succeeded" }),
      queuedJob({ job_id: "newer-validate", kind: "validate", status: "refused", message: "repair validation" }),
    ] });
    expect(screen.getByText("Full validation refused: repair validation").closest("[role]")).toHaveClass("status-danger");
    expect(screen.queryByText("Full validation current for revision 4")).toBeNull();
  });

  it("reports Quick checks separately from the latest Full validation", () => {
    // Kills the former unconditional "Quick and full validation are ready" claim.
    const blocked = state({ document: { ...state().document, validation: { ...state().document.validation, run_blocked: true } } });
    renderExecute({ session: blocked, jobs: [] });
    expect(screen.getByText("Quick checks need attention").closest("[role]")).toHaveClass("status-danger");
    expect(screen.getByText("Full validation not run").closest("[role]")).toHaveClass("status-neutral");
  });

  it("discovers every product but mounts settings only for the selected product", async () => {
    // Kills a regression that eagerly mounts all product controls and makes a 23-product session unwieldy.
    const user = userEvent.setup();
    renderExecute();
    expect(screen.queryAllByRole("group", { name: /product settings/i })).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Add product" }));
    expect(screen.getAllByRole("option")).toHaveLength(23);
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

  it("keeps a 23-product listbox on one keyboard tab stop and selects from its roving focus", async () => {
    // Kills a regression that exposes listbox options without Arrow/Home/End navigation or a single roving focus stop.
    const user = userEvent.setup();
    renderExecute();
    await user.click(screen.getByRole("button", { name: "Add product" }));
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(23);
    expect(options[0]).toHaveAttribute("tabindex", "0");
    for (const option of options.slice(1)) expect(option).toHaveAttribute("tabindex", "-1");
    expect(options[0]).toHaveFocus();
    await user.keyboard("{End}");
    // The last option, not a fixed index: {End} means "the end of the list",
    // and the length is pinned above, so this stays honest as the catalogue grows.
    const last = options[options.length - 1];
    expect(last).toHaveFocus();
    expect(last).toHaveAttribute("tabindex", "0");
    await user.keyboard("{Home}{ArrowDown}{Enter}");
    expect(screen.getByRole("group", { name: "aux product settings" })).toBeInTheDocument();
  });

  it("keeps a preview current across a Save that bumps the revision without changing the YAML", () => {
    // Kills reinstating `job.revision === revision` on the preview identity. `mark_saved` bumps
    // the revision and leaves `yaml_text` — and so the digest — untouched, and `Save YAML` calls
    // it, so one Save retired a verdict that still described this exact document and made this
    // reader disagree with OnboardingChecklist and with deriveFullValidation about the same job.
    const preview = queuedJob({
      kind: "preview_forward", status: "succeeded", revision: 3, yaml_digest: "current", stale: false,
    });
    renderExecute({ jobs: [preview] });

    expect(screen.getByRole("button", { name: "Run" })).toHaveClass("primary-action");
    expect(screen.getByRole("button", { name: "Preview forward" })).not.toHaveClass("primary-action");
  });

  it("retires a preview whose YAML digest the document has moved past, whatever its revision", () => {
    // The other side of the same identity: dropping the digest comparison would promote Run from
    // a preview of text this document no longer holds.
    const preview = queuedJob({
      kind: "preview_forward", status: "succeeded", revision: 4, yaml_digest: "old", stale: false,
    });
    renderExecute({ jobs: [preview] });

    expect(screen.getByRole("button", { name: "Preview forward" })).toHaveClass("primary-action");
    expect(screen.getByRole("button", { name: "Run" })).not.toHaveClass("primary-action");
  });

  it("describes a blocked action with every reason that holds, not the first one to hold", () => {
    // Kills `runBlocked ? "run-blocked-reason" : disabledReason`. Both causes can hold at once,
    // and picking one left the blocker the user can actually clear — the unsaved draft —
    // undescribed on every action button.
    const blocked = state({ document: { ...state().document, validation: { ...state().document.validation, run_blocked: true } } });
    function Harness() {
      const surface = useExecuteWorkspace({
        ...executeProps, session: blocked, disabledReason: "draft-blocked-reason",
      });
      return <>
        <p id="draft-blocked-reason">Editing is blocked by an unsaved draft.</p>
        {surface.main}
      </>;
    }
    render(<Harness />);

    const run = screen.getByRole("button", { name: "Run" });
    expect(run).toBeDisabled();
    const described = (run.getAttribute("aria-describedby") ?? "").split(" ").filter(Boolean);
    expect(described).toEqual(["draft-blocked-reason", "run-blocked-reason"]);
    for (const id of described) expect(document.getElementById(id)).not.toBeNull();
    expect(run).toHaveAccessibleDescription(
      "Editing is blocked by an unsaved draft. Run is blocked until validation is repaired.",
    );
  });

  it("lists the enabled products alone and leaves the other twenty to the Add product search", () => {
    // Kills replacing `products.filter((product) => product.enabled)` with `products`. §7.3 splits
    // this surface in two — the enabled products in view, all 23 reachable through search — and
    // asserting only that the two enabled ones are present cannot tell the split from no split.
    const session = state({ outputs: { ...state().outputs, products: productNames.map((name) => product(name, name === "arrays" || name === "chains")) } });
    renderExecute({ session });

    const enabled = screen.getByRole("list", { name: "Enabled products" });
    expect(within(enabled).getAllByRole("button").map((button) => button.textContent))
      .toEqual(["Expand arrays product settings", "Expand chains product settings"]);
    for (const name of productNames.filter((name) => name !== "arrays" && name !== "chains")) {
      expect(screen.queryByRole("button", { name: `Expand ${name} product settings` })).toBeNull();
    }
  });

  it("renders no enabled-products list at all when nothing is enabled", () => {
    // The other side of the same split: an empty list is a heading over nothing.
    renderExecute();

    expect(screen.queryByRole("list", { name: "Enabled products" })).toBeNull();
    expect(screen.queryAllByRole("button", { name: /^Expand .* product settings$/ })).toHaveLength(0);
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
    const onRun = vi.fn((action: () => Promise<EditorSession>, _message: string) => { captured = action; });
    function Harness() {
      const [session, setSession] = useState(initial);
      accept = setSession;
      const surface = useExecuteWorkspace({ ...executeProps, session, transport: api, jobs: [], onRun });
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

// Advanced disclosure is controlled: useExecuteWorkspace owns it in the app, and this harness
// owns it here so every existing case below renders unchanged.
function ExecutionActions(props: Omit<ExecutionActionsProps, "advanced" | "onAdvanced">) {
  const [advanced, setAdvanced] = useState(false);
  return <ExecutionActionsSurface {...props} advanced={advanced} onAdvanced={setAdvanced} />;
}

const actionDefaults: Omit<ExecutionActionsProps, "advanced" | "onAdvanced"> = {
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

  it("renders one control and one active reason for a kind declared twice", async () => {
    // Kills taking declaredKinds as given: a duplicate produced two buttons under one React key
    // and two spans sharing the id `execution-compare-active`, so the button's aria-describedby
    // named an ambiguous element and the same queued job was announced twice.
    const user = userEvent.setup();
    const { container } = renderExecutionActions({
      jobs: [queuedJob({ kind: "compare", revision: 4, yaml_digest: "current" })],
      declaredKinds: ["run", "compare", "compare"],
    });

    await user.click(screen.getByRole("button", { name: "Advanced actions" }));
    expect(screen.getAllByRole("button", { name: "Compare" })).toHaveLength(1);
    expect(container.querySelectorAll("#execution-compare-active")).toHaveLength(1);
    expect(screen.getAllByText("Queued Compare at revision 4")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Compare" }))
      .toHaveAccessibleDescription("Queued Compare at revision 4");
  });

  it("does not disclose advanced controls for undeclared jobs", () => {
    // Kills a regression that invents compare or benchmark actions absent from the accepted declaration.
    renderExecutionActions({ declaredKinds: ["run"] });
    expect(screen.queryByRole("button", { name: "Advanced actions" })).toBeNull();
  });
});

function renderSwitchable(overrides: Partial<ExecuteWorkspaceProps> = {}) {
  function Harness() {
    // Mirrors SessionEditor: every workspace hook runs on every render, only the surface is swapped.
    const [active, setActive] = useState(true);
    const surface = useExecuteWorkspace({ ...executeProps, ...overrides });
    return (
      <>
        <button type="button" onClick={() => setActive((value) => !value)}>Toggle workspace</button>
        {active ? <>{surface.main}{surface.inspector}</> : <p>Another workspace</p>}
      </>
    );
  }
  return render(<Harness />);
}

const openPicker = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: "Add product" }));
  return screen.getByRole("searchbox", { name: "Filter products" });
};

describe("execute product picker search", () => {
  it("exposes all 23 products for an empty query and restores them when the query is cleared", async () => {
    // Kills a regression that hides the catalogue behind a query and breaks spec 7.3 discoverability.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    expect(screen.getAllByRole("option")).toHaveLength(23);
    await user.type(filter, "chain");
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["chains"]);
    await user.clear(filter);
    expect(screen.getAllByRole("option")).toHaveLength(23);
  });

  it("filters case-insensitively on a substring of the product name", async () => {
    // Kills a regression that anchors the match at the start of the name or compares case-sensitively.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "PRE");
    expect(screen.getAllByRole("option").map((option) => option.textContent))
      .toEqual(["prediction_bands", "posterior_predictives"]);
  });

  it("roves the keyboard over the filtered options rather than the whole catalogue", async () => {
    // Kills a regression that indexes Arrow/Home/End into the unfiltered 23 and lands outside the filtered list.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.keyboard("{End}");
    const unfiltered = screen.getAllByRole("option");
    expect(unfiltered[unfiltered.length - 1]).toHaveFocus();
    await user.click(filter);
    await user.type(filter, "PRE");
    const filtered = screen.getAllByRole("option");
    expect(filtered).toHaveLength(2);
    expect(filtered[0]).toHaveAttribute("tabindex", "0");
    expect(filtered[1]).toHaveAttribute("tabindex", "-1");
    await user.keyboard("{ArrowDown}");
    expect(filtered[0]).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(filtered[1]).toHaveFocus();
    expect(filtered[1]).toHaveAttribute("tabindex", "0");
    await user.keyboard("{ArrowDown}");
    expect(filtered[0]).toHaveFocus();
    await user.keyboard("{End}");
    expect(filtered[1]).toHaveFocus();
    await user.keyboard("{Home}");
    expect(filtered[0]).toHaveFocus();
  });

  it("announces an empty result instead of leaving a silent blank listbox", async () => {
    // Kills a regression that renders an empty listbox with no readable, announced explanation.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "zzz");
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    const empty = screen.getByText('No products match "zzz"');
    expect(empty).toBeVisible();
    expect(empty.closest("[role]")).toHaveAttribute("role", "status");
  });

  it("selects the active filtered option with Enter", async () => {
    // Kills a regression that drops Enter selection once the list is filtered.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "CHAIN");
    await user.keyboard("{ArrowDown}{Enter}");
    expect(screen.getByRole("group", { name: "chains product settings" })).toBeInTheDocument();
    expect(document.activeElement).not.toBe(document.body);
  });

  it("selects the active filtered option with Space", async () => {
    // Kills dropping Space as a selection key; a bare keydown carries no synthesised button click behind it.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "CHAIN");
    await user.keyboard("{ArrowDown}");
    const active = screen.getByRole("option", { name: "chains" });
    expect(active).toHaveFocus();
    fireEvent.keyDown(active, { key: " " });
    expect(screen.getByRole("group", { name: "chains product settings" })).toBeInTheDocument();
  });

  it("selects a clicked filtered option with the mouse", async () => {
    // Kills a regression that maps the click index onto the unfiltered catalogue and opens the wrong product.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "recov");
    await user.click(screen.getByRole("option", { name: "recovery" }));
    expect(screen.getByRole("group", { name: "recovery product settings" })).toBeInTheDocument();
  });

  it("returns focus to the opener after a selection rather than dropping it on the document body", async () => {
    // Kills a regression that closes the picker without restoring focus and strands keyboard users on <body>.
    const user = userEvent.setup();
    renderExecute();
    const opener = screen.getByRole("button", { name: "Add product" });
    await user.click(opener);
    await user.click(screen.getByRole("option", { name: "arrays" }));
    expect(document.activeElement).not.toBe(document.body);
    expect(opener).toHaveFocus();
  });
});

describe("execute view state ownership", () => {
  it("keeps the open picker and its query across a workspace switch", async () => {
    // Kills a regression that parks picker disclosure inside the unmounted surface and loses it on every switch.
    const user = userEvent.setup();
    renderSwitchable();
    const filter = await openPicker(user);
    await user.type(filter, "PRE");
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.queryByRole("listbox", { name: "Available products" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.getByRole("searchbox", { name: "Filter products" })).toHaveValue("PRE");
    expect(screen.getAllByRole("option").map((option) => option.textContent))
      .toEqual(["prediction_bands", "posterior_predictives"]);
  });

  it("keeps the expanded product across a workspace switch", async () => {
    // Kills a regression that returns the expanded-product state to a component the workspace switch unmounts.
    const user = userEvent.setup();
    renderSwitchable();
    await openPicker(user);
    await user.click(screen.getByRole("option", { name: "timings" }));
    expect(screen.getByRole("group", { name: "timings product settings" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.queryByRole("group", { name: /product settings/i })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.getByRole("group", { name: "timings product settings" })).toBeInTheDocument();
  });
});

const REQUESTED_YAML = "outputs:\n  clobber: false\n\n  # trailing comment\n";
const RESOLVED_YAML = "runtime:\n  jax_enable_x64: true\noutputs:\n  clobber: false\n";
const RESOLUTION_NOTE = "Preset-merged preview; the final resolved audit file appears after Run.";
const comparisonSession = () => state({
  outputs: { ...state().outputs, requested_yaml: REQUESTED_YAML, resolved_yaml: RESOLVED_YAML, resolution_note: RESOLUTION_NOTE },
});

describe("execute advanced disclosure ownership", () => {
  it("takes advanced action disclosure from its props and reports the toggle upward", async () => {
    // Kills a regression that restores a local advanced useState inside ExecutionActions.
    const user = userEvent.setup();
    const onAdvanced = vi.fn();
    render(<ExecutionActionsSurface {...actionDefaults} declaredKinds={["run", "compare", "benchmark"]} advanced onAdvanced={onAdvanced} />);
    expect(screen.getByRole("button", { name: "Compare" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Advanced actions" })).toHaveAttribute("aria-expanded", "true");
    await user.click(screen.getByRole("button", { name: "Advanced actions" }));
    expect(onAdvanced).toHaveBeenCalledWith(false);
    expect(screen.getByRole("button", { name: "Compare" })).toBeInTheDocument();
  });

  it("keeps advanced action disclosure across a workspace switch", async () => {
    // Kills a regression that parks advanced disclosure inside the surface the workspace switch unmounts.
    const user = userEvent.setup();
    renderSwitchable();
    await user.click(screen.getByRole("button", { name: "Advanced actions" }));
    expect(screen.getByRole("button", { name: "Compare" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.queryByRole("button", { name: "Compare" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.getByRole("button", { name: "Advanced actions" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: "Compare" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Benchmark" })).toBeInTheDocument();
  });
});

describe("advanced requested and resolved comparison", () => {
  it("stays collapsed by default and sits last in the output request, ahead of the action bar", () => {
    // Kills a regression that promotes the advanced comparison above the primary Execute order or opens it eagerly.
    renderExecute({ session: comparisonSession() });
    const toggle = screen.getByRole("button", { name: "Requested and resolved YAML" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("region", { name: "Requested YAML" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Preset-resolved YAML" })).toBeNull();
    expect(screen.queryByText(RESOLUTION_NOTE)).toBeNull();
    expect(screen.queryByText(REQUESTED_YAML)).toBeNull();
    expect(screen.queryByText(RESOLVED_YAML)).toBeNull();
    const products = screen.getByRole("region", { name: "Scientific product selectors" });
    const report = screen.getByRole("button", { name: "Write report" });
    const actions = screen.getByRole("region", { name: "Execution actions" });
    expect(products.compareDocumentPosition(toggle) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(report.compareDocumentPosition(toggle) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(toggle.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("shows the exact requested YAML, resolved YAML and resolution note under their own labels", async () => {
    // Kills a regression that mislabels, swaps, reformats or re-serialises the two scientific YAML texts.
    const user = userEvent.setup();
    renderExecute({ session: comparisonSession() });
    await user.click(screen.getByRole("button", { name: "Requested and resolved YAML" }));
    expect(screen.getByRole("button", { name: "Requested and resolved YAML" })).toHaveAttribute("aria-expanded", "true");

    const requested = screen.getByRole("region", { name: "Requested YAML" });
    const resolved = screen.getByRole("region", { name: "Preset-resolved YAML" });
    const note = screen.getByRole("region", { name: "Resolution note" });
    expect(within(requested).getByRole("heading", { name: "Requested YAML" })).toBeVisible();
    expect(within(resolved).getByRole("heading", { name: "Preset-resolved YAML" })).toBeVisible();
    expect(within(note).getByRole("heading", { name: "Resolution note" })).toBeVisible();
    expect(requested.querySelector("pre")?.textContent).toBe(REQUESTED_YAML);
    expect(resolved.querySelector("pre")?.textContent).toBe(RESOLVED_YAML);
    expect(requested.querySelector("pre")?.textContent).not.toBe(RESOLVED_YAML);
    expect(resolved.querySelector("pre")?.textContent).not.toBe(REQUESTED_YAML);
    expect(within(note).getByText(RESOLUTION_NOTE)).toBeVisible();
  });

  it("restores the comparison without resurrecting the deleted legacy artefact surface", async () => {
    // Kills a regression that brings back the artefact tablist, audit-bundle region or audit file links.
    const user = userEvent.setup();
    renderExecute({ session: comparisonSession() });
    await user.click(screen.getByRole("button", { name: "Requested and resolved YAML" }));
    expect(screen.queryByRole("tablist", { name: "Configuration artefacts" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Completed audit bundles" })).toBeNull();
    expect(screen.queryByRole("link", { name: "config.resolved.yaml" })).toBeNull();
  });

  it("keeps the comparison disclosure across a workspace switch", async () => {
    // Kills a regression that parks the comparison disclosure in a component the workspace switch unmounts.
    const user = userEvent.setup();
    renderSwitchable({ session: comparisonSession() });
    await user.click(screen.getByRole("button", { name: "Requested and resolved YAML" }));
    expect(screen.getByRole("region", { name: "Requested YAML" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.queryByRole("region", { name: "Requested YAML" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Toggle workspace" }));
    expect(screen.getByRole("button", { name: "Requested and resolved YAML" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("region", { name: "Requested YAML" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Preset-resolved YAML" })).toBeInTheDocument();
  });
});

describe("execute full validation staleness", () => {
  it.each([
    ["the server stale flag alone", queuedJob({ kind: "validate", status: "succeeded", stale: true }), "Full validation stale from revision 4"],
    ["a digest the document has moved past", queuedJob({ kind: "validate", status: "succeeded", revision: 3, yaml_digest: "old", stale: false }), "Full validation stale from revision 3"],
    ["a refusal the repair superseded", queuedJob({ kind: "validate", status: "refused", revision: 3, yaml_digest: "old", stale: true, message: "repair validation" }), "Full validation stale from revision 3"],
    ["an internal error the repair superseded", queuedJob({ kind: "validate", status: "error", revision: 3, yaml_digest: "old", stale: true, message: "bounded failure" }), "Full validation stale from revision 3"],
    ["a queued job the repair superseded", queuedJob({ kind: "validate", status: "queued", stale: true }), "Full validation stale from revision 4"],
    ["a running job the repair superseded", queuedJob({ kind: "validate", status: "running", stale: true }), "Full validation stale from revision 4"],
  ] as const)("reports %s as stale rather than as progress on this document", (_case, job, label) => {
    // Kills applying staleness on the succeeded path alone, which claimed a superseded verdict for a repaired document.
    renderExecute({ jobs: [job] });
    const evidence = screen.getByText(label).closest("[role]");
    expect(evidence).toHaveAttribute("role", "status");
    expect(evidence).toHaveClass("status-stale");
  });

  it("never renders a superseded refusal as a refusal of the repaired document", () => {
    // Kills the release blocker: Execute showed red "refused" for a document the server had already marked stale.
    renderExecute({ jobs: [queuedJob({
      kind: "validate", status: "refused", revision: 3, yaml_digest: "old", stale: true, message: "repair validation",
    })] });
    expect(screen.queryByText("Full validation refused: repair validation")).toBeNull();
    expect(screen.queryByText(/Full validation refused/)).toBeNull();
    expect(screen.getByText("Full validation stale from revision 3").closest("[role]")).toHaveClass("status-stale");
  });

  it("prefers the last non-stale validate job over a newer stale one", () => {
    // Kills selecting the last validate job in array order and hiding a bound verdict behind a superseded one.
    renderExecute({ jobs: [
      queuedJob({ job_id: "bound", kind: "validate", status: "succeeded" }),
      queuedJob({ job_id: "superseded", kind: "validate", status: "refused", revision: 5, yaml_digest: "next", stale: true, message: "repair validation" }),
    ] });
    expect(screen.getByText("Full validation current for revision 4").closest("[role]")).toHaveClass("status-success");
    expect(screen.queryByText(/Full validation refused/)).toBeNull();
  });
});

describe("execute product picker dismissal", () => {
  it("declares the popup it opens on the opener itself", () => {
    // Kills an opener that expands a listbox while telling assistive technology nothing about it.
    renderExecute();
    const opener = screen.getByRole("button", { name: "Add product" });
    expect(opener).toHaveAttribute("aria-haspopup", "listbox");
    expect(opener).toHaveAttribute("aria-expanded", "false");
    expect(opener).toHaveAttribute("aria-controls", "available-products");
  });

  it("reports the picker as expanded and names the listbox it controls", async () => {
    // Kills a static aria-expanded that never follows the disclosure it claims to describe.
    const user = userEvent.setup();
    renderExecute();
    await openPicker(user);
    const opener = screen.getByRole("button", { name: "Add product" });
    expect(opener).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("listbox", { name: "Available products" }).id)
      .toBe(opener.getAttribute("aria-controls"));
  });

  it("dismisses the picker with Escape from the filter and returns focus to the opener", async () => {
    // Kills a picker whose only exit is committing a product the user never wanted.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.click(filter);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox", { name: "Available products" })).toBeNull();
    expect(screen.queryByRole("group", { name: /product settings/i })).toBeNull();
    expect(document.activeElement).not.toBe(document.body);
    expect(screen.getByRole("button", { name: "Add product" })).toHaveFocus();
  });

  it("dismisses the picker with Escape from an option and returns focus to the opener", async () => {
    // Kills an Escape bound to the filter alone, which strands a keyboard user roving the listbox.
    const user = userEvent.setup();
    renderExecute();
    await openPicker(user);
    expect(screen.getAllByRole("option")[0]).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox", { name: "Available products" })).toBeNull();
    expect(screen.queryByRole("group", { name: /product settings/i })).toBeNull();
    expect(document.activeElement).not.toBe(document.body);
    expect(screen.getByRole("button", { name: "Add product" })).toHaveFocus();
  });

  it("dismisses the picker with Escape when the query matches nothing", async () => {
    // Kills an Escape parked behind the empty-match guard, which is exactly when the user wants out.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.click(filter);
    await user.type(filter, "zzz");
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox", { name: "Available products" })).toBeNull();
    expect(screen.getByRole("button", { name: "Add product" })).toHaveFocus();
  });

  it("collapses the picker when the opener is pressed again", async () => {
    // Kills an opener that reports aria-expanded while a second press only wipes the query.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.type(filter, "PRE");
    const opener = screen.getByRole("button", { name: "Add product" });
    await user.click(opener);
    expect(screen.queryByRole("listbox", { name: "Available products" })).toBeNull();
    expect(opener).toHaveAttribute("aria-expanded", "false");
    expect(opener).toHaveFocus();
    await user.click(opener);
    expect(screen.getByRole("listbox", { name: "Available products" })).toBeInTheDocument();
    expect(screen.getByRole("searchbox", { name: "Filter products" })).toHaveValue("");
  });

  it("leaves Home and End to the filter caret while the arrows still enter the list", async () => {
    // Kills binding Home/End on the textbox, which steals the caret keys the combobox pattern reserves for it.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.click(filter);
    await user.type(filter, "PRE");
    expect(screen.getAllByRole("option")).toHaveLength(2);
    expect(fireEvent.keyDown(filter, { key: "Home" })).toBe(true);
    expect(filter).toHaveFocus();
    expect(fireEvent.keyDown(filter, { key: "End" })).toBe(true);
    expect(filter).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getAllByRole("option")[0]).toHaveFocus();
  });

  it("still reaches the last option with ArrowUp from the filter", async () => {
    // Kills dropping the arrow entry points along with the Home/End bindings.
    const user = userEvent.setup();
    renderExecute();
    const filter = await openPicker(user);
    await user.click(filter);
    await user.type(filter, "PRE");
    await user.keyboard("{ArrowUp}");
    expect(screen.getAllByRole("option")[1]).toHaveFocus();
  });
});
