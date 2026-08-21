import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DiagnosticsDrawer } from "../../../src/rheplicant/gui/react/DiagnosticsDrawer";
import { useExecuteWorkspace, type ExecuteWorkspaceProps } from "../../../src/rheplicant/gui/react/ExecuteWorkspace";
import {
  FULL_VALIDATION_TONE,
  deriveFullValidation,
  type FullValidationState,
} from "../../../src/rheplicant/gui/react/fullValidation";
import type { EditorSession, JobProjection, SessionTransport } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

function session(overrides: Partial<EditorSession> = {}): EditorSession {
  return {
    session_id: "session", revision: 4, yaml_digest: "current", dirty: false,
    validation_stale: false, can_undo: false, can_redo: false, jobs: [],
    outputs: {
      requested_yaml: "outputs: {}", resolved_yaml: "outputs: {}", resolution_note: "resolved",
      target_path: "/tmp/results", state: "ready_new", state_message: "ready", clobber: false,
      declared_runs: ["run"], products: [],
      report: { enabled: false, rows: [], columns: ["mean"], reference: null, relative: [], formats: ["json"], expected_paths: [] },
      audit_paths: [],
    },
    document: {
      yaml_text: "", svg: "", nodes: [], walk_order: [], forms: { sections: [], missing_required: [] },
      previews: {
        classes: [], axes: [], shapes: [],
        forward_cost: { label: "cost", estimated_milliseconds: 1, estimated_peak_megabytes: 1, n_freq: 1, nside: 1, lmax: 1, optimizations: [] },
        declared_run_kinds: ["run"],
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
  const unchanged = async () => session();
  return {
    refresh: vi.fn(unchanged),
    refreshJobs: vi.fn(async () => ({ session_id: "session", revision: 4, yaml_digest: "current", jobs: [] })),
    replaceYaml: vi.fn(unchanged), setField: vi.fn(unchanged), undo: vi.fn(unchanged), redo: vi.fn(unchanged),
    load: vi.fn(unchanged), save: vi.fn(unchanged), editNode: vi.fn(unchanged), moveNodeInstance: vi.fn(unchanged),
    composeNode: vi.fn(unchanged), placeNode: vi.fn(unchanged), setSnapshotBefore: vi.fn(unchanged),
    setOutputProduct: vi.fn(unchanged), setOutputReport: vi.fn(unchanged), submitJob: vi.fn(unchanged),
  };
}

const validateJob = (overrides: Partial<JobProjection> = {}): JobProjection => ({
  job_id: "validate-job", session_id: "session", kind: "validate", revision: 4, yaml_digest: "current",
  status: "succeeded", result: null, message: null, stale: false, ...overrides,
});

function renderExecute(jobs: JobProjection[]) {
  const props: ExecuteWorkspaceProps = {
    session: session(), jobs, transport: transport(),
    disabledReason: null, onSubmit: vi.fn(), onRun: vi.fn(),
  };
  function Harness() {
    const surface = useExecuteWorkspace(props);
    return <>{surface.main}</>;
  }
  return render(<Harness />);
}

function renderDrawer(jobs: JobProjection[]) {
  return render(
    <DiagnosticsDrawer
      session={session({ jobs })}
      onOpenConfigPath={vi.fn()}
      onOpenYamlPath={vi.fn()}
      onClose={vi.fn()}
    />,
  );
}

/** Each surface owns its own words; only the state behind them is shared. */
const EXECUTE_LABEL: Record<FullValidationState, string> = {
  "not-run": "Full validation not run",
  queued: "Full validation queued for revision 3",
  running: "Full validation running for revision 3",
  current: "Full validation current for revision 4",
  stale: "Full validation stale from revision 3",
  refused: "Full validation refused: repair validation",
  error: "Full validation internal error: bounded failure",
};

const DRAWER_LABEL: Record<FullValidationState, string> = {
  "not-run": "Not run for this YAML",
  queued: "Queued",
  running: "Running",
  current: "Current for revision 4",
  stale: "Stale for this YAML",
  refused: "Refused · repair validation",
  error: "Internal error · bounded failure",
};

// The last two members are per-case label overrides for the rows whose words the state alone does
// not fix: the two surfaces name an ABSENT reason differently from a reported one, and the state
// is "refused" either way.
const SHARED_CASES: Array<[FullValidationState, JobProjection[], string?, string?]> = [
  ["not-run", []],
  ["queued", [validateJob({ revision: 3, status: "queued" })]],
  ["running", [validateJob({ revision: 3, status: "running" })]],
  ["current", [validateJob({ status: "succeeded" })]],
  ["refused", [validateJob({ status: "refused", message: "repair validation" })]],
  ["error", [validateJob({ status: "error", message: "bounded failure" })]],
  ["stale", [validateJob({ revision: 3, status: "succeeded", stale: true, yaml_digest: "current" })]],
  ["stale", [validateJob({ revision: 3, status: "refused", stale: false, yaml_digest: "old", message: "repair validation" })]],
  ["stale", [validateJob({ revision: 3, status: "queued", stale: true, yaml_digest: "old" })]],
  // `?? NO_REASON` cannot see an empty message, and empty is exactly what the server sends: it
  // builds the field as `bounded_text(error)`, which is "" for an exception raised with no args,
  // and the frame validator only asks whether it is a string. Both surfaces then rendered the
  // dangling separator the fallback exists to prevent — "Full validation refused: " and
  // "Refused · ". Only `||` sees it.
  [
    "refused",
    [validateJob({ status: "refused", message: "" })],
    "Full validation refused: no reason reported",
    "Refused · no reason reported",
  ],
  [
    "error",
    [validateJob({ status: "error", message: "" })],
    "Full validation internal error: no detail reported",
    "Internal error · no detail reported",
  ],
];

describe("shared full-validation derivation", () => {
  it.each(SHARED_CASES)("derives %s once for every surface %#", (
    expected,
    jobs,
    executeOverride,
    drawerOverride,
  ) => {
    // Kills a second, divergent derivation of the same §7.5 vocabulary living inside either surface.
    const derived = deriveFullValidation(jobs, session().yaml_digest);
    expect(derived.state).toBe(expected);

    const executeLabel = executeOverride ?? EXECUTE_LABEL[expected];
    const drawerLabel = drawerOverride ?? DRAWER_LABEL[expected];

    renderExecute(jobs);
    const execute = screen.getByText(executeLabel).closest("[role]");
    expect(execute).toHaveAttribute("role", "status");
    expect(execute).toHaveClass(`status-${FULL_VALIDATION_TONE[expected]}`);
    // No dangling separator survives: the words end in a reason, never in the punctuation
    // that was supposed to introduce one.
    expect(executeLabel).not.toMatch(/[:·]\s*$/);
    cleanup();

    renderDrawer(jobs);
    const drawer = within(screen.getByRole("region", { name: "Full validation" })).getByRole("status");
    expect(drawer).toHaveTextContent(drawerLabel);
    expect(drawer).toHaveClass(`status-${FULL_VALIDATION_TONE[expected]}`);
    expect(drawer.textContent ?? "").not.toMatch(/[:·]\s*$/);
  });

  it("checks staleness before any status branch on both surfaces", () => {
    // Kills reinstating a status branch ahead of the staleness check, which reported a superseded refusal as danger.
    const jobs = [validateJob({ revision: 3, status: "refused", stale: true, yaml_digest: "old", message: "repair validation" })];
    expect(deriveFullValidation(jobs, "current").state).toBe("stale");

    renderExecute(jobs);
    expect(screen.queryByText(/Full validation refused/)).toBeNull();
    expect(screen.getByText("Full validation stale from revision 3").closest("[role]")).toHaveClass("status-stale");
    cleanup();

    renderDrawer(jobs);
    const drawer = screen.getByRole("region", { name: "Full validation" });
    expect(drawer).toHaveTextContent("Stale for this YAML");
    expect(drawer).not.toHaveTextContent("Refused");
  });

  it("treats the server flag alone as staleness even when the digest still matches", () => {
    // Kills dropping the server stale flag and trusting only the digest comparison.
    expect(deriveFullValidation([validateJob({ stale: true })], "current").state).toBe("stale");
  });

  it("treats a digest the document has moved past as staleness even when the flag is clear", () => {
    // Kills trusting only the server flag, which is how the drawer used to decide staleness alone.
    expect(deriveFullValidation([validateJob({ stale: false, yaml_digest: "old" })], "current").state).toBe("stale");
  });

  it("selects the last non-stale validate job and falls back to the last one overall", () => {
    // Kills selecting the last job in array order, which hides a bound verdict behind a superseded one.
    const bound = validateJob({ job_id: "bound", status: "succeeded" });
    const superseded = validateJob({ job_id: "superseded", status: "refused", yaml_digest: "next", stale: true, message: "repair validation" });
    expect(deriveFullValidation([bound, superseded], "current").job?.job_id).toBe("bound");
    expect(deriveFullValidation([superseded], "current").job?.job_id).toBe("superseded");
    expect(deriveFullValidation([], "current").job).toBeNull();
  });

  it("names an absent reason in the same words on both surfaces", () => {
    // Kills the divergence: Execute invented a fallback no test exercised while Diagnostics
    // rendered a bare "Refused", so one surface named the absence and the other hid it.
    const refused = [validateJob({ status: "refused", message: null })];
    const errored = [validateJob({ status: "error", message: null })];
    const drawerStatus = () => within(
      screen.getByRole("region", { name: "Full validation" }),
    ).getByRole("status");

    renderExecute(refused);
    expect(screen.getByText("Full validation refused: no reason reported")).toBeVisible();
    cleanup();
    renderDrawer(refused);
    expect(drawerStatus()).toHaveTextContent("Refused · no reason reported");
    cleanup();

    renderExecute(errored);
    expect(screen.getByText("Full validation internal error: no detail reported")).toBeVisible();
    cleanup();
    renderDrawer(errored);
    expect(drawerStatus()).toHaveTextContent("Internal error · no detail reported");
  });

  it("falls back to the newest superseded validate job when every one is stale", () => {
    // Kills falling back to the OLDEST stale validate job. Right after any edit every validate job
    // is stale, so this is the ordinary state, and the oldest one makes both surfaces name a
    // revision the user superseded several edits ago.
    const older = validateJob({ job_id: "stale-2", revision: 2, yaml_digest: "old-2", stale: true });
    const newer = validateJob({ job_id: "stale-7", revision: 7, yaml_digest: "old-7", stale: true });

    const derived = deriveFullValidation([older, newer], "current");
    expect(derived.state).toBe("stale");
    expect(derived.job?.job_id).toBe("stale-7");

    renderExecute([older, newer]);
    expect(screen.getByText("Full validation stale from revision 7")).toBeVisible();
    expect(screen.queryByText("Full validation stale from revision 2")).toBeNull();
  });

  it("ignores every job that is not a validate job", () => {
    // Kills widening the selection to preview or run jobs, which report a different question entirely.
    const other = validateJob({ job_id: "run-job", kind: "run", status: "refused", message: "repair validation" });
    expect(deriveFullValidation([other], "current").state).toBe("not-run");
  });
});
