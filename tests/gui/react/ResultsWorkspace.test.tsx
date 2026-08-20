import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useResultsWorkspace } from "../../../src/rheplicant/gui/react/ResultsWorkspace";
import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import type {
  EditorSession,
  GraphDiagram,
  JobPollProjection,
  JobProjection,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const FULL_JOB_ID = "abc123456789";

function job(overrides: Partial<JobProjection> = {}): JobProjection {
  return {
    job_id: FULL_JOB_ID,
    session_id: "s",
    kind: "run",
    revision: 8,
    yaml_digest: "digest-8",
    status: "succeeded",
    result: {},
    message: null,
    stale: false,
    ...overrides,
  };
}

function renderResults(
  jobs: JobProjection[],
  onSubmit = vi.fn(),
  options: { revision?: number; yamlDigest?: string; disabledReason?: string | null } = {},
) {
  function Harness() {
    const surface = useResultsWorkspace({
      jobs,
      revision: options.revision ?? 8,
      yamlDigest: options.yamlDigest ?? "digest-8",
      disabledReason: options.disabledReason ?? null,
      onSubmit,
    });
    return <>{surface.main}{surface.inspector}</>;
  }
  return { ...render(<Harness />), onSubmit };
}

async function selectJob(visibleId = "abc123…") {
  await userEvent.click(screen.getByRole("button", { name: `View job ${visibleId}` }));
}

describe("results workspace", () => {
  it("separates active, current, and stale evidence without losing the source revision", () => {
    renderResults([
      job({ job_id: "active123456", status: "running" }),
      job({ job_id: "current123456" }),
      job({ job_id: "stale123456", revision: 7, status: "refused", stale: true }),
    ]);

    expect(within(screen.getByRole("group", { name: "Active jobs" })).getByText("Running"))
      .toBeVisible();
    expect(within(screen.getByRole("group", { name: "Current history" })).getByText("Succeeded"))
      .toBeVisible();
    expect(within(screen.getByRole("group", { name: "Stale history" })).getByText("From revision 7"))
      .toBeVisible();
  });

  it("assigns every row to exactly one group, including stale active work", () => {
    renderResults([
      job({ job_id: "actcur-current", status: "queued" }),
      job({ job_id: "actsta-stale", status: "running", stale: true, revision: 7 }),
      job({ job_id: "current-only" }),
      job({ job_id: "stale-only", stale: true, revision: 7 }),
    ]);

    const ids = (name: string) => within(screen.getByRole("group", { name }))
      .queryAllByRole("button", { name: /^View job/ })
      .map((button) => button.getAttribute("aria-label"));
    expect(ids("Active jobs")).toEqual(["View job actcur…", "View job actsta…"]);
    expect(ids("Current history")).toEqual(["View job curren…"]);
    expect(ids("Stale history")).toEqual(["View job stale-…"]);
  });

  it("shortens the visual job id and copies the full id", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText");
    renderResults([job()]);

    expect(screen.getByText("abc123…")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy full job id" }));

    expect(writeText).toHaveBeenCalledWith(FULL_JOB_ID);
  });

  it("binds selection, copy, and audit href to the chosen row in multi-row history", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText");
    renderResults([
      job({
        job_id: "first-job-111",
        session_id: "first-session",
        result: { output: { audit_files: ["first.json"] } },
      }),
      job({
        job_id: "second-job-222",
        session_id: "second-session",
        result: { output: { audit_files: ["second.json"] } },
      }),
    ]);
    expect(screen.getByText("Select a job")).toBeVisible();

    const second = screen.getByRole("button", { name: "View job second…" });
    await user.click(second);
    const summary = screen.getByRole("region", { name: "Result summary" });
    expect(summary).toHaveTextContent("second-job-222");
    expect(summary).not.toHaveTextContent("first-job-111");
    const secondRow = second.closest("li");
    expect(secondRow).not.toBeNull();
    await user.click(within(secondRow as HTMLLIElement).getByRole("button", {
      name: "Copy full job id",
    }));
    expect(writeText).toHaveBeenCalledWith("second-job-222");
    expect(screen.getByRole("link", { name: "second.json" })).toHaveAttribute(
      "href",
      "/api/sessions/second-session/jobs/second-job-222/artifacts/second.json",
    );
    expect(screen.queryByRole("link", { name: "first.json" })).not.toBeInTheDocument();
  });

  it("reports a bounded clipboard failure instead of leaking a rejected promise", async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValueOnce(
      new Error("operating-system clipboard details"),
    );
    renderResults([job()]);

    await user.click(screen.getByRole("button", { name: "Copy full job id" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Could not copy the job id");
    expect(screen.getByRole("status")).not.toHaveTextContent("operating-system clipboard details");
  });

  it("reports the same bounded failure when the clipboard API is unavailable", async () => {
    const descriptor = Object.getOwnPropertyDescriptor(navigator, "clipboard");
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
    try {
      renderResults([job()]);
      fireEvent.click(screen.getByRole("button", { name: "Copy full job id" }));
      expect(await screen.findByRole("status"))
        .toHaveTextContent("Could not copy the job id");
    } finally {
      if (descriptor === undefined) delete (navigator as { clipboard?: Clipboard }).clipboard;
      else Object.defineProperty(navigator, "clipboard", descriptor);
    }
  });

  it.each([
    [job({ status: "refused", message: "document refused" }), "Refused"],
    [job({ status: "error", message: "RuntimeError: boom" }), "Internal error"],
    [job({
      status: "refused",
      result: { output: {
        state: "blocked_unsafe",
        state_message: "Output ancestry or access protections could not be proved safely.",
        target_path: "/data/unsafe-result",
      } },
    }), "Unsafe target"],
    [job({
      kind: "validate",
      result: {
        findings: [{
          check: "time_precision",
          severity: "warn",
          where: "observation.time",
          message: "precision warning",
          layer: "base",
        }],
      },
    }), "Warning"],
  ])("renders terminal evidence with the distinct %s label", async (row, label) => {
    renderResults([row]);
    await selectJob();

    expect(within(screen.getByRole("region", { name: "Result summary" })).getByText(label))
      .toBeVisible();
  });

  it("renders only closed result fields instead of dumping arbitrary payloads", async () => {
    renderResults([job({
      kind: "preview_forward",
      result: {
        waterfall: {
          shape: [8, 4],
          dtype: "float64",
          statistic: "value",
          minimum: -1,
          maximum: 2,
          mean: 0.5,
          private_payload: "do not render",
        },
        secret: "do not render",
      },
    })]);
    await selectJob();

    const summary = screen.getByRole("region", { name: "Result summary" });
    expect(summary).toHaveTextContent("8 × 4");
    expect(summary).toHaveTextContent("float64");
    expect(summary).toHaveTextContent("minimum -1");
    expect(summary).not.toHaveTextContent("private_payload");
    expect(summary).not.toHaveTextContent("do not render");
  });

  it("restores every bounded Forward result field with a capped waterfall table", async () => {
    renderResults([job({
      kind: "preview_forward",
      result: {
        waterfall: {
          shape: [2, 2],
          dtype: "float64",
          minimum: 1,
          maximum: 4,
          mean: 2.5,
          values: [[1, 2], [3, 4]],
        },
        taps: {
          raw: {
            shape: [2],
            dtype: "complex128",
            statistic: "magnitude",
            minimum: 5,
            maximum: 13,
            mean: 9,
          },
        },
        saturated_fraction: 0.125,
        uniform_sky_mean: { diffuse: 201.5 },
      },
    })]);
    await selectJob();

    expect(screen.getByRole("table", { name: "Predicted waterfall" }))
      .toHaveTextContent("1234");
    expect(screen.getByRole("list", { name: "Forward taps" }))
      .toHaveTextContent("raw: 2 complex128");
    expect(screen.getByText("saturation 12.50%")).toBeVisible();
    expect(screen.getByRole("list", { name: "Uniform-sky probes" }))
      .toHaveTextContent("diffuse: 201.5");
  });

  it.each([
    ["shape", { shape: [1, 2, 3, 4, 5], dtype: "float64" }],
    ["outer values", { shape: [65, 1], values: Array.from({ length: 65 }, () => [1]) }],
    ["inner values", { shape: [1, 65], values: [Array.from({ length: 65 }, () => 1)] }],
  ])("does not render an oversized waterfall %s", async (kind, waterfall) => {
    renderResults([job({ kind: "preview_forward", result: { waterfall } })]);
    await selectJob();

    if (kind === "shape") {
      expect(screen.queryByText("Shape 1 × 2 × 3 × 4 × 5")).not.toBeInTheDocument();
    } else {
      expect(screen.queryByRole("table", { name: "Predicted waterfall" }))
        .not.toBeInTheDocument();
    }
  });

  it("caps tap and uniform-sky collections at exactly 64 entries", async () => {
    const taps = Object.fromEntries(Array.from({ length: 65 }, (_, index) => [
      `tap-${index}`,
      { shape: [1], dtype: "float64" },
    ]));
    const uniformSkyMean = Object.fromEntries(Array.from({ length: 65 }, (_, index) => [
      `probe-${index}`,
      index,
    ]));
    renderResults([job({
      kind: "preview_forward",
      result: {
        waterfall: { shape: [1], dtype: "float64" },
        taps,
        uniform_sky_mean: uniformSkyMean,
      },
    })]);
    await selectJob();

    expect(within(screen.getByRole("list", { name: "Forward taps" })).getAllByRole("listitem"))
      .toHaveLength(64);
    expect(within(screen.getByRole("list", { name: "Uniform-sky probes" }))
      .getAllByRole("listitem")).toHaveLength(64);
    expect(screen.queryByText(/tap-64:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/probe-64:/)).not.toBeInTheDocument();
  });

  it("renders allowed output state and target without disclosing identity or unknown values", async () => {
    renderResults([job({
      result: { output: {
        state: "blocked_unsafe",
        state_message: "Unsafe target is blocked.",
        target_path: "/allowed/target",
        marker_id: "secret-marker-value",
        target_device: 314159,
        target_inode: 271828,
        internal_note: "secret-internal-value",
      } },
    })]);
    await selectJob();

    const inspector = screen.getByRole("complementary", { name: "Context inspector" });
    expect(inspector).toHaveTextContent("blocked_unsafe");
    expect(inspector).toHaveTextContent("/allowed/target");
    for (const secret of [
      "secret-marker-value",
      "314159",
      "271828",
      "secret-internal-value",
      "marker_id",
      "target_device",
      "target_inode",
      "internal_note",
    ]) expect(inspector).not.toHaveTextContent(secret);
  });

  it("opens an identity-checked audit route in a separate tab", async () => {
    renderResults([job({
      job_id: "j",
      result: {
        output: {
          target_path: "/data/result",
          marker_id: "marker",
          target_device: 12,
          target_inode: 34,
          audit_files: ["config.resolved.yaml"],
        },
      },
    })]);
    await selectJob("j");

    const link = screen.getByRole("link", { name: "config.resolved.yaml" });
    expect(link).toHaveAttribute(
      "href",
      "/api/sessions/s/jobs/j/artifacts/config.resolved.yaml",
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });

  it("submits a stale kind as a fresh request without forwarding historical identity", async () => {
    const onSubmit = vi.fn();
    renderResults([job({ kind: "compare", revision: 3, stale: true })], onSubmit);
    await selectJob();
    await userEvent.click(screen.getByRole("button", { name: "Re-run Compare" }));

    expect(onSubmit).toHaveBeenCalledWith("compare");
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("disables Re-run with an adjacent reason for parent blocking and exact active duplicates", async () => {
    const stale = job({ kind: "compare", revision: 3, yaml_digest: "digest-3", stale: true });
    const duplicate = job({
      job_id: "duplicate-current",
      kind: "compare",
      revision: 8,
      yaml_digest: "digest-8",
      status: "queued",
    });
    renderResults([stale], vi.fn(), {
      disabledReason: "Unsaved YAML draft",
    });
    await selectJob();
    expect(screen.getByRole("button", { name: "Re-run Compare" }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
    expect(screen.getByRole("button", { name: "Re-run Compare" })).toBeDisabled();

    cleanup();
    renderResults([stale, duplicate]);
    await selectJob();
    expect(screen.getByRole("button", { name: "Re-run Compare" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-run Compare" }))
      .toHaveAccessibleDescription("An identical Compare job is already active for revision 8.");
  });

  it.each([
    ["different kind", { kind: "run" as const }],
    ["different revision", { revision: 7 }],
    ["different digest", { yaml_digest: "digest-other" }],
    ["terminal status", { status: "succeeded" as const }],
  ])("does not over-block Re-run for an active-looking %s job", async (_label, difference) => {
    const stale = job({ kind: "compare", revision: 3, yaml_digest: "digest-3", stale: true });
    const other = job({
      job_id: "other-current",
      kind: "compare",
      revision: 8,
      yaml_digest: "digest-8",
      status: "queued",
      ...difference,
    });
    renderResults([stale, other]);
    await selectJob();
    expect(screen.getByRole("button", { name: "Re-run Compare" })).toBeEnabled();
  });
});

const DIAGRAM: GraphDiagram = {
  name: "base",
  svg: "<svg></svg>",
  nodes: [],
  walk_order: [],
  counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
  changed_nodes: [],
};

function session(jobs: JobProjection[], revision = 8): EditorSession {
  return {
    session_id: "session-1",
    revision,
    yaml_digest: `digest-${revision}`,
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    jobs,
    outputs: {
      requested_yaml: "outputs: {}\n",
      resolved_yaml: "outputs: {}\n",
      resolution_note: "No preset changes.",
      target_path: "/data/results",
      state: "ready_new",
      state_message: "Ready.",
      clobber: false,
      declared_runs: ["forward"],
      products: [],
      report: {
        enabled: false,
        rows: [],
        columns: [],
        reference: null,
        relative: [],
        formats: [],
        expected_paths: [],
      },
      audit_paths: [],
    },
    document: {
      yaml_text: "schema_version: 1\n",
      svg: "<svg></svg>",
      nodes: [],
      walk_order: [],
      forms: { sections: [], missing_required: [] },
      previews: {
        classes: [],
        axes: [],
        shapes: [],
        forward_cost: {
          label: "Cost unavailable",
          estimated_milliseconds: null,
          estimated_peak_megabytes: null,
          n_freq: null,
          nside: null,
          lmax: null,
          optimizations: [],
        },
        declared_run_kinds: [],
      },
      validation: {
        findings: [],
        section_badges: [],
        selected_presets: [],
        preset_changes: [],
        run_blocked: false,
      },
      base_diagram: DIAGRAM,
      backend_diagram: { ...DIAGRAM, name: "backend" },
      variant_diagrams: [],
    },
  };
}

function transport(initial: EditorSession) {
  const pendingRefresh = vi.fn(
    (_sessionId: string, _signal: AbortSignal) => new Promise<JobPollProjection>(() => undefined),
  );
  const unchanged = vi.fn(async () => initial);
  const submitJob = vi.fn(async () => initial);
  const api: SessionTransport = {
    refresh: unchanged,
    refreshJobs: pendingRefresh,
    replaceYaml: unchanged,
    setField: unchanged,
    undo: unchanged,
    redo: unchanged,
    load: unchanged,
    save: unchanged,
    editNode: unchanged,
    moveNodeInstance: unchanged,
    composeNode: unchanged,
    placeNode: unchanged,
    setSnapshotBefore: unchanged,
    setOutputProduct: unchanged,
    setOutputReport: unchanged,
    submitJob,
  };
  return { api, submitJob };
}

describe("results session integration", () => {
  it("keeps hook-owned selection across workspace changes while unmounting inactive controls", async () => {
    const current = session([job({ job_id: "selected123456", session_id: "session-1" })]);
    render(<SessionEditor initial={current} transport={transport(current).api} />);

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job select…" }));
    expect(screen.getByRole("region", { name: "Result summary" }))
      .toHaveTextContent("selected123456");

    fireEvent.click(screen.getByRole("tab", { name: "Model" }));
    expect(screen.queryByRole("group", { name: "Current history" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Result summary" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    expect(screen.getByRole("region", { name: "Result summary" }))
      .toHaveTextContent("selected123456");
  });

  it("renders terminal polling transitions from display jobs rather than the accepted-session snapshot", async () => {
    const running = job({
      job_id: "polled123456",
      session_id: "session-1",
      status: "running",
    });
    const initial = session([running]);
    let resolveJobs: ((projection: JobPollProjection) => void) | undefined;
    const candidate = transport(initial);
    candidate.api.refreshJobs = vi.fn(() => new Promise<JobPollProjection>((resolve) => {
      resolveJobs = resolve;
    }));
    render(<SessionEditor initial={initial} transport={candidate.api} />);

    resolveJobs?.({
      session_id: "session-1",
      revision: 8,
      yaml_digest: "digest-8",
      jobs: [{ ...running, status: "succeeded", result: { exit_code: 0 } }],
    });
    await waitFor(() => expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("succeeded"));

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    expect(within(screen.getByRole("group", { name: "Current history" })).getByText("Succeeded"))
      .toBeVisible();
    expect(screen.queryByRole("group", { name: "Active jobs" }))
      .not.toHaveTextContent("polled123456");
  });

  it("re-runs the historical kind through confirmation and the current revision path", async () => {
    const historical = job({
      job_id: "historical123",
      session_id: "session-1",
      kind: "compare",
      revision: 3,
      yaml_digest: "digest-3",
      stale: true,
    });
    const current = session([historical], 8);
    const candidate = transport(current);
    render(<SessionEditor initial={current} transport={candidate.api} />);

    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job histor…" }));
    fireEvent.click(screen.getByRole("button", { name: "Re-run Compare" }));
    expect(screen.getByRole("dialog", { name: "Trusted execution" }))
      .toHaveTextContent("Requested action: Compare");
    fireEvent.click(screen.getByRole("button", { name: "I understand, continue" }));

    await waitFor(() => expect(candidate.submitJob)
      .toHaveBeenCalledWith("session-1", "compare", 8));
    expect(candidate.submitJob).toHaveBeenCalledTimes(1);
  });

  it("projects validation and draft blocks into the Results Re-run reason", async () => {
    const historical = job({
      job_id: "blocked-history",
      session_id: "session-1",
      kind: "compare",
      revision: 3,
      yaml_digest: "digest-3",
      stale: true,
    });
    const validationBlocked = session([historical]);
    validationBlocked.document.validation = {
      ...validationBlocked.document.validation,
      run_blocked: true,
    };
    const first = render(
      <SessionEditor initial={validationBlocked} transport={transport(validationBlocked).api} />,
    );
    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job blocke…" }));
    expect(screen.getByRole("button", { name: "Re-run Compare" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-run Compare" }))
      .toHaveAccessibleDescription("Run is blocked until validation is repaired.");
    first.unmount();

    const draftSession = session([historical]);
    render(<SessionEditor initial={draftSession} transport={transport(draftSession).api} />);
    fireEvent.click(screen.getByRole("button", { name: "YAML" }));
    fireEvent.change(screen.getByRole("textbox", { name: "YAML source of truth" }), {
      target: { value: "schema_version: 1\n# exact draft\n" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job blocke…" }));
    expect(screen.getByRole("button", { name: "Re-run Compare" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-run Compare" }))
      .toHaveAccessibleDescription("Unsaved YAML draft");
  });

  it("projects parent busy state into the Results Re-run reason", async () => {
    const historical = job({
      job_id: "busy-history-1",
      session_id: "session-1",
      revision: 3,
      yaml_digest: "digest-3",
      stale: true,
    });
    const current = session([historical]);
    let resolveSave: ((next: EditorSession) => void) | undefined;
    const candidate = transport(current);
    candidate.api.save = vi.fn(() => new Promise<EditorSession>((resolve) => {
      resolveSave = resolve;
    }));
    render(
      <SessionEditor initial={current} transport={candidate.api} saveFile={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job busy-h…" }));
    fireEvent.click(screen.getByRole("button", { name: "Save YAML" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Re-run Run" }))
      .toBeDisabled());
    expect(screen.getByRole("button", { name: "Re-run Run" }))
      .toHaveAccessibleDescription("Another action is running");
    resolveSave?.(current);
    await waitFor(() => expect(screen.getByRole("button", { name: "Re-run Run" }))
      .toBeEnabled());
  });

  it("keeps the parent exact-active guard when a disabled Results handler is bypassed", () => {
    const historical = job({
      job_id: "history-guard",
      session_id: "session-1",
      kind: "compare",
      revision: 3,
      yaml_digest: "digest-3",
      stale: true,
    });
    const duplicate = job({
      job_id: "active-guard",
      session_id: "session-1",
      kind: "compare",
      revision: 8,
      yaml_digest: "digest-8",
      status: "running",
    });
    const current = session([historical, duplicate]);
    const candidate = transport(current);
    render(<SessionEditor initial={current} transport={candidate.api} />);
    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job histor…" }));
    const rerun = screen.getByRole("button", { name: "Re-run Compare" });
    expect(rerun).toBeDisabled();

    // Capture a connected opener, then call the React handler directly to model a stale
    // child-side guard. The parent runner must remain the final duplicate boundary.
    fireEvent.click(screen.getAllByRole("button", { name: "Copy full job id" })[0]);
    const reactPropsKey = Object.keys(rerun).find((key) => key.startsWith("__reactProps$"));
    expect(reactPropsKey).toBeDefined();
    const reactProps = (rerun as unknown as Record<string, { onClick(): void }>)[reactPropsKey!];
    act(() => reactProps.onClick());
    expect(screen.queryByRole("dialog", { name: "Trusted execution" })).not.toBeInTheDocument();
    expect(candidate.submitJob).not.toHaveBeenCalled();
  });

  it("rechecks exact active duplicates after confirmation opens and before transport", async () => {
    const historical = job({
      job_id: "history-race",
      session_id: "session-1",
      kind: "compare",
      revision: 3,
      yaml_digest: "digest-3",
      stale: true,
    });
    const current = session([historical]);
    let resolveJobs: ((projection: JobPollProjection) => void) | undefined;
    const candidate = transport(current);
    candidate.api.refreshJobs = vi.fn(() => new Promise<JobPollProjection>((resolve) => {
      resolveJobs = resolve;
    }));
    render(<SessionEditor initial={current} transport={candidate.api} />);
    fireEvent.click(screen.getByRole("tab", { name: "Results" }));
    fireEvent.click(screen.getByRole("button", { name: "View job histor…" }));
    fireEvent.click(screen.getByRole("button", { name: "Re-run Compare" }));
    expect(screen.getByRole("dialog", { name: "Trusted execution" })).toBeVisible();

    resolveJobs?.({
      session_id: "session-1",
      revision: 8,
      yaml_digest: "digest-8",
      jobs: [historical, job({
        job_id: "active-race",
        session_id: "session-1",
        kind: "compare",
        revision: 8,
        yaml_digest: "digest-8",
        status: "running",
      })],
    });
    await waitFor(() => expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("active-race"));
    fireEvent.click(screen.getByRole("button", { name: "I understand, continue" }));

    expect(candidate.submitJob).not.toHaveBeenCalled();
  });
});
