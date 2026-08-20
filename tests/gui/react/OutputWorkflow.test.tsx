import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OutputWorkflow } from "../../../src/rheplicant/gui/react/OutputWorkflow";
import { OutputTargetCard } from "../../../src/rheplicant/gui/react/OutputTargetCard";
import type {
  EditorSession,
  OutputProjection,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const OUTPUTS: OutputProjection = {
  requested_yaml: "outputs:\n  clobber: false\n",
  resolved_yaml: "runtime:\n  jax_enable_x64: true\noutputs:\n  clobber: false\n",
  resolution_note: "Preset-merged preview; the final resolved audit file appears after Run.",
  target_path: "/data/night.results",
  state: "blocked_existing",
  state_message: "The target exists and clobber is false.",
  clobber: false,
  declared_runs: ["fit", "compare"],
  products: [
    {
      name: "arrays",
      enabled: true,
      format: "npz",
      formats: ["npz"],
      runs: ["fit"],
      keys: [],
      themes: [],
      expected_paths: ["runs/n-666974/arrays.npz"],
    },
    {
      name: "chains",
      enabled: false,
      format: "npz",
      formats: ["npz", "netcdf"],
      runs: [],
      keys: [],
      themes: [],
      expected_paths: [],
    },
  ],
  report: {
    enabled: true,
    rows: ["fit"],
    columns: ["seconds"],
    reference: null,
    relative: [],
    formats: ["json"],
    expected_paths: ["report.json"],
  },
  audit_paths: [
    "config.input.yaml",
    "config.resolved.yaml",
    "provenance.json",
    "diagnostics.json",
    "products.json",
  ],
};

function state(): EditorSession {
  return {
    session_id: "session-1",
    revision: 3,
    yaml_digest: "abc",
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    outputs: OUTPUTS,
    jobs: [
      {
        job_id: "job-1",
        session_id: "session-1",
        kind: "run",
        revision: 3,
        yaml_digest: "abc",
        status: "succeeded",
        result: {
          output: {
            target_path: "/data/night.results",
            marker_id: "marker",
            audit_files: ["config.resolved.yaml", "provenance.json"],
          },
        },
        message: null,
        stale: false,
      },
    ],
    document: {} as EditorSession["document"],
  };
}

function transport(): SessionTransport {
  const unchanged = async () => state();
  return {
    refresh: vi.fn(unchanged),
    refreshJobs: vi.fn(async () => ({
      session_id: "session-1",
      revision: 3,
      yaml_digest: "abc",
      jobs: state().jobs,
    })),
    replaceYaml: vi.fn(unchanged),
    setField: vi.fn(unchanged),
    undo: vi.fn(unchanged),
    redo: vi.fn(unchanged),
    load: vi.fn(unchanged),
    save: vi.fn(unchanged),
    editNode: vi.fn(unchanged),
    moveNodeInstance: vi.fn(unchanged),
    composeNode: vi.fn(unchanged),
    placeNode: vi.fn(unchanged),
    setSnapshotBefore: vi.fn(unchanged),
    setOutputProduct: vi.fn(unchanged),
    setOutputReport: vi.fn(unchanged),
    submitJob: vi.fn(unchanged),
  };
}

describe("output request editor", () => {
  it("keeps terminal audit detail and requested/resolved tabs out of the execute request", () => {
    // Kills a regression that leaks legacy terminal detail or YAML artefact tabs back into Task 2 Execute.
    render(
      <OutputWorkflow
        session={state()}
        transport={transport()}
        onRun={vi.fn()}
      />,
    );

    expect(screen.queryByRole("tablist", { name: "Configuration artefacts" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Completed audit bundles" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "config.resolved.yaml" })).not.toBeInTheDocument();
    const existing = screen.getByText("Existing target blocked").closest("[role]");
    expect(existing).toHaveAttribute("role", "status");
    expect(existing).toHaveTextContent("Existing target blocked");
    expect(screen.getByText("The target exists and clobber is false.")).toBeVisible();
    expect(screen.getByText("/data/night.results")).toBeInTheDocument();
  });

  it.each([
    ["ready_new", "New target ready", "success"],
    ["blocked_existing", "Existing target blocked", "warning"],
    ["replace_owned", "Owned target can be replaced", "warning"],
    ["blocked_foreign", "Foreign target blocked", "danger"],
    ["ambiguous_recovery", "Recovery needs review", "danger"],
    ["blocked_unsafe", "Unsafe target blocked", "danger"],
    ["unavailable", "Target unavailable", "danger"],
  ] as const)("renders output state %s as visible non-urgent %s evidence", (
    outputState,
    label,
    tone,
  ) => {
    const session = state();
    render(
      <OutputWorkflow
        session={{
          ...session,
          outputs: {
            ...session.outputs,
            state: outputState,
            state_message: `Exact safety explanation for ${outputState}.`,
          },
        }}
        transport={transport()}
        onRun={vi.fn()}
      />,
    );

    const visibleLabel = screen.getByText(label);
    const stateEvidence = visibleLabel.closest("[role]");
    expect(visibleLabel).toBeVisible();
    expect(stateEvidence).toHaveAttribute("role", "status");
    expect(stateEvidence).toHaveClass(`status-${tone}`);
    expect(stateEvidence).toHaveTextContent(label);
    const exactMessage = screen.getByText(`Exact safety explanation for ${outputState}.`);
    expect(exactMessage).toBeVisible();
    expect(exactMessage.closest("[role]")).toBe(stateEvidence);
    expect(screen.getByText("/data/night.results").closest("[role]")).toBe(stateEvidence);
  });

  it("alerts only a newly reached danger state and keeps history quiet across rerender and remount", async () => {
    const ready = { ...OUTPUTS, state: "ready_new" as const };
    const danger = { ...OUTPUTS, state: "blocked_foreign" as const };
    const view = render(<OutputTargetCard output={ready} />);
    expect(screen.getByText("New target ready").closest("[role]"))
      .toHaveAttribute("role", "status");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    view.rerender(<OutputTargetCard output={danger} />);
    const foreignLabel = await screen.findByText("Foreign target blocked");
    await waitFor(() => expect(foreignLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    const alert = foreignLabel.closest("[role]");
    view.rerender(<OutputTargetCard output={danger} />);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByText("Foreign target blocked").closest("[role]")).toBe(alert);

    view.unmount();
    render(<OutputTargetCard output={danger} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("Foreign target blocked").closest("[role]"))
      .toHaveAttribute("role", "status");
  });

  it.each([
    ["blocked_foreign", "Foreign target blocked"],
    ["ambiguous_recovery", "Recovery needs review"],
    ["blocked_unsafe", "Unsafe target blocked"],
    ["unavailable", "Target unavailable"],
  ] as const)("alerts a mounted non-danger transition to %s", async (outputState, label) => {
    const view = render(<OutputTargetCard output={{ ...OUTPUTS, state: "ready_new" }} />);
    view.rerender(<OutputTargetCard output={{ ...OUTPUTS, state: outputState }} />);
    const visibleLabel = screen.getByText(label);
    await waitFor(() => expect(visibleLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    expect(visibleLabel).toBeVisible();
  });

  it("treats a new danger state, message, or path as distinct urgent evidence", async () => {
    const first = {
      ...OUTPUTS,
      state: "blocked_foreign" as const,
      state_message: "First foreign ownership refusal.",
    };
    const view = render(<OutputTargetCard output={{ ...OUTPUTS, state: "ready_new" }} />);
    view.rerender(<OutputTargetCard output={first} />);
    const firstLabel = screen.getByText("Foreign target blocked");
    await waitFor(() => expect(firstLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    const firstAlert = firstLabel.closest("[role]");

    const nextState = {
      ...first,
      state: "ambiguous_recovery" as const,
    };
    view.rerender(<OutputTargetCard output={nextState} />);
    const nextStateLabel = screen.getByText("Recovery needs review");
    await waitFor(() => expect(nextStateLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    const nextStateAlert = nextStateLabel.closest("[role]");
    expect(nextStateAlert).not.toBe(firstAlert);

    const nextMessage = {
      ...nextState,
      state_message: "A newly observed recovery owner is still ambiguous.",
    };
    view.rerender(<OutputTargetCard output={nextMessage} />);
    const nextMessageLabel = screen.getByText("Recovery needs review");
    await waitFor(() => expect(nextMessageLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    const nextMessageAlert = nextMessageLabel.closest("[role]");
    expect(nextMessageAlert).not.toBe(nextStateAlert);
    expect(nextMessageAlert).toHaveTextContent(nextMessage.state_message);
    expect(nextMessageAlert).toHaveTextContent(nextMessage.target_path);

    view.rerender(<OutputTargetCard output={{ ...nextMessage, target_path: "/new/target.results" }} />);
    const nextPathLabel = screen.getByText("Recovery needs review");
    await waitFor(() => expect(nextPathLabel.closest("[role]")).toHaveAttribute("role", "alert"));
    const nextPathAlert = nextPathLabel.closest("[role]");
    expect(nextPathAlert).not.toBe(nextMessageAlert);
    expect(nextPathAlert).toHaveTextContent(nextMessage.state_message);
    expect(nextPathAlert).toHaveTextContent("/new/target.results");
  });

  it("re-alerts a historical danger identity after an A-B-A cycle", async () => {
    const dangerA = {
      ...OUTPUTS,
      state: "blocked_foreign" as const,
      state_message: "Danger A.",
    };
    const dangerB = {
      ...dangerA,
      state: "ambiguous_recovery" as const,
      state_message: "Danger B.",
    };
    const view = render(<OutputTargetCard output={dangerA} />);
    expect(screen.getByText("Foreign target blocked").closest("[role]"))
      .toHaveAttribute("role", "status");

    view.rerender(<OutputTargetCard output={dangerB} />);
    const alertB = screen.getByText("Recovery needs review").closest("[role]");
    await waitFor(() => expect(alertB).toHaveAttribute("role", "alert"));
    view.rerender(<OutputTargetCard output={dangerA} />);
    await waitFor(() => expect(screen.getByText("Foreign target blocked").closest("[role]"))
      .toHaveAttribute("role", "alert"));
  });

  it("hands output mutations to the required parent runner", () => {
    // Kills a regression that restores a local OutputWorkflow transport fallback and bypasses SessionEditor's busy gate.
    const api = transport();
    const onRun = vi.fn();
    render(
      <OutputWorkflow session={state()} transport={api} onRun={onRun} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Add product" }));
    fireEvent.click(screen.getByRole("option", { name: "chains" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Write chains" }));
    expect(api.setOutputProduct).not.toHaveBeenCalled();
    expect(onRun).toHaveBeenCalledOnce();
    const [action, message] = onRun.mock.calls[0] as [() => Promise<EditorSession>, string];
    expect(message).toBe("Updated chains output");
    void action();
    expect(api.setOutputProduct).toHaveBeenCalledWith(
      "session-1", "chains", true, "npz", [], [], [], 3,
    );
    fireEvent.click(screen.getByRole("checkbox", { name: "Report row compare" }));
    expect(onRun).toHaveBeenCalledTimes(2);
  });

  it("shows a product's predictable paths only when its request is expanded", () => {
    // Kills a regression that restores eager settings or terminal-audit rendering to the request editor.
    render(
      <OutputWorkflow
        session={state()}
        transport={transport()}
        onRun={vi.fn()}
      />,
    );

    expect(screen.queryByText("runs/n-666974/arrays.npz")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add product" }));
    fireEvent.click(screen.getByRole("option", { name: "arrays" }));
    expect(screen.getByText("runs/n-666974/arrays.npz")).toBeInTheDocument();
    expect(screen.queryByText("job-1")).not.toBeInTheDocument();
  });
});
