import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OutputWorkflow } from "../../../src/rheplicant/gui/react/OutputWorkflow";
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
    expect(screen.getByRole("alert", { name: "Output target state" }))
      .toHaveTextContent("target exists and clobber is false");
    expect(screen.getByText("/data/night.results")).toBeInTheDocument();
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
