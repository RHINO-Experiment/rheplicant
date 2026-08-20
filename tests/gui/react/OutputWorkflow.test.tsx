import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
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

describe("output and recovery workflow", () => {
  it("renders keyboard-accessible requested/resolved tabs and the exact target state", () => {
    render(
      <OutputWorkflow
        session={state()}
        transport={transport()}
        onAccept={vi.fn()}
      />,
    );

    const tabs = screen.getByRole("tablist", { name: "Configuration artefacts" });
    const asked = within(tabs).getByRole("tab", { name: "What I asked for" });
    const resolved = within(tabs).getByRole("tab", { name: "What will run" });
    expect(asked).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("clobber: false");
    fireEvent.keyDown(asked, { key: "ArrowRight" });
    expect(resolved).toHaveAttribute("aria-selected", "true");
    expect(resolved).toHaveFocus();
    expect(screen.getByRole("tabpanel")).toHaveTextContent("jax_enable_x64");
    expect(screen.getByRole("alert", { name: "Output target state" }))
      .toHaveTextContent("target exists and clobber is false");
    expect(screen.getByText("/data/night.results")).toBeInTheDocument();
  });

  it("edits product formats/runs and report rows through revisioned transport", () => {
    const api = transport();
    render(
      <OutputWorkflow session={state()} transport={api} onAccept={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Write chains" }));
    expect(api.setOutputProduct).toHaveBeenCalledWith(
      "session-1", "chains", true, "npz", [], [], [], 3,
    );
    fireEvent.change(screen.getByRole("combobox", { name: "chains format" }), {
      target: { value: "netcdf" },
    });
    expect(api.setOutputProduct).toHaveBeenCalledWith(
      "session-1", "chains", false, "netcdf", [], [], [], 3,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Report row compare" }));
    expect(api.setOutputReport).toHaveBeenCalledWith(
      "session-1",
      true,
      ["fit", "compare"],
      ["seconds"],
      null,
      [],
      ["json"],
      3,
    );
  });

  it("shows predictable paths and identity-bound audit bundle links", () => {
    render(
      <OutputWorkflow
        session={state()}
        transport={transport()}
        onAccept={vi.fn()}
      />,
    );

    expect(screen.getByText("runs/n-666974/arrays.npz")).toBeInTheDocument();
    const links = screen.getByRole("region", { name: "Completed audit bundles" });
    expect(within(links).getByRole("link", { name: "config.resolved.yaml" }))
      .toHaveAttribute(
        "href",
        "/api/sessions/session-1/jobs/job-1/artifacts/config.resolved.yaml",
      );
    expect(within(links).getByRole("link", { name: "config.resolved.yaml" }))
      .toHaveAttribute("target", "_blank");
    expect(within(links).getByRole("link", { name: "provenance.json" }))
      .toBeInTheDocument();
  });
});
