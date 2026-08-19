import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import { RequestError } from "../../../src/rheplicant/gui/react/api";
import type {
  EditorSession,
  GraphDiagram,
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
const DIAGRAM: GraphDiagram = {
  name: "base",
  svg: SVG,
  nodes: [],
  walk_order: [],
  counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
  changed_nodes: [],
};

function documentState(yamlText = YAML) {
  return {
    yaml_text: yamlText,
    svg: SVG,
    nodes: [],
    walk_order: [],
    forms: FORMS,
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
  const unchanged = vi.fn(async () => initial);
  const transport: SessionTransport = {
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
  };
  return { transport, replaceYaml, undo, redo, load, save };
}

describe("durable React editor session", () => {
  it("applies YAML with the visible revision and projects dirty/stale state", async () => {
    const { transport, replaceYaml } = candidate();
    render(<SessionEditor initial={state()} transport={transport} />);

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

  it("loads a file only after user selection and replaces the whole projection", async () => {
    const { transport, load } = candidate();
    const readFile = vi.fn(async () => EDITED);
    render(
      <SessionEditor initial={state()} transport={transport} readFile={readFile} />,
    );

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
    expect(await screen.findByRole("status")).toHaveTextContent("download refused");
    expect(save).not.toHaveBeenCalled();
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
  });

  it("surfaces a revision conflict without overwriting the current YAML", async () => {
    const { transport, replaceYaml } = candidate();
    replaceYaml.mockRejectedValueOnce(
      new RequestError(
        409,
        "Editor command expected revision 0, but the current revision is 1.",
      ),
    );
    render(<SessionEditor initial={state()} transport={transport} />);
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    expect(await screen.findByRole("status")).toHaveTextContent("current revision is 1");
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
    expect(screen.queryByRole("alert", { name: "YAML parse diagnostic" }))
      .not.toBeInTheDocument();
  });

  it("keeps invalid YAML editable, reports the parse failure, and retains the last good projection", async () => {
    const { transport, replaceYaml } = candidate();
    replaceYaml.mockRejectedValueOnce(
      new RequestError(422, "GUI document: expected the node content"),
    );
    render(<SessionEditor initial={state()} transport={transport} />);
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });

    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
    fireEvent.change(mirror, { target: { value: "model: [" } });
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    expect(await screen.findByRole("alert", { name: "YAML parse diagnostic" }))
      .toHaveTextContent("expected the node content");
    expect(mirror).toHaveValue("model: [");
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

    const ledger = screen.getByRole("region", { name: "Pre-flight finding ledger" });
    expect(ledger).toHaveTextContent("unknown model node");
    expect(ledger).toHaveTextContent("suspicious run");
    expect(ledger).toHaveTextContent("recorded fact");
    expect(ledger).toHaveTextContent("variant:bad");
    expect(screen.getByRole("region", { name: "Diff against preset" }))
      .toHaveTextContent("runtime.jax_enable_x64");
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });
});
