import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import type {
  EditorSession,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const YAML = "model:\n  gain:\n    type: GainOperator\n    gain: 1.0\n";
const EDITED = YAML.replace("1.0", "1.25");
const SVG = '<svg><g data-node-id="gain" role="button"><text>gain</text></g></svg>';
const FORMS = { sections: [], missing_required: [] };

function state(overrides: Partial<EditorSession> = {}): EditorSession {
  return {
    session_id: "session-1",
    revision: 0,
    dirty: false,
    validation_stale: true,
    can_undo: false,
    can_redo: false,
    document: { yaml_text: YAML, svg: SVG, nodes: [], walk_order: [], forms: FORMS },
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
  const transport: SessionTransport = { replaceYaml, undo, redo, load, save };
  return { transport, replaceYaml, undo, redo, load, save };
}

describe("durable React editor session", () => {
  it("applies YAML with the visible revision and projects dirty/stale state", async () => {
    const { transport, replaceYaml } = candidate();
    render(<SessionEditor initial={state()} transport={transport} />);

    expect(screen.getByText("Saved")).toBeInTheDocument();
    expect(screen.getByText("Validation stale")).toBeInTheDocument();
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
      document: { yaml_text: EDITED, svg: SVG, nodes: [], walk_order: [], forms: FORMS },
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
      new Error("Editor command expected revision 0, but the current revision is 1."),
    );
    render(<SessionEditor initial={state()} transport={transport} />);
    const mirror = screen.getByRole("textbox", { name: "YAML source of truth" });
    fireEvent.change(mirror, { target: { value: EDITED } });
    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    expect(await screen.findByRole("status")).toHaveTextContent("current revision is 1");
    expect(mirror).toHaveValue(EDITED);
    expect(screen.getByText("Revision 0")).toBeInTheDocument();
  });
});
