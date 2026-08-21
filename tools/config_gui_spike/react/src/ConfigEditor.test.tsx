import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ConfigEditor } from "./ConfigEditor";
import type { EditorSnapshot, EditorTransport } from "./types";

afterEach(cleanup);

const YAML = "model:\n  gain:\n    type: GainOperator\n    gain: 1.0\n";
const SVG = `<svg viewBox="0 0 100 100">
  <g data-node-id="astro_sum" data-node-kind="junction" aria-disabled="true"><circle /></g>
  <g data-node-id="bandpass" data-node-kind="transform" role="button" tabindex="0" aria-label="Edit bandpass"><rect /><text>Band pass display</text></g>
  <g data-node-id="gain" data-node-kind="transform" role="button" tabindex="0" aria-label="Edit gain"><rect /><text>gain</text></g>
</svg>`;

const INITIAL: EditorSnapshot = {
  yaml_text: YAML,
  svg: SVG,
  walk_order: ["astro_sum", "bandpass", "gain"],
  nodes: [
    { node_id: "astro_sum", label: "astro sum", kind: "junction", description: "sum", editable: false, reserved: false, many: false, lit: true },
    { node_id: "bandpass", label: "bandpass", kind: "transform", description: "band", editable: true, reserved: false, many: false, lit: false },
    { node_id: "gain", label: "gain", kind: "transform", description: "gain", editable: true, reserved: false, many: false, lit: true },
  ],
};

const EMPTY_GAIN: EditorSnapshot = {
  ...INITIAL,
  yaml_text: "model: {}\n",
  nodes: INITIAL.nodes.map((node) => node.node_id === "gain" ? { ...node, lit: false } : node),
};

function candidate() {
  const editNode = vi.fn(async () => ({ ...INITIAL, yaml_text: "model: {}\n" }));
  const snapshot = vi.fn(async () => INITIAL);
  const transport: EditorTransport = { editNode, snapshot };
  const view = render(<ConfigEditor initial={INITIAL} transport={transport} />);
  return { ...view, editNode, snapshot };
}

describe("React signal-path candidate", () => {
  it("delegates click and hover through stable SVG ids, including child targets", () => {
    const { container } = candidate();
    const bandpass = container.querySelector('[data-node-id="bandpass"]')!;
    fireEvent.click(bandpass.querySelector("rect")!);
    expect(screen.getByText("Selected: bandpass")).toBeInTheDocument();
    expect(container.querySelector('[data-node-id="bandpass"]')).toBe(bandpass);
    fireEvent.mouseOver(bandpass.querySelector("text")!);
    expect(screen.getByText("Hovered: bandpass")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply node edit" })).toBeDisabled();
  });

  it("supports keyboard activation and skips composition nodes in walk order", () => {
    const { container } = candidate();
    const bandpass = container.querySelector('[data-node-id="bandpass"]')!;
    fireEvent.keyDown(bandpass, { key: "Enter" });
    expect(screen.getByText("Selected: bandpass")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next node" }));
    expect(screen.getByText("Selected: gain")).toBeInTheDocument();
  });

  it("posts content edits to the Python YAML transformation boundary", async () => {
    const { editNode } = candidate();
    fireEvent.change(screen.getByLabelText("gain"), { target: { value: "1.875" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply node edit" }));
    await waitFor(() => expect(editNode).toHaveBeenCalledTimes(1));
    expect(editNode).toHaveBeenCalledWith("gain", {
      yaml_text: YAML,
      enabled: true,
      settings: { type: "GainOperator", gain: 1.875 },
    });
    expect(await screen.findByRole("status")).toHaveTextContent("YAML transformed");
  });

  it("keeps direct YAML replacement behind the same transport", async () => {
    const { snapshot } = candidate();
    snapshot.mockResolvedValueOnce(EMPTY_GAIN);
    const mirror = screen.getByRole("textbox", { name: /YAML/ });
    fireEvent.change(mirror, { target: { value: "model: {}\n" } });
    fireEvent.click(screen.getByRole("button", { name: "Load YAML mirror" }));
    await waitFor(() => expect(snapshot).toHaveBeenCalledWith("model: {}\n"));
    await waitFor(() => expect(screen.getByRole("checkbox", { name: "Node enabled" })).not.toBeChecked());
  });
});
