import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { YamlDrawer } from "../../../src/rheplicant/gui/react/YamlDrawer";
import {
  canStartDraft,
  canUpdateDraft,
  clearDraft,
  draftBlocksMutation,
  type DraftEnvelope,
} from "../../../src/rheplicant/gui/react/drafts";

afterEach(cleanup);

describe("authoritative YAML drawer", () => {
  function FocusHarness() {
    const [open, setOpen] = useState(false);
    return <>
      <button type="button" onClick={() => setOpen(true)}>First YAML opener</button>
      <button type="button" onClick={() => setOpen(true)}>Actual YAML opener</button>
      {open && (
        <YamlDrawer
          acceptedYaml="model: {}\n"
          revision={4}
          draft={{ kind: "yaml", baseRevision: 4, text: "model: {gain: 2}\n" }}
          diagnostic={null}
          onChange={vi.fn()}
          onApply={vi.fn()}
          onDiscard={vi.fn()}
          onClose={() => setOpen(false)}
        />
      )}
    </>;
  }

  it("allows only one raw draft and blocks accepted mutation", () => {
    const yaml: DraftEnvelope = { kind: "yaml", baseRevision: 4, text: "model: [" };
    expect(draftBlocksMutation(yaml)).toBe(true);
    expect(canStartDraft(yaml, "field")).toBe(false);
    expect(clearDraft()).toEqual({ kind: "none" });
    expect(canStartDraft(yaml, "graph")).toBe(false);
    expect(canUpdateDraft(yaml, { kind: "yaml", baseRevision: 4, text: "next" })).toBe(true);
    expect(canUpdateDraft(yaml, { kind: "yaml", baseRevision: 5, text: "next" })).toBe(false);
    expect(canUpdateDraft(yaml, { kind: "graph", baseRevision: 4, path: "gain", rawValue: "{}" })).toBe(false);
    const graph = { kind: "graph" as const, baseRevision: 4, path: "base:gain:stages", rawValue: "[]" };
    expect(canUpdateDraft(graph, { ...graph, rawValue: "[{}]" })).toBe(true);
    expect(canUpdateDraft(graph, { ...graph, path: "base:gain:snapshot" })).toBe(false);
    expect(canUpdateDraft(graph, { ...graph, baseRevision: 5 })).toBe(false);
    expect(canUpdateDraft(graph, { kind: "field", baseRevision: 4, path: "base:gain:stages", rawValue: "x" })).toBe(false);
  });

  it("shows a single inline parse diagnostic while retaining the raw YAML", () => {
    render(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={4}
        draft={{ kind: "yaml", baseRevision: 4, text: "model: [" }}
        diagnostic="expected the node content"
        onChange={vi.fn()}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole("textbox", { name: "YAML source of truth" }))
      .toHaveValue("model: [");
    expect(screen.getByRole("alert", { name: "YAML parse diagnostic" }))
      .toHaveTextContent("expected the node content");
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });

  it("keeps close and discard available for a dirty draft", () => {
    const onClose = vi.fn();
    const onDiscard = vi.fn();
    render(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={4}
        draft={{ kind: "yaml", baseRevision: 4, text: "model: [" }}
        diagnostic={null}
        onChange={vi.fn()}
        onApply={vi.fn()}
        onDiscard={onDiscard}
        onClose={onClose}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Close YAML drawer" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    expect(onClose).toHaveBeenCalledOnce();
    expect(onDiscard).toHaveBeenCalledOnce();
  });

  it("keeps Close available but disables Discard while Apply is in flight", () => {
    render(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={4}
        draft={{ kind: "yaml", baseRevision: 4, text: "model: [" }}
        diagnostic={null}
        busy
        onChange={vi.fn()}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Close YAML drawer" })).toBeEnabled();
  });

  it("moves initial focus inside and wraps YAML focus in both directions", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);

    await user.click(screen.getByRole("button", { name: "Actual YAML opener" }));
    const close = screen.getByRole("button", { name: "Close YAML drawer" });
    const last = screen.getByRole("button", { name: "Discard draft" });
    await waitFor(() => expect(close).toHaveFocus());

    await user.tab({ shift: true });
    expect(last).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();
  });

  it("closes on Escape and restores focus to the actual YAML opener", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);
    const opener = screen.getByRole("button", { name: "Actual YAML opener" });

    await user.click(opener);
    await user.click(screen.getByRole("textbox", { name: "YAML source of truth" }));
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog", { name: "YAML drawer" })).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("restores focus to the actual YAML opener after explicit Close", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);
    const first = screen.getByRole("button", { name: "First YAML opener" });
    const opener = screen.getByRole("button", { name: "Actual YAML opener" });

    await user.click(opener);
    await user.click(screen.getByRole("button", { name: "Close YAML drawer" }));

    await waitFor(() => expect(opener).toHaveFocus());
    expect(first).not.toHaveFocus();
  });
});
