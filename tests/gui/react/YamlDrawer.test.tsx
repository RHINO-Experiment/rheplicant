import { useState } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

// Every id in aria-describedby resolved to an element that is actually in the document, in the
// style of reasonsFor in ReportDesigner.test.tsx. `not.toHaveAccessibleDescription()` cannot do
// this: an association to a MISSING element computes to the empty description too, so making
// aria-describedby unconditional passed it while leaving a dangling IDREF on the button.
function describedByReasons(control: HTMLElement): string[] {
  const attribute = control.getAttribute("aria-describedby");
  if (attribute === null) return [];
  return attribute.split(" ").filter(Boolean).map((id) => {
    const node = document.getElementById(id);
    expect(node, `aria-describedby names a missing element: ${id}`).not.toBeNull();
    const reason = node as HTMLElement;
    expect(reason).toBeVisible();
    return reason.textContent ?? "";
  });
}

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

  it("shows one quiet historical parse diagnostic while retaining the raw YAML", () => {
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
    const diagnostic = screen.getByText("Invalid YAML: expected the node content");
    expect(diagnostic).toBeVisible();
    expect(diagnostic.closest("[role]")).toHaveAttribute("role", "status");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
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

describe("copying a conflicted YAML draft", () => {
  // Leading and trailing whitespace is load-bearing: any trim, normalisation or re-serialisation
  // on the way to the clipboard changes this value and is caught below.
  const RAW_DRAFT = "\n  model: {gain: 2}\n\n";
  const CONFLICT = "Editor command expected revision 4, but the current revision is 5.";

  function stubClipboard(value: unknown) {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value });
  }

  afterEach(() => {
    Reflect.deleteProperty(navigator, "clipboard");
  });

  function conflicted() {
    return render(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={5}
        draft={{ kind: "yaml", baseRevision: 4, text: RAW_DRAFT }}
        diagnostic={null}
        conflict={CONFLICT}
        onChange={vi.fn()}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );
  }

  async function clickCopy() {
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy draft" }));
    });
    // Let the event loop turn so Node would have reported any unhandled rejection by now.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  function politeStatus(label: string) {
    const visibleLabel = screen.getByText(label);
    expect(visibleLabel).toBeVisible();
    const status = visibleLabel.closest("[role]");
    expect(status).toHaveAttribute("role", "status");
    expect(status).toHaveAttribute("aria-live", "polite");
    return status as HTMLElement;
  }

  it("hands the clipboard the exact raw draft and reports success politely", async () => {
    const writeText = vi.fn(async () => undefined);
    stubClipboard({ writeText });
    conflicted();
    expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(RAW_DRAFT);

    await clickCopy();

    expect(writeText).toHaveBeenCalledOnce();
    expect(writeText).toHaveBeenCalledWith(RAW_DRAFT);
    // Kills pinning every outcome to one tone: a success and a refusal must not look alike.
    expect(politeStatus("Copied the raw draft to the clipboard.")).toHaveClass("status-success");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy draft" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh accepted YAML" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Discard draft" })).toBeEnabled();
  });

  it("describes the Copy draft button with the outcome it announces", async () => {
    // Kills a copy status carrying an id no aria-describedby references, announced as nobody's description.
    const writeText = vi.fn(async () => undefined);
    stubClipboard({ writeText });
    conflicted();
    expect(describedByReasons(screen.getByRole("button", { name: "Copy draft" }))).toEqual([]);

    await clickCopy();

    expect(screen.getByRole("button", { name: "Copy draft" }))
      .toHaveAccessibleDescription(/Copied the raw draft to the clipboard\./);
    const reasons = describedByReasons(screen.getByRole("button", { name: "Copy draft" }));
    expect(reasons).toHaveLength(1);
    expect(reasons[0]).toContain("Copied the raw draft to the clipboard.");
  });

  it("reports a rejected write politely, boundedly, and without an unhandled rejection", async () => {
    const rejections: unknown[] = [];
    const capture = (reason: unknown) => { rejections.push(reason); };
    process.on("unhandledRejection", capture);
    try {
      const detail = `clipboard denied: ${"x".repeat(4000)}`;
      const writeText = vi.fn(async () => { throw new Error(detail); });
      stubClipboard({ writeText });
      conflicted();

      await clickCopy();

      // Asserted first, so a handler that stops catching is convicted here and not merely by the
      // status it also fails to render.
      expect(rejections).toEqual([]);
      expect(writeText).toHaveBeenCalledOnce();
      const status = politeStatus("Copy failed: the clipboard refused the draft.");
      expect(status).toHaveClass("status-warning");
      // Bounded: the rejection's own text never reaches the surface.
      expect(status.textContent ?? "").not.toContain("xxxxxxxxxx");
      expect((status.textContent ?? "").length).toBeLessThan(120);
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("textbox", { name: "YAML source of truth" })).toHaveValue(RAW_DRAFT);
      expect(rejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", capture);
    }
  });

  it("reports a missing Clipboard API instead of throwing at it", async () => {
    const rejections: unknown[] = [];
    const capture = (reason: unknown) => { rejections.push(reason); };
    process.on("unhandledRejection", capture);
    try {
      stubClipboard(undefined);
      conflicted();

      await clickCopy();

      expect(politeStatus("Copy unavailable: this browser exposes no clipboard."))
        .toHaveClass("status-warning");
      expect(screen.queryByText("Copy failed: the clipboard refused the draft."))
        .not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(rejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", capture);
    }
  });

  // Mirrors SessionEditor: editing the draft clears the revision conflict (updateYamlDraft calls
  // setYamlConflict(null)) while the drawer itself stays mounted, and a fresh Apply the server
  // rejects with 409 brings the same conflict back.
  function ConflictFlow() {
    const [text, setText] = useState(RAW_DRAFT);
    const [conflict, setConflict] = useState<string | null>(CONFLICT);
    return (
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={5}
        draft={{ kind: "yaml", baseRevision: 4, text }}
        diagnostic={null}
        conflict={conflict}
        onChange={(next) => { setText(next); setConflict(null); }}
        onApply={() => setConflict(CONFLICT)}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />
    );
  }

  it("never re-announces a copy of a draft the user has since edited", async () => {
    // Kills a copy status that outlives the draft it describes. The clipboard holds the pre-edit
    // text, and §9 offers Copy beside Discard as conflict recovery, so a user who trusts the
    // re-announced claim and discards loses the edit the clipboard never received.
    const writeText = vi.fn(async () => undefined);
    stubClipboard({ writeText });
    render(<ConflictFlow />);

    await clickCopy();
    politeStatus("Copied the raw draft to the clipboard.");

    fireEvent.change(screen.getByRole("textbox", { name: "YAML source of truth" }), {
      target: { value: `${RAW_DRAFT}  extra: 1\n` },
    });
    expect(screen.queryByRole("region", { name: "YAML revision conflict" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Apply YAML edit" }));

    const conflict = screen.getByRole("region", { name: "YAML revision conflict" });
    expect(conflict).toBeInTheDocument();
    expect(conflict).not.toHaveTextContent("Copied the raw draft to the clipboard.");
    expect(screen.queryByText("Copied the raw draft to the clipboard.")).not.toBeInTheDocument();
    expect(describedByReasons(screen.getByRole("button", { name: "Copy draft" }))).toEqual([]);
    // Nothing re-copied anything: the clipboard still holds only the pre-edit draft.
    expect(writeText).toHaveBeenCalledOnce();
    expect(writeText).toHaveBeenCalledWith(RAW_DRAFT);
  });

  it("re-announces nothing after a copy the draft has moved past, even without a remount", async () => {
    // The same invalidation seen inside one open conflict section: the status names one exact
    // draft, so a keystroke retires it rather than leaving it to describe the new text.
    const writeText = vi.fn(async () => undefined);
    stubClipboard({ writeText });
    const onChange = vi.fn();
    const view = render(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={5}
        draft={{ kind: "yaml", baseRevision: 4, text: RAW_DRAFT }}
        diagnostic={null}
        conflict={CONFLICT}
        onChange={onChange}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    await clickCopy();
    politeStatus("Copied the raw draft to the clipboard.");

    view.rerender(
      <YamlDrawer
        acceptedYaml="model: {}\n"
        revision={5}
        draft={{ kind: "yaml", baseRevision: 4, text: `${RAW_DRAFT}  extra: 1\n` }}
        diagnostic={null}
        conflict={CONFLICT}
        onChange={onChange}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        onClose={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.queryByText("Copied the raw draft to the clipboard.")).not.toBeInTheDocument();
    expect(describedByReasons(screen.getByRole("button", { name: "Copy draft" }))).toEqual([]);
  });

  it("keeps the conflict section quiet until a copy is actually attempted", () => {
    stubClipboard({ writeText: vi.fn(async () => undefined) });
    conflicted();

    const conflict = screen.getByRole("region", { name: "YAML revision conflict" });
    expect(conflict).not.toHaveTextContent("Copied the raw draft to the clipboard.");
    expect(conflict).not.toHaveTextContent("Copy failed");
    expect(conflict).not.toHaveTextContent("Copy unavailable");
    expect(screen.getByText("Draft base revision 4; accepted revision 5.")).toBeVisible();
  });
});
