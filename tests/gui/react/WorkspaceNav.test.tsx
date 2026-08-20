import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { WorkspaceNav } from "../../../src/rheplicant/gui/react/WorkspaceNav";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

type CompactListener = (event: { matches: boolean }) => void;

function installCompactMedia(initial: boolean) {
  let listener: CompactListener | undefined;
  const media = {
    matches: initial,
    media: "(max-width: 959px)",
    addEventListener: vi.fn((_type: string, changed: CompactListener) => {
      listener = changed;
    }),
    removeEventListener: vi.fn(),
  };
  const matchMedia = vi.fn(() => media);
  vi.stubGlobal("matchMedia", matchMedia);
  return {
    matchMedia,
    media,
    change(matches: boolean) {
      media.matches = matches;
      listener?.({ matches });
    },
  };
}

it("uses vertical arrows in the rail and horizontal arrows in compact mode", async () => {
  const onChange = vi.fn();
  const { rerender } = render(
    <WorkspaceNav active="model" orientation="vertical" onChange={onChange} />,
  );
  const model = screen.getByRole("tab", { name: "Model" });
  model.focus();
  await userEvent.keyboard("{ArrowDown}");
  expect(screen.getByRole("tab", { name: "Config" })).toHaveFocus();
  expect(onChange).toHaveBeenLastCalledWith("config");

  rerender(
    <WorkspaceNav active="config" orientation="horizontal" onChange={onChange} />,
  );
  screen.getByRole("tab", { name: "Config" }).focus();
  await userEvent.keyboard("{ArrowRight}");
  expect(screen.getByRole("tab", { name: "Execute" })).toHaveFocus();
  expect(onChange).toHaveBeenLastCalledWith("execute");
});

it("derives compact orientation while retaining one active workspace tab stop", () => {
  const compact = installCompactMedia(true);
  const { rerender } = render(
    <WorkspaceNav active="model" onChange={vi.fn()} />,
  );

  expect(compact.matchMedia).toHaveBeenCalledWith("(max-width: 959px)");
  expect(screen.getByRole("tablist")).toHaveAttribute("aria-orientation", "horizontal");
  expect(screen.getAllByRole("tab").filter((tab) => tab.tabIndex === 0))
    .toEqual([screen.getByRole("tab", { name: "Model" })]);

  rerender(<WorkspaceNav active="config" onChange={vi.fn()} />);
  expect(screen.getAllByRole("tab").filter((tab) => tab.tabIndex === 0))
    .toEqual([screen.getByRole("tab", { name: "Config" })]);

  act(() => compact.change(false));
  expect(screen.getByRole("tablist")).toHaveAttribute("aria-orientation", "vertical");
});

it("removes the exact compact media listener on unmount", () => {
  const compact = installCompactMedia(false);
  const { unmount } = render(<WorkspaceNav active="model" onChange={vi.fn()} />);
  const changed = compact.media.addEventListener.mock.calls[0]?.[1];

  unmount();

  expect(changed).toBeDefined();
  expect(compact.media.removeEventListener).toHaveBeenCalledWith("change", changed);
});
