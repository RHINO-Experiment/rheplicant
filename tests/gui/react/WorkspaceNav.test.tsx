import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { WorkspaceNav } from "../../../src/rheplicant/gui/react/WorkspaceNav";

afterEach(cleanup);

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
