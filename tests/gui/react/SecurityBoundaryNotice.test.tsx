import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { SecurityBoundaryNotice } from "../../../src/rheplicant/gui/react/SecurityBoundaryNotice";

afterEach(cleanup);

describe("SecurityBoundaryNotice", () => {
  it("keeps the compact trust boundary visible and discloses every server risk", async () => {
    const user = userEvent.setup();
    render(<SecurityBoundaryNotice />);

    const summary = screen.getByText(
      "Trusted YAML: plugins, python targets, paths and jobs run as the server account.",
    );
    expect(summary).toBeVisible();

    await user.click(summary);
    expect(screen.getByText("No authentication, tenant isolation or sandbox is provided."))
      .toBeVisible();
    expect(screen.getByText(
      "Paths read and write the server filesystem; jobs may consume CPU, accelerator time and wall time.",
    )).toBeVisible();
    expect(screen.getByText("Remote binding is explicit acknowledgement, not protection."))
      .toBeVisible();
  });
});
