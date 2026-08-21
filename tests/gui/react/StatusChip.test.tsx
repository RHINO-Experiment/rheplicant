import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { outputTone } from "../../../src/rheplicant/gui/react/OutputTargetCard";
import {
  StatusChip,
  type StatusTone,
} from "../../../src/rheplicant/gui/react/StatusChip";

afterEach(cleanup);

describe("closed workbench status semantics", () => {
  it.each([
    ["ready_new", "success"],
    ["blocked_existing", "warning"],
    ["replace_owned", "warning"],
    ["blocked_foreign", "danger"],
    ["ambiguous_recovery", "danger"],
    ["blocked_unsafe", "danger"],
    ["unavailable", "danger"],
  ] as const)("maps output state %s to %s", (state, tone) => {
    expect(outputTone(state)).toBe(tone);
  });

  it.each([
    ["neutral", "Queued validate at revision 7", "○"],
    ["success", "Current run succeeded", "✓"],
    ["warning", "Warning: target already exists", "⚠"],
    ["danger", "Refused by output safety checks", "!"],
    ["stale", "From revision 6", "↶"],
    ["disabled", "Disabled: unsaved YAML draft", "—"],
  ] as const)("renders %s with visible text, a distinct icon and polite status semantics", (
    tone,
    label,
    icon,
  ) => {
    const { container } = render(
      <StatusChip tone={tone as StatusTone} label={label} />,
    );

    const visibleLabel = screen.getByText(label);
    const chip = visibleLabel.closest("[role]");
    expect(visibleLabel).toBeVisible();
    expect(chip).toHaveAttribute("role", "status");
    expect(chip).toHaveTextContent(label);
    expect(chip).toHaveAttribute("aria-live", "polite");
    expect(chip).not.toHaveAttribute("aria-label");
    expect(chip).not.toHaveAttribute("aria-labelledby");
    expect(visibleLabel).not.toHaveAttribute("id");
    expect(chip).toHaveClass(`status-${tone}`);
    expect(container.querySelector("[aria-hidden='true']")).toHaveTextContent(icon);
  });

  it("uses different visible labels and icons for warning and stale evidence", () => {
    const { rerender, container } = render(
      <StatusChip tone="warning" label="Warning: review the target" />,
    );
    const warning = screen.getByText("Warning: review the target").closest("[role]");
    expect(warning).toHaveAttribute("role", "status");
    expect(warning).toHaveTextContent("Warning");
    expect(container.querySelector("[aria-hidden='true']")).toHaveTextContent("⚠");

    rerender(<StatusChip tone="stale" label="From revision 6" />);
    const stale = screen.getByText("From revision 6").closest("[role]");
    expect(stale).toHaveAttribute("role", "status");
    expect(stale).toHaveTextContent("From revision 6");
    expect(container.querySelector("[aria-hidden='true']")).toHaveTextContent("↶");
  });

  it("keeps one stable alert node for an unchanged refusal instead of remounting it", () => {
    const { rerender } = render(
      <StatusChip tone="danger" label="Refused: repair the accepted configuration" urgent />,
    );
    const alert = screen.getByText("Refused: repair the accepted configuration").closest("[role]");
    expect(alert).toHaveAttribute("role", "alert");
    expect(alert).toHaveTextContent("Refused");
    expect(alert).toHaveAttribute("aria-live", "assertive");

    rerender(
      <StatusChip tone="danger" label="Refused: repair the accepted configuration" urgent />,
    );
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByText("Refused: repair the accepted configuration").closest("[role]"))
      .toBe(alert);

    rerender(<StatusChip tone="neutral" label="Running run at revision 7" />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    rerender(
      <StatusChip tone="danger" label="Internal error: retry or report" urgent />,
    );
    const internalError = screen.getByText("Internal error: retry or report").closest("[role]");
    expect(internalError).toHaveAttribute("role", "alert");
    expect(internalError).toHaveTextContent("Internal error");
  });
});
