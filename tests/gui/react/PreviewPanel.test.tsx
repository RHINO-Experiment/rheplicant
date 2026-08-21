import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ForwardPreviewSummary } from "../../../src/rheplicant/gui/react/PreviewPanel";
import type { PreviewProjection } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const PREVIEWS: PreviewProjection = {
  classes: [
    { preview_id: "graph", label: "Signal path", cadence: "continuous", priced: false, description: "free" },
    { preview_id: "forward", label: "Preview forward", cadence: "explicit", priced: true, description: "priced" },
  ],
  axes: [{ axis: "freq", first: [60, 68, 76], last: [68, 76, 84], count: 4, spacing: 8, unit: "MHz", precision_ratio: null, precision_ok: null }],
  shapes: [{ symbol: "n_freq", value: 4 }],
  forward_cost: { label: "estimated 24.13 ms / 14.25 MB for 4 channels", estimated_milliseconds: 24.125, estimated_peak_megabytes: 14.25, n_freq: 4, nside: 64, lmax: 191, optimizations: [] },
  declared_run_kinds: ["forward", "compare"],
};

describe("forward preview summary", () => {
  it("renders free continuous previews and cost without mounting job controls", () => {
    // Kills a regression that retains submit controls in the pure preview summary instead of ExecutionActions.
    render(<ForwardPreviewSummary previews={PREVIEWS} />);
    const continuous = screen.getByRole("region", { name: "Continuous axis and shape previews" });
    expect(continuous).toHaveTextContent("60, 68, 76");
    expect(continuous).toHaveTextContent("count 4");
    expect(continuous).toHaveTextContent("n_freq = 4");
    expect(screen.getByText("estimated 24.13 ms / 14.25 MB for 4 channels")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Result summary" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Completed audit bundles" }))
      .not.toBeInTheDocument();
  });
});
