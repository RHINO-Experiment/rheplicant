import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PreviewPanel } from "../../../src/rheplicant/gui/react/PreviewPanel";
import type {
  JobProjection,
  PreviewProjection,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const PREVIEWS: PreviewProjection = {
  classes: [
    { preview_id: "graph", label: "Signal path", cadence: "continuous", priced: false, description: "free" },
    { preview_id: "axes_shapes", label: "Axes and shapes", cadence: "continuous", priced: false, description: "free" },
    { preview_id: "validate", label: "Validate", cadence: "explicit", priced: true, description: "priced" },
    { preview_id: "forward", label: "Preview forward", cadence: "explicit", priced: true, description: "priced" },
  ],
  axes: [
    {
      axis: "freq",
      first: [60, 68, 76],
      last: [68, 76, 84],
      count: 4,
      spacing: 8,
      unit: "MHz",
      precision_ratio: null,
      precision_ok: null,
    },
    {
      axis: "time",
      first: [0, 2, 4],
      last: [10, 12, 14],
      count: 8,
      spacing: 2,
      unit: "s",
      precision_ratio: 2e15,
      precision_ok: true,
    },
  ],
  shapes: [
    { symbol: "n_time", value: 8 },
    { symbol: "n_freq", value: 4 },
  ],
  forward_cost: {
    label: "estimated 24.13 ms / 14.25 MB for 4 channels",
    estimated_milliseconds: 24.125,
    estimated_peak_megabytes: 14.25,
    n_freq: 4,
    nside: 64,
    lmax: 191,
    optimizations: [],
  },
  declared_run_kinds: ["forward", "compare"],
};

const JOBS: JobProjection[] = [
  {
    job_id: "job-a",
    session_id: "session-1",
    kind: "validate",
    revision: 0,
    yaml_digest: "abc",
    status: "succeeded",
    result: { findings: [] },
    message: null,
    stale: true,
  },
];

const RESULTS: JobProjection[] = [
  {
    ...JOBS[0],
    job_id: "job-validation",
    stale: false,
    result: {
      findings: [
        { check: "C13", severity: "warn", where: "model.adc", message: "near rail", layer: "base" },
      ],
      layers: 1,
    },
  },
  {
    ...JOBS[0],
    job_id: "job-forward",
    kind: "preview_forward",
    stale: false,
    result: {
      waterfall: { shape: [2, 2], dtype: "float32", minimum: 1, maximum: 4, mean: 2.5, values: [[1, 2], [3, 4]] },
      taps: { raw: { shape: [2, 2], dtype: "float32", minimum: 1, maximum: 4, mean: 2.5 } },
      saturated_fraction: 0.25,
      uniform_sky_mean: { drift: 200 },
    },
  },
];

describe("the four preview classes", () => {
  it("renders free strips and shape symbols without an action", () => {
    render(
      <PreviewPanel
        previews={PREVIEWS}
        jobs={[]}
        disabled={false}
        blocked={false}
        onSubmit={vi.fn()}
      />,
    );

    const continuous = screen.getByRole("region", { name: "Continuous axis and shape previews" });
    expect(continuous).toHaveTextContent("freq");
    expect(continuous).toHaveTextContent("60, 68, 76");
    expect(continuous).toHaveTextContent("count 4");
    expect(continuous).toHaveTextContent("n_time = 8");
    expect(continuous).toHaveTextContent("precision safe");
  });

  it("labels priced actions, returns explicit job kinds, and exposes no fitting preview", () => {
    const submit = vi.fn();
    render(
      <PreviewPanel
        previews={PREVIEWS}
        jobs={[]}
        disabled={false}
        blocked={false}
        onSubmit={submit}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    fireEvent.click(screen.getByRole("button", { name: /Preview forward.*estimated 24\.13 ms/ }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(submit.mock.calls.map(([kind]) => kind)).toEqual([
      "validate",
      "preview_forward",
      "run",
      "compare",
    ]);
    expect(screen.getByRole("button", { name: "Benchmark" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /fit preview/i })).not.toBeInTheDocument();
  });

  it("blocks execution on draft/refusal state and marks old priced results stale", () => {
    render(
      <PreviewPanel
        previews={PREVIEWS}
        jobs={JOBS}
        disabled={true}
        blocked={true}
        onSubmit={vi.fn()}
      />,
    );

    for (const name of ["Validate", /Preview forward/, "Run", "Compare", "Benchmark"]) {
      expect(screen.getByRole("button", { name })).toBeDisabled();
    }
    const jobs = screen.getByRole("region", { name: "Explicit jobs" });
    expect(jobs).toHaveTextContent("job-a");
    expect(jobs).toHaveTextContent("succeeded");
    expect(jobs).toHaveTextContent("stale");
  });

  it("renders the priced validation ledger and bounded forward result", () => {
    render(
      <PreviewPanel
        previews={PREVIEWS}
        jobs={RESULTS}
        disabled={false}
        blocked={false}
        onSubmit={vi.fn()}
      />,
    );

    const jobs = screen.getByRole("region", { name: "Explicit jobs" });
    expect(jobs).toHaveTextContent("C13 · warn · model.adc · base · near rail");
    expect(jobs).toHaveTextContent("waterfall 2 × 2 · float32 · range 1…4 · mean 2.5");
    expect(jobs).toHaveTextContent("saturation 25.00%");
    expect(jobs).toHaveTextContent("raw: 2 × 2 float32");
    expect(jobs).toHaveTextContent("drift: 200");
    expect(screen.getByRole("table", { name: "Predicted waterfall" })).toHaveTextContent("4");
  });
});
