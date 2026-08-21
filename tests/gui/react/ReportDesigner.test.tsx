import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OutputWorkflow } from "../../../src/rheplicant/gui/react/OutputWorkflow";
import { CLOSED_PRODUCT_VIEW } from "../../../src/rheplicant/gui/react/ProductSelector";
import { ReportDesigner, describedBy } from "../../../src/rheplicant/gui/react/ReportDesigner";
import type {
  EditorSession,
  OutputProjection,
  OutputReportProjection,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const REPORT: OutputReportProjection = {
  enabled: true,
  rows: ["fit"],
  columns: ["seconds"],
  reference: null,
  relative: [],
  formats: ["json"],
  expected_paths: ["report.json"],
};

const OUTPUTS: OutputProjection = {
  requested_yaml: "outputs:\n  clobber: false\n",
  resolved_yaml: "outputs:\n  clobber: false\n",
  resolution_note: "Preset-merged preview.",
  target_path: "/data/night.results",
  state: "ready_new",
  state_message: "The target is absent and ready for a new run.",
  clobber: false,
  declared_runs: ["fit", "compare"],
  products: [],
  report: REPORT,
  audit_paths: [],
};

function state(outputs: Partial<OutputProjection> = {}): EditorSession {
  return {
    session_id: "session-1",
    revision: 3,
    yaml_digest: "abc",
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    outputs: { ...OUTPUTS, ...outputs },
    jobs: [],
    document: {} as EditorSession["document"],
  };
}

function transport(): SessionTransport {
  const unchanged = async () => state();
  return {
    refresh: vi.fn(unchanged),
    refreshJobs: vi.fn(async () => ({
      session_id: "session-1",
      revision: 3,
      yaml_digest: "abc",
      jobs: [],
    })),
    replaceYaml: vi.fn(unchanged),
    setField: vi.fn(unchanged),
    undo: vi.fn(unchanged),
    redo: vi.fn(unchanged),
    load: vi.fn(unchanged),
    save: vi.fn(unchanged),
    editNode: vi.fn(unchanged),
    moveNodeInstance: vi.fn(unchanged),
    composeNode: vi.fn(unchanged),
    placeNode: vi.fn(unchanged),
    setSnapshotBefore: vi.fn(unchanged),
    setOutputProduct: vi.fn(unchanged),
    setOutputReport: vi.fn(unchanged),
    submitJob: vi.fn(unchanged),
  };
}

// Resolves every id in aria-describedby to a rendered, visible element and returns their text.
// A dangling id fails here, so removing either half of the contract (the association or the
// visible reason) is caught rather than silently reported as "described".
function reasonsFor(control: HTMLElement): string {
  const ids = (control.getAttribute("aria-describedby") ?? "").split(" ").filter(Boolean);
  return ids.map((id) => {
    const node = document.getElementById(id);
    expect(node, `aria-describedby names a missing element: ${id}`).not.toBeNull();
    const reason = node as HTMLElement;
    expect(reason).toBeVisible();
    return reason.textContent ?? "";
  }).join(" ");
}

function designer(outputs: Partial<OutputProjection> = {}) {
  return render(
    <ReportDesigner
      session={state(outputs)}
      transport={transport()}
      onRun={vi.fn()}
      disabled={false}
    />,
  );
}

describe("describedBy names something or nothing at all", () => {
  it("returns undefined rather than an empty description", () => {
    // Kills dropping the empty guard. `named.join(" ")` over nothing is "", React renders
    // aria-describedby="", and reasonsFor cannot tell that from no attribute at all because
    // "".split(" ").filter(Boolean) is empty too — the assertion goes quietly vacuous.
    expect(describedBy()).toBeUndefined();
    expect(describedBy(undefined)).toBeUndefined();
    expect(describedBy(undefined, undefined)).toBeUndefined();
    expect(describedBy("", "")).toBeUndefined();
  });

  it("drops the empty ids and joins the rest with one space", () => {
    // Kills dropping the `id !== ""` half of the filter, which would emit a leading or doubled
    // space and, with it, an empty IDREF in the list.
    expect(describedBy("only")).toBe("only");
    expect(describedBy("first", "second")).toBe("first second");
    expect(describedBy("", "first", undefined, "second")).toBe("first second");
  });

  it("leaves an unexplained control carrying no aria-describedby attribute at all", () => {
    // The same guard seen from the DOM: absence, not an association to nobody.
    designer({ report: { ...REPORT, rows: ["fit", "compare"], columns: ["mean", "seconds"], formats: ["json", "text"], reference: "fit" } });

    for (const name of ["Report row fit", "mean", "json", "mean_sigma"]) {
      const control = screen.getByRole("checkbox", { name });
      expect(control).toBeEnabled();
      expect(control).not.toHaveAttribute("aria-describedby");
    }
  });
});

describe("heldValue answers only for a selection of exactly one", () => {
  it("holds nothing when the report has selected no rows at all", () => {
    // The lower side of the `=== 1` boundary. With no rows selected nothing is unremovable, so
    // every checkbox is free and the reason that explains an unremovable last row must be absent.
    designer({ report: { ...REPORT, rows: [] } });

    for (const name of ["Report row fit", "Report row compare"]) {
      const row = screen.getByRole("checkbox", { name });
      expect(row).toBeEnabled();
      expect(row).not.toBeChecked();
      expect(row).not.toHaveAttribute("aria-describedby");
    }
    expect(screen.queryByText("Report row disabled: the report keeps at least one row."))
      .not.toBeInTheDocument();
  });

  it("holds nothing when two of three columns are selected", () => {
    // The upper side, one step further out than the two-row case: `>= 1` and `> 0` both disable
    // the first selected value here, and `=== 1` leaves every one of them free.
    designer({ report: { ...REPORT, columns: ["mean", "std"] } });

    for (const name of ["mean", "std", "seconds"]) {
      expect(screen.getByRole("checkbox", { name })).toBeEnabled();
    }
    expect(screen.queryByText("Report column disabled: the report keeps at least one column."))
      .not.toBeInTheDocument();
  });
});

describe("disabled report controls carry an adjacent reason", () => {
  it("explains why the last remaining report row cannot be unchecked", () => {
    designer();

    const last = screen.getByRole("checkbox", { name: "Report row fit" });
    const other = screen.getByRole("checkbox", { name: "Report row compare" });
    expect(last).toBeDisabled();
    expect(other).toBeEnabled();
    expect(screen.getByText("Report row disabled: the report keeps at least one row."))
      .toBeVisible();
    expect(reasonsFor(last))
      .toContain("Report row disabled: the report keeps at least one row.");
  });

  it("drops the last-row reason once a second row makes unchecking legal", () => {
    designer({ report: { ...REPORT, rows: ["fit", "compare"] } });

    const fit = screen.getByRole("checkbox", { name: "Report row fit" });
    expect(fit).toBeEnabled();
    expect(reasonsFor(fit)).toBe("");
    expect(screen.queryByText("Report row disabled: the report keeps at least one row."))
      .not.toBeInTheDocument();
  });

  it("keeps the last-row reason off a row the document no longer declares", () => {
    // Kills dropping `candidates.includes(values[0])` from heldValue. A report holding a run that
    // declared_runs no longer names renders no checkbox for it, so the reason would stand alone:
    // a visible "disabled" explanation with no disabled control and no aria-describedby pointing
    // at it — which reasonsFor, reading from the control outward, structurally cannot detect.
    designer({ declared_runs: ["fit"], report: { ...REPORT, rows: ["ghost"] } });

    expect(screen.queryByText("Report row disabled: the report keeps at least one row."))
      .not.toBeInTheDocument();
    const fit = screen.getByRole("checkbox", { name: "Report row fit" });
    expect(fit).toBeEnabled();
    expect(fit).not.toBeChecked();
    expect(reasonsFor(fit)).toBe("");
    expect(screen.queryByRole("checkbox", { name: "Report row ghost" })).toBeNull();
  });

  it("explains why the last remaining report column cannot be unchecked", () => {
    designer();

    const columns = screen.getByRole("group", { name: "Report columns" });
    const last = screen.getByRole("checkbox", { name: "seconds" });
    expect(last).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "mean" })).toBeEnabled();
    expect(columns).toHaveTextContent("Report column disabled: the report keeps at least one column.");
    expect(reasonsFor(last))
      .toContain("Report column disabled: the report keeps at least one column.");
  });

  it("explains why the last remaining report format cannot be unchecked", () => {
    designer();

    const last = screen.getByRole("checkbox", { name: "json" });
    expect(last).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "text" })).toBeEnabled();
    expect(screen.getByText("Report format disabled: the report keeps at least one format."))
      .toBeVisible();
    expect(reasonsFor(last))
      .toContain("Report format disabled: the report keeps at least one format.");
  });

  it("explains why relative columns are disabled while no reference row is chosen", () => {
    designer();

    const meanSigma = screen.getByRole("checkbox", { name: "mean_sigma" });
    const widthRatio = screen.getByRole("checkbox", { name: "width_ratio" });
    expect(meanSigma).toBeDisabled();
    expect(widthRatio).toBeDisabled();
    expect(screen.getByText("Relative columns disabled: choose a reference row first."))
      .toBeVisible();
    // One shared reason element serves both controls rather than a duplicated id.
    expect(reasonsFor(meanSigma))
      .toContain("Relative columns disabled: choose a reference row first.");
    expect(reasonsFor(widthRatio))
      .toContain("Relative columns disabled: choose a reference row first.");
    expect(screen.getAllByText("Relative columns disabled: choose a reference row first."))
      .toHaveLength(1);
  });

  it("drops the relative reason once a reference row is chosen", () => {
    designer({ report: { ...REPORT, reference: "fit" } });

    const meanSigma = screen.getByRole("checkbox", { name: "mean_sigma" });
    expect(meanSigma).toBeEnabled();
    expect(reasonsFor(meanSigma)).toBe("");
    expect(screen.queryByText("Relative columns disabled: choose a reference row first."))
      .not.toBeInTheDocument();
  });

  it("explains why Write report is disabled when the document declares no run", () => {
    render(
      <OutputWorkflow
        session={state({ declared_runs: [], report: { ...REPORT, enabled: false } })}
        transport={transport()}
        onRun={vi.fn()}
        productView={CLOSED_PRODUCT_VIEW}
        onProductView={vi.fn()}
        comparisonOpen={false}
        onComparisonOpen={vi.fn()}
      />,
    );

    const write = screen.getByRole("button", { name: "Write report" });
    expect(write).toBeDisabled();
    expect(screen.getByText("Write report disabled: no run is declared.")).toBeVisible();
    expect(reasonsFor(write)).toContain("Write report disabled: no run is declared.");
  });

  it("leaves Write report enabled and unexplained once a run is declared", () => {
    render(
      <OutputWorkflow
        session={state({ report: { ...REPORT, enabled: false } })}
        transport={transport()}
        onRun={vi.fn()}
        productView={CLOSED_PRODUCT_VIEW}
        onProductView={vi.fn()}
        comparisonOpen={false}
        onComparisonOpen={vi.fn()}
      />,
    );

    const write = screen.getByRole("button", { name: "Write report" });
    expect(write).toBeEnabled();
    expect(reasonsFor(write)).toBe("");
    expect(screen.queryByText("Write report disabled: no run is declared."))
      .not.toBeInTheDocument();
  });

  it("keeps the outer blocked reason alongside the per-control reason", () => {
    render(<>
      <p id="outer-blocked-reason">Editing is blocked by an unsaved draft.</p>
      <ReportDesigner
        session={state()}
        transport={transport()}
        onRun={vi.fn()}
        disabled
        disabledReasonId="outer-blocked-reason"
      />
    </>);

    const last = screen.getByRole("checkbox", { name: "Report row fit" });
    const described = reasonsFor(last);
    expect(described).toContain("Editing is blocked by an unsaved draft.");
    expect(described).toContain("Report row disabled: the report keeps at least one row.");
  });
});
