import { useState } from "react";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DiagnosticsDrawer } from "../../../src/rheplicant/gui/react/DiagnosticsDrawer";
import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import type {
  EditorSession,
  JobProjection,
  LedgerFinding,
  SessionTransport,
  ValidationProjection,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const projectedFinding: LedgerFinding = {
  check: "channel-range",
  severity: "refuse",
  where: "observation.channel_start",
  message: "Choose a non-negative channel.",
  attribution: "base",
};

const yamlFinding: LedgerFinding = {
  check: "nested-path",
  severity: "warn",
  where: "observation.channel_start.extra",
  message: "This is not an exact projected field.",
  attribution: "variant:high_gain",
};

const validation: ValidationProjection = {
  findings: [projectedFinding, yamlFinding, {
    check: "reported",
    severity: "report",
    where: "runtime.seed",
    message: "Seed is recorded.",
    attribution: "variant:high_gain",
  }],
  section_badges: [],
  selected_presets: [],
  preset_changes: [],
  run_blocked: true,
};

function validateJob(
  overrides: Partial<JobProjection> = {},
): JobProjection {
  return {
    job_id: "validate-1",
    session_id: "session-1",
    kind: "validate",
    revision: 7,
    yaml_digest: "current-digest",
    status: "succeeded",
    result: {
      findings: [{
        check: "full-refusal",
        severity: "refuse",
        where: "observation.channel_start",
        message: "Full validation refused this field.",
        layer: "base",
      }, {
        check: "full-warning",
        severity: "warn",
        where: "model.hidden.path",
        message: "Full validation warned about YAML.",
        layer: "variant:high_gain",
      }],
    },
    message: null,
    stale: false,
    ...overrides,
  };
}

function session(jobs: JobProjection[] = [], overrides: Partial<EditorSession> = {}): EditorSession {
  return {
    session_id: "session-1",
    revision: 7,
    yaml_digest: "current-digest",
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    outputs: {
      requested_yaml: "",
      resolved_yaml: "",
      resolution_note: "",
      target_path: null,
      state: "unavailable",
      state_message: "Unavailable",
      clobber: false,
      declared_runs: [],
      products: [],
      report: {
        enabled: false,
        rows: [],
        columns: [],
        reference: null,
        relative: [],
        formats: [],
        expected_paths: [],
      },
      audit_paths: [],
    },
    jobs,
    document: {
      yaml_text: "observation:\n  channel_start: 0\n",
      svg: "<svg />",
      nodes: [],
      walk_order: [],
      forms: {
        missing_required: [],
        sections: [{
          section_id: "observation",
          label: "Observation",
          disabled: false,
          reason: null,
          widgets: [{
            path: "observation.channel_start",
            path_pattern: "observation.channel_start",
            label: "channel start",
            widget: "integer",
            choices: [],
            visible: true,
            present: true,
            must_decide: false,
            value: 0,
            dimension: null,
            unit_policy: null,
            delivery: null,
            disabled: false,
            reason: null,
          }],
        }, {
          section_id: "runtime",
          label: "Runtime",
          disabled: false,
          reason: null,
          widgets: [{
            path: "runtime.seed",
            path_pattern: "runtime.seed",
            label: "seed",
            widget: "text",
            choices: [],
            visible: true,
            present: true,
            must_decide: false,
            value: "1",
            dimension: null,
            unit_policy: null,
            delivery: null,
            disabled: false,
            reason: null,
          }],
        }],
      },
      previews: {
        classes: [],
        axes: [],
        shapes: [],
        forward_cost: {
          label: "Unavailable",
          estimated_milliseconds: null,
          estimated_peak_megabytes: null,
          n_freq: null,
          nside: null,
          lmax: null,
          optimizations: [],
        },
        declared_run_kinds: [],
      },
      validation,
      base_diagram: {
        name: "base",
        svg: "<svg />",
        nodes: [],
        walk_order: [],
        counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
        changed_nodes: [],
      },
      backend_diagram: {
        name: "backend",
        svg: "<svg />",
        nodes: [],
        walk_order: [],
        counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
        changed_nodes: [],
      },
      variant_diagrams: [],
    },
    ...overrides,
  };
}

function renderDrawer(candidate = session()) {
  const onOpenConfigPath = vi.fn();
  const onOpenYamlPath = vi.fn();
  const onClose = vi.fn();
  render(
    <DiagnosticsDrawer
      session={candidate}
      onOpenConfigPath={onOpenConfigPath}
      onOpenYamlPath={onOpenYamlPath}
      onClose={onClose}
    />,
  );
  return { onOpenConfigPath, onOpenYamlPath, onClose };
}

function transportFor(candidate: EditorSession): SessionTransport {
  const unchanged = vi.fn(async () => candidate);
  return {
    refresh: unchanged,
    refreshJobs: vi.fn(async () => ({
      session_id: candidate.session_id,
      revision: candidate.revision,
      yaml_digest: candidate.yaml_digest,
      jobs: candidate.jobs,
    })),
    replaceYaml: unchanged,
    setField: unchanged,
    undo: unchanged,
    redo: unchanged,
    load: unchanged,
    save: unchanged,
    editNode: unchanged,
    moveNodeInstance: unchanged,
    composeNode: unchanged,
    placeNode: unchanged,
    setSnapshotBefore: unchanged,
    setOutputProduct: unchanged,
    setOutputReport: unchanged,
    submitJob: unchanged,
  };
}

describe("global diagnostics drawer", () => {
  function FocusHarness() {
    const [open, setOpen] = useState(false);
    return <>
      <button type="button" onClick={() => setOpen(true)}>First diagnostics opener</button>
      <button type="button" onClick={() => setOpen(true)}>Actual diagnostics opener</button>
      {open && (
        <DiagnosticsDrawer
          session={session()}
          onOpenConfigPath={vi.fn()}
          onOpenYamlPath={vi.fn()}
          onClose={() => setOpen(false)}
        />
      )}
    </>;
  }

  it("keeps accepted Quick checks distinct from the current Full validation", () => {
    renderDrawer(session([validateJob()]));

    const quick = screen.getByRole("region", { name: "Quick checks" });
    const full = screen.getByRole("region", { name: "Full validation" });
    expect(within(quick).getByText("Choose a non-negative channel.")).toBeVisible();
    expect(within(quick).queryByText("Full validation refused this field.")).not.toBeInTheDocument();
    expect(within(full).getByText("Full validation refused this field.")).toBeVisible();
    expect(within(full).queryByText("Choose a non-negative channel.")).not.toBeInTheDocument();
    expect(full).toHaveTextContent("Current for revision 7");
  });

  it.each([
    ["not run", [], "Not run for this YAML"],
    ["queued", [validateJob({ status: "queued", result: null })], "Queued"],
    ["running", [validateJob({ status: "running", result: null })], "Running"],
    ["refused", [validateJob({ status: "refused", result: null, message: "Validation refused" })], "Refused · Validation refused"],
    ["error", [validateJob({ status: "error", result: null, message: "Worker failed" })], "Internal error · Worker failed"],
    ["stale", [validateJob({ revision: 6, stale: true })], "Stale for this YAML"],
  ])("renders the %s Validate job state without inventing progress", (_label, jobs, text) => {
    renderDrawer(session(jobs as JobProjection[]));

    const full = screen.getByRole("region", { name: "Full validation" });
    expect(full).toHaveTextContent(text);
    expect(within(full).getByRole("status")).toHaveTextContent(text);
    expect(full).not.toHaveTextContent("%");
  });

  it("uses the latest non-stale Validate identity across a revision-only transition", () => {
    renderDrawer(session([
      validateJob({ job_id: "old-current", status: "queued", result: null }),
      validateJob({ job_id: "current", message: "matching result" }),
      validateJob({
        job_id: "newer-but-stale",
        revision: 8,
        stale: true,
        result: { findings: [{
          check: "stale",
          severity: "warn",
          where: "runtime.seed",
          message: "stale result must not win",
          layer: "base",
        }] },
      }),
    ], { revision: 8 }));

    const full = screen.getByRole("region", { name: "Full validation" });
    expect(full).toHaveTextContent("Current for revision 7");
    expect(full).toHaveTextContent("Full validation refused this field.");
    expect(full).not.toHaveTextContent("stale result must not win");
  });

  it("groups base and variant findings by refuse, warn, and report", () => {
    renderDrawer(session([validateJob()]));

    const quick = screen.getByRole("region", { name: "Quick checks" });
    expect(within(quick).getByRole("region", { name: "Base refuse findings" }))
      .toHaveTextContent("Choose a non-negative channel.");
    expect(within(quick).getByRole("region", { name: "Variant high_gain warn findings" }))
      .toHaveTextContent("This is not an exact projected field.");
    expect(within(quick).getByRole("region", { name: "Variant high_gain report findings" }))
      .toHaveTextContent("Seed is recorded.");

    const full = screen.getByRole("region", { name: "Full validation" });
    expect(within(full).getByRole("region", { name: "Base refuse findings" }))
      .toHaveTextContent("Full validation refused this field.");
    expect(within(full).getByRole("region", { name: "Variant high_gain warn findings" }))
      .toHaveTextContent("Full validation warned about YAML.");
  });

  it("opens Config only for an exact projected path and routes every other path to YAML context", async () => {
    const user = userEvent.setup();
    const { onOpenConfigPath, onOpenYamlPath } = renderDrawer(session([validateJob()]));

    await user.click(screen.getAllByRole("button", { name: /observation\.channel_start$/ })[0]);
    expect(onOpenConfigPath).toHaveBeenCalledWith("observation.channel_start");
    expect(onOpenYamlPath).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /observation\.channel_start\.extra$/ }));
    expect(onOpenYamlPath).toHaveBeenCalledWith("observation.channel_start.extra");
    expect(onOpenConfigPath).toHaveBeenCalledTimes(1);
  });

  it("wires diagnostics navigation through the workbench without interpreting a YAML path", async () => {
    const user = userEvent.setup();
    const candidate = session([validateJob()]);
    render(<SessionEditor initial={candidate} transport={transportFor(candidate)} />);

    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    const diagnostics = screen.getByRole("dialog", { name: "Diagnostics" });
    await user.click(within(diagnostics).getAllByRole(
      "button",
      { name: /observation\.channel_start$/ },
    )[0]);
    expect(screen.getByRole("tabpanel", { name: "Config" })).toBeVisible();
    expect(screen.getByRole("article", { name: "observation.channel_start" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    await user.click(within(screen.getByRole("dialog", { name: "Diagnostics" })).getByRole(
      "button",
      { name: /observation\.channel_start\.extra$/ },
    ));
    expect(screen.getByRole("dialog", { name: "YAML drawer" })).toBeVisible();
    expect(screen.getByText("YAML context:")).toHaveTextContent(
      "YAML context: observation.channel_start.extra",
    );
  });

  it("consumes a Config finding command so the same finding navigates again", async () => {
    const user = userEvent.setup();
    const candidate = session([validateJob()]);
    render(<SessionEditor initial={candidate} transport={transportFor(candidate)} />);

    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    await user.click(within(screen.getByRole("dialog", { name: "Diagnostics" })).getAllByRole(
      "button",
      { name: /observation\.channel_start$/ },
    )[0]);
    expect(screen.getByRole("region", { name: "Observation form" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: /Runtime/ }));
    expect(screen.getByRole("region", { name: "Runtime form" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    await user.click(within(screen.getByRole("dialog", { name: "Diagnostics" })).getAllByRole(
      "button",
      { name: /observation\.channel_start$/ },
    )[0]);

    expect(screen.getByRole("region", { name: "Observation form" })).toBeVisible();
    expect(screen.getByRole("article", { name: "observation.channel_start" })).toBeVisible();
  });

  it("does not replay a consumed finding command when accepted forms refresh", async () => {
    const user = userEvent.setup();
    const candidate = session([validateJob()], { can_undo: true });
    let accepted = candidate;
    const transport = transportFor(candidate);
    transport.refreshJobs = vi.fn(async () => ({
      session_id: accepted.session_id,
      revision: accepted.revision,
      yaml_digest: accepted.yaml_digest,
      jobs: accepted.jobs,
    }));
    const refreshJobs = vi.mocked(transport.refreshJobs);
    transport.undo = vi.fn(async () => {
      accepted = session(candidate.jobs, {
        revision: 8,
        can_undo: false,
        document: {
          ...candidate.document,
          forms: {
            ...candidate.document.forms,
            sections: [...candidate.document.forms.sections],
          },
        },
      });
      return accepted;
    });
    render(<SessionEditor initial={candidate} transport={transport} />);
    await waitFor(() => expect(refreshJobs).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Diagnostics" }));
    await user.click(within(screen.getByRole("dialog", { name: "Diagnostics" })).getAllByRole(
      "button",
      { name: /observation\.channel_start$/ },
    )[0]);
    await user.click(screen.getByRole("button", { name: /Runtime/ }));
    expect(screen.getByRole("region", { name: "Runtime form" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(await screen.findByText("Undid YAML edit")).toBeVisible();
    expect(screen.getByText("Revision 8")).toBeVisible();
    expect(screen.getByRole("region", { name: "Runtime form" })).toBeVisible();
    await waitFor(() => expect(refreshJobs).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByRole("region", { name: "Jobs" }))
      .toHaveTextContent("Job polling idle"));
    expect(screen.queryByText(
      "Job refresh identity no longer matches this accepted session; refresh the session.",
    )).not.toBeInTheDocument();
  });

  it("keeps an empty preset diff collapsed and renders only projected nonempty changes", () => {
    const { rerender } = render(
      <DiagnosticsDrawer
        session={session()}
        onOpenConfigPath={vi.fn()}
        onOpenYamlPath={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const empty = screen.getByText("Diff against preset").closest("details");
    expect(empty).not.toHaveAttribute("open");
    expect(empty).toHaveTextContent("No effective scientific differences.");

    const changed = session([], {
      document: {
        ...session().document,
        validation: {
          ...validation,
          selected_presets: ["wideband"],
          preset_changes: [{
            path: "runtime.seed",
            kind: "changed",
            preset_value: 1,
            document_value: 2,
          }],
        },
      },
    });
    rerender(
      <DiagnosticsDrawer
        session={changed}
        onOpenConfigPath={vi.fn()}
        onOpenYamlPath={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const nonempty = screen.getByText("Diff against preset").closest("details");
    expect(nonempty).not.toHaveAttribute("open");
    expect(nonempty).toHaveTextContent("runtime.seed");
    expect(nonempty).toHaveTextContent("1 → 2");
  });

  it("moves initial focus inside and wraps diagnostics focus in both directions", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);

    await user.click(screen.getByRole("button", { name: "Actual diagnostics opener" }));
    const dialog = screen.getByRole("dialog", { name: "Diagnostics" });
    const close = within(dialog).getByRole("button", { name: "Close diagnostics" });
    const last = within(dialog).getByText("Diff against preset", { selector: "summary" });
    await waitFor(() => expect(close).toHaveFocus());

    await user.tab({ shift: true });
    expect(last).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();
  });

  it("restores the stable Diagnostics opener after YAML finding transitions", async () => {
    const user = userEvent.setup();
    const candidate = session([validateJob()]);
    render(<SessionEditor initial={candidate} transport={transportFor(candidate)} />);
    const opener = screen.getByRole("button", { name: "Diagnostics" });

    for (const closeWith of ["button", "escape"] as const) {
      await user.click(opener);
      await user.click(within(screen.getByRole("dialog", { name: "Diagnostics" })).getByRole(
        "button",
        { name: /observation\.channel_start\.extra$/ },
      ));
      const yaml = screen.getByRole("dialog", { name: "YAML drawer" });
      expect(yaml).toBeVisible();
      if (closeWith === "button") {
        await user.click(within(yaml).getByRole("button", { name: "Close YAML drawer" }));
      } else {
        await user.keyboard("{Escape}");
      }
      expect(screen.queryByRole("dialog", { name: "YAML drawer" })).not.toBeInTheDocument();
      await waitFor(() => expect(opener).toHaveFocus());
    }
  });

  it("closes on Escape and restores focus to the actual diagnostics opener", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);
    const opener = screen.getByRole("button", { name: "Actual diagnostics opener" });

    await user.click(opener);
    await user.click(screen.getByRole("heading", { name: "Diagnostics" }));
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog", { name: "Diagnostics" })).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("restores focus to the actual diagnostics opener after explicit Close", async () => {
    const user = userEvent.setup();
    render(<FocusHarness />);
    const first = screen.getByRole("button", { name: "First diagnostics opener" });
    const opener = screen.getByRole("button", { name: "Actual diagnostics opener" });

    await user.click(opener);
    await user.click(screen.getByRole("button", { name: "Close diagnostics" }));

    await waitFor(() => expect(opener).toHaveFocus());
    expect(first).not.toHaveFocus();
  });
});
