import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ConfigForms } from "../../../src/rheplicant/gui/react/ConfigForms";
import { useConfigWorkspace } from "../../../src/rheplicant/gui/react/ConfigWorkspace";
import { NO_DRAFT, type DraftEnvelope } from "../../../src/rheplicant/gui/react/drafts";
import { SessionEditor } from "../../../src/rheplicant/gui/react/SessionEditor";
import type {
  EditorSession,
  FormProjection,
  SectionBadge,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const forms: FormProjection = {
  missing_required: ["resources.beams.horn.normalize"],
  sections: [
    {
      section_id: "beam",
      label: "Beam",
      disabled: false,
      reason: null,
      widgets: [
        {
          path: "resources.beams.horn.normalize",
          path_pattern: "resources.beams.*.normalize",
          label: "normalize",
          widget: "select",
          choices: ["none", "pixel_sum", "solid_angle"],
          visible: true,
          present: false,
          must_decide: true,
          value: null,
          dimension: null,
          unit_policy: null,
          delivery: null,
          disabled: false,
          reason: "Normalization decides the output unit.",
        },
        {
          path: "resources.beams.horn.phi0_deg",
          path_pattern: "resources.beams.*.phi0_deg",
          label: "phi0 deg",
          widget: "value",
          choices: [],
          visible: false,
          present: false,
          must_decide: false,
          value: null,
          dimension: "deg",
          unit_policy: "optional",
          delivery: null,
          disabled: false,
          reason: null,
        },
      ],
    },
    {
      section_id: "observation",
      label: "Observation",
      disabled: false,
      reason: null,
      widgets: [
        {
          path: "observation.optional_note",
          path_pattern: "observation.optional_note",
          label: "optional note",
          widget: "text",
          choices: [],
          visible: true,
          present: false,
          must_decide: false,
          value: null,
          dimension: null,
          unit_policy: null,
          delivery: null,
          disabled: false,
          reason: null,
        },
        {
          path: "observation.calibrated",
          path_pattern: "observation.calibrated",
          label: "calibrated",
          widget: "toggle",
          choices: [],
          visible: true,
          present: true,
          must_decide: false,
          value: false,
          dimension: null,
          unit_policy: null,
          delivery: "traced",
          disabled: false,
          reason: null,
        },
        {
          path: "observation.mode",
          path_pattern: "observation.mode",
          label: "mode",
          widget: "select",
          choices: ["drift", "track"],
          visible: true,
          present: false,
          must_decide: true,
          value: null,
          dimension: null,
          unit_policy: null,
          delivery: null,
          disabled: false,
          reason: "Choose an observing mode.",
        },
        {
          path: "observation.channel_start",
          path_pattern: "observation.channel_start",
          label: "channel start",
          widget: "integer",
          choices: [],
          visible: true,
          present: true,
          must_decide: false,
          value: 0,
          dimension: "Hz",
          unit_policy: "optional",
          delivery: null,
          disabled: false,
          reason: null,
        },
        {
          path: "observation.note",
          path_pattern: "observation.note",
          label: "observer note",
          widget: "text",
          choices: [],
          visible: true,
          present: true,
          must_decide: false,
          value: "",
          dimension: null,
          unit_policy: null,
          delivery: null,
          disabled: false,
          reason: null,
        },
        {
          path: "observation.server_path",
          path_pattern: "observation.server_path",
          label: "input data",
          widget: "file",
          choices: [],
          visible: true,
          present: true,
          must_decide: false,
          value: "/srv/data/input.h5",
          dimension: null,
          unit_policy: null,
          delivery: null,
          disabled: false,
          reason: "A path on the server running Rheplicant.",
        },
      ],
    },
    {
      section_id: "instrument",
      label: "Instrument",
      disabled: false,
      reason: null,
      widgets: [
        {
          path: "model.foregrounds.ref_freq",
          path_pattern: "model.foregrounds.ref_freq",
          label: "ref freq",
          widget: "value",
          choices: [],
          visible: true,
          present: true,
          must_decide: false,
          value: 140000000,
          dimension: "Hz",
          unit_policy: "optional",
          delivery: "static_float",
          disabled: false,
          reason: null,
        },
      ],
    },
    {
      section_id: "campaign",
      label: "Campaign",
      disabled: true,
      reason: "Reserved for capability 4 (streaming evidence).",
      widgets: [],
    },
  ],
};

const badges: SectionBadge[] = [
  {
    section_id: "beam",
    incomplete: 1,
    refuse: 2,
    warn: 1,
    report: 4,
    preset_changes: 3,
  },
];

const diagram = {
  name: "base",
  svg: "<svg />",
  nodes: [],
  walk_order: [],
  counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
  changed_nodes: [],
};

function withFieldValue(
  projection: FormProjection,
  path: string,
  value: unknown,
): FormProjection {
  return {
    ...projection,
    sections: projection.sections.map((section) => ({
      ...section,
      widgets: section.widgets.map((widget) => widget.path === path
        ? { ...widget, value, present: true }
        : widget),
    })),
  };
}

function editorSession(
  revision = 4,
  projection: FormProjection = forms,
): EditorSession {
  return {
    session_id: "session-1",
    revision,
    dirty: false,
    validation_stale: false,
    can_undo: true,
    can_redo: true,
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
    jobs: [],
    document: {
      yaml_text: "observation:\n  server_path: /srv/data/input.h5\n",
      svg: "<svg />",
      nodes: [],
      walk_order: [],
      forms: projection,
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
      validation: {
        findings: [],
        section_badges: badges,
        selected_presets: [],
        preset_changes: [],
        run_blocked: false,
      },
      base_diagram: diagram,
      backend_diagram: { ...diagram, name: "backend" },
      variant_diagrams: [],
    },
  };
}

function transportFor(initial = editorSession()) {
  const unchanged = vi.fn(async () => initial);
  const setField = vi.fn(async () => initial);
  const transport: SessionTransport = {
    refresh: unchanged,
    replaceYaml: unchanged,
    setField,
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
  return { transport, setField };
}

interface HarnessProps {
  candidate?: EditorSession;
  transport?: SessionTransport;
  initialDraft?: DraftEnvelope;
  disabled?: boolean;
  disabledReason?: string | null;
  requestedPath?: string | null;
  onEditYaml?: (path: string) => void;
}

function ConfigHarness({
  candidate = editorSession(),
  transport = transportFor(candidate).transport,
  initialDraft = NO_DRAFT,
  disabled = false,
  disabledReason = null,
  requestedPath = null,
  onEditYaml = () => undefined,
}: HarnessProps) {
  const [draft, setDraft] = useState<DraftEnvelope>(initialDraft);
  const drafts = {
    draft,
    begin(next: Exclude<DraftEnvelope, { kind: "none" }>) {
      if (draft.kind !== "none") return false;
      setDraft(next);
      return true;
    },
    update(next: Exclude<DraftEnvelope, { kind: "none" }>) { setDraft(next); },
    clear() { setDraft(NO_DRAFT); },
  };
  const onAccept = vi.fn();
  const surface = useConfigWorkspace({
    session: candidate,
    transport,
    drafts,
    disabled,
    disabledReason,
    requestedPath,
    onAccept,
    onEditYaml,
    onRun(action, message) {
      void action().then((next) => {
        onAccept(next, message);
        drafts.clear();
      });
    },
  });
  return <>{disabledReason && <p id={disabledReason}>Editing is blocked</p>}{surface.main}{surface.inspector}</>;
}

describe("schema-projected config forms", () => {
  it("mounts one selected section and preserves false, zero, and empty-string values", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness />);

    const observation = screen.getByRole("button", { name: /Observation/ });
    await user.click(observation);

    expect(screen.getAllByRole("region", { name: /form$/ })).toHaveLength(1);
    expect(screen.getByRole("region", { name: "Observation form" })).toBeVisible();
    expect(observation).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: /Beam/ })).not.toHaveAttribute("aria-current");
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();

    const fields = screen.getAllByRole("article");
    expect(fields.map((field) => field.getAttribute("aria-label"))).toEqual([
      "observation.mode",
      "observation.calibrated",
      "observation.channel_start",
      "observation.note",
      "observation.server_path",
      "observation.optional_note",
    ]);
    const required = screen.getByRole("region", { name: "Missing required fields" });
    const present = screen.getByRole("region", { name: "Present values" });
    const optional = screen.getByRole("region", { name: "Optional fields not set" });
    expect(within(required).getByRole("heading", { name: "Missing required fields" }))
      .toBeVisible();
    expect(within(present).getByRole("heading", { name: "Present values" })).toBeVisible();
    expect(within(optional).getByRole("heading", { name: "Optional fields not set" }))
      .toBeVisible();
    expect(within(required).getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.mode"]);
    expect(within(present).getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual([
        "observation.calibrated",
        "observation.channel_start",
        "observation.note",
        "observation.server_path",
      ]);
    expect(within(optional).getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.optional_note"]);
    expect(within(screen.getByRole("article", { name: "observation.calibrated" }))
      .getByText("false")).toBeVisible();
    expect(within(screen.getByRole("article", { name: "observation.channel_start" }))
      .getByText("0")).toBeVisible();
    expect(within(screen.getByRole("article", { name: "observation.note" }))
      .getByText('\"\"')).toBeVisible();
    expect(within(screen.getByRole("article", { name: "observation.optional_note" }))
      .getByText("Not set")).toBeVisible();
    expect(forms.sections.find((section) => section.section_id === "observation")
      ?.widgets.map((widget) => widget.path)).toEqual([
      "observation.optional_note",
      "observation.calibrated",
      "observation.mode",
      "observation.channel_start",
      "observation.note",
      "observation.server_path",
    ]);
  });

  it("filters the active section by case-insensitive label or exact projected path only", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const filter = screen.getByRole("searchbox", { name: "Filter configuration fields" });

    await user.type(filter, "CALIBRATED");
    expect(screen.getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.calibrated"]);

    await user.clear(filter);
    await user.type(filter, "observation.channel_start");
    expect(screen.getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.channel_start"]);

    await user.clear(filter);
    await user.type(filter, "observation.channel");
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });

  it("keeps Campaign navigation enabled and opens one unavailable explanation", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness />);
    const campaign = screen.getByRole("button", { name: /Campaign.*Unavailable/ });

    expect(campaign).toBeEnabled();
    await user.click(campaign);

    expect(campaign).toHaveAttribute("aria-current", "page");
    expect(screen.getAllByRole("region", { name: /form$/ })).toHaveLength(1);
    expect(screen.getByRole("region", { name: "Campaign form" }))
      .toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Reserved for capability 4 (streaming evidence)."))
      .toBeVisible();
  });

  it("keeps projected delivery and source metadata in a native collapsed disclosure", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness />);
    await user.click(screen.getByRole("button", { name: /Instrument/ }));
    const field = screen.getByRole("article", { name: "model.foregrounds.ref_freq" });
    const summary = within(field).getByText("Delivery and source metadata");
    const details = summary.closest("details");

    expect(details).not.toHaveAttribute("open");
    await user.click(summary);
    expect(details).toHaveAttribute("open");
    expect(details).toHaveTextContent("model.foregrounds.ref_freq");
    expect(details).toHaveTextContent("static — part of the jit cache key");
    expect(details).toHaveTextContent("Hz");
  });

  it("retains completeness, severity, and preset-change badges on ordinary navigation", () => {
    render(<ConfigHarness />);

    expect(screen.getByRole("navigation", { name: "Configuration sections" }))
      .toHaveTextContent("Beam — 1 incomplete · 2 refuse · 1 warn · 4 report · 3 preset changes");
    expect(screen.getByRole("button", { name: /Beam/ })).toHaveTextContent(
      "1 incomplete · 2 refuse · 1 warn · 4 report · 3 preset changes",
    );
    expect(screen.getByRole("button", { name: /Beam/ })).toHaveAttribute("aria-current", "page");
  });

  it("keeps ConfigForms as a compatibility component for the split workspace", () => {
    const candidate = editorSession();
    const { transport } = transportFor(candidate);
    render(
      <ConfigForms
        session={candidate}
        transport={transport}
        drafts={{ draft: NO_DRAFT, begin: () => true, update: () => undefined, clear: () => undefined }}
        disabled={false}
        disabledReason={null}
        requestedPath={null}
        onAccept={() => undefined}
        onEditYaml={() => undefined}
        onRun={() => undefined}
      />,
    );

    expect(screen.getByRole("region", { name: "Schema-projected forms" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Beam form" })).toBeVisible();
    expect(screen.getByRole("complementary", { name: "Config inspector" })).toBeVisible();
    expect(screen.getByRole("button", { name: /Beam/ })).toHaveTextContent("1 incomplete");
    expect(screen.queryByText("phi0 deg")).not.toBeInTheDocument();
  });

  it("renders file metadata as a Server path textbox and submits one exact field draft", async () => {
    const user = userEvent.setup();
    const candidate = editorSession();
    const { transport, setField } = transportFor(candidate);
    render(<ConfigHarness candidate={candidate} transport={transport} />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));

    const input = screen.getByRole("textbox", { name: "Server path" });
    expect(input).toHaveValue("/srv/data/input.h5");
    expect(input).not.toHaveAttribute("type", "file");
    await user.clear(input);
    await user.type(input, "/srv/data/new.h5");
    await user.click(within(screen.getByRole("article", { name: "observation.server_path" }))
      .getByRole("button", { name: "Apply field" }));

    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1",
      "observation.server_path",
      "/srv/data/new.h5",
      false,
      4,
    ));
    expect(setField).toHaveBeenCalledTimes(1);
  });

  it("uses only the declared native primitive contracts and preserves false, zero, and empty", async () => {
    const user = userEvent.setup();
    const candidate = editorSession();
    const { transport, setField } = transportFor(candidate);
    render(<ConfigHarness candidate={candidate} transport={transport} />);

    const normalize = screen.getByRole("combobox", { name: "normalize" });
    await user.selectOptions(normalize, "solid_angle");
    await user.click(within(screen.getByRole("article", { name: "resources.beams.horn.normalize" }))
      .getByRole("button", { name: "Apply field" }));
    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1", "resources.beams.horn.normalize", "solid_angle", false, 4,
    ));

    await user.click(screen.getByRole("button", { name: /Observation/ }));
    expect(screen.getByRole("checkbox", { name: "calibrated" })).not.toBeChecked();
    expect(screen.getByRole("textbox", { name: "channel start" })).toHaveValue("0");
    expect(screen.getByRole("textbox", { name: "observer note" })).toHaveValue("");

    await user.click(screen.getByRole("checkbox", { name: "calibrated" }));
    await user.click(within(screen.getByRole("article", { name: "observation.calibrated" }))
      .getByRole("button", { name: "Apply field" }));
    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1", "observation.calibrated", true, false, 4,
    ));

    const channel = screen.getByRole("textbox", { name: "channel start" });
    await user.clear(channel);
    await user.type(channel, "-17");
    await user.click(within(screen.getByRole("article", { name: "observation.channel_start" }))
      .getByRole("button", { name: "Apply field" }));
    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1", "observation.channel_start", -17, false, 4,
    ));

    const note = screen.getByRole("textbox", { name: "observer note" });
    await user.type(note, "exact text");
    await user.click(within(screen.getByRole("article", { name: "observation.note" }))
      .getByRole("button", { name: "Apply field" }));
    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1", "observation.note", "exact text", false, 4,
    ));
  });

  it.each(["01", "+1", "1.0"])(
    "rejects the loose integer spelling %s without transport",
    async (raw) => {
      const user = userEvent.setup();
      const candidate = editorSession();
      const { transport, setField } = transportFor(candidate);
      render(<ConfigHarness candidate={candidate} transport={transport} />);
      await user.click(screen.getByRole("button", { name: /Observation/ }));
      const field = screen.getByRole("article", { name: "observation.channel_start" });
      const input = within(field).getByRole("textbox", { name: "channel start" });

      await user.clear(input);
      await user.type(input, raw);
      await user.click(within(field).getByRole("button", { name: "Apply field" }));

      expect(within(field).getByRole("alert")).toHaveTextContent("Enter a whole number");
      expect(setField).not.toHaveBeenCalled();
    },
  );

  it("rejects an unsafe integer while retaining its exact raw draft", async () => {
    const user = userEvent.setup();
    const candidate = editorSession();
    const { transport, setField } = transportFor(candidate);
    render(<ConfigHarness candidate={candidate} transport={transport} />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const field = screen.getByRole("article", { name: "observation.channel_start" });
    const input = within(field).getByRole("textbox", { name: "channel start" });

    await user.clear(input);
    await user.type(input, "9007199254740992");
    await user.click(within(field).getByRole("button", { name: "Apply field" }));

    expect(within(field).getByRole("alert"))
      .toHaveTextContent("Whole number is outside the safe range");
    expect(input).toHaveValue("9007199254740992");
    expect(setField).not.toHaveBeenCalled();
  });

  it("routes generic projected values to exact YAML context without inferring a scalar control", async () => {
    const user = userEvent.setup();
    const onEditYaml = vi.fn();
    render(<ConfigHarness onEditYaml={onEditYaml} />);
    await user.click(screen.getByRole("button", { name: /Instrument/ }));
    const field = screen.getByRole("article", { name: "model.foregrounds.ref_freq" });

    expect(within(field).queryByRole("spinbutton")).not.toBeInTheDocument();
    expect(within(field).queryByRole("textbox")).not.toBeInTheDocument();
    await user.click(within(field).getByRole("button", { name: "Edit in YAML" }));
    expect(onEditYaml).toHaveBeenCalledWith("model.foregrounds.ref_freq");
  });

  it("retains the owning raw draft and original base revision across an accepted refresh", async () => {
    const user = userEvent.setup();
    const original = editorSession(4);
    const refreshed = editorSession(
      9,
      withFieldValue(forms, "observation.server_path", "/srv/data/server-refresh.h5"),
    );
    const { transport, setField } = transportFor(refreshed);
    const view = render(<ConfigHarness candidate={original} transport={transport} />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const input = screen.getByRole("textbox", { name: "Server path" });
    await user.clear(input);
    await user.type(input, "/srv/data/my-draft.h5");

    view.rerender(<ConfigHarness candidate={refreshed} transport={transport} />);
    expect(screen.getByRole("textbox", { name: "Server path" }))
      .toHaveValue("/srv/data/my-draft.h5");
    await user.click(within(screen.getByRole("article", { name: "observation.server_path" }))
      .getByRole("button", { name: "Apply field" }));

    await waitFor(() => expect(setField).toHaveBeenCalledWith(
      "session-1", "observation.server_path", "/srv/data/my-draft.h5", false, 4,
    ));
  });

  it.each([
    { kind: "yaml", baseRevision: 4, text: "draft" } as DraftEnvelope,
    { kind: "graph", baseRevision: 4, path: "base:gain", rawValue: "{}" } as DraftEnvelope,
    { kind: "field", baseRevision: 4, path: "observation.note", rawValue: "mine" } as DraftEnvelope,
  ])("keeps a field read-only while a non-owning $kind draft exists", async (initialDraft) => {
    const user = userEvent.setup();
    render(
      <ConfigHarness
        initialDraft={initialDraft}
        disabledReason="blocked-reason"
      />,
    );
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const input = screen.getByRole("textbox", { name: "Server path" });
    expect(input).toBeDisabled();
    expect(input).toHaveAccessibleDescription("Editing is blocked");
  });

  it("keeps controls read-only during global busy and exposes the shared reason", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness disabled disabledReason="blocked-reason" />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));

    const input = screen.getByRole("textbox", { name: "Server path" });
    expect(input).toBeDisabled();
    expect(input).toHaveAccessibleDescription("Editing is blocked");
  });

  it("discards only the owning field draft and restores the accepted value", async () => {
    const user = userEvent.setup();
    render(<ConfigHarness />);
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const field = screen.getByRole("article", { name: "observation.server_path" });
    const input = within(field).getByRole("textbox", { name: "Server path" });
    await user.clear(input);
    await user.type(input, "/srv/data/discard.h5");

    await user.click(within(field).getByRole("button", { name: "Discard field" }));
    expect(input).toHaveValue("/srv/data/input.h5");
    expect(within(field).queryByRole("button", { name: "Apply field" })).not.toBeInTheDocument();
  });

  it("navigates a requested field only by an exact projected path", async () => {
    const view = render(<ConfigHarness requestedPath={null} />);
    expect(screen.getByRole("region", { name: "Beam form" })).toBeVisible();

    view.rerender(<ConfigHarness requestedPath="observation.channel_start" />);
    expect(screen.getByRole("region", { name: "Observation form" })).toBeVisible();
    expect(screen.getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.channel_start"]);

    view.rerender(<ConfigHarness requestedPath="observation.channel" />);
    expect(screen.getByRole("region", { name: "Observation form" })).toBeVisible();
    expect(screen.getAllByRole("article").map((field) => field.getAttribute("aria-label")))
      .toEqual(["observation.channel_start"]);
  });

  it("serializes field Apply through the parent runner and retains a rejected raw draft", async () => {
    const user = userEvent.setup();
    let rejectField: ((error: Error) => void) | undefined;
    const initial = editorSession(4);
    const { transport, setField } = transportFor(initial);
    setField.mockImplementationOnce(() => new Promise<EditorSession>((_resolve, reject) => {
      rejectField = reject;
    }));
    render(<SessionEditor initial={initial} transport={transport} />);
    await user.click(screen.getByRole("tab", { name: "Config" }));
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const field = screen.getByRole("article", { name: "observation.server_path" });
    const input = within(field).getByRole("textbox", { name: "Server path" });
    await user.clear(input);
    await user.type(input, "/srv/data/conflict.h5");
    const apply = within(field).getByRole("button", { name: "Apply field" });

    fireEvent.click(apply);
    fireEvent.click(apply);
    expect(setField).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByText("Another action is running")).toBeVisible());
    expect(screen.getByRole("button", { name: "Save YAML" }))
      .toHaveAccessibleDescription("Another action is running");

    await act(async () => rejectField?.(new Error("revision conflict")));
    expect(await screen.findByRole("alert")).toHaveTextContent("revision conflict");
    expect(screen.getByRole("textbox", { name: "Server path" }))
      .toHaveValue("/srv/data/conflict.h5");
    expect(screen.getByText("Unsaved field: observation.server_path")).toBeVisible();
  });

  it("installs the complete accepted field response and clears the draft only after success", async () => {
    const user = userEvent.setup();
    const initial = editorSession(4);
    const accepted = editorSession(
      5,
      withFieldValue(forms, "observation.server_path", "/srv/data/accepted.h5"),
    );
    const { transport, setField } = transportFor(initial);
    setField.mockResolvedValueOnce(accepted);
    render(<SessionEditor initial={initial} transport={transport} />);
    await user.click(screen.getByRole("tab", { name: "Config" }));
    await user.click(screen.getByRole("button", { name: /Observation/ }));
    const field = screen.getByRole("article", { name: "observation.server_path" });
    const input = within(field).getByRole("textbox", { name: "Server path" });
    await user.clear(input);
    await user.type(input, "/srv/data/accepted.h5");

    await user.click(within(field).getByRole("button", { name: "Apply field" }));
    expect(await screen.findByText("Updated observation.server_path")).toBeVisible();
    expect(screen.getByText("Revision 5")).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Server path" }))
      .toHaveValue("/srv/data/accepted.h5");
    expect(screen.queryByText("Unsaved field: observation.server_path")).not.toBeInTheDocument();
  });
});
