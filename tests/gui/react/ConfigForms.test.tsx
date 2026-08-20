import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { ConfigForms } from "../../../src/rheplicant/gui/react/ConfigForms";
import { useConfigWorkspace } from "../../../src/rheplicant/gui/react/ConfigWorkspace";
import type { FormProjection, SectionBadge } from "../../../src/rheplicant/gui/react/types";

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

function ConfigHarness({ sectionBadges = badges }: { sectionBadges?: SectionBadge[] }) {
  const surface = useConfigWorkspace({ forms, badges: sectionBadges });
  return <>{surface.main}{surface.inspector}</>;
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
    render(<ConfigForms forms={forms} />);

    expect(screen.getByRole("region", { name: "Schema-projected forms" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Beam form" })).toBeVisible();
    expect(screen.getByRole("complementary", { name: "Config inspector" })).toBeVisible();
    expect(screen.getByRole("button", { name: /Beam/ })).toHaveTextContent("1 incomplete");
    expect(screen.queryByText("phi0 deg")).not.toBeInTheDocument();
  });
});
