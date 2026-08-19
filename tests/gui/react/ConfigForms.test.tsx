import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ConfigForms } from "../../../src/rheplicant/gui/react/ConfigForms";
import type { FormProjection } from "../../../src/rheplicant/gui/react/types";

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

describe("schema-projected config forms", () => {
  it("shows required decisions, delivery and dimensions while hiding inactive fields", () => {
    render(<ConfigForms forms={forms} />);

    expect(screen.getByRole("navigation", { name: "Configuration sections" }))
      .toHaveTextContent("Beam — 1 incomplete");
    expect(screen.getByText("must decide")).toBeInTheDocument();
    expect(screen.getByText("static — part of the jit cache key")).toBeInTheDocument();
    expect(screen.getByText("Hz")).toBeInTheDocument();
    expect(screen.queryByText("phi0 deg")).not.toBeInTheDocument();
  });

  it("keeps the deferred campaign view visible and disabled", () => {
    render(<ConfigForms forms={forms} />);
    const campaign = screen.getByRole("button", { name: /Campaign/ });
    expect(campaign).toBeDisabled();
    expect(screen.getByText("Reserved for capability 4 (streaming evidence)."))
      .toBeInTheDocument();
  });

  it("renders completeness, pre-flight severity, and preset-diff section badges", () => {
    render(<ConfigForms forms={forms} badges={[
      {
        section_id: "beam",
        incomplete: 1,
        refuse: 2,
        warn: 1,
        report: 0,
        preset_changes: 3,
      },
    ]} />);

    expect(screen.getByRole("button", { name: /Beam/ })).toHaveTextContent(
      "1 incomplete · 2 refuse · 1 warn · 3 preset changes",
    );
  });
});
