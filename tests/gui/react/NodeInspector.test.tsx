import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { NodeInspector } from "../../../src/rheplicant/gui/react/NodeInspector";
import {
  NO_DRAFT,
  type DraftCoordinator,
  type DraftEnvelope,
} from "../../../src/rheplicant/gui/react/drafts";
import type {
  EditorSession,
  NodeCard,
  NodeField,
  NodeInstance,
  SessionTransport,
} from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

/** NodeInspector reads exactly two members of the session -- `session_id` and
 *  `revision`, both verified by grep -- so the fixture states those two rather
 *  than a hundred lines of unrelated projection. The cast is narrow on
 *  purpose: widening what the component reads should break this file. */
const SESSION = { session_id: "session-1", revision: 4 } as unknown as EditorSession;

function field(overrides: Partial<NodeField> = {}): NodeField {
  return {
    name: "depth",
    path: "model.global_signal.depth",
    label: "depth",
    control: "quantity",
    required: true,
    has_default: false,
    default: null,
    dimension: "K",
    unit_policy: "optional",
    units: ["K", "celsius"],
    choices: [],
    delivery: "traced",
    help: "trough depth [K] (positive number gives absorption).",
    resource_kind: null,
    forms: ["bare", "shorthand", "quantity"],
    typed: true,
    present: true,
    form: "quantity",
    number: 0.5,
    unit: "K",
    written: { value: 0.5, unit: "K" },
    ...overrides,
  };
}

function card(overrides: Partial<NodeCard> = {}): NodeCard {
  return {
    node_id: "global_signal",
    label: "global signal",
    kind: "source",
    description: "21 cm global signal",
    explanation: "Configure the global signal.",
    editable: true,
    reserved: false,
    many: false,
    segment: "forward",
    lit: true,
    count: 1,
    configuration: "single",
    settings: { depth: { value: 0.5, unit: "K" } },
    instances: [],
    stage_names: [],
    typed_form: true,
    typed_form_reason: null,
    type_choices: ["GlobalSignalOperator"],
    selected_type: "GlobalSignalOperator",
    fields: [field()],
    extra_keys: [],
  removed_by_type: {},
  stages: [],
  from_fields: [],
    ...overrides,
  };
}

function transport(): SessionTransport {
  return { editNode: vi.fn() } as unknown as SessionTransport;
}

/** The real coordinator, so a typed edit and the textarea share one draft in
 *  the same render rather than each keeping its own copy. */
function Harness({
  selected,
  onEdit,
}: {
  selected: NodeCard;
  onEdit?: (envelope: DraftEnvelope) => void;
}) {
  const [draft, setDraft] = useState<DraftEnvelope>(NO_DRAFT);
  const drafts: DraftCoordinator = {
    draft,
    begin: (next) => {
      setDraft(next);
      onEdit?.(next);
      return true;
    },
    update: (next) => {
      setDraft(next);
      onEdit?.(next);
    },
    clear: () => setDraft(NO_DRAFT),
  };
  return (
    <NodeInspector
      session={SESSION}
      transport={transport()}
      drafts={drafts}
      selected={selected}
      activeVariant={null}
      disabled={false}
      disabledReason={null}
      onAccept={vi.fn()}
      onStatus={vi.fn()}
    />
  );
}

function openSpellings(subject: string) {
  fireEvent.click(screen.getByRole("checkbox", { name: `${subject} value spellings` }));
}

function textarea() {
  return screen.getByRole("textbox", { name: "Node settings JSON" });
}

describe("typed node fields", () => {
  it("keeps the raw JSON textarea for every editable node, always", () => {
    render(<Harness selected={card()} />);

    expect(textarea()).toBeVisible();
    expect(screen.getByRole("group", { name: "global_signal typed fields" })).toBeVisible();
  });

  it("writes a typed edit into the same draft the textarea shows", () => {
    render(<Harness selected={card()} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "depth" }), {
      target: { value: "0.75" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ depth: { value: 0.75, unit: "K" } }, null, 2));
  });

  it("changes the unit without touching the number", () => {
    /** `celsius` is affine -- offset 273.15 -- so a control that "helpfully"
     *  converted would produce a finite, correctly-shaped, wrong answer that
     *  nothing downstream can detect. */
    render(<Harness selected={card({
      settings: { depth: { value: 293, unit: "K" } },
      fields: [field({ number: 293, written: { value: 293, unit: "K" } })],
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "depth unit" }), {
      target: { value: "celsius" },
    });

    expect(textarea()).toHaveValue(
      JSON.stringify({ depth: { value: 293, unit: "celsius" } }, null, 2),
    );
  });

  it("keeps a shorthand field written as shorthand", () => {
    render(<Harness selected={card({
      settings: { width: "5 MHz" },
      fields: [field({
        name: "width",
        label: "width",
        dimension: "Hz",
        units: ["Hz", "kHz", "MHz", "GHz"],
        form: "shorthand",
        number: 5,
        unit: "MHz",
        written: "5 MHz",
      })],
    })} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "width" }), {
      target: { value: "6" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ width: "6 MHz" }, null, 2));
  });

  it("deletes the key when the control is cleared rather than writing a null", () => {
    render(<Harness selected={card()} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "depth" }), {
      target: { value: "" },
    });

    expect(textarea()).toHaveValue("{}");
  });

  it("disables the typed controls when the textarea is not valid JSON", () => {
    render(<Harness selected={card()} />);

    fireEvent.change(textarea(), { target: { value: "{ not json" } });

    expect(screen.getByRole("spinbutton", { name: "depth" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "depth unit" })).toBeDisabled();
  });

  it("shows the gate reason instead of controls, and still shows the textarea", () => {
    render(<Harness selected={card({
      typed_form: false,
      typed_form_reason: "python: target; its class is not resolved in the browser.",
      type_choices: [],
      selected_type: null,
      fields: [],
    })} />);

    expect(
      screen.getByText("python: target; its class is not resolved in the browser."),
    ).toBeVisible();
    expect(screen.queryByRole("spinbutton", { name: "depth" })).toBeNull();
    expect(textarea()).toBeVisible();
  });

  it("leaves a value no control can represent read-only and says why", () => {
    render(<Harness selected={card({
      settings: { depth: { file: "depth.h5" } },
      fields: [field({ typed: false, form: "file", number: null, unit: null })],
    })} />);

    expect(screen.getByRole("spinbutton", { name: "depth" })).toBeDisabled();
    expect(screen.getByText(/written as file/)).toBeVisible();
  });

  it("offers no unit select for a dimension with one spelling", () => {
    render(<Harness selected={card({
      fields: [field({ dimension: "dimensionless", units: ["dimensionless"] })],
    })} />);

    expect(screen.queryByRole("combobox", { name: "depth unit" })).toBeNull();
  });

  it("offers no unit select where a unit is forbidden", () => {
    render(<Harness selected={card({
      fields: [field({ unit_policy: "forbidden", units: [] })],
    })} />);

    expect(screen.queryByRole("combobox", { name: "depth unit" })).toBeNull();
  });

  it("names the extra keys it has no control for", () => {
    render(<Harness selected={card({ extra_keys: ["snapshot_before", "eqx_leaves"] })} />);

    expect(screen.getByText(/snapshot_before, eqx_leaves/)).toBeVisible();
  });

  it("offers the classes as a select, with no class chosen yet", () => {
    render(<Harness selected={card({
      node_id: "noise",
      type_choices: ["NoiseOperator", "RadiometerNoiseOperator"],
      selected_type: null,
      fields: [],
    })} />);

    const select = screen.getByRole("combobox", { name: "noise type" });
    expect(select).toHaveValue("");
    expect(within(select).getAllByRole("option").map((option) => option.textContent))
      .toEqual(["not chosen", "NoiseOperator", "RadiometerNoiseOperator"]);
  });

  it("edits an enum field through a select carrying its own members", () => {
    render(<Harness selected={card({
      node_id: "cw_tone",
      settings: {},
      fields: [field({
        name: "lineshape",
        label: "lineshape",
        control: "select",
        required: false,
        dimension: "structural",
        unit_policy: "forbidden",
        units: [],
        choices: ["sinc2", "gaussian"],
        delivery: "static_str",
        present: false,
        form: "absent",
        number: null,
        unit: null,
        written: null,
      })],
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "lineshape" }), {
      target: { value: "gaussian" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ lineshape: "gaussian" }, null, 2));
  });

  it("starts no second draft: one Apply commits both views", () => {
    const seen: DraftEnvelope[] = [];
    render(<Harness selected={card()} onEdit={(envelope) => seen.push(envelope)} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "depth" }), {
      target: { value: "0.75" },
    });
    fireEvent.change(textarea(), {
      target: { value: JSON.stringify({ depth: { value: 0.9, unit: "K" } }, null, 2) },
    });

    expect(seen).toHaveLength(2);
    expect(new Set(seen.map((envelope) => envelope.kind === "graph" && envelope.path))).toEqual(
      new Set(["base:global_signal:settings"]),
    );
    expect(screen.getAllByRole("button", { name: /Apply configuration to global_signal/ }))
      .toHaveLength(1);
  });
});

/** `noise` is one of the two nodes that hold more than one operator class.
 *  The two share no fields at all, so every type change here destroys every
 *  written value -- which is why the change is confirmed rather than applied. */
function noiseCard(overrides: Partial<NodeCard> = {}): NodeCard {
  return card({
    node_id: "noise",
    label: "noise",
    kind: "transform",
    settings: { type: "NoiseOperator", sigma: { value: 0.05, unit: "K" } },
    type_choices: ["NoiseOperator", "RadiometerNoiseOperator"],
    selected_type: "NoiseOperator",
    fields: [field({
      name: "sigma",
      label: "sigma",
      path: "model.noise.sigma",
      number: 0.05,
      written: { value: 0.05, unit: "K" },
    })],
    removed_by_type: { NoiseOperator: [], RadiometerNoiseOperator: ["sigma"] },
    ...overrides,
  });
}

describe("changing the operator class", () => {
  it("offers no select where there is only one class to choose", () => {
    render(<Harness selected={card()} />);

    expect(screen.queryByRole("combobox", { name: "global_signal type" })).toBeNull();
  });

  it("names what the change would remove, and writes nothing yet", () => {
    render(<Harness selected={noiseCard()} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "RadiometerNoiseOperator" },
    });

    const confirm = screen.getByRole("group", { name: "noise type change" });
    expect(within(confirm).getByText(/sigma/)).toBeVisible();
    expect(textarea()).toHaveValue(
      JSON.stringify({ type: "NoiseOperator", sigma: { value: 0.05, unit: "K" } }, null, 2),
    );
  });

  it("removes exactly the named keys once confirmed", () => {
    render(<Harness selected={noiseCard()} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "RadiometerNoiseOperator" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Change noise to RadiometerNoiseOperator" }));

    expect(textarea()).toHaveValue(
      JSON.stringify({ type: "RadiometerNoiseOperator" }, null, 2),
    );
    expect(screen.queryByRole("group", { name: "noise type change" })).toBeNull();
  });

  it("keeps the keys no operator class owns", () => {
    /** `snapshot_before` belongs to the NODE, not to its class. A change of
     *  class has no business taking it. */
    render(<Harness selected={noiseCard({
      settings: {
        type: "NoiseOperator",
        sigma: { value: 0.05, unit: "K" },
        snapshot_before: "pre_noise",
      },
      extra_keys: ["snapshot_before"],
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "RadiometerNoiseOperator" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Change noise to RadiometerNoiseOperator" }));

    expect(textarea()).toHaveValue(
      JSON.stringify(
        { type: "RadiometerNoiseOperator", snapshot_before: "pre_noise" },
        null,
        2,
      ),
    );
  });

  it("cancelling writes nothing and puts the select back", () => {
    render(<Harness selected={noiseCard()} />);
    const select = screen.getByRole("combobox", { name: "noise type" });

    fireEvent.change(select, { target: { value: "RadiometerNoiseOperator" } });
    fireEvent.click(screen.getByRole("button", { name: "Keep NoiseOperator" }));

    expect(select).toHaveValue("NoiseOperator");
    expect(textarea()).toHaveValue(
      JSON.stringify({ type: "NoiseOperator", sigma: { value: 0.05, unit: "K" } }, null, 2),
    );
    expect(screen.queryByRole("group", { name: "noise type change" })).toBeNull();
  });

  it("applies at once when the document has written nothing to lose", () => {
    render(<Harness selected={noiseCard({
      settings: { type: "NoiseOperator" },
      fields: [],
      removed_by_type: { NoiseOperator: [], RadiometerNoiseOperator: [] },
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "RadiometerNoiseOperator" },
    });

    expect(screen.queryByRole("group", { name: "noise type change" })).toBeNull();
    expect(textarea()).toHaveValue(JSON.stringify({ type: "RadiometerNoiseOperator" }, null, 2));
  });

  it("writes the type into a document that had not chosen one", () => {
    render(<Harness selected={noiseCard({
      settings: {},
      selected_type: null,
      fields: [],
      removed_by_type: { NoiseOperator: [], RadiometerNoiseOperator: [] },
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "NoiseOperator" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ type: "NoiseOperator" }, null, 2));
  });

  it("still uses the one shared draft", () => {
    const seen: DraftEnvelope[] = [];
    render(<Harness selected={noiseCard()} onEdit={(envelope) => seen.push(envelope)} />);

    fireEvent.change(screen.getByRole("combobox", { name: "noise type" }), {
      target: { value: "RadiometerNoiseOperator" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Change noise to RadiometerNoiseOperator" }));

    expect(seen.map((envelope) => envelope.kind === "graph" && envelope.path))
      .toEqual(["base:noise:settings"]);
  });
});

/** A CHAIN of two filters. The node itself has no fields -- it is the list --
 *  and each entry carries its own. All three filter classes own `mode`, so a
 *  type change here keeps it. */
function filtersCard(overrides: Partial<NodeCard> = {}): NodeCard {
  const sidereal: NodeInstance = {
    instance_id: "filters_1",
    label: "filters 1",
    settings: { type: "SiderealFilter", n_days: 3, mode: "extract" },
    slot: ["0"],
    typed_form: true,
    typed_form_reason: null,
    type_choices: ["FourierBandFilter", "SiderealFilter", "SkySpaceFilter"],
    selected_type: "SiderealFilter",
    fields: [
      field({
        name: "n_days", label: "n days", path: "model.filters[].n_days",
        control: "integer", delivery: "static_int", dimension: "count",
        unit_policy: "optional", units: [], number: 3, unit: null,
        form: "bare", written: 3,
      }),
    ],
    extra_keys: [],
    removed_by_type: {
      FourierBandFilter: ["n_days"], SiderealFilter: [], SkySpaceFilter: ["n_days"],
    },
  };
  const fourier: NodeInstance = {
    ...sidereal,
    instance_id: "filters_2",
    label: "filters 2",
    slot: ["1"],
    settings: { type: "FourierBandFilter", axis: "time" },
    selected_type: "FourierBandFilter",
    fields: [
      field({
        name: "axis", label: "axis", path: "model.filters[].axis",
        control: "text", delivery: "static_str", dimension: "structural",
        unit_policy: "forbidden", units: [], number: null, unit: null,
        form: "bare", written: "time",
      }),
    ],
    removed_by_type: {
      FourierBandFilter: [], SiderealFilter: ["axis"], SkySpaceFilter: ["axis"],
    },
  };
  return card({
    node_id: "filters",
    label: "filters",
    kind: "transform",
    many: true,
    configuration: "chain",
    count: 2,
    settings: [sidereal.settings, fourier.settings],
    typed_form: false,
    typed_form_reason: "Many node: each instance carries its own fields.",
    type_choices: [],
    selected_type: null,
    fields: [],
    instances: [sidereal, fourier],
    ...overrides,
  });
}

describe("a many node's instances", () => {
  it("gives each entry its own fieldset and the node the reason it has none", () => {
    render(<Harness selected={filtersCard()} />);

    expect(screen.getByText("Many node: each instance carries its own fields.")).toBeVisible();
    expect(screen.getByRole("group", { name: "filters 1 typed fields" })).toBeVisible();
    expect(screen.getByRole("group", { name: "filters 2 typed fields" })).toBeVisible();
  });

  it("writes an entry's edit into that entry alone", () => {
    render(<Harness selected={filtersCard()} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "n days" }), {
      target: { value: "7" },
    });

    expect(textarea()).toHaveValue(JSON.stringify([
      { type: "SiderealFilter", n_days: 7, mode: "extract" },
      { type: "FourierBandFilter", axis: "time" },
    ], null, 2));
  });

  it("keeps the entries apart when two fields share a name", () => {
    render(<Harness selected={filtersCard()} />);
    const first = screen.getByRole("group", { name: "filters 1 typed fields" });
    const second = screen.getByRole("group", { name: "filters 2 typed fields" });

    expect(within(first).getByRole("spinbutton", { name: "n days" })).toHaveValue(3);
    expect(within(second).getByRole("textbox", { name: "axis" })).toHaveValue("time");
  });

  it("confirms a type change inside one entry and leaves the other alone", () => {
    render(<Harness selected={filtersCard()} />);
    const first = screen.getByRole("group", { name: "filters 1 typed fields" });

    fireEvent.change(within(first).getByRole("combobox", { name: "filters 1 type" }), {
      target: { value: "FourierBandFilter" },
    });
    expect(within(first).getByText(/n_days/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Change filters 1 to FourierBandFilter" }));

    expect(textarea()).toHaveValue(JSON.stringify([
      { type: "FourierBandFilter", mode: "extract" },
      { type: "FourierBandFilter", axis: "time" },
    ], null, 2));
  });

  it("keeps two entries of the SAME class apart", () => {
    /** The fixtures above have differently-named fields, so a control state
     *  shared across instances would be invisible in them. Two SiderealFilters
     *  both own `n_days`, and that is the case that can see it. */
    const first = filtersCard().instances[0];
    const second: NodeInstance = {
      ...first,
      instance_id: "filters_2",
      label: "filters 2",
      slot: ["1"],
      settings: { type: "SiderealFilter", n_days: 5 },
      fields: [{ ...first.fields[0], number: 5, written: 5 }],
    };
    render(<Harness selected={filtersCard({
      settings: [first.settings, second.settings],
      instances: [first, second],
    })} />);
    const groups = ["filters 1", "filters 2"].map((name) =>
      screen.getByRole("group", { name: `${name} typed fields` }));

    expect(within(groups[0]).getByRole("spinbutton", { name: "n days" })).toHaveValue(3);
    expect(within(groups[1]).getByRole("spinbutton", { name: "n days" })).toHaveValue(5);

    fireEvent.change(within(groups[1]).getByRole("spinbutton", { name: "n days" }), {
      target: { value: "9" },
    });

    expect(within(groups[0]).getByRole("spinbutton", { name: "n days" })).toHaveValue(3);
    expect(textarea()).toHaveValue(JSON.stringify([
      { type: "SiderealFilter", n_days: 3, mode: "extract" },
      { type: "SiderealFilter", n_days: 9 },
    ], null, 2));
  });

  it("edits a FAN label through its own name rather than an index", () => {
    const hot: NodeInstance = {
      ...filtersCard().instances[0],
      instance_id: "hot",
      label: "hot",
      slot: ["hot"],
      settings: { t_load: { value: 350, unit: "K" } },
      selected_type: "CalLoadOperator",
      type_choices: ["CalLoadOperator"],
      fields: [field({ name: "t_load", label: "t load", path: "model.cal_loads.*.t_load",
                       number: 350, unit: "K", written: { value: 350, unit: "K" } })],
      removed_by_type: { CalLoadOperator: [] },
    };
    render(<Harness selected={filtersCard({
      node_id: "cal_loads",
      configuration: "fan",
      count: 1,
      settings: { hot: hot.settings },
      instances: [hot],
    })} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "t load" }), {
      target: { value: "400" },
    });

    expect(textarea()).toHaveValue(
      JSON.stringify({ hot: { t_load: { value: 400, unit: "K" } } }, null, 2),
    );
  });
});

/** A composed node: two GainOperator stages one level deeper than an
 *  instance, at `settings.stages[i]`. */
function composedCard(): NodeCard {
  const stage = (name: string, gain: number, index: number): NodeInstance => ({
    ...filtersCard().instances[0],
    instance_id: `gain_stage_${index + 1}`,
    label: name,
    slot: ["stages", String(index)],
    settings: { name, type: "GainOperator", gain },
    type_choices: ["GainOperator"],
    selected_type: "GainOperator",
    fields: [field({
      name: "gain", label: "gain", path: "model.gain.gain",
      dimension: "dimensionless", units: ["dimensionless"],
      forms: ["bare", "shorthand", "quantity"],
      number: gain, unit: null, form: "bare", written: gain,
    })],
    removed_by_type: { GainOperator: [] },
  });
  return card({
    node_id: "gain",
    settings: {
      compose: "cascade",
      stages: [
        { name: "coarse", type: "GainOperator", gain: 1.1 },
        { name: "fine", type: "GainOperator", gain: 1.01 },
      ],
    },
    typed_form: false,
    typed_form_reason: "Composed node: the stages own the fields.",
    type_choices: [],
    selected_type: null,
    fields: [],
    stages: [stage("coarse", 1.1, 0), stage("fine", 1.01, 1)],
  });
}

describe("a composed node's stages", () => {
  it("gives each stage its own fieldset, named after the stage", () => {
    render(<Harness selected={composedCard()} />);

    expect(screen.getByText("Composed node: the stages own the fields.")).toBeVisible();
    expect(screen.getByRole("group", { name: "coarse typed fields" })).toBeVisible();
    expect(screen.getByRole("group", { name: "fine typed fields" })).toBeVisible();
  });

  it("writes a stage's edit one level deeper, into that stage alone", () => {
    render(<Harness selected={composedCard()} />);
    const fine = screen.getByRole("group", { name: "fine typed fields" });

    fireEvent.change(within(fine).getByRole("spinbutton", { name: "gain" }), {
      target: { value: "1.02" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({
      compose: "cascade",
      stages: [
        { name: "coarse", type: "GainOperator", gain: 1.1 },
        { name: "fine", type: "GainOperator", gain: 1.02 },
      ],
    }, null, 2));
  });

  it("never offers the stage name as a field", () => {
    render(<Harness selected={composedCard()} />);

    expect(screen.queryByRole("textbox", { name: "name" })).toBeNull();
  });
});

describe("a from: route", () => {
  it("offers the route's own keys beside the reason it is not a field set", () => {
    render(<Harness selected={card({
      node_id: "beam_spill",
      settings: { from: "projector", t_ground: { value: 300, unit: "K" } },
      typed_form: false,
      typed_form_reason: "from: projector is a constructor route, not a field set.",
      type_choices: [],
      selected_type: null,
      fields: [],
      from_fields: [
        field({
          name: "projector", label: "projector", path: "model.beam_spill.projector",
          control: "opaque", dimension: "structural", unit_policy: "forbidden",
          units: [], forms: ["bare"], typed: false, present: false,
          form: "absent", number: null, unit: null, written: null,
        }),
        field({
          name: "t_ground", label: "t ground", path: "model.beam_spill.t_ground",
          units: ["K", "celsius"], forms: ["bare", "shorthand", "quantity"],
          number: 300, unit: "K", written: { value: 300, unit: "K" },
        }),
      ],
    })} />);

    expect(
      screen.getByText("from: projector is a constructor route, not a field set."),
    ).toBeVisible();
    const route = screen.getByRole("group", { name: "beam_spill route typed fields" });
    expect(within(route).getByRole("spinbutton", { name: "t ground" })).toHaveValue(300);
    expect(within(route).getByRole("textbox", { name: "projector" })).toBeDisabled();
  });
});

describe("re-spelling one field", () => {
  it("stays out of the way until it is asked for", () => {
    render(<Harness selected={card()} />);

    expect(screen.queryByRole("combobox", { name: "depth form" })).toBeNull();
    expect(screen.getByRole("checkbox", { name: "global_signal value spellings" }))
      .not.toBeChecked();
  });

  it("offers no toggle where no field has a second spelling", () => {
    render(<Harness selected={card({
      fields: [field({ unit_policy: "forbidden", units: [], forms: ["bare"] })],
    })} />);

    expect(screen.queryByRole("checkbox", { name: "global_signal value spellings" }))
      .toBeNull();
  });

  it("offers only the shapes this layer can read back", () => {
    render(<Harness selected={card()} />);
    openSpellings("global_signal");

    const form = screen.getByRole("combobox", { name: "depth form" });
    expect(within(form).getAllByRole("option").map((option) => option.textContent))
      .toEqual(["bare", "shorthand", "quantity"]);
    expect(form).toHaveValue("quantity");
  });

  it("re-spells the envelope as the shorthand without touching the number", () => {
    render(<Harness selected={card()} />);
    openSpellings("global_signal");

    fireEvent.change(screen.getByRole("combobox", { name: "depth form" }), {
      target: { value: "shorthand" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ depth: "0.5 K" }, null, 2));
  });

  it("re-spells it as a bare scalar, and the unit goes with the spelling", () => {
    render(<Harness selected={card()} />);
    openSpellings("global_signal");

    fireEvent.change(screen.getByRole("combobox", { name: "depth form" }), {
      target: { value: "bare" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ depth: 0.5 }, null, 2));
  });

  it("keeps a later number edit in the chosen spelling", () => {
    render(<Harness selected={card()} />);
    openSpellings("global_signal");

    fireEvent.change(screen.getByRole("combobox", { name: "depth form" }), {
      target: { value: "shorthand" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "depth" }), {
      target: { value: "0.9" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ depth: "0.9 K" }, null, 2));
  });

  it("offers no switcher for a field with one shape even when opened", () => {
    render(<Harness selected={card({
      fields: [
        field({ name: "sigma", label: "sigma" }),
        field({ name: "mode", label: "mode", unit_policy: "forbidden", units: [], forms: ["bare"] }),
      ],
    })} />);
    openSpellings("global_signal");

    expect(screen.getByRole("combobox", { name: "sigma form" })).toBeVisible();
    expect(screen.queryByRole("combobox", { name: "mode form" })).toBeNull();
  });

  it("leaves a bare number bare when only its value changes", () => {
    /** P0 turned a bare number into an envelope the moment it was edited,
     *  because a unit was offered. The written spelling is the default now. */
    render(<Harness selected={card({
      settings: { depth: 0.5 },
      fields: [field({ form: "bare", number: 0.5, unit: null, written: 0.5 })],
    })} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "depth" }), {
      target: { value: "0.7" },
    });

    expect(textarea()).toHaveValue(JSON.stringify({ depth: 0.7 }, null, 2));
  });
});

describe("help text and resource pickers", () => {
  it("shows the sentence the operator writes about each field", () => {
    render(<Harness selected={card()} />);

    expect(
      screen.getByText("trough depth [K] (positive number gives absorption)."),
    ).toBeVisible();
  });

  it("says nothing where the operator says nothing", () => {
    render(<Harness selected={card({ fields: [field({ help: "" })] })} />);

    expect(screen.getByRole("group", { name: "global_signal typed fields" }))
      .toBeVisible();
    expect(screen.queryByText(/trough depth/)).toBeNull();
  });

  it("offers the declared resources and writes a ref", () => {
    render(<Harness selected={card({
      node_id: "observed_astro_sky",
      settings: {},
      fields: [field({
        name: "projector", label: "projector", path: "model.observed_astro_sky.projector",
        control: "resource", resource_kind: "projectors",
        choices: ["resources.projectors.drift", "resources.projectors.second"],
        dimension: "structural", unit_policy: "forbidden", units: [], forms: ["bare"],
        present: false, form: "absent", number: null, unit: null, written: null,
        help: "the projector this sky is convolved with.",
      })],
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "projector" }), {
      target: { value: "resources.projectors.second" },
    });

    expect(textarea()).toHaveValue(
      JSON.stringify({ projector: { ref: "resources.projectors.second" } }, null, 2),
    );
  });

  it("shows the reference already written", () => {
    render(<Harness selected={card({
      node_id: "observed_astro_sky",
      settings: { projector: { ref: "resources.projectors.drift" } },
      fields: [field({
        name: "projector", label: "projector", path: "model.observed_astro_sky.projector",
        control: "resource", resource_kind: "projectors",
        choices: ["resources.projectors.drift"],
        unit_policy: "forbidden", units: [], forms: ["bare"],
        form: "ref", number: null, unit: null,
        written: { ref: "resources.projectors.drift" },
      })],
    })} />);

    expect(screen.getByRole("combobox", { name: "projector" }))
      .toHaveValue("resources.projectors.drift");
  });

  it("clearing the picker removes the key rather than writing an empty ref", () => {
    render(<Harness selected={card({
      node_id: "observed_astro_sky",
      settings: { projector: { ref: "resources.projectors.drift" } },
      fields: [field({
        name: "projector", label: "projector", path: "model.observed_astro_sky.projector",
        control: "resource", resource_kind: "projectors",
        choices: ["resources.projectors.drift"],
        unit_policy: "forbidden", units: [], forms: ["bare"],
        form: "ref", number: null, unit: null,
        written: { ref: "resources.projectors.drift" },
      })],
    })} />);

    fireEvent.change(screen.getByRole("combobox", { name: "projector" }), {
      target: { value: "" },
    });

    expect(textarea()).toHaveValue("{}");
  });

  it("says so when the document has declared nothing to pick", () => {
    render(<Harness selected={card({
      node_id: "observed_astro_sky",
      settings: {},
      fields: [field({
        name: "projector", label: "projector", control: "resource",
        resource_kind: "projectors", choices: [],
        unit_policy: "forbidden", units: [], forms: ["bare"],
        present: false, form: "absent", number: null, unit: null, written: null,
      })],
    })} />);

    expect(screen.getByText(/no projectors declared/)).toBeVisible();
  });
});
