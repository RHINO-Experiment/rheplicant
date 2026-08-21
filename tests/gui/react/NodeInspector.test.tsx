import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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

  it("says which types are on offer when the document has chosen none", () => {
    render(<Harness selected={card({
      node_id: "noise",
      type_choices: ["NoiseOperator", "RadiometerNoiseOperator"],
      selected_type: null,
      fields: [],
    })} />);

    expect(screen.getByText(/NoiseOperator, RadiometerNoiseOperator/)).toBeVisible();
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
