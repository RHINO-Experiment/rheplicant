import { useEffect, useState } from "react";
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CLOSED_PRODUCT_VIEW,
  ProductSelector,
  type ProductView,
} from "../../../src/rheplicant/gui/react/ProductSelector";
import type { OutputProductProjection } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const NAMES = ["arrays", "aux", "prediction_bands", "posterior_predictives", "timings"];

function product(name: string): OutputProductProjection {
  return {
    name, enabled: false, format: "npz", formats: ["npz"], runs: [], keys: [], themes: [],
    expected_paths: [],
  };
}

// Controlled exactly as useExecuteWorkspace controls it in the app, except that the initial view
// is supplied here. The hook can only reach a subset of the view states the component must render:
// the products themselves come from the session and shrink under it without touching activeIndex.
function Picker({
  view: initial,
  products = NAMES.map(product),
}: {
  view: ProductView;
  products?: OutputProductProjection[];
}) {
  const [view, setView] = useState(initial);
  return (
    <ProductSelector
      products={products}
      declaredRuns={["fit"]}
      view={view}
      onView={setView}
      onChange={vi.fn()}
      disabled={false}
    />
  );
}

function openView(overrides: Partial<ProductView> = {}): ProductView {
  return { ...CLOSED_PRODUCT_VIEW, open: true, ...overrides };
}

function tabStops() {
  return screen.getAllByRole("option").map((option) => option.getAttribute("tabindex"));
}

// Scoped to the listbox on purpose: an expanded product mounts a <select> whose <option> children
// carry the same implicit role, and a bare getAllByRole("option") counts the format list too.
function listboxOptions() {
  return within(screen.getByRole("listbox", { name: "Available products" })).getAllByRole("option");
}

describe("the product listbox keeps exactly one reachable tab stop", () => {
  it("clamps an active index past the end of the list onto the last option", () => {
    // Kills dropping the clamp. Without it no option carries tabindex="0" at all, and a keyboard
    // user cannot tab into the listbox — the roving test above only pins the in-range case.
    render(<Picker view={openView({ activeIndex: 7 })} />);

    expect(screen.getAllByRole("option")).toHaveLength(5);
    expect(tabStops()).toEqual(["-1", "-1", "-1", "-1", "0"]);
  });

  it("clamps a negative active index onto the first option", () => {
    // The other side of the same clamp: Math.max(index, 0).
    render(<Picker view={openView({ activeIndex: -3 })} />);

    expect(tabStops()).toEqual(["0", "-1", "-1", "-1", "-1"]);
  });

  it("clamps onto the filtered list, not the catalogue behind it", () => {
    // An index legal for all five products and illegal for the two that match: the clamp answers
    // for the list actually rendered.
    render(<Picker view={openView({ query: "pre", activeIndex: 4 })} />);

    expect(screen.getAllByRole("option").map((option) => option.textContent))
      .toEqual(["prediction_bands", "posterior_predictives"]);
    expect(tabStops()).toEqual(["-1", "0"]);
  });

  it("moves focus into the listbox from the filter", () => {
    // The one focus mechanism, exercised: focusOption asks the post-render effect to place focus
    // on the clamped active option rather than focusing a stale element mid-render.
    render(<Picker view={openView()} />);
    const filter = screen.getByRole("searchbox", { name: "Filter products" });
    filter.focus();

    const user = userEvent.setup();
    return user.keyboard("{ArrowUp}").then(() => {
      expect(screen.getAllByRole("option")[4]).toHaveFocus();
      expect(tabStops()).toEqual(["-1", "-1", "-1", "-1", "0"]);
    });
  });
});

describe("focus restoration leaves no window on the document body", () => {
  // An earlier sibling's PASSIVE effect runs after every layout effect of the same commit and
  // before the picker's own passive effect, so it observes precisely the window between "the
  // focused option was removed" and "focus was put back".
  function Probe({ seen }: { seen: (Element | null)[] }) {
    useEffect(() => { seen.push(document.activeElement); });
    return null;
  }

  function ProbedPicker({ seen }: { seen: (Element | null)[] }) {
    const [view, setView] = useState<ProductView>(openView());
    return <>
      <Probe seen={seen} />
      <ProductSelector
        products={NAMES.map(product)}
        declaredRuns={["fit"]}
        view={view}
        onView={setView}
        onChange={vi.fn()}
        disabled={false}
      />
    </>;
  }

  it("puts focus back before a passive effect of the same commit can see the body", async () => {
    // Kills useEffect in place of useLayoutEffect. Dismissing the picker unmounts the element
    // that holds focus, so focus falls to <body> in that very commit; a passive restore hands it
    // back only after React yields, by which point the browser may have painted the stranded
    // focus and dispatched focusout at the body.
    const seen: (Element | null)[] = [];
    render(<ProbedPicker seen={seen} />);
    const options = screen.getAllByRole("option");
    options[0].focus();
    expect(options[0]).toHaveFocus();
    seen.length = 0;

    const user = userEvent.setup();
    await user.keyboard("{Escape}");

    expect(screen.getByRole("button", { name: "Add product" })).toHaveFocus();
    expect(seen.length).toBeGreaterThan(0);
    expect(seen).not.toContain(document.body);
    expect(seen.at(-1)).toBe(screen.getByRole("button", { name: "Add product" }));
  });
});

describe("Home and End are not the arrows", () => {
  // Every End in the suite was pressed from index 0 and every Home from the last index — exactly
  // the two positions where the modulo wrap makes Home ≡ (index + 1) % len and
  // End ≡ (index - 1 + len) % len, so neither key was ever distinguished from the arrow it is
  // not. Both are pressed from a MIDDLE index here, on a filtered list of four.
  const FILTERED = ["arrays", "prediction_bands", "posterior_predictives", "timings"];

  function middle() {
    render(<Picker view={openView({ query: "s", activeIndex: 1 })} />);
    const options = screen.getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual(FILTERED);
    options[1].focus();
    expect(options[1]).toHaveFocus();
    return options;
  }

  it("sends Home to the first option from the middle, where ArrowDown would go to the third", async () => {
    const options = middle();
    const user = userEvent.setup();

    await user.keyboard("{Home}");

    expect(options[0]).toHaveFocus();
    expect(tabStops()).toEqual(["0", "-1", "-1", "-1"]);
  });

  it("sends End to the last option from the middle, where ArrowUp would go to the first", async () => {
    const options = middle();
    const user = userEvent.setup();

    await user.keyboard("{End}");

    expect(options[3]).toHaveFocus();
    expect(tabStops()).toEqual(["-1", "-1", "-1", "0"]);
  });

  it("keeps the arrows wrapping while Home and End clamp", async () => {
    // The arrows are the other half of the contract: from the middle they step by one, and from
    // an end they wrap — which is exactly why pressing Home or End there proves nothing.
    const options = middle();
    const user = userEvent.setup();

    await user.keyboard("{ArrowUp}");
    expect(options[0]).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(options[3]).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(options[0]).toHaveFocus();
  });
});

describe("the product listbox names the current option", () => {
  it("marks the expanded product selected and every other option unselected", () => {
    // Kills a hard-coded aria-selected={false} (and a hard-coded true). For a role="listbox" this
    // is the attribute that tells a screen-reader user which option is current.
    render(<Picker view={openView({ expanded: "prediction_bands" })} />);

    expect(listboxOptions().map((option) => (
      [option.textContent, option.getAttribute("aria-selected")]
    ))).toEqual([
      ["arrays", "false"],
      ["aux", "false"],
      ["prediction_bands", "true"],
      ["posterior_predictives", "false"],
      ["timings", "false"],
    ]);
  });

  it("marks nothing selected before any product has been chosen", () => {
    render(<Picker view={openView()} />);

    expect(screen.getAllByRole("option").map((option) => option.getAttribute("aria-selected")))
      .toEqual(["false", "false", "false", "false", "false"]);
  });

  it("returns the roving tab stop to the selection when the picker is reopened", async () => {
    // Kills `activeIndex: 0` in openPicker. Reopening parked the one tab stop on the first product
    // in the catalogue, so a keyboard user had to walk back to their own selection every time.
    const user = userEvent.setup();
    render(<Picker view={CLOSED_PRODUCT_VIEW} />);

    await user.click(screen.getByRole("button", { name: "Add product" }));
    await user.click(screen.getByRole("option", { name: "posterior_predictives" }));
    expect(screen.queryByRole("listbox")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Add product" }));

    const options = listboxOptions();
    expect(options.map((option) => option.getAttribute("tabindex")))
      .toEqual(["-1", "-1", "-1", "0", "-1"]);
    expect(options[3]).toHaveFocus();
    expect(options[3]).toHaveTextContent("posterior_predictives");
    expect(options[3]).toHaveAttribute("aria-selected", "true");
  });
});

describe("the product filter", () => {
  it("matches on the trimmed query", () => {
    // Kills dropping query.trim(): a pasted or space-padded query would match nothing at all and
    // read as "no such product" rather than as whitespace.
    render(<Picker view={openView({ query: "  pre  " })} />);

    expect(screen.getAllByRole("option").map((option) => option.textContent))
      .toEqual(["prediction_bands", "posterior_predictives"]);
  });

  it("renders no listbox at all when nothing matches", () => {
    // Kills a role="listbox" with zero owned options. ARIA requires a listbox to own option or
    // group children, and the "No products match" chip is a sibling of the box, not inside it:
    // a screen-reader user in the listbox would hear an empty list with no explanation.
    render(<Picker view={openView({ query: "zzz" })} />);

    expect(screen.queryByRole("listbox")).toBeNull();
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    const empty = screen.getByText('No products match "zzz"');
    expect(empty).toBeVisible();
    expect(empty.closest("[role]")).toHaveAttribute("role", "status");
  });

  it("keeps the populated listbox owning every option it renders", () => {
    // The other side: whenever the box exists it owns its options directly.
    render(<Picker view={openView()} />);

    const listbox = screen.getByRole("listbox", { name: "Available products" });
    expect(screen.getAllByRole("option")).toHaveLength(5);
    for (const option of screen.getAllByRole("option")) expect(option.parentElement).toBe(listbox);
  });

  it("leaves the popup relationship on the opener rather than half-claiming it on the filter", () => {
    // The opener declares the popup it owns: aria-haspopup, aria-expanded and aria-controls all
    // live there. The filter is a plain searchbox. A bare aria-controls on it named a relationship
    // no pattern backs — the combobox pattern that would license it also requires role="combobox",
    // aria-expanded and aria-autocomplete, and changes the control's role. Adopt the whole pattern
    // deliberately or claim nothing; this pins the "nothing" the component chose.
    render(<Picker view={openView()} />);

    const filter = screen.getByRole("searchbox", { name: "Filter products" });
    for (const attribute of [
      "role", "aria-controls", "aria-expanded", "aria-autocomplete", "aria-activedescendant",
    ]) {
      expect(filter).not.toHaveAttribute(attribute);
    }
    expect(screen.getByRole("button", { name: "Add product" }))
      .toHaveAttribute("aria-controls", "available-products");
  });

  it("stops naming a listbox once the query empties it", async () => {
    // Kills an unconditional aria-controls. A collapsed disclosure may name the popup it will
    // open — that is the APG idiom and the one dangling IDREF axe tolerates, because
    // aria-expanded="false" says the target does not exist yet. An EXPANDED opener has no such
    // excuse, and the empty-listbox fix created exactly that state: expanded, controlling an id
    // nothing in the document carries.
    const user = userEvent.setup();
    render(<Picker view={CLOSED_PRODUCT_VIEW} />);
    const opener = screen.getByRole("button", { name: "Add product" });

    await user.click(opener);
    expect(opener).toHaveAttribute("aria-expanded", "true");
    expect(opener).toHaveAttribute("aria-controls", "available-products");
    expect(document.getElementById("available-products")).not.toBeNull();

    await user.type(screen.getByRole("searchbox", { name: "Filter products" }), "zzz");

    expect(screen.queryByRole("listbox")).toBeNull();
    expect(document.getElementById("available-products")).toBeNull();
    expect(opener).toHaveAttribute("aria-expanded", "true");
    expect(opener).not.toHaveAttribute("aria-controls");
  });

  it("never names an element the document does not carry, in any picker state", async () => {
    // The invariant behind the case above, asked of every state the picker can reach.
    const user = userEvent.setup();
    render(<Picker view={CLOSED_PRODUCT_VIEW} />);
    const opener = screen.getByRole("button", { name: "Add product" });
    const named = () => {
      const id = opener.getAttribute("aria-controls");
      // A collapsed disclosure is the one licensed exception; expanded, the target must exist.
      if (id === null || opener.getAttribute("aria-expanded") === "false") return;
      expect(document.getElementById(id), `aria-controls names a missing element: ${id}`)
        .not.toBeNull();
    };

    named();
    await user.click(opener);
    named();
    const filter = screen.getByRole("searchbox", { name: "Filter products" });
    for (const query of ["pre", "zzz", "", "aux"]) {
      await user.clear(filter);
      if (query !== "") await user.type(filter, query);
      named();
    }
    await user.click(opener);
    named();
  });
});
