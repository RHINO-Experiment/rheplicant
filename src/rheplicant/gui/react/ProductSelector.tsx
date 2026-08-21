import { useLayoutEffect, useRef } from "react";

import { StatusChip } from "./StatusChip";
import type { OutputProductProjection } from "./types";

/** Progressive-disclosure view state for the product picker. Owned by useExecuteWorkspace. */
export interface ProductView {
  open: boolean;
  query: string;
  activeIndex: number;
  expanded: string | null;
}

// Frozen: one module-level object is the initial value in several places, and a single accidental
// mutation would rewrite the starting view of every picker that has not opened yet.
export const CLOSED_PRODUCT_VIEW: ProductView = Object.freeze({ open: false, query: "", activeIndex: 0, expanded: null });

/** Case-insensitive substring match; an empty query keeps every product discoverable. */
function matchingProducts(products: OutputProductProjection[], query: string): OutputProductProjection[] {
  const needle = query.trim().toLowerCase();
  return needle === "" ? products : products.filter((product) => product.name.toLowerCase().includes(needle));
}

function toggled(values: string[], value: string, enabled: boolean) {
  return enabled ? [...values, value].filter((item, index, all) => all.indexOf(item) === index) : values.filter((item) => item !== value);
}

interface Props {
  products: OutputProductProjection[];
  declaredRuns: string[];
  view: ProductView;
  onView(next: ProductView): void;
  onChange(product: OutputProductProjection): void;
  disabled: boolean;
  disabledReasonId?: string;
}

const LISTBOX_ID = "available-products";

export function ProductSelector({ products, declaredRuns, view, onView, onChange, disabled, disabledReasonId }: Props) {
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const openerRef = useRef<HTMLButtonElement | null>(null);
  const focusIntent = useRef<"none" | "option" | "opener">("none");
  const expandedProduct = view.expanded === null ? null : products.find((product) => product.name === view.expanded) ?? null;
  const enabledProducts = products.filter((product) => product.enabled);
  const matches = matchingProducts(products, view.query);
  const activeIndex = matches.length === 0 ? 0 : Math.min(Math.max(view.activeIndex, 0), matches.length - 1);
  // The listbox exists only while the picker is open AND something matches.
  const listboxRendered = view.open && matches.length > 0;
  // aria-controls names the popup a collapsed disclosure WILL open — the APG disclosure idiom,
  // and the one dangling IDREF axe tolerates, precisely because aria-expanded="false" says the
  // target does not exist yet. Once expanded that excuse is gone: the empty-listbox fix created
  // an aria-expanded="true" opener controlling an id nothing in the document carries, and that is
  // the state the reference is dropped in.
  const controlsListbox = listboxRendered || !view.open;

  // Layout, not passive: the element that had focus is removed by the very commit this runs
  // after, so focus falls to <body> first and a passive effect moves it back only after the
  // browser may have painted and dispatched focusout. useLayoutEffect closes that window.
  useLayoutEffect(() => {
    const intent = focusIntent.current;
    focusIntent.current = "none";
    if (intent === "option") optionRefs.current[activeIndex]?.focus();
    if (intent === "opener") openerRef.current?.focus();
  });

  function update(row: OutputProductProjection, values: Partial<OutputProductProjection>) {
    onChange({ ...row, ...values });
  }
  // One mechanism for one job: the intent above places focus after the render that the new active
  // index produces. Focusing here as well aimed a second time at the pre-render element.
  function focusOption(index: number) {
    if (matches.length === 0) return;
    focusIntent.current = "option";
    onView({ ...view, activeIndex: index });
  }
  function select(index: number) {
    const chosen = matches[index];
    if (chosen === undefined) return;
    focusIntent.current = "opener";
    onView({ ...view, open: false, query: "", activeIndex: 0, expanded: chosen.name });
  }
  function openPicker() {
    const selected = products.findIndex((product) => product.name === view.expanded);
    focusIntent.current = "option";
    onView({ ...view, open: true, query: "", activeIndex: selected >= 0 ? selected : 0 });
  }
  /** Dismissal without a commitment: the picker closes and the opener takes the focus back. */
  function closePicker() {
    focusIntent.current = "opener";
    onView({ ...view, open: false, query: "", activeIndex: 0 });
  }
  function togglePicker() {
    if (view.open) closePicker();
    else openPicker();
  }
  function updateQuery(query: string) {
    focusIntent.current = "none";
    onView({ ...view, query, activeIndex: 0 });
  }
  function moveFromQuery(event: React.KeyboardEvent<HTMLInputElement>) {
    // Escape comes before the empty-match guard: no match is exactly when the user wants out.
    if (event.key === "Escape") { event.preventDefault(); closePicker(); return; }
    if (matches.length === 0) return;
    // Home and End stay with the textbox caret, as the WAI-ARIA combobox pattern reserves them;
    // only the arrows carry focus from the filter into the listbox.
    if (event.key === "ArrowDown") { event.preventDefault(); focusOption(0); return; }
    if (event.key === "ArrowUp") { event.preventDefault(); focusOption(matches.length - 1); }
  }
  function moveOption(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    if (event.key === "Escape") { event.preventDefault(); closePicker(); return; }
    let next: number | null = null;
    if (event.key === "ArrowDown") next = (index + 1) % matches.length;
    if (event.key === "ArrowUp") next = (index - 1 + matches.length) % matches.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = matches.length - 1;
    if (next !== null) {
      event.preventDefault();
      focusOption(next);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select(index);
    }
  }
  return (
    <section aria-label="Scientific product selectors">
      <h3>Products</h3>
      {enabledProducts.length > 0 && <ul aria-label="Enabled products">{enabledProducts.map((product) => <li key={product.name}><button type="button" disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onClick={() => onView({ ...view, expanded: product.name })}>{`Expand ${product.name} product settings`}</button></li>)}</ul>}
      <button ref={openerRef} type="button" disabled={disabled} aria-haspopup="listbox" aria-expanded={view.open} aria-controls={controlsListbox ? LISTBOX_ID : undefined} aria-describedby={disabled ? disabledReasonId : undefined} onClick={togglePicker}>Add product</button>
      {view.open && <div className="product-picker">
        {/* A plain searchbox: the opener declares the popup it owns. A bare aria-controls here
            named a relationship no pattern backs — the combobox pattern that licenses it also
            requires role="combobox", aria-expanded and aria-autocomplete. */}
        <label>Filter products<input type="search" value={view.query} onChange={(event) => updateQuery(event.target.value)} onKeyDown={moveFromQuery} /></label>
        {/* No box when it would own nothing: ARIA requires a listbox to own option or group
            children, and the empty-state chip below is its sibling, not its child. */}
        {listboxRendered && <div id={LISTBOX_ID} role="listbox" aria-label="Available products">{matches.map((row, index) => <button key={row.name} ref={(element) => { optionRefs.current[index] = element; }} role="option" aria-selected={row.name === view.expanded} tabIndex={index === activeIndex ? 0 : -1} type="button" onClick={() => select(index)} onKeyDown={(event) => moveOption(event, index)}>{row.name}</button>)}</div>}
        {matches.length === 0 && <StatusChip tone="neutral" label={`No products match "${view.query}"`} />}
      </div>}
      {expandedProduct !== null && <fieldset aria-label={`${expandedProduct.name} product settings`} className="output-product">
        <legend>{expandedProduct.name}</legend>
        <label><input type="checkbox" aria-label={`Write ${expandedProduct.name}`} checked={expandedProduct.enabled} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update(expandedProduct, { enabled: event.target.checked })} />Write {expandedProduct.name}</label>
        <label>Format<select aria-label={`${expandedProduct.name} format`} value={expandedProduct.format} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update(expandedProduct, { format: event.target.value })}>{expandedProduct.formats.map((format) => <option key={format} value={format}>{format}</option>)}</select></label>
        {declaredRuns.length > 0 && expandedProduct.name !== "assembly" && expandedProduct.name !== "signal_paths" && <fieldset><legend>Runs</legend>{declaredRuns.map((run) => <label key={run}><input type="checkbox" aria-label={`${expandedProduct.name} run ${run}`} checked={expandedProduct.runs.includes(run)} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update(expandedProduct, { runs: toggled(expandedProduct.runs, run, event.target.checked) })} />{run}</label>)}</fieldset>}
        {(expandedProduct.name === "aux" || expandedProduct.name === "taps") && <label>Keys, comma-separated<input aria-label={`${expandedProduct.name} keys`} value={expandedProduct.keys.join(", ")} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update(expandedProduct, { keys: event.target.value.split(",").map((item) => item.trim()).filter(Boolean) })} /></label>}
        {expandedProduct.name === "signal_paths" && <fieldset><legend>Themes</legend>{["light", "dark"].map((theme) => <label key={theme}><input type="checkbox" aria-label={`signal_paths theme ${theme}`} checked={expandedProduct.themes.includes(theme)} disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onChange={(event) => update(expandedProduct, { themes: toggled(expandedProduct.themes, theme, event.target.checked) })} />{theme}</label>)}</fieldset>}
        {expandedProduct.expected_paths.length > 0 && <ul aria-label={`${expandedProduct.name} expected paths`}>{expandedProduct.expected_paths.map((path) => <li key={path}><code>{path}</code></li>)}</ul>}
      </fieldset>}
    </section>
  );
}
