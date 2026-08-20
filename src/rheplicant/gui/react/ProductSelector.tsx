import { useEffect, useRef, useState } from "react";

import type { OutputProductProjection } from "./types";

function toggled(values: string[], value: string, enabled: boolean) {
  return enabled ? [...values, value].filter((item, index, all) => all.indexOf(item) === index) : values.filter((item) => item !== value);
}

interface Props {
  products: OutputProductProjection[];
  declaredRuns: string[];
  expanded: string | null;
  onExpand(name: string | null): void;
  onChange(product: OutputProductProjection): void;
  disabled: boolean;
  disabledReasonId?: string;
}

export function ProductSelector({ products, declaredRuns, expanded, onExpand, onChange, disabled, disabledReasonId }: Props) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const expandedProduct = expanded === null ? null : products.find((product) => product.name === expanded) ?? null;
  const enabledProducts = products.filter((product) => product.enabled);

  useEffect(() => {
    if (open) optionRefs.current[activeIndex]?.focus();
  }, [activeIndex, open]);

  function update(row: OutputProductProjection, values: Partial<OutputProductProjection>) {
    onChange({ ...row, ...values });
  }
  function select(index: number) {
    onExpand(products[index].name);
    setOpen(false);
  }
  function openPicker() {
    const selected = products.findIndex((product) => product.name === expanded);
    setActiveIndex(selected >= 0 ? selected : 0);
    setOpen(true);
  }
  function moveOption(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number | null = null;
    if (event.key === "ArrowDown") next = (index + 1) % products.length;
    if (event.key === "ArrowUp") next = (index - 1 + products.length) % products.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = products.length - 1;
    if (next !== null) {
      event.preventDefault();
      setActiveIndex(next);
      optionRefs.current[next]?.focus();
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
      {enabledProducts.length > 0 && <ul aria-label="Enabled products">{enabledProducts.map((product) => <li key={product.name}><button type="button" disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onClick={() => onExpand(product.name)}>{`Expand ${product.name} product settings`}</button></li>)}</ul>}
      <button type="button" disabled={disabled} aria-describedby={disabled ? disabledReasonId : undefined} onClick={openPicker}>Add product</button>
      {open && <div role="listbox" aria-label="Available products">{products.map((row, index) => <button key={row.name} ref={(element) => { optionRefs.current[index] = element; }} role="option" aria-selected={row.name === expanded} tabIndex={index === activeIndex ? 0 : -1} type="button" onClick={() => select(index)} onKeyDown={(event) => moveOption(event, index)}>{row.name}</button>)}</div>}
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
