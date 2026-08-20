import { FieldEditor, type FieldEditorProps } from "./FieldEditor";
import type { ProjectedWidget } from "./types";

function deliveryLabel(delivery: string | null) {
  if (delivery?.startsWith("static")) {
    return "static — part of the jit cache key";
  }
  return delivery === "traced" ? "traced — trainable array" : delivery;
}

function projectedValue(widget: ProjectedWidget) {
  if (!widget.present) return "Not set";
  if (widget.value === "") return '\"\"';
  if (widget.value === undefined) return "undefined";
  if (typeof widget.value === "string") return widget.value;
  const encoded = JSON.stringify(widget.value);
  return encoded ?? String(widget.value);
}

type Props = { widget: ProjectedWidget } & Omit<FieldEditorProps, "widget">;

export function ConfigField({ widget, ...editorProps }: Props) {
  const delivery = deliveryLabel(widget.delivery);
  return (
    <article aria-label={widget.path} aria-disabled={widget.disabled || undefined}>
      <header>
        <h3>{widget.label}</h3>
        {widget.must_decide && <strong>must decide</strong>}
      </header>
      <p>
        Current value: <output aria-label={`${widget.label} current value`}>
          {projectedValue(widget)}
        </output>
      </p>
      {widget.choices.length > 0 && <p>Choices: {widget.choices.join(" · ")}</p>}
      {widget.reason && <p>{widget.reason}</p>}
      <FieldEditor widget={widget} {...editorProps} />
      <details>
        <summary>Delivery and source metadata</summary>
        <dl>
          <dt>Projected path</dt>
          <dd>{widget.path}</dd>
          <dt>Catalog path</dt>
          <dd>{widget.path_pattern}</dd>
          <dt>Widget contract</dt>
          <dd>{widget.widget}</dd>
          {widget.dimension && <><dt>Dimension</dt><dd>{widget.dimension}</dd></>}
          {widget.unit_policy && <><dt>Unit policy</dt><dd>{widget.unit_policy}</dd></>}
          {delivery && <><dt>Delivery</dt><dd>{delivery}</dd></>}
        </dl>
      </details>
    </article>
  );
}
