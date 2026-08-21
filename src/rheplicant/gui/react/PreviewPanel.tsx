import type { PreviewProjection } from "./types";

function values(items: number[], unit: string | null) {
  return `${items.join(", ")}${unit ? ` ${unit}` : ""}`;
}

/** Text-derived preview information only; priced work is rendered by ExecutionActions. */
export function ForwardPreviewSummary({ previews }: { previews: PreviewProjection }) {
  return (
    <section aria-label="Forward preview summary">
      <h2>Previews and cost</h2>
      <p>{previews.forward_cost.label}</p>
      <section aria-label="Continuous axis and shape previews">
        <h3>Continuous previews</h3>
        <p>The signal-path graph, axes, and shapes update without evaluating the model.</p>
        {previews.axes.length === 0 ? <p>No text-declared axes are available.</p> : <dl>{previews.axes.map((axis) => <div key={axis.axis}><dt>{axis.axis}</dt><dd>{values(axis.first, axis.unit)} … {values(axis.last, axis.unit)}{` · count ${axis.count}`}{axis.spacing !== null && ` · spacing ${axis.spacing}`}{axis.precision_ok !== null && <span>{` · precision ${axis.precision_ok ? "safe" : "unsafe"}`}</span>}</dd></div>)}</dl>}
        <ul>{previews.shapes.map((shape) => <li key={shape.symbol}><code>{shape.symbol}</code> = {shape.value}</li>)}</ul>
      </section>
    </section>
  );
}
