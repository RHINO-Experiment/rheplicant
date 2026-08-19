import type { FormProjection, ProjectedWidget } from "./types";

interface Props {
  forms: FormProjection;
}

function deliveryLabel(delivery: string | null) {
  if (delivery?.startsWith("static")) {
    return "static — part of the jit cache key";
  }
  return delivery === "traced" ? "traced — trainable array" : null;
}

function Widget({ widget }: { widget: ProjectedWidget }) {
  const delivery = deliveryLabel(widget.delivery);
  return (
    <article aria-label={widget.path} aria-disabled={widget.disabled || undefined}>
      <header>
        <h3>{widget.label}</h3>
        {widget.must_decide && <strong>must decide</strong>}
      </header>
      {widget.choices.length > 0 && (
        <span>{widget.choices.join(" · ")}</span>
      )}
      {widget.dimension && <span>{widget.dimension}</span>}
      {delivery && <span>{delivery}</span>}
      {widget.reason && <p>{widget.reason}</p>}
    </article>
  );
}

export function ConfigForms({ forms }: Props) {
  return (
    <section aria-label="Schema-projected forms">
      <nav aria-label="Configuration sections">
        {forms.sections.map((section) => {
          const incomplete = section.widgets.filter((widget) => widget.must_decide).length;
          return (
            <button key={section.section_id} type="button" disabled={section.disabled}>
              {section.label}
              {incomplete > 0 && ` — ${incomplete} incomplete`}
            </button>
          );
        })}
      </nav>
      {forms.sections.map((section) => (
        <section
          key={section.section_id}
          aria-label={`${section.label} form`}
          aria-disabled={section.disabled || undefined}
        >
          <h2>{section.label}</h2>
          {section.reason && <p>{section.reason}</p>}
          {section.widgets
            .filter((widget) => widget.visible)
            .map((widget) => <Widget key={widget.path} widget={widget} />)}
        </section>
      ))}
    </section>
  );
}
