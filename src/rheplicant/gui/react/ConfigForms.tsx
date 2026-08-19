import type { FormProjection, ProjectedWidget, SectionBadge } from "./types";

interface Props {
  forms: FormProjection;
  badges?: SectionBadge[];
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

export function ConfigForms({ forms, badges = [] }: Props) {
  return (
    <section aria-label="Schema-projected forms">
      <nav aria-label="Configuration sections">
        {forms.sections.map((section) => {
          const badge = badges.find((candidate) => candidate.section_id === section.section_id);
          const incomplete = badge?.incomplete
            ?? section.widgets.filter((widget) => widget.must_decide).length;
          const labels = [
            incomplete > 0 ? `${incomplete} incomplete` : null,
            badge && badge.refuse > 0 ? `${badge.refuse} refuse` : null,
            badge && badge.warn > 0 ? `${badge.warn} warn` : null,
            badge && badge.report > 0 ? `${badge.report} report` : null,
            badge && badge.preset_changes > 0
              ? `${badge.preset_changes} preset changes`
              : null,
          ].filter((label): label is string => label !== null);
          return (
            <button key={section.section_id} type="button" disabled={section.disabled}>
              {section.label}
              {labels.length > 0 && ` — ${labels.join(" · ")}`}
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
