import type { ProjectedSection, SectionBadge } from "./types";

interface ConfigSectionNavProps {
  sections: ProjectedSection[];
  badges: SectionBadge[];
  activeSection: string | null;
  onSelect: (sectionId: string) => void;
}

function sectionLabels(section: ProjectedSection, badge: SectionBadge | undefined) {
  const incomplete = badge?.incomplete
    ?? section.widgets.filter((widget) => widget.must_decide).length;
  return [
    incomplete > 0 ? `${incomplete} incomplete` : null,
    badge && badge.refuse > 0 ? `${badge.refuse} refuse` : null,
    badge && badge.warn > 0 ? `${badge.warn} warn` : null,
    badge && badge.report > 0 ? `${badge.report} report` : null,
    badge && badge.preset_changes > 0
      ? `${badge.preset_changes} preset changes`
      : null,
  ].filter((label): label is string => label !== null);
}

export function ConfigSectionNav({
  sections,
  badges,
  activeSection,
  onSelect,
}: ConfigSectionNavProps) {
  return (
    <nav aria-label="Configuration sections">
      {sections.map((section) => {
        const badge = badges.find((candidate) => candidate.section_id === section.section_id);
        const labels = sectionLabels(section, badge);
        return (
          <button
            key={section.section_id}
            type="button"
            aria-current={section.section_id === activeSection ? "page" : undefined}
            onClick={() => onSelect(section.section_id)}
          >
            {section.label}
            {section.disabled && " — Unavailable"}
            {labels.length > 0 && ` — ${labels.join(" · ")}`}
          </button>
        );
      })}
    </nav>
  );
}
