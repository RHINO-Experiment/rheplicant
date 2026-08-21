import { useEffect, useState } from "react";

import { ConfigField } from "./ConfigField";
import { ConfigSectionNav } from "./ConfigSectionNav";
import type { DraftCoordinator } from "./drafts";
import type { WorkspaceSurface } from "./ModelWorkspace";
import type { EditorSession, ProjectedWidget, SessionTransport } from "./types";

export interface ConfigWorkspaceProps {
  session: EditorSession;
  transport: SessionTransport;
  drafts: DraftCoordinator;
  disabled: boolean;
  disabledReason: string | null;
  requestedPath: string | null;
  onAccept(next: EditorSession, message: string): void;
  onEditYaml(path: string): void;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

function matchesFilter(widget: ProjectedWidget, filter: string) {
  const needle = filter.trim().toLowerCase();
  return !needle
    || widget.label.toLowerCase().includes(needle)
    || widget.path.toLowerCase() === needle;
}

type FieldGroupProps = Omit<ConfigWorkspaceProps, "requestedPath"> & {
  label: string;
  widgets: ProjectedWidget[];
};

function FieldGroup({ label, widgets, ...editorProps }: FieldGroupProps) {
  if (widgets.length === 0) return null;
  return (
    <section aria-label={label}>
      <h3>{label}</h3>
      {widgets.map((widget) => (
        <ConfigField key={widget.path} widget={widget} {...editorProps} />
      ))}
    </section>
  );
}

export function useConfigWorkspace({
  session,
  transport,
  drafts,
  disabled,
  disabledReason,
  requestedPath,
  onAccept,
  onEditYaml,
  onRun,
}: ConfigWorkspaceProps): WorkspaceSurface {
  const forms = session.document.forms;
  const badges = session.document.validation.section_badges;
  const [activeSection, setActiveSection] = useState<string | null>(
    forms.sections[0]?.section_id ?? null,
  );
  const [filter, setFilter] = useState("");
  const selected = forms.sections.find((section) => section.section_id === activeSection)
    ?? forms.sections[0];
  const resolvedSection = selected?.section_id ?? null;

  useEffect(() => {
    if (activeSection !== resolvedSection) setActiveSection(resolvedSection);
  }, [activeSection, resolvedSection]);

  useEffect(() => {
    if (requestedPath === null) return;
    const section = forms.sections.find((candidate) => candidate.widgets.some(
      (widget) => widget.path === requestedPath,
    ));
    if (!section) return;
    setActiveSection(section.section_id);
    setFilter(requestedPath);
  }, [forms, requestedPath]);

  const visible = (selected?.widgets ?? [])
    .filter((widget) => widget.visible)
    .filter((widget) => matchesFilter(widget, filter));
  const missingRequired = visible.filter((widget) => widget.must_decide);
  const present = visible.filter((widget) => !widget.must_decide && widget.present);
  const optionalAbsent = visible.filter((widget) => !widget.must_decide && !widget.present);

  const main = (
    <section aria-label="Schema-projected forms">
      <header>
        <h2>Configuration</h2>
        <p>Browse one server-projected section at a time.</p>
        <label>
          Filter configuration fields
          <input
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </label>
      </header>
      <ConfigSectionNav
        sections={forms.sections}
        badges={badges}
        activeSection={resolvedSection}
        onSelect={setActiveSection}
      />
      {selected && (
        <section
          role="region"
          aria-label={`${selected.label} form`}
          aria-disabled={selected.disabled || undefined}
        >
          <h2>{selected.label}</h2>
          {selected.reason && <p>{selected.reason}</p>}
          <FieldGroup
            label="Missing required fields"
            widgets={missingRequired}
            session={session}
            transport={transport}
            drafts={drafts}
            disabled={disabled || selected.disabled}
            disabledReason={disabledReason}
            onAccept={onAccept}
            onEditYaml={onEditYaml}
            onRun={onRun}
          />
          <FieldGroup
            label="Present values"
            widgets={present}
            session={session}
            transport={transport}
            drafts={drafts}
            disabled={disabled || selected.disabled}
            disabledReason={disabledReason}
            onAccept={onAccept}
            onEditYaml={onEditYaml}
            onRun={onRun}
          />
          <FieldGroup
            label="Optional fields not set"
            widgets={optionalAbsent}
            session={session}
            transport={transport}
            drafts={drafts}
            disabled={disabled || selected.disabled}
            disabledReason={disabledReason}
            onAccept={onAccept}
            onEditYaml={onEditYaml}
            onRun={onRun}
          />
          {visible.length === 0 && !selected.disabled && <p>No fields match this filter.</p>}
        </section>
      )}
    </section>
  );
  const inspector = (
    <aside aria-label="Config inspector">
      <h2>{selected?.label ?? "Configuration"}</h2>
      <p>Field values and metadata are read directly from the accepted server projection.</p>
    </aside>
  );
  return { main, inspector };
}
