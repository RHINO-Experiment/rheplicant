import { useModelWorkspace, type ModelWorkspaceProps } from "./ModelWorkspace";

export type { ModelWorkspaceProps } from "./ModelWorkspace";

export function GraphEditor(props: ModelWorkspaceProps) {
  const surface = useModelWorkspace(props);
  return (
    <section aria-label="Graph-guided instrument editor">
      {surface.main}
      {surface.inspector}
    </section>
  );
}
