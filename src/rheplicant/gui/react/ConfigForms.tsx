import {
  useConfigWorkspace,
  type ConfigWorkspaceProps,
} from "./ConfigWorkspace";

export function ConfigForms(props: ConfigWorkspaceProps) {
  const surface = useConfigWorkspace(props);
  return <>{surface.main}{surface.inspector}</>;
}
