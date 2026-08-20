import type { ReactNode } from "react";

export interface WorkbenchShellProps {
  header: ReactNode;
  navigation: ReactNode;
  main: ReactNode;
  inspector: ReactNode;
  jobs: ReactNode;
  overlay: ReactNode;
}

export function WorkbenchShell({
  header,
  navigation,
  main,
  inspector,
  jobs,
  overlay,
}: WorkbenchShellProps) {
  return (
    <main className="rheplicant-editor workbench-shell">
      <header className="workbench-header">{header}</header>
      <div className="workbench-layout">
        <div className="workbench-navigation">{navigation}</div>
        <div className="workbench-main">{main}</div>
        <div className="workbench-inspector">{inspector}</div>
      </div>
      <footer className="workbench-jobs">{jobs}</footer>
      {overlay}
    </main>
  );
}
