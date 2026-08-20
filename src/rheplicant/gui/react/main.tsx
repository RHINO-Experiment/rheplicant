import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { SessionEditor } from "./SessionEditor";
import { StatusChip } from "./StatusChip";
import { createStarterSession, sessionTransport } from "./api";
import type { EditorSession } from "./types";
import { WorkbenchShell } from "./WorkbenchShell";

import "./tokens.css";
import "./editor.css";

export function BootstrapShell({ error }: { error: string }) {
  const loading = error === "";
  return (
    <WorkbenchShell
      header={
        <>
          <h1>Rheplicant configuration workbench</h1>
          <p>YAML is the sole scientific state; controls are projections.</p>
        </>
      }
      navigation={<nav aria-label="Workbench workspaces"><p>Loading workspaces…</p></nav>}
      main={
        <section aria-label="Workbench startup">
          {loading ? (
            <p role="status" aria-label="Workbench startup" aria-busy="true" aria-live="polite">
              Loading the canonical starter and creating the editor session…
            </p>
          ) : (
            <StatusChip
              tone="danger"
              label={`Could not start the editor: ${error}`}
              urgent
            />
          )}
        </section>
      }
      inspector={<aside aria-label="Context inspector"><p>Startup context will appear here.</p></aside>}
      jobs={<p>Startup jobs are unavailable until the editor session is ready.</p>}
      overlay={null}
    />
  );
}

export function App() {
  const [session, setSession] = useState<EditorSession | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    createStarterSession()
      .then((created) => {
        if (active) setSession(created);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, []);

  if (error || !session) return <BootstrapShell error={error} />;
  return <SessionEditor initial={session} transport={sessionTransport} />;
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
