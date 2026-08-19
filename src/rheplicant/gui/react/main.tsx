import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { SessionEditor } from "./SessionEditor";
import { createSession, sessionTransport } from "./api";
import type { EditorSession } from "./types";

const STARTER_YAML = `runtime:
  jax_enable_x64: true
model:
  gain:
    type: GainOperator
    gain: 1.0
runs:
  - name: forward
    kind: forward
`;

function App() {
  const [session, setSession] = useState<EditorSession | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    createSession(STARTER_YAML)
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

  if (error) return <p role="alert">Could not start the editor: {error}</p>;
  if (!session) return <p role="status">Creating local YAML editor session…</p>;
  return <SessionEditor initial={session} transport={sessionTransport} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
