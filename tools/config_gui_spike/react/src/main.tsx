import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { apiTransport } from "./api";
import { ConfigEditor } from "./ConfigEditor";
import type { EditorSnapshot } from "./types";
import "./style.css";

const DEFAULT_YAML = `runtime:
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
  const [initial, setInitial] = useState<EditorSnapshot | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiTransport.snapshot(DEFAULT_YAML).then(setInitial).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, []);

  if (error) return <p role="alert">{error}</p>;
  if (!initial) return <p>Loading canonical signal path…</p>;
  return <ConfigEditor initial={initial} transport={apiTransport} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
