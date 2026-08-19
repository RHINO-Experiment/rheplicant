import { useState } from "react";

import { ConfigForms } from "./ConfigForms";
import { GraphEditor } from "./GraphEditor";
import { ValidationLedger } from "./ValidationLedger";
import { RequestError } from "./api";
import type { EditorSession, SessionTransport } from "./types";

type ReadFile = (file: File) => Promise<string>;
type SaveFile = (yamlText: string) => Promise<void> | void;

interface Props {
  initial: EditorSession;
  transport: SessionTransport;
  readFile?: ReadFile;
  saveFile?: SaveFile;
}

function browserReadFile(file: File) {
  return file.text();
}

function browserSaveFile(yamlText: string) {
  const url = URL.createObjectURL(new Blob([yamlText], { type: "application/yaml" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "rheplicant.yaml";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function SessionEditor({
  initial,
  transport,
  readFile = browserReadFile,
  saveFile = browserSaveFile,
}: Props) {
  const [session, setSession] = useState(initial);
  const [yamlDraft, setYamlDraft] = useState(initial.document.yaml_text);
  const [status, setStatus] = useState("Ready");
  const [yamlDiagnostic, setYamlDiagnostic] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pendingYaml = yamlDraft !== session.document.yaml_text;

  function accept(next: EditorSession, message: string) {
    setSession(next);
    setYamlDraft(next.document.yaml_text);
    setYamlDiagnostic(null);
    setStatus(message);
  }

  async function run(
    action: () => Promise<EditorSession>,
    message: string,
  ) {
    setBusy(true);
    try {
      accept(await action(), message);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function load(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const yamlText = await readFile(file);
      accept(
        await transport.load(session.session_id, yamlText, session.revision),
        `Loaded ${file.name}`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      event.target.value = "";
      setBusy(false);
    }
  }

  async function save() {
    setBusy(true);
    try {
      await saveFile(session.document.yaml_text);
      accept(
        await transport.save(session.session_id, session.revision),
        "YAML saved",
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function applyYamlDraft() {
    setBusy(true);
    try {
      accept(
        await transport.replaceYaml(
          session.session_id,
          yamlDraft,
          session.revision,
        ),
        "YAML edit applied",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (error instanceof RequestError && error.status === 422) {
        setYamlDiagnostic(message);
      }
      setStatus(message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Rheplicant config editor</h1>
        <p>YAML is the sole scientific state; controls are projections.</p>
        <div aria-label="Editor session state">
          <strong>{session.dirty ? "Unsaved changes" : "Saved"}</strong>
          <span>{session.validation_stale ? "Validation stale" : "Validation current"}</span>
          <span>Revision {session.revision}</span>
        </div>
      </header>

      <nav aria-label="History and file actions">
        <button
          disabled={busy || pendingYaml || !session.can_undo}
          onClick={() => run(
            () => transport.undo(session.session_id, session.revision),
            "Undid YAML edit",
          )}
        >
          Undo
        </button>
        <button
          disabled={busy || pendingYaml || !session.can_redo}
          onClick={() => run(
            () => transport.redo(session.session_id, session.revision),
            "Redid YAML edit",
          )}
        >
          Redo
        </button>
        <label>
          Load YAML
          <input
            aria-label="Load YAML file"
            type="file"
            accept=".yaml,.yml,application/yaml,text/yaml,text/plain"
            disabled={busy}
            onChange={load}
          />
        </label>
        <button disabled={busy || pendingYaml} onClick={save}>Save YAML</button>
        <button
          disabled={busy || pendingYaml || session.document.validation.run_blocked}
        >
          Run
        </button>
      </nav>

      <GraphEditor
        session={session}
        transport={transport}
        disabled={busy || pendingYaml}
        onAccept={(next, message) => accept(next, message)}
      />

      <ConfigForms
        forms={session.document.forms}
        badges={session.document.validation.section_badges}
      />

      <ValidationLedger validation={session.document.validation} />

      <section aria-label="YAML source of truth">
        <textarea
          aria-label="YAML source of truth"
          value={yamlDraft}
          disabled={busy}
          onChange={(event) => {
            setYamlDraft(event.target.value);
            setYamlDiagnostic(null);
          }}
        />
        <button
          disabled={busy || yamlDraft === session.document.yaml_text}
          onClick={applyYamlDraft}
        >
          Apply YAML edit
        </button>
        {yamlDiagnostic && (
          <p role="alert" aria-label="YAML parse diagnostic">{yamlDiagnostic}</p>
        )}
      </section>
      <p role="status">{status}</p>
    </main>
  );
}
