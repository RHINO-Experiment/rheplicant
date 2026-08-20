import { useState } from "react";
import type { ReactNode } from "react";

import { ConfigForms } from "./ConfigForms";
import { GraphEditor } from "./GraphEditor";
import { OutputWorkflow } from "./OutputWorkflow";
import { PreviewPanel } from "./PreviewPanel";
import { ValidationLedger } from "./ValidationLedger";
import { WorkbenchShell } from "./WorkbenchShell";
import { WorkspaceNav } from "./WorkspaceNav";
import { RequestError } from "./api";
import type { EditorSession, JobKind, SessionTransport } from "./types";
import type { WorkspaceId } from "./WorkspaceNav";

type ReadFile = (file: File) => Promise<string>;
type SaveFile = (yamlText: string) => Promise<void> | void;
type WorkspaceSurface = { main: ReactNode; inspector: ReactNode };

const workspaceLabels: Record<WorkspaceId, string> = {
  model: "Model",
  config: "Config",
  execute: "Execute",
  results: "Results",
};

function emptyInspector(message: string) {
  return <aside aria-label="Context inspector"><p>{message}</p></aside>;
}

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
  const [statusError, setStatusError] = useState(false);
  const [yamlDiagnostic, setYamlDiagnostic] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>("model");
  const pendingYaml = yamlDraft !== session.document.yaml_text;

  function accept(next: EditorSession, message: string) {
    setSession(next);
    setYamlDraft(next.document.yaml_text);
    setYamlDiagnostic(null);
    setStatus(message);
    setStatusError(false);
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
      setStatusError(true);
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
      setStatusError(true);
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
      setStatusError(true);
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
      setStatusError(true);
    } finally {
      setBusy(false);
    }
  }

  function submitJob(kind: JobKind) {
    void run(
      () => transport.submitJob(session.session_id, kind, session.revision),
      `${kind === "preview_forward" ? "Forward preview" : kind} job submitted`,
    );
  }

  async function refreshJobs() {
    setBusy(true);
    try {
      setSession(await transport.refresh(session.session_id));
      setStatus("Job state refreshed");
      setStatusError(false);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setStatusError(true);
    } finally {
      setBusy(false);
    }
  }

  const yamlSource = (
    <section aria-label="YAML source of truth">
      <textarea
        aria-label="YAML source of truth"
        aria-invalid={yamlDiagnostic !== null}
        aria-describedby={yamlDiagnostic ? "yaml-diagnostic" : undefined}
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
        <p id="yaml-diagnostic" role="alert" aria-label="YAML parse diagnostic" className="error-surface">
          {yamlDiagnostic}
        </p>
      )}
    </section>
  );
  const surfaces: Record<WorkspaceId, WorkspaceSurface> = {
    model: {
      main: <GraphEditor session={session} transport={transport} onAccept={accept} disabled={busy || pendingYaml} />,
      inspector: emptyInspector("Select a graph node"),
    },
    config: {
      main: <ConfigForms forms={session.document.forms} badges={session.document.validation.section_badges} />,
      inspector: emptyInspector("Select a field"),
    },
    execute: {
      main: <><OutputWorkflow session={session} transport={transport} onAccept={accept} disabled={busy || pendingYaml} /><PreviewPanel previews={session.document.previews} jobs={session.jobs} disabled={busy || pendingYaml} blocked={session.document.validation.run_blocked} onSubmit={submitJob} /></>,
      inspector: emptyInspector("Select an output"),
    },
    results: {
      main: <ValidationLedger validation={session.document.validation} />,
      inspector: emptyInspector("Select a job"),
    },
  };
  const surface = surfaces[activeWorkspace];

  return (
    <WorkbenchShell
      header={
        <>
          <h1>Rheplicant configuration workbench</h1>
          <p>YAML is the sole scientific state; controls are projections.</p>
          <div aria-label="Editor session state">
            <strong>{session.dirty ? "Unsaved changes" : "Saved"}</strong>
            <span>{session.validation_stale ? "Validation stale" : "Validation current"}</span>
            <span>Revision {session.revision}</span>
          </div>
          <nav aria-label="History and file actions">
            <button disabled={busy || pendingYaml || !session.can_undo} onClick={() => run(() => transport.undo(session.session_id, session.revision), "Undid YAML edit")}>Undo</button>
            <button disabled={busy || pendingYaml || !session.can_redo} onClick={() => run(() => transport.redo(session.session_id, session.revision), "Redid YAML edit")}>Redo</button>
            <label>
              Load YAML
              <input aria-label="Load YAML file" type="file" accept=".yaml,.yml,application/yaml,text/yaml,text/plain" disabled={busy} onChange={load} />
            </label>
            <button disabled={busy || pendingYaml} onClick={save}>Save YAML</button>
            <button disabled={busy} onClick={refreshJobs}>Refresh jobs</button>
          </nav>
        </>
      }
      navigation={<WorkspaceNav active={activeWorkspace} onChange={setActiveWorkspace} />}
      main={<section id={`workspace-panel-${activeWorkspace}`} role="tabpanel" aria-label={workspaceLabels[activeWorkspace]} aria-labelledby={`workspace-tab-${activeWorkspace}`}>{surface.main}</section>}
      inspector={surface.inspector}
      jobs={
        <>
          {yamlSource}
          <p role={statusError ? "alert" : "status"} className={statusError ? "error-surface" : undefined}>
            {status}
          </p>
        </>
      }
      overlay={null}
    />
  );
}
