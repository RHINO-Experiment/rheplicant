import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { ConfigForms } from "./ConfigForms";
import { canUpdateDraft, draftBlocksMutation, draftLabel, NO_DRAFT, type DraftCoordinator, type DraftEnvelope } from "./drafts";
import { FirstJobConfirmation } from "./FirstJobConfirmation";
import { GraphEditor } from "./GraphEditor";
import { OnboardingChecklist } from "./OnboardingChecklist";
import { OutputWorkflow } from "./OutputWorkflow";
import { PreviewPanel } from "./PreviewPanel";
import { ValidationLedger } from "./ValidationLedger";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { WorkbenchShell } from "./WorkbenchShell";
import { WorkspaceNav } from "./WorkspaceNav";
import { YamlDrawer } from "./YamlDrawer";
import { RequestError } from "./api";
import type { EditorSession, JobKind, SessionTransport } from "./types";
import type { WorkspaceId } from "./WorkspaceNav";

type ReadFile = (file: File) => Promise<string>;
type SaveFile = (yamlText: string) => Promise<void> | void;
type WorkspaceSurface = { main: ReactNode; inspector: ReactNode };

const workspaceLabels: Record<WorkspaceId, string> = { model: "Model", config: "Config", execute: "Execute", results: "Results" };
function emptyInspector(message: string) { return <aside aria-label="Context inspector"><p>{message}</p></aside>; }

interface Props { initial: EditorSession; transport: SessionTransport; readFile?: ReadFile; saveFile?: SaveFile; }
function browserReadFile(file: File) { return file.text(); }
function browserSaveFile(yamlText: string) {
  const url = URL.createObjectURL(new Blob([yamlText], { type: "application/yaml" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "rheplicant.yaml";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function SessionEditor({ initial, transport, readFile = browserReadFile, saveFile = browserSaveFile }: Props) {
  const [session, setSession] = useState(initial);
  const [draft, setDraft] = useState<DraftEnvelope>(NO_DRAFT);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [statusError, setStatusError] = useState(false);
  const [yamlDiagnostic, setYamlDiagnostic] = useState<string | null>(null);
  const [yamlConflict, setYamlConflict] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>("model");
  const [pendingJob, setPendingJob] = useState<JobKind | null>(null);
  const [securityAcknowledged, setSecurityAcknowledged] = useState(false);
  const jobOpener = useRef<HTMLElement | null>(null);
  const capturedJobOpener = useRef<HTMLButtonElement | null>(null);
  const restoreJobFocus = useRef(false);

  useEffect(() => {
    if (pendingJob !== null || busy || !restoreJobFocus.current) return;
    restoreJobFocus.current = false;
    const opener = jobOpener.current;
    jobOpener.current = null;
    if (opener instanceof HTMLButtonElement && opener.isConnected && !opener.disabled) {
      opener.focus();
      return;
    }
    const fallback = document.getElementById(`workspace-tab-${activeWorkspace}`);
    if (fallback instanceof HTMLElement) fallback.focus();
  }, [pendingJob, busy, activeWorkspace]);

  const coordinator: DraftCoordinator = {
    draft,
    begin(next) { if (draft.kind !== "none") return false; setDraft(next); return true; },
    update(next) {
      if (!canUpdateDraft(draft, next)) throw new Error("Cannot replace a different editor draft");
      setDraft(next);
    },
    clear() { setDraft(NO_DRAFT); },
  };
  const mutationBlocked = busy || draftBlocksMutation(draft);
  const jobBlocked = mutationBlocked || session.document.validation.run_blocked;
  const mutationReason = busy ? "Another action is running" : draftLabel(draft);

  function accept(next: EditorSession, message: string, clearDraft = true) {
    setSession(next);
    if (clearDraft) coordinator.clear();
    setYamlDiagnostic(null);
    setYamlConflict(null);
    setStatus(message);
    setStatusError(false);
  }
  async function run(action: () => Promise<EditorSession>, message: string) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try { accept(await action(), message); }
    catch (error) { setStatus(error instanceof Error ? error.message : String(error)); setStatusError(true); }
    finally { busyRef.current = false; setBusy(false); }
  }
  async function load(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || mutationBlocked || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try { const yamlText = await readFile(file); accept(await transport.load(session.session_id, yamlText, session.revision), `Loaded ${file.name}`); }
    catch (error) { setStatus(error instanceof Error ? error.message : String(error)); setStatusError(true); }
    finally { event.target.value = ""; busyRef.current = false; setBusy(false); }
  }
  async function save() {
    if (mutationBlocked || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try { await saveFile(session.document.yaml_text); accept(await transport.save(session.session_id, session.revision), "YAML saved"); }
    catch (error) { setStatus(error instanceof Error ? error.message : String(error)); setStatusError(true); }
    finally { busyRef.current = false; setBusy(false); }
  }
  function updateYamlDraft(text: string) {
    setYamlDiagnostic(null);
    setYamlConflict(null);
    if (draft.kind === "none") {
      if (text !== session.document.yaml_text) coordinator.begin({ kind: "yaml", baseRevision: session.revision, text });
    } else if (draft.kind === "yaml") coordinator.update({ kind: "yaml", baseRevision: draft.baseRevision, text });
  }
  async function applyYamlDraft() {
    if (draft.kind !== "yaml") return;
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try { accept(await transport.replaceYaml(session.session_id, draft.text, draft.baseRevision), "YAML edit applied"); }
    catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (error instanceof RequestError && error.status === 422) { setYamlDiagnostic(message); setStatus("YAML edit needs repair"); setStatusError(false); }
      else if (error instanceof RequestError && error.status === 409) { setYamlConflict(message); setStatus("YAML revision conflict"); setStatusError(false); }
      else { setStatus(message); setStatusError(true); }
    } finally { busyRef.current = false; setBusy(false); }
  }
  function discardDraft() { coordinator.clear(); setYamlDiagnostic(null); setYamlConflict(null); }
  function submitJob(kind: JobKind) {
    if (jobBlocked) return;
    void run(() => transport.submitJob(session.session_id, kind, session.revision), `${kind === "preview_forward" ? "Forward preview" : kind} job submitted`);
  }
  function captureJobOpener(event: React.MouseEvent<HTMLDivElement>) {
    capturedJobOpener.current = event.target instanceof HTMLButtonElement
      ? event.target
      : null;
  }
  function requestJob(kind: JobKind) {
    const opener = capturedJobOpener.current;
    capturedJobOpener.current = null;
    if (
      jobBlocked
      || busyRef.current
      || drawerOpen
      || pendingJob !== null
      || opener === null
      || !opener.isConnected
      || opener.disabled
    ) return;
    if (securityAcknowledged) {
      submitJob(kind);
      return;
    }
    jobOpener.current = opener;
    setPendingJob(kind);
  }
  function cancelPendingJob() {
    restoreJobFocus.current = true;
    setPendingJob(null);
  }
  function confirmPendingJob() {
    if (pendingJob === null || jobBlocked || busyRef.current) return;
    const kind = pendingJob;
    restoreJobFocus.current = true;
    setSecurityAcknowledged(true);
    setPendingJob(null);
    submitJob(kind);
  }
  async function refreshJobs() {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      const next = await transport.refresh(session.session_id);
      setSession(next);
      setStatus("Job state refreshed");
      setStatusError(false);
    }
    catch (error) { setStatus(error instanceof Error ? error.message : String(error)); setStatusError(true); }
    finally { busyRef.current = false; setBusy(false); }
  }
  const reasonId = mutationReason ? "mutation-blocked-reason" : undefined;
  const surfaces: Record<WorkspaceId, WorkspaceSurface> = {
    model: { main: <GraphEditor session={session} transport={transport} onAccept={accept} disabled={busy || (draft.kind !== "none" && draft.kind !== "graph")} disabledReasonId={reasonId} coordinator={coordinator} onRun={(action, message) => void run(action, message)} />, inspector: emptyInspector("Select a graph node") },
    config: { main: <ConfigForms forms={session.document.forms} badges={session.document.validation.section_badges} />, inspector: emptyInspector("Select a field") },
    execute: { main: <><OutputWorkflow session={session} transport={transport} onAccept={accept} disabled={mutationBlocked} disabledReasonId={reasonId} onRun={(action, message) => void run(action, message)} /><div onClickCapture={captureJobOpener}><PreviewPanel previews={session.document.previews} jobs={session.jobs} disabled={mutationBlocked} blocked={session.document.validation.run_blocked} disabledReasonId={reasonId} onSubmit={requestJob} /></div></>, inspector: emptyInspector("Select an output") },
    results: { main: <ValidationLedger validation={session.document.validation} />, inspector: emptyInspector("Select a job") },
  };
  const surface = surfaces[activeWorkspace];
  const actionDescription = reasonId;

  return <WorkbenchShell
    header={<WorkbenchHeader dirty={session.dirty} validationStale={session.validation_stale} revision={session.revision} mutationBlocked={mutationBlocked} mutationReason={mutationReason} yamlBlocked={pendingJob !== null} onOpenYaml={() => { if (pendingJob === null) setDrawerOpen(true); }} actions={<>
      <button disabled={mutationBlocked || !session.can_undo} aria-describedby={actionDescription} onClick={() => void run(() => transport.undo(session.session_id, session.revision), "Undid YAML edit")}>Undo</button>
      <button disabled={mutationBlocked || !session.can_redo} aria-describedby={actionDescription} onClick={() => void run(() => transport.redo(session.session_id, session.revision), "Redid YAML edit")}>Redo</button>
      <label>Load YAML<input aria-label="Load YAML file" type="file" accept=".yaml,.yml,application/yaml,text/yaml,text/plain" disabled={mutationBlocked} aria-describedby={actionDescription} onChange={load} /></label>
      <button disabled={mutationBlocked} aria-describedby={actionDescription} onClick={save}>Save YAML</button>
      <button disabled={busy} aria-describedby={actionDescription} onClick={() => void refreshJobs()}>Refresh jobs</button>
    </>} />}
    navigation={<WorkspaceNav active={activeWorkspace} onChange={setActiveWorkspace} />}
    main={<><OnboardingChecklist missingRequired={session.document.forms.missing_required} runBlocked={session.document.validation.run_blocked} jobs={session.jobs} /><section id={`workspace-panel-${activeWorkspace}`} role="tabpanel" aria-label={workspaceLabels[activeWorkspace]} aria-labelledby={`workspace-tab-${activeWorkspace}`}>{surface.main}</section></>}
    inspector={surface.inspector}
    jobs={<p role={statusError ? "alert" : "status"} className={statusError ? "error-surface" : undefined}>{status}</p>}
    overlay={drawerOpen || pendingJob !== null ? <>
      {drawerOpen && <YamlDrawer acceptedYaml={session.document.yaml_text} revision={session.revision} draft={draft} diagnostic={yamlDiagnostic} conflict={yamlConflict} busy={busy} onChange={updateYamlDraft} onApply={() => void applyYamlDraft()} onDiscard={discardDraft} onClose={() => setDrawerOpen(false)} onRefresh={() => void refreshJobs()} />}
      {pendingJob !== null && <FirstJobConfirmation kind={pendingJob} blocked={jobBlocked} onConfirm={confirmPendingJob} onCancel={cancelPendingJob} />}
    </> : null}
  />;
}
