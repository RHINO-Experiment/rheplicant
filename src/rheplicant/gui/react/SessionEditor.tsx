import { useCallback, useEffect, useRef, useState } from "react";

import { useConfigWorkspace } from "./ConfigWorkspace";
import { DiagnosticsDrawer } from "./DiagnosticsDrawer";
import { useExecuteWorkspace } from "./ExecuteWorkspace";
import { canUpdateDraft, draftBlocksMutation, draftLabel, NO_DRAFT, type DraftCoordinator, type DraftEnvelope } from "./drafts";
import { FirstJobConfirmation } from "./FirstJobConfirmation";
import { JobsDrawer } from "./JobsDrawer";
import { useModelWorkspace, type WorkspaceSurface } from "./ModelWorkspace";
import { OnboardingChecklist } from "./OnboardingChecklist";
import { ValidationLedger } from "./ValidationLedger";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { WorkbenchShell } from "./WorkbenchShell";
import { WorkspaceNav } from "./WorkspaceNav";
import { YamlDrawer } from "./YamlDrawer";
import { RequestError } from "./api";
import { useJobPolling } from "./useJobPolling";
import type { EditorSession, JobKind, SessionTransport } from "./types";
import type { WorkspaceId } from "./WorkspaceNav";

type ReadFile = (file: File) => Promise<string>;
type SaveFile = (yamlText: string) => Promise<void> | void;
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
  const [displayJobs, setDisplayJobs] = useState(initial.jobs);
  const [draft, setDraft] = useState<DraftEnvelope>(NO_DRAFT);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [requestedConfigPath, setRequestedConfigPath] = useState<string | null>(null);
  const [yamlContext, setYamlContext] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready");
  const [statusError, setStatusError] = useState(false);
  const [yamlDiagnostic, setYamlDiagnostic] = useState<string | null>(null);
  const [yamlConflict, setYamlConflict] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const acceptedSessionRef = useRef(initial);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>("model");
  const [pendingJob, setPendingJob] = useState<JobKind | null>(null);
  const [securityAcknowledged, setSecurityAcknowledged] = useState(false);
  const jobOpener = useRef<HTMLElement | null>(null);
  const capturedJobOpener = useRef<HTMLButtonElement | null>(null);
  const restoreJobFocus = useRef(false);
  const refreshJobProjection = useCallback(
    (signal: AbortSignal) => transport.refreshJobs(session.session_id, signal),
    [transport, session.session_id],
  );
  const installPolledJobs = useCallback((jobs: EditorSession["jobs"]) => {
    if (acceptedSessionRef.current !== session) return;
    setDisplayJobs(jobs);
  }, [session]);
  const polling = useJobPolling({
    sessionId: session.session_id,
    revision: session.revision,
    yamlDigest: session.yaml_digest,
    jobs: displayJobs,
    refresh: refreshJobProjection,
    onJobs: installPolledJobs,
  });
  const displaySession = { ...session, jobs: displayJobs };

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
    acceptedSessionRef.current = next;
    setSession(next);
    setDisplayJobs(next.jobs);
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
      || diagnosticsOpen
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
  async function refreshAcceptedSession() {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      const next = await transport.refresh(session.session_id);
      acceptedSessionRef.current = next;
      setSession(next);
      setDisplayJobs(next.jobs);
      setStatus("Accepted YAML refreshed");
      setStatusError(false);
    }
    catch (error) { setStatus(error instanceof Error ? error.message : String(error)); setStatusError(true); }
    finally { busyRef.current = false; setBusy(false); }
  }
  const reasonId = mutationReason ? "mutation-blocked-reason" : undefined;
  const modelSurface = useModelWorkspace({
    session,
    transport,
    drafts: coordinator,
    disabled: busy || (draft.kind !== "none" && draft.kind !== "graph"),
    disabledReason: reasonId ?? null,
    onAccept: accept,
    onRun: (action, message) => void run(action, message),
  });
  const configSurface = useConfigWorkspace({
    session,
    transport,
    drafts: coordinator,
    disabled: busy || (draft.kind !== "none" && draft.kind !== "field"),
    disabledReason: reasonId ?? null,
    requestedPath: requestedConfigPath,
    onAccept: accept,
    onEditYaml(path) {
      if (pendingJob !== null) return;
      setYamlContext(path);
      setDiagnosticsOpen(false);
      setDrawerOpen(true);
    },
    onRun: (action, message) => void run(action, message),
  });
  const executeSurface = useExecuteWorkspace({
    session,
    jobs: displayJobs,
    transport,
    drafts: coordinator,
    disabledReason: reasonId ?? null,
    onAccept: accept,
    onSubmit: requestJob,
    onRun: (action, message) => void run(action, message),
  });
  useEffect(() => {
    if (requestedConfigPath !== null && activeWorkspace === "config" && !diagnosticsOpen) {
      setRequestedConfigPath(null);
    }
  }, [activeWorkspace, diagnosticsOpen, requestedConfigPath]);
  const surfaces: Record<WorkspaceId, WorkspaceSurface> = {
    model: modelSurface,
    config: configSurface,
    execute: { main: <div onClickCapture={captureJobOpener}>{executeSurface.main}</div>, inspector: executeSurface.inspector },
    results: { main: <ValidationLedger validation={session.document.validation} />, inspector: emptyInspector("Select a job") },
  };
  const surface = surfaces[activeWorkspace];
  const actionDescription = reasonId;

  return <WorkbenchShell
    header={<WorkbenchHeader dirty={session.dirty} validationStale={session.validation_stale} revision={session.revision} mutationBlocked={mutationBlocked} mutationReason={mutationReason} yamlBlocked={pendingJob !== null} onOpenYaml={() => { if (pendingJob === null) { setYamlContext(null); setDiagnosticsOpen(false); setDrawerOpen(true); } }} actions={<>
      <button type="button" disabled={pendingJob !== null} onClick={() => { setDrawerOpen(false); setDiagnosticsOpen(true); }}>Diagnostics</button>
      <button disabled={mutationBlocked || !session.can_undo} aria-describedby={actionDescription} onClick={() => void run(() => transport.undo(session.session_id, session.revision), "Undid YAML edit")}>Undo</button>
      <button disabled={mutationBlocked || !session.can_redo} aria-describedby={actionDescription} onClick={() => void run(() => transport.redo(session.session_id, session.revision), "Redid YAML edit")}>Redo</button>
      <label>Load YAML<input aria-label="Load YAML file" type="file" accept=".yaml,.yml,application/yaml,text/yaml,text/plain" disabled={mutationBlocked} aria-describedby={actionDescription} onChange={load} /></label>
      <button disabled={mutationBlocked} aria-describedby={actionDescription} onClick={save}>Save YAML</button>
    </>} />}
    navigation={<WorkspaceNav active={activeWorkspace} onChange={setActiveWorkspace} />}
    main={<><OnboardingChecklist missingRequired={session.document.forms.missing_required} runBlocked={session.document.validation.run_blocked} jobs={displayJobs} /><section id={`workspace-panel-${activeWorkspace}`} role="tabpanel" aria-label={workspaceLabels[activeWorkspace]} aria-labelledby={`workspace-tab-${activeWorkspace}`}>{surface.main}</section></>}
    inspector={surface.inspector}
    jobs={<><JobsDrawer jobs={displayJobs} {...polling} disabled={busy} disabledReasonId={actionDescription} /><p role={statusError ? "alert" : "status"} className={statusError ? "error-surface" : undefined}>{status}</p></>}
    overlay={drawerOpen || diagnosticsOpen || pendingJob !== null ? <>
      {drawerOpen && <>{yamlContext && <p>YAML context: <code>{yamlContext}</code></p>}<YamlDrawer acceptedYaml={session.document.yaml_text} revision={session.revision} draft={draft} diagnostic={yamlDiagnostic} conflict={yamlConflict} busy={busy} onChange={updateYamlDraft} onApply={() => void applyYamlDraft()} onDiscard={discardDraft} onClose={() => setDrawerOpen(false)} onRefresh={() => void refreshAcceptedSession()} /></>}
      {diagnosticsOpen && <DiagnosticsDrawer session={displaySession} onOpenConfigPath={(path) => { setRequestedConfigPath(path); setActiveWorkspace("config"); setDiagnosticsOpen(false); }} onOpenYamlPath={(path) => { setYamlContext(path); setDiagnosticsOpen(false); setDrawerOpen(true); }} onClose={() => setDiagnosticsOpen(false)} />}
      {pendingJob !== null && <FirstJobConfirmation kind={pendingJob} blocked={jobBlocked} onConfirm={confirmPendingJob} onCancel={cancelPendingJob} />}
    </> : null}
  />;
}
