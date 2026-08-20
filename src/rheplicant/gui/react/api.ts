import type { EditorSession, JobKind, SessionTransport } from "./types";

export class RequestError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "RequestError";
  }
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new RequestError(
      response.status,
      body.detail ?? `Request failed with status ${response.status}`,
    );
  }
  return body as T;
}

export function createSession(yamlText: string) {
  return requestJson<EditorSession>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ yaml_text: yamlText }),
  });
}

export function getStarter(): Promise<{ yaml_text: string }> {
  return requestJson("/api/starter", { method: "GET" });
}

export async function createStarterSession(): Promise<EditorSession> {
  const { yaml_text } = await getStarter();
  return createSession(yaml_text);
}

function postRevision(path: string, expectedRevision: number) {
  return requestJson<EditorSession>(path, {
    method: "POST",
    body: JSON.stringify({ expected_revision: expectedRevision }),
  });
}

function nodePath(sessionId: string, nodeId: string, suffix = "") {
  return `/api/sessions/${encodeURIComponent(sessionId)}/nodes/${encodeURIComponent(nodeId)}${suffix}`;
}

export const sessionTransport: SessionTransport = {
  refresh(sessionId) {
    return requestJson<EditorSession>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: "GET",
    });
  },
  replaceYaml(sessionId, yamlText, expectedRevision) {
    return requestJson<EditorSession>(`/api/sessions/${encodeURIComponent(sessionId)}/yaml`, {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        yaml_text: yamlText,
      }),
    });
  },
  undo(sessionId, expectedRevision) {
    return postRevision(
      `/api/sessions/${encodeURIComponent(sessionId)}/undo`,
      expectedRevision,
    );
  },
  redo(sessionId, expectedRevision) {
    return postRevision(
      `/api/sessions/${encodeURIComponent(sessionId)}/redo`,
      expectedRevision,
    );
  },
  load(sessionId, yamlText, expectedRevision) {
    return requestJson<EditorSession>(`/api/sessions/${encodeURIComponent(sessionId)}/load`, {
      method: "POST",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        yaml_text: yamlText,
      }),
    });
  },
  save(sessionId, expectedRevision) {
    return postRevision(
      `/api/sessions/${encodeURIComponent(sessionId)}/save`,
      expectedRevision,
    );
  },
  editNode(sessionId, nodeId, enabled, settings, expectedRevision, variant) {
    return requestJson<EditorSession>(nodePath(sessionId, nodeId), {
      method: "PATCH",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        enabled,
        settings,
        variant,
      }),
    });
  },
  moveNodeInstance(
    sessionId,
    nodeId,
    fromIndex,
    toIndex,
    expectedRevision,
    variant,
  ) {
    return requestJson<EditorSession>(nodePath(sessionId, nodeId, "/move"), {
      method: "POST",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        from_index: fromIndex,
        to_index: toIndex,
        variant,
      }),
    });
  },
  composeNode(sessionId, nodeId, compose, stages, expectedRevision, variant) {
    return requestJson<EditorSession>(nodePath(sessionId, nodeId, "/compose"), {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        compose,
        stages,
        variant,
      }),
    });
  },
  placeNode(sessionId, nodeId, at, settings, expectedRevision, variant) {
    return requestJson<EditorSession>(nodePath(sessionId, nodeId, "/placement"), {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        at,
        settings,
        variant,
      }),
    });
  },
  setSnapshotBefore(
    sessionId,
    nodeId,
    snapshotName,
    expectedRevision,
    variant,
  ) {
    return requestJson<EditorSession>(nodePath(sessionId, nodeId, "/snapshot-before"), {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        snapshot_name: snapshotName,
        variant,
      }),
    });
  },
  setOutputProduct(
    sessionId,
    name,
    enabled,
    format,
    runs,
    keys,
    themes,
    expectedRevision,
  ) {
    return requestJson<EditorSession>(
      `/api/sessions/${encodeURIComponent(sessionId)}/outputs/products/${encodeURIComponent(name)}`,
      {
        method: "PUT",
        body: JSON.stringify({
          expected_revision: expectedRevision,
          enabled,
          format,
          runs,
          keys,
          themes,
        }),
      },
    );
  },
  setOutputReport(
    sessionId,
    enabled,
    rows,
    columns,
    reference,
    relative,
    formats,
    expectedRevision,
  ) {
    return requestJson<EditorSession>(`/api/sessions/${encodeURIComponent(sessionId)}/outputs/report`, {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        enabled,
        rows,
        columns,
        reference,
        relative,
        formats,
      }),
    });
  },
  submitJob(sessionId, kind: JobKind, expectedRevision) {
    return requestJson<EditorSession>(`/api/sessions/${encodeURIComponent(sessionId)}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        kind,
      }),
    });
  },
};
