import type { EditorSession, SessionTransport } from "./types";

export class RequestError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "RequestError";
  }
}

async function request(path: string, init: RequestInit): Promise<EditorSession> {
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
  return body as EditorSession;
}

function postRevision(path: string, expectedRevision: number) {
  return request(path, {
    method: "POST",
    body: JSON.stringify({ expected_revision: expectedRevision }),
  });
}

function nodePath(sessionId: string, nodeId: string, suffix = "") {
  return `/api/sessions/${encodeURIComponent(sessionId)}/nodes/${encodeURIComponent(nodeId)}${suffix}`;
}

export const sessionTransport: SessionTransport = {
  replaceYaml(sessionId, yamlText, expectedRevision) {
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/yaml`, {
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
    return request(`/api/sessions/${encodeURIComponent(sessionId)}/load`, {
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
    return request(nodePath(sessionId, nodeId), {
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
    return request(nodePath(sessionId, nodeId, "/move"), {
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
    return request(nodePath(sessionId, nodeId, "/compose"), {
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
    return request(nodePath(sessionId, nodeId, "/placement"), {
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
    return request(nodePath(sessionId, nodeId, "/snapshot-before"), {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        snapshot_name: snapshotName,
        variant,
      }),
    });
  },
};
