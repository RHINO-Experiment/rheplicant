import type { EditorSession, SessionTransport } from "./types";

async function request(path: string, init: RequestInit): Promise<EditorSession> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }
  return body as EditorSession;
}

function postRevision(path: string, expectedRevision: number) {
  return request(path, {
    method: "POST",
    body: JSON.stringify({ expected_revision: expectedRevision }),
  });
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
};
