import type { EditorSnapshot, EditorTransport, NodeEdit } from "./types";

async function request(path: string, init: RequestInit): Promise<EditorSnapshot> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }
  return body as EditorSnapshot;
}

export const apiTransport: EditorTransport = {
  snapshot(yamlText) {
    return request("/api/snapshot", {
      method: "POST",
      body: JSON.stringify({ yaml_text: yamlText }),
    });
  },
  editNode(nodeId: string, edit: NodeEdit) {
    return request(`/api/nodes/${encodeURIComponent(nodeId)}`, {
      method: "PATCH",
      body: JSON.stringify(edit),
    });
  },
};
