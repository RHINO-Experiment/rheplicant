export interface NodeCard {
  node_id: string;
  label: string;
  kind: "source" | "transform" | "junction" | "selector";
  description: string;
  editable: boolean;
  reserved: boolean;
  many: boolean;
  lit: boolean;
}

export interface EditorSnapshot {
  yaml_text: string;
  svg: string;
  nodes: NodeCard[];
  walk_order: string[];
}

export interface EditorSession {
  session_id: string;
  revision: number;
  dirty: boolean;
  validation_stale: boolean;
  can_undo: boolean;
  can_redo: boolean;
  document: EditorSnapshot;
}

export interface SessionTransport {
  replaceYaml(
    sessionId: string,
    yamlText: string,
    expectedRevision: number,
  ): Promise<EditorSession>;
  undo(sessionId: string, expectedRevision: number): Promise<EditorSession>;
  redo(sessionId: string, expectedRevision: number): Promise<EditorSession>;
  load(
    sessionId: string,
    yamlText: string,
    expectedRevision: number,
  ): Promise<EditorSession>;
  save(sessionId: string, expectedRevision: number): Promise<EditorSession>;
}
