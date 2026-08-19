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

export interface NodeEdit {
  yaml_text: string;
  enabled: boolean;
  settings?: Record<string, unknown> | null;
}

export interface EditorTransport {
  snapshot(yamlText: string): Promise<EditorSnapshot>;
  editNode(nodeId: string, edit: NodeEdit): Promise<EditorSnapshot>;
}
