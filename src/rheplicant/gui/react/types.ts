export interface NodeCard {
  node_id: string;
  label: string;
  kind: "source" | "transform" | "junction" | "selector";
  description: string;
  explanation: string;
  editable: boolean;
  reserved: boolean;
  many: boolean;
  segment: "forward" | "processing";
  lit: boolean;
  count: number;
  configuration:
    | "single"
    | "sum"
    | "fan"
    | "chain"
    | "compose"
    | "region"
    | "reserved"
    | "junction"
    | "selector";
  settings: unknown;
  instances: NodeInstance[];
  stage_names: string[];
}

export interface NodeInstance {
  instance_id: string;
  label: string;
  settings: unknown;
}

export interface GraphCounts {
  lit: number;
  skipped: number;
  reserved: number;
  instances: number;
  materialized: number;
}

export interface GraphDiagram {
  name: string;
  svg: string;
  nodes: NodeCard[];
  walk_order: string[];
  counts: GraphCounts;
  changed_nodes: string[];
}

export interface EditorSnapshot {
  yaml_text: string;
  svg: string;
  nodes: NodeCard[];
  walk_order: string[];
  forms: FormProjection;
  base_diagram: GraphDiagram;
  backend_diagram: GraphDiagram;
  variant_diagrams: GraphDiagram[];
}

export interface ProjectedWidget {
  path: string;
  path_pattern: string;
  label: string;
  widget: string;
  choices: string[];
  visible: boolean;
  present: boolean;
  must_decide: boolean;
  value: unknown;
  dimension: string | null;
  unit_policy: string | null;
  delivery: string | null;
  disabled: boolean;
  reason: string | null;
}

export interface ProjectedSection {
  section_id: string;
  label: string;
  disabled: boolean;
  reason: string | null;
  widgets: ProjectedWidget[];
}

export interface FormProjection {
  sections: ProjectedSection[];
  missing_required: string[];
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
  editNode(
    sessionId: string,
    nodeId: string,
    enabled: boolean,
    settings: unknown,
    expectedRevision: number,
    variant: string | null,
  ): Promise<EditorSession>;
  moveNodeInstance(
    sessionId: string,
    nodeId: string,
    fromIndex: number,
    toIndex: number,
    expectedRevision: number,
    variant: string | null,
  ): Promise<EditorSession>;
  composeNode(
    sessionId: string,
    nodeId: string,
    compose: "cascade" | "sum",
    stages: Record<string, unknown>[],
    expectedRevision: number,
    variant: string | null,
  ): Promise<EditorSession>;
  placeNode(
    sessionId: string,
    nodeId: string,
    at: string | string[],
    settings: Record<string, unknown>,
    expectedRevision: number,
    variant: string | null,
  ): Promise<EditorSession>;
  setSnapshotBefore(
    sessionId: string,
    nodeId: string,
    snapshotName: string,
    expectedRevision: number,
    variant: string | null,
  ): Promise<EditorSession>;
}
