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
  previews: PreviewProjection;
  validation: ValidationProjection;
  base_diagram: GraphDiagram;
  backend_diagram: GraphDiagram;
  variant_diagrams: GraphDiagram[];
}

export interface PreviewClass {
  preview_id: "graph" | "axes_shapes" | "validate" | "forward";
  label: string;
  cadence: "continuous" | "explicit";
  priced: boolean;
  description: string;
}

export interface AxisPreview {
  axis: "time" | "freq";
  first: number[];
  last: number[];
  count: number;
  spacing: number | null;
  unit: string | null;
  precision_ratio: number | null;
  precision_ok: boolean | null;
}

export interface ShapePreview {
  symbol: string;
  value: number;
}

export interface ForwardCost {
  label: string;
  estimated_milliseconds: number | null;
  estimated_peak_megabytes: number | null;
  n_freq: number | null;
  nside: number | null;
  lmax: number | null;
  optimizations: string[];
}

export interface PreviewProjection {
  classes: PreviewClass[];
  axes: AxisPreview[];
  shapes: ShapePreview[];
  forward_cost: ForwardCost;
  declared_run_kinds: string[];
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

export interface LedgerFinding {
  check: string;
  severity: "refuse" | "warn" | "report";
  where: string;
  message: string;
  attribution: string;
}

export interface PresetChange {
  path: string;
  kind: "added" | "changed" | "removed";
  preset_value: unknown;
  document_value: unknown;
}

export interface SectionBadge {
  section_id: string;
  incomplete: number;
  refuse: number;
  warn: number;
  report: number;
  preset_changes: number;
}

export interface ValidationProjection {
  findings: LedgerFinding[];
  section_badges: SectionBadge[];
  selected_presets: string[];
  preset_changes: PresetChange[];
  run_blocked: boolean;
}

export interface EditorSession {
  session_id: string;
  revision: number;
  yaml_digest: string;
  dirty: boolean;
  validation_stale: boolean;
  can_undo: boolean;
  can_redo: boolean;
  outputs: OutputProjection;
  jobs: JobProjection[];
  document: EditorSnapshot;
}

export type OutputState =
  | "ready_new"
  | "blocked_existing"
  | "blocked_foreign"
  | "replace_owned"
  | "ambiguous_recovery"
  | "blocked_unsafe"
  | "unavailable";

export interface OutputProductProjection {
  name: string;
  enabled: boolean;
  format: string;
  formats: string[];
  runs: string[];
  keys: string[];
  themes: string[];
  expected_paths: string[];
}

export interface OutputReportProjection {
  enabled: boolean;
  rows: string[];
  columns: string[];
  reference: string | null;
  relative: string[];
  formats: string[];
  expected_paths: string[];
}

export interface OutputProjection {
  requested_yaml: string;
  resolved_yaml: string;
  resolution_note: string;
  target_path: string | null;
  state: OutputState;
  state_message: string;
  clobber: boolean;
  declared_runs: string[];
  products: OutputProductProjection[];
  report: OutputReportProjection;
  audit_paths: string[];
}

export type JobKind =
  | "validate"
  | "preview_forward"
  | "run"
  | "compare"
  | "benchmark";

export interface JobProjection {
  job_id: string;
  session_id: string;
  kind: JobKind;
  revision: number;
  yaml_digest: string;
  status: "queued" | "running" | "succeeded" | "refused" | "error";
  result: unknown;
  message: string | null;
  stale: boolean;
}

export interface JobPollProjection {
  session_id: string;
  revision: number;
  yaml_digest: string;
  jobs: JobProjection[];
}

export interface SessionTransport {
  refresh(sessionId: string): Promise<EditorSession>;
  refreshJobs(sessionId: string, signal: AbortSignal): Promise<JobPollProjection>;
  replaceYaml(
    sessionId: string,
    yamlText: string,
    expectedRevision: number,
  ): Promise<EditorSession>;
  setField(
    sessionId: string,
    path: string,
    value: unknown,
    remove: boolean,
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
  setOutputProduct(
    sessionId: string,
    name: string,
    enabled: boolean,
    format: string,
    runs: string[],
    keys: string[],
    themes: string[],
    expectedRevision: number,
  ): Promise<EditorSession>;
  setOutputReport(
    sessionId: string,
    enabled: boolean,
    rows: string[],
    columns: string[],
    reference: string | null,
    relative: string[],
    formats: string[],
    expectedRevision: number,
  ): Promise<EditorSession>;
  submitJob(
    sessionId: string,
    kind: JobKind,
    expectedRevision: number,
  ): Promise<EditorSession>;
}
