export type DraftKind = "yaml" | "field" | "graph";

export type DraftEnvelope =
  | { kind: "none" }
  | { kind: "yaml"; baseRevision: number; text: string }
  | { kind: "field" | "graph"; baseRevision: number; path: string; rawValue: string };

export interface DraftCoordinator {
  draft: DraftEnvelope;
  begin(next: Exclude<DraftEnvelope, { kind: "none" }>): boolean;
  update(next: Exclude<DraftEnvelope, { kind: "none" }>): void;
  clear(): void;
}

export const NO_DRAFT: DraftEnvelope = { kind: "none" };

export function clearDraft(): DraftEnvelope {
  return NO_DRAFT;
}

export function draftBlocksMutation(draft: DraftEnvelope): boolean {
  return draft.kind !== "none";
}

export function canStartDraft(draft: DraftEnvelope, _kind: DraftKind): boolean {
  return draft.kind === "none";
}

export function canUpdateDraft(
  current: DraftEnvelope,
  next: Exclude<DraftEnvelope, { kind: "none" }>,
): boolean {
  if (current.kind === "none" || current.baseRevision !== next.baseRevision) return false;
  return (current.kind === "yaml" && next.kind === "yaml")
    || ((current.kind === "field" || current.kind === "graph")
      && current.kind === next.kind
      && current.path === next.path);
}

export function draftLabel(draft: DraftEnvelope): string | null {
  if (draft.kind === "none") return null;
  return draft.kind === "yaml" ? "Unsaved YAML draft" : `Unsaved ${draft.kind}: ${draft.path}`;
}
