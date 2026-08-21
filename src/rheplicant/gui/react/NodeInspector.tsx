import { useEffect, useState } from "react";

import type { DraftCoordinator } from "./drafts";
import { TypedFieldset } from "./TypedFieldset";
import type {
  EditorSession,
  NodeCard,
  NodeField,
  SessionTransport,
  TypedFields,
} from "./types";

export interface NodeInspectorProps {
  session: EditorSession;
  transport: SessionTransport;
  drafts?: DraftCoordinator;
  selected: NodeCard | undefined;
  activeVariant: string | null;
  disabled: boolean;
  disabledReason: string | null;
  onAccept: (next: EditorSession, message: string) => void;
  onRun?: (action: () => Promise<EditorSession>, message: string) => void;
  onStatus: (message: string, error: boolean) => void;
}

function jsonText(value: unknown, fallback: unknown) {
  return JSON.stringify(value ?? fallback, null, 2);
}

function parseObject(text: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(text);
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("Settings JSON must be an object.");
  }
  return parsed as Record<string, unknown>;
}

/** What one typed control writes back, in the shape the field already had.
 *
 *  A field written as the `<number> <unit>` shorthand stays shorthand and one
 *  written as the `{value, unit}` envelope stays an envelope: normalising
 *  between them would rewrite a line the user did not touch. The number is
 *  carried through verbatim -- changing a unit NEVER scales it, because
 *  `celsius` is affine and a silent conversion is a finite, correctly-shaped
 *  wrong answer. */
function quantityValue(field: NodeField, numberText: string, unit: string) {
  const parsed = Number(numberText);
  if (!unit) return parsed;
  if (field.form === "shorthand") return `${numberText} ${unit}`;
  return { value: parsed, unit };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** One operator's settings inside the draft: the whole object for a
 *  single-slot node, one list entry or one FAN label for a `many` one.
 *
 *  Which of the two a key means is read off the DRAFT rather than off the
 *  graph -- an array indexes, a mapping keys -- so a FAN label that happens to
 *  look like a number still finds its own entry. */
function readSlot(source: string, slot: string | null): Record<string, unknown> | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return null;
  }
  if (slot === null) return isPlainObject(parsed) ? parsed : null;
  if (Array.isArray(parsed)) {
    const entry: unknown = parsed[Number(slot)];
    return isPlainObject(entry) ? entry : null;
  }
  if (isPlainObject(parsed)) {
    const entry = parsed[slot];
    return isPlainObject(entry) ? entry : null;
  }
  return null;
}

function replaceSlot(
  source: string,
  slot: string | null,
  next: Record<string, unknown>,
): string | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return null;
  }
  if (slot === null) return JSON.stringify(next, null, 2);
  if (Array.isArray(parsed)) {
    const index = Number(slot);
    return JSON.stringify(
      parsed.map((entry: unknown, at: number) => (at === index ? next : entry)),
      null,
      2,
    );
  }
  if (isPlainObject(parsed)) return JSON.stringify({ ...parsed, [slot]: next }, null, 2);
  return null;
}

function parseSettings(text: string): unknown {
  const parsed: unknown = JSON.parse(text);
  if (parsed === null || typeof parsed !== "object") {
    throw new Error("Node settings JSON must be an object or list.");
  }
  return parsed;
}

export function NodeInspector({
  session,
  transport,
  drafts,
  selected,
  activeVariant,
  disabled,
  disabledReason,
  onAccept,
  onRun,
  onStatus,
}: NodeInspectorProps) {
  const [settingsText, setSettingsText] = useState("{}");
  const [stagesText, setStagesText] = useState("[]");
  const [regionText, setRegionText] = useState("");
  const [snapshotName, setSnapshotName] = useState("raw");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // What the typed controls currently SHOW, seeded from the server's own
  // decomposition of each field and kept here until the draft is applied or
  // discarded. Deliberately not re-read from the JSON draft: reading it back
  // would need a second value-form parser in the browser, and a second parser
  // is a second authority on what a document says.
  // Keyed by slot and field name together, so two instances of one node
  // editing fields of the same name do not share a control's value.
  const [typedEdits, setTypedEdits] = useState<Record<string, { number?: string; unit?: string }>>({});
  // A class the user has picked but not yet confirmed. Held here rather than
  // written straight through, because the two classes at a node share no
  // fields: changing one destroys every value written for the other, and a
  // control that does that silently is a control that loses work.
  const [pendingTypes, setPendingTypes] = useState<Record<string, string>>({});

  const selectedAt = selected?.settings && typeof selected.settings === "object"
    ? (selected.settings as { at?: unknown }).at
    : undefined;
  const selectedSettings = selected
    ? jsonText(selected.settings, selected.configuration === "fan" ? {} : selected.many ? [] : {})
    : "{}";
  const selectedStages = selected?.settings && typeof selected.settings === "object"
    ? jsonText((selected.settings as { stages?: unknown }).stages, [])
    : "[]";
  const selectedRegion = Array.isArray(selectedAt)
    ? selectedAt.join(", ")
    : typeof selectedAt === "string"
      ? selectedAt
      : "";
  const selectedSnapshot = selected?.settings && typeof selected.settings === "object"
    && typeof (selected.settings as { snapshot_before?: unknown }).snapshot_before === "string"
    ? (selected.settings as { snapshot_before: string }).snapshot_before
    : "";

  useEffect(() => {
    if (drafts || !selected) return;
    const fallback = selected.configuration === "fan" ? {} : selected.many ? [] : {};
    setSettingsText(jsonText(selected.settings, fallback));
    const stages = selected.settings && typeof selected.settings === "object"
      ? (selected.settings as { stages?: unknown }).stages
      : undefined;
    setStagesText(jsonText(stages, [{ name: "stage_1" }, { name: "stage_2" }]));
    setRegionText(selectedRegion);
    setSnapshotName(selectedSnapshot || "raw");
  }, [activeVariant, drafts, selected, selectedRegion, selectedSnapshot]);

  useEffect(() => {
    setAdvancedOpen(false);
  }, [selected?.node_id]);

  useEffect(() => {
    setTypedEdits({});
    setPendingTypes({});
  }, [selected?.node_id, activeVariant, session.revision]);

  if (!selected) {
    return <aside aria-label="Selected graph node"><p>Select a graph node</p></aside>;
  }
  const node = selected;

  function path(suffix: string) {
    return `${activeVariant ?? "base"}:${node.node_id}:${suffix}`;
  }

  function rawGraphValue(draftPath: string, fallback: string) {
    const draft = drafts?.draft;
    return draft?.kind === "graph" && draft.path === draftPath ? draft.rawValue : fallback;
  }

  function ownsGraphDraft(draftPath: string) {
    return drafts?.draft.kind === "graph" && drafts.draft.path === draftPath;
  }

  function graphControlDisabled(draftPath: string) {
    return disabled || (drafts !== undefined && drafts.draft.kind !== "none" && !ownsGraphDraft(draftPath));
  }

  function graphRevision(draftPath: string) {
    const current = drafts?.draft;
    return current?.kind === "graph" && current.path === draftPath
      ? current.baseRevision
      : session.revision;
  }

  function placementValues(draftPath: string) {
    const draft = drafts?.draft;
    const fallback = { settings: selectedSettings, region: selectedRegion };
    if (draft?.kind !== "graph" || draft.path !== draftPath) return fallback;
    try {
      const parsed = JSON.parse(draft.rawValue) as { settings?: unknown; region?: unknown };
      return {
        settings: typeof parsed.settings === "string" ? parsed.settings : fallback.settings,
        region: typeof parsed.region === "string" ? parsed.region : fallback.region,
      };
    } catch {
      return fallback;
    }
  }

  function updatePlacement(draftPath: string, key: "settings" | "region", value: string) {
    if (!drafts) {
      if (key === "settings") setSettingsText(value);
      else setRegionText(value);
      return;
    }
    const values = placementValues(draftPath);
    const current = drafts.draft;
    const next = {
      kind: "graph" as const,
      path: draftPath,
      rawValue: JSON.stringify({ ...values, [key]: value }),
      baseRevision: current.kind === "graph" && current.path === draftPath
        ? current.baseRevision
        : session.revision,
    };
    if (current.kind === "none") drafts.begin(next);
    else if (current.kind === "graph" && current.path === draftPath) drafts.update(next);
  }

  function updateRawGraph(draftPath: string, rawValue: string, setLocal: (value: string) => void) {
    if (!drafts) {
      setLocal(rawValue);
      return;
    }
    const current = drafts.draft;
    const next = {
      kind: "graph" as const,
      path: draftPath,
      rawValue,
      baseRevision: current.kind === "graph" && current.path === draftPath
        ? current.baseRevision
        : session.revision,
    };
    if (current.kind === "none") drafts.begin(next);
    else if (current.kind === "graph" && current.path === draftPath) drafts.update(next);
  }

  async function run(action: () => Promise<EditorSession>, message: string) {
    if (onRun) {
      onRun(action, message);
      return;
    }
    try {
      const next = await action();
      onAccept(next, message);
      onStatus(message, false);
    } catch (error) {
      onStatus(error instanceof Error ? error.message : String(error), true);
    }
  }

  function applyNode() {
    const draftPath = path("settings");
    if (graphControlDisabled(draftPath)) return;
    try {
      const settings = parseSettings(rawGraphValue(
        draftPath,
        drafts ? selectedSettings : settingsText,
      ));
      void run(
        () => transport.editNode(
          session.session_id,
          node.node_id,
          true,
          settings,
          graphRevision(draftPath),
          activeVariant,
        ),
        `Configured ${node.node_id}`,
      );
    } catch (error) {
      onStatus(error instanceof Error ? error.message : String(error), true);
    }
  }

  function disableNode() {
    if (graphControlDisabled("node-action")) return;
    void run(
      () => transport.editNode(
        session.session_id,
        node.node_id,
        false,
        null,
        session.revision,
        activeVariant,
      ),
      `Disabled ${node.node_id}`,
    );
  }

  function moveInstance(index: number) {
    if (graphControlDisabled("node-action")) return;
    void run(
      () => transport.moveNodeInstance(
        session.session_id,
        node.node_id,
        index,
        index - 1,
        session.revision,
        activeVariant,
      ),
      `Moved ${node.instances[index].label}`,
    );
  }

  function applyComposition() {
    const draftPath = path("stages");
    if (graphControlDisabled(draftPath)) return;
    try {
      const stages = JSON.parse(rawGraphValue(
        draftPath,
        drafts ? selectedStages : stagesText,
      )) as unknown;
      if (!Array.isArray(stages) || stages.some(
        (stage) => stage === null || Array.isArray(stage) || typeof stage !== "object",
      )) {
        throw new Error("Composition stages JSON must be a list of objects.");
      }
      const compose = node.kind === "source" ? "sum" : "cascade";
      void run(
        () => transport.composeNode(
          session.session_id,
          node.node_id,
          compose,
          stages as Record<string, unknown>[],
          graphRevision(draftPath),
          activeVariant,
        ),
        `Composed ${node.node_id}`,
      );
    } catch (error) {
      onStatus(error instanceof Error ? error.message : String(error), true);
    }
  }

  function applyPlacement() {
    const draftPath = path("placement");
    if (graphControlDisabled(draftPath)) return;
    try {
      const values = drafts
        ? placementValues(draftPath)
        : { settings: settingsText, region: regionText };
      const settings = parseObject(values.settings);
      const nodes = values.region.split(",").map((item) => item.trim()).filter(Boolean);
      if (nodes.length === 0) throw new Error("A placement needs at least one node.");
      void run(
        () => transport.placeNode(
          session.session_id,
          node.node_id,
          nodes.length === 1 ? nodes[0] : nodes,
          settings,
          graphRevision(draftPath),
          activeVariant,
        ),
        `Placed ${node.node_id}`,
      );
    } catch (error) {
      onStatus(error instanceof Error ? error.message : String(error), true);
    }
  }

  const settingsPath = path("settings");
  const settingsSource = rawGraphValue(settingsPath, drafts ? selectedSettings : settingsText);
  const draftError = (() => {
    try {
      JSON.parse(settingsSource);
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  })();

  function editKey(slot: string | null, name: string) {
    return `${slot ?? ""}\u0000${name}`;
  }

  function shownNumber(slot: string | null, field: NodeField) {
    const edited = typedEdits[editKey(slot, field.name)]?.number;
    if (edited !== undefined) return edited;
    if (field.control === "select" || field.control === "text") {
      return typeof field.written === "string" ? field.written : "";
    }
    return field.number === null ? "" : String(field.number);
  }

  function shownUnit(slot: string | null, field: NodeField) {
    const edited = typedEdits[editKey(slot, field.name)]?.unit;
    if (edited !== undefined) return edited;
    return field.unit ?? (field.units[0] ?? "");
  }

  function remember(slot: string | null, field: NodeField, patch: { number?: string; unit?: string }) {
    setTypedEdits((current) => ({
      ...current,
      [editKey(slot, field.name)]: { ...current[editKey(slot, field.name)], ...patch },
    }));
  }

  function writeTypedField(slot: string | null, name: string, value: unknown) {
    const current = readSlot(settingsSource, slot);
    if (current === null) return;
    const next = { ...current };
    if (value === undefined) delete next[name];
    else next[name] = value;
    const text = replaceSlot(settingsSource, slot, next);
    if (text === null) return;
    updateRawGraph(settingsPath, text, setSettingsText);
  }

  function editNumber(slot: string | null, field: NodeField, text: string) {
    remember(slot, field, { number: text });
    if (text === "") {
      writeTypedField(slot, field.name, undefined);
      return;
    }
    if (!Number.isFinite(Number(text))) return;
    const value = field.control === "quantity"
      ? quantityValue(field, text, shownUnit(slot, field))
      : Number(text);
    writeTypedField(slot, field.name, value);
  }

  function editUnit(slot: string | null, field: NodeField, unit: string) {
    remember(slot, field, { unit });
    const number = shownNumber(slot, field);
    // Nothing to hang a unit on yet. A lone `{unit: K}` is not a value node,
    // so the choice is remembered and written with the first number.
    if (number === "" || !Number.isFinite(Number(number))) return;
    writeTypedField(slot, field.name, quantityValue(field, number, unit));
  }

  function editText(slot: string | null, field: NodeField, value: string) {
    remember(slot, field, { number: value });
    writeTypedField(slot, field.name, value === "" ? undefined : value);
  }

  function writeType(slot: string | null, next: string, removed: string[]) {
    const current = readSlot(settingsSource, slot);
    if (current === null) return;
    // Rebuilt by iteration rather than by delete-and-reassign so `type:` keeps
    // the line it was written on instead of moving to the end of the mapping.
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(current)) {
      if (key === "type") out[key] = next;
      else if (!removed.includes(key)) out[key] = value;
    }
    if (!("type" in out)) out.type = next;
    const text = replaceSlot(settingsSource, slot, out);
    if (text === null) return;
    setTypedEdits({});
    updateRawGraph(settingsPath, text, setSettingsText);
  }

  function chooseType(slot: string | null, owner: TypedFields, next: string) {
    const key = slot ?? "";
    if (next === "" || next === owner.selected_type) {
      setPendingTypes(({ [key]: _dropped, ...rest }) => rest);
      return;
    }
    const removed = owner.removed_by_type[next] ?? [];
    if (removed.length === 0) {
      setPendingTypes(({ [key]: _dropped, ...rest }) => rest);
      writeType(slot, next, removed);
      return;
    }
    setPendingTypes((current) => ({ ...current, [key]: next }));
  }

  function fieldsetProps(owner: TypedFields, subject: string, slot: string | null) {
    const key = slot ?? "";
    const pending = pendingTypes[key] ?? null;
    return {
      owner,
      subject,
      disabled: draftError !== null
        || readSlot(settingsSource, slot) === null
        || graphControlDisabled(settingsPath),
      shownNumber: (field: NodeField) => shownNumber(slot, field),
      shownUnit: (field: NodeField) => shownUnit(slot, field),
      onNumber: (field: NodeField, text: string) => editNumber(slot, field, text),
      onUnit: (field: NodeField, unit: string) => editUnit(slot, field, unit),
      onText: (field: NodeField, value: string) => editText(slot, field, value),
      pendingType: pending,
      onChooseType: (next: string) => chooseType(slot, owner, next),
      onConfirmType: () => {
        if (pending === null) return;
        writeType(slot, pending, owner.removed_by_type[pending] ?? []);
        setPendingTypes(({ [key]: _dropped, ...rest }) => rest);
      },
      onCancelType: () => setPendingTypes(({ [key]: _dropped, ...rest }) => rest),
    };
  }

  const stagesPath = path("stages");
  const placementPath = path("placement");
  const snapshotPath = path("snapshot");

  return (
    <aside aria-label={`${selected.node_id} settings`}>
      <h3>{selected.label}</h3>
      <p>{selected.explanation}</p>
      {selected.reserved && <strong>reserved — python: only</strong>}
      {selected.many && <strong>{selected.configuration.toUpperCase()} · {selected.count} instances</strong>}
      {selected.stage_names.length > 0 && (
        <ol aria-label={`${selected.node_id} composed stages`}>
          {selected.stage_names.map((name) => <li key={name}>{name}</li>)}
        </ol>
      )}
      {Array.isArray(selectedAt) && selectedAt.length > 1 && (
        <p>Region: {selectedAt.join(" → ")} · addressed as {selectedAt[selectedAt.length - 1]}</p>
      )}

      {selected.editable && (
        <>
          {draftError !== null && (
            <p>Typed fields need valid JSON: {draftError}</p>
          )}
          <TypedFieldset {...fieldsetProps(selected, selected.node_id, null)} />
          {selected.instances.map((instance, index) => (
            <TypedFieldset
              key={instance.instance_id}
              {...fieldsetProps(
                instance,
                instance.label,
                // The FAN keys by label and the list by position, and the
                // draft itself decides which of the two this is.
                selected.configuration === "fan" ? instance.instance_id : String(index),
              )}
            />
          ))}
          <label>
            Node settings JSON
            <textarea
              aria-label="Node settings JSON"
              value={rawGraphValue(settingsPath, drafts ? selectedSettings : settingsText)}
              disabled={graphControlDisabled(settingsPath)}
              aria-describedby={graphControlDisabled(settingsPath) ? disabledReason ?? undefined : undefined}
              onChange={(event) => updateRawGraph(settingsPath, event.target.value, setSettingsText)}
            />
          </label>
          <button
            type="button"
            disabled={graphControlDisabled(settingsPath)}
            aria-describedby={graphControlDisabled(settingsPath) ? disabledReason ?? undefined : undefined}
            onClick={applyNode}
          >
            {selected.lit
              ? `Apply configuration to ${selected.node_id}`
              : `Light and configure ${selected.node_id}`}
          </button>
          {selected.lit && (
            <button
              type="button"
              disabled={graphControlDisabled("node-action")}
              aria-describedby={graphControlDisabled("node-action") ? disabledReason ?? undefined : undefined}
              onClick={disableNode}
            >
              Disable {selected.node_id}
            </button>
          )}
          {drafts?.draft.kind === "graph" && (
            <button
              type="button"
              disabled={disabled}
              aria-describedby={disabled ? disabledReason ?? undefined : undefined}
              onClick={() => drafts.clear()}
            >
              Discard graph draft
            </button>
          )}

          <details open={advancedOpen}>
            <summary
              onClick={(event) => {
                event.preventDefault();
                setAdvancedOpen((open) => !open);
              }}
            >
              Advanced node controls
            </summary>
            {advancedOpen && (
              <>
                {selected.instances.length > 0 && (
                  <ul aria-label={`${selected.node_id} instances`}>
                    {selected.instances.map((instance, index) => (
                      <li key={instance.instance_id}>
                        <span>{instance.label}</span>
                        {selected.configuration === "chain" && index > 0 && (
                          <button
                            type="button"
                            disabled={graphControlDisabled("node-action")}
                            aria-describedby={graphControlDisabled("node-action") ? disabledReason ?? undefined : undefined}
                            aria-label={`Move ${instance.label} up`}
                            onClick={() => moveInstance(index)}
                          >
                            Move up
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}

                {!selected.many && !selected.reserved && (
                  <fieldset>
                    <legend>Compose stages</legend>
                    <textarea
                      aria-label="Composition stages JSON"
                      value={rawGraphValue(stagesPath, drafts ? selectedStages : stagesText)}
                      disabled={graphControlDisabled(stagesPath)}
                      aria-describedby={graphControlDisabled(stagesPath) ? disabledReason ?? undefined : undefined}
                      onChange={(event) => updateRawGraph(stagesPath, event.target.value, setStagesText)}
                    />
                    <button
                      type="button"
                      disabled={graphControlDisabled(stagesPath)}
                      aria-describedby={graphControlDisabled(stagesPath) ? disabledReason ?? undefined : undefined}
                      onClick={applyComposition}
                    >
                      Apply {selected.kind === "source" ? "sum" : "cascade"}
                    </button>
                  </fieldset>
                )}

                <fieldset>
                  <legend>Place python operator with at:</legend>
                  <label>
                    Placement settings JSON
                    <textarea
                      aria-label="Placement settings JSON"
                      value={drafts ? placementValues(placementPath).settings : settingsText}
                      disabled={graphControlDisabled(placementPath)}
                      aria-describedby={graphControlDisabled(placementPath) ? disabledReason ?? undefined : undefined}
                      onChange={(event) => updatePlacement(placementPath, "settings", event.target.value)}
                    />
                  </label>
                  <label>
                    Covered nodes in signal order
                    <input
                      value={drafts ? placementValues(placementPath).region : regionText}
                      disabled={graphControlDisabled(placementPath)}
                      aria-describedby={graphControlDisabled(placementPath) ? disabledReason ?? undefined : undefined}
                      placeholder="noise, emi"
                      onChange={(event) => updatePlacement(placementPath, "region", event.target.value)}
                    />
                  </label>
                  <button
                    type="button"
                    disabled={graphControlDisabled(placementPath)}
                    aria-describedby={graphControlDisabled(placementPath) ? disabledReason ?? undefined : undefined}
                    onClick={applyPlacement}
                  >
                    Apply placement
                  </button>
                </fieldset>

                {selected.segment === "processing" && !selected.many && (
                  <label>
                    Snapshot name
                    <input
                      aria-label="Snapshot name"
                      value={rawGraphValue(snapshotPath, drafts ? selectedSnapshot : snapshotName)}
                      disabled={graphControlDisabled(snapshotPath)}
                      aria-describedby={graphControlDisabled(snapshotPath) ? disabledReason ?? undefined : undefined}
                      onChange={(event) => updateRawGraph(snapshotPath, event.target.value, setSnapshotName)}
                    />
                    <button
                      type="button"
                      disabled={graphControlDisabled(snapshotPath)
                        || !rawGraphValue(snapshotPath, drafts ? selectedSnapshot : snapshotName)}
                      aria-describedby={graphControlDisabled(snapshotPath) ? disabledReason ?? undefined : undefined}
                      onClick={() => !graphControlDisabled(snapshotPath) && void run(
                        () => transport.setSnapshotBefore(
                          session.session_id,
                          selected.node_id,
                          rawGraphValue(snapshotPath, drafts ? selectedSnapshot : snapshotName),
                          graphRevision(snapshotPath),
                          activeVariant,
                        ),
                        `Snapshot before ${selected.node_id}`,
                      )}
                    >
                      Keep raw data before {selected.node_id}
                    </button>
                  </label>
                )}
              </>
            )}
          </details>
        </>
      )}
    </aside>
  );
}
