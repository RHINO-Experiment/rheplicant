import { useEffect, useState } from "react";

import type { DraftCoordinator } from "./drafts";
import type { EditorSession, NodeCard, SessionTransport } from "./types";

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
