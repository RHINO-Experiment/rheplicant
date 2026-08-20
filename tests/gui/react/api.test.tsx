import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../../../src/rheplicant/gui/react/main";
import {
  createSession,
  createStarterSession,
  getStarter,
} from "../../../src/rheplicant/gui/react/api";
import type { EditorSession } from "../../../src/rheplicant/gui/react/types";

function response(body: object, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function createdSession(yamlText: string): EditorSession {
  const diagram = {
    name: "base",
    svg: "<svg></svg>",
    nodes: [],
    walk_order: [],
    counts: { lit: 0, skipped: 0, reserved: 0, instances: 0, materialized: 0 },
    changed_nodes: [],
  };
  return {
    session_id: "created",
    revision: 0,
    yaml_digest: "server-digest",
    dirty: false,
    validation_stale: false,
    can_undo: false,
    can_redo: false,
    jobs: [],
    outputs: {
      requested_yaml: yamlText,
      resolved_yaml: yamlText,
      resolution_note: "Preset-merged preview.",
      target_path: "/results",
      state: "ready_new",
      state_message: "Ready.",
      clobber: false,
      declared_runs: ["forward"],
      products: [],
      report: {
        enabled: false,
        rows: [],
        columns: ["mean"],
        reference: null,
        relative: [],
        formats: ["text"],
        expected_paths: [],
      },
      audit_paths: [],
    },
    document: {
      yaml_text: yamlText,
      svg: "<svg></svg>",
      nodes: [],
      walk_order: [],
      forms: { sections: [], missing_required: [] },
      previews: {
        classes: [],
        axes: [],
        shapes: [],
        forward_cost: {
          label: "Cost unavailable",
          estimated_milliseconds: null,
          estimated_peak_megabytes: null,
          n_freq: null,
          nside: null,
          lmax: null,
          optimizations: [],
        },
        declared_run_kinds: ["forward"],
      },
      validation: {
        findings: [],
        section_badges: [],
        selected_presets: [],
        preset_changes: [],
        run_blocked: false,
      },
      base_diagram: diagram,
      backend_diagram: { ...diagram, name: "backend" },
      variant_diagrams: [],
    },
  };
}

describe("production session bootstrap", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("creates the first session from the exact starter YAML", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: "created" }),
    });
    vi.stubGlobal("fetch", fetch);
    const yamlText = "model: {}\nruns: []\n";

    await createSession(yamlText);

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ yaml_text: yamlText }),
      }),
    );
  });

  it("gets the canonical starter", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ yaml_text: "schema_version: 1\nruns: []\n" }),
    });
    vi.stubGlobal("fetch", fetch);

    expect(await getStarter()).toEqual({
      yaml_text: "schema_version: 1\nruns: []\n",
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/starter",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("fetches the starter before creating a session from its exact YAML bytes", async () => {
    const yamlText = "schema_version: 1\nmodel: {}\nruns: []\n";
    const fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ yaml_text: yamlText }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ session_id: "created" }),
      });
    vi.stubGlobal("fetch", fetch);

    await createStarterSession();

    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "/api/starter",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "/api/sessions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ yaml_text: yamlText }),
      }),
    );
  });

  it("keeps a named busy workbench frame while the starter request is pending", () => {
    const starter = deferred<ReturnType<typeof response>>();
    const fetch = vi.fn().mockReturnValue(starter.promise);
    vi.stubGlobal("fetch", fetch);
    render(<App />);

    expect(screen.getByRole("status", { name: "Workbench startup" }))
      .toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      "/api/starter",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("keeps the workbench frame when the starter request fails", async () => {
    const fetch = vi.fn().mockResolvedValue(response({ detail: "starter unavailable" }, false));
    vi.stubGlobal("fetch", fetch);
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("starter unavailable");
    expect(screen.getByRole("main")).toBeInTheDocument();
  });

  it("keeps a named busy workbench frame while session creation is pending", async () => {
    const session = deferred<ReturnType<typeof response>>();
    const fetch = vi.fn()
      .mockResolvedValueOnce(response({ yaml_text: "schema_version: 1\n" }))
      .mockReturnValueOnce(session.promise);
    vi.stubGlobal("fetch", fetch);
    render(<App />);

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("status", { name: "Workbench startup" }))
      .toHaveAttribute("aria-busy", "true");
  });

  it("keeps the workbench frame when session creation fails", async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(response({ yaml_text: "schema_version: 1\n" }))
      .mockResolvedValueOnce(response({ detail: "session unavailable" }, false));
    vi.stubGlobal("fetch", fetch);
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("session unavailable");
    expect(screen.getByRole("main")).toBeInTheDocument();
  });

  it("shows the editor after both bootstrap requests succeed", async () => {
    const yamlText = "schema_version: 1\n";
    const fetch = vi.fn()
      .mockResolvedValueOnce(response({ yaml_text: yamlText }))
      .mockResolvedValueOnce(response(createdSession(yamlText)));
    vi.stubGlobal("fetch", fetch);
    render(<App />);

    expect(await screen.findByRole("tab", { name: "Model" })).toBeVisible();
  });

  it("does not update the workbench after unmount during bootstrap", async () => {
    const session = deferred<ReturnType<typeof response>>();
    const fetch = vi.fn()
      .mockResolvedValueOnce(response({ yaml_text: "schema_version: 1\n" }))
      .mockReturnValueOnce(session.promise);
    vi.stubGlobal("fetch", fetch);
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { container, unmount } = render(<App />);

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    unmount();
    session.resolve(response(createdSession("schema_version: 1\n")));
    await Promise.resolve();
    await Promise.resolve();

    expect(container).toBeEmptyDOMElement();
    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });
});
