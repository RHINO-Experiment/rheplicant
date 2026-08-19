import { afterEach, describe, expect, it, vi } from "vitest";

import { createSession } from "../../../src/rheplicant/gui/react/api";

describe("production session bootstrap", () => {
  afterEach(() => {
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
});
