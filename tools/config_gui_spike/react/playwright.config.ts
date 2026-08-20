import { defineConfig } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const sourceBaseUrl = "http://127.0.0.1:8127";

type Environment = Record<string, string | undefined>;

export function buildPlaywrightConfig(environment: Environment) {
  const external = environment.RHEPLICANT_E2E_BASE_URL;
  const usesExternalServer = external !== undefined;

  return defineConfig({
    testDir: resolve(repository, "tests/gui/e2e"),
    outputDir: resolve(
      repository,
      ".superpowers/sdd/2026-08-20-config-plan-6c-execution-results-release/playwright-results",
    ),
    snapshotPathTemplate: "{testDir}/snapshots/{arg}{ext}",
    use: {
      baseURL: external ?? sourceBaseUrl,
      colorScheme: "light",
      screenshot: "only-on-failure",
      trace: "retain-on-failure",
    },
    projects: [
      { name: "wide", use: { viewport: { width: 1440, height: 1000 } } },
      { name: "laptop", use: { viewport: { width: 1024, height: 768 } } },
      { name: "compact", use: { viewport: { width: 768, height: 900 } } },
      { name: "narrow", use: { viewport: { width: 640, height: 900 } } },
    ],
    webServer: usesExternalServer
      ? undefined
      : {
          command:
            "PYTHONPATH=src .venv/bin/python -m rheplicant.gui.launcher --host 127.0.0.1 --port 8127 --log-level warning",
          cwd: repository,
          url: `${sourceBaseUrl}/`,
          reuseExistingServer: false,
        },
  });
}

export default buildPlaywrightConfig(process.env);
