import { access, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  COMPLETE_YAML,
  GRAPH_VARIANT_YAML,
  INVALID_YAML_DRAFT,
  STALE_JOB_CURRENT_YAML,
  STALE_JOB_SOURCE_YAML,
  STARTER_YAML,
  AxeBuilder,
  allocateOutputTarget,
  expect,
  outputYaml,
  refusalJobYaml,
  sameDirectoryIdentity,
  successfulJobYaml,
  test,
} from "./fixtures";
import {
  DETERMINISTIC_MOTION_CSS,
  JOB_TERMINAL_TIMEOUT_MS,
  gotoWorkbench,
} from "./helpers";

const repository = resolve(process.cwd(), "../../..");

function rootKeyCount(yaml: string, key: string): number {
  return yaml.match(new RegExp(`^${key}:`, "gm"))?.length ?? 0;
}

test("production workbench exposes four workspaces", async ({ page }) => {
  await gotoWorkbench(page);
  await expect(page.getByRole("tab", { name: "Model" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Config" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Execute" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Results" })).toBeVisible();
  expect(await page.locator("body").evaluate((body) => getComputedStyle(body).animationDuration))
    .toBe("1e-06s");
});

test("harness fixes the source server and four viewport projects", async ({}, testInfo) => {
  expect(testInfo.config.projects.map(({ name }) => name)).toEqual([
    "wide",
    "laptop",
    "compact",
    "narrow",
  ]);
  expect(testInfo.config.projects.map(({ use }) => use.viewport)).toEqual([
    { width: 1440, height: 1000 },
    { width: 1024, height: 768 },
    { width: 768, height: 900 },
    { width: 640, height: 900 },
  ]);
  expect(testInfo.project.use).toMatchObject({
    baseURL: process.env.RHEPLICANT_E2E_BASE_URL ?? "http://127.0.0.1:8127",
    colorScheme: "light",
  });

  const source = await readFile(
    resolve(repository, "tools/config_gui_spike/react/playwright.config.ts"),
    "utf8",
  );
  expect(source).toContain('snapshotPathTemplate: "{testDir}/snapshots/{arg}{ext}"');
  expect(source).toContain(
    "PYTHONPATH=src .venv/bin/python -m rheplicant.gui.launcher --host 127.0.0.1 --port 8127 --log-level warning",
  );
  expect(source).toContain("const usesExternalServer = external !== undefined;");
  expect(source).toMatch(/webServer: usesExternalServer\s*\? undefined/);
  expect(source).toContain("reuseExistingServer: false");

  const smoke = await readFile(resolve(repository, "tests/gui/e2e/smoke.spec.ts"), "utf8");
  for (const workspace of ["Model", "Config", "Execute", "Results"]) {
    expect(smoke).toContain(`page.getByRole("tab", { name: "${workspace}" })`);
  }
});

test("package scripts and development dependencies retain exact pins", async () => {
  const packageRoot = resolve(repository, "tools/config_gui_spike/react");
  const manifest = JSON.parse(await readFile(resolve(packageRoot, "package.json"), "utf8"));
  const lock = JSON.parse(await readFile(resolve(packageRoot, "package-lock.json"), "utf8"));

  expect(manifest.scripts["test:e2e"]).toBe("playwright test");
  expect(manifest.scripts["test:e2e:update"]).toBe("playwright test --update-snapshots");
  expect(manifest.scripts["check:e2e"]).toBe(
    "tsc --noEmit --strict --skipLibCheck --target ES2022 --module ESNext --moduleResolution Bundler --types node playwright.config.ts ../../../tests/gui/e2e/fixtures.ts ../../../tests/gui/e2e/helpers.ts ../../../tests/gui/e2e/smoke.spec.ts ../../../tests/gui/e2e/editor-journeys.spec.ts ../../../tests/gui/e2e/execution-results.spec.ts ../../../tests/gui/e2e/responsive.spec.ts ../../../tests/gui/e2e/accessibility.spec.ts",
  );
  expect(manifest.devDependencies["@playwright/test"]).toBe("1.62.1");
  expect(manifest.devDependencies["@axe-core/playwright"]).toBe("4.10.2");
  expect(manifest.devDependencies["@types/node"]).toBe("25.9.5");
  expect(lock.packages[""].devDependencies["@playwright/test"]).toBe("1.62.1");
  expect(lock.packages[""].devDependencies["@axe-core/playwright"]).toBe("4.10.2");
  expect(lock.packages[""].devDependencies["@types/node"]).toBe("25.9.5");
  expect(lock.packages["node_modules/@playwright/test"].version).toBe("1.62.1");
  expect(lock.packages["node_modules/@axe-core/playwright"].version).toBe("4.10.2");
  expect(lock.packages["node_modules/@types/node"].version).toBe("25.9.5");
  expect(lock.packages["node_modules/playwright"].version).toBe("1.62.1");
  expect(lock.packages["node_modules/playwright-core"].version).toBe("1.62.1");
  expect(typeof AxeBuilder).toBe("function");
});

test("responsive coverage retains vertical overflow and header-bound assertions", async () => {
  const responsive = await readFile(
    resolve(repository, "tests/gui/e2e/responsive.spec.ts"),
    "utf8",
  );

  expect(responsive).toContain("verticalDelta: 0,");
  expect(responsive).toContain("bodyVerticalDelta: 0,");
  expect(responsive).toContain("expect(await headerActionsOverflow(page)).toEqual");
  expect(responsive).toContain("expect(firstPreviewMain.usableHeight).toBeGreaterThanOrEqual(128);");
  expect(responsive).toContain("expect(firstPreviewMain.nextActionOverflow).toEqual");
});

test("output fixture adds safe root keys exactly once", () => {
  expect(rootKeyCount(COMPLETE_YAML, "outputs")).toBe(0);
  expect(rootKeyCount(COMPLETE_YAML, "plugins")).toBe(0);

  const target = '/tmp/rheplicant output "quoted"';
  const rendered = outputYaml(target);

  expect(rootKeyCount(rendered, "outputs")).toBe(1);
  expect(rootKeyCount(rendered, "plugins")).toBe(1);
  expect(rendered.match(/^  - json$/gm)).toHaveLength(1);
  expect(rendered).toContain(`  dir: ${JSON.stringify(target)}`);
  const encodedTarget = rendered.match(/^  dir: (.+)$/m)?.[1];
  expect(encodedTarget).toBeDefined();
  expect(JSON.parse(encodedTarget!)).toBe(target);
});

test("fixture documents name starter, graph, invalid and stale states explicitly", () => {
  expect(STARTER_YAML).toContain("schema_version: 1\n");
  expect(GRAPH_VARIANT_YAML).toBe(COMPLETE_YAML);
  expect(rootKeyCount(GRAPH_VARIANT_YAML, "variants")).toBe(1);
  expect(INVALID_YAML_DRAFT).toBe("schema_version: [\n");
  expect(STALE_JOB_SOURCE_YAML).toBe(COMPLETE_YAML);
  expect(STALE_JOB_CURRENT_YAML).toBe(`${COMPLETE_YAML}# accepted after job submission\n`);
  expect(STALE_JOB_CURRENT_YAML).not.toBe(STALE_JOB_SOURCE_YAML);
});

test("source starter and complete fixture are accepted byte-for-byte", async ({ request }) => {
  const starter = await request.get("/api/starter");
  expect(starter.ok()).toBe(true);
  expect((await starter.json()).yaml_text).toBe(STARTER_YAML);

  const created = await request.post("/api/sessions", {
    data: { yaml_text: COMPLETE_YAML },
  });
  expect(created.status()).toBe(201);
  expect((await created.json()).document.yaml_text).toBe(COMPLETE_YAML);
});

test("output fixture supplies distinct success and existing-target refusal documents", async ({
  outputDocument,
  outputTarget,
  request,
}) => {
  expect(outputDocument).toBe(successfulJobYaml(outputTarget));
  const created = await request.post("/api/sessions", {
    data: { yaml_text: outputDocument },
  });
  expect(created.status()).toBe(201);
  expect((await created.json()).document.yaml_text).toBe(outputDocument);

  const allocation = await allocateOutputTarget();
  try {
    const refusal = refusalJobYaml(allocation.directory);
    expect(refusal).toContain(`  dir: ${JSON.stringify(allocation.directory)}`);
    expect(refusal).not.toBe(outputDocument);
  } finally {
    await allocation.cleanup();
  }
});

test("helpers retain deterministic motion and bounded semantic waits", async () => {
  expect(DETERMINISTIC_MOTION_CSS).toContain("animation-duration: 0.001ms !important");
  expect(DETERMINISTIC_MOTION_CSS).toContain("transition-duration: 0.001ms !important");
  expect(DETERMINISTIC_MOTION_CSS).toContain("scroll-behavior: auto !important");
  expect(JOB_TERMINAL_TIMEOUT_MS).toBe(60_000);

  const source = await readFile(resolve(repository, "tests/gui/e2e/helpers.ts"), "utf8");
  expect(source).toContain('.getByRole("listitem")');
  expect(source).not.toContain("waitForTimeout(");
  expect(source).not.toMatch(/\.locator\(["'`]\./);
});

test("temporary targets are unique and cleanup removes only its own allocation", async () => {
  const first = await allocateOutputTarget();
  const second = await allocateOutputTarget();
  try {
    expect(first.directory).not.toBe(second.directory);
    expect(first.target).not.toBe(second.target);
    expect(await realpath(first.directory)).toBe(first.directory);
    expect(await realpath(second.directory)).toBe(second.directory);
    await writeFile(resolve(second.directory, "keep.txt"), "second allocation\n", "utf8");

    await first.cleanup();

    await expect(access(first.directory)).rejects.toThrow();
    await expect(access(resolve(second.directory, "keep.txt"))).resolves.toBeUndefined();
  } finally {
    await first.cleanup();
    await second.cleanup();
  }
});

test("cleanup refuses a replaced allocation without touching its destination", async () => {
  const guarded = await allocateOutputTarget();
  const destination = await allocateOutputTarget();
  try {
    await rm(guarded.directory, { recursive: true });
    await symlink(destination.directory, guarded.directory, "dir");

    await expect(guarded.cleanup()).rejects.toThrow(/replaced temporary output allocation/);
    await expect(access(destination.directory)).resolves.toBeUndefined();
  } finally {
    await rm(guarded.directory, { force: true });
    await guarded.cleanup();
    await destination.cleanup();
  }
});

test("cleanup refuses a same-path directory replacement and preserves its contents", async () => {
  const allocation = await allocateOutputTarget();
  const keep = resolve(allocation.directory, "keep.txt");
  try {
    await rm(allocation.directory, { recursive: true });
    await mkdir(allocation.directory);
    await writeFile(keep, "replacement directory\n", "utf8");

    await expect(allocation.cleanup()).rejects.toThrow(/replaced temporary output allocation/);
    await expect(access(keep)).resolves.toBeUndefined();
  } finally {
    await rm(allocation.directory, { force: true, recursive: true });
    await allocation.cleanup();
  }
});

test("directory identity requires both the original device and inode", () => {
  const original = { dev: 7, ino: 11 };
  expect(sameDirectoryIdentity(original, { dev: 7, ino: 11 })).toBe(true);
  expect(sameDirectoryIdentity(original, { dev: 8, ino: 11 })).toBe(false);
  expect(sameDirectoryIdentity(original, { dev: 7, ino: 12 })).toBe(false);
});
