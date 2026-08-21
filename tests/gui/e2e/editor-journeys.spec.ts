import {
  COMPLETE_YAML,
  GRAPH_VARIANT_YAML,
  INVALID_YAML_DRAFT,
  expect,
  test,
} from "./fixtures";
import {
  assertPageAudit,
  beginPageAudit,
  closeContextInspector,
  closeYamlDrawer,
  expectHandledHttpError,
  gotoWorkbench,
  gotoWorkbenchSession,
  loadYaml,
  openContextInspector,
  openWorkspace,
  openYaml,
  replaceYamlDraft,
  waitForSuccess,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  beginPageAudit(page);
});

test.afterEach(async ({ page }) => {
  await assertPageAudit(page);
});

test("first preview follows the trusted, bounded workbench journey", async ({ page }) => {
  await gotoWorkbench(page);

  const guide = page.getByRole("region", { name: "First preview checklist" });
  await expect(guide).toContainText("Required choices complete");
  await expect(guide).toContainText("Quick checks clean");
  await expect(guide).toContainText("Forward preview not run");
  await expect(guide).toContainText("Next: run Preview forward.");
  await expect(page.getByText(
    "Trusted YAML: plugins, python targets, paths and jobs run as the server account.",
  )).toBeVisible();

  await openWorkspace(page, "Execute");
  await page.getByRole("button", { name: "Preview forward" }).click();
  const confirmation = page.getByRole("dialog", { name: "Trusted execution" });
  await expect(confirmation).toContainText("Requested action: Preview forward");
  await expect(confirmation.getByRole("button", { name: "Cancel trusted execution" }))
    .toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(confirmation.getByRole("button", { name: "I understand, continue" }))
    .toBeFocused();
  await page.keyboard.press("Tab");
  await expect(confirmation.getByRole("button", { name: "Cancel trusted execution" }))
    .toBeFocused();
  await confirmation.getByRole("button", { name: "I understand, continue" }).click();

  await waitForSuccess(page, "preview_forward");
  await expect(page.getByRole("button", { name: "Help" })).toBeVisible();
  await expect(page.getByRole("region", { name: "First preview checklist" }))
    .toHaveCount(0);
  await openWorkspace(page, "Results");
  await expect(page.getByRole("group", { name: "Current history" }))
    .toContainText("preview_forward");
});

test("graph editing preserves base and variant context through Undo and Redo", async ({ page }) => {
  await gotoWorkbench(page);
  await loadYaml(page, GRAPH_VARIANT_YAML);
  await closeYamlDrawer(page);

  const graph = page.getByRole("group", { name: "Signal path" });
  await graph.getByRole("button", { name: "Edit gain" }).click();
  await openContextInspector(page);
  const settings = page.getByRole("textbox", { name: "Node settings JSON" });
  await expect(settings).toContainText('"value": 1.1');
  await settings.fill(JSON.stringify({ gain: { value: 1.2, unit: "dimensionless" } }, null, 2));
  await page.getByRole("button", { name: "Apply configuration to gain" }).click();
  await expect(page.getByText("Configured gain", { exact: true })).toBeVisible();
  await expect(settings).toContainText('"value": 1.2');

  await closeContextInspector(page);
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(page.getByText("Undid YAML edit", { exact: true })).toBeVisible();
  await openContextInspector(page);
  await expect(settings).toContainText('"value": 1.1');
  await closeContextInspector(page);
  await page.getByRole("button", { name: "Redo" }).click();
  await expect(page.getByText("Redid YAML edit", { exact: true })).toBeVisible();
  await openContextInspector(page);
  await expect(settings).toContainText('"value": 1.2');

  await closeContextInspector(page);
  await page.getByRole("combobox", { name: "Editing layer" }).selectOption("low_gain");
  await graph.getByRole("button", { name: "Edit gain" }).click();
  await openContextInspector(page);
  await expect(settings).toContainText('"value": 0.9');
  await closeContextInspector(page);
  await page.getByRole("button", { name: "Compare" }).click();
  const comparison = page.getByRole("region", { name: "Base versus selected variant" });
  await expect(comparison.getByRole("heading", { name: "Base" })).toBeVisible();
  await expect(comparison.getByRole("heading", { name: "low_gain" })).toBeVisible();
  await expect(comparison).toContainText("Changed nodes: gain");
});

test("projected primitives mutate YAML while file controls remain server paths", async ({ page }) => {
  await gotoWorkbench(page);
  await loadYaml(page, COMPLETE_YAML);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Config");

  const filter = page.getByRole("searchbox", { name: "Filter configuration fields" });
  await filter.fill("runtime.seed");
  const seed = page.getByRole("article", { name: "runtime.seed" });
  await seed.getByRole("textbox", { name: "seed" }).fill("20260821");
  await seed.getByRole("button", { name: "Apply field" }).click();
  await expect(page.getByText("Updated runtime.seed", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /^Instrument/ }).click();
  await filter.fill("model.gain.eqx_leaves");
  const pathField = page.getByRole("article", { name: "model.gain.eqx_leaves" });
  const serverPath = pathField.getByRole("textbox", { name: "Server path" });
  await expect(serverPath).toHaveAttribute("type", "text");
  await serverPath.fill("/srv/rheplicant/gain.eqx");
  await expect(page.getByText("Unsaved field: model.gain.eqx_leaves", { exact: true }))
    .toBeVisible();
  await expect(pathField.getByRole("button", { name: "Apply field" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Load YAML file" })).toHaveCount(0);
  await expect(page.getByLabel("Load YAML file")).toHaveAttribute("type", "file");
  await pathField.getByRole("button", { name: "Discard field" }).click();
  await expect(serverPath).toHaveValue("");
});

test("invalid YAML retains the last-good graph and blocks accepted mutations", async ({ page }) => {
  const initial = await gotoWorkbenchSession(page);
  await loadYaml(page, COMPLETE_YAML);
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  await replaceYamlDraft(page, INVALID_YAML_DRAFT);
  expectHandledHttpError(page, "PUT", `/api/sessions/${initial.session_id}/yaml`, 422);
  await drawer.getByRole("button", { name: "Apply YAML edit" }).click();

  await expect(drawer.getByRole("alert")).toContainText("Invalid YAML:");
  await expect(drawer.getByRole("textbox", { name: "YAML source of truth" }))
    .toHaveValue(INVALID_YAML_DRAFT);
  await expect(page.getByRole("button", { name: "Undo" })).toBeDisabled();
  await closeYamlDrawer(page);
  await expect(page.getByRole("group", { name: "Signal path" }))
    .toContainText("gain");
  await openWorkspace(page, "Execute");
  await expect(page.getByRole("button", { name: "Run" })).toBeDisabled();
});

test("revision conflict retains the raw draft and exposes comparison actions", async ({ page, request }) => {
  const initial = await gotoWorkbenchSession(page);
  const drawer = await openYaml(page);
  const rawDraft = `${initial.document.yaml_text}# retained browser draft\n`;
  await replaceYamlDraft(page, rawDraft);

  const external = await request.patch(
    `/api/sessions/${encodeURIComponent(initial.session_id)}/fields`,
    {
      data: {
        expected_revision: initial.revision,
        path: "runtime.seed",
        value: 20260821,
        remove: false,
      },
    },
  );
  expect(external.ok()).toBe(true);
  expectHandledHttpError(page, "PUT", `/api/sessions/${initial.session_id}/yaml`, 409);
  await drawer.getByRole("button", { name: "Apply YAML edit" }).click();

  const conflict = drawer.getByRole("region", { name: "YAML revision conflict" });
  await expect(conflict).toContainText("Revision conflict:");
  await expect(conflict).toContainText(
    `Draft base revision ${initial.revision}; accepted revision ${initial.revision}.`,
  );
  await expect(drawer.getByRole("textbox", { name: "YAML source of truth" }))
    .toHaveValue(rawDraft);
  await expect(conflict.getByRole("button", { name: "Copy draft" })).toBeEnabled();
  await expect(conflict.getByRole("button", { name: "Refresh accepted YAML" })).toBeEnabled();
  await expect(drawer.getByRole("button", { name: "Discard draft" })).toBeEnabled();
});

test("wide first-use baseline", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  await gotoWorkbench(page);
  await expect(page.getByRole("region", { name: "First preview checklist" })).toBeVisible();
  await expect(page).toHaveScreenshot("first-use.png", { fullPage: true });
});

test("wide graph-selection baseline", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  await gotoWorkbench(page);
  await loadYaml(page, GRAPH_VARIANT_YAML);
  await closeYamlDrawer(page);
  await page.getByRole("group", { name: "Signal path" })
    .getByRole("button", { name: "Edit gain" }).click();
  await openContextInspector(page);
  await expect(page.getByRole("complementary", { name: "gain settings" })).toBeVisible();
  await expect(page).toHaveScreenshot("graph-selection.png", { fullPage: true });
});

test("wide required-config baseline", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  await gotoWorkbench(page);
  await loadYaml(page, "schema_version: 1\n");
  await closeYamlDrawer(page);
  await openWorkspace(page, "Config");
  const sections = page.getByRole("navigation", { name: "Configuration sections" });
  await sections.getByRole("button").filter({ hasText: "incomplete" }).first().click();
  const missing = page.getByRole("region", { name: "Missing required fields" });
  await expect(missing).toBeVisible();
  await missing.scrollIntoViewIfNeeded();
  await expect(page).toHaveScreenshot("required-config.png", { fullPage: true });
});

test("wide invalid-YAML baseline", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  const initial = await gotoWorkbenchSession(page);
  await loadYaml(page, COMPLETE_YAML);
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  await replaceYamlDraft(page, INVALID_YAML_DRAFT);
  expectHandledHttpError(page, "PUT", `/api/sessions/${initial.session_id}/yaml`, 422);
  await drawer.getByRole("button", { name: "Apply YAML edit" }).click();
  await expect(drawer.getByRole("alert")).toContainText("Invalid YAML:");
  await expect(page).toHaveScreenshot("invalid-yaml.png", { fullPage: true });
});
