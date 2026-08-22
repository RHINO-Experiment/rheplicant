import { dirname } from "node:path";

import {
  COMPLETE_YAML,
  STALE_JOB_CURRENT_YAML,
  expect,
  refusalJobYaml,
  test,
} from "./fixtures";
import {
  assertPageAudit,
  beginPageAudit,
  closeYamlDrawer,
  gotoWorkbench,
  gotoWorkbenchSession,
  loadYaml,
  openContextInspector,
  openWorkspace,
  submitTrustedJob,
  waitForSuccess,
  waitForTerminal,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  beginPageAudit(page);
});

test.afterEach(async ({ page }) => {
  await assertPageAudit(page);
});

test("successful fixture explicitly resolves the partial pointing preset", async ({ request }) => {
  expect(COMPLETE_YAML).toContain(
    "  pointing:\n    materialise: []\n",
  );
  const created = await request.post("/api/sessions", {
    data: { yaml_text: COMPLETE_YAML },
  });
  expect(created.status()).toBe(201);
  expect((await created.json()).document.yaml_text).toBe(COMPLETE_YAML);
});

test("Validate and Forward preview use the accepted outputs/plugin document", async ({
  outputDocument,
  outputTarget,
  page,
}) => {
  await gotoWorkbench(page);
  await loadYaml(page, outputDocument);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");
  await expect(page.getByRole("region", { name: "Output target" }))
    .toContainText(outputTarget);

  await submitTrustedJob(page, "Validate");
  await waitForSuccess(page, "validate");
  await page.getByRole("button", { name: "Preview forward" }).click();
  await expect(page.getByRole("dialog", { name: "Trusted execution" }))
    .toHaveCount(0);
  await waitForSuccess(page, "preview_forward");

  await openWorkspace(page, "Results");
  const current = page.getByRole("group", { name: "Current history" });
  await expect(current).toContainText("validate");
  await expect(current).toContainText("preview_forward");
});

test("an existing target is visibly refused until YAML recovers to a new target", async ({
  outputDocument,
  outputTarget,
  page,
}) => {
  const initial = await gotoWorkbenchSession(page);
  const existingTarget = dirname(outputTarget);
  await loadYaml(page, refusalJobYaml(existingTarget));
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");

  const output = page.getByRole("region", { name: "Output target" });
  await expect(output.getByText("Existing target blocked", { exact: true })).toBeVisible();
  await expect(output.getByText(
    "The target exists and outputs.clobber is false.",
    { exact: true },
  )).toBeVisible();
  await expect(output.getByText(existingTarget, { exact: true })).toBeVisible();
  const run = page.getByRole("button", { name: "Run", exact: true });
  await expect(run).toBeDisabled();
  await expect(run).toHaveAccessibleDescription("Run disabled: repair the output target.");

  const currentResponse = await page.request.get(
    `/api/sessions/${encodeURIComponent(initial.session_id)}`,
  );
  expect(currentResponse.ok()).toBe(true);
  const current = await currentResponse.json() as { revision: number };
  const refusedResponse = await page.request.post(
    `/api/sessions/${encodeURIComponent(initial.session_id)}/jobs`,
    { data: { expected_revision: current.revision, kind: "run" } },
  );
  expect(refusedResponse.status()).toBe(202);
  await page.getByRole("button", { name: "Refresh jobs" }).click();
  const refused = await waitForTerminal(page, "run");
  await expect(refused).toContainText(/·\s*run\s*·\s*refused\b/i);
  await expect(waitForSuccess(page, "run")).rejects.toThrow();

  const yaml = page.getByRole("button", { name: "YAML", exact: true });
  await expect(yaml).toBeEnabled();
  await loadYaml(page, outputDocument);
  await closeYamlDrawer(page);
  await expect(output.getByText("New target ready", { exact: true })).toBeVisible();
  await expect(output.getByText(outputTarget, { exact: true })).toBeVisible();
  await expect(run).toBeEnabled();
  await submitTrustedJob(page, "Run");
  await waitForSuccess(page, "run");
});

test("one product, report, Run and audit bundle complete without traversing the catalog", async ({
  outputDocument,
  outputTarget,
  page,
}) => {
  const initial = await gotoWorkbenchSession(page);
  await loadYaml(page, outputDocument);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");

  await page.getByRole("button", { name: "Add product" }).click();
  const products = page.getByRole("listbox", { name: "Available products" });
  await expect(products.getByRole("option")).toHaveCount(23);
  await products.getByRole("option", { name: "timings" }).click();
  const timings = page.getByRole("group", { name: "timings product settings" });
  await timings.getByRole("checkbox", { name: "Write timings" }).click();
  await expect(page.getByText("Updated timings output", { exact: true })).toBeVisible();
  await expect(timings.getByRole("checkbox", { name: "Write timings" })).toBeChecked();
  await expect(page.getByRole("list", { name: "Enabled products" }))
    .toContainText("Expand timings product settings");

  await page.getByRole("button", { name: "Write report" }).click();
  const report = page.getByRole("group", { name: "Report selector" });
  await expect(report).toBeVisible();
  await expect(report.getByRole("checkbox", { name: "Report row forward" }))
    .toBeChecked();
  const mean = report.getByRole("checkbox", { name: "mean", exact: true });
  const standardDeviation = report.getByRole("checkbox", { name: "std", exact: true });
  await mean.click();
  await expect(mean).not.toBeChecked();
  await standardDeviation.click();
  await expect(standardDeviation).not.toBeChecked();
  await expect(report.getByRole("checkbox", { name: "seconds", exact: true })).toBeChecked();

  await submitTrustedJob(page, "Run");
  const terminal = await waitForSuccess(page, "run");
  const jobId = await terminal.locator("code").innerText();
  const visibleJobId = jobId.length > 6 ? `${jobId.slice(0, 6)}…` : jobId;
  await openWorkspace(page, "Results");
  const history = page.getByRole("group", { name: "Current history" });
  const runItem = history.getByRole("listitem").filter({ hasText: /run/ }).last();
  const selector = runItem.getByRole("button", { name: `View job ${visibleJobId}`, exact: true });
  await selector.click();
  await expect(selector).toHaveAttribute("aria-pressed", "true");
  await openContextInspector(page);
  const result = page.getByRole("region", { name: "Result summary" });
  await expect(result).toContainText("Succeeded");
  await expect(result).toContainText("run · revision");
  await expect(result.getByText(jobId, { exact: true })).toBeVisible();
  const publishedOutput = result.getByRole("region", { name: "Published output summary" });
  await expect(publishedOutput).toContainText(outputTarget);
  expect(await publishedOutput.evaluate((region) => region.scrollWidth - region.clientWidth))
    .toBe(0);
  const pathCode = publishedOutput.getByText(outputTarget, { exact: true });
  const pathOverflow = await pathCode.evaluate((code) => {
    const region = code.closest<HTMLElement>('[aria-label="Published output summary"]');
    if (region === null) throw new Error("Output path has no published-output owner.");
    const codeBox = code.getBoundingClientRect();
    const regionBox = region.getBoundingClientRect();
    return {
      left: Math.max(0, regionBox.left - codeBox.left),
      right: Math.max(0, codeBox.right - regionBox.right),
    };
  });
  expect(pathOverflow).toEqual({ left: 0, right: 0 });
  const audit = page.getByRole("region", { name: "Completed audit bundles" });
  const file = "products.json";
  const link = audit.getByRole("link", { name: file, exact: true });
  const artifactPath = `/api/sessions/${encodeURIComponent(initial.session_id)}/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(file)}`;
  const artifactUrl = new URL(artifactPath, page.url()).href;
  await expect(link).toHaveAttribute("href", artifactPath);
  await expect(link).toHaveAttribute("target", "_blank");
  const [artifactPage, artifactResponse] = await Promise.all([
    page.waitForEvent("popup"),
    page.context().waitForEvent("response", (response) => (
      response.url() === artifactUrl
      && response.request().method() === "GET"
      && response.request().resourceType() === "document"
    )),
    link.click(),
  ]);
  await expect.poll(() => artifactPage.url()).toBe(artifactUrl);
  expect(artifactResponse.status()).toBe(200);
  const artifact = await artifactResponse.text();
  expect(artifact.length).toBeLessThan(100_000);
  expect(artifact).toContain('"arrays"');
  expect(artifact).toContain('"timings"');
  await expect(selector).toHaveAttribute("aria-pressed", "true");
  await expect(result.getByText(jobId, { exact: true })).toBeVisible();
  await artifactPage.close();
});

test("terminal evidence becomes stale and Re-run submits current YAML", async ({ page }) => {
  await gotoWorkbench(page);
  await loadYaml(page, COMPLETE_YAML);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");
  await submitTrustedJob(page, "Validate");
  await waitForSuccess(page, "validate");

  await loadYaml(page, STALE_JOB_CURRENT_YAML);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Results");
  const stale = page.getByRole("group", { name: "Stale history" });
  await expect(stale).toContainText("From revision");
  await stale.getByRole("button", { name: /^View job / }).click();
  await openContextInspector(page);
  await expect(page.getByRole("region", { name: "Result summary" }))
    .toContainText("From revision");
  const rerun = page.getByRole("button", { name: "Re-run Validate" });
  await expect(rerun).toBeEnabled();
  await rerun.click();
  await waitForSuccess(page, "validate");
  await expect(page.getByRole("group", { name: "Current history" }))
    .toContainText("validate");
});

test("journeys retain source and packaged-server compatible public contracts", async ({ page, request }) => {
  await gotoWorkbench(page);
  const starter = await request.get("/api/starter");
  expect(starter.ok()).toBe(true);
  expect((await starter.json()).yaml_text).toContain("schema_version: 1");
  for (const workspace of ["Model", "Config", "Execute", "Results"] as const) {
    await openWorkspace(page, workspace);
    await expect(page.getByRole("tabpanel", { name: workspace })).toBeVisible();
  }
});

test("wide execute-setup baseline", async ({
  outputDocument,
  outputTarget,
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  await gotoWorkbench(page);
  await loadYaml(page, outputDocument);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");
  await page.getByRole("button", { name: "Dismiss setup guide" }).click();
  await expect(page.getByRole("button", { name: "Help" })).toBeVisible();
  const output = page.getByRole("region", { name: "Output target" });
  await expect(output).toContainText(outputTarget);
  await output.evaluate((element) => {
    const scroller = element.closest<HTMLElement>('[class~="workbench-main"]');
    if (scroller === null) throw new Error("Output target has no workbench scroller.");
    scroller.scrollTop += element.getBoundingClientRect().top
      - scroller.getBoundingClientRect().top;
  });
  await expect(output).toBeInViewport();
  await expect(page).toHaveScreenshot("execute-setup.png", {
    fullPage: true,
    mask: [page.getByText(outputTarget, { exact: true })],
  });
});

test("wide results baseline", async ({
  outputDocument,
  outputTarget,
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "wide", "Canonical screenshots use the wide viewport.");
  await gotoWorkbench(page);
  await loadYaml(page, outputDocument);
  await closeYamlDrawer(page);
  await openWorkspace(page, "Execute");
  await submitTrustedJob(page, "Run");
  const terminal = await waitForSuccess(page, "run");
  const jobId = await terminal.locator("code").innerText();
  const visibleJobId = jobId.length > 6 ? `${jobId.slice(0, 6)}…` : jobId;

  await openWorkspace(page, "Results");
  const history = page.getByRole("group", { name: "Current history" });
  await history.getByRole("button", { name: /^View job / }).click();
  await openContextInspector(page);
  await expect(page.getByRole("region", { name: "Result summary" }))
    .toContainText("Succeeded");
  await expect(page.getByRole("region", { name: "Completed audit bundles" }))
    .toBeVisible();
  await expect(page).toHaveScreenshot("results.png", {
    fullPage: true,
    mask: [
      page.getByText(jobId, { exact: true }),
      page.getByText(visibleJobId, { exact: true }),
      page.getByText(outputTarget, { exact: true }),
    ],
  });
});
