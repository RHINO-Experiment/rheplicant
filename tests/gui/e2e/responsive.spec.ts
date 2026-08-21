import { expect, test } from "./fixtures";
import {
  activeMainUsability,
  activeWorkspaceTabStops,
  assertPageAudit,
  beginPageAudit,
  documentOverflow,
  gotoWorkbench,
  headerActionsOverflow,
  openWorkspace,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  beginPageAudit(page);
});

test.afterEach(async ({ page }) => {
  await assertPageAudit(page);
});

test("each viewport contains page overflow and bounds active workspace tab stops", async ({
  page,
}) => {
  await gotoWorkbench(page);
  const firstPreviewMain = await activeMainUsability(page);
  expect(firstPreviewMain.usableHeight).toBeGreaterThanOrEqual(128);
  expect(firstPreviewMain.nextActionOverflow).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
  expect(await documentOverflow(page)).toEqual({
    delta: 0,
    bodyDelta: 0,
    verticalDelta: 0,
    bodyVerticalDelta: 0,
  });
  expect(await headerActionsOverflow(page)).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
  for (const workspace of ["Model", "Config", "Execute", "Results"] as const) {
    await openWorkspace(page, workspace);
    expect(await documentOverflow(page)).toEqual({
      delta: 0,
      bodyDelta: 0,
      verticalDelta: 0,
      bodyVerticalDelta: 0,
    });
    expect(await headerActionsOverflow(page)).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
    expect((await activeMainUsability(page)).usableHeight).toBeGreaterThanOrEqual(128);
    expect(await activeWorkspaceTabStops(page)).toBeLessThan(60);
  }
});

test("workspace orientation keys keep exactly one selected tab stop", async ({ page }, testInfo) => {
  await gotoWorkbench(page);
  const tablist = page.getByRole("tablist");
  const horizontal = testInfo.project.name === "compact" || testInfo.project.name === "narrow";
  await expect(tablist).toHaveAttribute("aria-orientation", horizontal ? "horizontal" : "vertical");

  const model = page.getByRole("tab", { name: "Model" });
  await model.focus();
  await model.press(horizontal ? "ArrowRight" : "ArrowDown");
  const config = page.getByRole("tab", { name: "Config" });
  await expect(config).toBeFocused();
  await expect(config).toHaveAttribute("aria-selected", "true");
  await expect(page.locator('[role="tab"][tabindex="0"]')).toHaveCount(1);
});

test("graph Home, End and arrow keys follow the server walk order", async ({ page }) => {
  await gotoWorkbench(page);
  const graph = page.getByRole("group", { name: "Signal path" });
  const globalSignal = graph.getByRole("button", { name: "Edit global_signal" });
  const filters = graph.getByRole("button", { name: "Edit filters" });
  await globalSignal.focus();
  await globalSignal.press("End");
  await expect(filters).toBeFocused();
  await expect(filters).toHaveAttribute("aria-pressed", "true");
  await filters.press("Home");
  await expect(globalSignal).toBeFocused();
  await globalSignal.press("ArrowRight");
  await expect(graph.getByRole("button", { name: "Edit foregrounds" })).toBeFocused();
  await expect(graph.locator('[role="button"][tabindex="0"]')).toHaveCount(1);
});

test("Campaign remains keyboard reachable even while unavailable", async ({ page }) => {
  await gotoWorkbench(page);
  await openWorkspace(page, "Config");
  const campaign = page.getByRole("button", { name: /^Campaign — Unavailable/ });
  await expect(campaign).toBeEnabled();
  await campaign.focus();
  await expect(campaign).toBeFocused();
  await campaign.press("Enter");
  const form = page.getByRole("region", { name: "Campaign form" });
  await expect(form).toBeVisible();
  await expect(form).toHaveAttribute("aria-disabled", "true");
  await expect(form).toContainText("Reserved for capability 4 (streaming evidence).");
});

test("YAML drawer traps focus and restores its opener", async ({ page }) => {
  await gotoWorkbench(page);
  const opener = page.getByRole("button", { name: "YAML", exact: true });
  await opener.focus();
  await opener.click();
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  const close = drawer.getByRole("button", { name: "Close YAML drawer" });
  const source = drawer.getByRole("textbox", { name: "YAML source of truth" });
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(source).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(drawer).toHaveCount(0);
  await expect(opener).toBeFocused();
});
