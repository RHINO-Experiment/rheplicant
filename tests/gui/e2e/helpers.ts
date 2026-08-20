import { expect, type Locator, type Page } from "./fixtures";

export const WORKBENCH_TIMEOUT_MS = 15_000;
export const JOB_TERMINAL_TIMEOUT_MS = 60_000;

export const DETERMINISTIC_MOTION_CSS = `
*, *::before, *::after {
  animation-delay: 0s !important;
  animation-duration: 0.001ms !important;
  animation-iteration-count: 1 !important;
  caret-color: transparent !important;
  scroll-behavior: auto !important;
  transition-delay: 0s !important;
  transition-duration: 0.001ms !important;
}
`;

function escapeRegularExpression(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function installDeterministicMotion(page: Page): Promise<void> {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addStyleTag({ content: DETERMINISTIC_MOTION_CSS });
}

export async function gotoWorkbench(page: Page): Promise<void> {
  await page.goto("/", {
    waitUntil: "domcontentloaded",
    timeout: WORKBENCH_TIMEOUT_MS,
  });
  await expect(page.getByRole("tab", { name: "Model" })).toBeVisible({
    timeout: WORKBENCH_TIMEOUT_MS,
  });
  await installDeterministicMotion(page);
}

export async function openWorkspace(
  page: Page,
  name: "Model" | "Config" | "Execute" | "Results",
): Promise<void> {
  const tab = page.getByRole("tab", { name, exact: true });
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true", {
    timeout: WORKBENCH_TIMEOUT_MS,
  });
}

export async function openYaml(page: Page): Promise<Locator> {
  await page.getByRole("button", { name: "YAML", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  await expect(drawer).toBeVisible({ timeout: WORKBENCH_TIMEOUT_MS });
  return drawer;
}

export async function replaceYamlDraft(page: Page, yaml: string): Promise<void> {
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  await drawer.getByRole("textbox", { name: "YAML source of truth" }).fill(yaml);
}

export async function loadYaml(page: Page, yaml: string): Promise<void> {
  const drawer = await openYaml(page);
  await replaceYamlDraft(page, yaml);
  await drawer.getByRole("button", { name: "Apply YAML edit" }).click();
  await expect(drawer.getByText(/^Accepted revision [1-9][0-9]*$/)).toBeVisible({
    timeout: WORKBENCH_TIMEOUT_MS,
  });
}

export async function waitForTerminal(page: Page, kind: string): Promise<Locator> {
  const jobs = page.getByRole("region", { name: "Jobs" });
  const item = jobs
    .getByRole("listitem")
    .filter({ hasText: new RegExp(escapeRegularExpression(kind), "i") })
    .first();
  await expect(item).toContainText(/succeeded|refused|error/i, {
    timeout: JOB_TERMINAL_TIMEOUT_MS,
  });
  return item;
}
