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

interface PageAudit {
  readonly consoleErrors: string[];
  readonly pageErrors: string[];
  readonly failedRequests: string[];
  readonly httpErrors: HttpError[];
  readonly expectedHttpErrors: ExpectedHttpError[];
}

interface HttpError {
  readonly method: string;
  readonly path: string;
  readonly status: number;
  readonly statusText: string;
}

interface ExpectedHttpError {
  readonly method: string;
  readonly path: string;
  readonly status: number;
}

const pageAudits = new WeakMap<Page, PageAudit>();

export function beginPageAudit(page: Page): void {
  if (pageAudits.has(page)) throw new Error("Page audit was started twice.");
  const audit: PageAudit = {
    consoleErrors: [],
    pageErrors: [],
    failedRequests: [],
    httpErrors: [],
    expectedHttpErrors: [],
  };
  pageAudits.set(page, audit);
  page.on("console", (message) => {
    if (message.type() === "error") audit.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => audit.pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    audit.failedRequests.push(
      `${request.method()} ${request.url()}: ${request.failure()?.errorText ?? "unknown failure"}`,
    );
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    audit.httpErrors.push({
      method: response.request().method(),
      path: new URL(response.url()).pathname,
      status: response.status(),
      statusText: response.statusText(),
    });
  });
}

export function expectHandledHttpError(
  page: Page,
  method: string,
  path: string,
  status: number,
): void {
  const audit = pageAudits.get(page);
  if (audit === undefined) throw new Error("Page audit was not started.");
  audit.expectedHttpErrors.push({ method, path, status });
}

export async function assertPageAudit(page: Page): Promise<void> {
  const audit = pageAudits.get(page);
  if (audit === undefined) throw new Error("Page audit was not started.");
  const unmatchedHttp = [...audit.httpErrors];
  const unmatchedConsole = [...audit.consoleErrors];
  for (const expected of audit.expectedHttpErrors) {
    const at = unmatchedHttp.findIndex((found) => (
      found.method === expected.method
      && found.path === expected.path
      && found.status === expected.status
    ));
    expect(at, `handled HTTP ${expected.method} ${expected.path} ${expected.status}`)
      .toBeGreaterThanOrEqual(0);
    const [found] = unmatchedHttp.splice(at, 1);
    const browserMessage = `Failed to load resource: the server responded with a status of ${found.status} (${found.statusText})`;
    const consoleAt = unmatchedConsole.indexOf(browserMessage);
    expect(consoleAt, `Chromium console evidence for ${expected.status}`)
      .toBeGreaterThanOrEqual(0);
    unmatchedConsole.splice(consoleAt, 1);
  }
  expect(unmatchedHttp, "unexpected HTTP error responses").toEqual([]);
  expect(unmatchedConsole, "unexpected browser console errors").toEqual([]);
  expect(audit.pageErrors, "uncaught browser errors").toEqual([]);
  expect(audit.failedRequests, "failed browser requests").toEqual([]);
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

export interface CreatedSessionIdentity {
  readonly session_id: string;
  readonly revision: number;
  readonly document: { readonly yaml_text: string };
}

export async function gotoWorkbenchSession(page: Page): Promise<CreatedSessionIdentity> {
  const created = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && new URL(response.url()).pathname === "/api/sessions"
    && response.status() === 201
  ));
  await gotoWorkbench(page);
  return await (await created).json() as CreatedSessionIdentity;
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

export async function closeYamlDrawer(page: Page): Promise<void> {
  const drawer = page.getByRole("dialog", { name: "YAML drawer" });
  await drawer.getByRole("button", { name: "Close YAML drawer" }).click();
  await expect(drawer).toHaveCount(0);
}

export async function openContextInspector(page: Page): Promise<Locator> {
  const summary = page.locator("summary").filter({ hasText: "Context inspector" });
  const inspector = page.locator("details").filter({ has: summary });
  if (await inspector.getAttribute("open") === null) await summary.click();
  await expect(inspector).toHaveAttribute("open", "");
  return inspector;
}

export async function closeContextInspector(page: Page): Promise<void> {
  const summary = page.locator("summary").filter({ hasText: "Context inspector" });
  const inspector = page.locator("details").filter({ has: summary });
  if (await summary.isVisible() && await inspector.getAttribute("open") !== null) {
    await summary.click();
    await expect(inspector).not.toHaveAttribute("open", "");
  }
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

export async function waitForSuccess(page: Page, kind: string): Promise<Locator> {
  const item = await waitForTerminal(page, kind);
  const escapedKind = escapeRegularExpression(kind);
  await expect(item).toContainText(
    new RegExp(`·\\s*${escapedKind}\\s*·\\s*succeeded\\b`, "i"),
  );
  return item;
}

export async function submitTrustedJob(page: Page, label: string): Promise<void> {
  await page.getByRole("button", { name: label, exact: true }).click();
  const confirmation = page.getByRole("dialog", { name: "Trusted execution" });
  if (await confirmation.count() > 0) {
    await confirmation.getByRole("button", { name: "I understand, continue" }).click();
  }
}

export async function activeWorkspaceTabStops(page: Page): Promise<number> {
  return await page.getByRole("tabpanel").evaluate((panel) => {
    const selector = [
      "a[href]",
      "button",
      "input",
      "select",
      "textarea",
      "summary",
      '[role="button"]',
      '[tabindex]:not([tabindex="-1"])',
    ].join(",");
    return Array.from(panel.querySelectorAll<HTMLElement>(selector)).filter((element) => {
      if (element.closest("[inert]") || element.matches(":disabled")) return false;
      if (element.getAttribute("tabindex") === "-1") return false;
      const style = getComputedStyle(element);
      return style.display !== "none"
        && style.visibility !== "hidden"
        && element.getClientRects().length > 0;
    }).length;
  });
}

export async function documentOverflow(page: Page): Promise<{
  delta: number;
  bodyDelta: number;
  verticalDelta: number;
  bodyVerticalDelta: number;
}> {
  return await page.evaluate(() => ({
    delta: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    bodyDelta: document.body.scrollWidth - document.body.clientWidth,
    verticalDelta: document.documentElement.scrollHeight - document.documentElement.clientHeight,
    bodyVerticalDelta: document.body.scrollHeight - document.body.clientHeight,
  }));
}

export async function headerActionsOverflow(page: Page): Promise<{
  top: number;
  right: number;
  bottom: number;
  left: number;
}> {
  return await page.getByRole("navigation", { name: "History and file actions" })
    .evaluate((actions) => {
      const header = actions.closest<HTMLElement>(".workbench-header");
      if (header === null) throw new Error("Header actions have no workbench header.");
      const headerBox = header.getBoundingClientRect();
      const actionsBox = actions.getBoundingClientRect();
      const style = getComputedStyle(header);
      const inner = {
        top: headerBox.top + Number.parseFloat(style.borderTopWidth),
        right: headerBox.right - Number.parseFloat(style.borderRightWidth),
        bottom: headerBox.bottom - Number.parseFloat(style.borderBottomWidth),
        left: headerBox.left + Number.parseFloat(style.borderLeftWidth),
      };
      return {
        top: Math.max(0, inner.top - actionsBox.top),
        right: Math.max(0, actionsBox.right - inner.right),
        bottom: Math.max(0, actionsBox.bottom - inner.bottom),
        left: Math.max(0, inner.left - actionsBox.left),
      };
    });
}

export async function activeMainUsability(page: Page): Promise<{
  usableHeight: number;
  nextActionOverflow: { top: number; right: number; bottom: number; left: number } | null;
}> {
  return await page.getByRole("tabpanel").evaluate((panel) => {
    const main = panel.closest<HTMLElement>(".workbench-main");
    if (main === null) throw new Error("Active workspace has no workbench main owner.");
    const mainBox = main.getBoundingClientRect();
    const style = getComputedStyle(main);
    const inner = {
      top: mainBox.top + Number.parseFloat(style.borderTopWidth)
        + Number.parseFloat(style.paddingTop),
      right: mainBox.right - Number.parseFloat(style.borderRightWidth)
        - Number.parseFloat(style.paddingRight),
      bottom: mainBox.bottom - Number.parseFloat(style.borderBottomWidth)
        - Number.parseFloat(style.paddingBottom),
      left: mainBox.left + Number.parseFloat(style.borderLeftWidth)
        + Number.parseFloat(style.paddingLeft),
    };
    const nextAction = Array.from(main.querySelectorAll("p")).find(
      (paragraph) => paragraph.textContent?.trim() === "Next: run Preview forward.",
    );
    const nextBox = nextAction?.getBoundingClientRect();
    return {
      usableHeight: inner.bottom - inner.top,
      nextActionOverflow: nextBox === undefined
        ? null
        : {
          top: Math.max(0, inner.top - nextBox.top),
          right: Math.max(0, nextBox.right - inner.right),
          bottom: Math.max(0, nextBox.bottom - inner.bottom),
          left: Math.max(0, inner.left - nextBox.left),
        },
    };
  });
}

interface Colour {
  readonly red: number;
  readonly green: number;
  readonly blue: number;
  readonly alpha: number;
}

function parseColour(value: string): Colour {
  const hex = value.trim().match(/^#([0-9a-f]{3,8})$/i)?.[1];
  if (hex !== undefined) {
    const expanded = hex.length === 3 || hex.length === 4
      ? [...hex].map((digit) => `${digit}${digit}`).join("")
      : hex;
    if (expanded.length === 6 || expanded.length === 8) {
      return {
        red: Number.parseInt(expanded.slice(0, 2), 16),
        green: Number.parseInt(expanded.slice(2, 4), 16),
        blue: Number.parseInt(expanded.slice(4, 6), 16),
        alpha: expanded.length === 8 ? Number.parseInt(expanded.slice(6, 8), 16) / 255 : 1,
      };
    }
  }
  const functional = value.match(/^rgba?\(([^)]+)\)$/i)?.[1];
  if (functional !== undefined) {
    const channels = functional.replace("/", " ").split(/[ ,]+/).filter(Boolean).map(Number);
    if (channels.length === 3 || channels.length === 4) {
      return {
        red: channels[0],
        green: channels[1],
        blue: channels[2],
        alpha: channels[3] ?? 1,
      };
    }
  }
  throw new Error(`Unsupported computed colour ${JSON.stringify(value)}.`);
}

function composite(foreground: Colour, background: Colour): Colour {
  const alpha = foreground.alpha + background.alpha * (1 - foreground.alpha);
  if (alpha === 0) return { red: 0, green: 0, blue: 0, alpha: 0 };
  const channel = (front: number, back: number) => (
    (front * foreground.alpha + back * background.alpha * (1 - foreground.alpha)) / alpha
  );
  return {
    red: channel(foreground.red, background.red),
    green: channel(foreground.green, background.green),
    blue: channel(foreground.blue, background.blue),
    alpha,
  };
}

function luminance(colour: Colour): number {
  const linear = (channel: number) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(colour.red)
    + 0.7152 * linear(colour.green)
    + 0.0722 * linear(colour.blue);
}

function contrast(foreground: Colour, background: Colour): number {
  const front = luminance(foreground);
  const back = luminance(background);
  return (Math.max(front, back) + 0.05) / (Math.min(front, back) + 0.05);
}

function contrastFromStrings(foreground: string, backgrounds: string[]): number {
  const canvas = { red: 255, green: 255, blue: 255, alpha: 1 };
  const background = backgrounds
    .map(parseColour)
    .reverse()
    .reduce((behind, layer) => composite(layer, behind), canvas);
  return contrast(composite(parseColour(foreground), background), background);
}

export async function cssVariableContrast(
  page: Page,
  foreground: string,
  background: string,
): Promise<number> {
  const values = await page.evaluate(([frontName, backName]) => {
    const style = getComputedStyle(document.documentElement);
    return [style.getPropertyValue(frontName).trim(), style.getPropertyValue(backName).trim()];
  }, [foreground, background]);
  return contrastFromStrings(values[0], [values[1]]);
}

export async function expectTextContrast(locator: Locator, minimum: number): Promise<void> {
  const evidence = await locator.evaluate((element) => {
    const backgrounds: string[] = [];
    for (let current: Element | null = element; current !== null; current = current.parentElement) {
      backgrounds.push(getComputedStyle(current).backgroundColor);
    }
    return { foreground: getComputedStyle(element).color, backgrounds };
  });
  expect(contrastFromStrings(evidence.foreground, evidence.backgrounds))
    .toBeGreaterThanOrEqual(minimum);
}

export async function svgTextContrast(text: Locator, surface: Locator): Promise<number> {
  const foreground = await text.evaluate((element) => getComputedStyle(element).fill);
  const background = await surface.evaluate((element) => getComputedStyle(element).fill);
  return contrastFromStrings(foreground, [background]);
}
