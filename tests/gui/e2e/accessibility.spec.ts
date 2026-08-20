import { AxeBuilder, GRAPH_VARIANT_YAML, expect, test } from "./fixtures";
import {
  assertPageAudit,
  beginPageAudit,
  closeYamlDrawer,
  cssVariableContrast,
  expectTextContrast,
  gotoWorkbench,
  loadYaml,
  svgTextContrast,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  beginPageAudit(page);
});

test.afterEach(async ({ page }) => {
  await assertPageAudit(page);
});

for (const colorScheme of ["light", "dark"] as const) {
  test(`${colorScheme} workbench has no serious or critical WCAG 2.2 AA axe findings`, async ({
    page,
  }) => {
    await page.emulateMedia({ colorScheme });
    await gotoWorkbench(page);
    const results = await new AxeBuilder({ page })
      .include(".rheplicant-editor")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    expect(results.violations.filter((violation) => (
      violation.impact === "serious" || violation.impact === "critical"
    ))).toEqual([]);
  });

  test(`${colorScheme} muted, dim and state text retain computed 4.5:1 contrast`, async ({
    page,
  }) => {
    await page.emulateMedia({ colorScheme });
    await gotoWorkbench(page);
    expect(await cssVariableContrast(page, "--rh-text-muted", "--rh-surface-work"))
      .toBeGreaterThanOrEqual(4.5);
    await expectTextContrast(page.getByRole("status").filter({ hasText: "Saved" }).first(), 4.5);
    await expectTextContrast(
      page.getByRole("status").filter({ hasText: "Validation current" }).first(),
      4.5,
    );
    await expectTextContrast(page.getByRole("button", { name: "Undo" }), 4.5);
    const dimNode = page.getByRole("tab", { name: "Model" });
    await dimNode.click();
    const foregrounds = page.getByRole("group", { name: "Signal path" })
      .getByRole("button", { name: "Edit foregrounds" });
    expect(await svgTextContrast(
      foregrounds.locator("text"),
      foregrounds.locator("rect"),
    )).toBeGreaterThanOrEqual(4.5);
  });

  test(`${colorScheme} comparison graphs expose one non-interactive fitted named image each`, async ({
    page,
  }) => {
    await page.emulateMedia({ colorScheme });
    await gotoWorkbench(page);
    await loadYaml(page, GRAPH_VARIANT_YAML);
    await closeYamlDrawer(page);
    const compare = page.getByRole("button", { name: "Compare" });
    await compare.click();

    const comparison = page.getByRole("region", { name: "Base versus selected variant" });
    const diagrams = comparison.locator('.graph-viewport[role="img"]');
    await expect(diagrams).toHaveCount(2);
    for (const diagram of await diagrams.all()) {
      await expect(diagram.locator(".graph-markup > svg")).toHaveAttribute("aria-hidden", "true");
      const bounds = await diagram.evaluate((outer) => {
        const svg = outer.querySelector<SVGSVGElement>(".graph-markup > svg");
        if (svg === null) throw new Error("Readonly comparison graph has no SVG.");
        const outerBox = outer.getBoundingClientRect();
        const svgBox = svg.getBoundingClientRect();
        return {
          horizontalOverflow: Math.max(
            0,
            outerBox.left - svgBox.left,
            svgBox.right - outerBox.right,
          ),
          verticalOverflow: Math.max(
            0,
            outerBox.top - svgBox.top,
            svgBox.bottom - outerBox.bottom,
          ),
          scrollDelta: outer.scrollWidth - outer.clientWidth,
        };
      });
      expect(bounds).toEqual({ horizontalOverflow: 0, verticalOverflow: 0, scrollDelta: 0 });
      await expect(diagram).not.toHaveAttribute("tabindex");
      for (const node of await diagram.locator("[data-node-id]").all()) {
        await expect(node).not.toHaveAttribute("role");
        await expect(node).not.toHaveAttribute("tabindex");
        await expect(node).not.toHaveAttribute("aria-pressed");
        await expect(node).not.toHaveAttribute("aria-disabled");
      }
    }
    await compare.focus();
    let returnedToCompare = false;
    for (let tab = 0; tab < 60; tab += 1) {
      await page.keyboard.press("Tab");
      expect(await diagrams.evaluateAll((elements) => elements.some((element) => (
        element === document.activeElement || element.contains(document.activeElement)
      )))).toBe(false);
      if (await compare.evaluate((element) => element === document.activeElement)) {
        returnedToCompare = true;
        break;
      }
    }
    expect(returnedToCompare).toBe(true);
    const results = await new AxeBuilder({ page })
      .include(".rheplicant-editor")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    expect(results.violations.filter((violation) => (
      violation.impact === "serious" || violation.impact === "critical"
    ))).toEqual([]);
  });
}

test("forced colours retain system boundaries and a visible focus indicator", async ({ page }) => {
  await page.emulateMedia({ forcedColors: "active" });
  await gotoWorkbench(page);
  const model = page.getByRole("tab", { name: "Model" });
  await model.focus();
  const evidence = await page.evaluate(() => {
    const root = getComputedStyle(document.documentElement);
    const selected = document.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]');
    const chip = document.querySelector<HTMLElement>(".status-chip");
    if (!selected || !chip) throw new Error("forced-colour evidence is missing");
    const focus = getComputedStyle(selected);
    const boundary = getComputedStyle(chip);
    const systemColour = (property: "color" | "borderColor", value: string) => {
      const probe = document.createElement("div");
      probe.style.forcedColorAdjust = "none";
      probe.style[property] = value;
      probe.style.borderStyle = "solid";
      document.body.append(probe);
      const resolved = property === "color"
        ? getComputedStyle(probe).color
        : getComputedStyle(probe).borderTopColor;
      probe.remove();
      return resolved;
    };
    return {
      borderToken: root.getPropertyValue("--rh-border").trim(),
      focusToken: root.getPropertyValue("--rh-focus").trim(),
      focusStyle: focus.outlineStyle,
      focusWidth: Number.parseFloat(focus.outlineWidth),
      boundaryStyle: boundary.borderStyle,
      boundaryWidth: Number.parseFloat(boundary.borderWidth),
      focusAdjust: focus.forcedColorAdjust,
      boundaryAdjust: boundary.forcedColorAdjust,
      focusColour: focus.outlineColor,
      resolvedHighlight: systemColour("color", "Highlight"),
      boundaryColour: boundary.borderTopColor,
      resolvedCanvasText: systemColour("borderColor", "CanvasText"),
    };
  });
  expect(evidence).toMatchObject({
    borderToken: "CanvasText",
    focusToken: "Highlight",
    focusStyle: "solid",
    focusWidth: 3,
    boundaryStyle: "solid",
    boundaryWidth: 2,
    focusAdjust: "auto",
    boundaryAdjust: "auto",
  });
  expect(evidence.focusColour).toBe(evidence.resolvedHighlight);
  expect(evidence.boundaryColour).toBe(evidence.resolvedCanvasText);
});
