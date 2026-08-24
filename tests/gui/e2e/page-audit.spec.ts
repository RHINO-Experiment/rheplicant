import { COMPLETE_YAML, expect, test } from "./fixtures";
import {
  assertPageAudit,
  auditedCancelledPolls,
  auditedFailures,
  beginPageAudit,
  gotoWorkbench,
  loadYaml,
} from "./helpers";

/**
 * The audit helper judged against itself.
 *
 * ``assertPageAudit`` used to fail any ``requestfailed``, and the job poller
 * cancels its own in-flight request whenever the session identity or the set of
 * active jobs changes -- which Chromium reports as ``net::ERR_ABORTED``, the
 * same error text a real network failure would carry. That made the whole e2e
 * suite flaky in exactly one place: a cancel needs an identity change to land
 * while a poll is still open, which needs a slow enough server, which is why it
 * appeared once on a loaded 14-worker run and never on an idle one.
 *
 * These three tests are the exemption in both directions. Without the middle
 * and last ones it would be indistinguishable from "ignore aborts".
 */
test.beforeEach(({}, testInfo) => {
  test.skip(
    testInfo.project.name !== "wide",
    "the audit reads requests, not pixels -- one viewport answers it",
  );
});

/**
 * Long enough that the identity change below always lands first.
 *
 * The route FULFILS rather than continuing: a poll held open against the real
 * server holds a server connection too, and this suite runs at fourteen
 * workers against one launcher. The response is never read -- the poller
 * aborts the request before this resolves -- so its body does not matter, and
 * keeping the server out of it keeps this test from slowing its neighbours.
 */
const HOLD_THE_POLL_OPEN_MS = 10_000;

test("a jobs poll the page supersedes is a cancel, not a failure", async ({ page }) => {
  beginPageAudit(page);
  await page.route("**/api/sessions/*/jobs*", async (route) => {
    await new Promise((resume) => setTimeout(resume, HOLD_THE_POLL_OPEN_MS));
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });

  await gotoWorkbench(page);
  await loadYaml(page, COMPLETE_YAML);
  await page.waitForTimeout(500);

  // The cancel really happened -- otherwise the assertion below would pass on
  // a run that never exercised the exemption at all.
  expect(auditedCancelledPolls(page).length).toBeGreaterThan(0);
  expect(auditedCancelledPolls(page)[0]).toContain("net::ERR_ABORTED");
  expect(auditedFailures(page)).toEqual([]);
  await assertPageAudit(page);
});

test("an aborted request that is not a jobs poll is still a failure", async ({ page }) => {
  beginPageAudit(page);
  await gotoWorkbench(page);

  // The page cancels a fetch of its own, to an endpoint that is not the poll.
  await page.evaluate(async () => {
    const controller = new AbortController();
    const pending = fetch("/api/sessions", { signal: controller.signal });
    controller.abort();
    await pending.catch(() => undefined);
  });
  await expect.poll(() => auditedFailures(page).length).toBeGreaterThan(0);

  expect(auditedFailures(page)[0]).toContain("net::ERR_ABORTED");
  expect(auditedFailures(page)[0]).toContain("/api/sessions");
  expect(auditedCancelledPolls(page)).toEqual([]);
});

test("a jobs poll that fails for any other reason is still a failure", async ({ page }) => {
  beginPageAudit(page);
  await page.route("**/api/sessions/*/jobs*", (route) => route.abort("connectionfailed"));

  await gotoWorkbench(page);
  await expect.poll(() => auditedFailures(page).length).toBeGreaterThan(0);

  expect(auditedFailures(page)[0]).toContain("/jobs");
  expect(auditedFailures(page)[0]).not.toContain("net::ERR_ABORTED");
  expect(auditedCancelledPolls(page)).toEqual([]);
});
