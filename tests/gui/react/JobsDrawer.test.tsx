import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { JobsDrawer } from "../../../src/rheplicant/gui/react/JobsDrawer";
import type { JobPollingState } from "../../../src/rheplicant/gui/react/useJobPolling";
import type { JobProjection } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

function polling(): JobPollingState {
  return { status: "idle", error: null, nextRetryMs: null, refreshNow: vi.fn() };
}

// job-a is submitted first, job-b second: the backend orders by submission, never by completion.
const RUNNING_A: JobProjection = {
  job_id: "job-a",
  session_id: "session-1",
  kind: "run",
  revision: 7,
  yaml_digest: "digest-7",
  status: "running",
  result: null,
  message: null,
  stale: false,
};
const SUCCEEDED_B: JobProjection = {
  ...RUNNING_A,
  job_id: "job-b",
  kind: "validate",
  status: "succeeded",
};
const RUNNING_C: JobProjection = { ...RUNNING_A, job_id: "job-c", kind: "benchmark" };

function evidence(label: string) {
  const visibleLabel = screen.getByText(label);
  expect(visibleLabel).toBeVisible();
  return visibleLabel.closest("[role]") as HTMLElement;
}

describe("jobs drawer tracks the real latest terminal transition", () => {
  it("shows the job that just completed, not the one that is merely array-latest", async () => {
    // job-b finished before job-a was submitted, so it sits last in submission order while job-a
    // is the genuinely newer completion. Reading `.at(-1)` shows job-b and hides the news.
    const view = render(<JobsDrawer jobs={[RUNNING_A, SUCCEEDED_B]} {...polling()} />);
    expect(evidence("Running run at revision 7")).toHaveAttribute("role", "status");
    expect(evidence("Current validate succeeded at revision 7")).toHaveClass("status-success");

    view.rerender(
      <JobsDrawer jobs={[{ ...RUNNING_A, status: "succeeded" }, SUCCEEDED_B]} {...polling()} />,
    );

    const completed = await screen.findByText("Current run succeeded at revision 7");
    expect(completed).toBeVisible();
    expect(completed.closest("[role]")).toHaveAttribute("role", "status");
    expect(completed.closest("[role]")).toHaveClass("status-success");
    await waitFor(() => expect(screen.queryByText("Current validate succeeded at revision 7"))
      .not.toBeInTheDocument());
    const drawer = screen.getByRole("region", { name: "Jobs" });
    expect(drawer).toHaveTextContent("job-a");
    expect(drawer).not.toHaveTextContent("job-b");
    // An ordinary completion is news, not an emergency: it is shown, but it does not alert.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("prefers a same-batch danger transition to an ordinary one that lands later in the array", async () => {
    const runningB = { ...SUCCEEDED_B, status: "running" as const };
    const view = render(<JobsDrawer jobs={[RUNNING_A, runningB]} {...polling()} />);

    view.rerender(<JobsDrawer
      jobs={[
        { ...RUNNING_A, status: "refused", message: "batched refusal" },
        { ...runningB, status: "succeeded" },
      ]}
      {...polling()}
    />);

    const refusal = await screen.findByText("Refused run at revision 7: batched refusal");
    expect(refusal).toBeVisible();
    await waitFor(() => expect(refusal.closest("[role]")).toHaveAttribute("role", "alert"));
    expect(refusal.closest("[role]")).toHaveClass("status-danger");
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.queryByText("Current validate succeeded at revision 7"))
      .not.toBeInTheDocument();
  });

  it("stands the alert down for a later ordinary completion while still showing it", async () => {
    // job-b stays array-latest throughout, so nothing here can be satisfied by array position.
    const runningC = RUNNING_C;
    const refusedA = { ...RUNNING_A, status: "refused" as const, message: "first refusal" };
    const view = render(
      <JobsDrawer jobs={[RUNNING_A, runningC, SUCCEEDED_B]} {...polling()} />,
    );
    expect(evidence("Current validate succeeded at revision 7")).toHaveClass("status-success");

    view.rerender(<JobsDrawer jobs={[refusedA, runningC, SUCCEEDED_B]} {...polling()} />);
    const refusal = await screen.findByText("Refused run at revision 7: first refusal");
    await waitFor(() => expect(refusal.closest("[role]")).toHaveAttribute("role", "alert"));

    view.rerender(<JobsDrawer
      jobs={[refusedA, { ...runningC, status: "succeeded" }, SUCCEEDED_B]}
      {...polling()}
    />);

    const completed = await screen.findByText("Current benchmark succeeded at revision 7");
    expect(completed).toBeVisible();
    expect(completed.closest("[role]")).toHaveAttribute("role", "status");
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.queryByText("Refused run at revision 7: first refusal")).not.toBeInTheDocument();
    expect(screen.queryByText("Current validate succeeded at revision 7")).not.toBeInTheDocument();
    const drawer = screen.getByRole("region", { name: "Jobs" });
    expect(drawer).toHaveTextContent("job-c");
    expect(drawer).not.toHaveTextContent("job-b");
  });

  it("stands down when a job_id comes back at a status it never alerted on", async () => {
    // The identity the drawer remembers is job_id AND status. job-a refused, so the drawer
    // alerted; the same id then reappears RUNNING — a resubmission that reuses it, or a poll that
    // raced the retirement. Remembering the id alone, the drawer still believes "job-a" is the
    // refusal it alerted on and announces a running job through role="alert".
    const refusedA = { ...RUNNING_A, status: "refused" as const, message: "first refusal" };
    const view = render(<JobsDrawer jobs={[RUNNING_A]} {...polling()} />);

    view.rerender(<JobsDrawer jobs={[refusedA]} {...polling()} />);
    const refusal = await screen.findByText("Refused run at revision 7: first refusal");
    await waitFor(() => expect(refusal.closest("[role]")).toHaveAttribute("role", "alert"));

    view.rerender(<JobsDrawer jobs={[RUNNING_A]} {...polling()} />);

    const running = await screen.findByText("Running run at revision 7");
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(running.closest("[role]")).toHaveAttribute("role", "status");
    expect(screen.queryByText("Refused run at revision 7: first refusal")).not.toBeInTheDocument();
  });

  it("keeps alerting the same terminal row across a poll that changed nothing", async () => {
    // The other side of the retention rule: an unchanged poll is not a reason to forget. Without
    // it every refusal would be announced once and then stand itself down on the next tick.
    const refusedA = { ...RUNNING_A, status: "refused" as const, message: "first refusal" };
    const view = render(<JobsDrawer jobs={[RUNNING_A, SUCCEEDED_B]} {...polling()} />);

    view.rerender(<JobsDrawer jobs={[refusedA, SUCCEEDED_B]} {...polling()} />);
    const refusal = await screen.findByText("Refused run at revision 7: first refusal");
    await waitFor(() => expect(refusal.closest("[role]")).toHaveAttribute("role", "alert"));

    // A fresh array with identical contents: exactly what the poller hands over every tick.
    view.rerender(<JobsDrawer jobs={[{ ...refusedA }, { ...SUCCEEDED_B }]} {...polling()} />);

    expect(screen.getByText("Refused run at revision 7: first refusal").closest("[role]"))
      .toHaveAttribute("role", "alert");
    expect(screen.queryByText("Current validate succeeded at revision 7")).not.toBeInTheDocument();
  });

  it("stops believing in a job the server has retired, and hears it as news if it returns", async () => {
    // A job put in, taken out, and put back. While it is gone nothing on screen may still be
    // sourced from it, and its return is a fresh transition rather than a memory the drawer never
    // let go of — job-b stays array-latest throughout, so array position can satisfy none of this.
    const refusedA = { ...RUNNING_A, status: "refused" as const, message: "first refusal" };
    const view = render(<JobsDrawer jobs={[RUNNING_A, SUCCEEDED_B]} {...polling()} />);

    view.rerender(<JobsDrawer jobs={[refusedA, SUCCEEDED_B]} {...polling()} />);
    const refusal = await screen.findByText("Refused run at revision 7: first refusal");
    await waitFor(() => expect(refusal.closest("[role]")).toHaveAttribute("role", "alert"));

    view.rerender(<JobsDrawer jobs={[SUCCEEDED_B]} {...polling()} />);

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.queryByText("Refused run at revision 7: first refusal")).not.toBeInTheDocument();
    const drawer = screen.getByRole("region", { name: "Jobs" });
    expect(drawer).not.toHaveTextContent("job-a");
    expect(drawer).toHaveTextContent("job-b");
    expect(evidence("Current validate succeeded at revision 7")).toHaveClass("status-success");

    view.rerender(<JobsDrawer jobs={[refusedA, SUCCEEDED_B]} {...polling()} />);

    const returned = await screen.findByText("Refused run at revision 7: first refusal");
    await waitFor(() => expect(returned.closest("[role]")).toHaveAttribute("role", "alert"));
    expect(screen.queryByText("Current validate succeeded at revision 7")).not.toBeInTheDocument();
  });

  it("shows a retained terminal row rather than the array-latest one, and shows only terminal rows", async () => {
    // What the retained-evidence lookup is for: job-b is array-latest and terminal throughout, so
    // preferring job-a can only come from the remembered transition. Every row the drawer shows
    // beside the active ones is terminal — a queued or running row has no business being reported
    // as the latest completion.
    const view = render(<JobsDrawer jobs={[RUNNING_A, RUNNING_C, SUCCEEDED_B]} {...polling()} />);
    view.rerender(<JobsDrawer
      jobs={[{ ...RUNNING_A, status: "succeeded" }, RUNNING_C, SUCCEEDED_B]}
      {...polling()}
    />);

    await screen.findByText("Current run succeeded at revision 7");
    const drawer = screen.getByRole("region", { name: "Jobs" });
    expect(drawer).toHaveTextContent("job-a");
    expect(drawer).toHaveTextContent("job-c");
    expect(drawer).not.toHaveTextContent("job-b");
    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(rows.map((row) => row.textContent?.includes("running") ?? false))
      .toEqual([true, false]);
  });

  it("falls back to the array-latest terminal row when no transition has been observed", () => {
    // On a first mount every status is already known, so §7.4's bound is all the drawer can honour.
    render(<JobsDrawer
      jobs={[{ ...RUNNING_A, status: "succeeded" }, RUNNING_C, SUCCEEDED_B]}
      {...polling()}
    />);

    expect(evidence("Current validate succeeded at revision 7")).toHaveClass("status-success");
    expect(evidence("Running benchmark at revision 7")).toHaveAttribute("role", "status");
    expect(screen.queryByText("Current run succeeded at revision 7")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
