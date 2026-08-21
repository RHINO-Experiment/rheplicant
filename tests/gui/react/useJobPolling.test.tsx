import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useJobPolling } from "../../../src/rheplicant/gui/react/useJobPolling";
import { sessionTransport } from "../../../src/rheplicant/gui/react/api";
import type {
  JobPollProjection,
  JobProjection,
} from "../../../src/rheplicant/gui/react/types";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
  });
}

function job(
  status: JobProjection["status"] = "running",
  overrides: Partial<JobProjection> = {},
): JobProjection {
  return {
    job_id: "job-1",
    session_id: "session-1",
    kind: "run",
    revision: 4,
    yaml_digest: "digest-4",
    status,
    result: null,
    message: null,
    stale: false,
    ...overrides,
  };
}

function pollAt(
  revision = 4,
  yamlDigest = `digest-${revision}`,
  jobs: JobProjection[] = [job()],
  sessionId = "session-1",
): JobPollProjection {
  return {
    session_id: sessionId,
    revision,
    yaml_digest: yamlDigest,
    jobs,
  };
}

interface HookSession {
  sessionId: string;
  revision: number;
  yamlDigest: string;
  jobs: JobProjection[];
}

function sessionAt(
  revision = 4,
  yamlDigest = `digest-${revision}`,
  jobs: JobProjection[] = [job()],
  sessionId = "session-1",
): HookSession {
  return { sessionId, revision, yamlDigest, jobs };
}

function renderPolling(
  initial: HookSession,
  refresh: (signal: AbortSignal) => Promise<JobPollProjection>,
) {
  const onJobs = vi.fn();
  const rendered = renderHook(
    ({ candidate }: { candidate: HookSession }) => useJobPolling({
      ...candidate,
      refresh,
      onJobs,
    }),
    { initialProps: { candidate: initial } },
  );
  return { ...rendered, onJobs };
}

describe("race-safe jobs polling", () => {
  it("passes the exact AbortSignal through the jobs-only transport", async () => {
    const response = pollAt(4);
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => response,
    }));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await sessionTransport.refreshJobs("session /1", controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/session%20%2F1/jobs",
      expect.objectContaining({ method: "GET", signal: controller.signal }),
    );
  });

  it("discards a late response for an older accepted revision", async () => {
    const oldRequest = deferred<JobPollProjection>();
    const newRequest = deferred<JobPollProjection>();
    const refresh = vi.fn()
      .mockReturnValueOnce(oldRequest.promise)
      .mockReturnValueOnce(newRequest.promise);
    const { rerender, onJobs } = renderPolling(sessionAt(4), refresh);

    rerender({ candidate: sessionAt(5) });
    oldRequest.resolve(pollAt(4));
    await flushPromises();

    expect(onJobs).not.toHaveBeenCalled();
  });

  it.each([
    ["session", pollAt(4, "digest-4", [job()], "session-other")],
    ["revision", pollAt(5, "digest-4")],
    ["digest", pollAt(4, "digest-other")],
  ])("reports a live %s identity mismatch without installing jobs", async (_field, found) => {
    vi.useFakeTimers();
    const refresh = vi.fn(async () => found);
    const { result, onJobs } = renderPolling(sessionAt(4), refresh);

    await flushPromises();

    expect(onJobs).not.toHaveBeenCalled();
    expect(result.current).toMatchObject({
      status: "retrying",
      error: "Job refresh identity no longer matches this accepted session; refresh the session.",
      nextRetryMs: 1000,
    });
  });

  it("aborts the exact in-flight request on unmount", () => {
    let signal: AbortSignal | undefined;
    const refresh = vi.fn((candidate: AbortSignal) => {
      signal = candidate;
      return new Promise<JobPollProjection>(() => undefined);
    });
    const { unmount } = renderPolling(sessionAt(4), refresh);

    expect(signal?.aborted).toBe(false);
    unmount();

    expect(signal?.aborted).toBe(true);
  });

  it("clears the scheduled timer and exact visibility listener on unmount", async () => {
    vi.useFakeTimers();
    const addListener = vi.spyOn(document, "addEventListener");
    const removeListener = vi.spyOn(document, "removeEventListener");
    const active = job("running");
    const refresh = vi.fn(async () => pollAt(4, "digest-4", [active]));
    const { unmount } = renderPolling(sessionAt(4, "digest-4", [active]), refresh);
    await flushPromises();

    const visibilityCall = addListener.mock.calls.find(([type]) => type === "visibilitychange");
    expect(visibilityCall).toBeDefined();
    expect(vi.getTimerCount()).toBe(1);
    unmount();

    expect(vi.getTimerCount()).toBe(0);
    expect(removeListener).toHaveBeenCalledWith("visibilitychange", visibilityCall?.[1]);
  });

  it("discards a superseded same-identity manual response", async () => {
    const oldRequest = deferred<JobPollProjection>();
    const current = pollAt(4, "digest-4", [job("running", { job_id: "current" })]);
    const refresh = vi.fn()
      .mockReturnValueOnce(oldRequest.promise)
      .mockResolvedValueOnce(current);
    const { result, onJobs } = renderPolling(sessionAt(4), refresh);

    act(() => result.current.refreshNow());
    await flushPromises();
    expect(onJobs).toHaveBeenCalledTimes(1);
    expect(onJobs).toHaveBeenLastCalledWith(current.jobs);

    oldRequest.resolve(pollAt(4, "digest-4", [job("succeeded", { job_id: "old" })]));
    await flushPromises();
    expect(onJobs).toHaveBeenCalledTimes(1);
  });

  it("backs off 1 2 4 8 10 seconds and resets after success", async () => {
    vi.useFakeTimers();
    const refresh = vi.fn().mockRejectedValue(new Error("offline"));
    const active = job("running");
    const { result, onJobs } = renderPolling(sessionAt(4, "digest-4", [active]), refresh);

    await flushPromises();
    expect(refresh).toHaveBeenCalledTimes(1);
    for (const [delay, calls] of [
      [1000, 2],
      [2000, 3],
      [4000, 4],
      [8000, 5],
      [10000, 6],
    ] as const) {
      await act(async () => vi.advanceTimersByTimeAsync(delay));
      expect(refresh).toHaveBeenCalledTimes(calls);
    }
    expect(result.current.nextRetryMs).toBe(10000);

    refresh.mockResolvedValueOnce(pollAt(4, "digest-4", [active]));
    await act(async () => vi.advanceTimersByTimeAsync(10000));
    expect(onJobs).toHaveBeenLastCalledWith([active]);

    refresh.mockClear();
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(result.current.nextRetryMs).toBe(1000);
  });

  it("pauses while hidden and resumes immediately on visibility", async () => {
    vi.useFakeTimers();
    let hidden = true;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    const active = job("running");
    const refresh = vi.fn(async () => pollAt(4, "digest-4", [active]));
    renderPolling(sessionAt(4, "digest-4", [active]), refresh);

    await flushPromises();
    expect(refresh).not.toHaveBeenCalled();
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await flushPromises();
    expect(refresh).toHaveBeenCalledTimes(1);

    refresh.mockClear();
    hidden = true;
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(refresh).not.toHaveBeenCalled();
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await flushPromises();
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("replaces an active timer when visibility triggers an immediate poll", async () => {
    vi.useFakeTimers();
    let hidden = false;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    const active = job("running");
    const refresh = vi.fn(async () => pollAt(4, "digest-4", [active]));
    const { unmount } = renderPolling(sessionAt(4, "digest-4", [active]), refresh);

    await flushPromises();
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);
    await act(async () => vi.advanceTimersByTimeAsync(500));

    hidden = true;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await flushPromises();

    expect(refresh).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(1);
    await act(async () => vi.advanceTimersByTimeAsync(999));
    expect(refresh).toHaveBeenCalledTimes(2);
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(refresh).toHaveBeenCalledTimes(3);
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(refresh).toHaveBeenCalledTimes(4);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("clears an active timer when visibility polling becomes terminal", async () => {
    vi.useFakeTimers();
    let hidden = false;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    const active = job("running");
    const terminal = job("succeeded");
    const refresh = vi.fn()
      .mockResolvedValueOnce(pollAt(4, "digest-4", [active]))
      .mockResolvedValueOnce(pollAt(4, "digest-4", [terminal]));
    renderPolling(sessionAt(4, "digest-4", [active]), refresh);

    await flushPromises();
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);
    await act(async () => vi.advanceTimersByTimeAsync(500));

    hidden = true;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    await flushPromises();

    expect(refresh).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("stops idle without scheduling after a terminal projection", async () => {
    vi.useFakeTimers();
    const terminal = job("succeeded");
    const refresh = vi.fn(async () => pollAt(4, "digest-4", [terminal]));
    const { result, onJobs } = renderPolling(sessionAt(4), refresh);

    await flushPromises();
    expect(onJobs).toHaveBeenLastCalledWith([terminal]);
    expect(result.current).toMatchObject({ status: "idle", error: null, nextRetryMs: null });
    await act(async () => vi.advanceTimersByTimeAsync(20000));
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("keeps the effect stable for fresh arrays with the same active key", () => {
    const firstRequest = deferred<JobPollProjection>();
    const secondRequest = deferred<JobPollProjection>();
    const refresh = vi.fn()
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise);
    const queued = job("queued");
    const { rerender } = renderPolling(sessionAt(4, "digest-4", [queued]), refresh);
    const firstSignal = refresh.mock.calls[0][0] as AbortSignal;

    rerender({ candidate: sessionAt(4, "digest-4", [{ ...queued }]) });
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(firstSignal.aborted).toBe(false);

    rerender({ candidate: sessionAt(4, "digest-4", [{ ...queued, status: "running" }]) });
    expect(refresh).toHaveBeenCalledTimes(2);
    expect(firstSignal.aborted).toBe(true);
  });

  it("runs an immediate manual refresh after terminal idle", async () => {
    const terminal = job("succeeded");
    const refresh = vi.fn(async () => pollAt(4, "digest-4", [terminal]));
    const { result } = renderPolling(sessionAt(4, "digest-4", [terminal]), refresh);
    await flushPromises();
    refresh.mockClear();

    act(() => result.current.refreshNow());
    await flushPromises();

    expect(refresh).toHaveBeenCalledTimes(1);
  });
});
