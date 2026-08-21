import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { JobPollProjection, JobProjection } from "./types";

const RETRY_MS = [1000, 2000, 4000, 8000, 10000] as const;

interface JobPollingOptions {
  sessionId: string;
  revision: number;
  yamlDigest: string;
  jobs: JobProjection[];
  refresh(signal: AbortSignal): Promise<JobPollProjection>;
  onJobs(jobs: JobProjection[]): void;
}

export interface JobPollingState {
  status: "idle" | "polling" | "retrying";
  error: string | null;
  nextRetryMs: number | null;
  refreshNow(): void;
}

function isActive(job: JobProjection) {
  return job.status === "queued" || job.status === "running";
}

export function useJobPolling({
  sessionId,
  revision,
  yamlDigest,
  jobs,
  refresh,
  onJobs,
}: JobPollingOptions): JobPollingState {
  const [state, setState] = useState<Omit<JobPollingState, "refreshNow">>({
    status: "idle",
    error: null,
    nextRetryMs: null,
  });
  const [requestToken, setRequestToken] = useState(0);
  const requestSequence = useRef(0);
  const refreshNow = useCallback(() => setRequestToken((value) => value + 1), []);
  const activeKey = useMemo(() => jobs
    .filter(isActive)
    .map((job) => `${job.job_id}:${job.status}`)
    .sort()
    .join("|"), [jobs]);

  useEffect(() => {
    let cancelled = false;
    let inFlight = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const controller = new AbortController();
    const identity = { sessionId, revision, yamlDigest };

    const clearTimer = () => {
      if (timer === null) return;
      clearTimeout(timer);
      timer = null;
    };
    const schedule = (delay: number) => {
      if (cancelled) return;
      clearTimer();
      timer = setTimeout(() => {
        timer = null;
        void poll();
      }, delay);
    };
    const poll = async () => {
      if (cancelled || inFlight || document.hidden) return;
      inFlight = true;
      const sequence = ++requestSequence.current;
      setState({ status: "polling", error: null, nextRetryMs: null });
      try {
        const found = await refresh(controller.signal);
        if (cancelled || sequence !== requestSequence.current) return;
        if (
          found.session_id !== identity.sessionId
          || found.revision !== identity.revision
          || found.yaml_digest !== identity.yamlDigest
        ) {
          throw new Error(
            "Job refresh identity no longer matches this accepted session; refresh the session.",
          );
        }
        failures = 0;
        const active = found.jobs.some(isActive);
        setState({
          status: active ? "polling" : "idle",
          error: null,
          nextRetryMs: null,
        });
        onJobs(found.jobs);
        if (active) schedule(RETRY_MS[0]);
      } catch (error) {
        if (cancelled || sequence !== requestSequence.current) return;
        const delay = RETRY_MS[Math.min(failures, RETRY_MS.length - 1)];
        failures += 1;
        setState({
          status: "retrying",
          error: error instanceof Error ? error.message : String(error),
          nextRetryMs: delay,
        });
        schedule(delay);
      } finally {
        inFlight = false;
      }
    };
    const visibility = () => {
      if (!document.hidden) {
        clearTimer();
        void poll();
      }
    };

    document.addEventListener("visibilitychange", visibility);
    void poll();
    return () => {
      cancelled = true;
      requestSequence.current += 1;
      controller.abort();
      clearTimer();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [sessionId, revision, yamlDigest, activeKey, refresh, onJobs, requestToken]);

  return { ...state, refreshNow };
}
