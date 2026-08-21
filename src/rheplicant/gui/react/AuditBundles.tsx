import type { JobProjection } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function auditFiles(job: JobProjection) {
  if (!isRecord(job.result) || !isRecord(job.result.output)) return [];
  const files = job.result.output.audit_files;
  if (!Array.isArray(files)) return [];
  return files.filter((file): file is string => typeof file === "string");
}

function artifactHref(job: JobProjection, file: string) {
  return `/api/sessions/${encodeURIComponent(job.session_id)}/jobs/${encodeURIComponent(job.job_id)}/artifacts/${encodeURIComponent(file)}`;
}

export function AuditBundles({ job }: { job: JobProjection }) {
  const files = auditFiles(job);
  return (
    <section aria-label="Completed audit bundles">
      <h2>Audit bundle</h2>
      {files.length === 0 ? <p>No published audit bundle.</p> : (
        <ul>{files.map((file) => (
          <li key={file}>
            <a href={artifactHref(job, file)} target="_blank" rel="noreferrer">{file}</a>
          </li>
        ))}</ul>
      )}
    </section>
  );
}
