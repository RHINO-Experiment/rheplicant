export function SecurityBoundaryNotice() {
  return (
    <details className="security-boundary">
      <summary>
        Trusted YAML: plugins, python targets, paths and jobs run as the server account.
      </summary>
      <p>No authentication, tenant isolation or sandbox is provided.</p>
      <p>
        Paths read and write the server filesystem; jobs may consume CPU,
        accelerator time and wall time.
      </p>
      <p>Remote binding is explicit acknowledgement, not protection.</p>
    </details>
  );
}
