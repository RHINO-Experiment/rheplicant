import type { ValidationProjection } from "./types";

interface Props {
  validation: ValidationProjection;
}

function value(value: unknown) {
  return value === null ? "—" : JSON.stringify(value);
}

export function ValidationLedger({ validation }: Props) {
  return (
    <section aria-label="Live config validation">
      <section aria-label="Pre-flight finding ledger">
        <h2>Pre-flight findings</h2>
        {validation.findings.length === 0 ? (
          <p>No text-level findings.</p>
        ) : (
          <ol>
            {validation.findings.map((finding, index) => (
              <li key={`${finding.attribution}:${finding.where}:${finding.check}:${index}`}>
                <strong>{finding.severity}</strong>
                {finding.check && <span> {finding.check}</span>}
                <span> · {finding.attribution} · {finding.where}</span>
                <p>{finding.message}</p>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section aria-label="Diff against preset">
        <h2>Diff against preset</h2>
        {validation.selected_presets.length === 0 ? (
          <p>No package preset selected.</p>
        ) : (
          <p>Compared with {validation.selected_presets.join(" + ")}</p>
        )}
        {validation.preset_changes.length === 0 ? (
          <p>No effective scientific differences.</p>
        ) : (
          <ol>
            {validation.preset_changes.map((change) => (
              <li key={change.path}>
                <strong>{change.kind}</strong> <code>{change.path}</code>
                <span> · {value(change.preset_value)} → {value(change.document_value)}</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </section>
  );
}
