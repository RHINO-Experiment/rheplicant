import type { LedgerFinding, ValidationProjection } from "./types";

interface Props {
  validation: ValidationProjection;
  ledgerLabel?: string;
  heading?: string;
  showFindings?: boolean;
  showPresetDiff?: boolean;
  onFinding?: (finding: LedgerFinding) => void;
}

function value(value: unknown) {
  return value === null ? "—" : JSON.stringify(value);
}

const severities: LedgerFinding["severity"][] = ["refuse", "warn", "report"];

function layerLabel(attribution: string) {
  if (attribution === "base") return "Base";
  if (attribution.startsWith("variant:")) return `Variant ${attribution.slice(8)}`;
  return attribution.charAt(0).toUpperCase() + attribution.slice(1);
}

function FindingGroups({
  findings,
  onFinding,
}: {
  findings: LedgerFinding[];
  onFinding?: (finding: LedgerFinding) => void;
}) {
  const layers = Array.from(new Set(findings.map((finding) => finding.attribution)));
  return <>{layers.flatMap((layer) => severities.map((severity) => {
    const selected = findings.filter((finding) => (
      finding.attribution === layer && finding.severity === severity
    ));
    if (selected.length === 0) return null;
    return (
      <section key={`${layer}:${severity}`} aria-label={`${layerLabel(layer)} ${severity} findings`}>
        <h3>{layerLabel(layer)} · {severity}</h3>
        <ol>
          {selected.map((finding, index) => (
            <li key={`${finding.where}:${finding.check}:${index}`}>
              <strong>{finding.severity}</strong>
              {finding.check && <span> {finding.check}</span>}
              <span> · {finding.attribution} · </span>
              {onFinding ? (
                <button
                  type="button"
                  aria-label={`${finding.severity} finding ${finding.where}`}
                  onClick={() => onFinding(finding)}
                >
                  {finding.where}
                </button>
              ) : <span>{finding.where}</span>}
              <p>{finding.message}</p>
            </li>
          ))}
        </ol>
      </section>
    );
  }))}</>;
}

export function ValidationLedger({
  validation,
  ledgerLabel = "Pre-flight finding ledger",
  heading = "Pre-flight findings",
  showFindings = true,
  showPresetDiff = true,
  onFinding,
}: Props) {
  return (
    <section aria-label="Live config validation">
      {showFindings && (
        <section aria-label={ledgerLabel}>
          <h2>{heading}</h2>
          {validation.findings.length === 0 ? (
            <p>No text-level findings.</p>
          ) : <FindingGroups findings={validation.findings} onFinding={onFinding} />}
        </section>
      )}

      {showPresetDiff && (
        <section aria-label="Diff against preset">
          <details>
            <summary>Diff against preset</summary>
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
          </details>
        </section>
      )}
    </section>
  );
}
