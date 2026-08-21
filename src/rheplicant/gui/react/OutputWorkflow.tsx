import { OutputTargetCard } from "./OutputTargetCard";
import { ProductSelector, type ProductView } from "./ProductSelector";
import { ReportDesigner, describedBy } from "./ReportDesigner";
import { StatusChip } from "./StatusChip";
import type { EditorSession, OutputProductProjection, SessionTransport } from "./types";

const NO_DECLARED_RUN_REASON_ID = "output-write-report-reason";

export interface OutputRequestEditorProps {
  session: EditorSession;
  transport: SessionTransport;
  disabled?: boolean;
  disabledReasonId?: string;
  productView: ProductView;
  onProductView(next: ProductView): void;
  comparisonOpen: boolean;
  onComparisonOpen(next: boolean): void;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

export function OutputRequestEditor({ session, transport, disabled = false, disabledReasonId, productView, onProductView, comparisonOpen, onComparisonOpen, onRun }: OutputRequestEditorProps) {
  const output = session.outputs;
  const report = output.report;
  const noDeclaredRun = output.declared_runs.length === 0;
  function updateProduct(product: OutputProductProjection) {
    onRun(
      () => transport.setOutputProduct(session.session_id, product.name, product.enabled, product.format, product.runs, product.keys, product.themes, session.revision),
      `Updated ${product.name} output`,
    );
  }
  function enableReport() {
    const rows = report.rows.length > 0 ? report.rows : output.declared_runs.slice(0, 1);
    onRun(
      () => transport.setOutputReport(session.session_id, true, rows, report.columns, report.reference, report.relative, report.formats, session.revision),
      "Enabled report output",
    );
  }
  return (
    <section aria-label="Output request">
      <OutputTargetCard output={output} />
      <ProductSelector products={output.products} declaredRuns={output.declared_runs} view={productView} onView={onProductView} onChange={updateProduct} disabled={disabled} disabledReasonId={disabledReasonId} />
      {report.enabled
        ? <ReportDesigner session={session} transport={transport} onRun={onRun} disabled={disabled} disabledReasonId={disabledReasonId} />
        : <>
          <button type="button" disabled={disabled || noDeclaredRun} aria-describedby={describedBy(disabled ? disabledReasonId : undefined, noDeclaredRun ? NO_DECLARED_RUN_REASON_ID : undefined)} onClick={enableReport}>Write report</button>
          {noDeclaredRun && <span id={NO_DECLARED_RUN_REASON_ID}><StatusChip tone="disabled" label="Write report disabled: no run is declared." /></span>}
        </>}
      <section aria-label="Advanced YAML comparison">
        <button type="button" aria-expanded={comparisonOpen} onClick={() => onComparisonOpen(!comparisonOpen)}>Requested and resolved YAML</button>
        {comparisonOpen && <>
          <section aria-label="Resolution note"><h3>Resolution note</h3><p>{output.resolution_note}</p></section>
          {/* Raw scientific YAML, rendered verbatim: React never parses, re-serialises or diffs it.

              Each block is a named, focusable scroll region. `.rheplicant-editor pre` caps the
              height at 32rem, so a real document leaves ~1100 and ~1300 pixels below the fold;
              without a tab stop those pixels are keyboard-unreachable in Firefox and WebKit
              (Chromium alone focuses scroll containers for free), which is WCAG 2.1.1 Level A.
              The name is what makes the stop worth arriving at: role="region" plus aria-label
              tells a screen-reader user which of the two blocks they have landed in. */}
          <section aria-label="Requested YAML"><h3>Requested YAML</h3><pre tabIndex={0} role="region" aria-label="Requested YAML source">{output.requested_yaml}</pre></section>
          <section aria-label="Preset-resolved YAML"><h3>Preset-resolved YAML</h3><pre tabIndex={0} role="region" aria-label="Preset-resolved YAML source">{output.resolved_yaml}</pre></section>
        </>}
      </section>
    </section>
  );
}

export { OutputRequestEditor as OutputWorkflow };
