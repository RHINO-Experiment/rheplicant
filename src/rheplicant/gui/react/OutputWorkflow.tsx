import { useState } from "react";

import { OutputTargetCard } from "./OutputTargetCard";
import { ProductSelector } from "./ProductSelector";
import { ReportDesigner } from "./ReportDesigner";
import type { EditorSession, OutputProductProjection, SessionTransport } from "./types";

export interface OutputRequestEditorProps {
  session: EditorSession;
  transport: SessionTransport;
  disabled?: boolean;
  disabledReasonId?: string;
  onRun(action: () => Promise<EditorSession>, message: string): void;
}

export function OutputRequestEditor({ session, transport, disabled = false, disabledReasonId, onRun }: OutputRequestEditorProps) {
  const [expandedProduct, setExpandedProduct] = useState<string | null>(null);
  const output = session.outputs;
  const report = output.report;
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
      <ProductSelector products={output.products} declaredRuns={output.declared_runs} expanded={expandedProduct} onExpand={setExpandedProduct} onChange={updateProduct} disabled={disabled} disabledReasonId={disabledReasonId} />
      {report.enabled
        ? <ReportDesigner session={session} transport={transport} onRun={onRun} disabled={disabled} disabledReasonId={disabledReasonId} />
        : <button type="button" disabled={disabled || output.declared_runs.length === 0} aria-describedby={disabled ? disabledReasonId : undefined} onClick={enableReport}>Write report</button>}
    </section>
  );
}

export { OutputRequestEditor as OutputWorkflow };
