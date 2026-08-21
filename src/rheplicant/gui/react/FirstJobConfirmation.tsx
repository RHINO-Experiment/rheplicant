import { useEffect, useRef } from "react";

import type { JobKind } from "./types";

interface Props {
  kind: JobKind;
  blocked: boolean;
  onConfirm(): void;
  onCancel(): void;
}

function jobLabel(kind: JobKind) {
  return {
    validate: "Validate",
    preview_forward: "Preview forward",
    run: "Run",
    compare: "Compare",
    benchmark: "Benchmark",
  }[kind];
}

export function FirstJobConfirmation({ kind, blocked, onConfirm, onCancel }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    cancelRef.current?.focus();
    function retainFocus(event: FocusEvent) {
      const dialog = dialogRef.current;
      if (dialog && event.target instanceof Node && !dialog.contains(event.target)) {
        cancelRef.current?.focus();
      }
    }
    document.addEventListener("focusin", retainFocus);
    return () => document.removeEventListener("focusin", retainFocus);
  }, []);

  function keyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel();
      return;
    }
    if (event.key !== "Tab") return;
    const controls = [cancelRef.current, confirmRef.current]
      .filter((control): control is HTMLButtonElement => control !== null && !control.disabled);
    if (controls.length === 0) return;
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="trusted-execution-title"
      onKeyDown={keyDown}
    >
      <h2 id="trusted-execution-title">Trusted execution</h2>
      <p>Requested action: {jobLabel(kind)}</p>
      <ul>
        <li>Trusted YAML may execute plugins and python targets.</li>
        <li>Paths may read from and write to the server filesystem.</li>
        <li>Jobs may consume CPU, accelerator time and wall time.</li>
        <li>Execution uses the shared server process and account.</li>
      </ul>
      <button
        ref={cancelRef}
        type="button"
        onClick={onCancel}
      >
        Cancel trusted execution
      </button>
      <button
        ref={confirmRef}
        type="button"
        disabled={blocked}
        onClick={onConfirm}
      >
        I understand, continue
      </button>
    </div>
  );
}
