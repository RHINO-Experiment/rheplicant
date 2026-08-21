import type { ReactNode } from "react";

export type StatusTone =
  | "neutral"
  | "success"
  | "warning"
  | "danger"
  | "stale"
  | "disabled";

interface StatusChipProps {
  tone: StatusTone;
  label: ReactNode;
  urgent?: boolean;
}

const statusIcon: Record<StatusTone, string> = {
  neutral: "○",
  success: "✓",
  warning: "⚠",
  danger: "!",
  stale: "↶",
  disabled: "—",
};

export function StatusChip({
  tone,
  label,
  urgent = false,
}: StatusChipProps) {
  return (
    <span
      className={`status-chip status-${tone}`}
      role={urgent ? "alert" : "status"}
      aria-live={urgent ? "assertive" : "polite"}
    >
      <span className="status-chip-icon" aria-hidden="true">{statusIcon[tone]}</span>
      <span>{label}</span>
    </span>
  );
}
