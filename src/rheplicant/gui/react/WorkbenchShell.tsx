import {
  useCallback,
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent,
  type MutableRefObject,
  type ReactNode,
} from "react";

const WorkbenchOpenerContext = createContext<MutableRefObject<HTMLElement | null> | null>(null);

export function useWorkbenchOpener() {
  return useContext(WorkbenchOpenerContext);
}

const MODAL_CONTROLS = [
  "button:not([disabled])",
  "a[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "summary",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

function modalControls(dialog: HTMLElement) {
  return Array.from(dialog.querySelectorAll<HTMLElement>(MODAL_CONTROLS))
    .filter((control) => control.isConnected && !control.closest("[inert]"));
}

function canRestoreFocus(candidate: HTMLElement | null): candidate is HTMLElement {
  return candidate !== null
    && candidate.isConnected
    && !candidate.closest("[inert]")
    && !(candidate instanceof HTMLButtonElement && candidate.disabled);
}

export function useWorkbenchModal(onClose: () => void) {
  const workbenchOpener = useWorkbenchOpener();
  const dialogRef = useRef<HTMLElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const openerCaptured = useRef(false);
  if (!openerCaptured.current) {
    const active = typeof document !== "undefined" && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const body = typeof document === "undefined" ? null : document.body;
    opener.current = workbenchOpener?.current ?? (active === body ? null : active);
    openerCaptured.current = true;
  }

  const closeModal = useCallback(() => {
    const focusAtClose = document.activeElement;
    onClose();
    const restore = () => {
      const active = document.activeElement;
      const focusIsUnclaimed = active === focusAtClose
        || active === null
        || active === document.body
        || !active.isConnected;
      if (!focusIsUnclaimed) return;
      const selectedWorkspace = document.querySelector<HTMLElement>(
        '[role="tab"][aria-selected="true"]',
      );
      const target = [opener.current, workbenchOpener?.current ?? null, selectedWorkspace]
        .find(canRestoreFocus);
      target?.focus();
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(restore);
    else setTimeout(restore, 0);
  }, [onClose, workbenchOpener]);

  useLayoutEffect(() => {
    const first = dialogRef.current && modalControls(dialogRef.current)[0];
    first?.focus();
  }, []);

  useEffect(() => {
    function retainFocus(event: FocusEvent) {
      const dialog = dialogRef.current;
      if (!dialog || dialog.contains(event.target as Node)) return;
      modalControls(dialog)[0]?.focus();
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closeModal();
    }
    document.addEventListener("focusin", retainFocus);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("focusin", retainFocus);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [closeModal]);

  function handleModalKeyDown(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key !== "Tab") return;
    const controls = modalControls(event.currentTarget);
    if (controls.length === 0) {
      event.preventDefault();
      return;
    }
    const first = controls[0];
    const last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return { dialogRef, closeModal, handleModalKeyDown };
}

function compactInspectorMatches() {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(max-width: 1279px)").matches;
}

export interface WorkbenchShellProps {
  header: ReactNode;
  navigation: ReactNode;
  main: ReactNode;
  inspector: ReactNode;
  jobs: ReactNode;
  overlay: ReactNode;
}

export function WorkbenchShell({
  header,
  navigation,
  main,
  inspector,
  jobs,
  overlay,
}: WorkbenchShellProps) {
  const opener = useRef<HTMLElement | null>(null);
  const [inspectorState, setInspectorState] = useState(() => {
    const compact = compactInspectorMatches();
    return { compact, open: !compact };
  });

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const viewport = window.matchMedia("(max-width: 1279px)");
    const updateInspector = () => setInspectorState({
      compact: viewport.matches,
      open: !viewport.matches,
    });
    updateInspector();
    viewport.addEventListener("change", updateInspector);
    return () => viewport.removeEventListener("change", updateInspector);
  }, []);

  const overlayOpen = Boolean(overlay);

  function captureOpener(event: MouseEvent<HTMLElement>) {
    if (!(event.target instanceof Element)) return;
    if (event.target.closest(".workbench-drawer")) return;
    const control = event.target.closest<HTMLElement>(
      "button, a[href], input, select, textarea, [tabindex]",
    );
    if (control && event.currentTarget.contains(control)) opener.current = control;
  }

  return (
    <WorkbenchOpenerContext.Provider value={opener}>
      <main className="rheplicant-editor workbench-shell" onClickCapture={captureOpener}>
        <header className="workbench-header" inert={overlayOpen ? true : undefined}>{header}</header>
        <div className="workbench-layout">
          <div className="workbench-navigation" inert={overlayOpen ? true : undefined}>{navigation}</div>
          <div className="workbench-main" inert={overlayOpen ? true : undefined}>{main}</div>
          <details
            className="workbench-inspector"
            inert={overlayOpen ? true : undefined}
            open={inspectorState.open}
            onToggle={(event) => {
              if (!inspectorState.compact) return;
              const open = event.currentTarget.open;
              setInspectorState((current) => ({ ...current, open }));
            }}
          >
            <summary>Context inspector</summary>
            <div>{inspector}</div>
          </details>
        </div>
        <footer className="workbench-jobs" inert={overlayOpen ? true : undefined}>{jobs}</footer>
        {overlay && <div className="workbench-drawer">{overlay}</div>}
      </main>
    </WorkbenchOpenerContext.Provider>
  );
}
