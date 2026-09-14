import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type ViewMode = "simplified" | "advanced";

const STORAGE_KEY = "rr-view-mode";

const ViewModeContext = createContext<{
  mode: ViewMode;
  setMode: (m: ViewMode) => void;
  isAdvanced: boolean;
}>({ mode: "simplified", setMode: () => {}, isAdvanced: false });

export function ViewModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ViewMode>(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      return saved === "advanced" ? "advanced" : "simplified";
    } catch {
      return "simplified";
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      /* private mode — ignore */
    }
    document.documentElement.dataset.view = mode;
  }, [mode]);

  const setMode = useCallback((m: ViewMode) => setModeState(m), []);

  return (
    <ViewModeContext.Provider value={{ mode, setMode, isAdvanced: mode === "advanced" }}>
      {children}
    </ViewModeContext.Provider>
  );
}

export function useViewMode() {
  return useContext(ViewModeContext);
}

/** Segmented Simplified / Advanced toggle. No emoji — text + sliding indicator. */
export function ViewToggle() {
  const { mode, setMode } = useViewMode();
  const options: Array<{ id: ViewMode; label: string; hint: string }> = [
    { id: "simplified", label: "Simplified", hint: "Normal users" },
    { id: "advanced", label: "Advanced", hint: "Developers" },
  ];
  return (
    <div
      role="group"
      aria-label="View mode: simplified for normal users, advanced for developers"
      className="relative grid grid-cols-2 rounded-sm border border-line bg-panel p-0.5 text-xs"
    >
      <span
        aria-hidden
        className={`absolute inset-y-0.5 w-[calc(50%-2px)] rounded-[3px] bg-card shadow-card transition-transform duration-200 ease-out ${
          mode === "advanced" ? "translate-x-[calc(100%+2px)] left-0.5" : "translate-x-0 left-0.5"
        }`}
      />
      {options.map((o) => {
        const active = mode === o.id;
        return (
          <button
            key={o.id}
            type="button"
            aria-pressed={active}
            title={o.hint}
            onClick={() => setMode(o.id)}
            className={`relative z-10 rounded-[3px] px-3 py-1.5 font-medium transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 cursor-pointer ${
              active ? "text-ink" : "text-ink-faint hover:text-ink-soft"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
