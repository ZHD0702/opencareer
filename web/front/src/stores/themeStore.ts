import { create } from "zustand"

export type ThemeMode = "light" | "dark" | "system"

function getStoredTheme(): ThemeMode {
  try {
    const stored = localStorage.getItem("oc-theme")
    if (stored === "light" || stored === "dark" || stored === "system") return stored
  } catch { /* ignore */ }
  return "system"
}

function applyTheme(mode: ThemeMode) {
  const root = document.documentElement
  root.classList.remove("light", "dark")
  if (mode === "light") {
    root.classList.add("light")
  } else if (mode === "dark") {
    root.classList.add("dark")
  }
  // "system" — neither class, let media query handle it
}

interface ThemeState {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
}

export const useThemeStore = create<ThemeState>((set) => {
  const initial = getStoredTheme()
  applyTheme(initial)
  return {
    mode: initial,
    setMode: (mode) => {
      localStorage.setItem("oc-theme", mode)
      applyTheme(mode)
      set({ mode })
    },
  }
})
