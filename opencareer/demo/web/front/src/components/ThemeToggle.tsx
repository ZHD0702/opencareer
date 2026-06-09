import { motion } from "framer-motion"
import { Sun, Moon, Monitor } from "lucide-react"
import { useThemeStore, type ThemeMode } from "../stores/themeStore"
import { cn } from "../lib/utils"

const modes: { id: ThemeMode; label: string; icon: typeof Sun }[] = [
  { id: "light", label: "亮色", icon: Sun },
  { id: "dark", label: "暗色", icon: Moon },
  { id: "system", label: "跟随系统", icon: Monitor },
]

export function ThemeToggle() {
  const { mode, setMode } = useThemeStore()

  return (
    <div className="flex items-center gap-0.5 rounded-lg bg-muted p-0.5">
      {modes.map((m) => {
        const isActive = mode === m.id
        return (
          <button
            key={m.id}
            type="button"
            onClick={() => setMode(m.id)}
            title={m.label}
            className={cn(
              "relative flex items-center justify-center w-9 h-8 rounded-md text-sm transition-colors",
              isActive
                ? "text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {isActive && (
              <motion.div
                layoutId="theme-active"
                className="absolute inset-0 bg-primary rounded-md"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
            <m.icon className="relative z-10 w-5 h-5" />
          </button>
        )
      })}
    </div>
  )
}
