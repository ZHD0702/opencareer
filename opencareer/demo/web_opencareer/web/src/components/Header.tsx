import { motion } from "framer-motion"
import { PanelLeftClose, PanelLeftOpen, Settings } from "lucide-react"
import { ThemeToggle } from "./ThemeToggle"

interface HeaderProps {
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
}

export function Header({ sidebarCollapsed, onToggleSidebar }: HeaderProps) {
  return (
    <header className="h-12 border-b border-border bg-background flex items-center justify-between px-3 shrink-0">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="p-1.5 rounded-md hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          title={sidebarCollapsed ? "展开侧栏" : "折叠侧栏"}
        >
          <motion.div
            animate={{ rotate: sidebarCollapsed ? 0 : 180 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
          >
            {sidebarCollapsed ? (
              <PanelLeftOpen className="w-4 h-4" />
            ) : (
              <PanelLeftClose className="w-4 h-4" />
            )}
          </motion.div>
        </button>
        <h1 className="text-sm font-semibold text-foreground select-none">
          OpenCareer
        </h1>
      </div>

      <div className="flex items-center gap-2">
        <ThemeToggle />
        <button
          type="button"
          className="p-1.5 rounded-md hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          title="设置（即将上线）"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}
