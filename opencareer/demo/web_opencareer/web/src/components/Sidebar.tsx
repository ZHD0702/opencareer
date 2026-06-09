import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Heart, FileText, BarChart3, Sparkles, Clock } from "lucide-react"
import { useSessionStore, type SidebarTab } from "../stores/sessionStore"
import { cn } from "../lib/utils"
import { EmotionTab } from "./EmotionTab"
import { ResumeTab } from "./ResumeTab"
import { JobProgressTab } from "./JobProgressTab"
import { SkillAssessmentTab } from "./SkillAssessmentTab"
import { SessionHistoryTab } from "./SessionHistoryTab"

const tabs: { id: SidebarTab; label: string; icon: typeof Heart }[] = [
  { id: "emotion", label: "情绪", icon: Heart },
  { id: "resume", label: "简历", icon: FileText },
  { id: "job-progress", label: "求职进度", icon: BarChart3 },
  { id: "skill-assessment", label: "技能评估", icon: Sparkles },
  { id: "session-history", label: "历史会话", icon: Clock },
]

const tabContent: Record<SidebarTab, React.ComponentType> = {
  emotion: EmotionTab,
  resume: ResumeTab,
  "job-progress": JobProgressTab,
  "skill-assessment": SkillAssessmentTab,
  "session-history": SessionHistoryTab,
}

export function Sidebar() {
  const { sidebarTab, setSidebarTab, sidebarCollapsed } = useSessionStore()
  const [hoveredTab, setHoveredTab] = useState<SidebarTab | null>(null)
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout>>()

  const handleTabHover = (tabId: SidebarTab) => {
    if (!sidebarCollapsed) return
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    setHoveredTab(tabId)
  }

  const handleTabLeave = () => {
    if (!sidebarCollapsed) return
    hoverTimeoutRef.current = setTimeout(() => setHoveredTab(null), 200)
  }

  const ActiveContent = tabContent[sidebarTab]

  return (
    <motion.aside
      className="border-r border-sidebar-border bg-sidebar flex flex-col shrink-0 overflow-hidden relative"
      animate={{ width: sidebarCollapsed ? 48 : 280 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {/* Tab icons (always visible) */}
      <nav className="flex flex-col items-center py-2 gap-1">
        {tabs.map((tab) => {
          const isActive = sidebarTab === tab.id
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setSidebarTab(tab.id)}
              onMouseEnter={() => handleTabHover(tab.id)}
              onMouseLeave={handleTabLeave}
              className={cn(
                "relative w-full flex items-center gap-3 px-3 py-2 text-sm transition-colors rounded-md mx-1",
                isActive
                  ? "text-primary"
                  : "text-sidebar-muted hover:text-sidebar-foreground",
                sidebarCollapsed && "justify-center px-0"
              )}
            >
              <tab.icon className="w-4 h-4 shrink-0" />
              {!sidebarCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-xs font-medium whitespace-nowrap"
                >
                  {tab.label}
                </motion.span>
              )}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-tab"
                  className="absolute left-0 top-1 bottom-1 w-0.5 rounded-r-full bg-primary"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          )
        })}
      </nav>

      {/* Panel content (expanded) */}
      <AnimatePresence>
        {!sidebarCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="flex-1 px-3 pb-3 overflow-y-auto"
          >
            <ActiveContent />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hover popup (collapsed mode) */}
      <AnimatePresence>
        {sidebarCollapsed && hoveredTab && (
          <motion.div
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.2 }}
            className="absolute left-[52px] top-2 bottom-2 w-[280px] bg-sidebar border border-sidebar-border rounded-lg shadow-lg z-50 p-3 overflow-y-auto"
            onMouseEnter={() => {
              if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
            }}
            onMouseLeave={() => setHoveredTab(null)}
          >
            <p className="text-xs font-medium text-sidebar-foreground mb-2">
              {tabs.find((t) => t.id === hoveredTab)?.label}
            </p>
            {(() => {
              const HoverContent = tabContent[hoveredTab]
              return <HoverContent />
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.aside>
  )
}
