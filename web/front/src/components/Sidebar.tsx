import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Heart, FileText, BarChart3, Sparkles, Clock, SquarePen } from "lucide-react"
import { useSessionStore, type SidebarTab } from "../stores/sessionStore"
import { useChatStore } from "../stores/chatStore"
import { cn } from "../lib/utils"
import { EmotionTab } from "./EmotionTab"
import { ResumeTab } from "./ResumeTab"
import { JobProgressTab } from "./JobProgressTab"
import { SkillAssessmentTab } from "./SkillAssessmentTab"
import { SessionHistoryTab } from "./SessionHistoryTab"

const tabs: { id: SidebarTab; label: string; icon: typeof Heart }[] = [
  { id: "session-history", label: "历史会话", icon: Clock },
  { id: "emotion", label: "情绪", icon: Heart },
  { id: "resume", label: "简历", icon: FileText },
  { id: "job-progress", label: "求职进度", icon: BarChart3 },
  { id: "skill-assessment", label: "技能评估", icon: Sparkles },
]

const tabContent: Record<SidebarTab, React.ComponentType> = {
  emotion: EmotionTab,
  resume: ResumeTab,
  "job-progress": JobProgressTab,
  "skill-assessment": SkillAssessmentTab,
  "session-history": SessionHistoryTab,
}

export function Sidebar() {
  const { sidebarTab, setSidebarTab, sidebarCollapsed, setSessionId } = useSessionStore()
  const resetConversation = useChatStore((s) => s.resetConversation)
  const [hoveredTab, setHoveredTab] = useState<SidebarTab | null>(null)
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  const handleCreateNewSession = () => {
    setSessionId(null)
    resetConversation()
    setSidebarTab("session-history")
    setHoveredTab(null)
  }

  return (
    <motion.aside
      className="border-r border-sidebar-border bg-sidebar flex flex-col shrink-0 overflow-hidden relative"
      animate={{ width: sidebarCollapsed ? 56 : 340 }}
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
                "relative w-full flex items-center gap-3 px-4 py-3 text-base transition-colors rounded-md mx-1",
                isActive
                  ? "text-primary"
                  : "text-sidebar-muted hover:text-sidebar-foreground",
                sidebarCollapsed && "justify-center px-0"
              )}
            >
              <tab.icon className="w-5 h-5 shrink-0" />
              {!sidebarCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm font-medium whitespace-nowrap"
                >
                  {tab.label}
                </motion.span>
              )}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-tab"
                  className="absolute left-0 top-1 bottom-1 w-1 rounded-r-full bg-primary"
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
            className="flex-1 px-4 pb-4 overflow-y-auto"
          >
            <ActiveContent />
          </motion.div>
        )}
      </AnimatePresence>

      <div className={cn(
        "border-t border-sidebar-border bg-sidebar/95 p-3",
        sidebarCollapsed && "px-2"
      )}>
        <motion.button
          type="button"
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleCreateNewSession}
          title="创建新会话"
          className={cn(
            "group relative flex w-full items-center justify-center gap-2 rounded-lg border border-sidebar-border",
            "bg-background/70 text-sidebar-foreground shadow-sm transition-colors",
            "hover:border-primary/40 hover:bg-primary/10 hover:text-primary",
            "focus:outline-none focus:ring-2 focus:ring-primary/25",
            sidebarCollapsed ? "h-10 px-0" : "px-3 py-2.5"
          )}
        >
          <SquarePen className="h-4 w-4 shrink-0" />
          {!sidebarCollapsed && (
            <span className="text-sm font-medium whitespace-nowrap">创建新会话</span>
          )}
          {sidebarCollapsed && (
            <span className="pointer-events-none absolute left-[48px] z-50 rounded-md border border-sidebar-border bg-sidebar px-2 py-1 text-xs text-sidebar-foreground opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
              创建新会话
            </span>
          )}
        </motion.button>
      </div>

      {/* Hover popup (collapsed mode) */}
      <AnimatePresence>
        {sidebarCollapsed && hoveredTab && (
          <motion.div
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.2 }}
            className="absolute left-[60px] top-2 bottom-2 w-[340px] bg-sidebar border border-sidebar-border rounded-lg shadow-lg z-50 p-4 overflow-y-auto"
            onMouseEnter={() => {
              if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
            }}
            onMouseLeave={() => setHoveredTab(null)}
          >
            <p className="text-sm font-medium text-sidebar-foreground mb-3">
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
