import { useEmotionTrends, type EmotionTrendItem } from "../hooks/useEmotionTrends"
import { useSessionStore } from "../stores/sessionStore"

const moodColors: Record<string, string> = {
  happy: "bg-emerald-500",
  confident: "bg-emerald-400",
  neutral: "bg-slate-400",
  stressed: "bg-amber-500",
  anxious: "bg-orange-500",
  discouraged: "bg-red-500",
  crisis: "bg-red-700",
}

const moodLabels: Record<string, string> = {
  happy: "开心",
  confident: "自信",
  neutral: "平静",
  stressed: "有压力",
  anxious: "焦虑",
  discouraged: "沮丧",
  crisis: "危机",
}

const stateColors: Record<string, string> = {
  positive: "bg-emerald-500",
  neutral: "bg-slate-400",
  negative: "bg-amber-500",
  crisis: "bg-red-600",
}

const stateLabels: Record<string, string> = {
  positive: "积极",
  neutral: "中性",
  negative: "消极",
  crisis: "危机",
}

const trendConfig: Record<string, { icon: string; label: string; color: string }> = {
  improving: { icon: "\u2191", label: "改善中", color: "text-emerald-500" },
  declining: { icon: "\u2193", label: "需关注", color: "text-red-500" },
  stable: { icon: "\u2192", label: "平稳", color: "text-slate-400" },
  insufficient_data: { icon: "\u2014", label: "数据不足", color: "text-slate-400" },
}

function getMoodBadge(mood: string | null) {
  if (!mood) return { color: "bg-slate-300", label: "暂无" }
  return {
    color: moodColors[mood] || "bg-slate-400",
    label: moodLabels[mood] || mood,
  }
}

function formatTime(iso: string): string {
  if (!iso) return ""
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
  } catch {
    return ""
  }
}

function TrendBars({ history }: { history: EmotionTrendItem[] }) {
  if (history.length === 0) {
    return <p className="text-xs text-sidebar-muted text-center mt-2">暂无情绪数据</p>
  }

  const display = history.slice(-8)

  return (
    <div className="flex items-end gap-1.5 h-20 mt-2">
      {display.map((item, i) => {
        const heightPct = Math.max(15, (item.confidence || 0.5) * 100)
        const stateColor = stateColors[item.overall_state] || "bg-slate-400"
        const stateLabel = stateLabels[item.overall_state] || item.overall_state
        return (
          <div key={i} className="flex-1 flex flex-col items-center gap-1 min-w-0">
            <div
              className={`w-full rounded-sm ${stateColor} transition-all`}
              style={{ height: `${heightPct}%` }}
              title={`${stateLabel} (${Math.round((item.confidence || 0) * 100)}%)`}
            />
            <span className="text-[10px] text-sidebar-muted leading-none">
              {formatTime(item.timestamp)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function HistoryList({ history }: { history: EmotionTrendItem[] }) {
  if (history.length === 0) return null

  const recent = [...history].reverse().slice(0, 6)

  return (
    <div className="mt-4 space-y-2">
      <h3 className="text-xs font-medium text-sidebar-foreground">近期情绪记录</h3>
      {recent.map((item, i) => {
        const stateColor = stateColors[item.overall_state] || "bg-slate-400"
        const stateLabel = stateLabels[item.overall_state] || item.overall_state
        const emotionLabels = item.emotions.length > 0
          ? item.emotions.join("\u00B7")
          : stateLabel

        return (
          <div key={i} className="flex items-start gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${stateColor} mt-0.5 shrink-0`} />
            <div className="flex-1 min-w-0">
              <div className="flex justify-between">
                <span className="text-sidebar-foreground truncate">{emotionLabels}</span>
                <span className="text-sidebar-muted ml-2 shrink-0">
                  {formatTime(item.timestamp)}
                </span>
              </div>
              <div className="text-sidebar-muted">
                {item.demand_type !== "unknown" && (
                  <span>需求: {item.demand_type}</span>
                )}
                {item.support_intensity !== "none" && (
                  <span className="ml-2">
                    支持: {item.support_intensity === "high" ? "强" : item.support_intensity === "medium" ? "中" : "低"}
                  </span>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function EmotionTab() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const { data, isLoading, isError } = useEmotionTrends(sessionId)

  if (!sessionId) {
    return (
      <div className="text-sm text-sidebar-muted">
        <p className="text-center mt-8">等待会话创建...</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="text-sm text-sidebar-muted animate-pulse">
        <div className="h-4 bg-muted rounded w-3/4 mx-auto mt-8" />
        <div className="h-20 bg-muted rounded mt-4" />
        <div className="h-4 bg-muted rounded w-1/2 mx-auto mt-4" />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="text-sm text-sidebar-muted">
        <p className="text-center mt-8 text-red-500">无法加载情绪数据</p>
        <p className="text-center text-xs mt-2">请检查后端服务是否运行</p>
      </div>
    )
  }

  const moodBadge = getMoodBadge(data.current_mood)
  const trend = trendConfig[data.trend] || trendConfig.insufficient_data

  return (
    <div className="text-sm">
      {/* Current mood + trend */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${moodBadge.color}`} />
          <span className="font-medium text-sidebar-foreground">{moodBadge.label}</span>
        </div>
        <span className={`text-xs ${trend.color}`}>
          {trend.icon} {trend.label}
        </span>
      </div>

      {/* Trend bars */}
      <div className="bg-muted/50 rounded-lg p-3">
        <div className="flex justify-between text-xs text-sidebar-muted">
          <span>情绪趋势</span>
          <span>
            负面比: {Math.round(data.negative_ratio * 100)}%
          </span>
        </div>
        <TrendBars history={data.history} />
      </div>

      {/* Intervention warning */}
      {data.needs_intervention && (
        <div className="mt-3 p-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800">
          <p className="text-xs text-red-600 dark:text-red-400 font-medium">
            需要关注
          </p>
          <p className="text-xs text-red-500 dark:text-red-400 mt-1">
            {data.reason || "用户情绪趋势显示需要更多关怀"}
          </p>
        </div>
      )}

      {/* Consecutive negative indicator */}
      {data.consecutive_negative >= 2 && (
        <div className="mt-2 text-xs text-amber-500">
          连续 {data.consecutive_negative} 轮情绪偏低
        </div>
      )}

      {/* History list */}
      <HistoryList history={data.history} />
    </div>
  )
}
