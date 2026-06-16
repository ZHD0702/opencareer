import { AlertTriangle, HeartPulse, ShieldCheck } from "lucide-react"
import { useEmotionTrends, type EmotionTrendItem } from "../hooks/useEmotionTrends"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"

const moodConfig: Record<string, { label: string; dot: string; text: string }> = {
  positive: { label: "积极", dot: "bg-emerald-500", text: "text-emerald-600" },
  happy: { label: "开心", dot: "bg-emerald-500", text: "text-emerald-600" },
  confident: { label: "自信", dot: "bg-emerald-500", text: "text-emerald-600" },
  neutral: { label: "平静", dot: "bg-slate-400", text: "text-slate-500" },
  uneasy: { label: "有点不安", dot: "bg-amber-400", text: "text-amber-600" },
  stressed: { label: "有压力", dot: "bg-amber-500", text: "text-amber-600" },
  anxious: { label: "焦虑", dot: "bg-orange-500", text: "text-orange-600" },
  angry: { label: "愤怒", dot: "bg-orange-600", text: "text-orange-700" },
  distressed: { label: "高压", dot: "bg-red-500", text: "text-red-600" },
  discouraged: { label: "沮丧", dot: "bg-red-500", text: "text-red-600" },
  crisis: { label: "危机", dot: "bg-red-700", text: "text-red-700" },
}

const stateColors: Record<string, string> = {
  positive: "bg-emerald-500",
  neutral: "bg-slate-400",
  negative: "bg-amber-500",
  crisis: "bg-red-600",
}

const trendConfig: Record<string, { label: string; color: string; icon: string }> = {
  improving: { label: "改善中", color: "text-emerald-600", icon: "↑" },
  declining: { label: "需关注", color: "text-red-600", icon: "↓" },
  stable: { label: "平稳", color: "text-slate-500", icon: "→" },
  insufficient_data: { label: "数据不足", color: "text-slate-400", icon: "—" },
}

const supportLabels: Record<string, string> = {
  none: "无需支持",
  low: "轻度支持",
  medium: "中度支持",
  high: "优先疏导",
  crisis: "安全优先",
}

function moodFor(value?: string | null) {
  if (!value) return moodConfig.neutral
  return moodConfig[value] || { label: value, dot: "bg-slate-400", text: "text-slate-500" }
}

function formatTime(value?: string): string {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

function supportLabel(value?: string | null) {
  if (!value) return supportLabels.none
  return supportLabels[value] || value
}

function TrendBars({ history }: { history: EmotionTrendItem[] }) {
  if (history.length === 0) {
    return (
      <div className="h-20 grid place-items-center text-xs text-sidebar-muted">
        暂无情绪数据
      </div>
    )
  }

  const display = [...history].reverse().slice(-8)

  return (
    <div className="flex items-end gap-1.5 h-20 mt-3">
      {display.map((item, index) => {
        const heightPct = Math.max(16, Math.min(100, (item.confidence || 0.45) * 100))
        const stateColor = stateColors[item.overall_state] || "bg-slate-400"

        return (
          <div key={`${item.timestamp}-${index}`} className="flex-1 flex flex-col items-center gap-1 min-w-0">
            <div
              className={`w-full rounded-sm ${stateColor}`}
              style={{ height: `${heightPct}%` }}
              title={`${item.overall_state} ${Math.round((item.confidence || 0) * 100)}%`}
            />
            <span className="text-[10px] leading-none text-sidebar-muted">
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

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-medium text-sidebar-foreground">最近记录</p>
      {history.slice(0, 5).map((item, index) => {
        const stateColor = stateColors[item.overall_state] || "bg-slate-400"
        const title = item.emotions.length > 0 ? item.emotions.join(" · ") : moodFor(item.current_mood).label

        return (
          <div key={`${item.timestamp}-${index}`} className="flex gap-2 text-xs">
            <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${stateColor}`} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sidebar-foreground">{title}</span>
                <span className="shrink-0 text-sidebar-muted">{formatTime(item.timestamp)}</span>
              </div>
              <div className="mt-0.5 text-sidebar-muted">
                {supportLabel(item.support_intensity)}
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
  const liveEmotion = useChatStore((s) => s.emotionAnalysis)
  const { data, isLoading, isError } = useEmotionTrends(sessionId)

  if (!sessionId) {
    return (
      <div className="grid h-28 place-items-center text-sm text-sidebar-muted">
        等待会话创建...
      </div>
    )
  }

  if (isLoading && !liveEmotion) {
    return (
      <div className="space-y-3 pt-2 text-sm text-sidebar-muted animate-pulse">
        <div className="h-6 rounded bg-muted" />
        <div className="h-24 rounded-lg bg-muted" />
        <div className="h-16 rounded-lg bg-muted" />
      </div>
    )
  }

  if (isError && !liveEmotion) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-600 dark:border-red-900 dark:bg-red-950/30">
        无法加载情绪数据
      </div>
    )
  }

  const liveForSession = liveEmotion?.session_id === sessionId ? liveEmotion : null
  const currentMood = liveForSession?.current_mood || data?.current_mood || "neutral"
  const currentState = liveForSession?.overall_state || data?.current_overall_state || "neutral"
  const emotions = liveForSession?.emotions || data?.current_emotions || []
  const confidence = liveForSession?.confidence ?? data?.confidence ?? 0
  const supportIntensity = liveForSession?.support_intensity || data?.support_intensity || "none"
  const needsIntervention = liveForSession?.should_intervene ?? data?.needs_intervention ?? false
  const reason = liveForSession?.reason || data?.reason || "情绪状态平稳"
  const trendKey = liveForSession?.trend || data?.trend || "stable"
  const trend = trendConfig[trendKey] || trendConfig.stable
  const mood = moodFor(currentMood)
  const negativeRatio = data?.negative_ratio ?? (currentState === "negative" || currentState === "crisis" ? 1 : 0)
  const history = data?.history || []

  return (
    <div className="text-sm">
      <div className="flex items-center justify-between">
        <div className="flex min-w-0 items-center gap-2">
          <span className={`h-3 w-3 shrink-0 rounded-full ${mood.dot}`} />
          <div className="min-w-0">
            <div className={`font-medium ${mood.text}`}>{mood.label}</div>
            <div className="truncate text-xs text-sidebar-muted">
              {emotions.length > 0 ? emotions.join(" · ") : "暂无明显情绪波动"}
            </div>
          </div>
        </div>
        <span className={`shrink-0 text-xs ${trend.color}`}>
          {trend.icon} {trend.label}
        </span>
      </div>

      <div className="mt-3 rounded-lg bg-muted/50 p-3">
        <div className="flex items-center justify-between text-xs text-sidebar-muted">
          <span>情绪趋势</span>
          <span>负面比: {Math.round(negativeRatio * 100)}%</span>
        </div>
        <TrendBars history={history} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <div className="rounded-md border border-sidebar-border bg-sidebar/60 p-2">
          <div className="text-[11px] text-sidebar-muted">支持强度</div>
          <div className="mt-1 text-xs font-medium text-sidebar-foreground">
            {supportLabel(supportIntensity)}
          </div>
        </div>
        <div className="rounded-md border border-sidebar-border bg-sidebar/60 p-2">
          <div className="text-[11px] text-sidebar-muted">置信度</div>
          <div className="mt-1 text-xs font-medium text-sidebar-foreground">
            {Math.round(confidence * 100)}%
          </div>
        </div>
        <div className="rounded-md border border-sidebar-border bg-sidebar/60 p-2">
          <div className="text-[11px] text-sidebar-muted">连续负面</div>
          <div className="mt-1 text-xs font-medium text-sidebar-foreground">
            {data?.consecutive_negative ?? 0} 轮
          </div>
        </div>
        <div className="rounded-md border border-sidebar-border bg-sidebar/60 p-2">
          <div className="text-[11px] text-sidebar-muted">服务策略</div>
          <div className="mt-1 flex items-center gap-1 text-xs font-medium text-sidebar-foreground">
            {needsIntervention ? <AlertTriangle className="h-3.5 w-3.5 text-red-500" /> : <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />}
            {needsIntervention ? "先疏导" : "可继续"}
          </div>
        </div>
      </div>

      {needsIntervention && (
        <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950/30">
          <div className="flex items-center gap-2 text-xs font-medium text-red-600 dark:text-red-400">
            <HeartPulse className="h-4 w-4" />
            需要优先关注
          </div>
          <p className="mt-1 text-xs leading-relaxed text-red-600/80 dark:text-red-300/80">
            {reason}
          </p>
        </div>
      )}

      <HistoryList history={history} />
    </div>
  )
}
