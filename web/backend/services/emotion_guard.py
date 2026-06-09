from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from db.crud import get_emotion_records, save_emotion_record


@dataclass
class EmotionAssessment:
    overall_state: str = "neutral"
    current_mood: str = "neutral"
    emotions: list[str] = field(default_factory=list)
    confidence: float = 0.4
    suggested_action: str = "work"
    support_intensity: str = "none"
    should_intervene: bool = False
    reason: str = "未检测到明显情绪风险"
    trend: str = "stable"
    matched_keywords: list[str] = field(default_factory=list)

    def to_event(self) -> dict[str, Any]:
        return {
            "overall_state": self.overall_state,
            "current_mood": self.current_mood,
            "emotions": self.emotions,
            "confidence": self.confidence,
            "suggested_action": self.suggested_action,
            "support_intensity": self.support_intensity,
            "should_intervene": self.should_intervene,
            "reason": self.reason,
            "trend": self.trend,
            "matched_keywords": self.matched_keywords,
        }


class EmotionGuard:
    """Emotion recognition and routing guard for the GUI chat flow."""

    CRISIS_KEYWORDS = (
        "不想活", "不想活了", "想死", "自杀", "轻生", "结束生命", "活不下去",
        "没有活着的意义", "伤害自己", "我撑不住了", "撑不下去",
    )
    HIGH_KEYWORDS = (
        "崩溃", "受不了了", "受不了", "绝望", "没希望", "好痛苦", "很痛苦",
        "特别痛苦", "喘不过气", "快疯了", "麻木", "耗尽", "筋疲力尽",
        "孤立无援", "没人帮", "没人理解", "无助", "被孤立", "撑不住",
    )
    MEDIUM_KEYWORDS = (
        "焦虑", "压力", "紧张", "害怕", "担心", "迷茫", "沮丧", "难过",
        "心累", "疲惫", "烦", "烦躁", "不开心", "失眠", "被拒", "挫败",
        "怀疑自己", "我不行", "没能力", "太菜", "比不过",
    )
    ANGER_KEYWORDS = (
        "生气", "气死", "气人", "窝火", "憋屈", "委屈", "烦死", "恶心",
        "傻逼", "煞笔", "sb", "蠢", "垃圾", "全是傻", "周围人",
    )
    LOW_KEYWORDS = (
        "有点慌", "有点累", "不太舒服", "没状态", "没动力", "不顺",
        "有点担心", "有点焦虑",
    )
    POSITIVE_KEYWORDS = ("开心", "好多了", "可以了", "谢谢", "有信心", "期待", "顺利")

    NEGATIVE_STATES = {"negative", "crisis"}
    NEGATIVE_MOODS = {"negative", "anxious", "distressed", "crisis", "uneasy", "angry"}
    INTENSITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "crisis": 4}

    def assess(self, session_id: str, user_message: str) -> EmotionAssessment:
        text = (user_message or "").strip().lower()
        recent_records = get_emotion_records(session_id, limit=8)

        assessment = self._assess_current_message(text)
        assessment = self._carry_recent_negative_state(assessment, recent_records)
        assessment.trend = self._detect_trend(recent_records, assessment)

        consecutive_negative = self._count_consecutive_negative(recent_records)
        high_recent_count = self._count_recent_high_support(recent_records)

        if assessment.support_intensity == "crisis":
            assessment.should_intervene = True
            assessment.reason = "检测到危机表达，需要暂停工作服务并优先关注安全"
        elif assessment.support_intensity == "high":
            assessment.should_intervene = True
            assessment.reason = "检测到高强度负面情绪，需要先进行心理疏导"
        elif assessment.support_intensity == "medium" and consecutive_negative >= 2:
            assessment.should_intervene = True
            assessment.support_intensity = "high"
            assessment.reason = "连续多轮负面情绪，暂时切换到心理疏导"
        elif high_recent_count >= 2:
            assessment.should_intervene = True
            assessment.support_intensity = "high"
            assessment.reason = "近期高强度情绪出现较多，需要先稳定情绪"
        elif assessment.support_intensity == "medium":
            assessment.reason = "检测到明显情绪压力，本轮会先共情再继续工作服务"

        if assessment.should_intervene:
            assessment.suggested_action = "emotional_support"

        save_emotion_record(
            session_id=session_id,
            overall_state=assessment.overall_state,
            current_mood=assessment.current_mood,
            emotions=assessment.emotions,
            confidence=assessment.confidence,
            demand_type=assessment.suggested_action,
            support_intensity=assessment.support_intensity,
        )
        return assessment

    def build_support_response(self, assessment: EmotionAssessment, user_message: str) -> str:
        if assessment.support_intensity == "crisis":
            return (
                "我先不继续聊求职的事了。\n\n"
                "听起来你现在真的很痛苦，这件事比任何简历、面试都更重要。\n\n"
                "如果你有伤害自己的冲动，先把自己放到一个相对安全的地方，立刻联系身边可信任的人，"
                "或者拨打当地紧急电话/心理援助热线。\n\n"
                "你也可以先只回我一句：你现在是一个人吗？"
            )

        if assessment.support_intensity == "high":
            return (
                "我先陪你缓一缓。\n\n"
                "现在不急着处理工作问题。你说到的这些感受，已经不是普通的烦一下了，"
                "更像是真的被压得很紧。\n\n"
                "我们先做一件很小的事：把眼前最难受的那个点说出来就好，不需要整理得很清楚。\n\n"
                "是害怕找不到工作，还是觉得自己撑不住这种状态了？"
            )

        return (
            "我听出来你现在有点被情绪卡住了。\n\n"
            "这很正常，求职本来就会把人的不确定感放大。\n\n"
            "我们先不急着给方案，你可以先跟我说说：最让你难受的是哪一块？"
        )

    def _assess_current_message(self, text: str) -> EmotionAssessment:
        if not text:
            return EmotionAssessment()

        if matched := self._match_keywords(text, self.CRISIS_KEYWORDS):
            return EmotionAssessment(
                overall_state="crisis",
                current_mood="crisis",
                emotions=["危机", "痛苦"],
                confidence=0.96,
                suggested_action="emotional_support",
                support_intensity="crisis",
                should_intervene=True,
                matched_keywords=matched,
            )

        if matched := self._match_keywords(text, self.HIGH_KEYWORDS):
            return EmotionAssessment(
                overall_state="negative",
                current_mood="distressed",
                emotions=["高压", "痛苦"],
                confidence=0.88,
                support_intensity="high",
                matched_keywords=matched,
            )

        if matched := self._match_keywords(text, self.ANGER_KEYWORDS):
            return EmotionAssessment(
                overall_state="negative",
                current_mood="angry",
                emotions=["愤怒", "挫败"],
                confidence=0.82,
                support_intensity="medium",
                matched_keywords=matched,
            )

        if matched := self._match_keywords(text, self.MEDIUM_KEYWORDS):
            return EmotionAssessment(
                overall_state="negative",
                current_mood="anxious",
                emotions=["焦虑", "压力"],
                confidence=0.76,
                support_intensity="medium",
                matched_keywords=matched,
            )

        if matched := self._match_keywords(text, self.LOW_KEYWORDS):
            return EmotionAssessment(
                overall_state="negative",
                current_mood="uneasy",
                emotions=["轻度压力"],
                confidence=0.62,
                support_intensity="low",
                matched_keywords=matched,
            )

        if matched := self._match_keywords(text, self.POSITIVE_KEYWORDS):
            return EmotionAssessment(
                overall_state="positive",
                current_mood="positive",
                emotions=["积极"],
                confidence=0.66,
                support_intensity="none",
                matched_keywords=matched,
            )

        return EmotionAssessment()

    def _match_keywords(self, text: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if keyword.lower() in text]

    def _carry_recent_negative_state(
        self,
        current: EmotionAssessment,
        records: list[dict[str, Any]],
    ) -> EmotionAssessment:
        if current.overall_state != "neutral" or not records:
            return current

        latest = records[0]
        latest_intensity = latest.get("support_intensity", "none")
        latest_state = latest.get("overall_state", "neutral")
        latest_mood = latest.get("current_mood", "neutral")

        if (
            latest_state not in self.NEGATIVE_STATES
            and latest_mood not in self.NEGATIVE_MOODS
            and self.INTENSITY_RANK.get(latest_intensity, 0) < 2
        ):
            return current

        carried_intensity = "medium" if latest_intensity in {"high", "crisis"} else "low"
        carried_mood = "anxious" if carried_intensity == "medium" else "uneasy"
        carried_emotions = ["情绪延续", "压力"] if carried_intensity == "medium" else ["轻度压力"]

        return EmotionAssessment(
            overall_state="negative",
            current_mood=carried_mood,
            emotions=carried_emotions,
            confidence=0.58 if carried_intensity == "medium" else 0.5,
            support_intensity=carried_intensity,
            reason="近期仍有负面情绪延续，暂不判定为完全平静",
            matched_keywords=[],
        )

    def _detect_trend(self, records: list[dict[str, Any]], current: EmotionAssessment) -> str:
        if current.overall_state == "crisis":
            return "declining"
        if len(records) < 2:
            return "stable"

        recent_score = sum(
            self.INTENSITY_RANK.get(r.get("support_intensity", "none"), 0)
            for r in records[:3]
        )
        current_score = self.INTENSITY_RANK.get(current.support_intensity, 0)
        recent_avg = recent_score / max(min(len(records), 3), 1)

        if current_score >= recent_avg + 1:
            return "declining"
        if current_score == 0 and recent_score > 0:
            return "improving"
        return "stable"

    def _count_consecutive_negative(self, records: list[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            if (
                record.get("overall_state") in self.NEGATIVE_STATES
                or record.get("current_mood") in self.NEGATIVE_MOODS
            ):
                count += 1
            else:
                break
        return count

    def _count_recent_high_support(self, records: list[dict[str, Any]]) -> int:
        return sum(
            1 for record in records[:5]
            if record.get("support_intensity") in {"high", "crisis"}
        )
