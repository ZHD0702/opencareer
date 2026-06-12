from typing import List, Dict, Optional
from datetime import datetime
import json
import sqlite3

from db.crud import (
    get_messages, save_emotion_record, get_emotion_records,
    get_skill_records, get_pending_skill_follow_up, get_session, list_skill_evidence, upsert_skill_evidence
)
from services.resume_builder_service import ResumeBuilderService
from services.career_tracking_service import get_skill_evidence_chains


class AnalysisService:
    """分析服务 - 提供情绪、技能、简历等分析功能"""
    
    def __init__(self):
        self.db_path = "./data/app.db"
    
    def get_emotion_trends(self, session_id: str) -> Optional[Dict]:
        """
        获取情绪趋势分析
        
        Args:
            session_id: 会话 ID
        
        Returns:
            情绪趋势数据字典
        """
        try:
            records = get_emotion_records(session_id, limit=20)
            
            if not records:
                return {
                    "session_id": session_id,
                    "current_mood": "neutral",
                    "current_overall_state": "neutral",
                    "current_emotions": [],
                    "confidence": 0.0,
                    "support_intensity": "none",
                    "suggested_action": "work",
                    "trend": "stable",
                    "consecutive_negative": 0,
                    "negative_ratio": 0.0,
                    "needs_intervention": False,
                    "reason": "没有足够数据",
                    "history": []
                }
            
            negative_states = {"negative", "crisis"}
            negative_moods = {"negative", "anxious", "distressed", "crisis", "uneasy", "angry"}
            elevated_support = {"medium", "high", "crisis"}
            latest = records[0]
            negative_count = sum(
                1 for r in records
                if r.get("overall_state") in negative_states
                or r.get("support_intensity") in elevated_support
            )
            
            ratio = negative_count / len(records) if records else 0
            consecutive_negative = 0
            
            for record in records:
                mood = record.get("current_mood", "neutral")
                if mood in negative_moods:
                    consecutive_negative += 1
                else:
                    break
            
            recent_high = sum(
                1 for r in records[:5]
                if r.get("support_intensity") in {"high", "crisis"}
            )
            needs_intervention = (
                ratio > 0.5
                or consecutive_negative >= 3
                or recent_high >= 2
                or any(r.get("support_intensity") == "crisis" for r in records[:3])
            )
            reason = self._generate_intervention_reason(ratio, consecutive_negative)
            
            return {
                "session_id": session_id,
                "current_mood": latest.get("current_mood", "neutral"),
                "current_overall_state": latest.get("overall_state", "neutral"),
                "current_emotions": latest.get("emotions", []),
                "confidence": latest.get("confidence") or 0.0,
                "support_intensity": latest.get("support_intensity", "none"),
                "suggested_action": latest.get("demand_type", "work"),
                "trend": self._determine_trend(records),
                "consecutive_negative": consecutive_negative,
                "negative_ratio": round(ratio, 2),
                "needs_intervention": needs_intervention,
                "reason": reason,
                "history": [self._normalize_emotion_record(record) for record in records[:10]]
            }
        except Exception as e:
            print(f"[AnalysisService] get_emotion_trends error: {e}")
            return None
    
    def get_skill_assessment(self, session_id: str) -> Optional[Dict]:
        """
        获取技能评估
        
        Args:
            session_id: 会话 ID
        
        Returns:
            技能评估数据字典
        """
        try:
            session = get_session(session_id)
            if not session:
                return None
            
            skills = list_skill_evidence(session_id)
            chains = get_skill_evidence_chains(session_id)
            for skill in skills:
                skill["evidence_chain"] = chains.get(skill["skill_name"], [])
            counts = {
                "proven": sum(1 for item in skills if item["status"] == "proven"),
                "mentioned": sum(1 for item in skills if item["status"] == "mentioned"),
                "gap": sum(1 for item in skills if item["status"] == "gap"),
            }
            
            return {
                "session_id": session_id,
                "target_role": session.get("target_role", "通用职业"),
                "counts": counts,
                "skills": skills,
                "gaps": [item for item in skills if item["status"] == "gap"],
                "pending_follow_up": get_pending_skill_follow_up(session_id),
            }
        except Exception as e:
            print(f"[AnalysisService] get_skill_assessment error: {e}")
            return None
    
    def update_skill_assessment(self, session_id: str, target_role: Optional[str] = None, 
                               skills: Optional[List] = None) -> Optional[Dict]:
        """
        更新技能评估
        
        Args:
            session_id: 会话 ID
            target_role: 目标角色
            skills: 技能列表
        
        Returns:
            更新后的技能评估
        """
        if target_role:
            from db.crud import update_session
            update_session(session_id, target_role=target_role)
        for skill in skills or []:
            name = skill.get("skill_name") or skill.get("name")
            if name:
                upsert_skill_evidence(session_id, {**skill, "skill_name": name, "source": "manual"})
        
        return self.get_skill_assessment(session_id)
    
    def get_resume(self, session_id: str) -> Optional[Dict]:
        """
        获取简历信息
        
        Args:
            session_id: 会话 ID
        
        Returns:
            简历信息字典
        """
        try:
            session = get_session(session_id)
            if not session:
                return None

            resume_service = ResumeBuilderService()
            state = resume_service.load_state(session_id)
            data = resume_service.to_resume_response_data(state)

            if session.get("target_role") and not data.get("target_role"):
                data["target_role"] = session.get("target_role")

            return {
                "session_id": session_id,
                "data": data,
                "last_updated": state.get("updated_at") or datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"[AnalysisService] get_resume error: {e}")
            return None
    
    def update_resume(self, session_id: str, data: Dict) -> Optional[Dict]:
        """
        更新简历信息
        
        Args:
            session_id: 会话 ID
            data: 简历数据
        
        Returns:
            更新后的简历信息
        """
        if data.get("target_role"):
            from db.crud import update_session
            update_session(session_id, target_role=data.get("target_role"))

        ResumeBuilderService().apply_manual_update(session_id, data)
        return self.get_resume(session_id)
    
    def _determine_trend(self, records: List[Dict]) -> str:
        """判断情绪趋势"""
        if len(records) < 3:
            return "stable"
        
        recent = records[:3]
        older = records[3:6] if len(records) > 6 else records
        
        negative_states = {"negative", "crisis"}
        elevated_support = {"medium", "high", "crisis"}
        recent_negative = sum(
            1 for r in recent
            if r.get("overall_state") in negative_states
            or r.get("support_intensity") in elevated_support
        )
        older_negative = sum(
            1 for r in older
            if r.get("overall_state") in negative_states
            or r.get("support_intensity") in elevated_support
        )
        
        if recent_negative < older_negative:
            return "improving"
        elif recent_negative > older_negative:
            return "declining"
        else:
            return "stable"

    def _normalize_emotion_record(self, record: Dict) -> Dict:
        item = dict(record)
        item["timestamp"] = item.get("created_at", "")
        return item
    
    def _generate_intervention_reason(self, ratio: float, consecutive: int) -> str:
        """生成干预原因说明"""
        if ratio > 0.7:
            return "近期负面情绪较多，建议进行积极干预"
        elif consecutive >= 3:
            return f"连续 {consecutive} 次负面情绪，需要关注"
        elif ratio > 0.5:
            return "负面情绪比例偏高，注意情绪状态"
        else:
            return "情绪状态整体稳定"
    
    def _generate_skill_suggestion(self, skill: Dict) -> str:
        """生成技能提升建议"""
        gap = skill["required"] - skill["level"]
        
        if gap <= 2:
            return "保持练习，稳步提升即可"
        elif gap <= 5:
            return "建议集中学习，提升较快"
        else:
            return "需要重点关注，制定学习计划"
