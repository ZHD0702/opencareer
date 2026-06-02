from typing import List, Dict, Optional
from datetime import datetime
import json
import sqlite3

from db.crud import (
    get_messages, save_emotion_record, get_emotion_records,
    get_skill_records, get_session
)


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
                    "trend": "stable",
                    "consecutive_negative": 0,
                    "negative_ratio": 0.0,
                    "needs_intervention": False,
                    "reason": "没有足够数据",
                    "history": []
                }
            
            recent_moods = [r.get("current_mood", "neutral") for r in records]
            negative_count = sum(1 for r in records 
                               if r.get("overall_state") in ["negative", "anxious"])
            
            ratio = negative_count / len(records) if records else 0
            consecutive_negative = 0
            
            for mood in reversed(recent_moods):
                if mood in ["negative", "anxious"]:
                    consecutive_negative += 1
                else:
                    break
            
            needs_intervention = ratio > 0.5 or consecutive_negative >= 3
            reason = self._generate_intervention_reason(ratio, consecutive_negative)
            
            return {
                "session_id": session_id,
                "current_mood": records[0].get("current_mood", "neutral"),
                "trend": self._determine_trend(records),
                "consecutive_negative": consecutive_negative,
                "negative_ratio": round(ratio, 2),
                "needs_intervention": needs_intervention,
                "reason": reason,
                "history": records[:10]
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
            
            records = get_skill_records(session_id)
            
            skills = []
            gaps = []
            total_match = 0
            
            for record in records:
                skill = {
                    "name": record.get("skill_name", ""),
                    "level": record.get("level", 0),
                    "required": record.get("required_level", 5),
                    "category": record.get("category", "general")
                }
                skills.append(skill)
                
                gap = skill["required"] - skill["level"]
                if gap > 0:
                    gaps.append({
                        "skill": skill["name"],
                        "gap": gap,
                        "suggestion": self._generate_skill_suggestion(skill)
                    })
                
                total_match += min(skill["level"], skill["required"])
            
            match_rate = int((total_match / (len(skills) * 10)) * 100) if skills else 0
            
            if not skills:
                skills = [
                    {"name": "编程能力", "level": 5, "required": 7, "category": "technical"},
                    {"name": "沟通能力", "level": 6, "required": 8, "category": "soft"},
                    {"name": "学习能力", "level": 7, "required": 8, "category": "cognitive"}
                ]
            
            return {
                "session_id": session_id,
                "target_role": session.get("target_role", "通用职业"),
                "match_rate": match_rate,
                "skills": skills,
                "gaps": gaps
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
            
            return {
                "session_id": session_id,
                "data": {
                    "grade_level": None,
                    "major": None,
                    "school": None,
                    "target_role": session.get("target_role"),
                    "job_search_stage": "exploring",
                    "skill_focus": [],
                    "common_concerns": [],
                    "background_summary": None
                },
                "last_updated": datetime.utcnow().isoformat()
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
        
        return self.get_resume(session_id)
    
    def _determine_trend(self, records: List[Dict]) -> str:
        """判断情绪趋势"""
        if len(records) < 3:
            return "stable"
        
        recent = records[:3]
        older = records[3:6] if len(records) > 6 else records
        
        recent_negative = sum(1 for r in recent 
                            if r.get("overall_state") in ["negative", "anxious"])
        older_negative = sum(1 for r in older 
                           if r.get("overall_state") in ["negative", "anxious"])
        
        if recent_negative < older_negative:
            return "improving"
        elif recent_negative > older_negative:
            return "worsening"
        else:
            return "stable"
    
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
