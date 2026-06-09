"""
OpenCareer Web API Server

FastAPI application serving the Web GUI backend on port 8080.
Provides REST endpoints for session management, emotion trends,
resume data, and a WebSocket endpoint for streaming chat.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure opencareer package is importable
_demo_dir = Path(__file__).resolve().parent.parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

from web_api.models import (
    EmotionTrendItem,
    EmotionTrendsResponse,
    GapItem,
    HealthResponse,
    JobCard,
    JobCardCreateRequest,
    JobCardMoveRequest,
    JobProgressResponse,
    ResumeData,
    ResumeResponse,
    ResumeUpdateRequest,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfo,
    SkillAssessmentResponse,
    SkillAssessmentUpdateRequest,
    SkillItem,
    WsMessage,
)
from web_api.session_store import session_store
from web_api.ws_manager import ws_manager
from web_api.agent_pipeline import AgentPipeline
from web_api.conversation_context_adapter import ConversationContextAdapter
from opencareer.agents.llm_client import create_llm_client

logger = logging.getLogger("web_api.server")

SYSTEM_PROMPT = """你是 AI Career Companion，一个专业的职业发展助手。你可以帮助用户进行：

- 职业规划与建议：分析用户背景，提供职业发展方向建议
- 简历优化：帮助用户改进简历内容和格式
- 面试准备：提供面试技巧、模拟面试问题
- 职业心理支持：倾听用户的职业焦虑，提供情绪支持
- 求职策略：建议求职渠道、薪资谈判技巧

请用中文回复，语气友好、专业、富有同理心。回答要具体、有针对性，避免空洞的套话。"""

# Lazy-initialized LLM client (created on first use)
_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client()
        if _llm_client is None:
            logger.warning("LLM client not available — DEEPSEEK_API_KEY not set")
    return _llm_client

# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("Web API server starting on port 8080")
    yield
    logger.info("Web API server shutting down")
    # Cleanup all sessions
    for sid in session_store.active_sessions:
        ws_manager.remove_handler(sid)
        session_store.delete(sid)


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenCareer Web API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow Vite dev server and local access
    cors_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:8080,http://localhost:3000",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # REST endpoints
    # ------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse()

    @app.post("/api/sessions", response_model=SessionCreateResponse)
    async def create_session(req: SessionCreateRequest):
        """Create a new conversation session."""
        result = session_store.create(user_id=req.user_id)
        return SessionCreateResponse(**result)

    @app.get("/api/sessions/{session_id}", response_model=SessionInfo)
    async def get_session(session_id: str):
        """Get session metadata."""
        info = session_store.get_info(session_id)
        if info is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)
        return SessionInfo(**info)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        ws_manager.remove_handler(session_id)
        existed = session_store.delete(session_id)
        return {"deleted": existed}

    @app.get("/api/sessions")
    async def list_sessions():
        """List all active sessions."""
        return session_store.list_sessions()

    # ------------------------------------------------------------------
    # Emotion trends
    # ------------------------------------------------------------------

    @app.get("/api/emotion/trends/{session_id}", response_model=EmotionTrendsResponse)
    async def get_emotion_trends(session_id: str):
        """Get emotion trends data for the sidebar visualization.

        Reads emotion_history and mood trends from the ConversationContext
        and returns structured data for the EmotionTab component.
        """
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        # Build emotion history items
        history_items = []
        for entry in ctx.emotion_history:
            emotions = entry.get("emotions", [])
            if isinstance(emotions, list):
                emotion_list = [
                    e.get("label", str(e)) if isinstance(e, dict) else str(e)
                    for e in emotions
                ]
            else:
                emotion_list = []

            history_items.append(EmotionTrendItem(
                emotions=emotion_list,
                overall_state=entry.get("overall_state", "neutral"),
                support_intensity=entry.get("support_intensity", "none"),
                demand_type=entry.get("demand_type", "unknown"),
                confidence=entry.get("confidence", 0.0),
                timestamp=entry.get("timestamp", ""),
            ))

        # Get mood trends
        mood_trends = ctx.detect_mood_trends()

        # Get accumulated emotion assessment
        accumulated = ctx.get_accumulated_emotion_assessment()

        trend = mood_trends.get("trend", "insufficient_data")
        current_mood = mood_trends.get("current_mood")

        # Determine if intervention is needed
        needs_intervention = (
            mood_trends.get("needs_intervention", False)
            or accumulated.get("should_defer_work", False)
        )

        return EmotionTrendsResponse(
            session_id=session_id,
            current_mood=current_mood,
            trend=trend,
            consecutive_negative=mood_trends.get("consecutive_low", 0),
            negative_ratio=accumulated.get("negative_ratio", 0.0),
            needs_intervention=needs_intervention,
            reason=accumulated.get("reason", ""),
            history=history_items,
        )

    # ------------------------------------------------------------------
    # Resume / Profile
    # ------------------------------------------------------------------

    @app.get("/api/resume/{session_id}", response_model=ResumeResponse)
    async def get_resume(session_id: str):
        """Get the user's resume/profile data extracted from conversation.

        Returns fields from ConversationContext.user_profile that have been
        passively extracted by agents during the conversation.
        """
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        profile = ctx.user_profile

        resume_data = ResumeData(
            grade_level=profile.get("grade_level"),
            major=profile.get("major"),
            school=profile.get("school"),
            target_role=profile.get("target_role"),
            job_search_stage=profile.get("job_search_stage"),
            skill_focus=list(profile.get("skill_focus", [])),
            common_concerns=list(profile.get("common_concerns", [])),
            background_summary=profile.get("background_summary"),
        )

        # Use most recent conversation turn timestamp if available
        last_updated = ""
        if ctx.history:
            last_updated = ctx.history[-1].get("timestamp", "")

        return ResumeResponse(
            session_id=session_id,
            data=resume_data,
            last_updated=last_updated,
        )

    @app.patch("/api/resume/{session_id}", response_model=ResumeResponse)
    async def update_resume(session_id: str, req: ResumeUpdateRequest):
        """Update specific profile fields with user-edited values.

        Accepts a partial update — only non-None fields in the request
        are written to the user_profile.
        """
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        profile = ctx.user_profile

        # Map request fields to profile keys (only update if value is not None)
        field_map = {
            "grade_level": req.grade_level,
            "major": req.major,
            "school": req.school,
            "target_role": req.target_role,
            "job_search_stage": req.job_search_stage,
            "skill_focus": req.skill_focus,
            "common_concerns": req.common_concerns,
            "background_summary": req.background_summary,
        }

        for key, value in field_map.items():
            if value is not None:
                profile[key] = value

        logger.info(f"Resume updated for session={session_id}: {[k for k, v in field_map.items() if v is not None]}")

        resume_data = ResumeData(
            grade_level=profile.get("grade_level"),
            major=profile.get("major"),
            school=profile.get("school"),
            target_role=profile.get("target_role"),
            job_search_stage=profile.get("job_search_stage"),
            skill_focus=list(profile.get("skill_focus", [])),
            common_concerns=list(profile.get("common_concerns", [])),
            background_summary=profile.get("background_summary"),
        )

        last_updated = ""
        if ctx.history:
            last_updated = ctx.history[-1].get("timestamp", "")

        return ResumeResponse(
            session_id=session_id,
            data=resume_data,
            last_updated=last_updated,
        )

    # ------------------------------------------------------------------
    # Job Progress
    # ------------------------------------------------------------------

    def _find_card(session_id: str, card_id: str) -> tuple:
        """Find a job card across all stages. Returns (stage_name, card_dict) or (None, None)."""
        ctx = session_store.get(session_id)
        if ctx is None:
            return None, None
        for stage, cards in ctx.job_progress.items():
            for card in cards:
                if card.get("id") == card_id:
                    return stage, card
        return None, None

    @app.get("/api/job-progress/{session_id}", response_model=JobProgressResponse)
    async def get_job_progress(session_id: str):
        """Get all job application cards grouped by stage."""
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        stages = {
            stage: [JobCard(**card) for card in cards]
            for stage, cards in ctx.job_progress.items()
        }
        return JobProgressResponse(session_id=session_id, stages=stages)

    @app.post("/api/job-progress/{session_id}", response_model=JobCard)
    async def create_job_card(session_id: str, req: JobCardCreateRequest):
        """Add a new job application card to a stage."""
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        if req.stage not in ctx.job_progress:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": f"Invalid stage: {req.stage}"}, status_code=400)

        import uuid
        card = {
            "id": uuid.uuid4().hex[:8],
            "company": req.company,
            "role": req.role,
            "date": req.date or "",
            "note": req.note,
        }
        ctx.job_progress[req.stage].append(card)
        logger.info(f"Job card added: {card['id']} to {req.stage} (session={session_id})")
        return JobCard(**card)

    @app.patch("/api/job-progress/{session_id}/{card_id}", response_model=JobCard)
    async def move_job_card(session_id: str, card_id: str, req: JobCardMoveRequest):
        """Move a job card to a different stage."""
        stage, card = _find_card(session_id, card_id)
        if card is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Card not found"}, status_code=404)

        ctx = session_store.get(session_id)
        if req.to_stage not in ctx.job_progress:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": f"Invalid stage: {req.to_stage}"}, status_code=400)

        # Remove from current stage, add to target stage
        ctx.job_progress[stage] = [c for c in ctx.job_progress[stage] if c.get("id") != card_id]
        ctx.job_progress[req.to_stage].append(card)
        logger.info(f"Job card {card_id} moved: {stage} -> {req.to_stage} (session={session_id})")
        return JobCard(**card)

    @app.delete("/api/job-progress/{session_id}/{card_id}")
    async def delete_job_card(session_id: str, card_id: str):
        """Delete a job application card."""
        stage, card = _find_card(session_id, card_id)
        if card is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Card not found"}, status_code=404)

        ctx = session_store.get(session_id)
        ctx.job_progress[stage] = [c for c in ctx.job_progress[stage] if c.get("id") != card_id]
        logger.info(f"Job card deleted: {card_id} from {stage} (session={session_id})")
        return {"deleted": True}

    # ------------------------------------------------------------------
    # Skill Assessment
    # ------------------------------------------------------------------

    @app.get("/api/skill-assessment/{session_id}", response_model=SkillAssessmentResponse)
    async def get_skill_assessment(session_id: str):
        """Get skill assessment data including skills, gaps, and match rate.

        Gaps are computed server-side from the skills list:
        any skill where required > level produces a gap entry.
        """
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        sa = ctx.skill_assessment
        skills = [SkillItem(**s) for s in sa.get("skills", [])]
        target_role = sa.get("target_role", "")

        # Compute gaps
        gaps = []
        total_level = 0
        total_required = 0
        for skill in skills:
            gap = skill.required - skill.level
            if gap > 0:
                gaps.append(GapItem(
                    skill=skill.name,
                    gap=gap,
                    suggestion="",
                ))
            total_level += skill.level
            total_required += skill.required

        # Compute overall match rate
        if total_required > 0:
            match_rate = round(total_level / total_required * 100, 1)
        else:
            match_rate = 0.0

        return SkillAssessmentResponse(
            session_id=session_id,
            target_role=target_role,
            match_rate=match_rate,
            skills=skills,
            gaps=gaps,
        )

    @app.patch("/api/skill-assessment/{session_id}", response_model=SkillAssessmentResponse)
    async def update_skill_assessment(session_id: str, req: SkillAssessmentUpdateRequest):
        """Update skill assessment data (partial update)."""
        ctx = session_store.get(session_id)
        if ctx is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        sa = ctx.skill_assessment

        if req.target_role is not None:
            sa["target_role"] = req.target_role

        if req.skills is not None:
            sa["skills"] = [s.model_dump() for s in req.skills]

        logger.info(f"Skill assessment updated for session={session_id}")

        # Return full updated response (same computation as GET)
        skills = [SkillItem(**s) for s in sa.get("skills", [])]
        target_role = sa.get("target_role", "")

        gaps = []
        total_level = 0
        total_required = 0
        for skill in skills:
            gap = skill.required - skill.level
            if gap > 0:
                gaps.append(GapItem(skill=skill.name, gap=gap, suggestion=""))
            total_level += skill.level
            total_required += skill.required

        match_rate = round(total_level / total_required * 100, 1) if total_required > 0 else 0.0

        return SkillAssessmentResponse(
            session_id=session_id,
            target_role=target_role,
            match_rate=match_rate,
            skills=skills,
            gaps=gaps,
        )

    # ------------------------------------------------------------------
    # WebSocket endpoint (stub for Phase 1)
    # ------------------------------------------------------------------

    @app.websocket("/ws/{session_id}")
    async def websocket_chat(ws: WebSocket, session_id: str):
        """WebSocket endpoint for streaming chat.

        Phase 2: Streaming LLM responses via DeepSeek API.
        On "chat" message, streams tokens back to the client in real-time.
        """
        await ws_manager.connect(session_id, ws)

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg_type == "chat":
                    user_content = msg.get("content", "")
                    if not user_content:
                        continue

                    llm = get_llm_client()
                    if llm is None:
                        await ws.send_json({
                            "type": "error",
                            "message": "LLM 服务未配置 (DEEPSEEK_API_KEY 缺失)",
                        })
                        continue

                    # Get or create agent pipeline for this session (lazy init on first chat)
                    pipeline = ws_manager.get_handler(session_id)
                    if pipeline is None:
                        session = session_store.get(session_id)
                        if session is None:
                            await ws.send_json({
                                "type": "error",
                                "message": "会话未找到，请刷新页面重新创建",
                            })
                            continue
                        ctx = ConversationContextAdapter(session)
                        pipeline = AgentPipeline(llm, ctx)
                        ws_manager.set_handler(session_id, pipeline)

                    try:
                        await pipeline.process_message(user_content, ws.send_json)
                    except Exception as exc:
                        logger.error(f"Agent pipeline error: {exc}")
                        await ws.send_json({"type": "error", "message": str(exc)})

                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error(f"WebSocket error session={session_id}: {exc}")
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            await ws_manager.disconnect(session_id, ws)

    return app


# ------------------------------------------------------------------
# App instance (imported by launch_web.py)
# ------------------------------------------------------------------

app = create_app()
