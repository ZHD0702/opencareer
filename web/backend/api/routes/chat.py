from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
import asyncio
import re

from api.exceptions import SessionNotFoundException, LLMServiceException
from db.crud import save_message, get_messages, get_session, update_session
from agent_factory import get_agent_factory
from careers_config import config
from services.chat_style import GUI_FRIENDLY_STYLE_PROMPT, apply_gui_style_to_career_agent
from services.emotion_guard import EmotionGuard
from services.resume_builder_service import ResumeBuilderService
from services.session_title_service import generate_session_title
from services.resume_pdf_service import find_new_resume_pdf, register_resume_pdf, snapshot_resume_pdfs
from services.career_tracking_service import get_skill_evidence_chains, update_career_tracking
from services.job_search_readiness import build_job_match_action
from services.mcp_resume_service import (
    build_resume_skill_payload,
    build_incomplete_resume_response,
    call_resume_skill,
    is_resume_pdf_request,
    validate_resume_payload,
)
from utils.text_fragmenter import TextFragmenter

router = APIRouter()
logger = logging.getLogger(__name__)

# 全局 CareerAgent 实例管理（每个会话一个实例）
_career_agents = {}
_session_chat_locks: dict[str, asyncio.Lock] = {}

_PROFILE_ENRICHMENT_MARKERS = (
    "简历", "求职", "岗位", "职位", "实习", "工作", "项目", "经历", "负责",
    "专业", "学校", "学历", "薪资", "期望", "城市", "技能", "熟悉", "使用过",
    "开发", "后端", "前端", "产品", "运营", "设计", "面试", "投递",
)


def _get_session_chat_lock(session_id: str) -> asyncio.Lock:
    lock = _session_chat_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_chat_locks[session_id] = lock
    return lock


def _build_resume_chat_hint(resume_update: dict, include_question: bool = True) -> str:
    changes = resume_update.get("changes") or []
    questions = resume_update.get("next_questions") or []

    if not changes:
        return ""

    if questions and include_question:
        return questions[0]

    return ""


def _contains_question(text: str) -> bool:
    return "？" in text or "?" in text


def _strip_internal_tool_markers(text: str) -> str:
    return re.sub(r"\s*\[调用工具\s*[:：][^\]]+\]\s*", "\n", text or "").strip()


def _needs_profile_enrichment(text: str) -> bool:
    normalized = (text or "").strip()
    if len(normalized) < 4:
        return False
    return any(marker in normalized for marker in _PROFILE_ENRICHMENT_MARKERS)


async def _get_career_agent(session_id: str):
    """获取或创建 CareerAgent 实例"""
    if session_id not in _career_agents:
        factory = get_agent_factory()
        agent = factory.create_agent(
            "career",
            mcp_url=config.MCP_URL,
            use_mcp=config.USE_MCP,
            memory_file=str(config.get_session_memory_path(session_id))
        )
        apply_gui_style_to_career_agent(agent)
        
        # 连接 MCP（如果启用）
        if hasattr(agent, "connect_mcp") and config.USE_MCP:
            try:
                await asyncio.wait_for(agent.connect_mcp(), timeout=7)
            except asyncio.TimeoutError:
                logger.warning("MCP connection timed out for session %s; continuing without tools", session_id)
            except Exception as e:
                logger.warning(f"Failed to connect MCP for session {session_id}: {e}")
        
        _career_agents[session_id] = agent
    
    return _career_agents[session_id]


@router.post("/chat/{session_id}")
async def chat_stream(session_id: str, request: Request):
    """
    SSE 流式对话接口 - 微信风格碎片化聊天
    
    优先使用 CareerAgent（Phase 3），带有完整的 LangChain 记忆系统
    支持智能断句，自动创建多个气泡
    """
    chat_lock: asyncio.Lock | None = None
    lock_acquired = False
    try:
        logger.info(f"收到聊天请求: session_id={session_id}")
        
        body = await request.json()
        user_message = body.get("message", "")
        
        logger.info(f"用户消息: \"{user_message}\"")
        
        if not user_message or not user_message.strip():
            logger.warning("消息内容为空")
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        if len(user_message) > 2000:
            logger.warning(f"消息过长: {len(user_message)} 字符")
            raise HTTPException(status_code=400, detail="消息内容过长，请控制在2000字以内")
        
        session = get_session(session_id)
        if not session:
            logger.warning(f"会话不存在: {session_id}")
            raise SessionNotFoundException(session_id)
        
        chat_lock = _get_session_chat_lock(session_id)
        if chat_lock.locked():
            raise HTTPException(status_code=409, detail="上一条消息仍在处理中，请稍候")
        await chat_lock.acquire()
        lock_acquired = True

        logger.info(f"保存用户消息到数据库")
        save_message(session_id, "user", user_message.strip())
        recent_messages = get_messages(session_id, limit=20)
        user_message_count = len([msg for msg in recent_messages if msg["role"] == "user"])
        if user_message_count == 1 and not session.get("title"):
            update_session(session_id, title=generate_session_title(user_message.strip()))

        emotion_guard = EmotionGuard()
        emotion_assessment = emotion_guard.assess(session_id, user_message.strip())
        resume_pdf_requested = is_resume_pdf_request(user_message)
        resume_service = ResumeBuilderService()
        if resume_pdf_requested:
            resume_update = resume_service.update_from_user_message(session_id, user_message.strip())
            tracking_update = {
                "session_id": session_id,
                "updated": False,
                "follow_up": None,
            }
        elif _needs_profile_enrichment(user_message):
            try:
                resume_update = await asyncio.wait_for(
                    resume_service.update_from_user_message_async(session_id, user_message.strip()),
                    timeout=13,
                )
            except asyncio.TimeoutError:
                logger.warning("Resume enrichment timed out for session %s; using local extraction", session_id)
                resume_update = resume_service.update_from_user_message(session_id, user_message.strip())
            try:
                tracking_update = await asyncio.wait_for(
                    update_career_tracking(
                        session_id,
                        user_message.strip(),
                        resume_update,
                        allow_follow_up=not emotion_assessment.should_intervene,
                    ),
                    timeout=13,
                )
            except asyncio.TimeoutError:
                logger.warning("Career tracking enrichment timed out for session %s", session_id)
                tracking_update = {
                    "session_id": session_id,
                    "updated": False,
                    "follow_up": None,
                }
        else:
            resume_update = resume_service.update_from_user_message(session_id, user_message.strip())
            tracking_update = {
                "session_id": session_id,
                "updated": False,
                "follow_up": None,
            }
        job_match_action = build_job_match_action(
            session_id,
            allow_action=not emotion_assessment.should_intervene,
        )
        pdf_snapshot = snapshot_resume_pdfs()
        
        # 获取 CareerAgent 实例（优先使用）
        agent = None
        if not emotion_assessment.should_intervene and not resume_pdf_requested:
            agent = await _get_career_agent(session_id)
        
        logger.info("开始调用 Agent 生成回复")
        
        # 收集完整的 AI 回复用于保存到数据库
        full_ai_response = []
        
        async def event_generator():
            try:
                # 先发送"对方正在输入..."信号
                yield f"data: {json.dumps({'type': 'typing_start'})}\n\n"
                emotion_event = emotion_assessment.to_event()
                emotion_event["session_id"] = session_id
                yield f"data: {json.dumps({'type': 'emotion_analysis', 'data': emotion_event}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'resume_update', 'data': resume_update}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'career_tracking_update', 'data': tracking_update}, ensure_ascii=False)}\n\n"
                
                # 检查是否是 CareerAgent
                if emotion_assessment.should_intervene:
                    logger.info(
                        "Emotion intervention triggered for session %s: %s",
                        session_id,
                        emotion_assessment.reason,
                    )
                    full_ai_response.append(
                        emotion_guard.build_support_response(emotion_assessment, user_message)
                    )
                elif resume_pdf_requested:
                    logger.info("Explicit PDF resume request detected; invoking resume_skill via MCP")
                    try:
                        resume_state = resume_update.get("state") or ResumeBuilderService().load_state(session_id)
                        payload = build_resume_skill_payload(
                            resume_state,
                            get_skill_evidence_chains(session_id),
                        )
                        validation = validate_resume_payload(payload)
                        if not validation["complete"]:
                            full_ai_response.append(build_incomplete_resume_response(validation))
                        else:
                            result = await call_resume_skill(payload)
                            filename = result.get("pdf_path", "").replace("\\", "/").rsplit("/", 1)[-1]
                            full_ai_response.append(
                                f"好，信息已经完整了，我把 PDF 简历生成好了。{filename or 'PDF 简历'}会在右侧打开，你也可以从左侧简历记录里再次查看或下载。"
                            )
                    except Exception as exc:
                        logger.error("Explicit resume PDF generation failed: %s", exc, exc_info=True)
                        full_ai_response.append(
                            "这次 PDF 没有生成成功，MCP 的简历工具连接出了问题。我已经保留了当前简历信息，请稍后重试，不需要重新讲一遍。"
                        )
                elif hasattr(agent, "stream_chat"):
                    # CareerAgent - 使用完整的 LangChain 记忆系统
                    logger.info("使用 CareerAgent (带有 LangChain 记忆系统)")
                    
                    async def collect_agent_response():
                        async for token in agent.stream_chat(user_message):
                            if token != "__FRAGMENT_BREAK__":
                                full_ai_response.append(token)

                    try:
                        await asyncio.wait_for(collect_agent_response(), timeout=50)
                    except asyncio.TimeoutError:
                        logger.error("CareerAgent response timed out for session %s", session_id)
                        full_ai_response.append(
                            "这次模型回复等待太久，已经先停下来了。你可以再发一次，我会接着当前对话继续。"
                        )
                    
                else:
                    # SimpleAgent 回退
                    logger.info("使用 SimpleAgent (回退)")
                    
                    messages = get_messages(session_id, limit=50)
                    history = [
                        {"role": "assistant" if msg["role"] == "ai" else msg["role"], "content": msg["content"]}
                        for msg in messages
                        if msg["role"] in ["user", "ai"]
                    ]
                    
                    if hasattr(agent, "chat"):
                        async for event in agent.chat(session_id, history, GUI_FRIENDLY_STYLE_PROMPT):
                            if event.startswith('data: '):
                                try:
                                    data_str = event[6:].strip()
                                    if data_str:
                                        data = json.loads(data_str)
                                        if data.get('type') == 'sentence':
                                            full_ai_response.append(data.get('content', ''))
                                except:
                                    pass
                            # 不发送事件，只收集内容

                skill_follow_up = tracking_update.get("follow_up")
                agent_text = "".join(full_ai_response)
                if (
                    skill_follow_up
                    and not _contains_question(agent_text)
                    and not emotion_assessment.should_intervene
                    and not resume_pdf_requested
                ):
                    full_ai_response.append(f"\n\n{skill_follow_up['question']}")
                
                # 发送"对方正在输入结束"信号
                yield f"data: {json.dumps({'type': 'typing_end'})}\n\n"
                
                # 对回复进行碎片化处理
                if full_ai_response:
                    full_text = _strip_internal_tool_markers("".join(full_ai_response))
                    logger.info(f"AI回复完成: {len(full_text)} 字符，开始碎片化处理")
                    
                    # 使用碎片化器处理文本（每次创建新实例确保加载最新代码）
                    fragmenter = TextFragmenter()
                    fragments = fragmenter.fragment_with_particles(
                        full_text,
                        user_message=user_message,
                    )
                    
                    logger.info(f"碎片化完成: 生成 {len(fragments)} 个碎片")
                    
                    # 发送碎片化信号
                    yield f"data: {json.dumps({'type': 'fragmentation_start', 'count': len(fragments)})}\n\n"
                    
                    # 逐个发送碎片（带延迟）
                    for i, fragment in enumerate(fragments):
                        # 发送碎片
                        yield f"data: {json.dumps({
                                'type': 'fragment',
                                'content': fragment.content,
                                'emotion': fragment.emotion,
                                'index': i,
                                'kind': fragment.kind,
                                'mode': fragment.mode,
                                'delay_ms': fragment.delay_ms,
                            })}\n\n"
                        
                        # 碎片之间添加延迟（最后一个碎片不需要延迟）
                        if i < len(fragments) - 1:
                            yield f"data: {json.dumps({'type': 'typing_start'})}\n\n"
                            await asyncio.sleep(fragment.delay_ms / 1000.0)
                            yield f"data: {json.dumps({'type': 'typing_end'})}\n\n"
                    
                    # 发送碎片化结束信号
                    yield f"data: {json.dumps({'type': 'fragmentation_end'})}\n\n"
                    
                    # 保存原始回复到数据库（不含语气词）
                    logger.info(f"保存 AI 回复到数据库: {len(full_text)} 字符")
                    message_actions = [job_match_action] if job_match_action else []
                    saved_message_id = save_message(session_id, "ai", full_text, actions=message_actions)
                    if job_match_action:
                        yield f"data: {json.dumps({'type': 'assistant_action', 'message_id': str(saved_message_id), 'data': job_match_action}, ensure_ascii=False)}\n\n"

                generated_pdf = find_new_resume_pdf(pdf_snapshot)
                if generated_pdf:
                    document = register_resume_pdf(session_id, generated_pdf)
                    version = document.get("updated_at") or document.get("created_at") or ""
                    pdf_event = {
                        "id": document["id"],
                        "session_id": session_id,
                        "filename": document["file_name"],
                        "created_at": document.get("created_at"),
                        "preview_url": f"/api/resume/{session_id}/pdfs/{document['id']}/content?v={version}",
                        "download_url": f"/api/resume/{session_id}/pdfs/{document['id']}/download",
                    }
                    yield f"data: {json.dumps({'type': 'resume_pdf', 'data': pdf_event}, ensure_ascii=False)}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"事件生成器错误: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'typing_end'})}\n\n"
                yield f"data: {json.dumps({'type': 'error', 'message': '这次回复没有处理成功，请再试一次。'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            finally:
                if lock_acquired and chat_lock.locked():
                    chat_lock.release()
        
        logger.info("返回 StreamingResponse")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except SessionNotFoundException as e:
        if lock_acquired and chat_lock and chat_lock.locked():
            chat_lock.release()
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except LLMServiceException as e:
        if lock_acquired and chat_lock and chat_lock.locked():
            chat_lock.release()
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        if lock_acquired and chat_lock and chat_lock.locked():
            chat_lock.release()
        raise
    except Exception as e:
        if lock_acquired and chat_lock and chat_lock.locked():
            chat_lock.release()
        logger.error(f"聊天接口错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")


@router.get("/chat/memory/{session_id}")
async def get_chat_memory(session_id: str):
    """获取对话记忆摘要（兼容原接口）"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "get_memory_summary"):
            memory_summary = agent.get_memory_summary()
            return {
                "success": True,
                "memory": memory_summary,
                "agent_type": "career_agent",
                "has_langchain_memory": True
            }
        else:
            return {
                "success": True,
                "memory": None,
                "agent_type": "simple_agent",
                "has_langchain_memory": False,
                "message": "当前使用的 Agent 不支持 LangChain 记忆功能"
            }
    except Exception as e:
        logger.error(f"获取记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "memory": None,
            "message": str(e)
        }


@router.delete("/chat/memory/{session_id}")
async def clear_chat_memory(session_id: str):
    """清除对话记忆（兼容原接口）"""
    try:
        agent = await _get_career_agent(session_id)
        
        if hasattr(agent, "clear_memory"):
            agent.clear_memory()
            return {
                "success": True,
                "message": "记忆已清除",
                "agent_type": "career_agent"
            }
        else:
            return {
                "success": False,
                "message": "当前使用的 Agent 不支持记忆清除功能",
                "agent_type": "simple_agent"
            }
    except Exception as e:
        logger.error(f"清除记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e)
        }
