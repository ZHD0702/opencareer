"""MultiAgent Orchestrator API Routes"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json
import logging

from agents.orchestrator import get_orchestrator
from llm.registry import list_available_llms, get_llm_adapter
from knowledge.knowledge_base import get_knowledge_base

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/orchestrator/chat/{session_id}")
async def orchestrator_chat_stream(session_id: str, request: Request):
    """
    MultiAgent 编排器流式对话接口
    
    根据用户意图自动选择最合适的 Agent 来处理请求
    """
    try:
        logger.info(f"Orchestrator chat request: session_id={session_id}")
        
        body = await request.json()
        user_message = body.get("message", "")
        
        if not user_message or not user_message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        if len(user_message) > 2000:
            raise HTTPException(status_code=400, detail="消息内容过长，请控制在2000字以内")
        
        orchestrator = get_orchestrator()
        
        full_response = []
        
        async def event_generator():
            try:
                async for token in orchestrator.route(user_message):
                    full_response.append(token)
                    yield f"data: {json.dumps({'type': 'sentence', 'content': token})}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Orchestrator error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orchestrator API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/orchestrator/delegate/{session_id}")
async def delegate_to_agent(session_id: str, request: Request):
    """
    直接委托给指定类型的 Agent
    """
    try:
        body = await request.json()
        user_message = body.get("message", "")
        agent_type = body.get("agent_type", "general")
        
        if not user_message or not user_message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        orchestrator = get_orchestrator()
        
        full_response = []
        
        async def event_generator():
            try:
                async for token in orchestrator.delegate(user_message, agent_type):
                    full_response.append(token)
                    yield f"data: {json.dumps({'type': 'sentence', 'content': token})}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Delegate error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delegate API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/orchestrator/agents")
async def get_available_agents():
    """获取所有可用的 Agent 列表"""
    try:
        orchestrator = get_orchestrator()
        agents = orchestrator.get_available_agents()
        
        result = []
        for agent_type in agents:
            info = orchestrator.get_agent_info(agent_type)
            result.append({
                "type": agent_type,
                **info
            })
        
        return {"success": True, "agents": result}
    
    except Exception as e:
        logger.error(f"Get agents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取 Agent 列表失败")


@router.get("/llm/providers")
async def get_llm_providers():
    """获取所有可用的 LLM 提供者"""
    try:
        providers = list_available_llms()
        return {"success": True, "providers": providers}
    
    except Exception as e:
        logger.error(f"Get LLM providers error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取 LLM 提供者失败")


@router.get("/llm/health/{provider}")
async def check_llm_health(provider: str):
    """检查指定 LLM 提供者的健康状态"""
    try:
        adapter = get_llm_adapter(provider)
        
        # 尝试调用 invoke 方法检查连接
        test_response = await adapter.invoke([{"role": "user", "content": "Hello"}])
        
        return {
            "success": True,
            "provider": provider,
            "status": "healthy",
            "test_response": test_response[:50] + "..."
        }
    
    except Exception as e:
        logger.warning(f"LLM health check failed for {provider}: {e}")
        return {
            "success": False,
            "provider": provider,
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/knowledge/search")
async def search_knowledge(request: Request):
    """搜索知识库"""
    try:
        body = await request.json()
        query = body.get("query", "")
        max_results = body.get("max_results", 5)
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        kb = get_knowledge_base()
        results = kb.search(query, k=max_results)
        
        return {"success": True, "results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="搜索失败")


@router.post("/knowledge/query")
async def query_knowledge(request: Request):
    """查询知识库并返回格式化结果"""
    try:
        body = await request.json()
        query = body.get("query", "")
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        kb = get_knowledge_base()
        result = kb.query(query)
        
        return {"success": True, "answer": result}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="查询失败")


@router.post("/knowledge/add")
async def add_to_knowledge(request: Request):
    """添加文档到知识库"""
    try:
        body = await request.json()
        content = body.get("content", "")
        metadata = body.get("metadata", {})
        
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="文档内容不能为空")
        
        kb = get_knowledge_base()
        kb.add_document(content, metadata)
        
        stats = kb.get_stats()
        
        return {"success": True, "message": "文档添加成功", "stats": stats}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="添加文档失败")


@router.get("/knowledge/stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        kb = get_knowledge_base()
        stats = kb.get_stats()
        
        return {"success": True, "stats": stats}
    
    except Exception as e:
        logger.error(f"Get knowledge stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取统计信息失败")


@router.delete("/knowledge/clear")
async def clear_knowledge():
    """清空知识库"""
    try:
        kb = get_knowledge_base()
        kb.clear()
        
        return {"success": True, "message": "知识库已清空"}
    
    except Exception as e:
        logger.error(f"Clear knowledge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="清空知识库失败")