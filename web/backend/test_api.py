"""
OpenCareer API 单元测试

测试覆盖：
1. 会话管理
2. 聊天功能
3. 情绪分析
4. 技能评估
5. 简历管理
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from fastapi.testclient import TestClient
from main import app
from db.crud import init_sync_db, create_session, get_session
from utils.text_fragmenter import TextFragmenter
from api.exceptions import (
    SessionNotFoundException,
    LLMServiceException,
    ValidationException
)

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """测试前初始化数据库"""
    init_sync_db()
    yield


class TestSessionAPI:
    """会话管理API测试"""
    
    def test_create_session(self):
        """测试创建会话"""
        response = client.post(
            "/api/sessions",
            json={"user_id": "test_user", "target_role": "软件工程师"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["user_id"] == "test_user"
        assert data["target_role"] == "软件工程师"
    
    def test_create_session_empty_user_id(self):
        """测试空用户ID"""
        response = client.post(
            "/api/sessions",
            json={"user_id": ""}
        )
        assert response.status_code == 400
    
    def test_get_session(self):
        """测试获取会话"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_2"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
    
    def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        response = client.get("/api/sessions/nonexistent_id")
        assert response.status_code == 404


class TestChatAPI:
    """聊天API测试"""
    
    def test_empty_message(self):
        """测试空消息"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_chat"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.post(
            f"/api/chat/{session_id}",
            json={"message": ""}
        )
        assert response.status_code == 400
    
    def test_message_too_long(self):
        """测试消息过长"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_long"}
        )
        session_id = create_response.json()["session_id"]
        
        long_message = "a" * 3000
        response = client.post(
            f"/api/chat/{session_id}",
            json={"message": long_message}
        )
        assert response.status_code == 400
    
    def test_chat_nonexistent_session(self):
        """测试向不存在的会话发送消息"""
        response = client.post(
            "/api/chat/nonexistent_session",
            json={"message": "你好"}
        )
        assert response.status_code == 404


class TestEmotionAPI:
    """情绪API测试"""
    
    def test_get_emotion_trends(self):
        """测试获取情绪趋势"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_emotion"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.get(f"/api/emotion/trends/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "current_mood" in data
        assert "trend" in data
    
    def test_get_emotion_nonexistent_session(self):
        """测试获取不存在会话的情绪趋势"""
        response = client.get("/api/emotion/trends/nonexistent")
        assert response.status_code == 404


class TestSkillAPI:
    """技能API测试"""
    
    def test_get_skill_assessment(self):
        """测试获取技能评估"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_skill"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.get(f"/api/skill-assessment/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert "match_rate" in data
    
    def test_update_skill_assessment(self):
        """测试更新技能评估"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_skill_update"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.patch(
            f"/api/skill-assessment/{session_id}",
            params={"target_role": "前端工程师"}
        )
        assert response.status_code == 200


class TestResumeAPI:
    """简历API测试"""
    
    def test_get_resume(self):
        """测试获取简历"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_resume"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.get(f"/api/resume/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "grade_level" in data["data"]
    
    def test_update_resume(self):
        """测试更新简历"""
        create_response = client.post(
            "/api/sessions",
            json={"user_id": "test_user_resume_update"}
        )
        session_id = create_response.json()["session_id"]
        
        response = client.patch(
            f"/api/resume/{session_id}",
            json={
                "grade_level": "本科",
                "major": "计算机科学",
                "school": "清华大学"
            }
        )
        assert response.status_code == 200


class TestHealthCheck:
    """健康检查测试"""
    
    def test_root(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_health(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestTextFragmenter:
    """文本碎片化测试"""
    
    def test_split_short_text(self):
        """测试短文本分割"""
        fragmenter = TextFragmenter()
        text = "你好，我是职业顾问。"
        sentences = fragmenter.split(text)
        
        assert len(sentences) >= 1
        assert sentences[0].content == "你好"
    
    def test_split_long_text(self):
        """测试长文本分割"""
        fragmenter = TextFragmenter()
        text = "今天天气真好！我们要加油学习，找到好工作。"
        sentences = fragmenter.split(text)
        
        assert len(sentences) >= 2
    
    def test_split_empty_text(self):
        """测试空文本"""
        fragmenter = TextFragmenter()
        sentences = fragmenter.split("")
        
        assert len(sentences) == 0


class TestExceptions:
    """异常测试"""
    
    def test_session_not_found_exception(self):
        """测试会话不存在异常"""
        exc = SessionNotFoundException("test_id")
        assert exc.status_code == 404
        assert "test_id" in exc.message
    
    def test_llm_service_exception(self):
        """测试LLM服务异常"""
        exc = LLMServiceException()
        assert exc.status_code == 503
        assert "不可用" in exc.message
    
    def test_validation_exception(self):
        """测试验证异常"""
        exc = ValidationException("测试错误", field="test_field")
        assert exc.status_code == 422
        assert exc.field == "test_field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
