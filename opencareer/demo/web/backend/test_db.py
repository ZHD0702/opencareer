#!/usr/bin/env python3
"""
数据库测试脚本
测试 Phase 1 任务 2: 数据库配置
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.crud import (
    init_sync_db, create_session, get_session, update_session,
    save_message, get_messages, save_emotion_record, get_emotion_records,
    get_skill_records
)
from datetime import datetime, timedelta
import uuid


def test_database():
    print("=" * 60)
    print("Phase 1 任务 2: 数据库配置测试")
    print("=" * 60)
    
    # 1. 初始化数据库
    print("\n1. 初始化数据库...")
    try:
        init_sync_db()
        print("   ✅ 数据库初始化成功")
    except Exception as e:
        print(f"   ❌ 数据库初始化失败: {e}")
        return False
    
    # 2. 测试会话管理
    print("\n2. 测试会话管理...")
    
    session_id = str(uuid.uuid4())
    user_id = "test_user_001"
    target_role = "前端开发工程师"
    
    try:
        session = create_session(session_id, user_id, target_role)
        print(f"   ✅ 创建会话成功: {session['session_id']}")
    except Exception as e:
        print(f"   ❌ 创建会话失败: {e}")
        return False
    
    try:
        retrieved = get_session(session_id)
        if retrieved and retrieved["user_id"] == user_id:
            print(f"   ✅ 获取会话成功: {retrieved['user_id']}")
        else:
            print("   ❌ 获取会话失败")
            return False
    except Exception as e:
        print(f"   ❌ 获取会话失败: {e}")
        return False
    
    try:
        update_session(session_id, target_role="全栈开发工程师")
        updated = get_session(session_id)
        if updated and updated["target_role"] == "全栈开发工程师":
            print(f"   ✅ 更新会话成功: 目标岗位已变更")
        else:
            print("   ❌ 更新会话失败")
            return False
    except Exception as e:
        print(f"   ❌ 更新会话失败: {e}")
        return False
    
    # 3. 测试消息存储
    print("\n3. 测试消息存储...")
    try:
        save_message(session_id, "user", "我想找一份前端开发工作", "career_advice")
        save_message(session_id, "ai", "理解你的需求，让我来帮你...", "career_advice")
        print("   ✅ 保存消息成功")
    except Exception as e:
        print(f"   ❌ 保存消息失败: {e}")
        return False
    
    try:
        messages = get_messages(session_id, limit=10)
        if len(messages) >= 2:
            print(f"   ✅ 获取消息成功: {len(messages)} 条")
            for msg in messages:
                print(f"      - {msg['role']}: {msg['content'][:30]}...")
        else:
            print("   ❌ 获取消息失败")
            return False
    except Exception as e:
        print(f"   ❌ 获取消息失败: {e}")
        return False
    
    # 4. 测试情绪记录
    print("\n4. 测试情绪记录...")
    try:
        save_emotion_record(
            session_id,
            "neutral",
            "期待",
            ["希望", "期待"],
            0.85,
            "career_advice",
            "medium"
        )
        print("   ✅ 保存情绪记录成功")
    except Exception as e:
        print(f"   ❌ 保存情绪记录失败: {e}")
        return False
    
    try:
        emotions = get_emotion_records(session_id, limit=10)
        if len(emotions) >= 1:
            print(f"   ✅ 获取情绪记录成功: {len(emotions)} 条")
        else:
            print("   ❌ 获取情绪记录失败")
            return False
    except Exception as e:
        print(f"   ❌ 获取情绪记录失败: {e}")
        return False
    
    # 5. 测试技能记录（占位）
    print("\n5. 测试技能记录...")
    try:
        skills = get_skill_records(session_id)
        print(f"   ✅ 获取技能记录成功: {len(skills)} 条")
    except Exception as e:
        print(f"   ❌ 获取技能记录失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有数据库测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)
