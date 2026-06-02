import sqlite3
import json
from typing import Optional, List, Dict
from datetime import datetime
import os

DB_PATH = "./data/app.db"

def init_sync_db():
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            target_role TEXT,
            current_phase TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            intent TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            overall_state TEXT,
            current_mood TEXT,
            emotions TEXT,
            confidence REAL,
            demand_type TEXT,
            support_intensity TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            level INTEGER,
            required_level INTEGER,
            category TEXT,
            source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

def create_session(session_id: str, user_id: str, target_role: Optional[str] = None) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO sessions (id, user_id, target_role, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, user_id, target_role, now, now))
    
    conn.commit()
    conn.close()
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "target_role": target_role,
        "created_at": now,
        "updated_at": now
    }

def get_session(session_id: str) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, user_id, target_role, current_phase, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "session_id": row[0],
        "user_id": row[1],
        "target_role": row[2],
        "current_phase": row[3],
        "created_at": row[4],
        "updated_at": row[5]
    }

def update_session(session_id: str, **kwargs):
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    for key, value in kwargs.items():
        if value is not None:
            updates.append(f"{key} = ?")
            values.append(value)
    
    if updates:
        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(session_id)
        
        cursor.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
    
    conn.close()

def save_message(session_id: str, role: str, content: str, intent: Optional[str] = None):
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, intent, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, content, intent, now))
    
    conn.commit()
    conn.close()

def get_messages(session_id: str, limit: int = 50) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, session_id, role, content, intent, created_at 
        FROM messages 
        WHERE session_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "intent": row[4],
            "created_at": row[5]
        }
        for row in rows
    ]

def get_emotion_records(session_id: str, limit: int = 20) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, session_id, overall_state, current_mood, emotions, confidence, demand_type, support_intensity, created_at
        FROM emotion_records 
        WHERE session_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "overall_state": row[2],
            "current_mood": row[3],
            "emotions": json.loads(row[4]) if row[4] else [],
            "confidence": row[5],
            "demand_type": row[6],
            "support_intensity": row[7],
            "created_at": row[8]
        }
        for row in rows
    ]

def save_emotion_record(session_id: str, overall_state: str, current_mood: str, emotions: List[str], confidence: float, demand_type: str, support_intensity: str):
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO emotion_records (session_id, overall_state, current_mood, emotions, confidence, demand_type, support_intensity, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, overall_state, current_mood, json.dumps(emotions), confidence, demand_type, support_intensity, now))
    
    conn.commit()
    conn.close()

def get_skill_records(session_id: str) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, session_id, skill_name, level, required_level, category, source, created_at
        FROM skill_records 
        WHERE session_id = ?
    """, (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "skill_name": row[2],
            "level": row[3],
            "required_level": row[4],
            "category": row[5],
            "source": row[6],
            "created_at": row[7]
        }
        for row in rows
    ]

init_sync_db()
