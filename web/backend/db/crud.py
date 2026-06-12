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
            title TEXT,
            target_role TEXT,
            current_phase TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(sessions)")
    session_columns = {row[1] for row in cursor.fetchall()}
    if "title" not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
    
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            company TEXT NOT NULL,
            role TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT 'saved',
            next_action TEXT,
            deadline TEXT,
            jd_text TEXT,
            note TEXT,
            source TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            status TEXT NOT NULL,
            category TEXT DEFAULT '通用',
            evidence TEXT,
            requirement TEXT,
            suggestion TEXT,
            source TEXT DEFAULT 'conversation',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, skill_name),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_evidence_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            context_key TEXT,
            scenario TEXT,
            task TEXT,
            action TEXT,
            result TEXT,
            metric TEXT,
            user_role TEXT,
            used_at TEXT,
            raw_text TEXT,
            source TEXT DEFAULT 'conversation',
            confidence REAL DEFAULT 0.5,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_follow_ups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            evidence_id INTEGER,
            skill_name TEXT NOT NULL,
            missing_field TEXT NOT NULL,
            question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            asked_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (evidence_id) REFERENCES skill_evidence_items(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_states (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute("PRAGMA table_info(resume_documents)")
    resume_document_columns = {row[1] for row in cursor.fetchall()}
    if resume_document_columns and "id" not in resume_document_columns:
        cursor.execute("ALTER TABLE resume_documents RENAME TO resume_documents_legacy")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resume_documents_session_created
        ON resume_documents(session_id, created_at DESC)
    """)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='resume_documents_legacy'")
    if cursor.fetchone():
        cursor.execute("""
            INSERT INTO resume_documents (session_id, file_path, file_name, created_at)
            SELECT session_id, file_path, file_name, COALESCE(updated_at, created_at)
            FROM resume_documents_legacy
        """)
        cursor.execute("DROP TABLE resume_documents_legacy")
    
    conn.commit()
    conn.close()

def create_session(session_id: str, user_id: str, target_role: Optional[str] = None) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO sessions (id, user_id, title, target_role, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, user_id, None, target_role, now, now))
    
    conn.commit()
    conn.close()
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "title": None,
        "target_role": target_role,
        "created_at": now,
        "updated_at": now
    }

def get_session(session_id: str) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, user_id, title, target_role, current_phase, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "session_id": row[0],
        "user_id": row[1],
        "title": row[2],
        "target_role": row[3],
        "current_phase": row[4],
        "created_at": row[5],
        "updated_at": row[6]
    }

def list_sessions(user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    safe_limit = max(1, min(limit, 500))
    if user_id:
        cursor.execute("""
            SELECT id, user_id, title, target_role, current_phase, created_at, updated_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
        """, (user_id, safe_limit))
    else:
        cursor.execute("""
            SELECT id, user_id, title, target_role, current_phase, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
        """, (safe_limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "session_id": row[0],
            "user_id": row[1],
            "title": row[2],
            "target_role": row[3],
            "current_phase": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }
        for row in rows
    ]

def delete_session(session_id: str) -> bool:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    exists = cursor.fetchone() is not None
    if exists:
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM emotion_records WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM skill_records WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM skill_follow_ups WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM skill_evidence_items WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM skill_evidence WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM job_applications WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM resume_states WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM resume_documents WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

    conn.close()
    return exists


def save_resume_document(session_id: str, file_path: str, file_name: str) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO resume_documents (session_id, file_path, file_name, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, file_path, file_name, now))
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": document_id,
        "session_id": session_id,
        "file_path": file_path,
        "file_name": file_name,
        "created_at": now,
    }


def get_resume_document(session_id: str) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, file_path, file_name, created_at
        FROM resume_documents
        WHERE session_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "session_id": row[1],
        "file_path": row[2],
        "file_name": row[3],
        "created_at": row[4],
    }


def get_resume_document_by_id(session_id: str, document_id: int) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, file_path, file_name, created_at
        FROM resume_documents
        WHERE session_id = ? AND id = ?
    """, (session_id, document_id))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None
    return {
        "id": row[0],
        "session_id": row[1],
        "file_path": row[2],
        "file_name": row[3],
        "created_at": row[4],
    }


def list_resume_documents(session_id: str) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_id, file_path, file_name, created_at
        FROM resume_documents
        WHERE session_id = ?
        ORDER BY created_at DESC, id DESC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "file_path": row[2],
            "file_name": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]

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

    cursor.execute("""
        UPDATE sessions
        SET updated_at = ?
        WHERE id = ?
    """, (now, session_id))
    
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


def list_job_applications(session_id: str) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM job_applications
        WHERE session_id = ?
        ORDER BY updated_at DESC, id DESC
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_job_application(session_id: str, data: Dict) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().isoformat()
    card_id = data.get("id")
    if card_id:
        allowed = ["company", "role", "stage", "next_action", "deadline", "jd_text", "note", "source"]
        fields = [key for key in allowed if key in data]
        if fields:
            values = [data[key] for key in fields]
            assignments = ", ".join(f"{key} = ?" for key in fields)
            conn.execute(
                f"UPDATE job_applications SET {assignments}, updated_at = ? WHERE id = ? AND session_id = ?",
                (*values, now, card_id, session_id),
            )
    else:
        cursor = conn.execute("""
            INSERT INTO job_applications
            (session_id, company, role, stage, next_action, deadline, jd_text, note, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, data.get("company", "待确认公司"), data.get("role", "待确认岗位"),
            data.get("stage", "saved"), data.get("next_action"), data.get("deadline"),
            data.get("jd_text"), data.get("note"), data.get("source", "manual"), now, now,
        ))
        card_id = cursor.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM job_applications WHERE id = ?", (card_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def delete_job_application(session_id: str, card_id: int) -> bool:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("DELETE FROM job_applications WHERE id = ? AND session_id = ?", (card_id, session_id))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def upsert_skill_evidence(session_id: str, data: Dict) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO skill_evidence
        (session_id, skill_name, status, category, evidence, requirement, suggestion, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, skill_name) DO UPDATE SET
            status = CASE
                WHEN skill_evidence.status = 'proven' AND excluded.status = 'gap' THEN 'proven'
                ELSE excluded.status
            END,
            category = excluded.category,
            evidence = COALESCE(excluded.evidence, skill_evidence.evidence),
            requirement = COALESCE(excluded.requirement, skill_evidence.requirement),
            suggestion = COALESCE(excluded.suggestion, skill_evidence.suggestion),
            source = excluded.source,
            updated_at = excluded.updated_at
    """, (
        session_id, data["skill_name"], data.get("status", "mentioned"), data.get("category", "通用"),
        data.get("evidence"), data.get("requirement"), data.get("suggestion"), data.get("source", "conversation"),
        now, now,
    ))
    conn.commit()
    row = conn.execute(
        "SELECT * FROM skill_evidence WHERE session_id = ? AND skill_name = ?",
        (session_id, data["skill_name"]),
    ).fetchone()
    conn.close()
    return dict(row)


def list_skill_evidence(session_id: str) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM skill_evidence
        WHERE session_id = ?
        ORDER BY CASE status WHEN 'gap' THEN 0 WHEN 'mentioned' THEN 1 ELSE 2 END, updated_at DESC
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_skill_evidence_item(session_id: str, data: Dict) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().isoformat()
    evidence_id = data.get("id")
    fields = (
        "skill_name", "context_key", "scenario", "task", "action", "result",
        "metric", "user_role", "used_at", "raw_text", "source", "confidence",
    )
    if evidence_id:
        updates = [key for key in fields if key in data and data.get(key) is not None]
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE skill_evidence_items SET {assignments}, updated_at = ? WHERE id = ? AND session_id = ?",
                (*[data[key] for key in updates], now, evidence_id, session_id),
            )
    else:
        cursor = conn.execute("""
            INSERT INTO skill_evidence_items
            (session_id, skill_name, context_key, scenario, task, action, result, metric,
             user_role, used_at, raw_text, source, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, data["skill_name"], data.get("context_key"), data.get("scenario"),
            data.get("task"), data.get("action"), data.get("result"), data.get("metric"),
            data.get("user_role"), data.get("used_at"), data.get("raw_text"),
            data.get("source", "conversation"), data.get("confidence", 0.5), now, now,
        ))
        evidence_id = cursor.lastrowid
    conn.commit()
    row = conn.execute(
        "SELECT * FROM skill_evidence_items WHERE id = ? AND session_id = ?",
        (evidence_id, session_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def list_skill_evidence_items(session_id: str, skill_name: Optional[str] = None) -> List[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if skill_name:
        rows = conn.execute("""
            SELECT * FROM skill_evidence_items
            WHERE session_id = ? AND skill_name = ?
            ORDER BY updated_at DESC, id DESC
        """, (session_id, skill_name)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM skill_evidence_items
            WHERE session_id = ?
            ORDER BY updated_at DESC, id DESC
        """, (session_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_pending_skill_follow_up(session_id: str) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT * FROM skill_follow_ups
        WHERE session_id = ? AND status = 'pending'
        ORDER BY updated_at DESC, id DESC LIMIT 1
    """, (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_skill_follow_up(session_id: str, data: Dict) -> Dict:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE skill_follow_ups SET status = 'superseded', updated_at = ? WHERE session_id = ? AND status = 'pending'",
        (now, session_id),
    )
    cursor = conn.execute("""
        INSERT INTO skill_follow_ups
        (session_id, evidence_id, skill_name, missing_field, question, status, asked_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
    """, (
        session_id, data.get("evidence_id"), data["skill_name"], data["missing_field"],
        data["question"], data.get("asked_count", 0), now, now,
    ))
    conn.commit()
    row = conn.execute("SELECT * FROM skill_follow_ups WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def resolve_skill_follow_up(session_id: str, follow_up_id: int) -> None:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE skill_follow_ups SET status = 'answered', updated_at = ? WHERE id = ? AND session_id = ?",
        (datetime.utcnow().isoformat(), follow_up_id, session_id),
    )
    conn.commit()
    conn.close()

def get_resume_state(session_id: str) -> Optional[Dict]:
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT state_json, updated_at
        FROM resume_states
        WHERE session_id = ?
    """, (session_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    try:
        state = json.loads(row[0])
    except json.JSONDecodeError:
        state = {}

    state["updated_at"] = row[1]
    return state

def save_resume_state(session_id: str, state: Dict):
    init_sync_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()
    payload = dict(state)
    payload.pop("updated_at", None)

    cursor.execute("""
        INSERT INTO resume_states (session_id, state_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
    """, (session_id, json.dumps(payload, ensure_ascii=False), now))

    conn.commit()
    conn.close()

init_sync_db()
