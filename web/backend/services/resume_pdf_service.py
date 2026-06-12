from __future__ import annotations

from pathlib import Path
import re
import shutil
from datetime import datetime
from typing import Dict, Optional

from db.crud import (
    get_resume_document,
    get_resume_document_by_id,
    list_resume_documents,
    save_resume_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESUME_OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "resume").resolve()


def snapshot_resume_pdfs() -> Dict[str, int]:
    if not RESUME_OUTPUT_DIR.exists():
        return {}

    snapshot: Dict[str, int] = {}
    for path in RESUME_OUTPUT_DIR.rglob("*.pdf"):
        try:
            resolved = path.resolve()
            if "_history" in resolved.parts:
                continue
            snapshot[str(resolved)] = resolved.stat().st_mtime_ns
        except OSError:
            continue
    return snapshot


def find_new_resume_pdf(before: Dict[str, int]) -> Optional[Path]:
    after = snapshot_resume_pdfs()
    changed = [
        Path(path)
        for path, modified_at in after.items()
        if path not in before or before[path] != modified_at
    ]
    if not changed:
        return None
    return max(changed, key=lambda path: path.stat().st_mtime_ns)


def register_resume_pdf(session_id: str, path: Path) -> dict:
    safe_path = validate_resume_pdf_path(path)
    safe_session_id = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
    history_dir = RESUME_OUTPUT_DIR / "_history" / safe_session_id
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived_path = history_dir / f"{timestamp}_{safe_path.name}"
    shutil.copy2(safe_path, archived_path)
    return save_resume_document(session_id, str(archived_path), safe_path.name)


def get_session_resume_pdf(session_id: str) -> Optional[dict]:
    document = get_resume_document(session_id)
    if not document:
        return None

    try:
        path = validate_resume_pdf_path(Path(document["file_path"]))
    except (FileNotFoundError, ValueError):
        return None

    return {**document, "path": path}


def get_session_resume_pdf_by_id(session_id: str, document_id: int) -> Optional[dict]:
    document = get_resume_document_by_id(session_id, document_id)
    if not document:
        return None
    try:
        path = validate_resume_pdf_path(Path(document["file_path"]))
    except (FileNotFoundError, ValueError):
        return None
    return {**document, "path": path}


def list_session_resume_pdfs(session_id: str) -> list[dict]:
    documents = []
    for document in list_resume_documents(session_id):
        try:
            path = validate_resume_pdf_path(Path(document["file_path"]))
        except (FileNotFoundError, ValueError):
            continue
        documents.append({**document, "path": path})
    return documents


def validate_resume_pdf_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("Only PDF resume files can be served.")
    if not resolved.is_relative_to(RESUME_OUTPUT_DIR):
        raise ValueError("Resume PDF is outside the allowed output directory.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved
