# Skill Evidence Chain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build explainable multi-item skill evidence chains and context-aware automatic follow-up questions.

**Architecture:** Keep `skill_evidence` as the compatible profile summary, store detailed STAR-like records in `skill_evidence_items`, and connect pending questions to records through `skill_follow_ups`. Extract facts with the configured LLM, calculate levels with deterministic backend rules, and let the emotion guard decide whether a follow-up may be shown.

**Tech Stack:** FastAPI, SQLite, Python async services, React, TypeScript, TanStack Query.

---

### Task 1: Persistence

- Add evidence-item and pending-follow-up tables.
- Add CRUD operations for creating, merging, listing and resolving records.
- Include the new tables in session deletion.

### Task 2: Extraction And Follow-Up

- Extract structured skill facts with the configured LLM.
- Provide deterministic fallback extraction.
- Merge answers into pending evidence records.
- Calculate evidence level and completeness with backend rules.
- Select no more than one follow-up question per turn.

### Task 3: Chat Integration

- Run skill extraction after emotion assessment and resume extraction.
- Disable follow-ups during emotion intervention.
- Give a skill follow-up priority over other generated questions.

### Task 4: User Interface

- Return evidence chains with the skill-assessment API.
- Show evidence fields, level, completeness and missing information.
- Allow users to prefill the chat input from the skill workspace.

### Task 5: Verification

- Test multi-turn evidence merging and level upgrades.
- Test emotion-mode suppression.
- Run Python compilation and the frontend production build.
