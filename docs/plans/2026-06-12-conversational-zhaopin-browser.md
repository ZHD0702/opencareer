# Conversational Zhaopin Browser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Offer a persistent “匹配岗位” chat action when the user profile is ready, then run an automated Zhaopin search in a right-side Playwright browser panel.

**Architecture:** A deterministic readiness service combines resume requirements, proven skill evidence and explicit job-search intent. Chat actions are stored with assistant messages. A backend Playwright session manager owns isolated browser contexts and exposes status, screenshots and lifecycle endpoints to a polling React side panel.

**Tech Stack:** FastAPI, SQLite, Playwright Python, React, TypeScript, Zustand.

---

### Task 1: Readiness And Persistent Actions

- Extend messages with JSON actions.
- Detect explicit job-search intent from conversation history.
- Require target role, city, background and sufficient skill evidence.
- Attach a structured action to the assistant response when ready.

### Task 2: Browser Session Service

- Create one isolated Playwright context per OpenCareer session.
- Open Zhaopin and automate keyword and city search.
- Record task status and current URL.
- Provide PNG screenshots without exposing browser internals.
- Stop on login or verification challenges and request user intervention.

### Task 3: Browser API

- Add create, status, screenshot and stop endpoints.
- Validate session ownership and search-plan fields.
- Close all browser contexts during application shutdown.

### Task 4: Chat And Sidebar UI

- Render structured actions below the final assistant bubble.
- Open a Codex-style right panel when matching starts.
- Poll task state and screenshot frames.
- Provide refresh, stop, open-original-page and close controls.

### Task 5: Verification

- Unit-test readiness rules and action persistence.
- Compile backend modules and build the frontend.
- Launch Chromium locally and verify screenshot generation.
