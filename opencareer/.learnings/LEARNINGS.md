# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice
**Areas**: frontend | backend | infra | tests | docs | config
**Statuses**: pending | in_progress | resolved | wont_fix | promoted | promoted_to_skill

## Status Definitions

| Status | Meaning |
|--------|---------|
| `pending` | Not yet addressed |
| `in_progress` | Actively being worked on |
| `resolved` | Issue fixed or knowledge integrated |
| `wont_fix` | Decided not to address (reason in Resolution) |
| `promoted` | Elevated to CLAUDE.md, AGENTS.md, or copilot-instructions.md |
| `promoted_to_skill` | Extracted as a reusable skill |

## Learning Entry Format

```markdown
## [LRN-YYYYMMDD-XXX] category

**Logged**: ISO-8601 timestamp
**Priority**: low | medium | high | critical
**Status**: pending
**Area**: frontend | backend | infra | tests | docs | config

### Summary
One-line description of what was learned

### Details
Full context: what happened, what was wrong, what's correct

### Suggested Action
Specific fix or improvement to make

### Metadata
- Source: conversation | error | user_feedback
- Related Files: path/to/file.ext
- Tags: tag1, tag2
- See Also: LRN-20250110-001 (if related to existing entry)
- Pattern-Key: simplify.dead_code | harden.input_validation (optional, for recurring-pattern tracking)
- Recurrence-Count: 1 (optional)
- First-Seen: 2025-01-15 (optional)
- Last-Seen: 2025-01-15 (optional)

---
```

## [LRN-20260418-001] best_practice

**Logged**: 2026-04-18T10:35:00Z
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Skill installation troubleshooting: copy skill to global skills directory when not recognized

### Details
When installing skills via SkillHub, sometimes skills are not automatically recognized by Claude Code. The skill `self-improving-agent` was installed but not recognized initially. The fix was to copy the skill from the project's `skills/` directory to the global skills directory `~/.claude/skills/`. This ensures the skill is available across all projects.

### Suggested Action
When a skill is not recognized after installation, check if it exists in the global skills directory. If not, copy it manually.

### Metadata
- Source: error
- Related Files: ~/.claude/skills/self-improving-agent/SKILL.md, F:\AI Career Companion\OpenCareer\skills\self-improving-agent\SKILL.md
- Tags: skillhub, claude-code, skill-installation
- See Also: ERR-20260418-001
- Pattern-Key: infra.skill_installation
- Recurrence-Count: 1
- First-Seen: 2026-04-18
- Last-Seen: 2026-04-18

---

```