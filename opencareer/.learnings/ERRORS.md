# Errors

Command failures, exceptions, and unexpected behavior captured during development.

**Priorities**: high (default) | critical
**Areas**: frontend | backend | infra | tests | docs | config
**Statuses**: pending | in_progress | resolved | wont_fix

## Error Entry Format

```markdown
## [ERR-YYYYMMDD-XXX] skill_or_command_name

**Logged**: ISO-8601 timestamp
**Priority**: high
**Status**: pending
**Area**: frontend | backend | infra | tests | docs | config

### Summary
Brief description of what failed

### Error
```
Actual error message or output
```

### Context
- Command/operation attempted
- Input or parameters used
- Environment details if relevant

### Suggested Fix
If identifiable, what might resolve this

### Metadata
- Reproducible: yes | no | unknown
- Related Files: path/to/file.ext
- See Also: ERR-20250110-001 (if recurring)

---
```

## [ERR-20260418-001] skillhub_install_superpowers_mode

**Logged**: 2026-04-18T10:30:00Z
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
skillhub install superpowers-mode failed with HTTP 404 error

### Error
```
HTTP 404: The remote skill package could not be found at https://skillhub-1388575217.cos.ap-guangzhou.myqcloud.com/skills/superpowers-mode.zip
```

### Context
- Command: `skillhub install superpowers-mode`
- Attempted with `--force` flag and updated SkillHub index
- Also tried GitHub URL template: `https://github.com/skillhub-store/superpowers-mode/archive/refs/heads/main.zip`
- Environment: Windows 11, SkillHub CLI installed

### Suggested Fix
Check if the skill exists in the SkillHub store; may need to contact skill maintainer or find alternative source.

### Metadata
- Reproducible: yes
- Related Files: ~/.skillhub/metadata.json, ~/.skillhub/config.json
- See Also: 

---

```