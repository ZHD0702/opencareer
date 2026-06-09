---
name: skill-template
description: "Template for creating new SKILLs. Provides directory structure conventions and SKILL.md writing standards."
---

# SKILL Template

Use this template when creating a new SKILL for OpenCareer. It defines the required structure and documentation format.

## Directory Structure

```
skill-name/
├── SKILL.md          # (Required) YAML frontmatter + Markdown instructions
├── scripts/          # (Optional) Executable code for deterministic/repetitive tasks
├── references/       # (Optional) Reference documents loaded on demand
└── assets/           # (Optional) Output templates, icons, fonts
```

## Workflow

### Step 1: Determine Call Intent

When an agent invokes this SKILL, first validate input parameters:
- Check required parameters are present
- Ask user for missing information if needed
- Confirm operation mode if multiple modes exist

### Step 2: Execute Core Logic

Execute business logic based on input parameters:

**Standard processing flow:**
1. Validate parameter format and range
2. Load auxiliary materials from `references/` if available
3. Call executable scripts from `scripts/` for data processing if needed
4. Generate structured output

### Step 3: Assemble Output

Assemble execution results into a unified output format, then respond in Chinese.

### Step 4: Return and Log

Return results to the calling agent. The logging agent records this invocation.

## Output Format

用中文回复。返回结构：
- `result`: Core result data
- `metadata`: Execution metadata (duration, version, etc.)

## Creating a New SKILL

1. Create `skill-name/` directory under `skills/`
2. Create `SKILL.md` with YAML frontmatter and markdown instructions
3. (Optional) Add executable code in `scripts/`
4. (Optional) Add reference documents in `references/`
5. (Optional) Add templates or resources in `assets/`
