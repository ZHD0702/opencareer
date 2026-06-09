"""
Persona definition for the OpenCareer conversational agents.

Provides a shared communication identity that all conversational agents
(BrainAgent, WorkAgent, EmotionAgent) adhere to. This ensures consistent
tone, style, and behavioral principles regardless of which agent is
currently handling the conversation.

Usage:
    from agents.persona import PERSONA_SYSTEM_PROMPT

    # Use as the base system prompt for any conversational agent:
    system_prompt = PERSONA_SYSTEM_PROMPT + agent_specific_instructions

All prompts are loaded from YAML files via PromptRegistry,
enabling easy editing without touching Python code.
"""

from opencareer.prompts.registry import get_global_registry

_registry = get_global_registry()


def _load_base_section(key: str) -> str:
    """Load a section from ``personas.base``, falling back to empty string."""
    return _registry.get(f"personas.base.{key}", "")


# ---------------------------------------------------------------------------
# Core identity — who the agent is
# ---------------------------------------------------------------------------

AGENT_IDENTITY = _load_base_section("agent_identity")

# ---------------------------------------------------------------------------
# Communication principles
# ---------------------------------------------------------------------------

COMMUNICATION_PRINCIPLES = _load_base_section("communication_principles")

# ---------------------------------------------------------------------------
# Behavioral guidelines
# ---------------------------------------------------------------------------

BEHAVIORAL_GUIDELINES = _load_base_section("behavioral_guidelines")

# ---------------------------------------------------------------------------
# Response structure template
# ---------------------------------------------------------------------------

RESPONSE_STRUCTURE = _load_base_section("response_structure")

# ---------------------------------------------------------------------------
# Complete system prompt — use as the base for all conversational agents
# ---------------------------------------------------------------------------

PERSONA_SYSTEM_PROMPT = "\n\n".join(
    filter(None, [AGENT_IDENTITY, COMMUNICATION_PRINCIPLES,
                  BEHAVIORAL_GUIDELINES, RESPONSE_STRUCTURE])
)

# ---------------------------------------------------------------------------
# Agent-specific additions — append to PERSONA_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

BRAIN_SPECIFIC = _registry.get("personas.brain.agent_specific", "")
WORK_SPECIFIC = _registry.get("personas.work.agent_specific", "")
EMOTION_SPECIFIC = _registry.get("personas.emotion.agent_specific", "")


def get_agent_persona(agent_name: str) -> str:
    """Get the full persona prompt for a specific agent.

    Args:
        agent_name: ``"brain"``, ``"work_agent"``, or ``"emotion_agent"``

    Returns:
        Complete system prompt string combining shared persona + agent-specific rules.
    """
    additions = {
        "brain": BRAIN_SPECIFIC,
        "work_agent": WORK_SPECIFIC,
        "emotion_agent": EMOTION_SPECIFIC,
    }
    extra = additions.get(agent_name, "")
    return f"{PERSONA_SYSTEM_PROMPT}\n\n{extra}" if extra else PERSONA_SYSTEM_PROMPT
