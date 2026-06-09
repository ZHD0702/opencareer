"""
Emotion Support SKILL for OpenCareer.

This SKILL provides emotional support and encouragement for job seekers
to help them maintain motivation and positive mindset during their job search.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base_skill import BaseSkill, SkillMetadata, SkillCategory
from ..skill_registry import SkillRegistry


class EmotionSupportSkill(BaseSkill):
    """Emotion Support SKILL for maintaining motivation and positive mindset."""

    def __init__(self):
        """Initialize Emotion Support SKILL."""
        metadata = SkillMetadata(
            name="emotion_support",
            version="1.0.0",
            description="Provide emotional support and encouragement for job seekers",
            author="OpenCareer Team",
            category=SkillCategory.EMOTION,
            tags=["emotion", "motivation", "support", "encouragement"],
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "current_mood": {
                        "type": "string",
                        "enum": ["happy", "neutral", "stressed", "anxious", "discouraged", "confident"],
                        "description": "Current emotional state"
                    },
                    "recent_experience": {
                        "type": "string",
                        "description": "Recent experience or situation description"
                    },
                    "support_type": {
                        "type": "string",
                        "enum": ["encouragement", "motivation", "advice", "validation", "listening"],
                        "description": "Type of support requested"
                    },
                    "job_search_stage": {
                        "type": "string",
                        "enum": ["starting", "applying", "interviewing", "waiting", "negotiating", "accepted"],
                        "description": "Current job search stage"
                    }
                },
                "required": ["current_mood"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "emotional_response": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "support_type": {"type": "string"},
                            "suggested_actions": {"type": "array", "items": {"type": "string"}},
                            "affirmations": {"type": "array", "items": {"type": "string"}},
                            "resources": {"type": "array", "items": {"type": "object"}}
                        }
                    },
                    "mood_improvement_tips": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "follow_up_questions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "recommended_break_time": {
                        "type": "integer",
                        "description": "Suggested break time in minutes"
                    }
                }
            },
            examples=[
                {
                    "input": {
                        "current_mood": "stressed",
                        "recent_experience": "Got rejected from a job interview",
                        "support_type": "encouragement",
                        "job_search_stage": "interviewing"
                    },
                    "output": {
                        "emotional_response": {
                            "message": "Rejections are a normal part of the job search process. Each interview is a learning experience.",
                            "support_type": "encouragement",
                            "suggested_actions": ["Take a short break", "Review what went well", "Practice common questions"],
                            "affirmations": ["You are capable and qualified", "The right opportunity will come"],
                            "resources": [{"type": "article", "title": "Coping with Job Rejection"}]
                        },
                        "mood_improvement_tips": ["Go for a walk", "Talk to a friend", "Practice deep breathing"],
                        "follow_up_questions": ["What did you learn from this interview?", "How can you improve for next time?"],
                        "recommended_break_time": 30
                    }
                }
            ]
        )
        super().__init__(metadata)
        self.responses = self._load_responses()
        self.affirmations = self._load_affirmations()
        self.resources = self._load_resources()

    def _load_responses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load emotional responses for different moods and situations.

        Returns:
            Dictionary of responses by mood and support type
        """
        return {
            "stressed": [
                {
                    "message": "Job searching can be stressful, but remember that stress is temporary. You're building resilience.",
                    "support_type": "validation",
                    "suggested_actions": ["Take deep breaths", "Prioritize tasks", "Set small achievable goals"],
                    "affirmation_type": "capability"
                },
                {
                    "message": "It's normal to feel stressed during a job search. Your feelings are valid and acknowledged.",
                    "support_type": "validation",
                    "suggested_actions": ["Write down your thoughts", "Practice mindfulness", "Get some fresh air"],
                    "affirmation_type": "validation"
                }
            ],
            "anxious": [
                {
                    "message": "Anxiety often comes from uncertainty. Focus on what you can control in your job search.",
                    "support_type": "advice",
                    "suggested_actions": ["Create a schedule", "Prepare for interviews", "Research companies"],
                    "affirmation_type": "control"
                },
                {
                    "message": "Take things one step at a time. You don't have to solve everything at once.",
                    "support_type": "encouragement",
                    "suggested_actions": ["Break tasks into smaller steps", "Celebrate small wins", "Practice self-care"],
                    "affirmation_type": "progress"
                }
            ],
            "discouraged": [
                {
                    "message": "Feeling discouraged is understandable, but remember that every 'no' brings you closer to a 'yes'.",
                    "support_type": "encouragement",
                    "suggested_actions": ["Review your accomplishments", "Update your portfolio", "Network with others"],
                    "affirmation_type": "persistence"
                },
                {
                    "message": "Your worth is not defined by job rejections. You have unique skills and value to offer.",
                    "support_type": "validation",
                    "suggested_actions": ["List your strengths", "Seek feedback", "Consider different approaches"],
                    "affirmation_type": "worth"
                }
            ],
            "neutral": [
                {
                    "message": "Maintaining a balanced mindset is key to sustainable job searching.",
                    "support_type": "motivation",
                    "suggested_actions": ["Set clear goals", "Track progress", "Stay consistent"],
                    "affirmation_type": "consistency"
                }
            ],
            "happy": [
                {
                    "message": "It's great to hear you're feeling positive! Let's build on this momentum.",
                    "support_type": "encouragement",
                    "suggested_actions": ["Network proactively", "Apply for stretch positions", "Help others in their search"],
                    "affirmation_type": "momentum"
                }
            ],
            "confident": [
                {
                    "message": "Confidence is a powerful asset in job searching. Trust in your abilities.",
                    "support_type": "validation",
                    "suggested_actions": ["Apply for dream jobs", "Negotiate confidently", "Share your knowledge"],
                    "affirmation_type": "confidence"
                }
            ]
        }

    def _load_affirmations(self) -> Dict[str, List[str]]:
        """Load affirmations by type.

        Returns:
            Dictionary of affirmations by affirmation type
        """
        return {
            "capability": [
                "You have the skills and abilities to succeed",
                "You are capable of learning and growing",
                "Your experiences have prepared you for this moment"
            ],
            "validation": [
                "Your feelings are valid and important",
                "It's okay to have ups and downs",
                "You're doing your best, and that's enough"
            ],
            "control": [
                "You can only control your actions, not the outcome",
                "Focus on what you can influence",
                "Your preparation creates confidence"
            ],
            "progress": [
                "Every step forward, no matter how small, is progress",
                "You're moving in the right direction",
                "Growth happens through consistency"
            ],
            "persistence": [
                "Persistence is more important than perfection",
                "Success often comes after multiple attempts",
                "Your determination will pay off"
            ],
            "worth": [
                "Your value extends beyond your job title",
                "You are worthy of respect and opportunity",
                "Your unique perspective matters"
            ],
            "consistency": [
                "Consistent effort leads to meaningful results",
                "Showing up every day builds momentum",
                "Small daily actions create big changes over time"
            ],
            "momentum": [
                "Positive momentum attracts positive outcomes",
                "Your energy creates opportunities",
                "Success builds upon success"
            ],
            "confidence": [
                "You deserve to feel confident in your abilities",
                "Confidence comes from preparation and practice",
                "Believe in yourself as others believe in you"
            ]
        }

    def _load_resources(self) -> List[Dict[str, Any]]:
        """Load emotional support resources.

        Returns:
            List of support resources
        """
        return [
            {
                "type": "article",
                "title": "Managing Job Search Stress",
                "description": "Tips for maintaining mental health during job search",
                "url": "https://example.com/stress-management",
                "duration_minutes": 10
            },
            {
                "type": "exercise",
                "title": "5-Minute Mindfulness",
                "description": "Quick mindfulness exercise to reduce anxiety",
                "instructions": "Find a quiet place, focus on your breath for 5 minutes",
                "duration_minutes": 5
            },
            {
                "type": "activity",
                "title": "Gratitude Journaling",
                "description": "Write down three things you're grateful for each day",
                "instructions": "Keep a daily journal of positive experiences",
                "duration_minutes": 5
            },
            {
                "type": "community",
                "title": "Job Search Support Groups",
                "description": "Connect with others going through similar experiences",
                "url": "https://example.com/support-groups",
                "duration_minutes": 60
            }
        ]

    async def _initialize(self) -> None:
        """Initialize the SKILL."""
        self.logger.info("Initializing Emotion Support SKILL")
        # Load additional resources, connect to databases, etc.
        # For demo purposes, we'll just simulate loading
        await asyncio.sleep(0.1)
        self.logger.info("Emotion Support SKILL initialized")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the emotion support skill.

        Args:
            input_data: Input data containing mood and context
            context: Additional context (user info, session, etc.)

        Returns:
            Emotional support response
        """
        # Validate input
        await self.validate_input(input_data)

        try:
            self.logger.info(f"Providing emotion support for mood: {input_data.get('current_mood')}")

            # Extract input parameters
            user_id = input_data.get("user_id", "anonymous")
            current_mood = input_data.get("current_mood", "neutral")
            recent_experience = input_data.get("recent_experience", "")
            support_type = input_data.get("support_type", "encouragement")
            job_search_stage = input_data.get("job_search_stage", "applying")

            # Select appropriate response
            response = self._select_response(current_mood, support_type, recent_experience, job_search_stage)

            # Select affirmations
            affirmations = self._select_affirmations(response.get("affirmation_type", "validation"))

            # Select resources
            resources = self._select_resources(current_mood, support_type)

            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(current_mood, recent_experience, job_search_stage)

            # Calculate recommended break time
            recommended_break_time = self._calculate_break_time(current_mood)

            # Build response
            result = {
                "emotional_response": {
                    "message": response["message"],
                    "support_type": response["support_type"],
                    "suggested_actions": response["suggested_actions"],
                    "affirmations": affirmations,
                    "resources": resources[:2]  # Limit to 2 resources
                },
                "mood_improvement_tips": self._get_mood_improvement_tips(current_mood),
                "follow_up_questions": follow_up_questions,
                "recommended_break_time": recommended_break_time,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "mood_tracking": {
                    "initial_mood": current_mood,
                    "suggested_next_check": "tomorrow"
                }
            }

            self.logger.info(f"Emotion support provided successfully for user: {user_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error providing emotion support: {e}")
            raise RuntimeError(f"Failed to provide emotion support: {str(e)}")

    def _select_response(self, mood: str, support_type: str,
                        recent_experience: str, job_search_stage: str) -> Dict[str, Any]:
        """Select an appropriate emotional response.

        Args:
            mood: Current mood
            support_type: Requested support type
            recent_experience: Recent experience description
            job_search_stage: Current job search stage

        Returns:
            Selected response dictionary
        """
        # Get responses for the current mood
        mood_responses = self.responses.get(mood, self.responses.get("neutral", []))

        if not mood_responses:
            # Default response
            return {
                "message": "I'm here to support you in your job search journey.",
                "support_type": "listening",
                "suggested_actions": ["Take a moment to breathe", "Express how you're feeling"],
                "affirmation_type": "validation"
            }

        # Filter by support type if specified and not "listening"
        if support_type and support_type != "listening":
            filtered_responses = [
                r for r in mood_responses
                if r.get("support_type") == support_type
            ]
            if filtered_responses:
                return random.choice(filtered_responses)

        # Otherwise return a random response for the mood
        return random.choice(mood_responses)

    def _select_affirmations(self, affirmation_type: str) -> List[str]:
        """Select affirmations of a specific type.

        Args:
            affirmation_type: Type of affirmations needed

        Returns:
            List of affirmations
        """
        affirmations = self.affirmations.get(affirmation_type, self.affirmations.get("validation", []))
        if not affirmations:
            return ["You're doing your best, and that's enough"]

        # Return 2-3 random affirmations
        count = min(3, len(affirmations))
        return random.sample(affirmations, count)

    def _select_resources(self, mood: str, support_type: str) -> List[Dict[str, Any]]:
        """Select appropriate resources.

        Args:
            mood: Current mood
            support_type: Support type needed

        Returns:
            List of resources
        """
        # For stressed/anxious moods, recommend mindfulness resources
        if mood in ["stressed", "anxious"]:
            return [r for r in self.resources if r["type"] in ["exercise", "article"]]

        # For discouraged moods, recommend community and articles
        if mood == "discouraged":
            return [r for r in self.resources if r["type"] in ["community", "article"]]

        # For positive moods, recommend activities
        if mood in ["happy", "confident"]:
            return [r for r in self.resources if r["type"] in ["activity", "community"]]

        # Default: mix of resources
        return random.sample(self.resources, min(2, len(self.resources)))

    def _generate_follow_up_questions(self, mood: str, recent_experience: str,
                                    job_search_stage: str) -> List[str]:
        """Generate follow-up questions to continue the conversation.

        Args:
            mood: Current mood
            recent_experience: Recent experience description
            job_search_stage: Current job search stage

        Returns:
            List of follow-up questions
        """
        questions = []

        # Mood-specific questions
        if mood in ["stressed", "anxious"]:
            questions.extend([
                "What's causing the most stress in your job search right now?",
                "How can I help you feel more in control?",
                "What self-care practices have helped you manage stress in the past?"
            ])
        elif mood == "discouraged":
            questions.extend([
                "What accomplishments are you most proud of in your career?",
                "What would success look like for you in this job search?",
                "Who supports you in your career journey?"
            ])
        elif mood in ["happy", "confident"]:
            questions.extend([
                "What's going well in your job search right now?",
                "How can we build on this positive momentum?",
                "What opportunities are you most excited about?"
            ])

        # Job search stage questions
        stage_questions = {
            "starting": ["What kind of roles are you looking for?", "What's your timeline for finding a new position?"],
            "applying": ["How many applications have you sent?", "What response rate are you seeing?"],
            "interviewing": ["How do you prepare for interviews?", "What interview feedback have you received?"],
            "waiting": ["How do you handle the waiting period?", "What's your follow-up strategy?"],
            "negotiating": ["What's your ideal compensation package?", "How do you approach salary negotiations?"],
            "accepted": ["How are you preparing for your new role?", "What are you most excited about?"]
        }

        questions.extend(stage_questions.get(job_search_stage, []))

        # Add general questions
        questions.extend([
            "What support would be most helpful to you right now?",
            "How can I better assist you with your job search?",
            "What's one small step you could take today?"
        ])

        return questions[:3]  # Return top 3 questions

    def _get_mood_improvement_tips(self, mood: str) -> List[str]:
        """Get tips for improving mood.

        Args:
            mood: Current mood

        Returns:
            List of mood improvement tips
        """
        tips_by_mood = {
            "stressed": [
                "Take 5 deep breaths",
                "Go for a 10-minute walk",
                "Listen to calming music",
                "Drink a glass of water",
                "Stretch for 5 minutes"
            ],
            "anxious": [
                "Practice 4-7-8 breathing",
                "Write down your worries",
                "Focus on the present moment",
                "Limit caffeine intake",
                "Talk to a supportive person"
            ],
            "discouraged": [
                "List 3 things you're good at",
                "Review past successes",
                "Help someone else",
                "Try something new",
                "Set a small, achievable goal"
            ],
            "neutral": [
                "Plan something enjoyable",
                "Learn something new",
                "Connect with a friend",
                "Exercise for 15 minutes",
                "Express gratitude"
            ],
            "happy": [
                "Share your positivity",
                "Set ambitious goals",
                "Help others in their search",
                "Celebrate your progress",
                "Plan for continued success"
            ],
            "confident": [
                "Share your knowledge",
                "Mentor someone",
                "Apply for stretch roles",
                "Network proactively",
                "Document your achievements"
            ]
        }

        return tips_by_mood.get(mood, tips_by_mood["neutral"])

    def _calculate_break_time(self, mood: str) -> int:
        """Calculate recommended break time based on mood.

        Args:
            mood: Current mood

        Returns:
            Recommended break time in minutes
        """
        break_times = {
            "stressed": 30,
            "anxious": 20,
            "discouraged": 45,
            "neutral": 15,
            "happy": 10,
            "confident": 5
        }

        return break_times.get(mood, 15)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Emotion Support SKILL resources")
        # Close database connections, cleanup temporary files, etc.
        await asyncio.sleep(0.1)
        self.logger.info("Emotion Support SKILL cleaned up")


# Factory function for easy instantiation
def create_emotion_support_skill() -> EmotionSupportSkill:
    """Create an EmotionSupportSkill instance.

    Returns:
        EmotionSupportSkill instance
    """
    return EmotionSupportSkill()