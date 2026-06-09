"""
Learning Plan SKILL for OpenCareer.

This SKILL provides personalized learning plans and skill development recommendations
for job seekers based on their career goals and current skill level.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base_skill import BaseSkill, SkillMetadata, SkillCategory
from ..skill_registry import SkillRegistry


class LearningPlanSkill(BaseSkill):
    """Learning Plan SKILL for personalized career development."""

    def __init__(self):
        """Initialize Learning Plan SKILL."""
        metadata = SkillMetadata(
            name="learning_plan",
            version="1.0.0",
            description="Create personalized learning plans based on career goals and skills",
            author="OpenCareer Team",
            category=SkillCategory.LEARNING,
            tags=["learning", "career", "planning", "skills"],
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "career_goal": {
                        "type": "string",
                        "description": "Desired career goal or position"
                    },
                    "current_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of current skills"
                    },
                    "timeframe_weeks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 52,
                        "description": "Learning timeframe in weeks"
                    },
                    "learning_style": {
                        "type": "string",
                        "enum": ["visual", "auditory", "reading", "kinesthetic", "mixed"],
                        "description": "Preferred learning style"
                    }
                },
                "required": ["career_goal", "current_skills"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "learning_plan": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "timeline_weeks": {"type": "integer"},
                            "weekly_schedule": {"type": "array", "items": {"type": "object"}},
                            "resources": {"type": "array", "items": {"type": "object"}},
                            "milestones": {"type": "array", "items": {"type": "object"}}
                        }
                    },
                    "skill_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identified skill gaps"
                    },
                    "recommended_courses": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "estimated_completion": {
                        "type": "string",
                        "format": "date"
                    }
                }
            },
            examples=[
                {
                    "input": {
                        "career_goal": "Frontend Developer",
                        "current_skills": ["HTML", "CSS", "JavaScript"],
                        "timeframe_weeks": 12,
                        "learning_style": "visual"
                    },
                    "output": {
                        "learning_plan": {
                            "goal": "Become a Frontend Developer",
                            "timeline_weeks": 12,
                            "weekly_schedule": [
                                {"week": 1, "topics": ["Advanced JavaScript", "ES6+ Features"]},
                                {"week": 2, "topics": ["React Fundamentals", "JSX"]}
                            ],
                            "skill_gaps": ["React", "TypeScript", "State Management"],
                            "estimated_completion": "2024-01-15"
                        }
                    }
                }
            ]
        )
        super().__init__(metadata)
        self.learning_resources = self._load_learning_resources()

    def _load_learning_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load learning resources for different skill areas.

        Returns:
            Dictionary of learning resources by skill category
        """
        # This would typically come from a database or external API
        return {
            "programming": [
                {
                    "title": "Python for Everybody",
                    "platform": "Coursera",
                    "type": "course",
                    "duration_hours": 20,
                    "url": "https://www.coursera.org/specializations/python"
                },
                {
                    "title": "JavaScript: The Complete Guide",
                    "platform": "Udemy",
                    "type": "course",
                    "duration_hours": 40,
                    "url": "https://www.udemy.com/course/javascript-the-complete-guide-2020-beginner-advanced/"
                }
            ],
            "frontend": [
                {
                    "title": "React - The Complete Guide",
                    "platform": "Udemy",
                    "type": "course",
                    "duration_hours": 48,
                    "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/"
                },
                {
                    "title": "Vue.js Fundamentals",
                    "platform": "Pluralsight",
                    "type": "course",
                    "duration_hours": 15,
                    "url": "https://www.pluralsight.com/courses/vuejs-fundamentals"
                }
            ],
            "soft_skills": [
                {
                    "title": "Communication Skills for Career Success",
                    "platform": "LinkedIn Learning",
                    "type": "course",
                    "duration_hours": 8,
                    "url": "https://www.linkedin.com/learning/communication-skills-for-career-success"
                }
            ]
        }

    async def _initialize(self) -> None:
        """Initialize the SKILL."""
        self.logger.info("Initializing Learning Plan SKILL")
        # Load additional resources, connect to databases, etc.
        # For demo purposes, we'll just simulate loading
        await asyncio.sleep(0.1)
        self.logger.info("Learning Plan SKILL initialized")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the learning plan creation.

        Args:
            input_data: Input data containing career goal and skills
            context: Additional context (user info, session, etc.)

        Returns:
            Personalized learning plan
        """
        # Validate input
        await self.validate_input(input_data)

        try:
            self.logger.info(f"Creating learning plan for goal: {input_data.get('career_goal')}")

            # Extract input parameters
            career_goal = input_data.get("career_goal", "")
            current_skills = input_data.get("current_skills", [])
            timeframe_weeks = input_data.get("timeframe_weeks", 12)
            learning_style = input_data.get("learning_style", "mixed")
            user_id = input_data.get("user_id", "anonymous")

            # Analyze skill gaps (simplified logic)
            skill_gaps = self._analyze_skill_gaps(career_goal, current_skills)

            # Create learning plan
            learning_plan = self._create_learning_plan(
                career_goal=career_goal,
                skill_gaps=skill_gaps,
                timeframe_weeks=timeframe_weeks,
                learning_style=learning_style
            )

            # Get recommended resources
            recommended_resources = self._recommend_resources(skill_gaps, learning_style)

            # Calculate estimated completion date
            estimated_completion = datetime.now() + timedelta(weeks=timeframe_weeks)

            # Build response
            result = {
                "learning_plan": learning_plan,
                "skill_gaps": skill_gaps,
                "recommended_courses": recommended_resources,
                "estimated_completion": estimated_completion.strftime("%Y-%m-%d"),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Learning plan created successfully for user: {user_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error creating learning plan: {e}")
            raise RuntimeError(f"Failed to create learning plan: {str(e)}")

    def _analyze_skill_gaps(self, career_goal: str, current_skills: List[str]) -> List[str]:
        """Analyze skill gaps based on career goal.

        Args:
            career_goal: Desired career position
            current_skills: List of current skills

        Returns:
            List of missing skills
        """
        # Define required skills for different career paths
        career_skill_requirements = {
            "frontend developer": ["HTML", "CSS", "JavaScript", "React", "Git", "Responsive Design"],
            "backend developer": ["Python", "Django/Flask", "SQL", "REST APIs", "Docker", "Git"],
            "data scientist": ["Python", "Pandas", "NumPy", "Machine Learning", "SQL", "Statistics"],
            "devops engineer": ["Docker", "Kubernetes", "AWS/Azure", "CI/CD", "Linux", "Networking"],
            "product manager": ["Product Strategy", "User Research", "Agile", "Analytics", "Communication"]
        }

        # Normalize career goal
        goal_lower = career_goal.lower()

        # Find matching career path
        required_skills = []
        for career_path, skills in career_skill_requirements.items():
            if career_path in goal_lower:
                required_skills = skills
                break

        # If no exact match, use default
        if not required_skills:
            required_skills = ["Communication", "Problem Solving", "Technical Skills", "Industry Knowledge"]

        # Normalize current skills (lowercase for comparison)
        current_skills_lower = [skill.lower() for skill in current_skills]

        # Find gaps
        skill_gaps = []
        for required_skill in required_skills:
            required_lower = required_skill.lower()
            # Check if any current skill contains required skill or vice versa
            found = False
            for current_skill in current_skills_lower:
                if required_lower in current_skill or current_skill in required_lower:
                    found = True
                    break
            if not found:
                skill_gaps.append(required_skill)

        return skill_gaps

    def _create_learning_plan(self, career_goal: str, skill_gaps: List[str],
                             timeframe_weeks: int, learning_style: str) -> Dict[str, Any]:
        """Create a structured learning plan.

        Args:
            career_goal: Career goal
            skill_gaps: Identified skill gaps
            timeframe_weeks: Learning timeframe in weeks
            learning_style: Preferred learning style

        Returns:
            Structured learning plan
        """
        # Create weekly schedule
        weekly_schedule = []
        weeks_per_skill = max(1, timeframe_weeks // max(1, len(skill_gaps)))

        current_week = 1
        for i, skill in enumerate(skill_gaps):
            if current_week > timeframe_weeks:
                break

            # Determine weeks for this skill
            skill_weeks = min(weeks_per_skill, timeframe_weeks - current_week + 1)

            weekly_schedule.append({
                "week": current_week,
                "skill": skill,
                "duration_weeks": skill_weeks,
                "topics": self._generate_topics_for_skill(skill, skill_weeks, learning_style),
                "learning_objectives": [
                    f"Understand core concepts of {skill}",
                    f"Apply {skill} in practical scenarios",
                    f"Build a project using {skill}"
                ]
            })

            current_week += skill_weeks

        # Create milestones
        milestones = []
        quarter_point = timeframe_weeks // 4
        for i in range(1, 5):
            week = i * quarter_point
            if week <= timeframe_weeks:
                milestones.append({
                    "week": week,
                    "milestone": f"Complete {i * 25}% of learning plan",
                    "assessment": f"Review progress and adjust plan if needed"
                })

        return {
            "goal": f"Become a {career_goal}",
            "timeline_weeks": timeframe_weeks,
            "weekly_schedule": weekly_schedule,
            "milestones": milestones,
            "learning_style": learning_style,
            "created_date": datetime.now().strftime("%Y-%m-%d")
        }

    def _generate_topics_for_skill(self, skill: str, weeks: int, learning_style: str) -> List[str]:
        """Generate learning topics for a skill.

        Args:
            skill: Skill to learn
            weeks: Number of weeks allocated
            learning_style: Preferred learning style

        Returns:
            List of learning topics
        """
        # Define topics for different skills
        skill_topics = {
            "python": ["Syntax & Basics", "Data Structures", "Functions", "OOP", "Libraries"],
            "javascript": ["ES6+", "DOM Manipulation", "Async Programming", "Frameworks"],
            "react": ["Components", "State & Props", "Hooks", "Routing", "State Management"],
            "sql": ["Queries", "Joins", "Indexing", "Normalization", "Optimization"],
            "docker": ["Containers", "Images", "Dockerfile", "Compose", "Orchestration"]
        }

        # Find matching topics
        skill_lower = skill.lower()
        topics = []
        for skill_key, skill_topic_list in skill_topics.items():
            if skill_key in skill_lower:
                topics = skill_topic_list[:weeks]  # Limit topics based on weeks
                break

        # If no match, create generic topics
        if not topics:
            topics = [f"{skill} Fundamentals"]
            if weeks > 1:
                topics.append(f"Advanced {skill}")
            if weeks > 2:
                topics.append(f"{skill} in Practice")

        # Adapt topics based on learning style
        if learning_style == "visual":
            topics = [f"Visual guide to {topic}" for topic in topics]
        elif learning_style == "auditory":
            topics = [f"Audio lessons on {topic}" for topic in topics]

        return topics[:weeks]  # Ensure we don't have more topics than weeks

    def _recommend_resources(self, skill_gaps: List[str], learning_style: str) -> List[Dict[str, Any]]:
        """Recommend learning resources.

        Args:
            skill_gaps: Skill gaps to address
            learning_style: Preferred learning style

        Returns:
            List of recommended resources
        """
        recommended = []

        for skill in skill_gaps:
            skill_lower = skill.lower()

            # Find matching resources
            for category, resources in self.learning_resources.items():
                if any(category in skill_lower or skill_lower in category for word in skill_lower.split()):
                    # Filter by learning style if possible
                    style_resources = [
                        resource for resource in resources
                        if learning_style in resource.get("tags", []) or learning_style == "mixed"
                    ]
                    if style_resources:
                        recommended.extend(style_resources[:2])  # Limit to 2 per skill
                    else:
                        recommended.extend(resources[:2])

        # Remove duplicates
        unique_resources = []
        seen_urls = set()
        for resource in recommended:
            url = resource.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_resources.append(resource)

        return unique_resources[:10]  # Limit to 10 resources total

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Learning Plan SKILL resources")
        # Close database connections, cleanup temporary files, etc.
        await asyncio.sleep(0.1)
        self.logger.info("Learning Plan SKILL cleaned up")


# Factory function for easy instantiation
def create_learning_plan_skill() -> LearningPlanSkill:
    """Create a LearningPlanSkill instance.

    Returns:
        LearningPlanSkill instance
    """
    return LearningPlanSkill()