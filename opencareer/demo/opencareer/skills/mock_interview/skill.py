"""
Mock Interview SKILL for OpenCareer.

This SKILL provides mock interview practice with AI-powered feedback and evaluation
to help job seekers improve their interview skills.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base_skill import BaseSkill, SkillMetadata, SkillCategory


class MockInterviewSkill(BaseSkill):
    """Mock Interview SKILL for interview practice and feedback."""

    def __init__(self):
        """Initialize Mock Interview SKILL."""
        metadata = SkillMetadata(
            name="mock_interview",
            version="1.0.0",
            description="Practice mock interviews with AI feedback and evaluation",
            author="OpenCareer Team",
            category=SkillCategory.INTERVIEW,
            tags=["interview", "practice", "feedback", "career", "assessment"],
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "position": {
                        "type": "string",
                        "description": "Target job position"
                    },
                    "experience_level": {
                        "type": "string",
                        "enum": ["entry", "mid", "senior", "executive"],
                        "description": "Experience level for the position"
                    },
                    "interview_type": {
                        "type": "string",
                        "enum": ["technical", "behavioral", "mixed", "system_design"],
                        "description": "Type of interview"
                    },
                    "question_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of questions to ask"
                    },
                    "time_limit_minutes": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 120,
                        "description": "Total time limit in minutes"
                    }
                },
                "required": ["position", "experience_level"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "interview_session": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "questions": {"type": "array", "items": {"type": "object"}},
                            "start_time": {"type": "string", "format": "date-time"},
                            "estimated_duration_minutes": {"type": "integer"}
                        }
                    },
                    "feedback": {
                        "type": "object",
                        "properties": {
                            "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
                            "strengths": {"type": "array", "items": {"type": "string"}},
                            "improvement_areas": {"type": "array", "items": {"type": "string"}},
                            "detailed_feedback": {"type": "array", "items": {"type": "object"}}
                        }
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Personalized recommendations for improvement"
                    }
                }
            },
            examples=[
                {
                    "input": {
                        "position": "Software Engineer",
                        "experience_level": "mid",
                        "interview_type": "technical",
                        "question_count": 10
                    },
                    "output": {
                        "interview_session": {
                            "session_id": "intv_12345",
                            "questions": [
                                {"id": 1, "question": "Explain time complexity of binary search"},
                                {"id": 2, "question": "What is the difference between REST and GraphQL?"}
                            ],
                            "estimated_duration_minutes": 30
                        }
                    }
                }
            ]
        )
        super().__init__(metadata)
        self.interview_questions = self._load_questions()
        self.evaluation_criteria = self._load_evaluation_criteria()

    def _load_questions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load interview questions database.

        Returns:
            Dictionary of questions by category
        """
        return {
            "technical": [
                {
                    "id": "tech_001",
                    "question": "Explain the difference between lists and tuples in Python.",
                    "category": "programming",
                    "difficulty": "easy",
                    "expected_keywords": ["mutable", "immutable", "performance", "use cases"]
                },
                {
                    "id": "tech_002",
                    "question": "What is the time complexity of binary search?",
                    "category": "algorithms",
                    "difficulty": "easy",
                    "expected_keywords": ["O(log n)", "sorted array", "divide and conquer"]
                },
                {
                    "id": "tech_003",
                    "question": "Explain how React's virtual DOM works.",
                    "category": "frontend",
                    "difficulty": "medium",
                    "expected_keywords": ["diffing algorithm", "reconciliation", "performance"]
                },
                {
                    "id": "tech_004",
                    "question": "What are ACID properties in database transactions?",
                    "category": "database",
                    "difficulty": "medium",
                    "expected_keywords": ["atomicity", "consistency", "isolation", "durability"]
                },
                {
                    "id": "tech_005",
                    "question": "Describe the CAP theorem and its implications.",
                    "category": "system_design",
                    "difficulty": "hard",
                    "expected_keywords": ["consistency", "availability", "partition tolerance", "trade-offs"]
                }
            ],
            "behavioral": [
                {
                    "id": "behav_001",
                    "question": "Tell me about a time you faced a difficult challenge at work and how you handled it.",
                    "category": "problem_solving",
                    "difficulty": "medium",
                    "expected_keywords": ["STAR method", "specific example", "outcome", "learning"]
                },
                {
                    "id": "behav_002",
                    "question": "Describe a situation where you had to work with a difficult teammate.",
                    "category": "teamwork",
                    "difficulty": "medium",
                    "expected_keywords": ["conflict resolution", "communication", "collaboration"]
                },
                {
                    "id": "behav_003",
                    "question": "How do you prioritize tasks when working on multiple projects?",
                    "category": "time_management",
                    "difficulty": "easy",
                    "expected_keywords": ["prioritization", "deadlines", "importance vs urgency"]
                }
            ],
            "system_design": [
                {
                    "id": "sys_001",
                    "question": "Design a URL shortening service like bit.ly.",
                    "category": "system_design",
                    "difficulty": "medium",
                    "expected_keywords": ["hash function", "database", "scalability", "cache"]
                },
                {
                    "id": "sys_002",
                    "question": "Design a distributed key-value store.",
                    "category": "system_design",
                    "difficulty": "hard",
                    "expected_keywords": ["consistency", "replication", "partitioning", "failure handling"]
                }
            ]
        }

    def _load_evaluation_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Load evaluation criteria for different aspects.

        Returns:
            Dictionary of evaluation criteria
        """
        return {
            "technical_knowledge": {
                "weight": 0.4,
                "aspects": ["accuracy", "depth", "relevance", "examples"],
                "levels": {
                    "excellent": "Comprehensive and accurate with practical examples",
                    "good": "Mostly accurate with some depth",
                    "fair": "Basic understanding with some gaps",
                    "poor": "Significant gaps or inaccuracies"
                }
            },
            "communication": {
                "weight": 0.3,
                "aspects": ["clarity", "structure", "conciseness", "confidence"],
                "levels": {
                    "excellent": "Clear, structured, and confident communication",
                    "good": "Generally clear with minor issues",
                    "fair": "Somewhat unclear or disorganized",
                    "poor": "Unclear, disorganized, or lacks confidence"
                }
            },
            "problem_solving": {
                "weight": 0.2,
                "aspects": ["approach", "creativity", "efficiency", "adaptability"],
                "levels": {
                    "excellent": "Systematic approach with creative solutions",
                    "good": "Logical approach with room for improvement",
                    "fair": "Basic approach with limited creativity",
                    "poor": "Ineffective or no clear approach"
                }
            },
            "cultural_fit": {
                "weight": 0.1,
                "aspects": ["enthusiasm", "teamwork", "values", "growth_mindset"],
                "levels": {
                    "excellent": "Strong alignment with company values and culture",
                    "good": "Good fit with minor reservations",
                    "fair": "Some misalignment",
                    "poor": "Significant cultural mismatch"
                }
            }
        }

    async def _initialize(self) -> None:
        """Initialize the SKILL."""
        self.logger.info("Initializing Mock Interview SKILL")
        # Load additional resources, connect to databases, etc.
        await asyncio.sleep(0.1)
        self.logger.info("Mock Interview SKILL initialized")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute mock interview session.

        Args:
            input_data: Input data for interview configuration
            context: Additional context (user info, previous interviews, etc.)

        Returns:
            Interview session details and feedback
        """
        # Validate input
        await self.validate_input(input_data)

        try:
            self.logger.info(f"Starting mock interview for position: {input_data.get('position')}")

            # Extract parameters
            position = input_data.get("position", "")
            experience_level = input_data.get("experience_level", "mid")
            interview_type = input_data.get("interview_type", "mixed")
            question_count = input_data.get("question_count", 10)
            time_limit_minutes = input_data.get("time_limit_minutes", 60)
            user_id = input_data.get("user_id", "anonymous")

            # Generate interview session
            session_id = f"intv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
            questions = self._generate_questions(position, experience_level, interview_type, question_count)

            # Calculate estimated duration
            estimated_duration = self._estimate_duration(questions, experience_level)

            # Generate initial feedback template
            feedback = self._generate_initial_feedback_template(position, experience_level)

            # Build response
            result = {
                "interview_session": {
                    "session_id": session_id,
                    "position": position,
                    "experience_level": experience_level,
                    "interview_type": interview_type,
                    "questions": questions,
                    "start_time": datetime.now().isoformat(),
                    "estimated_duration_minutes": min(estimated_duration, time_limit_minutes),
                    "time_limit_minutes": time_limit_minutes,
                    "user_id": user_id
                },
                "feedback": feedback,
                "instructions": {
                    "how_to_proceed": "Answer each question verbally or in writing. Take your time to think before answering.",
                    "scoring_criteria": "Responses will be evaluated based on accuracy, clarity, and completeness.",
                    "time_management": f"Try to complete within {time_limit_minutes} minutes."
                }
            }

            self.logger.info(f"Mock interview session created: {session_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error creating mock interview: {e}")
            raise RuntimeError(f"Failed to create mock interview: {str(e)}")

    async def evaluate_response(self, session_id: str, question_id: str,
                              user_response: str, response_time_seconds: int = None) -> Dict[str, Any]:
        """Evaluate a user's response to an interview question.

        Args:
            session_id: Interview session ID
            question_id: Question ID
            user_response: User's response text
            response_time_seconds: Time taken to respond

        Returns:
            Evaluation results
        """
        try:
            self.logger.info(f"Evaluating response for question {question_id} in session {session_id}")

            # Find the question
            question = None
            for category_questions in self.interview_questions.values():
                for q in category_questions:
                    if q["id"] == question_id:
                        question = q
                        break
                if question:
                    break

            if not question:
                raise ValueError(f"Question not found: {question_id}")

            # Evaluate the response
            evaluation = self._evaluate_single_response(question, user_response, response_time_seconds)

            return {
                "session_id": session_id,
                "question_id": question_id,
                "question": question["question"],
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat(),
                "suggested_improvements": self._suggest_improvements(evaluation, question)
            }

        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            raise RuntimeError(f"Failed to evaluate response: {str(e)}")

    def _generate_questions(self, position: str, experience_level: str,
                          interview_type: str, question_count: int) -> List[Dict[str, Any]]:
        """Generate interview questions.

        Args:
            position: Target position
            experience_level: Experience level
            interview_type: Type of interview
            question_count: Number of questions

        Returns:
            List of interview questions
        """
        # Determine question categories based on interview type
        if interview_type == "technical":
            categories = ["technical"]
        elif interview_type == "behavioral":
            categories = ["behavioral"]
        elif interview_type == "system_design":
            categories = ["system_design"]
        else:  # mixed
            categories = ["technical", "behavioral"]

        # Select questions
        selected_questions = []
        questions_per_category = max(1, question_count // len(categories))

        for category in categories:
            if category in self.interview_questions:
                category_questions = self.interview_questions[category]
                # Filter by difficulty based on experience level
                difficulty_map = {
                    "entry": ["easy"],
                    "mid": ["easy", "medium"],
                    "senior": ["medium", "hard"],
                    "executive": ["medium", "hard"]
                }
                allowed_difficulties = difficulty_map.get(experience_level, ["easy", "medium"])

                filtered_questions = [
                    q for q in category_questions
                    if q["difficulty"] in allowed_difficulties
                ]

                # Randomly select questions
                if filtered_questions:
                    selected = random.sample(
                        filtered_questions,
                        min(questions_per_category, len(filtered_questions))
                    )
                    selected_questions.extend(selected)

        # Ensure we have the requested number of questions
        if len(selected_questions) < question_count:
            # Add more questions from any category
            all_questions = []
            for category_questions in self.interview_questions.values():
                all_questions.extend(category_questions)

            additional_needed = question_count - len(selected_questions)
            available_questions = [q for q in all_questions if q not in selected_questions]
            if available_questions:
                additional = random.sample(
                    available_questions,
                    min(additional_needed, len(available_questions))
                )
                selected_questions.extend(additional)

        # Format questions for response
        formatted_questions = []
        for i, question in enumerate(selected_questions[:question_count], 1):
            formatted_questions.append({
                "id": question["id"],
                "order": i,
                "question": question["question"],
                "category": question["category"],
                "difficulty": question["difficulty"],
                "expected_keywords": question.get("expected_keywords", []),
                "suggested_time_minutes": self._get_suggested_time(question["difficulty"])
            })

        return formatted_questions

    def _get_suggested_time(self, difficulty: str) -> int:
        """Get suggested time for answering a question.

        Args:
            difficulty: Question difficulty

        Returns:
            Suggested time in minutes
        """
        time_map = {
            "easy": 2,
            "medium": 5,
            "hard": 10
        }
        return time_map.get(difficulty, 3)

    def _estimate_duration(self, questions: List[Dict[str, Any]], experience_level: str) -> int:
        """Estimate total interview duration.

        Args:
            questions: List of questions
            experience_level: Experience level

        Returns:
            Estimated duration in minutes
        """
        base_duration = 0
        for question in questions:
            base_duration += question.get("suggested_time_minutes", 3)

        # Adjust based on experience level
        experience_multiplier = {
            "entry": 0.8,  # Less experience, quicker answers
            "mid": 1.0,
            "senior": 1.2,  # More senior, more detailed answers
            "executive": 1.5
        }

        multiplier = experience_multiplier.get(experience_level, 1.0)
        return int(base_duration * multiplier)

    def _generate_initial_feedback_template(self, position: str, experience_level: str) -> Dict[str, Any]:
        """Generate initial feedback template.

        Args:
            position: Target position
            experience_level: Experience level

        Returns:
            Feedback template
        """
        return {
            "overall_score": None,  # Will be calculated after evaluation
            "strengths": [],
            "improvement_areas": [],
            "detailed_feedback": [],
            "criteria": self.evaluation_criteria,
            "position": position,
            "experience_level": experience_level
        }

    def _evaluate_single_response(self, question: Dict[str, Any], user_response: str,
                                 response_time_seconds: Optional[int]) -> Dict[str, Any]:
        """Evaluate a single response.

        Args:
            question: Question details
            user_response: User's response
            response_time_seconds: Time taken to respond

        Returns:
            Evaluation results
        """
        # Simplified evaluation logic
        # In a real system, this would use NLP or AI for evaluation

        response_lower = user_response.lower()
        expected_keywords = question.get("expected_keywords", [])

        # Check for keywords
        found_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)

        # Calculate keyword coverage
        keyword_coverage = len(found_keywords) / max(1, len(expected_keywords))

        # Evaluate response length
        word_count = len(user_response.split())
        adequate_length = 20 <= word_count <= 500  # Reasonable response length

        # Evaluate structure (simplified)
        has_structure = any(marker in user_response for marker in [". ", "\n", ";", "Firstly", "Secondly"])

        # Calculate score
        score = 0.0
        score += keyword_coverage * 50  # 50% for content
        score += (1 if adequate_length else 0.5) * 30  # 30% for length
        score += (1 if has_structure else 0.5) * 20  # 20% for structure

        # Adjust for response time if provided
        if response_time_seconds:
            expected_time = self._get_suggested_time(question["difficulty"]) * 60  # Convert to seconds
            time_ratio = expected_time / max(1, response_time_seconds)
            if 0.5 <= time_ratio <= 2.0:  # Within reasonable range
                time_adjustment = 0
            elif time_ratio < 0.5:  # Too slow
                time_adjustment = -10
            else:  # Too fast
                time_adjustment = -5
            score = max(0, score + time_adjustment)

        # Determine level
        if score >= 80:
            level = "excellent"
        elif score >= 60:
            level = "good"
        elif score >= 40:
            level = "fair"
        else:
            level = "poor"

        return {
            "score": round(score, 1),
            "level": level,
            "keyword_coverage": f"{int(keyword_coverage * 100)}%",
            "found_keywords": found_keywords,
            "missing_keywords": [k for k in expected_keywords if k not in found_keywords],
            "word_count": word_count,
            "adequate_length": adequate_length,
            "has_structure": has_structure,
            "response_time_seconds": response_time_seconds
        }

    def _suggest_improvements(self, evaluation: Dict[str, Any], question: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on evaluation.

        Args:
            evaluation: Evaluation results
            question: Question details

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        score = evaluation.get("score", 0)
        level = evaluation.get("level", "fair")

        if score < 60:
            suggestions.append("Provide more specific examples in your answers.")
            suggestions.append("Structure your response with clear points.")

        if evaluation.get("keyword_coverage", "0%") < "50%":
            suggestions.append(f"Address key concepts like: {', '.join(evaluation.get('missing_keywords', []))}")

        if not evaluation.get("adequate_length", False):
            if evaluation.get("word_count", 0) < 20:
                suggestions.append("Provide more detailed explanations.")
            else:
                suggestions.append("Try to be more concise while maintaining clarity.")

        if not evaluation.get("has_structure", False):
            suggestions.append("Use bullet points or numbered lists to organize your thoughts.")

        # Add question-specific suggestions
        if question["difficulty"] == "hard":
            suggestions.append("For complex questions, break them down into smaller parts.")
        elif question["category"] == "behavioral":
            suggestions.append("Use the STAR method (Situation, Task, Action, Result) for behavioral questions.")

        return suggestions[:5]  # Limit to 5 suggestions

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Mock Interview SKILL resources")
        await asyncio.sleep(0.1)
        self.logger.info("Mock Interview SKILL cleaned up")


# Factory function for easy instantiation
def create_mock_interview_skill() -> MockInterviewSkill:
    """Create a MockInterviewSkill instance.

    Returns:
        MockInterviewSkill instance
    """
    return MockInterviewSkill()