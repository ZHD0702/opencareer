"""
Log Agent for OpenCareer system.

This agent extracts and structures information from conversations,
then stores it in the memory system for future reference.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base_agent import BaseAgent, AgentMessage


class LogAgent(BaseAgent):
    """Log Agent responsible for information extraction and structured storage."""

    def __init__(self, memory_manager = None):
        """Initialize Log Agent.

        Args:
            memory_manager: Memory manager for storing extracted information
        """
        super().__init__(
            name="log_agent",
            description="Extracts key information from conversations and stores it in structured format"
        )

        self.memory_manager = memory_manager

        # Information extraction patterns
        self.extraction_patterns = {
            "skill_mentions": ["skill", "learn", "study", "practice", "expertise", "knowledge"],
            "career_goals": ["goal", "aspire", "want to become", "dream job", "career path"],
            "job_interests": ["job", "position", "role", "work as", "apply for", "hiring"],
            "education_background": ["education", "degree", "university", "college", "major"],
            "work_experience": ["experience", "worked", "job at", "position at", "years in"],
            "emotional_state": ["stress", "anxious", "happy", "frustrated", "confident", "nervous"]
        }

        # Register capabilities
        self.capabilities = [
            "information_extraction",
            "structured_storage",
            "conversation_analysis",
            "user_profile_update"
        ]

        self.logger = logging.getLogger("agent.log_agent")

    async def _initialize(self) -> None:
        """Initialize Log Agent."""
        self.logger.info("Initializing Log Agent")
        # In a real implementation, this might load ML models or configuration
        await asyncio.sleep(0.1)
        self.logger.info("Log Agent initialized")

    async def process_user_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user request by extracting and storing information.

        Args:
            user_input: User's input text
            context: Additional context (user info, session, etc.)

        Returns:
            Information extraction result
        """
        self.logger.info(f"Processing log request: {user_input[:100]}...")

        # Step 1: Extract information from user input
        extracted_info = await self._extract_information(user_input, context)

        # Step 2: Store extracted information
        storage_result = await self._store_extracted_info(extracted_info, context)

        # Step 3: Generate response
        response = await self._generate_response(extracted_info, storage_result, context)

        return response

    async def _extract_information(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract structured information from user input.

        Args:
            user_input: User's input text
            context: Additional context

        Returns:
            Extracted information
        """
        self.logger.info("Extracting information from user input")

        extracted = {
            "raw_input": user_input,
            "timestamp": datetime.now().isoformat(),
            "extracted_categories": {},
            "entities": [],
            "sentiment": "neutral",
            "extraction_confidence": 0.0
        }

        # Add context if available
        if context:
            extracted["context"] = {k: v for k, v in context.items()
                                  if k not in ["raw_input", "extracted_info"]}

        # Simple rule-based extraction for demo
        # In a real implementation, this would use NLP models
        input_lower = user_input.lower()

        # Check each extraction pattern
        for category, triggers in self.extraction_patterns.items():
            matches = []
            for trigger in triggers:
                if trigger in input_lower:
                    matches.append(trigger)

            if matches:
                extracted["extracted_categories"][category] = {
                    "triggers_found": matches,
                    "confidence": min(1.0, len(matches) * 0.3),  # Simple confidence calculation
                    "snippets": self._extract_relevant_snippets(user_input, matches)
                }

        # Extract potential entities (simple version)
        entities = self._extract_entities(user_input)
        if entities:
            extracted["entities"] = entities

        # Simple sentiment analysis
        sentiment = self._analyze_sentiment(user_input)
        extracted["sentiment"] = sentiment

        # Calculate overall confidence
        if extracted["extracted_categories"]:
            total_confidence = sum(info["confidence"]
                                 for info in extracted["extracted_categories"].values())
            extracted["extraction_confidence"] = total_confidence / len(extracted["extracted_categories"])
        else:
            extracted["extraction_confidence"] = 0.0

        self.logger.info(f"Extracted {len(extracted['extracted_categories'])} categories")
        return extracted

    def _extract_relevant_snippets(self, text: str, triggers: List[str]) -> List[str]:
        """Extract text snippets around trigger words.

        Args:
            text: Original text
            triggers: List of trigger words found

        Returns:
            List of relevant snippets
        """
        snippets = []
        words = text.split()

        for trigger in triggers:
            # Find trigger in words (case insensitive)
            for i, word in enumerate(words):
                if trigger in word.lower():
                    # Extract context around trigger
                    start = max(0, i - 5)
                    end = min(len(words), i + 6)
                    snippet = " ".join(words[start:end])
                    snippets.append(snippet)
                    break  # Only need first occurrence per trigger

        return snippets[:3]  # Limit to 3 snippets

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (simplified for demo).

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = []
        words = text.split()

        # Simple patterns for demo
        # In a real system, use NER models
        tech_keywords = ["python", "javascript", "react", "java", "sql", "docker", "aws"]
        job_titles = ["developer", "engineer", "manager", "analyst", "designer", "architect"]
        education_terms = ["university", "college", "bachelor", "master", "phd", "degree"]

        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')

            # Check for technology keywords
            if word_lower in tech_keywords:
                entities.append({
                    "text": word,
                    "type": "technology",
                    "category": "skill",
                    "confidence": 0.8
                })

            # Check for job titles
            elif word_lower in job_titles:
                entities.append({
                    "text": word,
                    "type": "job_title",
                    "category": "career",
                    "confidence": 0.7
                })

            # Check for education terms
            elif word_lower in education_terms:
                entities.append({
                    "text": word,
                    "type": "education",
                    "category": "background",
                    "confidence": 0.6
                })

        return entities

    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis.

        Args:
            text: Input text

        Returns:
            Sentiment category
        """
        positive_words = ["good", "great", "excellent", "happy", "excited", "confident", "proud"]
        negative_words = ["bad", "terrible", "stress", "anxious", "nervous", "worried", "frustrated"]

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    async def _store_extracted_info(self, extracted_info: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store extracted information in memory system.

        Args:
            extracted_info: Extracted information
            context: Additional context

        Returns:
            Storage result
        """
        self.logger.info("Storing extracted information")

        storage_result = {
            "stored_items": 0,
            "storage_errors": [],
            "memory_ids": [],
            "timestamp": datetime.now().isoformat()
        }

        if not self.memory_manager:
            self.logger.warning("No memory manager available, skipping storage")
            storage_result["storage_errors"].append("memory_manager_not_available")
            return storage_result

        # Get user ID from context or use default
        user_id = "anonymous"
        if context and "user_id" in context:
            user_id = context["user_id"]

        try:
            # Store conversation
            conversation_data = {
                "user_input": extracted_info.get("raw_input", ""),
                "extracted_info": extracted_info,
                "timestamp": extracted_info.get("timestamp"),
                "sentiment": extracted_info.get("sentiment")
            }

            conversation_id = await self.memory_manager.store_conversation(
                user_id=user_id,
                conversation=conversation_data
            )

            storage_result["memory_ids"].append({
                "type": "conversation",
                "id": conversation_id
            })
            storage_result["stored_items"] += 1

            # Store extracted entities as skills if applicable
            entities = extracted_info.get("entities", [])
            for entity in entities:
                if entity.get("category") == "skill":
                    skill_data = {
                        "name": entity["text"],
                        "type": entity["type"],
                        "confidence": entity["confidence"],
                        "source": "extracted_from_conversation",
                        "extraction_timestamp": datetime.now().isoformat()
                    }

                    try:
                        skill_id = await self.memory_manager.store_skill(
                            user_id=user_id,
                            skill_data=skill_data
                        )
                        storage_result["memory_ids"].append({
                            "type": "skill",
                            "id": skill_id,
                            "skill_name": entity["text"]
                        })
                        storage_result["stored_items"] += 1
                    except Exception as e:
                        storage_result["storage_errors"].append(f"skill_storage_error: {str(e)}")

            # Update user profile if career goals are mentioned
            if "career_goals" in extracted_info.get("extracted_categories", {}):
                profile_update = {
                    "last_career_goal_mention": datetime.now().isoformat(),
                    "extracted_goals": extracted_info["extracted_categories"]["career_goals"]["snippets"]
                }

                # Get existing profile
                existing_profile = await self.memory_manager.get_user_profile(user_id)
                if existing_profile:
                    # Merge with existing profile
                    updated_profile = {**existing_profile, **profile_update}
                else:
                    # Create new profile
                    updated_profile = {
                        "user_id": user_id,
                        "created_at": datetime.now().isoformat(),
                        **profile_update
                    }

                try:
                    profile_id = await self.memory_manager.store_user_profile(
                        user_id=user_id,
                        profile=updated_profile
                    )
                    storage_result["memory_ids"].append({
                        "type": "user_profile",
                        "id": profile_id
                    })
                    storage_result["stored_items"] += 1
                except Exception as e:
                    storage_result["storage_errors"].append(f"profile_storage_error: {str(e)}")

            self.logger.info(f"Stored {storage_result['stored_items']} items successfully")

        except Exception as e:
            self.logger.error(f"Error storing extracted info: {e}")
            storage_result["storage_errors"].append(f"general_storage_error: {str(e)}")

        return storage_result

    async def _generate_response(self, extracted_info: Dict[str, Any],
                                storage_result: Dict[str, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response based on extraction and storage results.

        Args:
            extracted_info: Extracted information
            storage_result: Storage operation result
            context: Additional context

        Returns:
            Response to send back
        """
        self.logger.info("Generating log agent response")

        # Determine response type based on what was extracted
        categories_found = list(extracted_info.get("extracted_categories", {}).keys())

        if not categories_found:
            summary = "I've noted this conversation for future reference."
        else:
            category_descriptions = {
                "skill_mentions": "skills and learning topics",
                "career_goals": "career goals and aspirations",
                "job_interests": "job interests and opportunities",
                "education_background": "educational background",
                "work_experience": "work experience",
                "emotional_state": "emotional state and feelings"
            }

            readable_categories = [category_descriptions.get(cat, cat)
                                 for cat in categories_found[:3]]  # Limit to 3

            if len(readable_categories) == 1:
                summary = f"I've noted your {readable_categories[0]} for future reference."
            else:
                summary = f"I've noted information about {', '.join(readable_categories[:-1])} and {readable_categories[-1]}."

        # Build response
        response = {
            "agent": self.name,
            "response_type": "log_acknowledgment",
            "summary": summary,
            "extraction_summary": {
                "categories_found": len(categories_found),
                "entities_extracted": len(extracted_info.get("entities", [])),
                "sentiment": extracted_info.get("sentiment", "neutral"),
                "confidence": extracted_info.get("extraction_confidence", 0.0)
            },
            "storage_summary": {
                "items_stored": storage_result.get("stored_items", 0),
                "storage_errors": len(storage_result.get("storage_errors", [])),
                "successful": storage_result.get("stored_items", 0) > 0
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processing_time_ms": 0,  # Would be calculated in real implementation
                "memory_available": self.memory_manager is not None
            }
        }

        # Add user ID if available
        if context and "user_id" in context:
            response["user_id"] = context["user_id"]

        return response

    async def handle_text_message(self, message: AgentMessage) -> None:
        """Handle incoming text messages.

        Args:
            message: The message to handle
        """
        self.logger.info(f"Log Agent handling text message from {message.sender}")

        # Extract user input and context
        content = message.content
        user_input = content.get("text", "")
        context = content.get("context", {})

        # Process the request (but don't send response - log agent works in background)
        try:
            result = await self.process_user_request(user_input, context)
            self.logger.info(f"Log Agent processed message, extracted {result['extraction_summary']['categories_found']} categories")

            # In a real implementation, we might store the result or trigger further processing
            # For now, just log it
            self.logger.debug(f"Extraction result: {result}")

        except Exception as e:
            self.logger.error(f"Error processing log message: {e}")

    async def background_work(self) -> None:
        """Perform background work for log agent."""
        # Example: Periodic cleanup of temporary extraction data
        # In a real implementation, this might handle batch processing
        await asyncio.sleep(1)

    async def extract_and_store_batch(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and store information from multiple conversations.

        Args:
            conversations: List of conversation data

        Returns:
            Batch processing result
        """
        self.logger.info(f"Processing batch of {len(conversations)} conversations")

        results = {
            "total_conversations": len(conversations),
            "successfully_processed": 0,
            "failed_processing": 0,
            "total_items_stored": 0,
            "processing_errors": []
        }

        for i, conversation in enumerate(conversations):
            try:
                user_input = conversation.get("user_input", "")
                context = conversation.get("context", {})

                # Extract information
                extracted_info = await self._extract_information(user_input, context)

                # Store extracted information
                storage_result = await self._store_extracted_info(extracted_info, context)

                results["successfully_processed"] += 1
                results["total_items_stored"] += storage_result.get("stored_items", 0)

            except Exception as e:
                results["failed_processing"] += 1
                results["processing_errors"].append({
                    "conversation_index": i,
                    "error": str(e)
                })
                self.logger.error(f"Error processing conversation {i}: {e}")

        self.logger.info(f"Batch processing complete: {results['successfully_processed']} successful, {results['failed_processing']} failed")
        return results


# Factory function for easy instantiation
def create_log_agent(memory_manager = None) -> LogAgent:
    """Create a LogAgent instance.

    Args:
        memory_manager: Memory manager for storing extracted information

    Returns:
        LogAgent instance
    """
    return LogAgent(memory_manager=memory_manager)