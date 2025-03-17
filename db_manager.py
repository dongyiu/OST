# db_manager.py

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, data_dir="./data"):
        logger.info("Initializing File-based Database Manager")
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
    
    def _get_user_file_path(self, user_id):
        return os.path.join(self.data_dir, f"{user_id}.json")
        
    def _load_user_data(self, user_id):
        file_path = self._get_user_file_path(user_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                return {}
        return {}
        
    def _save_user_data(self, user_id, data):
        file_path = self._get_user_file_path(user_id)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            return False
    
    def save_user_profile(self, user_id, profile_data):
        """Save user profile with industry and variables."""
        logger.info(f"Saving user profile for user {user_id}")
        
        user_data = self._load_user_data(user_id)
        user_data["user_profile"] = profile_data
        
        return self._save_user_data(user_id, user_data)
    
    def get_user_profile(self, user_id):
        """Get user profile."""
        logger.info(f"Getting user profile for user {user_id}")
        user_data = self._load_user_data(user_id)
        return user_data.get("user_profile")
    
    def save_questions(self, user_id, questions):
        """Save generated questions."""
        logger.info(f"Saving {len(questions)} questions for user {user_id}")
        
        # Add question IDs if not present
        for i, q in enumerate(questions):
            if "id" not in q:
                q["id"] = f"q_{i}"
        
        user_data = self._load_user_data(user_id)
        user_data["questions"] = questions
        
        return self._save_user_data(user_id, user_data)
    
    def get_questions(self, user_id):
        """Get all questions."""
        logger.info(f"Getting questions for user {user_id}")
        user_data = self._load_user_data(user_id)
        return user_data.get("questions", [])
    
    def get_question(self, user_id, question_id):
        """Get specific question by ID."""
        logger.info(f"Getting question {question_id} for user {user_id}")
        questions = self.get_questions(user_id)
        for q in questions:
            if q.get("id") == question_id:
                return q
        return None
    
    def get_variable(self, user_id, variable_id):
        """Get variable information."""
        logger.info(f"Getting variable {variable_id} for user {user_id}")
        user_profile = self.get_user_profile(user_id)
        if not user_profile:
            return None
            
        for var in user_profile.get("variables", []):
            if var.get("id") == variable_id:
                return var
        return None
    
    def get_variables_by_confidence(self, user_id, threshold=70):
        """Get variables below confidence threshold."""
        logger.info(f"Getting variables below {threshold}% confidence for user {user_id}")
        user_data = self._load_user_data(user_id)
        var_contexts = user_data.get("variable_contexts", {})
        
        low_conf_vars = []
        for var_id, context in var_contexts.items():
            if context.get("current_confidence", 0) < threshold:
                low_conf_vars.append(var_id)
                
        return low_conf_vars
    
    def add_variable_context(self, user_id, variable_id, context):
        """Add new context for a variable."""
        logger.info(f"Adding context for variable {variable_id}")
        
        # Validate context has required fields
        if not isinstance(context, dict):
            logger.error(f"Invalid context format: {context}")
            return False
            
        # Ensure normalized_value and relevance are valid numbers
        if "normalized_value" not in context or not isinstance(context["normalized_value"], (int, float)):
            logger.warning(f"Invalid normalized_value in context: {context}")
            # Set default values
            context["normalized_value"] = 5.0  # Default middle value
            
        if "relevance" not in context or not isinstance(context["relevance"], (int, float)):
            logger.warning(f"Invalid relevance in context: {context}")
            # Set default values
            context["relevance"] = 0.5  # Default middle relevance
            
        if "confidence" not in context or not isinstance(context["confidence"], (int, float)):
            logger.warning(f"Invalid confidence in context: {context}")
            # Set default values
            context["confidence"] = 50.0  # Default middle confidence
        
        # Load user data
        user_data = self._load_user_data(user_id)
        
        if "variable_contexts" not in user_data:
            user_data["variable_contexts"] = {}
            
        if variable_id not in user_data["variable_contexts"]:
            user_data["variable_contexts"][variable_id] = {"contexts": []}
            
        # Add new context
        user_data["variable_contexts"][variable_id]["contexts"].append(context)
        
        # Get contexts and filter out any with invalid values
        contexts = user_data["variable_contexts"][variable_id]["contexts"]
        valid_contexts = [
            c for c in contexts 
            if isinstance(c.get("normalized_value"), (int, float)) and 
            isinstance(c.get("relevance"), (int, float)) and
            c.get("relevance") > 0  # Prevent division by zero
        ]
        
        # Calculate new values if we have valid contexts
        if valid_contexts:
            # Calculate new confidence (weighted average)
            total_relevance = sum(c["relevance"] for c in valid_contexts)
            weighted_confidence = sum(c["confidence"] * c["relevance"] for c in valid_contexts) / total_relevance
            
            # Calculate final value (weighted average)
            weighted_value = sum(c["normalized_value"] * c["relevance"] for c in valid_contexts) / total_relevance
            
            logger.info(f"New weighted value for {variable_id}: {weighted_value:.2f} (confidence: {weighted_confidence:.2f}%)")
            
            # Update calculated values
            user_data["variable_contexts"][variable_id]["current_confidence"] = weighted_confidence
            user_data["variable_contexts"][variable_id]["final_value"] = weighted_value
        else:
            # No valid contexts, use the most recent context values directly
            if contexts:
                latest = contexts[-1]
                user_data["variable_contexts"][variable_id]["current_confidence"] = latest.get("confidence", 50.0)
                user_data["variable_contexts"][variable_id]["final_value"] = latest.get("normalized_value", 5.0)
                logger.info(f"Using direct values for {variable_id} due to no valid contexts")
            else:
                # Default values if no contexts at all
                user_data["variable_contexts"][variable_id]["current_confidence"] = 0.0
                user_data["variable_contexts"][variable_id]["final_value"] = 0.0
                logger.warning(f"No valid contexts for {variable_id}, using defaults")
        
        return self._save_user_data(user_id, user_data)

    def get_answered_questions(self, user_id):
        """Get IDs of answered questions."""
        logger.info(f"Getting answered questions for user {user_id}")
        user_data = self._load_user_data(user_id)
        return user_data.get("answered_questions", [])
    
    def mark_question_answered(self, user_id, question_id):
        """Mark a question as answered."""
        logger.info(f"Marking question {question_id} as answered")
        
        user_data = self._load_user_data(user_id)
        
        if "answered_questions" not in user_data:
            user_data["answered_questions"] = []
            
        if question_id not in user_data["answered_questions"]:
            user_data["answered_questions"].append(question_id)
            
        return self._save_user_data(user_id, user_data)