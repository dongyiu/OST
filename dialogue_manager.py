# dialogue_manager.py

from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DialogueManager:
    def __init__(self, questions_engine, response_interpreter, db_manager):
        logger.info("Initializing Dialogue Manager")
        self.questions_engine = questions_engine
        self.interpreter = response_interpreter
        self.db = db_manager
        
    def initialize_conversation(self, user_id):
        """Start a new conversation flow."""
        logger.info(f"Initializing conversation for user {user_id}")
        
        # Generate initial questions
        questions = self.questions_engine.generate_questions(user_id)
        logger.info(f"Generated {len(questions)} initial questions")
        
        # Store in database
        self.db.save_questions(user_id, questions)
        
        # Get first question
        next_question = self.get_next_question(user_id)
        logger.info(f"Selected first question: {next_question['q']}")
        
        return next_question
        
    def get_next_question(self, user_id):
        """Get the next question to ask."""
        logger.info(f"Finding next question for user {user_id}")
        
        # Get variables that need information (low confidence)
        low_conf_vars = self.db.get_variables_by_confidence(user_id, threshold=70)
        
        if low_conf_vars:
            logger.info(f"Found {len(low_conf_vars)} variables with low confidence")
        
        # Get all questions
        all_questions = self.db.get_questions(user_id)
        
        # If we have low confidence variables, prioritize questions about them
        if low_conf_vars:
            for q in all_questions:
                if any(var in q["variables_involved"] for var in low_conf_vars):
                    logger.info(f"Selected question for low confidence variable: {q['q']}")
                    return q
        
        # Otherwise get the next unanswered question
        answered_questions = self.db.get_answered_questions(user_id)
        logger.debug(f"Already answered {len(answered_questions)} questions")
        
        for q in all_questions:
            if q["id"] not in answered_questions:
                logger.info(f"Selected next unanswered question: {q['q']}")
                return q
        
        # If all questions answered, check if clarification needed
        if low_conf_vars:
            # Generate clarification question for first low conf variable
            var_id = low_conf_vars[0]
            var_info = self.db.get_variable(user_id, var_id)
            current_value = var_info.get("value")
            
            clarification_q = {
                "id": f"clarify_{var_id}",
                "q": f"I understand you value {var_info['name']} at approximately {current_value}. Is that accurate?",
                "variables_involved": [var_id],
                "is_clarification": True
            }
            
            logger.info(f"Generated clarification question: {clarification_q['q']}")
            return clarification_q
        
        # All done!
        logger.info("All questions answered with high confidence, conversation complete")
        return {"id": "complete", "q": "Thank you! I have all the information I need."}
        
    def process_response(self, user_id, question_id, response_text):
        """Process user response to a question."""
        logger.info(f"Processing response for question {question_id}")
        logger.debug(f"Response text: {response_text}")
        
        # Get question details
        question = self.db.get_question(user_id, question_id)
        
        # For each variable involved
        for var_id in question["variables_involved"]:
            # Get variable info
            var_info = self.db.get_variable(user_id, var_id)
            
            # Interpret response for this variable
            interpretation = self.interpreter.interpret_response(
                question["q"], 
                response_text, 
                var_info
            )
            
            if interpretation:
                # Handle both list and dictionary responses
                if isinstance(interpretation, list):
                    # Find the interpretation for this variable
                    for interp in interpretation:
                        if interp.get("variable_id") == var_id:
                            # Update variable with new context
                            context = {
                                "raw_value": response_text,
                                "normalized_value": interp.get("normalized_value", 5.0),
                                "confidence": interp.get("confidence", 50.0),
                                "relevance": interp.get("relevance", 0.5),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            logger.info(f"Adding context for variable {var_id} with value {context['normalized_value']} (confidence: {context['confidence']}%)")
                            self.db.add_variable_context(user_id, var_id, context)
                else:
                    # Update variable with new context
                    context = {
                        "raw_value": response_text,
                        "normalized_value": interpretation.get("normalized_value", 5.0),
                        "confidence": interpretation.get("confidence", 50.0),
                        "relevance": interpretation.get("relevance", 0.5),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"Adding context for variable {var_id} with value {context['normalized_value']} (confidence: {context['confidence']}%)")
                    self.db.add_variable_context(user_id, var_id, context)
                    
        # Mark question as answered
        self.db.mark_question_answered(user_id, question_id)
        
        # Return next question
        return self.get_next_question(user_id)
