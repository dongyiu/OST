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
        self.min_questions_to_ask = 3  # Ensure at least this many questions are asked
        
    def initialize_conversation(self, user_id):
        """Start a new conversation flow."""
        logger.info(f"Initializing conversation for user {user_id}")
        
        # Generate initial questions
        questions = self.questions_engine.generate_questions(user_id)
        logger.info(f"Generated {len(questions)} initial questions")
        
        # Store in database
        self.db.save_questions(user_id, questions)
        
        # Initialize a counter for questions asked
        user_data = self.db._load_user_data(user_id)
        user_data["questions_asked"] = 0
        self.db._save_user_data(user_id, user_data)
        
        # Get first question
        next_question = self.get_next_question(user_id)
        logger.info(f"Selected first question: {next_question['q']}")
        
        return next_question
        
    def get_next_question(self, user_id):
        """Get the next question to ask."""
        logger.info(f"Finding next question for user {user_id}")
        
        # Load user data for tracking question count
        user_data = self.db._load_user_data(user_id)
        questions_asked = user_data.get("questions_asked", 0)
        
        # Get all questions
        all_questions = self.db.get_questions(user_id)
        answered_questions = self.db.get_answered_questions(user_id)
        
        # If we haven't asked enough questions yet, prioritize getting more questions
        if questions_asked < self.min_questions_to_ask:
            logger.info(f"Only {questions_asked} questions asked, need to ask at least {self.min_questions_to_ask}")
            
            # Find an unanswered question
            for q in all_questions:
                if q["id"] not in answered_questions:
                    logger.info(f"Selected next question: {q['q']} (enforcing minimum questions)")
                    return q
        
        # Get variables that need information (low confidence)
        low_conf_vars = self.db.get_variables_by_confidence(user_id, threshold=70)
        
        if low_conf_vars:
            logger.info(f"Found {len(low_conf_vars)} variables with low confidence")
            
            # If we have low confidence variables, prioritize questions about them
            for q in all_questions:
                if q["id"] not in answered_questions and any(var in q["variables_involved"] for var in low_conf_vars):
                    logger.info(f"Selected question for low confidence variable: {q['q']}")
                    return q
        
        # Otherwise get the next unanswered question
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
            
            # Get the current value for this variable
            var_context = user_data.get("variable_contexts", {}).get(var_id, {})
            current_value = var_context.get("final_value", 0)
            
            clarification_q = {
                "id": f"clarify_{var_id}",
                "q": f"I understand you value {var_info['name']} at approximately {current_value:.2f}. Is that accurate?",
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
        
        # Get all variable information at once
        variables_involved = []
        for var_id in question["variables_involved"]:
            var_info = self.db.get_variable(user_id, var_id)
            if var_info:
                variables_involved.append(var_info)
        
        # Process all variables at once with a single API call
        if variables_involved:
            interpretations = self.interpreter.interpret_response(
                question["q"], 
                response_text, 
                variables_involved
            )
            
            if interpretations:
                # Process each interpretation
                for interp in interpretations:
                    var_id = interp.get("variable_id")
                    if var_id:
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
        
        # Mark question as answered
        self.db.mark_question_answered(user_id, question_id)
        
        # Increment the questions asked counter
        user_data = self.db._load_user_data(user_id)
        user_data["questions_asked"] = user_data.get("questions_asked", 0) + 1
        self.db._save_user_data(user_id, user_data)
        
        # Return next question
        return self.get_next_question(user_id)