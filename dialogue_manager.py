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
        self.min_questions_to_ask = 3
        
    def initialize_conversation(self, user_id):
        logger.info(f"Initializing conversation for user {user_id}")
        questions = self.questions_engine.generate_questions(user_id)
        logger.info(f"Generated {len(questions)} initial questions")
        self.db.save_questions(user_id, questions)
        
        user_data = self.db._load_user_data(user_id)
        user_data["questions_asked"] = 0
        self.db._save_user_data(user_id, user_data)
        
        next_question = self.get_next_question(user_id)
        logger.info(f"Selected first question: {next_question['q']}")
        return next_question
        
    def get_next_question(self, user_id):
        logger.info(f"Finding next question for user {user_id}")
        user_data = self.db._load_user_data(user_id)
        questions_asked = user_data.get("questions_asked", 0)
        all_questions = self.db.get_questions(user_id)
        answered_questions = self.db.get_answered_questions(user_id)
        
        if questions_asked < self.min_questions_to_ask:
            logger.info(f"Only {questions_asked} questions asked, need to ask at least {self.min_questions_to_ask}")
            for q in all_questions:
                if q["id"] not in answered_questions:
                    logger.info(f"Selected next question: {q['q']} (enforcing minimum questions)")
                    return q
        
        low_conf_vars = self.db.get_variables_by_confidence(user_id, threshold=70)
        if low_conf_vars:
            logger.info(f"Found {len(low_conf_vars)} variables with low confidence")
            for q in all_questions:
                if q["id"] not in answered_questions and any(var in q["variables_involved"] for var in low_conf_vars):
                    logger.info(f"Selected question for low confidence variable: {q['q']}")
                    return q
        
        logger.debug(f"Already answered {len(answered_questions)} questions")
        for q in all_questions:
            if q["id"] not in answered_questions:
                logger.info(f"Selected next unanswered question: {q['q']}")
                return q
        
        if low_conf_vars:
            var_id = low_conf_vars[0]
            var_info = self.db.get_variable(user_id, var_id)
            var_context = user_data.get("variable_contexts", {}).get(var_id, {})
            current_value = var_context.get("final_value", 50)
            
            clarification_q = {
                "id": f"clarify_{var_id}",
                "q": f"I’m not sure I understood your preference for {var_info['name']}. Could you tell me more about what you’d like here?",
                "variables_involved": [var_id],
                "is_clarification": True
            }
            logger.info(f"Generated clarification question: {clarification_q['q']}")
            return clarification_q
        
        logger.info("All questions answered with high confidence, conversation complete")
        return {"id": "complete", "q": "Thank you! I have all the information I need."}
        
    def process_response(self, user_id, question_id, response_text):
        logger.info(f"Processing response for question {question_id}")
        logger.debug(f"Response text: {response_text}")
        
        question = self.db.get_question(user_id, question_id)
        if not question:
            logger.error(f"Question {question_id} not found")
            return self.get_next_question(user_id)
        
        variables_involved = []
        for var_id in question["variables_involved"]:
            var_info = self.db.get_variable(user_id, var_id)
            if var_info:
                variables_involved.append(var_info)
        
        if variables_involved:
            interpretations = self.interpreter.interpret_response(
                question["q"], 
                response_text, 
                variables_involved
            )
            
            if interpretations:
                needs_clarification = False
                for interp in interpretations:
                    var_id = interp.get("variable_id")
                    if var_id:
                        context = {
                            "raw_value": response_text,
                            "normalized_value": interp.get("normalized_value", 50.0),
                            "confidence": interp.get("confidence", 50.0),
                            "relevance": interp.get("relevance", 0.5),
                            "timestamp": datetime.now().isoformat()
                        }
                        logger.info(f"Adding context for variable {var_id} with value {context['normalized_value']} (confidence: {context['confidence']}%)")
                        self.db.add_variable_context(user_id, var_id, context)
                        
                        # Check if clarification is needed
                        if interp.get("relevance", 0) < 0.2 or interp.get("confidence", 0) < 20:
                            needs_clarification = True
                            logger.info(f"Low relevance ({interp['relevance']}) or confidence ({interp['confidence']}) for {var_id}, may need clarification")
                
                # Mark question as answered only if no clarification is needed
                if not needs_clarification:
                    self.db.mark_question_answered(user_id, question_id)
                    user_data = self.db._load_user_data(user_id)
                    user_data["questions_asked"] = user_data.get("questions_asked", 0) + 1
                    self.db._save_user_data(user_id, user_data)
                else:
                    var_id = interpretations[0]["variable_id"]
                    var_info = self.db.get_variable(user_id, var_id)
                    clarification_q = {
                        "id": f"clarify_{question_id}_{var_id}",
                        "q": f"Sorry, I didn't quite catch that. Could you clarify what you mean about {var_info['name']}?",
                        "variables_involved": [var_id],
                        "is_clarification": True
                    }
                    logger.info(f"Response unclear, asking clarification: {clarification_q['q']}")
                    return clarification_q
        
        else:
            self.db.mark_question_answered(user_id, question_id)
            user_data = self.db._load_user_data(user_id)
            user_data["questions_asked"] = user_data.get("questions_asked", 0) + 1
            self.db._save_user_data(user_id, user_data)
        
        return self.get_next_question(user_id)
    