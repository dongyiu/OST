# adaptive_questions.py

import json
import logging
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)
reasoning_logger = logging.getLogger('reasoning')

class AdaptiveQuestionsEngine:
    def __init__(self, gemini_client, db_manager):
        logger.info("Initializing Adaptive Questions Engine")
        self.gemini = gemini_client
        self.db = db_manager
        
    def generate_questions(self, user_id):
        """Generate questions based on industry variables."""
        logger.info(f"Generating questions for user {user_id}")
        
        # Get user profile and industry
        user_profile = self.db.get_user_profile(user_id)
        if not user_profile:
            logger.error(f"No user profile found for user {user_id}")
            return []
            
        industry = user_profile.get("industry")
        variables = user_profile.get("variables", [])
        
        logger.info(f"Generating questions for {industry} industry with {len(variables)} variables")
        
        # Create prompt for Gemini
        prompt = f"""
        Generate conversational interview questions to determine user preferences for job variables 
        in the {industry} industry. The variables are:
        
        {json.dumps(variables, indent=2)}
        
        For each variable, create a question that:
        1. Is conversational and natural to answer
        2. Allows for qualitative responses (not just numerical)
        3. Avoids asking for exact percentages or numbers when possible
        4. Focuses on preferences and priorities

        For example, instead of "What percentage of remote work do you want?", ask 
        "Do you prefer working remotely, in-office, or a hybrid arrangement?"
        
        Return your response as a JSON array where each object has:
        - id: a unique question ID (q_0, q_1, etc.)
        - q: the natural, conversational question text
        - variables_involved: list of variable ids this question measures
        
        Make sure every variable is covered by at least one question.
        """
        
        reasoning_logger.info(f"Questions generation reasoning:\nGenerating natural questions for {industry} industry variables")
        logger.debug("Sending question generation prompt to Gemini")
        
        response = self.gemini.generate(prompt, structured_output=True)
        
        try:
            questions = json.loads(response)
            logger.info(f"Generated {len(questions)} questions")
            
            # Create a mapping of variable IDs to names for better logging
            var_id_to_name = {v["id"]: v["name"] for v in variables}
            
            # Log each question with the variable names it's targeting
            question_summary = []
            for q in questions:
                var_names = [var_id_to_name.get(var_id, var_id) for var_id in q.get("variables_involved", [])]
                var_str = ", ".join(var_names)
                question_summary.append(f"- Question: {q['q']}\n  Variables: {var_str}")
                
            reasoning_logger.info(f"Generated {len(questions)} questions:\n\n" + "\n\n".join(question_summary))
            
            # Validate questions cover all variables
            covered_vars = set()
            for q in questions:
                covered_vars.update(q.get("variables_involved", []))
                
            all_vars = {v["id"] for v in variables}
            missing_vars = all_vars - covered_vars
            
            if missing_vars:
                missing_var_names = [var_id_to_name.get(var_id, var_id) for var_id in missing_vars]
                logger.warning(f"Missing questions for variables: {missing_vars}")
                reasoning_logger.info(f"Warning: Missing questions for variables: {', '.join(missing_var_names)}")
                
            return questions
        except:
            logger.error("Failed to parse questions response")
            return []