# response_interpreter.py

import json
import logging
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class ResponseInterpreter:
    def __init__(self, gemini_client):
        logger.info("Initializing Response Interpreter")
        self.gemini = gemini_client
        
    def interpret_response(self, question, response, variable_info):
        """Convert user response to quantifiable values."""
        logger.info(f"Interpreting response for variable: {variable_info['id']}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Response: {response}")
        
        prompt = f"""
        Question: {question}
        User response: {response}
        
        This question relates to the variable: {variable_info['name']}
        Description: {variable_info['description']}
        Scale: {variable_info['scale']}
        
        Your task: Convert this qualitative response into a quantifiable value, even if it's flexible.
        
        Guidelines:
        - If the user is flexible or open to multiple options, assign a middle value with high confidence
        - If they express a preference but are open to alternatives, use a value that leans toward their preference
        - If they give a clear preference, use a value that strongly reflects that
        
        For a variable like "Remote Work Percentage":
        - "Hybrid is ok, fully remote is fine too, in office is also okay" → This shows complete flexibility, 
        so a value of 50% with 90% confidence would be appropriate
        
        Return your answer as JSON with:
        {{
        "variable_id": "{variable_info['id']}",
        "normalized_value": 50,  // Replace with appropriate numerical value
        "confidence": 90,  // Higher confidence for flexible answers
        "relevance": 0.9  // How directly the response relates to the variable
        }}
        
        Importantly: When a user expresses flexibility, that's actually a high-confidence answer
        (they're confident they're flexible), not a low-confidence one.
        """
        
        logger.debug("Sending interpretation prompt to Gemini")
        response = self.gemini.generate(prompt, structured_output=True)
        
        try:
            interpretation = json.loads(response)
            
            if isinstance(interpretation, list):
                for item in interpretation:
                    logger.info(f"Interpreted value for {item['variable_id']}: {item['normalized_value']} (confidence: {item['confidence']}%)")
            else:
                logger.info(f"Interpreted value for {interpretation['variable_id']}: {interpretation['normalized_value']} (confidence: {interpretation['confidence']}%)")
            
            return interpretation
        except:
            logger.error("Failed to parse interpreter response")
            # Return a default interpretation with high confidence for flexibility
            return {
                "variable_id": variable_info['id'],
                "normalized_value": 50.0,  # Middle value representing flexibility
                "confidence": 80.0,  # High confidence to prevent repeated questions
                "relevance": 0.9
            }
        