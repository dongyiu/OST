import json
import logging
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)
reasoning_logger = logging.getLogger('reasoning')

class ResponseInterpreter:
    def __init__(self, gemini_client):
        logger.info("Initializing Response Interpreter")
        self.gemini = gemini_client
        
    def interpret_response(self, question, response, variable_infos):
        """Convert user response to quantifiable values for multiple variables at once."""
        var_ids = [var_info['id'] for var_info in variable_infos]
        logger.info(f"Interpreting response for variables: {', '.join(var_ids)}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Response: {response}")
        
        # Format variable information for the prompt
        var_details = []
        for var in variable_infos:
            var_details.append(f"Variable ID: {var['id']}\nName: {var['name']}\nDescription: {var['description']}\nScale: {var['scale']}")
        
        var_info_text = "\n\n".join(var_details)
        
        prompt = f"""
        Question: {question}
        User response: {response}
        
        This question relates to the following variables:
        
        {var_info_text}
        
        Your task: Convert this qualitative response into quantifiable values for EACH variable, even if it's flexible.
        
        Guidelines:
        - If the user is flexible or open to multiple options, assign a middle value with high confidence
        - If they express a preference but are open to alternatives, use a value that leans toward their preference
        - If they give a clear preference, use a value that strongly reflects that
        - If the response isn't relevant to a variable, assign a lower relevance score
        
        Return your answer as a JSON array where each object contains:
        - variable_id: the ID of the variable
        - normalized_value: numerical value interpretation (0-100 scale)
        - confidence: how confident you are in this interpretation (0-100%)
        - relevance: how relevant the response is to this variable (0-1)
        
        Example response format:
        [
          {{
            "variable_id": "{var_ids[0]}",
            "normalized_value": 75,
            "confidence": 90,
            "relevance": 0.9
          }},
          ... (one object for each variable)
        ]
        
        Importantly: When a user expresses flexibility, that's actually a high-confidence answer
        (they're confident they're flexible), not a low-confidence one.
        """
        
        var_names = [v['name'] for v in variable_infos]
        reasoning_logger.info(f"Response interpretation reasoning:\nInterpreting user response for variables: {', '.join(var_names)}:\nQ: {question}\nA: {response}")
        logger.debug("Sending interpretation prompt to Gemini")
        
        response = self.gemini.generate(prompt, structured_output=True)
        
        try:
            # Parse the response (could be a single object or an array)
            interpretation = json.loads(response)
            
            # Ensure interpretation is a list
            if not isinstance(interpretation, list):
                interpretation = [interpretation]
                
            # Log each interpretation
            for item in interpretation:
                # Find the variable info for this item
                var_info = next((v for v in variable_infos if v['id'] == item['variable_id']), None)
                if var_info:
                    var_name = var_info['name']
                    logger.info(f"Interpreted value for {var_name} ({item['variable_id']}): {item['normalized_value']} (confidence: {item['confidence']}%)")
                    reasoning_logger.info(f"Interpretation result: Variable '{var_name}' ({item['variable_id']}) = {item['normalized_value']} with {item['confidence']}% confidence")
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Failed to parse interpreter response: {e}")
            
            # Return default interpretations with high confidence for flexibility
            default_interpretations = []
            for var_info in variable_infos:
                default_interpretations.append({
                    "variable_id": var_info['id'],
                    "normalized_value": 50.0,  # Middle value representing flexibility
                    "confidence": 80.0,  # High confidence to prevent repeated questions
                    "relevance": 0.9
                })
            return default_interpretations