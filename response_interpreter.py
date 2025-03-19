# response_interpreter.py

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
        
        # Format variable information for the prompt, escaping curly braces in descriptions
        var_details = []
        for var in variable_infos:
            description = var['description'].replace('{', '{{').replace('}', '}}')
            var_details.append(f"Variable ID: {var['id']}\nName: {var['name']}\nDescription: {description}\nScale: {var['scale']}")
        
        var_info_text = "\n\n".join(var_details)
        
        prompt = f"""
        Question: {question}
        User response: {response}
        
        This question relates to the following variables:
        
        {var_info_text}
        
        Your task: Convert this qualitative response into quantifiable values for EACH variable, even if it's flexible.
        
        Guidelines:
        - IMPORTANT: If the user's response is nonsensical, irrelevant, or doesn't address the question (e.g., "ice cream"), assign a normalized_value of 50, confidence of 10% or lower, and relevance of 0.1 or lower for ALL variables.
        - If the user is flexible or open to multiple options, assign a middle value (50) with high confidence (70-90%)
        - If they express a preference but are open to alternatives, use a value that leans toward their preference (60-80 or 20-40)
        - If they give a clear preference, use a value that strongly reflects that (80-100 or 0-20)
        - If they mention a variable but provide no clear preference, assign medium relevance (0.5-0.7) and low confidence (30-50%)
        - If they don't mention a variable at all, assign low relevance (0-0.3)
        
        Return your answer as a JSON array where each object contains:
        - variable_id: the ID of the variable
        - normalized_value: numerical value interpretation (0-100 scale)
        - confidence: how confident you are in this interpretation (0-100%)
        - relevance: how relevant the response is to this variable (0-1)
        
        Example response format:
        [
          {{
            "variable_id": "var1",
            "normalized_value": 75,
            "confidence": 90,
            "relevance": 0.9
          }},
          ... (one object for each variable)
        ]
        
        Note: Only give high confidence scores (>70%) when there is clear, meaningful information in the response related to the variable.
        """

        var_names = [v['name'] for v in variable_infos]
        reasoning_logger.info(f"Response interpretation reasoning:\nInterpreting user response for variables: {', '.join(var_names)}:\nQ: {question}\nA: {response}")
        logger.debug("Sending interpretation prompt to Gemini")
        
        response = self.gemini.generate(prompt, structured_output=True)
        
        try:
            interpretation = json.loads(response)
            if not isinstance(interpretation, list):
                interpretation = [interpretation]
            for item in interpretation:
                if item.get("relevance", 0) <= 0:
                    item["confidence"] = 0
                    logger.info(f"Set confidence to 0 for {item['variable_id']} due to relevance <= 0")
            for item in interpretation:
                var_info = next((v for v in variable_infos if v['id'] == item['variable_id']), None)
                if var_info:
                    var_name = var_info['name']
                    logger.info(f"Interpreted value for {var_name} ({item['variable_id']}): {item['normalized_value']} (confidence: {item['confidence']}%)")
                    reasoning_logger.info(f"Interpretation result: Variable '{var_name}' ({item['variable_id']}) = {item['normalized_value']} with {item['confidence']}% confidence")
            return interpretation
        except Exception as e:
            logger.error(f"Failed to parse interpreter response: {e}")
            default_interpretations = []
            for var_info in variable_infos:
                default_interpretations.append({
                    "variable_id": var_info['id'],
                    "normalized_value": 50.0,
                    "confidence": 50.0,
                    "relevance": 0.5
                })
            return default_interpretations