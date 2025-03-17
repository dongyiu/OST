import logging
import json
import google.generativeai as genai
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
reasoning_logger = logging.getLogger('reasoning')

class GeminiClient:
    def __init__(self):
        logger.info("Initializing Gemini client")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate(self, prompt, structured_output=False):
        logger.debug(f"Generating content with prompt: {prompt[:100]}...")
        
        # Log the AI-to-AI reasoning process
        reasoning_logger.info(f"Prompt sent to Gemini:\n{prompt}")
        
        try:
            # If structured output is requested, modify the prompt
            if structured_output:
                logger.info("Requesting structured JSON output via prompt")
                # Add JSON instructions to the prompt
                prompt = f"{prompt}\n\nYour response must be in valid JSON format only, with no additional text, explanations, or markdown formatting."
                
            response = self.model.generate_content(prompt)
            
            logger.info("Successfully received response from Gemini")
            reasoning_logger.info(f"Response from Gemini:\n{response.text}")
            
            # Try to clean and parse JSON responses
            if structured_output:
                try:
                    # Clean up response if necessary (remove markdown code blocks if present)
                    response_text = response.text
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    # Validate it's proper JSON by attempting to parse it
                    json.loads(response_text)
                    logger.info("Successfully validated JSON response")
                    return response_text
                except json.JSONDecodeError as e:
                    logger.warning(f"Response looks like JSON but failed to parse: {e}")
                    logger.debug(f"Raw response: {response.text}")
                    
            return response.text
            
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return None