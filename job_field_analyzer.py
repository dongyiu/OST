import json
import logging
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)
reasoning_logger = logging.getLogger('reasoning')

class JobFieldAnalyzer:
    def __init__(self, gemini_client):
        logger.info("Initializing Job Field Analyzer")
        self.gemini = gemini_client
        
    def analyze_resume(self, resume_text):
        """Analyze resume to identify industry and variables."""
        logger.info("Analyzing resume to identify industry and variables")
        
        prompt = f"""
        Analyze this resume to identify the person's primary industry, then generate important variables 
        that someone in this industry would consider when evaluating job offers.

        Resume:
        {resume_text}

        First, determine the industry this person works in based on their experience, skills, and education.
        
        Then, generate 10-50 quantifiable variables that would be important for someone in this industry 
        when evaluating job offers. The amount of variables will depend on the industry. These should include:
        
        1. Compensation factors (salary, bonuses, equity, etc.)
        2. Work-life balance factors (hours, remote work, flexibility)
        3. Career growth factors (promotion timeline, learning opportunities)
        4. Job security and stability
        5. Industry-specific benefits or perks
        
        For each variable, provide:
        - A clear name
        - A description of why it matters
        - An appropriate scale (monetary value, percentage, rating scale)
        - A reasonable default/baseline value for the industry
        
        Return your response as a valid JSON object with:
        ```json
        {{
        "industry": "The identified industry",
        "confidence": 90, 
        "variables": [
            {{
            "id": "var1",
            "name": "Base Salary",
            "description": "Annual base compensation before taxes and benefits",
            "scale": "dollar amount",
            "baseline": 75000
            }},
            // More variables...
        ]
        }}
        ```
        
        Your response must be a valid JSON object with no additional text.
        """
        
        logger.debug("Sending resume analysis prompt to Gemini")
        reasoning_logger.info(f"Resume analysis reasoning:\nIndustry identification and variable generation for resume")
        
        response = self.gemini.generate(prompt, structured_output=True)
        
        try:
            result = json.loads(response)
            industry = result.get("industry")
            confidence = result.get("confidence")
            variables = result.get("variables", [])
            
            logger.info(f"Identified industry: {industry} with {confidence}% confidence")
            logger.info(f"Generated {len(variables)} industry-specific variables")
            
            # Log each variable with its name and ID for better clarity
            variable_summary = "\n".join([f"- {v['name']} ({v['id']})" for v in variables])
            reasoning_logger.info(f"Industry identified: {industry} with {confidence}% confidence\n\nGenerated variables:\n{variable_summary}")
            
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse analyzer response - invalid JSON")
            logger.debug(f"Raw response: {response}")
            return None