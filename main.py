import argparse
import logging
import os
import sys
import json
from config import GEMINI_API_KEY

from pdf_parser import extract_text_from_pdf
from job_field_analyzer import JobFieldAnalyzer
from adaptive_questions import AdaptiveQuestionsEngine
from response_interpreter import ResponseInterpreter
from dialogue_manager import DialogueManager
from db_manager import DatabaseManager
from gemini_client import GeminiClient

logger = logging.getLogger(__name__)

def main():
    # Default resume path
    resume_path = "resume.pdf"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Job Offer Data Collection Tool")
    parser.add_argument("--resume", help="Path to resume PDF file")
    
    args = parser.parse_args()
    resume_file = args.resume if args.resume else resume_path

    # Check if required environment variables are set
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY must be set in .env file")
        sys.exit(1)
    
    # Check if resume file exists
    if not os.path.exists(resume_file):
        print(f"Error: Resume file not found: {resume_file}")
        sys.exit(1)
    
    # Initialize components
    try:
        gemini_client = GeminiClient()
        db_manager = DatabaseManager()  # We'll still use this for temporary storage
        job_analyzer = JobFieldAnalyzer(gemini_client)
        questions_engine = AdaptiveQuestionsEngine(gemini_client, db_manager)
        response_interpreter = ResponseInterpreter(gemini_client)
        dialogue_manager = DialogueManager(questions_engine, response_interpreter, db_manager)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        print(f"Error initializing application: {e}")
        sys.exit(1)
    
    # Use a temporary session ID
    session_id = f"session_{os.urandom(4).hex()}"
    logger.info(f"Starting session: {session_id}")
    
    # Process resume
    logger.info(f"Processing resume: {resume_file}")
    print(f"Processing resume: {resume_file}...")
    
    # Extract resume text
    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text:
        print("Error: Failed to extract text from resume")
        sys.exit(1)
        
    # Analyze resume
    analysis = job_analyzer.analyze_resume(resume_text)
    if not analysis:
        print("Error: Failed to analyze resume")
        sys.exit(1)
        
    # Save analysis for this session
    db_manager.save_user_profile(session_id, analysis)
    
    # Start conversation
    next_question = dialogue_manager.initialize_conversation(session_id)
    
    # Print industry identification
    print(f"\nIdentified industry: {analysis.get('industry')} (Confidence: {analysis.get('confidence')}%)")
    print(f"Generated {len(analysis.get('variables', []))} industry-specific variables\n")
    
    # Main conversation loop
    while next_question and next_question.get("id") != "complete":
        print(f"Question: {next_question['q']}")
        response = input("Your answer: ")
        
        if response.lower() in ['exit', 'quit']:
            print("\nExiting conversation.")
            break
            
        # Process response
        next_question = dialogue_manager.process_response(session_id, next_question["id"], response)
    
    # Conversation complete
    if next_question and next_question.get("id") == "complete":
        print("\n" + next_question["q"])
        
        # Get final data
        user_data = db_manager._load_user_data(session_id)
        var_contexts = user_data.get("variable_contexts", {})
        industry_info = user_data.get("user_profile", {})
        
        print("\n--- Collected Data Summary ---")
        print(f"Industry: {industry_info.get('industry')}")
        
        if var_contexts:
            print("\nVariable values:")
            for var_id, context in var_contexts.items():
                var_info = next((v for v in industry_info.get("variables", []) if v["id"] == var_id), {})
                var_name = var_info.get("name", var_id)
                value = context.get("final_value")
                confidence = context.get("current_confidence")
                
                print(f"- {var_name}: {value:.2f} (Confidence: {confidence:.2f}%)")
        
        print("\nThank you for using the Job Offer Data Collection Tool!")
        
        # Clean up session data
        try:
            os.remove(db_manager._get_user_file_path(session_id))
            logger.info(f"Cleaned up session data: {session_id}")
        except:
            logger.warning(f"Could not clean up session data: {session_id}")

if __name__ == "__main__":
    main()