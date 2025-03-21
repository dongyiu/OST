from typing import TypedDict, List
import google.generativeai as genai
import PyPDF2
from pymongo import MongoClient
from langgraph.graph import StateGraph
import time
import numpy as np
from difflib import SequenceMatcher
import random
import backoff
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError

# Simple cache implementation
class SimpleCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove a random item if cache is full
            random_key = random.choice(list(self.cache.keys()))
            del self.cache[random_key]
        self.cache[key] = value

# Create a cache instance
api_cache = SimpleCache(max_size=200)

# Rate-limited API wrapper
class RateLimitedAPI:
    def __init__(self, model, min_delay=1.0, max_delay=5.0):
        self.model = model
        self.min_delay = min_delay  # Minimum delay between API calls
        self.max_delay = max_delay  # Maximum delay for exponential backoff
        self.last_call_time = 0
        self.consecutive_errors = 0
    
    # Exponential backoff decorator for handling rate limits
    @backoff.on_exception(
        backoff.expo,
        (ResourceExhausted, ServiceUnavailable),
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter
    )
    def generate_content(self, prompt):
        # First check if we have this in cache
        cache_key = hash(prompt)
        cached_result = api_cache.get(cache_key)
        if cached_result:
            print("📋 Using cached result")
            return cached_result
        
        # Ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        # Calculate delay based on consecutive errors (exponential backoff)
        delay = self.min_delay * (2 ** self.consecutive_errors)
        delay = min(delay, self.max_delay)  # Cap at max_delay
        
        if time_since_last_call < delay:
            sleep_time = delay - time_since_last_call
            print(f"⏱️ Rate limiting: Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        try:
            print(f"🔄 Making API call (delay: {delay:.2f}s)")
            response = self.model.generate_content(prompt)
            
            # Reset consecutive errors on success
            self.consecutive_errors = 0
            
            # Update last call time
            self.last_call_time = time.time()
            
            # Cache the result
            api_cache.set(cache_key, response)
            
            return response
        
        except (ResourceExhausted, ServiceUnavailable) as e:
            # Increment consecutive errors to increase backoff
            self.consecutive_errors += 1
            print(f"⚠️ API Rate limit hit. Consecutive errors: {self.consecutive_errors}")
            # Let backoff handler retry
            raise
            
        except GoogleAPIError as e:
            print(f"❌ Google API Error: {str(e)}")
            # For other API errors, also backoff but handle them specially
            self.consecutive_errors += 1
            # Create a simple response object with error info
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.text = f"Error occurred: {error_msg}. Using fallback response."
            
            return ErrorResponse(str(e))
        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            # For unexpected errors, return a stub response
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.text = f"Unexpected error: {error_msg}. Using fallback response."
            
            return ErrorResponse(str(e))

# Configure AI model (replace with your API key)
# GOOGLE_API_KEY = "your_api_key_here"
# genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
rate_limited_model = RateLimitedAPI(model, min_delay=2.0, max_delay=10.0)

# Meta-Reasoning Agent class - monitors and analyzes the reasoning process
class MetaReasoningAgent:
    def __init__(self, max_consecutive_calls=5, max_reasoning_per_variable=3):
        self.consecutive_calls = 0
        self.max_consecutive_calls = max_consecutive_calls
        self.total_calls = 0
        self.reasoning_history = {}  # Track reasoning per variable
        self.max_reasoning_per_variable = max_reasoning_per_variable
        self.last_responses = {}  # Track last AI response per variable for similarity check
        self.last_analysis_time = time.time()
    
    def increment(self):
        """Increment both consecutive and total call counters"""
        self.consecutive_calls += 1
        self.total_calls += 1
        print(f"API Call {self.total_calls} (Consecutive: {self.consecutive_calls}/{self.max_consecutive_calls})")
        
    def reset_consecutive(self):
        """Reset just the consecutive counter (after user input)"""
        self.consecutive_calls = 0
        print("Reset consecutive API call counter due to user interaction")
    
    def track_reasoning(self, variable, response, confidence):
        """Track reasoning history for a specific variable"""
        if variable not in self.reasoning_history:
            self.reasoning_history[variable] = []
        
        self.reasoning_history[variable].append({
            "response": response,
            "confidence": confidence,
            "time": time.time()
        })
        
        # Store latest response text for similarity comparison
        current_response = str(response)
        if variable in self.last_responses:
            similarity = SequenceMatcher(None, self.last_responses[variable], current_response).ratio()
            print(f"Response similarity for {variable}: {similarity:.2f}")
        self.last_responses[variable] = current_response
    
    def should_continue_reasoning(self, variable, confidence):
        """Analyze if more reasoning is productive for this variable"""
        # If we haven't reasoned about this variable much, continue
        if variable not in self.reasoning_history or len(self.reasoning_history[variable]) < 2:
            return True, "Initial reasoning"
        
        # If confidence is already high enough, no need for more reasoning
        if confidence >= 0.7:
            return False, "Confidence threshold met"
        
        # Check if we've done too much reasoning on this variable
        if len(self.reasoning_history[variable]) >= self.max_reasoning_per_variable:
            return False, f"Maximum reasoning attempts ({self.max_reasoning_per_variable}) reached for {variable}"
        
        # Check for progress in confidence
        history = self.reasoning_history[variable]
        if len(history) >= 2:
            confidence_progress = history[-1]["confidence"] - history[-2]["confidence"]
            # If confidence is decreasing or not improving much, stop
            if confidence_progress <= 0.05:
                return False, f"Minimal confidence improvement: {confidence_progress:.2f}"
        
        # Check similarity with previous response (circular reasoning check)
        if variable in self.last_responses and self.last_responses[variable]:
            similarity = SequenceMatcher(None, self.last_responses[variable], str(history[-1]["response"])).ratio()
            if similarity > 0.8:  # High similarity threshold
                return False, f"Potential circular reasoning detected (similarity: {similarity:.2f})"
        
        return True, "Reasoning is still productive"
    
    def analyze_reasoning_process(self):
        """Provides meta-analysis of the entire reasoning process"""
        # Only analyze after some time has passed to avoid too frequent analyses
        current_time = time.time()
        if current_time - self.last_analysis_time < 5:  # At least 5 seconds between analyses
            return None
        
        self.last_analysis_time = current_time
        
        if self.consecutive_calls >= 3:  # Analyze after 3+ consecutive calls
            total_variables = len(self.reasoning_history)
            variables_at_threshold = sum(1 for var in self.reasoning_history 
                                        if len(self.reasoning_history[var]) >= self.max_reasoning_per_variable)
            
            avg_confidence = np.mean([history[-1]["confidence"] 
                                     for var, history in self.reasoning_history.items() 
                                     if history])
            
            # If we're analyzing too much with little progress
            if variables_at_threshold > 0 or avg_confidence < 0.5:
                return {
                    "status": "inefficient",
                    "message": f"Reasoning appears inefficient. Analyzed {total_variables} variables with avg confidence {avg_confidence:.2f}.",
                    "recommendation": "Consider asking user for clarification rather than additional reasoning."
                }
            
            # If we're making reasonable progress
            if avg_confidence >= 0.6:
                return {
                    "status": "productive",
                    "message": f"Reasoning is productive with {total_variables} variables and avg confidence {avg_confidence:.2f}.",
                    "recommendation": "Continue current approach."
                }
        
        return None
    
    def check_and_handle(self, variable=None, confidence=None):
        """
        Check if we should continue reasoning and handle accordingly.
        Returns True to proceed with reasoning, False to skip.
        """
        # Track variable-specific reasoning if provided
        if variable is not None and confidence is not None:
            continue_reasoning, reason = self.should_continue_reasoning(variable, confidence)
            if not continue_reasoning:
                print(f"⚠️ Stopping further reasoning on '{variable}': {reason}")
                return False
        
        # Check overall consecutive call limit
        if self.consecutive_calls >= self.max_consecutive_calls:
            analysis = self.analyze_reasoning_process()
            if analysis:
                print(f"\n🔍 Meta-analysis: {analysis['message']}")
                print(f"Recommendation: {analysis['recommendation']}")
            
            print(f"\n⚠️ Rate limit reached: {self.consecutive_calls} consecutive AI reasoning steps.")
            print(f"Total API calls so far: {self.total_calls}")
            user_choice = input("Press Enter to continue reasoning, or type 'skip' to use simpler results: ")
            if user_choice.lower() == 'skip':
                return False
            self.reset_consecutive()
        
        # Perform occasional reasoning analysis
        elif self.consecutive_calls >= 3:
            analysis = self.analyze_reasoning_process()
            if analysis and analysis['status'] == 'inefficient':
                print(f"\n🔍 Meta-analysis: {analysis['message']}")
                print(f"Recommendation: {analysis['recommendation']}")
                user_choice = input("Continue with more reasoning? (y/n): ")
                if user_choice.lower() != 'y':
                    return False
                
        return True

# Create meta-reasoning agent
meta_agent = MetaReasoningAgent(max_consecutive_calls=5, max_reasoning_per_variable=3)

# State structure for LangGraph
class State(TypedDict):
    pdf_path: str
    resume_text: str
    job_field: str
    questions: List[dict]  # {"variable": str, "question": str}
    user_data: List[dict]  # {"question_type": str, "question": str, "response": float, "data_type": str}
    conversation_history: List[dict]  # {"variable": str, "question": str, "response": str}

# Standard variables (all numerical)
standard_variables = [
    {"key": "min_salary", "type": "number", "min": 0, "max": 10000000, "description": "Minimum acceptable annual salary in USD"},
    {"key": "compensation_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of compensation (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "career_growth_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of career growth (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "work_life_balance_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of work-life balance (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "company_reputation_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of company reputation (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "location_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of location (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "role_responsibilities_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of role responsibilities (1-5, 1=unimportant, 5=extremely important)"},
    {"key": "risk_tolerance", "type": "number", "min": 1, "max": 10, "description": "Willingness to wait for better offers (1-10, 1=low, 10=high)"},
    {"key": "financial_runway", "type": "number", "min": 0, "max": 120, "description": "Months you can afford to search for a job"},
    {"key": "job_search_urgency", "type": "number", "min": 1, "max": 10, "description": "Urgency to find a new job (1-10, 1=not urgent, 10=very urgent)"},
    {"key": "current_salary", "type": "number", "min": 0, "max": 10000000, "description": "Current or last annual salary in USD"}
]

# Node 1: Parse PDF
def pdf_parser(state: State) -> dict:
    try:
        with open(state["pdf_path"], "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        return {"resume_text": text, "conversation_history": []}
    except Exception as e:
        print(f"❌ Error parsing PDF: {str(e)}")
        return {"resume_text": "Failed to parse resume", "conversation_history": []}

# Node 2: Extract job field
def job_field_extractor(state: State) -> dict:
    # User input happens before this node, so reset consecutive counter
    meta_agent.reset_consecutive()
    
    prompt = f"Extract the primary job field in 1-2 words from the resume: {state['resume_text']}"
    
    try:
        response = rate_limited_model.generate_content(prompt)
        meta_agent.increment()
        
        job_field = response.text.strip()
        meta_agent.track_reasoning("job_field", job_field, 0.9)  # Typically high confidence
        
        return {"job_field": job_field}
    except Exception as e:
        print(f"❌ Error in job_field_extractor: {str(e)}")
        # Fallback value
        fallback_field = "unknown"
        meta_agent.track_reasoning("job_field", fallback_field, 0.5)
        return {"job_field": fallback_field}

# Node 3: Generate initial personalized questions
def question_generator(state: State) -> dict:
    # Check if we should continue reasoning or use simple output
    if not meta_agent.check_and_handle():
        # Return default questions if meta-agent suggests skipping
        default_questions = [
            {"variable": "min_salary", "question": "What is your minimum acceptable salary?"},
            {"variable": "work_life_balance_weight", "question": "How important is work-life balance to you (1-5)?"},
            {"variable": "job_search_urgency", "question": "How urgently do you need to find a job (1-10)?"}
        ]
        meta_agent.track_reasoning("question_generation", default_questions, 0.5)
        return {"questions": default_questions}
    
    try:
        standard_keys = ", ".join([var["key"] for var in standard_variables])
        prompt = f"""
        Based on the resume, generate personalized questions for these standard variables:
        - min_salary: Minimum acceptable annual salary in USD
        - compensation_weight: Importance of compensation (1-5, 1=unimportant, 5=extremely important)
        - career_growth_weight: Importance of career growth (1-5, 1=unimportant, 5=extremely important)
        - work_life_balance_weight: Importance of work-life balance (1-5, 1=unimportant, 5=extremely important)
        - company_reputation_weight: Importance of company reputation (1-5, 1=unimportant, 5=extremely important)
        - location_weight: Importance of location (1-5, 1=unimportant, 5=extremely important)
        - role_responsibilities_weight: Importance of role responsibilities (1-5, 1=unimportant, 5=extremely important)
        - risk_tolerance: Willingness to wait for better offers (1-10, 1=low, 10=high)
        - financial_runway: Months you can afford to search for a job
        - job_search_urgency: Urgency to find a new job (1-10, 1=not urgent, 10=very urgent)
        - current_salary: Current or last annual salary in USD
        Add 3-5 extra questions based on the resume, each asking for a 1-5 rating, with unique variable names not in the standard list ({standard_keys}).
        Ensure all questions are distinct and tailored to the user's background.
        Format:
        Variable: [variable_name]
        Question: [question_text]
        Resume: {state['resume_text']}
        """
        
        response = rate_limited_model.generate_content(prompt)
        meta_agent.increment()
        
        lines = response.text.strip().split("\n")
        questions = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("Variable:"):
                variable = lines[i].split(":", 1)[1].strip()
                i += 1
                if i < len(lines) and lines[i].startswith("Question:"):
                    question = lines[i].split(":", 1)[1].strip()
                    questions.append({"variable": variable, "question": question})
                else:
                    print(f"Warning: Question missing for {variable}")
            i += 1
        
        if not questions:
            # Fallback if parsing failed
            questions = [
                {"variable": "min_salary", "question": "What is your minimum acceptable salary?"},
                {"variable": "work_life_balance_weight", "question": "How important is work-life balance to you (1-5)?"},
                {"variable": "job_search_urgency", "question": "How urgently do you need to find a job (1-10)?"}
            ]
            print("⚠️ Failed to parse questions, using fallbacks")
        
        # Track this reasoning with high confidence as it's a generative task
        meta_agent.track_reasoning("question_generation", questions, 0.8)
        return {"questions": questions}
        
    except Exception as e:
        print(f"❌ Error in question_generator: {str(e)}")
        # Fallback questions in case of error
        default_questions = [
            {"variable": "min_salary", "question": "What is your minimum acceptable salary?"},
            {"variable": "work_life_balance_weight", "question": "How important is work-life balance to you (1-5)?"},
            {"variable": "job_search_urgency", "question": "How urgently do you need to find a job (1-10)?"}
        ]
        meta_agent.track_reasoning("question_generation", default_questions, 0.5)
        return {"questions": default_questions}

# Interpretation function with confidence
def interpret_response(variable, question, responses, schema, job_field, conversation_history):
    # Get the last confidence for this variable if available
    last_confidence = 0
    if variable in meta_agent.reasoning_history and meta_agent.reasoning_history[variable]:
        last_confidence = meta_agent.reasoning_history[variable][-1]["confidence"]
    
    # Check if we should continue reasoning based on previous attempts
    if not meta_agent.check_and_handle(variable, last_confidence):
        # Return best estimate so far or midpoint if no history
        if variable in meta_agent.reasoning_history and meta_agent.reasoning_history[variable]:
            best_estimate = meta_agent.reasoning_history[variable][-1]["response"]
            confidence = meta_agent.reasoning_history[variable][-1]["confidence"]
            return best_estimate, confidence
        return (schema['min'] + schema['max']) / 2, 0.7
    
    try:
        history_text = "\n".join([f"Q: {h['question']}\nA: {h['response']}" for h in conversation_history])
        prompt = f"""
        Estimate a numerical value for '{variable}' based on the user's responses.
        Initial question: '{question}'.
        Current responses for this question: {responses}
        Job field: '{job_field}'.
        Full conversation history:
        {history_text}

        Rules:
        - Output a number between {schema['min']} and {schema['max']}.
        - Output a confidence score (0 to 1) for your estimate.
        - Use all responses and history to inform the estimate.
        - If a number is provided (e.g., '50k'), convert it ('k'=1000, 'm'=1000000).
        - For text, estimate based on sentiment, context, and job field norms.
        - For related variables (e.g., financial_runway, risk_tolerance), ensure consistency with financial data if provided.

        Examples:
        - Variable: min_salary, Responses: ['50k'] -> Number: 50000, Confidence: 1.0
        - Variable: financial_runway, Responses: ['50k savings', '3k per week rent'] -> Number: 4.17, Confidence: 1.0
        - Variable: risk_tolerance, Responses: ['Can't wait long'], History: '50k savings, 3k/week rent' -> Number: 2, Confidence: 0.9

        Output format:
        Number: [estimated_number]
        Confidence: [confidence_score]
        """
        ai_response = rate_limited_model.generate_content(prompt)
        meta_agent.increment()
        
        print(f"AI response for {variable}: {ai_response.text}")
        lines = ai_response.text.strip().split("\n")
        number = confidence = None
        for line in lines:
            if line.startswith("Number:"):
                try:
                    number = float(line.split(":", 1)[1].strip())
                except ValueError:
                    print(f"⚠️ Warning: Could not parse number from '{line}'")
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    print(f"⚠️ Warning: Could not parse confidence from '{line}'")
        
        if number is not None and 0 <= confidence <= 1:
            result = max(schema['min'], min(schema['max'], number))
            # Track this reasoning
            meta_agent.track_reasoning(variable, result, confidence)
            return result, confidence
            
    except Exception as e:
        print(f"❌ Error in interpret_response for {variable}: {str(e)}")
    
    # Fallback for errors or parsing failures
    print(f"⚠️ AI failed for '{variable}', using midpoint and low confidence")
    result = (schema['min'] + schema['max']) / 2
    meta_agent.track_reasoning(variable, result, 0.5)
    return result, 0.5

# Generate follow-up question
def generate_followup_question(variable, responses, job_field, conversation_history):
    # Check if we should continue reasoning
    if not meta_agent.check_and_handle():
        # Return a simple follow-up if meta-agent suggests skipping
        return f"Please provide a clearer numerical value for {variable}:"
    
    try:
        history_text = "\n".join([f"Q: {h['question']}\nA: {h['response']}" for h in conversation_history])
        prompt = f"""
        The user's responses for '{variable}' were unclear.
        Responses: {responses}
        Job field: '{job_field}'.
        Conversation history:
        {history_text}
        Generate a follow-up question to clarify '{variable}' and elicit a precise numerical value.
        Output only the question.
        """
        
        response = rate_limited_model.generate_content(prompt)
        meta_agent.increment()
        followup = response.text.strip()
        
        # Track this reasoning with medium confidence
        meta_agent.track_reasoning(f"{variable}_followup", followup, 0.7)
        return followup
        
    except Exception as e:
        print(f"❌ Error in generate_followup_question for {variable}: {str(e)}")
        # Fallback question in case of error
        fallback = f"Could you please provide a more specific number for {variable}?"
        meta_agent.track_reasoning(f"{variable}_followup", fallback, 0.5)
        return fallback

# Node 4: Dynamic question management
def dynamic_question_manager(state: State) -> dict:
    user_data = []
    schemas = {var["key"]: var for var in standard_variables}
    variables_to_ask = {q["variable"]: q["question"] for q in state["questions"]}
    asked_variables = set()

    while variables_to_ask:
        # Check for meta-reasoning insights after processing some variables
        if len(asked_variables) > 0 and len(asked_variables) % 3 == 0:
            analysis = meta_agent.analyze_reasoning_process()
            if analysis:
                print(f"\n🔍 Meta-analysis: {analysis['message']}")
                if analysis['status'] == 'inefficient':
                    print(f"Recommendation: {analysis['recommendation']}")
                    if input("Continue with detailed questioning? (y/n): ").lower() != 'y':
                        # Use simplified approach for remaining variables
                        for var, question in list(variables_to_ask.items()):
                            if var not in schemas:
                                schemas[var] = {"type": "number", "min": 1, "max": 5}
                            schema = schemas[var]
                            default_value = (schema['min'] + schema['max']) / 2
                            user_data.append({
                                "question_type": var, 
                                "question": question, 
                                "response": default_value, 
                                "data_type": "number"
                            })
                        break
        
        try:
            variable = next(iter(variables_to_ask))
            initial_question = variables_to_ask.pop(variable)
            if variable not in schemas:
                schemas[variable] = {"type": "number", "min": 1, "max": 5}
            schema = schemas[variable]

            responses = []
            confidence = 0
            max_followups = 2
            followup_count = 0
            current_question = initial_question

            while confidence < 0.7 and followup_count <= max_followups:
                print(f"\n{current_question}")
                response = input("Your answer: ")
                
                # User input resets the consecutive call counter
                meta_agent.reset_consecutive()
                
                responses.append(response)
                state["conversation_history"].append({"variable": variable, "question": current_question, "response": response})
                
                number, confidence = interpret_response(variable, initial_question, responses, schema, state['job_field'], state["conversation_history"])
                
                # Check if meta-agent suggests we should stop follow-ups for this variable
                should_continue = meta_agent.check_and_handle(variable, confidence)
                
                if confidence < 0.7 and followup_count < max_followups and should_continue:
                    current_question = generate_followup_question(variable, responses, state['job_field'], state["conversation_history"])
                    followup_count += 1
                else:
                    # Break the loop if meta-agent suggests stopping or we have sufficient confidence
                    break

            user_data.append({"question_type": variable, "question": initial_question, "response": number, "data_type": "number"})
            asked_variables.add(variable)

            # Check if history can answer other variables, but limit batch size to avoid over-analysis
            vars_to_check = list(variables_to_ask.keys())[:3]  # Check at most 3 at a time
            for var in vars_to_check.copy():  # Use copy to avoid modification during iteration
                try:
                    if var not in schemas:
                        schemas[var] = {"type": "number", "min": 1, "max": 5}
                    
                    num, conf = interpret_response(var, variables_to_ask[var], [], schemas[var], state['job_field'], state["conversation_history"])
                    
                    if conf >= 0.7:
                        user_data.append({"question_type": var, "question": variables_to_ask[var], "response": num, "data_type": "number"})
                        variables_to_ask.pop(var)
                except Exception as e:
                    print(f"❌ Error checking variable {var}: {str(e)}")
                    # Continue with other variables
                    
        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
            # Handle error gracefully and continue with next variable if possible
            if variables_to_ask:
                continue
            else:
                break

    print(f"\n✅ Completed with {meta_agent.total_calls} total API calls")
    return {"user_data": user_data}

# Node 5: Store data
def data_storage(state: State) -> dict:
    try:
        client = MongoClient("localhost", 27017, serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        
        db = client["mydatabase"]
        collection = db["user_responses"]
        document = {
            "user_session": "some_session_id",
            "job_field": state["job_field"],
            "user_data": state["user_data"],
            "api_stats": {
                "total_calls": meta_agent.total_calls,
                "variables_analyzed": len(meta_agent.reasoning_history)
            }
        }
        collection.insert_one(document)
        print("✅ Data stored in MongoDB!")
        return {}
    except Exception as e:
        print(f"❌ MongoDB Error: {str(e)}")
        print("⚠️ Could not store data in MongoDB. Here's the data that would have been stored:")
        print(f"Job field: {state['job_field']}")
        print(f"User data: {state['user_data']}")
        print(f"Total API calls: {meta_agent.total_calls}")
        return {}

# Build workflow
def build_graph():
    graph = StateGraph(State)
    graph.add_node("pdf_parser", pdf_parser)
    graph.add_node("job_field_extractor", job_field_extractor)
    graph.add_node("question_generator", question_generator)
    graph.add_node("dynamic_question_manager", dynamic_question_manager)
    graph.add_node("data_storage", data_storage)

    graph.add_edge("pdf_parser", "job_field_extractor")
    graph.add_edge("job_field_extractor", "question_generator")
    graph.add_edge("question_generator", "dynamic_question_manager")
    graph.add_edge("dynamic_question_manager", "data_storage")
    graph.set_entry_point("pdf_parser")

    return graph.compile()

# Run the workflow with error handling
def run_workflow(resume_path):
    try:
        compiled_graph = build_graph()
        initial_state = {"pdf_path": resume_path}
        result = compiled_graph.invoke(initial_state)
        print("✅ Process completed successfully!")
        return result
    except Exception as e:
        print(f"❌ Critical error in workflow: {str(e)}")
        print("⚠️ Workflow execution failed. Please check the error message above.")
        return None

# Main execution
if __name__ == "__main__":
    resume_path = 'resume.pdf'  # Get path from user
    run_workflow(resume_path)