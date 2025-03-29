# main.py

from typing import TypedDict, List, Dict, Any, Callable, Optional, Union, Type
import google.generativeai as genai
import PyPDF2
from pymongo import MongoClient
from langgraph.graph import StateGraph
import time
import numpy as np
from difflib import SequenceMatcher
import random
import backoff
import json
import os
import uuid
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
from dotenv import load_dotenv
from ost import (MongoDBUtility, create_user_profile, DEFAULT_FIELD_PROFILES, DEFAULT_FIELD_PREFERENCES,
                add_new_field, SemanticOST, EnhancedPreferenceProcessor)

load_dotenv()

###########################################
# INFRASTRUCTURE LAYER
###########################################

# Get API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file")
    print("Please create a .env file with your API key: GEMINI_API_KEY=your_api_key_here")
    api_key = "YOUR_API_KEY_GOES_IN_ENV_FILE"

# Configure Gemini API
genai.configure(api_key=api_key)

class SimpleCache:
    """
    Generic caching implementation that can be used across different workflows.
    """
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
    
    def clear(self):
        """Clear the entire cache"""
        self.cache = {}
    
    def remove(self, key):
        """Remove a specific key from cache"""
        if key in self.cache:
            del self.cache[key]


class AIModelInterface:
    """
    Abstract base class defining the interface for AI model interactions.
    This enables easy swapping of AI providers or models.
    """
    def generate_content(self, prompt: str) -> Any:
        """Generate content from the AI model"""
        raise NotImplementedError("Subclasses must implement generate_content")


class RateLimitedAPI(AIModelInterface):
    """
    Rate-limited wrapper for any AI API that implements exponential backoff
    and caching to optimize API usage.
    """
    def __init__(self, model, cache=None, min_delay=1.0, max_delay=5.0):
        self.model = model
        self.cache = cache or SimpleCache(max_size=200)
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
        """
        Generate content from the AI model with rate limiting, caching and error handling.
        
        Args:
            prompt: The input prompt to send to the model
            
        Returns:
            The response from the model, or an error response
        """
        # First check if we have this in cache
        cache_key = hash(prompt)
        cached_result = self.cache.get(cache_key)
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
            self.cache.set(cache_key, response)
            
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


class StorageInterface:
    """
    Abstract base class for data storage operations.
    This allows for swapping storage backends (MongoDB, SQL, etc.)
    """
    def store_data(self, collection_name: str, data: Dict) -> bool:
        """Store data in the specified collection"""
        raise NotImplementedError("Subclasses must implement store_data")
        
    def retrieve_data(self, collection_name: str, query: Dict) -> List[Dict]:
        """Retrieve data from the specified collection matching the query"""
        raise NotImplementedError("Subclasses must implement retrieve_data")


class MongoDBStorage(StorageInterface):
    """
    MongoDB implementation of the storage interface.
    """
    def __init__(self, connection_string="mongodb://localhost:27017/", database_name="ost_db"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        
        # Initialize shared MongoDB utility from ost.py
        self.mongodb_util = MongoDBUtility()
        
    def _get_client(self):
        """Get or create MongoDB client"""
        if not self.client:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
        return self.client
        
    def store_data(self, collection_name: str, data: Dict) -> bool:
        """Store data in MongoDB"""
        try:
            client = self._get_client()
            db = client[self.database_name]
            collection = db[collection_name]
            collection.insert_one(data)
            return True
        except Exception as e:
            print(f"❌ MongoDB Error: {str(e)}")
            return False
            
    def retrieve_data(self, collection_name: str, query: Dict) -> List[Dict]:
        """Retrieve data from MongoDB"""
        try:
            client = self._get_client()
            db = client[self.database_name]
            collection = db[collection_name]
            return list(collection.find(query))
        except Exception as e:
            print(f"❌ MongoDB Error: {str(e)}")
            return []

###########################################
# OST INTEGRATION LAYER
###########################################

class OSTDataTransformer:
    """
    Utility class for transforming data between main.py and ost.py formats.
    """
    @staticmethod
    def transform_user_data_to_preferences(user_data):
        """
        Transform user data from main.py format to ost.py preferences format.
        
        Args:
            user_data: List of user data items from main.py
                [{"question_type": "variable_name", "response": value, ...}]
                
        Returns:
            Dictionary of preferences in ost.py format
                {"variable_name": value, ...}
        """
        preferences = {}
        
        for item in user_data:
            var_name = item["question_type"]
            var_value = item["response"]
            preferences[var_name] = var_value
            
        return preferences
    
    @staticmethod
    def map_job_field_to_ost_field(job_field):
        """
        Map job field from main.py to ost.py field name.
        
        Args:
            job_field: Job field name from main.py
            
        Returns:
            Field name compatible with ost.py
        """
        # Common mappings between main.py job fields and ost.py fields
        field_mappings = {
            "software engineer": "software_engineering",
            "software developer": "software_engineering",
            "software development": "software_engineering",
            "marketing": "marketing",
            "marketing specialist": "marketing",
            "marketing manager": "marketing",
            # Add more mappings as needed
        }
        
        # Normalize the job field (lowercase, remove extra spaces)
        normalized_field = job_field.lower().strip()
        
        # Try direct match
        if normalized_field in field_mappings:
            return field_mappings[normalized_field]
        
        # Try partial match
        for key, value in field_mappings.items():
            if key in normalized_field or normalized_field in key:
                return value
        
        # Default to software_engineering if no match is found
        print(f"⚠️ Unknown job field '{job_field}', defaulting to software_engineering")
        return "software_engineering"
    
    @staticmethod
    def create_ost_profile_from_main_data(job_field, user_data):
        """
        Create a full OST user profile and preferences from main.py data.
        
        Args:
            job_field: Job field extracted from resume
            user_data: User responses from main.py
            
        Returns:
            Tuple of (user_profile, user_preferences) for ost.py
        """
        # Map job field to OST field
        ost_field = OSTDataTransformer.map_job_field_to_ost_field(job_field)
        
        # Get default profile and preferences for this field
        user_profile, user_preferences = create_user_profile(ost_field)
        
        # Transform user data to preferences format
        main_preferences = OSTDataTransformer.transform_user_data_to_preferences(user_data)
        
        # Update OST preferences with values from main.py
        for key, value in main_preferences.items():
            user_preferences[key] = value
        
        # Ensure required fields exist
        expected_fields = [
            "min_salary", "compensation_weight", "career_growth_weight", 
            "work_life_balance_weight", "risk_tolerance", "job_search_urgency"
        ]
        
        for field in expected_fields:
            if field not in user_preferences:
                print(f"⚠️ Missing expected field '{field}' in user preferences, using default")
        
        # Add/update field indicator in profile and preferences
        user_profile["field"] = ost_field
        user_preferences["field"] = ost_field
        
        return user_profile, user_preferences
    
    @staticmethod
    def save_to_ost_collections(user_profile, user_preferences, user_id=None):
        """
        Save user profile and preferences to OST's MongoDB collections.
        
        Args:
            user_profile: User profile data
            user_preferences: User preferences data
            user_id: Optional user ID to associate with the data
            
        Returns:
            Dictionary with success status and user_id
        """
        try:
            # Create MongoDB utility
            mongodb_util = MongoDBUtility()
            
            # Generate a user ID if none provided
            if not user_id:
                user_id = str(uuid.uuid4())
            
            # Add user_id to profile and preferences
            user_profile["user_id"] = user_id
            user_preferences["user_id"] = user_id
            
            # Save profile and preferences
            field = user_profile.get("field", "unknown")
            mongodb_util.save_document("user_profiles", user_profile, 
                                      {"user_id": user_id, "field": field})
            mongodb_util.save_document("user_preferences", user_preferences, 
                                      {"user_id": user_id, "field": field})
            
            return {"success": True, "user_id": user_id}
        except Exception as e:
            print(f"❌ Error saving to OST collections: {str(e)}")
            return {"success": False, "user_id": user_id if user_id else None}

###########################################
# AGENT FRAMEWORK LAYER
###########################################

class BaseAgent:
    """
    Abstract base class for all agent types.
    Provides common functionality for agent operations.
    """
    def __init__(self, ai_model: AIModelInterface):
        self.ai_model = ai_model
        
    def process(self, input_data: Dict) -> Dict:
        """Process input data and return processed output"""
        raise NotImplementedError("Subclasses must implement process")


class ReasoningHistory:
    """
    A container class to track reasoning history for specific variables.
    Extracted from MetaReasoningAgent to be more reusable.
    """
    def __init__(self):
        self.history = {}  # variable -> list of reasoning steps
        self.last_responses = {}  # variable -> most recent response
        
    def add(self, variable: str, response: Any, confidence: float):
        """Add a reasoning step to the history"""
        if variable not in self.history:
            self.history[variable] = []
            
        self.history[variable].append({
            "response": response,
            "confidence": confidence,
            "time": time.time()
        })
        
        # Store latest response text for similarity comparison
        self.last_responses[variable] = str(response)
        
    def get_last_confidence(self, variable: str) -> float:
        """Get the confidence of the most recent reasoning step for a variable"""
        if variable in self.history and self.history[variable]:
            return self.history[variable][-1]["confidence"]
        return 0.0
        
    def get_last_response(self, variable: str) -> Any:
        """Get the most recent response for a variable"""
        if variable in self.history and self.history[variable]:
            return self.history[variable][-1]["response"]
        return None
        
    def get_similarity(self, variable: str, new_response: str) -> float:
        """Calculate similarity between new response and last response"""
        if variable in self.last_responses:
            return SequenceMatcher(None, self.last_responses[variable], new_response).ratio()
        return 0.0
        
    def get_confidence_progress(self, variable: str) -> float:
        """Calculate confidence improvement between last two steps"""
        if variable in self.history and len(self.history[variable]) >= 2:
            return self.history[variable][-1]["confidence"] - self.history[variable][-2]["confidence"]
        return 1.0  # Default to positive progress for first step


class MetaReasoningAgent(BaseAgent):
    """
    Monitors and analyzes the reasoning process to prevent inefficient reasoning loops
    and optimize the number of AI calls needed.
    """
    def __init__(self, 
                 ai_model: AIModelInterface, 
                 max_consecutive_calls=5, 
                 max_reasoning_per_variable=3,
                 similarity_threshold=0.8,
                 confidence_threshold=0.7,
                 progress_threshold=0.05):
        super().__init__(ai_model)
        self.consecutive_calls = 0
        self.max_consecutive_calls = max_consecutive_calls
        self.total_calls = 0
        self.reasoning_history = ReasoningHistory()
        self.max_reasoning_per_variable = max_reasoning_per_variable
        self.last_analysis_time = time.time()
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.progress_threshold = progress_threshold
    
    def increment(self):
        """Increment both consecutive and total call counters"""
        self.consecutive_calls += 1
        self.total_calls += 1
        print(f"API Call {self.total_calls} (Consecutive: {self.consecutive_calls}/{self.max_consecutive_calls})")
        
    def reset_consecutive(self):
        """Reset just the consecutive counter (after user input)"""
        self.consecutive_calls = 0
        print("Reset consecutive API call counter due to user interaction")
    
    def track_reasoning(self, variable: str, response: Any, confidence: float):
        """Track reasoning history for a specific variable"""
        # Calculate similarity with previous response if it exists
        current_response = str(response)
        similarity = self.reasoning_history.get_similarity(variable, current_response)
        if similarity > 0:
            print(f"Response similarity for {variable}: {similarity:.2f}")
            
        # Add to history
        self.reasoning_history.add(variable, response, confidence)
        
        return response, confidence
    
    def should_continue_reasoning(self, variable: str, confidence: float) -> tuple[bool, str]:
        """Analyze if more reasoning is productive for this variable"""
        # If we haven't reasoned about this variable much, continue
        if variable not in self.reasoning_history.history or len(self.reasoning_history.history[variable]) < 2:
            return True, "Initial reasoning"
        
        # If confidence is already high enough, no need for more reasoning
        if confidence >= self.confidence_threshold:
            return False, "Confidence threshold met"
        
        # Check if we've done too much reasoning on this variable
        if len(self.reasoning_history.history[variable]) >= self.max_reasoning_per_variable:
            return False, f"Maximum reasoning attempts ({self.max_reasoning_per_variable}) reached for {variable}"
        
        # Check for progress in confidence
        confidence_progress = self.reasoning_history.get_confidence_progress(variable)
        # If confidence is decreasing or not improving much, stop
        if confidence_progress <= self.progress_threshold:
            return False, f"Minimal confidence improvement: {confidence_progress:.2f}"
        
        # Check similarity with previous response (circular reasoning check)
        current_response = str(self.reasoning_history.get_last_response(variable))
        similarity = self.reasoning_history.get_similarity(variable, current_response)
        if similarity > self.similarity_threshold:  # High similarity threshold
            return False, f"Potential circular reasoning detected (similarity: {similarity:.2f})"
        
        return True, "Reasoning is still productive"
    
    def analyze_reasoning_process(self) -> Optional[Dict]:
        """Provides meta-analysis of the entire reasoning process"""
        # Only analyze after some time has passed to avoid too frequent analyses
        current_time = time.time()
        if current_time - self.last_analysis_time < 5:  # At least 5 seconds between analyses
            return None
        
        self.last_analysis_time = current_time
        
        if self.consecutive_calls >= 3:  # Analyze after 3+ consecutive calls
            total_variables = len(self.reasoning_history.history)
            variables_at_threshold = sum(1 for var in self.reasoning_history.history 
                                        if len(self.reasoning_history.history[var]) >= self.max_reasoning_per_variable)
            
            avg_confidence = np.mean([history[-1]["confidence"] 
                                     for var, history in self.reasoning_history.history.items() 
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
    
    def check_and_handle(self, variable: Optional[str] = None, confidence: Optional[float] = None) -> bool:
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
    
    def process(self, input_data: Dict) -> Dict:
        """
        Generic processing method that handles meta-reasoning.
        This can be used directly or extended by subclasses.
        """
        # Implementation depends on use case, this is a placeholder
        return input_data

class AutomatedMetaReasoningAgent(MetaReasoningAgent):
    """
    A version of MetaReasoningAgent that never prompts for user input.
    Always continues processing with sensible defaults.
    """
    
    def check_and_handle(self, variable: Optional[str] = None, confidence: Optional[float] = None) -> bool:
        """
        Automated version that never requests user input.
        Returns True to proceed with reasoning, False to skip based on automated rules.
        """
        # Track variable-specific reasoning if provided
        if variable is not None and confidence is not None:
            continue_reasoning, reason = self.should_continue_reasoning(variable, confidence)
            if not continue_reasoning:
                print(f"⚠️ Automatically stopping reasoning on '{variable}': {reason}")
                return False
        
        # Check overall consecutive call limit
        if self.consecutive_calls >= self.max_consecutive_calls:
            analysis = self.analyze_reasoning_process()
            if analysis:
                print(f"\n🔍 Meta-analysis: {analysis['message']}")
                print(f"Recommendation: {analysis['recommendation']}")
            
            print(f"\n⚠️ Rate limit reached: {self.consecutive_calls} consecutive AI reasoning steps.")
            print(f"Total API calls so far: {self.total_calls}")
            print("Automatically continuing reasoning (no user input required)")
            
            # Always reset consecutive counter and continue
            self.reset_consecutive()
        
        # Perform occasional reasoning analysis
        elif self.consecutive_calls >= 3:
            analysis = self.analyze_reasoning_process()
            if analysis and analysis['status'] == 'inefficient':
                print(f"\n🔍 Meta-analysis: {analysis['message']}")
                print(f"Recommendation: {analysis['recommendation']}")
                
                # Make automated decision based on confidence
                # If confidence is very low, we might want to stop
                should_continue = True
                if 'avg_confidence' in analysis and analysis.get('avg_confidence', 0.5) < 0.3:
                    print("Automatically stopping further reasoning due to very low confidence")
                    should_continue = False
                else:
                    print("Automatically continuing with reasoning")
                
                return should_continue
        
        return True
    
    def analyze_reasoning_process(self) -> Optional[Dict]:
        """
        Overridden to include average confidence in the analysis results
        to support automated decision making.
        """
        # Only analyze after some time has passed to avoid too frequent analyses
        current_time = time.time()
        if current_time - self.last_analysis_time < 5:  # At least 5 seconds between analyses
            return None
        
        self.last_analysis_time = current_time
        
        if self.consecutive_calls >= 3:  # Analyze after 3+ consecutive calls
            total_variables = len(self.reasoning_history.history)
            variables_at_threshold = sum(1 for var in self.reasoning_history.history 
                                        if len(self.reasoning_history.history[var]) >= self.max_reasoning_per_variable)
            
            confidences = [history[-1]["confidence"] 
                           for var, history in self.reasoning_history.history.items() 
                           if history]
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # If we're analyzing too much with little progress
            if variables_at_threshold > 0 or avg_confidence < 0.5:
                return {
                    "status": "inefficient",
                    "message": f"Reasoning appears inefficient. Analyzed {total_variables} variables with avg confidence {avg_confidence:.2f}.",
                    "recommendation": "Consider simplifying reasoning approach.",
                    "avg_confidence": avg_confidence
                }
            
            # If we're making reasonable progress
            if avg_confidence >= 0.6:
                return {
                    "status": "productive",
                    "message": f"Reasoning is productive with {total_variables} variables and avg confidence {avg_confidence:.2f}.",
                    "recommendation": "Continue current approach.",
                    "avg_confidence": avg_confidence
                }
        
        return None
    
###########################################
# WORKFLOW FRAMEWORK LAYER
###########################################

class BaseWorkflowState(TypedDict):
    """
    Base class for workflow state definitions.
    All specific workflow states should extend this.
    """
    workflow_id: str
    meta_data: Dict[str, Any]
    conversation_history: List[Dict]  # {"variable": str, "question": str, "response": str}


class WorkflowRegistry:
    """
    Registry for workflows and their components.
    Enables the management of multiple workflows within the application.
    """
    def __init__(self):
        self.workflows = {}
        self.nodes = {}
        
    def register_workflow(self, workflow_id: str, workflow_graph: StateGraph):
        """Register a workflow"""
        self.workflows[workflow_id] = workflow_graph
        
    def register_node(self, node_id: str, node_function: Callable):
        """Register a node function"""
        self.nodes[node_id] = node_function
        
    def get_workflow(self, workflow_id: str) -> StateGraph:
        """Get a workflow by ID"""
        return self.workflows.get(workflow_id)
        
    def get_node(self, node_id: str) -> Callable:
        """Get a node function by ID"""
        return self.nodes.get(node_id)
        
    def list_workflows(self) -> List[str]:
        """List all registered workflows"""
        return list(self.workflows.keys())
        
    def list_nodes(self) -> List[str]:
        """List all registered nodes"""
        return list(self.nodes.keys())


class BaseWorkflowBuilder:
    """
    Base class for workflow builders.
    Provides common functionality for building workflows.
    """
    def __init__(self, 
                 registry: WorkflowRegistry,
                 meta_agent: Optional[MetaReasoningAgent] = None, 
                 storage: Optional[StorageInterface] = None):
        self.registry = registry
        self.meta_agent = meta_agent
        self.storage = storage
        
    def build_graph(self) -> StateGraph:
        """Build and return a workflow graph"""
        raise NotImplementedError("Subclasses must implement build_graph")
        
    def register_workflow(self, workflow_id: str):
        """Register the workflow with the registry"""
        workflow = self.build_graph()
        self.registry.register_workflow(workflow_id, workflow)
        return workflow


###########################################
# DOMAIN-SPECIFIC IMPLEMENTATIONS
###########################################

# Resume Workflow Components
class ResumeWorkflowState(BaseWorkflowState):
    """State structure for the resume workflow"""
    pdf_path: str
    resume_text: str
    job_field: str
    questions: List[dict]  # {"variable": str, "question": str}
    user_data: List[dict]  # {"question_type": str, "question": str, "response": float, "data_type": str}
    ost_profile: Dict
    ost_preferences: Dict


class VariableSchema:
    """Schema definition for variables used in workflows"""
    def __init__(self, key: str, type_name: str, min_val: float, max_val: float, description: str):
        self.key = key
        self.type = type_name
        self.min = min_val
        self.max = max_val
        self.description = description
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'VariableSchema':
        """Create a schema from a dictionary"""
        return cls(
            key=data["key"],
            type_name=data["type"],
            min_val=data["min"],
            max_val=data["max"],
            description=data["description"]
        )
        
    def to_dict(self) -> Dict:
        """Convert schema to dictionary"""
        return {
            "key": self.key,
            "type": self.type,
            "min": self.min,
            "max": self.max,
            "description": self.description
        }


class WorkflowConfig:
    """Configuration for workflows"""
    def __init__(self, 
                 workflow_id: str, 
                 variables: List[Dict],
                 collection_name: str = "workflow_data",
                 max_followups: int = 2,
                 confidence_threshold: float = 0.7):
        self.workflow_id = workflow_id
        self.variables = [VariableSchema.from_dict(var) for var in variables]
        self.collection_name = collection_name
        self.max_followups = max_followups
        self.confidence_threshold = confidence_threshold
        
    @classmethod
    def load_from_file(cls, filepath: str) -> 'WorkflowConfig':
        """Load configuration from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return cls(**config)
        except Exception as e:
            print(f"❌ Error loading config: {str(e)}")
            # Return default config
            return cls(
                workflow_id="default",
                variables=[{
                    "key": "default",
                    "type": "number",
                    "min": 1,
                    "max": 5,
                    "description": "Default variable"
                }]
            )
            
    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to a JSON file"""
        try:
            config = {
                "workflow_id": self.workflow_id,
                "variables": [var.to_dict() for var in self.variables],
                "collection_name": self.collection_name,
                "max_followups": self.max_followups,
                "confidence_threshold": self.confidence_threshold
            }
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving config: {str(e)}")
            return False
            
    def get_schema(self, variable_key: str) -> Optional[VariableSchema]:
        """Get schema for a specific variable"""
        for schema in self.variables:
            if schema.key == variable_key:
                return schema
        return None


class ResumeWorkflowBuilder(BaseWorkflowBuilder):
    """
    Builder for the resume workflow.
    """
    def __init__(self, 
                 registry: WorkflowRegistry,
                 meta_agent: MetaReasoningAgent,
                 storage: StorageInterface,
                 config: WorkflowConfig):
        super().__init__(registry, meta_agent, storage)
        self.config = config
        
    def build_graph(self) -> StateGraph:
        """Build the resume workflow graph"""
        
        # Define the node functions with proper closure to capture dependencies
        def pdf_parser(state: ResumeWorkflowState) -> dict:
            try:
                with open(state["pdf_path"], "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "".join(page.extract_text() for page in reader.pages)
                return {"resume_text": text, "conversation_history": []}
            except Exception as e:
                print(f"❌ Error parsing PDF: {str(e)}")
                return {"resume_text": "Failed to parse resume", "conversation_history": []}
        
        def job_field_extractor(state: ResumeWorkflowState) -> dict:
            # User input happens before this node, so reset consecutive counter
            self.meta_agent.reset_consecutive()
            
            prompt = f"Extract the primary job field in 1-2 words from the resume: {state['resume_text']}"
            
            try:
                response = self.meta_agent.ai_model.generate_content(prompt)
                self.meta_agent.increment()
                
                job_field = response.text.strip()
                self.meta_agent.track_reasoning("job_field", job_field, 0.9)  # Typically high confidence
                
                return {"job_field": job_field}
            except Exception as e:
                print(f"❌ Error in job_field_extractor: {str(e)}")
                # Fallback value
                fallback_field = "unknown"
                self.meta_agent.track_reasoning("job_field", fallback_field, 0.5)
                return {"job_field": fallback_field}
        
        def question_generator(state: ResumeWorkflowState) -> dict:
            # Check if we should continue reasoning or use simple output
            if not self.meta_agent.check_and_handle():
                # Return default questions if meta-agent suggests skipping
                default_questions = [
                    {"variable": "min_salary", "question": "What is your minimum acceptable salary?"},
                    {"variable": "work_life_balance_weight", "question": "How important is work-life balance to you (1-5)?"},
                    {"variable": "job_search_urgency", "question": "How urgently do you need to find a job (1-10)?"}
                ]
                self.meta_agent.track_reasoning("question_generation", default_questions, 0.5)
                return {"questions": default_questions}
            
            try:
                standard_keys = ", ".join([var.key for var in self.config.variables])
                variable_descriptions = "\n".join([f"- {var.key}: {var.description}" for var in self.config.variables])
                
                prompt = f"""
                Based on the resume, generate personalized questions for these standard variables:
                {variable_descriptions}
                
                Add 3-5 extra questions based on the resume, each asking for a 1-5 rating, with unique variable names not in the standard list ({standard_keys}).
                Ensure all questions are distinct and tailored to the user's background.
                Format:
                Variable: [variable_name]
                Question: [question_text]
                Resume: {state['resume_text']}
                """
                
                response = self.meta_agent.ai_model.generate_content(prompt)
                self.meta_agent.increment()
                
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
                self.meta_agent.track_reasoning("question_generation", questions, 0.8)
                return {"questions": questions}
                
            except Exception as e:
                print(f"❌ Error in question_generator: {str(e)}")
                # Fallback questions in case of error
                default_questions = [
                    {"variable": "min_salary", "question": "What is your minimum acceptable salary?"},
                    {"variable": "work_life_balance_weight", "question": "How important is work-life balance to you (1-5)?"},
                    {"variable": "job_search_urgency", "question": "How urgently do you need to find a job (1-10)?"}
                ]
                self.meta_agent.track_reasoning("question_generation", default_questions, 0.5)
                return {"questions": default_questions}
        
        def dynamic_question_manager(state: ResumeWorkflowState) -> dict:
            user_data = []
            schemas = {var.key: var.to_dict() for var in self.config.variables}
            variables_to_ask = {q["variable"]: q["question"] for q in state["questions"]}
            asked_variables = set()

            while variables_to_ask:
                # Check for meta-reasoning insights after processing some variables
                if len(asked_variables) > 0 and len(asked_variables) % 3 == 0:
                    analysis = self.meta_agent.analyze_reasoning_process()
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
                    followup_count = 0
                    current_question = initial_question

                    while confidence < self.config.confidence_threshold and followup_count <= self.config.max_followups:
                        print(f"\n{current_question}")
                        response = input("Your answer: ")
                        
                        # User input resets the consecutive call counter
                        self.meta_agent.reset_consecutive()
                        
                        responses.append(response)
                        state["conversation_history"].append({"variable": variable, "question": current_question, "response": response})
                        
                        number, confidence = self._interpret_response(
                            variable, initial_question, responses, schema, 
                            state['job_field'], state["conversation_history"]
                        )
                        
                        # Check if meta-agent suggests we should stop follow-ups for this variable
                        should_continue = self.meta_agent.check_and_handle(variable, confidence)
                        
                        if confidence < self.config.confidence_threshold and followup_count < self.config.max_followups and should_continue:
                            current_question = self._generate_followup_question(
                                variable, responses, state['job_field'], state["conversation_history"]
                            )
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
                            
                            num, conf = self._interpret_response(
                                var, variables_to_ask[var], [], schemas[var], 
                                state['job_field'], state["conversation_history"]
                            )
                            
                            if conf >= self.config.confidence_threshold:
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

            print(f"\n✅ Completed with {self.meta_agent.total_calls} total API calls")
            return {"user_data": user_data}
        
        def ost_integration(state: ResumeWorkflowState) -> dict:
            """
            Transform and save data in the format that OST requires.
            Create the profile and preferences for OST.
            """
            try:
                # Create OST profile and preferences from collected data
                user_profile, user_preferences = OSTDataTransformer.create_ost_profile_from_main_data(
                    state["job_field"], state["user_data"]
                )
                
                # Save to OST collections and get user_id
                result = OSTDataTransformer.save_to_ost_collections(user_profile, user_preferences)
                
                if result.get("success"):
                    print("✅ Data successfully transformed and saved for OST")
                    user_id = result.get("user_id")
                    if user_id:
                        # Update the profile and preferences with the saved user_id
                        user_profile["user_id"] = user_id
                        user_preferences["user_id"] = user_id
                else:
                    print("⚠️ Warning: Could not save data to OST collections")
                
                # Return the created profile and preferences for future use
                return {
                    "ost_profile": user_profile,
                    "ost_preferences": user_preferences
                }
                
            except Exception as e:
                print(f"❌ Error in OST integration: {str(e)}")
                # Return empty dictionaries as fallback
                return {
                    "ost_profile": {},
                    "ost_preferences": {}
                }
        
        def data_storage(state: ResumeWorkflowState) -> dict:
            try:
                # Extract user_id from OST profile if available
                user_id = state.get("ost_profile", {}).get("user_id")
                
                document = {
                    "workflow_id": self.config.workflow_id,
                    "job_field": state["job_field"],
                    "user_data": state["user_data"],
                    "ost_profile": state.get("ost_profile", {}),
                    "ost_preferences": state.get("ost_preferences", {}),
                    "user_id": user_id,
                    "api_stats": {
                        "total_calls": self.meta_agent.total_calls,
                        "variables_analyzed": len(self.meta_agent.reasoning_history.history)
                    }
                }
                
                # Store data using the storage interface
                success = self.storage.store_data(
                    self.config.collection_name, 
                    document
                )
                
                if success:
                    print("✅ Data stored successfully!")
                    if user_id:
                        print(f"👤 User ID: {user_id}")
                        print("Use this ID in the application process to load your specific preferences.")
                else:
                    print("⚠️ Failed to store data")
                    
                return {"user_id": user_id} if user_id else {}
            except Exception as e:
                print(f"❌ Storage Error: {str(e)}")
                print("⚠️ Could not store data. Here's the data that would have been stored:")
                print(f"Job field: {state['job_field']}")
                print(f"User data: {state['user_data']}")
                print(f"Total API calls: {self.meta_agent.total_calls}")
                return {}
                
        # Register nodes with the registry
        self.registry.register_node("pdf_parser", pdf_parser)
        self.registry.register_node("job_field_extractor", job_field_extractor)
        self.registry.register_node("question_generator", question_generator)
        self.registry.register_node("dynamic_question_manager", dynamic_question_manager)
        self.registry.register_node("ost_integration", ost_integration)
        self.registry.register_node("data_storage", data_storage)
        
        # Build the graph
        graph = StateGraph(ResumeWorkflowState)
        graph.add_node("pdf_parser", pdf_parser)
        graph.add_node("job_field_extractor", job_field_extractor)
        graph.add_node("question_generator", question_generator)
        graph.add_node("dynamic_question_manager", dynamic_question_manager)
        graph.add_node("ost_integration", ost_integration)
        graph.add_node("data_storage", data_storage)

        graph.add_edge("pdf_parser", "job_field_extractor")
        graph.add_edge("job_field_extractor", "question_generator")
        graph.add_edge("question_generator", "dynamic_question_manager")
        graph.add_edge("dynamic_question_manager", "ost_integration")
        graph.add_edge("ost_integration", "data_storage")
        graph.set_entry_point("pdf_parser")

        return graph.compile()
    
    def _interpret_response(self, variable, question, responses, schema, job_field, conversation_history):
        """Helper method to interpret user responses"""
        # Get the last confidence for this variable if available
        last_confidence = self.meta_agent.reasoning_history.get_last_confidence(variable)
        
        # Check if we should continue reasoning based on previous attempts
        if not self.meta_agent.check_and_handle(variable, last_confidence):
            # Return best estimate so far or midpoint if no history
            if variable in self.meta_agent.reasoning_history.history:
                best_estimate = self.meta_agent.reasoning_history.get_last_response(variable)
                confidence = last_confidence
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
            ai_response = self.meta_agent.ai_model.generate_content(prompt)
            self.meta_agent.increment()
            
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
                self.meta_agent.track_reasoning(variable, result, confidence)
                return result, confidence
                
        except Exception as e:
            print(f"❌ Error in interpret_response for {variable}: {str(e)}")
        
        # Fallback for errors or parsing failures
        print(f"⚠️ AI failed for '{variable}', using midpoint and low confidence")
        result = (schema['min'] + schema['max']) / 2
        self.meta_agent.track_reasoning(variable, result, 0.5)
        return result, 0.5
    
    def _generate_followup_question(self, variable, responses, job_field, conversation_history):
        """Helper method to generate follow-up questions"""
        # Check if we should continue reasoning
        if not self.meta_agent.check_and_handle():
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
            
            response = self.meta_agent.ai_model.generate_content(prompt)
            self.meta_agent.increment()
            followup = response.text.strip()
            
            # Track this reasoning with medium confidence
            self.meta_agent.track_reasoning(f"{variable}_followup", followup, 0.7)
            return followup
            
        except Exception as e:
            print(f"❌ Error in generate_followup_question for {variable}: {str(e)}")
            # Fallback question in case of error
            fallback = f"Could you please provide a more specific number for {variable}?"
            self.meta_agent.track_reasoning(f"{variable}_followup", fallback, 0.5)
            return fallback


class OSTWorkflowState(BaseWorkflowState):
    """State structure for the OST job evaluation workflow"""
    field: str
    experience_level: str
    job_offers: List[Dict]
    evaluation_results: List[Dict]


class OSTWorkflowBuilder(BaseWorkflowBuilder):
    """
    Builder for the OST job evaluation workflow.
    """
    def __init__(self, 
                 registry: WorkflowRegistry,
                 meta_agent: Optional[MetaReasoningAgent] = None,
                 storage: Optional[StorageInterface] = None,
                 max_offers: int = 10):
        super().__init__(registry, meta_agent, storage)
        self.max_offers = max_offers
        
    def build_graph(self) -> StateGraph:
        """Build the OST workflow graph"""
        
        def load_user_data(state: OSTWorkflowState) -> dict:
            """Load user profile and preferences from OST system"""
            try:
                # Use user_id to fetch specific user data if provided
                user_id = state.get("user_id")
                mongodb_util = MongoDBUtility()
                
                if user_id:
                    print(f"🔍 Fetching user data for user_id: {user_id}")
                    # Try to load user profile and preferences by user_id
                    user_profile = mongodb_util.load_data("user_profiles", {"user_id": user_id})
                    user_preferences = mongodb_util.load_data("user_preferences", {"user_id": user_id})
                    
                    # If we found both user profile and preferences, use them
                    if user_profile and user_preferences:
                        print(f"✅ Found user data for user_id: {user_id}")
                        return {
                            "user_profile": user_profile,
                            "user_preferences": user_preferences
                        }
                    else:
                        print(f"⚠️ User data not found for user_id: {user_id}, using defaults")
                
                # Fall back to creating profile by field if user_id not provided or data not found
                user_profile, user_preferences = create_user_profile(state["field"])
                
                return {
                    "user_profile": user_profile,
                    "user_preferences": user_preferences
                }
            except Exception as e:
                print(f"❌ Error loading user data: {str(e)}")
                return {
                    "user_profile": DEFAULT_FIELD_PROFILES.get(state["field"], DEFAULT_FIELD_PROFILES["software_engineering"]),
                    "user_preferences": DEFAULT_FIELD_PREFERENCES.get(state["field"], DEFAULT_FIELD_PREFERENCES["software_engineering"])
                }
        
        def initialize_ost(state: OSTWorkflowState) -> dict:
            """Initialize OST algorithm with user profile and preferences"""
            try:
                # Create preference processor and OST algorithm
                preference_processor = EnhancedPreferenceProcessor(
                    state["user_profile"], 
                    state["user_preferences"]
                )
                
                ost_algorithm = SemanticOST(
                    state["user_profile"],
                    state["user_preferences"],
                    offer_arrival_rate=0.5,  # Adjust as needed
                    time_horizon=30,  # 30 day horizon
                    preference_processor=preference_processor
                )
                
                return {
                    "ost_algorithm": ost_algorithm,
                    "preference_processor": preference_processor
                }
            except Exception as e:
                print(f"❌ Error initializing OST: {str(e)}")
                return {}
        
        def evaluate_job_offers(state: OSTWorkflowState) -> dict:
            """Evaluate job offers using the OST algorithm"""
            try:
                evaluation_results = []
                
                for offer in state["job_offers"]:
                    # Calculate offer utility
                    utility = state["preference_processor"].calculate_offer_utility(offer)
                    
                    # Get threshold at current time
                    current_time = state.get("current_time", 0)
                    threshold = state["ost_algorithm"].get_threshold(current_time)
                    
                    # Determine if offer should be accepted
                    should_accept = utility >= threshold
                    
                    # Add evaluation result
                    evaluation_results.append({
                        "offer": offer,
                        "utility": utility,
                        "threshold": threshold,
                        "accept_recommendation": should_accept,
                        "evaluation_time": current_time
                    })
                
                return {"evaluation_results": evaluation_results}
            except Exception as e:
                print(f"❌ Error evaluating job offers: {str(e)}")
                return {"evaluation_results": []}
        
        def save_evaluation_results(state: OSTWorkflowState) -> dict:
            """Save evaluation results to storage"""
            try:
                document = {
                    "workflow_id": state["workflow_id"],
                    "field": state["field"],
                    "evaluation_results": state["evaluation_results"],
                    "timestamp": time.time()
                }
                
                if self.storage:
                    success = self.storage.store_data("ost_evaluations", document)
                    if success:
                        print("✅ Evaluation results stored successfully!")
                    else:
                        print("⚠️ Failed to store evaluation results")
                
                return {}
            except Exception as e:
                print(f"❌ Error saving evaluation results: {str(e)}")
                return {}
        
        # Register nodes with the registry
        self.registry.register_node("load_user_data", load_user_data)
        self.registry.register_node("initialize_ost", initialize_ost)
        self.registry.register_node("evaluate_job_offers", evaluate_job_offers)
        self.registry.register_node("save_evaluation_results", save_evaluation_results)
        
        # Build the graph
        graph = StateGraph(OSTWorkflowState)
        graph.add_node("load_user_data", load_user_data)
        graph.add_node("initialize_ost", initialize_ost)
        graph.add_node("evaluate_job_offers", evaluate_job_offers)
        graph.add_node("save_evaluation_results", save_evaluation_results)
        
        graph.add_edge("load_user_data", "initialize_ost")
        graph.add_edge("initialize_ost", "evaluate_job_offers")
        graph.add_edge("evaluate_job_offers", "save_evaluation_results")
        graph.set_entry_point("load_user_data")
        
        return graph.compile()


###########################################
# WORKFLOW EXECUTOR
###########################################

class WorkflowExecutor:
    """
    Executes workflows and handles errors.
    """
    def __init__(self, registry: WorkflowRegistry):
        self.registry = registry
        
    def run_workflow(self, workflow_id: str, initial_state: Dict) -> Dict:
        """
        Run a workflow with the given initial state.
        
        Args:
            workflow_id: ID of the workflow to run
            initial_state: Initial state for the workflow
            
        Returns:
            The final state of the workflow
        """
        try:
            workflow = self.registry.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow '{workflow_id}' not found")
                
            print(f"🚀 Starting workflow: {workflow_id}")
            result = workflow.invoke(initial_state)
            print(f"✅ Workflow {workflow_id} completed successfully!")
            return result
        except Exception as e:
            print(f"❌ Critical error in workflow {workflow_id}: {str(e)}")
            print("⚠️ Workflow execution failed.")
            return None


###########################################
# APPLICATION SETUP AND USAGE EXAMPLE
###########################################

def setup_resume_workflow():
    """
    Set up and configure the resume workflow.
    """
    # 1. Define standard variables for resume workflow
    standard_variables = [
        {"key": "min_salary", "type": "number", "min": 0, "max": 10000000, "description": "Minimum acceptable annual salary"},
        {"key": "compensation_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of compensation (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "career_growth_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of career growth (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "work_life_balance_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of work-life balance (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "company_reputation_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of company reputation (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "remote_work_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of remote work options (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "tech_stack_alignment_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of tech stack alignment (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "team_collaboration_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of team collaboration (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "role_responsibilities_weight", "type": "number", "min": 1, "max": 5, "description": "Importance of role responsibilities (1-5, 1=unimportant, 5=extremely important)"},
        {"key": "risk_tolerance", "type": "number", "min": 1, "max": 10, "description": "Willingness to wait for better offers (1-10, 1=low, 10=high)"},
        {"key": "financial_runway", "type": "number", "min": 0, "max": 120, "description": "Months you can afford to search for a job"},
        {"key": "job_search_urgency", "type": "number", "min": 1, "max": 10, "description": "Urgency to find a new job (1-10, 1=not urgent, 10=very urgent)"},
        {"key": "current_salary", "type": "number", "min": 0, "max": 10000000, "description": "Current or last annual salary"}
    ]

    # 2. Create workflow configuration
    config = WorkflowConfig(
        workflow_id="resume_workflow",
        variables=standard_variables,
        collection_name="resume_responses",
        max_followups=2,
        confidence_threshold=0.7
    )

    # 3. Initialize components
    registry = WorkflowRegistry()
    
    # Configure AI model (replace with your API key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    ai_model = RateLimitedAPI(model, min_delay=0.5, max_delay=10.0)
    
    meta_agent = AutomatedMetaReasoningAgent(
        ai_model=ai_model,
        max_consecutive_calls=5,
        max_reasoning_per_variable=3
    )
    
    storage = MongoDBStorage()
    
    # 4. Build and register workflow
    builder = ResumeWorkflowBuilder(registry, meta_agent, storage, config)
    builder.register_workflow("resume_workflow")
    
    return registry, config


def setup_ost_workflow():
    """
    Set up and configure the OST job evaluation workflow.
    """
    # 1. Initialize components
    registry = WorkflowRegistry()
    storage = MongoDBStorage()
    
    # 2. Build and register workflow
    builder = OSTWorkflowBuilder(registry, storage=storage, max_offers=10)
    builder.register_workflow("ost_workflow")
    
    return registry


def main():
    """
    Main application entry point.
    """
    print("📋 Setting up resume workflow...")
    registry, config = setup_resume_workflow()
    
    # Create workflow executor
    executor = WorkflowExecutor(registry)
    
    # Run the workflow
    resume_path = "resume.pdf"  # Path to the resume PDF file
    if not os.path.exists(resume_path):
        print(f"❌ File not found: {resume_path}")
        return
    
    # Run the workflow
    initial_state = {
        "workflow_id": config.workflow_id,
        "pdf_path": resume_path,
        "meta_data": {"source": "command_line"}
    }
    
    result = executor.run_workflow("resume_workflow", initial_state)
    if result:
        print("✅ Workflow completed successfully!")
        print(f"Job field: {result.get('job_field', 'unknown')}")
        print(f"User data collected: {len(result.get('user_data', []))} variables")
        
        # Show OST profile created
        if 'ost_profile' in result and result['ost_profile']:
            print("\n📊 OST Profile created:")
            print(f"Field: {result['ost_profile'].get('field', 'unknown')}")
            print(f"Experience level: {result['ost_profile'].get('experience_level', 'unknown')}")
            
            # Display user_id
            user_id = result['ost_profile'].get('user_id')
            if user_id:
                print(f"\n🔑 Your User ID: {user_id}")
                print("Save this ID to use in application_process.py for loading your preferences")
        else:
            print("\n⚠️ No OST Profile was created")


def run_ost_evaluation(field, job_offers, user_id=None):
    """
    Run OST job evaluation standalone.
    
    Args:
        field: The professional field
        job_offers: List of job offers to evaluate
        user_id: Optional user ID to fetch specific user preferences
    """
    print(f"📊 Running OST evaluation for field: {field}")
    if user_id:
        print(f"🔑 Using user ID: {user_id}")
    
    # Setup OST workflow
    registry = setup_ost_workflow()
    executor = WorkflowExecutor(registry)
    
    # Run the workflow
    initial_state = {
        "workflow_id": "ost_workflow",
        "field": field,
        "job_offers": job_offers,
        "current_time": 0,  # Day 0 of job search
        "meta_data": {"source": "command_line"},
        "user_id": user_id
    }
    
    result = executor.run_workflow("ost_workflow", initial_state)
    if result:
        print("\n✅ OST evaluation completed successfully!")
        print(f"Evaluated {len(result.get('evaluation_results', []))} job offers")
        
        # Display results
        for i, eval_result in enumerate(result.get('evaluation_results', [])):
            print(f"\nOffer {i+1}:")
            print(f"Utility: {eval_result.get('utility', 0):.2f}")
            print(f"Threshold: {eval_result.get('threshold', 0):.2f}")
            print(f"Recommendation: {'Accept' if eval_result.get('accept_recommendation', False) else 'Reject'}")
    else:
        print("⚠️ OST evaluation failed")


if __name__ == "__main__":
    main()