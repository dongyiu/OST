import random
import uuid
import numpy as np
import datetime
import math
import os
import json
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Callable
from scipy.stats import beta, norm
import pymongo
from dotenv import load_dotenv

# ===========================================================================
# 0. MONGODB CONNECTION UTILITY
# ===========================================================================

class MongoDBUtility:
    """Utility class for interacting with MongoDB, including loading and saving data."""
    
    def __init__(self):
        """Initialize MongoDB connection."""
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection string from environment variable
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        
        try:
            # Initialize MongoDB client
            self.client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.connected = True
            
            # Select database
            self.db = self.client["ost_db"]
            print("Successfully connected to MongoDB")
        except pymongo.errors.ServerSelectionTimeoutError:
            print("Warning: Could not connect to MongoDB. Using default values.")
            self.connected = False
    
    def load_data(self, collection_name, filter_query=None, default_data=None):
        """
        Load data from MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            filter_query: Optional query to filter results
            default_data: Default data to return if query returns no results or to save if collection is empty
            
        Returns:
            The data from MongoDB or default_data if not connected
        """
        # If not connected to MongoDB, return default data
        if not hasattr(self, 'connected') or not self.connected:
            return default_data
            
        try:
            collection = self.db[collection_name]
            
            # Check if collection exists and has data
            if collection.count_documents({}) == 0 and default_data:
                if isinstance(default_data, list):
                    # Insert list of documents
                    collection.insert_many(default_data)
                elif isinstance(default_data, dict):
                    # Insert single document
                    collection.insert_one(default_data)
                
                print(f"Initialized {collection_name} collection with default data")
            
            # Special handling for known collections with multiple documents
            if collection_name in ["common_tradeoffs", "field_arrival_factors", "field_arrival_rate_factors", 
                                "seasonality_factors", "field_companies", "field_positions", "field_contexts",
                                "experience_contexts", "location_contexts"] and not filter_query:
                # For these collections, we need to merge multiple documents
                results = {}
                documents = list(collection.find({}, {"_id": 0}))
                
                # Merge all documents into a single dictionary
                for doc in documents:
                    # Merge document contents into results
                    results.update(doc)
                
                # If we found data, return it, otherwise use default
                if results:
                    # Special handling for field_contexts to ensure 'general' is always available
                    if collection_name == "field_contexts" and "general" not in results:
                        # Add a general field as fallback 
                        results["general"] = {"base_salary": {"median": 60000, "std_dev": 10000}}
                    return results
                    
                # Use defaults if nothing found
                return default_data
            
            # Standard query handling
            if filter_query:
                data = collection.find_one(filter_query)
                if data:
                    # Remove MongoDB _id field
                    if '_id' in data:
                        del data['_id']
            else:
                data = list(collection.find({}, {"_id": 0}))
                
                # If only one document and default_data is a dictionary, return that document
                if len(data) == 1 and isinstance(default_data, dict):
                    data = data[0]
            
            return data
        except Exception as e:
            print(f"Error accessing MongoDB collection {collection_name}: {str(e)}")
            return default_data
    
    def load_dict_with_defaults(self, collection_name, default_dict, id_field="name"):
        """
        Load a dictionary from MongoDB, saving defaults if they don't exist.
        
        Args:
            collection_name: Name of the collection
            default_dict: Default dictionary to use if collection is empty
            id_field: Field to use as the unique identifier
            
        Returns:
            Dictionary with data from MongoDB or default_dict if not connected
        """
        # If not connected to MongoDB, return default dictionary
        if not hasattr(self, 'connected') or not self.connected:
            return default_dict
            
        try:
            collection = self.db[collection_name]
            result_dict = {}
            
            # Check if collection exists and has data
            if collection.count_documents({}) == 0:
                # Convert dictionary to list of documents
                documents = []
                for key, value in default_dict.items():
                    # Convert objects to dictionaries if needed
                    if hasattr(value, '__dict__'):
                        doc = value.__dict__.copy()
                    else:
                        doc = value
                    
                    # Add dictionary key as id_field
                    if id_field not in doc:
                        doc[id_field] = key
                        
                    documents.append(doc)
                
                # Insert all documents
                if documents:
                    collection.insert_many(documents)
                
                print(f"Initialized {collection_name} collection with {len(documents)} documents")
                return default_dict
            
            # Query all documents
            documents = list(collection.find({}, {"_id": 0}))
            
            # Convert back to dictionary
            for doc in documents:
                key = doc.get(id_field)
                if key:
                    # Convert back to object if original was an object
                    if key in default_dict and hasattr(default_dict[key], '__dict__'):
                        obj = default_dict[key].__class__(**doc)
                        result_dict[key] = obj
                    else:
                        result_dict[key] = doc
            
            return result_dict
        except Exception as e:
            print(f"Error loading dictionary from MongoDB collection {collection_name}: {str(e)}")
            return default_dict
    
    def save_document(self, collection_name, document, filter_query=None):
        """
        Save a document to MongoDB, updating if it exists.
        
        Args:
            collection_name: Name of the collection
            document: The document to save
            filter_query: Query to find existing document to update
            
        Returns:
            Result of the save operation or None if not connected
        """
        # If not connected to MongoDB, return None
        if not hasattr(self, 'connected') or not self.connected:
            return None
            
        try:
            collection = self.db[collection_name]
            
            if filter_query:
                return collection.update_one(filter_query, {"$set": document}, upsert=True)
            else:
                return collection.insert_one(document)
        except Exception as e:
            print(f"Error saving document to MongoDB collection {collection_name}: {str(e)}")
            return None

# Initialize MongoDB utility
mongodb_util = MongoDBUtility()

# ===========================================================================
# 1. SEMANTIC VARIABLE FRAMEWORK
# ===========================================================================

class VariableType:
    """Defines the semantic meaning and processing rules for job variables."""
    
    def __init__(self, 
                 name: str,
                 var_type: str,
                 higher_is_better: bool,
                 context_dependent: bool,
                 can_be_hard_constraint: bool,
                 normalization_method: str,
                 description: str):
        """
        Initialize a variable type definition.
        
        Args:
            name: Variable identifier
            var_type: Category of variable (monetary, quality, time, etc.)
            higher_is_better: Whether higher values are preferred
            context_dependent: Whether interpretation depends on field/experience
            can_be_hard_constraint: Whether this can be a deal-breaker
            normalization_method: How to normalize this variable
            description: Human-readable explanation of the variable
        """
        self.name = name
        self.var_type = var_type
        self.higher_is_better = higher_is_better
        self.context_dependent = context_dependent
        self.can_be_hard_constraint = can_be_hard_constraint
        self.normalization_method = normalization_method
        self.description = description
        
    def __repr__(self):
        return f"VariableType({self.name}, {self.var_type}, {'↑' if self.higher_is_better else '↓'})"


# Define default variable types for initialization
DEFAULT_VARIABLE_TYPES = {
    # Monetary variables
    "base_salary": VariableType(
        name="base_salary",
        var_type="monetary",
        higher_is_better=True,
        context_dependent=True,  # Depends on field, experience, location
        can_be_hard_constraint=True,
        normalization_method="field_relative",
        description="Annual base salary before bonuses or benefits"
    ),
    "total_compensation": VariableType(
        name="total_compensation",
        var_type="monetary",
        higher_is_better=True,
        context_dependent=True,
        can_be_hard_constraint=True,
        normalization_method="field_relative",
        description="Total annual compensation including salary, bonuses, and benefits"
    ),
    
    # Quality variables
    "work_life_balance": VariableType(
        name="work_life_balance",
        var_type="quality",
        higher_is_better=True,
        context_dependent=False,  # Universal preference
        can_be_hard_constraint=True,
        normalization_method="absolute_scale",
        description="Balance between work demands and personal time (1-10)"
    ),
    "career_growth": VariableType(
        name="career_growth",
        var_type="quality",
        higher_is_better=True,
        context_dependent=True,  # Different at different career stages
        can_be_hard_constraint=True,
        normalization_method="absolute_scale",
        description="Opportunities for advancement and skill development (1-10)"
    ),
    "company_reputation": VariableType(
        name="company_reputation",
        var_type="quality",
        higher_is_better=True,
        context_dependent=True,  # Industry-specific
        can_be_hard_constraint=False,
        normalization_method="absolute_scale",
        description="Overall reputation of the company (1-10)"
    ),
    "team_collaboration": VariableType(
        name="team_collaboration",
        var_type="quality",
        higher_is_better=True,
        context_dependent=False,
        can_be_hard_constraint=True,
        normalization_method="absolute_scale",
        description="Quality of team collaboration and communication (1-10)"
    ),
    
    # Location/logistics variables
    "commute_time": VariableType(
        name="commute_time",
        var_type="time",
        higher_is_better=False,  # Lower is better
        context_dependent=True,  # Location-dependent
        can_be_hard_constraint=True,
        normalization_method="inverse_scale",
        description="Daily commute time in minutes (one-way)"
    ),
    "remote_work": VariableType(
        name="remote_work",
        var_type="boolean",
        higher_is_better=True,
        context_dependent=True,  # Different fields value differently
        can_be_hard_constraint=True,
        normalization_method="binary",
        description="Whether remote work is available (0=No, 10=Full remote)"
    ),
    
    # Role-specific variables
    "tech_stack_alignment": VariableType(
        name="tech_stack_alignment",
        var_type="alignment",
        higher_is_better=True,
        context_dependent=True,  # Field-specific
        can_be_hard_constraint=True,
        normalization_method="absolute_scale",
        description="Alignment with preferred technologies and tools (1-10)"
    ),
    "role_responsibilities": VariableType(
        name="role_responsibilities",
        var_type="alignment",
        higher_is_better=True,
        context_dependent=True,
        can_be_hard_constraint=True,
        normalization_method="absolute_scale",
        description="Alignment with preferred job responsibilities (1-10)"
    )
}

# Load variable types from MongoDB
VARIABLE_TYPES = mongodb_util.load_dict_with_defaults("variable_types", DEFAULT_VARIABLE_TYPES)

# ===========================================================================
# 2. CONTEXT SYSTEM
# ===========================================================================

class ContextSystem:
    """
    System for understanding how variables should be interpreted in different
    contexts (fields, experience levels, locations, etc.)
    """
    
    def __init__(self):
        """Initialize with context definitions."""
        # Set up MongoDB utility for accessing user-specific data
        self.mongodb_util = MongoDBUtility()
        # Define default field-specific context information
        default_field_contexts = {
            "software_engineering": {
                "base_salary": {"median": 120000, "std_dev": 30000},
                "remote_work": {"importance_factor": 1.5},
                "tech_stack_alignment": {"applicable": True, "importance_factor": 1.3},
                "variables": {
                    "tech_stack_alignment": True,
                    "team_collaboration": 1.2, 
                }
            },
            "data_science": {
                "base_salary": {"median": 110000, "std_dev": 25000},
                "remote_work": {"importance_factor": 1.3},
                "tech_stack_alignment": {"applicable": True, "importance_factor": 1.4},
            },
            "marketing": {
                "base_salary": {"median": 75000, "std_dev": 20000},
                "remote_work": {"importance_factor": 0.9},
                "tech_stack_alignment": {"applicable": False},
            },
            "finance": {
                "base_salary": {"median": 100000, "std_dev": 35000},
                "remote_work": {"importance_factor": 0.7},
                "tech_stack_alignment": {"applicable": False},
            },
            "general": {  # Default fallback
                "base_salary": {"median": 80000, "std_dev": 25000},
                "remote_work": {"importance_factor": 1.0},
                "tech_stack_alignment": {"applicable": False},
            }
        }
        
        # Define default experience level modifiers
        default_experience_contexts = {
            "entry": {
                "salary_modifier": 0.7,
                "career_growth_importance": 1.6,
                "company_reputation_importance": 1.3,
            },
            "mid": {
                "salary_modifier": 1.0,
                "career_growth_importance": 1.2,
                "company_reputation_importance": 1.0,
            },
            "senior": {
                "salary_modifier": 1.4,
                "career_growth_importance": 0.9,
                "company_reputation_importance": 0.8,
            },
            "executive": {
                "salary_modifier": 2.0,
                "career_growth_importance": 0.7,
                "company_reputation_importance": 1.5,
            }
        }
        
        # Define default location-based cost of living adjustments
        default_location_contexts = {
            "san_francisco": {"cost_modifier": 1.5},
            "new_york": {"cost_modifier": 1.4},
            "seattle": {"cost_modifier": 1.3},
            "austin": {"cost_modifier": 1.1},
            "remote": {"cost_modifier": 1.0},
            "other": {"cost_modifier": 1.0}
        }
        
        # Define default variable interaction matrix
        default_variable_interactions = {
            # Format: "influencing_var": {"affected_var": weight_modifier}
            "work_life_balance": {"base_salary": 0.15, "total_compensation": 0.12},
            "remote_work": {"base_salary": 0.12, "commute_time": 0.8},
            "career_growth": {"base_salary": 0.1},
            "company_reputation": {"base_salary": 0.08, "career_growth": 0.2},
            "team_collaboration": {"work_life_balance": 0.15},
            "tech_stack_alignment": {"career_growth": 0.2}
        }
        
        # Define default common "would accept lower X for better Y" preferences
        default_common_tradeoffs = {
            "software_engineering": [
                ("remote_work", "base_salary", 0.15),  # Would accept 15% less salary for remote work
                ("work_life_balance", "base_salary", 0.2),  # Would accept 20% less salary for better WLB
                ("tech_stack_alignment", "base_salary", 0.1)  # Would accept 10% less for better tech
            ],
            "marketing": [
                ("company_reputation", "base_salary", 0.18),  # Reputation matters more in marketing
                ("career_growth", "base_salary", 0.15)
            ],
            "general": [
                ("work_life_balance", "base_salary", 0.15),
                ("career_growth", "base_salary", 0.1)
            ]
        }
        
        # Load data from MongoDB, using defaults if they don't exist
        self.field_contexts = mongodb_util.load_data("field_contexts", default_data=default_field_contexts)
        self.experience_contexts = mongodb_util.load_data("experience_contexts", default_data=default_experience_contexts)
        self.location_contexts = mongodb_util.load_data("location_contexts", default_data=default_location_contexts)
        self.variable_interactions = mongodb_util.load_data("variable_interactions", default_data=default_variable_interactions)
        self.common_tradeoffs = mongodb_util.load_data("common_tradeoffs", default_data=default_common_tradeoffs)
    
    def get_context(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete context object based on user profile.
        
        Args:
            user_profile: User's profile containing field, experience, location etc.
            
        Returns:
            Dictionary with all relevant context information
        """
        context = {}
        
        # Check if user has a specific user_id
        user_id = user_profile.get("user_id")
        
        # Get field context (default to general if not found)
        field = user_profile.get("field", "general")
        
        # Check for user-specific field context if user_id exists
        if user_id:
            try:
                # Query MongoDB directly to find user-specific field context
                user_field_context = self.mongodb_util.db["field_contexts"].find_one({"user_id": user_id})
                if user_field_context and field in user_field_context:
                    # Use user-specific field context
                    context["field"] = user_field_context[field]
                    print(f"✅ Using user-specific field context for user_id: {user_id}")
                    # Continue with the rest of context setup
                else:
                    # Fall back to default field context
                    if "general" not in self.field_contexts:
                        # Make sure we have at least a general field context
                        self.field_contexts["general"] = {"base_salary": {"median": 60000, "std_dev": 10000}}
                        
                    if field not in self.field_contexts:
                        field = "general"
                        
                    context["field"] = self.field_contexts[field]
            except Exception as e:
                print(f"❌ Error retrieving user-specific field context: {str(e)}")
                # Fall back to default field context
                if "general" not in self.field_contexts:
                    self.field_contexts["general"] = {"base_salary": {"median": 60000, "std_dev": 10000}}
                
                if field not in self.field_contexts:
                    field = "general"
                    
                context["field"] = self.field_contexts[field]
        else:
            # Use default field context
            # Create default field context if needed
            if "general" not in self.field_contexts:
                # Make sure we have at least a general field context
                self.field_contexts["general"] = {"base_salary": {"median": 60000, "std_dev": 10000}}
                
            if field not in self.field_contexts:
                field = "general"
                
            context["field"] = self.field_contexts[field]
        
        # Make sure field context has base_salary
        if "base_salary" not in context["field"]:
            context["field"]["base_salary"] = {"median": 60000, "std_dev": 10000}
        
        # Get experience context
        experience = user_profile.get("experience_level", "mid")
        
        # Check for user-specific experience context if user_id exists
        if user_id:
            try:
                # Query MongoDB directly to find user-specific experience context
                user_exp_contexts = list(self.mongodb_util.db["experience_contexts"].find({"user_id": user_id}))
                for exp_context in user_exp_contexts:
                    if experience in exp_context:
                        # Use user-specific experience context
                        context["experience"] = exp_context[experience]
                        print(f"✅ Using user-specific experience context for user_id: {user_id}")
                        break
                else:
                    # If no user-specific context found, fall back to default
                    if "mid" not in self.experience_contexts:
                        self.experience_contexts["mid"] = {"salary_modifier": 1.0}
                        
                    if experience not in self.experience_contexts:
                        experience = "mid"
                        
                    context["experience"] = self.experience_contexts[experience]
            except Exception as e:
                print(f"❌ Error retrieving user-specific experience context: {str(e)}")
                # Fall back to default experience context
                if "mid" not in self.experience_contexts:
                    self.experience_contexts["mid"] = {"salary_modifier": 1.0}
                    
                if experience not in self.experience_contexts:
                    experience = "mid"
                    
                context["experience"] = self.experience_contexts[experience]
        else:
            # Use default experience context
            # Make sure mid experience exists
            if "mid" not in self.experience_contexts:
                self.experience_contexts["mid"] = {"salary_modifier": 1.0}
                
            if experience not in self.experience_contexts:
                experience = "mid"
                
            context["experience"] = self.experience_contexts[experience]
        
        # Make sure experience has salary_modifier
        if "salary_modifier" not in context["experience"]:
            context["experience"]["salary_modifier"] = 1.0
        
        # Get location context
        location = user_profile.get("location", "other")
        
        # Make sure other location exists
        if "other" not in self.location_contexts:
            self.location_contexts["other"] = {"cost_modifier": 1.0}
            
        if location not in self.location_contexts:
            location = "other"
            
        context["location"] = self.location_contexts[location]
        
        # Make sure location has cost_modifier
        if "cost_modifier" not in context["location"]:
            context["location"]["cost_modifier"] = 1.0
        
        # Combined parameters for convenience
        context["expected_salary"] = context["field"]["base_salary"]["median"] * \
                                  context["experience"]["salary_modifier"] * \
                                  context["location"]["cost_modifier"]
        
        # Make sure general tradeoffs exist
        if "general" not in self.common_tradeoffs:
            self.common_tradeoffs["general"] = [
                ["work_life_balance", "base_salary", 0.15],
                ["career_growth", "base_salary", 0.1]
            ]
            
        # Add field-specific tradeoffs
        context["tradeoffs"] = self.common_tradeoffs.get(field, self.common_tradeoffs["general"])
        
        return context
    
    def apply_variable_relationships(self, offer: Dict[str, Any], context: Dict[str, Any], 
                                   preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply dynamic variable relationship models to modify perceptions.
        
        Args:
            offer: The job offer data
            context: The context information
            preferences: User preferences
            
        Returns:
            Modified offer with adjusted perceptions
        """
        adjusted_offer = offer.copy()
        
        # Process all potential variable interactions dynamically
        for influencing_var, affected_vars in self.variable_interactions.items():
            # Skip if influencing variable is not in the offer
            if influencing_var not in offer:
                continue
                
            influence_value = offer[influencing_var]
            
            # Skip non-numeric values
            if not isinstance(influence_value, (int, float)):
                continue
                
            # Check if this variable is above average (normalized to 1-10 scale)
            # For boolean variables like remote_work, any value > 5 is considered positive
            is_positive_influence = influence_value > 5
            
            # Get user's weight for this variable if available
            var_weight = preferences.get(f"{influencing_var}_weight", 3) / 5  # Normalize to 0-1 scale
            
            # Apply influence to each affected variable
            for affected_var, base_modifier in affected_vars.items():
                # Skip if affected variable not in offer
                if affected_var not in offer:
                    continue
                    
                # Skip non-numeric values
                if not isinstance(offer[affected_var], (int, float)):
                    continue
                
                # Calculate influence magnitude based on:
                # 1. How much above/below average the influencing variable is
                # 2. The user's personal weight for the influencing variable
                # 3. The base modifier for this relationship
                deviation_from_avg = (influence_value - 5) / 5  # -1 to +1 scale
                influence_magnitude = deviation_from_avg * var_weight * base_modifier
                
                # Apply the influence (only positive influences for now)
                if is_positive_influence:
                    # For monetary variables (positive adjustment)
                    if affected_var in ["base_salary", "total_compensation"]:
                        adjustment_factor = 1 + influence_magnitude
                        adjusted_value = offer[affected_var] * adjustment_factor
                    # For quality variables on 1-10 scale (cap at 10)
                    else:
                        adjustment_amount = influence_magnitude * 10  # Scale to 1-10
                        adjusted_value = min(10, offer[affected_var] + adjustment_amount)
                    
                    # Store the adjusted value
                    adjusted_key = f"adjusted_{affected_var}"
                    adjusted_offer[adjusted_key] = adjusted_value
        
        # Apply field-specific tradeoffs from user's "would accept lower X for better Y"
        if "would_accept_lower_salary_for" in preferences:
            for tradeoff_var in preferences["would_accept_lower_salary_for"]:
                if tradeoff_var in offer and offer.get(tradeoff_var, 0) > 7:  # Good value in tradeoff var
                    # Find base modifier from context tradeoffs
                    for (var1, var2, modifier) in context.get("tradeoffs", []):
                        if var1 == tradeoff_var and var2 == "base_salary":
                            # Apply personal tradeoff adjustment to salary perception
                            if "adjusted_base_salary" not in adjusted_offer:
                                salary_perception = offer["base_salary"] * (1 + modifier)
                                adjusted_offer["adjusted_base_salary"] = salary_perception
                            break
        
        return adjusted_offer

# ===========================================================================
# 3. ENHANCED PREFERENCE PROCESSOR
# ===========================================================================

class EnhancedPreferenceProcessor:
    """
    Enhanced processor for user preferences with semantic understanding
    and non-linear utility functions.
    """
    
    def __init__(self, user_profile: Dict[str, Any], user_preferences: Dict[str, Any]):
        """
        Initialize preference processor with user profile and preferences.
        
        Args:
            user_profile: User information (field, experience, location, etc.)
            user_preferences: User's stated preferences and weights
        """
        self.user_profile = user_profile
        self.user_preferences = user_preferences
        
        # Set up MongoDB utility for accessing user-specific data
        self.mongodb_util = MongoDBUtility()
        
        # Set up context system
        self.context_system = ContextSystem()
        self.context = self.context_system.get_context(user_profile)
        
        # Extract hard constraints
        self.hard_constraints = self._extract_hard_constraints()
        
        # Extract variable weights (with defaults)
        self.variable_weights = self._extract_weights()
        
        # Set up normalization functions
        self.normalization_functions = {
            "field_relative": self._normalize_field_relative,
            "absolute_scale": self._normalize_absolute_scale,
            "inverse_scale": self._normalize_inverse_scale,
            "binary": self._normalize_binary
        }
    
    def _extract_hard_constraints(self) -> Dict[str, Any]:
        """Extract hard constraints from preferences."""
        constraints = {}
        
        # Minimum salary is a common hard constraint
        if "min_salary" in self.user_preferences:
            constraints["base_salary"] = self.user_preferences["min_salary"]
        
        # Extract other hard constraints marked with "min_" or "must_"
        for key, value in self.user_preferences.items():
            if key.startswith("min_") and key != "min_salary":
                var_name = key[4:]  # Remove "min_" prefix
                if var_name in VARIABLE_TYPES and VARIABLE_TYPES[var_name].can_be_hard_constraint:
                    constraints[var_name] = value
                    
            elif key.startswith("must_"):
                var_name = key[5:]  # Remove "must_" prefix
                if var_name in VARIABLE_TYPES and VARIABLE_TYPES[var_name].can_be_hard_constraint:
                    constraints[var_name] = True
        
        # Include deal-breakers
        if "deal_breakers" in self.user_profile:
            for item in self.user_profile["deal_breakers"]:
                constraints[item] = True
        
        return constraints
    
    def _extract_weights(self) -> Dict[str, float]:
        """Extract and normalize preference weights."""
        weights = {}
        
        # Extract explicit weights
        for key, value in self.user_preferences.items():
            if key.endswith("_weight"):
                var_name = key.replace("_weight", "")
                weights[var_name] = float(value)
        
        # Get user_id from user_profile or user_preferences
        user_id = self.user_profile.get('user_id') or self.user_preferences.get('user_id')
        
        # Adjust weights based on context
        for var_name, weight in list(weights.items()):
            # Adjust based on experience level
            if var_name == "career_growth":
                # First check for user-specific experience context
                experience_context = None
                
                # Look for user-specific experience context
                if user_id and "experience" in self.context:
                    for exp_context in self.mongodb_util.db["experience_contexts"].find({"user_id": user_id}):
                        if exp_context:
                            # Found user-specific context
                            experience_level = self.user_profile.get("experience_level", "mid")
                            if experience_level in exp_context:
                                experience_context = exp_context[experience_level]
                                break
                
                # Get career_growth_importance either from user-specific context or default context
                if experience_context and "career_growth_importance" in experience_context:
                    career_growth_importance = experience_context["career_growth_importance"]
                else:
                    # Fall back to default context
                    career_growth_importance = self.context.get("experience", {}).get("career_growth_importance", 1.0)
                
                weights[var_name] *= career_growth_importance
            
            # Adjust based on field
            if var_name in self.context["field"].get("variables", {}):
                # Check if there are user-specific field variables
                field_variables = self.context["field"]["variables"]
                
                # Look for user-specific field variables
                if user_id:
                    # Query MongoDB for user-specific field context
                    user_field_context = self.mongodb_util.db["field_contexts"].find_one(
                        {"user_id": user_id, self.user_profile.get("field", "general"): {"$exists": True}}
                    )
                    
                    if user_field_context and var_name in user_field_context.get(self.user_profile.get("field", "general"), {}).get("variables", {}):
                        # Use user-specific field factor
                        field_factor = user_field_context[self.user_profile.get("field", "general")]["variables"][var_name]
                        if isinstance(field_factor, (int, float)):
                            weights[var_name] *= field_factor
                            continue
                
                # Fall back to default field variables
                field_factor = field_variables.get(var_name)
                if isinstance(field_factor, (int, float)):
                    weights[var_name] *= field_factor
        
        return weights
    
    def check_hard_constraints(self, offer: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if an offer meets all hard constraints.
        
        Args:
            offer: The job offer data
            
        Returns:
            Tuple of (meets_constraints, reason)
        """
        for var_name, constraint_value in self.hard_constraints.items():
            if var_name not in offer:
                continue
                
            var_type = VARIABLE_TYPES.get(var_name)
            if not var_type:
                continue
                
            offer_value = offer[var_name]
                
            # For numeric constraints
            if isinstance(constraint_value, (int, float)) and isinstance(offer_value, (int, float)):
                # Check if var_type is a dictionary or an object
                if isinstance(var_type, dict):
                    # If it's a dictionary, access the higher_is_better key
                    higher_is_better = var_type.get('higher_is_better', True)
                else:
                    # If it's an object, access the higher_is_better attribute
                    higher_is_better = var_type.higher_is_better
                    
                if higher_is_better and offer_value < constraint_value:
                    return False, f"{var_name} ({offer_value}) below minimum requirement ({constraint_value})"
                elif not higher_is_better and offer_value > constraint_value:
                    return False, f"{var_name} ({offer_value}) above maximum acceptable ({constraint_value})"
            
            # For boolean constraints
            elif isinstance(constraint_value, bool) and constraint_value:
                if not offer_value:
                    return False, f"Required {var_name} not available"
        
        return True, "Meets all hard constraints"
    
    def _normalize_field_relative(self, var_name: str, value: float) -> float:
        """
        Normalize a value relative to field expectations.
        
        Args:
            var_name: Variable name
            value: Raw value
            
        Returns:
            Normalized value (0-1 scale)
        """
        if var_name == "base_salary" or var_name == "total_compensation":
            # Get expected salary from context
            expected_salary = self.context["expected_salary"]
            std_dev = self.context["field"]["base_salary"]["std_dev"] * \
                     self.context["experience"]["salary_modifier"]
            
            # Calculate z-score (how many standard deviations from the mean)
            z_score = (value - expected_salary) / std_dev
            
            # Convert to 0-1 scale using sigmoid function
            normalized = 1 / (1 + math.exp(-z_score))
            
            # Scale to ensure a reasonable range (0.2 for -2 std dev, 0.8 for +2 std dev)
            normalized = max(0.01, min(0.99, normalized))
            
            return normalized
        
        # Fallback for other field-relative variables
        return value / 10.0
    
    def _normalize_absolute_scale(self, var_name: str, value: float) -> float:
        """Normalize a value on an absolute 1-10 scale to 0-1."""
        # Simple linear normalization for 1-10 scale
        normalized = (value - 1) / 9.0
        return max(0, min(1, normalized))
    
    def _normalize_inverse_scale(self, var_name: str, value: float) -> float:
        """
        Normalize variables where lower is better (like commute time).
        
        Args:
            var_name: Variable name
            value: Raw value
            
        Returns:
            Normalized value (0-1 scale, higher is better)
        """
        if var_name == "commute_time":
            # Commute time: 0 min = 1.0, 60+ min = 0.0
            max_acceptable = 60.0
            normalized = max(0, 1.0 - (value / max_acceptable))
            return normalized
            
        # Generic inverse for other variables
        return 1.0 - (value / 10.0)
    
    def _normalize_binary(self, var_name: str, value: float) -> float:
        """Normalize binary variables."""
        # For boolean values converted to 0-10 scale
        return value / 10.0
    
    def normalize_variable(self, var_name: str, value: float) -> float:
        """
        Normalize a variable to 0-1 scale using appropriate method.
        
        Args:
            var_name: Variable name
            value: Raw value
            
        Returns:
            Normalized value (0-1 scale)
        """
        # Get variable type
        var_type = VARIABLE_TYPES.get(var_name)
        if not var_type:
            # Default normalization for unknown variables
            return value / 10.0
            
        # Get normalization method for this variable
        # Check if var_type is a dictionary or an object
        if isinstance(var_type, dict):
            # If it's a dictionary, access the normalization_method key
            method = var_type.get('normalization_method', 'absolute_scale')
        else:
            # If it's an object, access the normalization_method attribute
            method = var_type.normalization_method
            
        normalize_fn = self.normalization_functions.get(method, self._normalize_absolute_scale)
        
        # Apply normalization function
        return normalize_fn(var_name, value)
    
    def apply_non_linear_utility(self, var_name: str, normalized_value: float) -> float:
        """
        Apply non-linear utility function to a normalized value.
        
        Args:
            var_name: Variable name
            normalized_value: Normalized value (0-1 scale)
            
        Returns:
            Utility value with non-linear adjustments
        """
        # Get the weight for this variable
        weight = self.variable_weights.get(var_name, 1.0)
        
        # Apply different utility curves based on variable type and weight
        if weight > 3.5:  # High importance
            # Convex curve for high-importance variables (steep improvement at higher values)
            # This means small differences at the high end matter a lot
            return normalized_value ** 0.7 * weight
        elif weight < 2.5:  # Low importance
            # Concave curve for low-importance variables (diminishing returns)
            return normalized_value ** 1.3 * weight
        else:  # Medium importance
            # Linear utility for medium importance
            return normalized_value * weight
    
    def calculate_offer_utility(self, offer: Dict[str, Any]) -> float:
        """
        Calculate normalized weighted utility of an offer with semantic understanding.
        
        Args:
            offer: The job offer data
            
        Returns:
            Utility value (0-10 scale)
        """
        # First apply variable relationships to get adjusted perceptions
        adjusted_offer = self.context_system.apply_variable_relationships(
            offer, self.context, self.user_preferences
        )
        
        # Check hard constraints first
        meets_constraints, _ = self.check_hard_constraints(adjusted_offer)
        if not meets_constraints:
            return -10.0  # Strong penalty for violating hard constraints
        
        # Calculate utility components
        utility = 0.0
        total_weight = 0.0
        
        # Process each attribute with semantic understanding
        for var_name, var_type in VARIABLE_TYPES.items():
            # Skip if not in offer
            if var_name not in adjusted_offer:
                continue
                
            # Check for adjusted value first (from relationship model)
            if f"adjusted_{var_name}" in adjusted_offer:
                value = adjusted_offer[f"adjusted_{var_name}"]
            else:
                value = adjusted_offer[var_name]
                
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
                
            # Get weight for this variable
            weight = self.variable_weights.get(var_name, 1.0)
            
            # Normalize value
            normalized_value = self.normalize_variable(var_name, value)
            
            # Apply non-linear utility function
            component_utility = self.apply_non_linear_utility(var_name, normalized_value)
            
            # Add to total utility
            utility += component_utility
            total_weight += weight
        
        # Scale to 0-10 range if we have weights
        if total_weight > 0:
            scaled_utility = (utility / total_weight) * 10
        else:
            scaled_utility = utility
            
        return scaled_utility

# ===========================================================================
# 4. ENHANCED BAYESIAN BELIEF MODEL
# ===========================================================================

class EnhancedBayesianModel:
    """
    Enhanced Bayesian framework for modeling job market with semantic understanding
    and field-specific priors.
    """
    
    def __init__(self, user_profile: Dict[str, Any], user_preferences: Dict[str, Any]):
        """Initialize Bayesian belief model with semantic understanding."""
        self.user_profile = user_profile
        self.user_preferences = user_preferences
        
        # Set up context system
        self.context_system = ContextSystem()
        self.context = self.context_system.get_context(user_profile)
        
        # Initialize field-specific salary model
        expected_salary = self.context["expected_salary"]
        salary_std = self.context["field"]["base_salary"]["std_dev"] * \
                   self.context["experience"]["salary_modifier"]
        
        # Salary model: Normal distribution with conjugate Normal-Inverse-Gamma prior
        self.salary_model = {
            'mu': expected_salary,
            'kappa': 2.0,  # Prior strength
            'alpha': 3.0,  # Shape parameter
            'beta': 0.5 * salary_std,  # Scale parameter
            'observations': []
        }
        
        # Initialize attribute models
        self.attribute_models = {}
        
        # Set up priors for each attribute based on semantic understanding
        for var_name, var_type in VARIABLE_TYPES.items():
            if var_name != "base_salary" and var_name != "total_compensation":
                # Check if var_type is a VariableType object or a dictionary
                if isinstance(var_type, dict):
                    is_context_dependent = var_type.get('context_dependent', False)
                else:
                    is_context_dependent = var_type.context_dependent
                
                # Set field-specific priors when applicable
                if is_context_dependent and var_name in self.context["field"].get("variables", {}):
                    field_info = self.context["field"]["variables"][var_name]
                    # For boolean values
                    if isinstance(field_info, bool):
                        if field_info:
                            self.attribute_models[var_name] = {
                                'alpha': 4.0,  # More confident prior
                                'beta': 2.0,   # Skewed toward positive
                                'observations': []
                            }
                        else:
                            self.attribute_models[var_name] = {
                                'alpha': 2.0,
                                'beta': 4.0,   # Skewed toward negative
                                'observations': []
                            }
                    # For numeric modifiers
                    elif isinstance(field_info, (int, float)):
                        if field_info > 1.0:  # More important in this field
                            self.attribute_models[var_name] = {
                                'alpha': 3.0 * field_info,
                                'beta': 3.0,
                                'observations': []
                            }
                        else:
                            self.attribute_models[var_name] = {
                                'alpha': 3.0,
                                'beta': 3.0 / field_info if field_info > 0 else 6.0,
                                'observations': []
                            }
                else:
                    # Default uninformative prior
                    self.attribute_models[var_name] = {
                        'alpha': 3.0,
                        'beta': 3.0,
                        'observations': []
                    }
        
        # Market condition model with field-specific prior
        # MODIFICATION 1: Strengthen market condition
        self.market_model = {
            'alpha': 8.0,  # Changed from 5.0 to 8.0
            'beta': 3.0,   # Changed from 5.0 to 3.0
            'observations': []
        }
        
        # Track offers and rejections
        self.num_offers = 0
        self.num_rejections = 0
        
        # Define default seasonality model
        default_seasonality_factors = [
            1.05, 1.08, 1.10, 1.05,  # Q1
            0.98, 0.95, 0.92, 0.90,  # Q2
            0.88, 0.90, 0.93, 0.95,  # Q3
            1.02, 1.08, 1.12, 1.10,  # Q4
        ]
        
        # Load seasonality factors from MongoDB
        seasonality_data = mongodb_util.load_data("seasonality_factors", 
                                              default_data={"factors": default_seasonality_factors})
        self.seasonality_factors = seasonality_data.get("factors", default_seasonality_factors)
    
    def update_with_offer(self, offer: Dict[str, Any], was_accepted: bool = None):
        """
        Update Bayesian model with a new observed job offer.
        
        Args:
            offer: The job offer data
            was_accepted: Whether this offer was accepted by the user
        """
        # Update salary model
        if 'base_salary' in offer:
            # Normalize salary based on context before updating
            normalized_salary = offer['base_salary'] / self.context["location"]["cost_modifier"]
            self._update_salary_model(normalized_salary)
        
        # Update attribute models
        for attr_name, model in self.attribute_models.items():
            if attr_name in offer:
                # Get variable type
                var_type = VARIABLE_TYPES.get(attr_name)
                if not var_type:
                    continue
                    
                # Get value and normalize to 0-1 scale for Beta model
                value = offer[attr_name]
                if not isinstance(value, (int, float)):
                    continue
                    
                # Normalize based on variable type
                # Check if var_type is a dictionary or an object
                if isinstance(var_type, dict):
                    # If it's a dictionary, access the var_type key
                    var_type_value = var_type.get('var_type', '')
                    if var_type_value in ["quality", "alignment"]:
                        normalized_value = (value - 1) / 9.0  # 1-10 scale to 0-1
                    elif var_type_value == "boolean":
                        normalized_value = value / 10.0  # 0-10 scale to 0-1
                    else:
                        # Skip variables we don't know how to normalize
                        continue
                else:
                    # If it's an object, access the var_type attribute
                    if var_type.var_type == "quality" or var_type.var_type == "alignment":
                        normalized_value = (value - 1) / 9.0  # 1-10 scale to 0-1
                    elif var_type.var_type == "boolean":
                        normalized_value = value / 10.0  # 0-10 scale to 0-1
                    else:
                        # Skip variables we don't know how to normalize
                        continue
                
                # Update model
                self._update_attribute_model(attr_name, normalized_value)
        
        # Update offer count
        self.num_offers += 1
        
        # Learn from rejection if applicable
        if was_accepted is False:
            self.num_rejections += 1
            self._learn_from_rejection(offer)
    
    def _update_salary_model(self, observed_salary: float):
        """Update salary model with conjugate Normal-Inverse-Gamma update."""
        # Store raw observation
        self.salary_model['observations'].append(observed_salary)
        
        # Extract current parameters
        mu0 = self.salary_model['mu']
        kappa0 = self.salary_model['kappa']
        alpha0 = self.salary_model['alpha']
        beta0 = self.salary_model['beta']
        
        # Compute updated parameters (conjugate update)
        kappa1 = kappa0 + 1
        mu1 = (kappa0 * mu0 + observed_salary) / kappa1
        alpha1 = alpha0 + 0.5
        beta1 = beta0 + 0.5 * kappa0 / kappa1 * (observed_salary - mu0)**2
        
        # Store updated parameters
        self.salary_model['mu'] = mu1
        self.salary_model['kappa'] = kappa1
        self.salary_model['alpha'] = alpha1
        self.salary_model['beta'] = beta1
    
    def _update_attribute_model(self, attr_name: str, observed_value: float):
        """Update attribute model with Beta conjugate update."""
        # Skip if attribute doesn't exist in our model
        if attr_name not in self.attribute_models:
            return
        
        # Store raw observation
        self.attribute_models[attr_name]['observations'].append(observed_value)
        
        # Extract current parameters
        alpha0 = self.attribute_models[attr_name]['alpha']
        beta0 = self.attribute_models[attr_name]['beta']
        
        # Beta conjugate update
        # For continuous 0-1 values, we treat them as "success proportion"
        alpha1 = alpha0 + observed_value
        beta1 = beta0 + (1 - observed_value)
        
        # Store updated parameters
        self.attribute_models[attr_name]['alpha'] = alpha1
        self.attribute_models[attr_name]['beta'] = beta1
    
    def _learn_from_rejection(self, offer: Dict[str, Any]):
        """Learn from rejected offers to improve future market model."""
        # Identify key features of the rejected offer
        salary_percentile = self._compute_salary_percentile(offer.get('base_salary', 0))
        
        # For each attribute, note if it was particularly low
        low_attributes = []
        for attr_name, model in self.attribute_models.items():
            if attr_name in offer:
                # Get variable type
                var_type = VARIABLE_TYPES.get(attr_name)
                if not var_type:
                    continue
                    
                # Skip non-numeric values
                if not isinstance(offer[attr_name], (int, float)):
                    continue
                    
                # Normalize value
                # Check if var_type is a dictionary or an object
                if isinstance(var_type, dict):
                    # If it's a dictionary, access the var_type key
                    var_type_value = var_type.get('var_type', '')
                    if var_type_value in ["quality", "alignment"]:
                        normalized_value = (offer[attr_name] - 1) / 9.0
                    elif var_type_value == "boolean":
                        normalized_value = offer[attr_name] / 10.0
                    else:
                        continue
                else:
                    # If it's an object, access the var_type attribute
                    if var_type.var_type == "quality" or var_type.var_type == "alignment":
                        normalized_value = (offer[attr_name] - 1) / 9.0
                    elif var_type.var_type == "boolean":
                        normalized_value = offer[attr_name] / 10.0
                    else:
                        continue
                
                # Check if this attribute was below expected
                posterior_mean = model['alpha'] / (model['alpha'] + model['beta'])
                if normalized_value < posterior_mean - 0.2:  # Quite below average
                    low_attributes.append(attr_name)
        
        # Update market model based on rejection
        self._update_market_model_with_rejection(salary_percentile, low_attributes)
    
    def _compute_salary_percentile(self, salary: float) -> float:
        """Compute the percentile of a salary in current distribution."""
        if salary <= 0:
            return 0.0
            
        # Normalize salary based on location
        normalized_salary = salary / self.context["location"]["cost_modifier"]
            
        # Extract current parameters
        mu = self.salary_model['mu']
        alpha = self.salary_model['alpha']
        beta = self.salary_model['beta']
        
        # Compute degrees of freedom
        df = 2 * alpha
        
        # Compute scale parameter
        scale = math.sqrt(beta / alpha)
        
        # Approximate with normal distribution
        return norm.cdf(normalized_salary, loc=mu, scale=scale)
    
    def _update_market_model_with_rejection(self, salary_percentile: float, low_attributes: List[str]):
        """Update market model based on rejection pattern."""
        # If high salary percentile was rejected, market might be weak
        if salary_percentile > 0.7:
            # Evidence of weak market (candidates can't be too picky)
            self.market_model['beta'] += 0.5
        elif salary_percentile < 0.3 and len(low_attributes) > 0:
            # Evidence of strong market (candidates can be picky)
            self.market_model['alpha'] += 0.5
    
    def generate_salary_sample(self) -> float:
        """Generate a salary sample from current beliefs with semantic understanding."""
        # Extract current parameters
        mu = self.salary_model['mu']
        alpha = self.salary_model['alpha']
        beta = self.salary_model['beta']
        
        # Generate salary from t-distribution (more robust than normal)
        if alpha > 1:
            # Scale parameter
            scale = math.sqrt(beta / (alpha - 1))
            
            # Degrees of freedom
            df = 2 * alpha
            
            # Sample from t-distribution
            t_sample = np.random.standard_t(df)
            salary_sample = mu + scale * t_sample
        else:
            # Fall back to normal if alpha too small
            std = math.sqrt(beta)
            salary_sample = np.random.normal(mu, std)
        
        # Apply location adjustment
        adjusted_salary = salary_sample * self.context["location"]["cost_modifier"]
        
        # Don't enforce minimum salary at generation time - let the OST algorithm decide
        # We want a realistic range of offers, including some below the minimum threshold
        return max(30000, adjusted_salary)  # Just ensure it's not unreasonably low
    
    def generate_attribute_sample(self, attr_name: str) -> float:
        """Generate attribute sample from current beliefs."""
        if attr_name not in self.attribute_models:
            return 5.0  # Default for unknown attributes
        
        # Extract current parameters
        alpha = self.attribute_models[attr_name]['alpha']
        beta = self.attribute_models[attr_name]['beta']
        
        # Sample from Beta distribution (0-1 scale)
        value_0_1 = np.random.beta(alpha, beta)
        
        # Get variable type
        var_type = VARIABLE_TYPES.get(attr_name)
        if not var_type:
            # Default to 1-10 scale
            return 1 + 9 * value_0_1
        
        # Check if var_type is a dictionary or an object
        if isinstance(var_type, dict):
            # If it's a dictionary, access the var_type key
            var_type_value = var_type.get('var_type', 'quality')
            if var_type_value in ["quality", "alignment"]:
                return 1 + 9 * value_0_1  # 0-1 to 1-10 scale
            elif var_type_value == "boolean":
                return 10 * value_0_1  # 0-1 to 0-10 scale
            else:
                return 1 + 9 * value_0_1  # Default to 1-10 scale
        else:
            # If it's an object, access the var_type attribute
            if var_type.var_type == "quality" or var_type.var_type == "alignment":
                return 1 + 9 * value_0_1  # 0-1 to 1-10 scale
            elif var_type.var_type == "boolean":
                return 10 * value_0_1  # 0-1 to 0-10 scale
            else:
                return 1 + 9 * value_0_1  # Default to 1-10 scale
    
    def get_market_condition(self, time_point: float) -> float:
        """Get current market condition with semantic understanding."""
        # Base market strength from Beta distribution
        alpha = self.market_model['alpha']
        beta = self.market_model['beta']
        market_strength = alpha / (alpha + beta)
        
        # Apply seasonal adjustment
        week_of_year = int(time_point * 52) % 52
        season_index = (week_of_year // 3) % 16
        seasonal_factor = self.seasonality_factors[season_index]
        
        # Adjust based on field-specific factors
        field = self.user_profile.get("field", "general")
        field_factor = 1.0
        
        # Some fields have more seasonal variation than others
        if field == "education":
            # Education hiring follows academic year strongly
            seasonal_factor = seasonal_factor ** 1.5
        elif field == "retail":
            # Retail has even stronger seasonality
            seasonal_factor = seasonal_factor ** 2
        elif field == "software_engineering":
            # Tech tends to be less seasonal
            seasonal_factor = seasonal_factor ** 0.5
        
        # Final market condition
        adjusted_condition = market_strength * seasonal_factor * field_factor
        
        # Ensure reasonable range
        return max(0.5, min(1.5, adjusted_condition))
    
    def generate_job_offer(self, time_point: float) -> Dict[str, Any]:
        """Generate a job offer based on current beliefs with semantic understanding."""
        # Generate company and position appropriate for the field
        field = self.user_profile.get("field", "general")
        
        # Define default field companies
        default_field_companies = {
            "software_engineering": [
                "TechInnovate", "CodeCraft", "ByteBuilders", "CloudCore", "AIVentures"
            ],
            "data_science": [
                "DataDynamics", "QuantumQueries", "AIVentures", "Analytica", "DataCore"
            ],
            "marketing": [
                "BrandBurst", "MarketMind", "EngageNow", "DigitalDomains", "MediaMakers"
            ],
            "finance": [
                "CapitalCore", "FinanceFirst", "WealthWise", "InvestRight", "FiscalFocus"
            ],
            "general": [
                "GeneriCorp", "AllFields", "BusinessBasics", "CorporateConcepts", "EnterpriseCo"
            ]
        }
        
        # Define default field positions
        default_field_positions = {
            "software_engineering": [
                "Software Engineer", "Frontend Developer", "Backend Developer", 
                "DevOps Engineer", "Full Stack Developer"
            ],
            "data_science": [
                "Data Scientist", "ML Engineer", "Data Analyst", 
                "Data Engineer", "Research Scientist"
            ],
            "marketing": [
                "Marketing Manager", "Digital Marketer", "Content Strategist",
                "Brand Manager", "Social Media Specialist"
            ],
            "finance": [
                "Financial Analyst", "Investment Banker", "Accountant",
                "Finance Manager", "Financial Advisor"
            ],
            "general": [
                "Business Analyst", "Project Manager", "Operations Manager",
                "HR Specialist", "Office Manager"
            ]
        }
        
        # Define default offer quality weights
        default_offer_quality_weights = {
            "good": 0.3,  # 30% good
            "mixed": 0.5, # 50% mixed
            "bad": 0.2    # 20% bad
        }
        
        # Load data from MongoDB
        field_companies = mongodb_util.load_data("field_companies", default_data=default_field_companies)
        field_positions = mongodb_util.load_data("field_positions", default_data=default_field_positions)
        offer_quality_weights = mongodb_util.load_data("offer_quality_weights", default_data=default_offer_quality_weights)
        
        # Get companies and positions for this field (or fall back to general)
        companies = field_companies.get(field, field_companies["general"])
        positions = field_positions.get(field, field_positions["general"])
        
        company = random.choice(companies)
        position = random.choice(positions)
        
        # Get market condition
        market_condition = self.get_market_condition(time_point)
        
        # Generate salary based on current beliefs and market with more variance
        # Calculate user's minimum salary (from preferences) or default to 70% of the expected salary
        min_salary = self.user_preferences.get("min_salary", self.salary_model['mu'] * 0.7)
        
        # Generate base salary with increased variance
        base_salary = self.generate_salary_sample() * market_condition
        
        # For variety, sometimes generate salaries well above min_salary to have some good offers
        if random.random() < 0.5:  # 50% chance to get a higher salary
            # Boosted salary between min_salary and min_salary * 1.8
            boost_factor = 1.0 + random.uniform(0.3, 0.8)  
            base_salary = max(base_salary, min_salary * boost_factor)
            
        # Ensure we occasionally get very good offers (10% chance)
        if random.random() < 0.1:
            # Generate a top-tier offer
            boost_factor = 1.5 + random.uniform(0.3, 0.5)  # 1.8-2.0x multiplier
            base_salary = max(base_salary, min_salary * boost_factor)
        
        # Create offer
        offer = {
            "company": company,
            "position": position,
            "base_salary": base_salary,
            "total_compensation": base_salary * (1 + random.uniform(0.1, 0.3))
        }
        
        # Decide offer quality profile using weights from MongoDB
        # Check if offer_quality_weights is a list containing a dictionary (as returned from MongoDB)
        if isinstance(offer_quality_weights, list) and len(offer_quality_weights) > 0 and isinstance(offer_quality_weights[0], dict):
            # Use the first dictionary in the list
            quality_weights = offer_quality_weights[0].get('weights', {})
            if quality_weights:
                offer_quality = random.choices(
                    list(quality_weights.keys()),
                    weights=list(quality_weights.values())
                )[0]
            else:
                # Fallback to default "medium" quality
                offer_quality = "medium"
        else:
            # Original behavior if offer_quality_weights is a dictionary as expected
            try:
                offer_quality = random.choices(
                    list(offer_quality_weights.keys()),
                    weights=list(offer_quality_weights.values())
                )[0]
            except (TypeError, ValueError):
                # Fallback to default "medium" quality
                offer_quality = "medium"
        
        # Add attributes based on semantic understanding
        for attr_name, var_type in VARIABLE_TYPES.items():
            # Skip monetary attributes (already handled)
            if attr_name in ["base_salary", "total_compensation"]:
                continue
            
            # Skip attributes not applicable to this field
            if attr_name in self.context["field"] and \
               "applicable" in self.context["field"][attr_name] and \
               not self.context["field"][attr_name]["applicable"]:
                continue
            
            # Generate attribute value based on offer quality
            base_value = self.generate_attribute_sample(attr_name)
            
            if offer_quality == "good":
                # Boost values for good offers
                attr_value = min(10, base_value + random.uniform(0, 3))
            elif offer_quality == "bad":
                # Reduce values for bad offers
                attr_value = max(1, base_value - random.uniform(0, 3))
            else:  # mixed
                # More variability for mixed offers
                attr_value = max(1, min(10, base_value + random.uniform(-2, 2)))
            
            offer[attr_name] = attr_value
        
        return offer
    
    def get_expected_salary_range(self) -> Tuple[float, float]:
        """Get the expected range for salary with semantic understanding."""
        mu = self.salary_model['mu']
        alpha = self.salary_model['alpha']
        beta = self.salary_model['beta']
        
        # If we have enough data, use t-distribution
        if alpha > 1:
            # Compute standard deviation
            std = math.sqrt(beta / (alpha - 1))
            
            # 90% confidence interval
            return (mu - 1.645 * std, mu + 1.645 * std)
        else:
            # Not enough data, use wide range
            return (mu * 0.7, mu * 1.3)
    
    def get_attribute_expectation(self, attr_name: str) -> float:
        """Get the expected value for an attribute with semantic understanding."""
        if attr_name not in self.attribute_models:
            return 5.0  # Default middle value
        
        alpha = self.attribute_models[attr_name]['alpha']
        beta = self.attribute_models[attr_name]['beta']
        
        # Expected value of Beta distribution
        expected_value = alpha / (alpha + beta)
        
        # Get variable type
        var_type = VARIABLE_TYPES.get(attr_name)
        if not var_type:
            # Default to 1-10 scale
            return 1 + 9 * expected_value
            
        # Convert to appropriate scale based on variable type
        # Check if var_type is a dictionary or an object
        if isinstance(var_type, dict):
            # If it's a dictionary, access the var_type key
            var_type_value = var_type.get('var_type', '')
            if var_type_value in ["quality", "alignment"]:
                return 1 + 9 * expected_value  # 0-1 to 1-10 scale
            elif var_type_value == "boolean":
                return 10 * expected_value  # 0-1 to 0-10 scale
            else:
                return 1 + 9 * expected_value  # Default to 1-10 scale
        else:
            # If it's an object, access the var_type attribute
            if var_type.var_type == "quality" or var_type.var_type == "alignment":
                return 1 + 9 * expected_value  # 0-1 to 1-10 scale
            elif var_type.var_type == "boolean":
                return 10 * expected_value  # 0-1 to 0-10 scale
            else:
                return 1 + 9 * expected_value  # Default to 1-10 scale

# ===========================================================================
# 5. ENHANCED OPTIMAL STOPPING ALGORITHM
# ===========================================================================

class SemanticOST:
    """
    Enhanced Optimal Stopping Theory with semantic understanding of job variables
    and context-aware decision making.
    """
    
    def __init__(self, user_profile: Dict[str, Any], user_preferences: Dict[str, Any], 
                max_time: float = 1.0, time_units: str = "years"):
        """Initialize the enhanced semantic OST model."""
        self.user_profile = user_profile
        self.user_preferences = user_preferences
        self.max_time = max_time
        self.time_units = time_units
        
        # Create enhanced preference processor with semantic understanding
        self.preference_processor = EnhancedPreferenceProcessor(user_profile, user_preferences)
        
        # Create enhanced Bayesian belief model
        self.belief_model = EnhancedBayesianModel(user_profile, user_preferences)
        
        # Extract key parameters with semantic understanding
        self.current_salary = user_preferences.get('current_salary', 40000)
        self.min_salary = user_preferences.get('min_salary', 30000)
        # MODIFICATION 2: Increase financial runway
        self.financial_runway = user_preferences.get('financial_runway', 12) / 12  # Convert months to years
        self.risk_tolerance = user_preferences.get('risk_tolerance', 5) / 10  # Normalize to 0-1
        self.job_search_urgency = user_preferences.get('job_search_urgency', 5) / 10  # Normalize to 0-1
        self.employment_status = user_profile.get('employment_status', 'employed')
        
        # Default financial parameters
        default_financial_params = {
            "discount_rate": 0.05,  # Annual discount rate
            "search_cost_multiplier": 0.02,  # Base cost of searching as percentage of current salary
            "offer_arrival_rate": 12,  # Expected offers per year
            "unemployed_search_cost_multiplier": 0.1,  # Higher opportunity cost when unemployed
            "unemployed_discount_rate": 0.1,  # Higher time pressure when unemployed
            "interview_cost_multiplier": 0.005,  # Cost per interview as percentage of current salary
            "rejection_cost_multiplier": 0.001,  # Reputation cost of rejecting an offer
            "fatigue_factor": 1.0  # Increases with each interview
        }
        
        # Load financial parameters from MongoDB
        financial_params = mongodb_util.load_data("financial_parameters", default_data=default_financial_params)
        
        # Set financial parameters
        self.discount_rate = financial_params.get("discount_rate", 0.05)
        self.search_cost_rate = self.current_salary * financial_params.get("search_cost_multiplier", 0.02)
        self.offer_arrival_rate = financial_params.get("offer_arrival_rate", 12)
        
        # Adjust parameters based on employment status
        if self.employment_status == 'unemployed':
            self.search_cost_rate = self.min_salary * financial_params.get("unemployed_search_cost_multiplier", 0.1)
            self.discount_rate = financial_params.get("unemployed_discount_rate", 0.1)
        
        # Transaction costs (with semantic understanding)
        self.interview_cost = financial_params.get("interview_cost_multiplier", 0.005) * self.current_salary
        self.rejection_cost = financial_params.get("rejection_cost_multiplier", 0.001) * self.current_salary
        self.fatigue_factor = financial_params.get("fatigue_factor", 1.0)
        
        # State variables
        self.current_time = 0.0
        self.best_offer_so_far = None
        self.best_utility_so_far = -float('inf')
        self.observed_offers = []
        self.rejections_made = 0
        
        # Value function and thresholds
        self.time_grid = np.linspace(0, max_time, 100)
        self.value_function = np.zeros_like(self.time_grid)
        self.reservation_utilities = np.zeros_like(self.time_grid)
        
        # Initialize the value function and thresholds
        self._initialize_value_function()
    
    def _initialize_value_function(self):
        """Initialize the value function and reservation utilities using backward induction."""
        # Terminal value is 0
        self.value_function[-1] = 0
        
        # # MODIFICATION 9: Add initial debugging information
        # print("\nDEBUG: Starting value function calculation")
        # print(f"  Discount rate: {self.discount_rate:.4f}")
        # print(f"  Search cost rate: {self.search_cost_rate:.2f}")
        # print(f"  Offer arrival rate: {self.offer_arrival_rate:.2f}")
        # print(f"  Financial runway: {self.financial_runway:.2f} years")
        
        # Backward induction with semantic understanding
        for i in range(len(self.time_grid) - 2, -1, -1):
            t = self.time_grid[i]
            dt = self.time_grid[i+1] - t
            
            # Expected improvement from continuing
            expected_improvement = self._expected_improvement(t)
            
            # MODIFICATION 10: Fix numerical instability in value function
            # When calculating discounted_next_value, check if next value is unreasonably negative
            next_value = self.value_function[i+1]
            if next_value < -100:
                next_value = -100  # Limit the cascade of negative values
            
            # Discounted future value
            discounted_next_value = next_value * np.exp(-self.discount_rate * dt)
            
            # Search cost with fatigue component and semantic understanding
            fatigue_multiplier = 1 + (self.fatigue_factor - 1) * (t / self.max_time)
            
            # Adjust search cost based on employment status
            if self.employment_status == 'employed':
                # Employed people have lower search costs but higher opportunity costs
                search_cost = self.search_cost_rate * dt * fatigue_multiplier * 0.8
            else:
                # Unemployed can search more intensively but financial pressure is higher
                search_cost = self.search_cost_rate * dt * fatigue_multiplier * 1.2
            
            # Interview cost with semantic understanding
            interview_cost = self.interview_cost * self.offer_arrival_rate * dt
            
            # Total cost
            total_cost = search_cost + interview_cost
            
            # Value of continuing
            continuing_value = expected_improvement + discounted_next_value - total_cost
            
            # Adjust for financial runway with semantic understanding
            if t > self.financial_runway:
                # Financial pressure depends on employment status
                if self.employment_status == 'unemployed':
                    # Severe penalty for unemployed past runway
                    runway_penalty = search_cost * (2 + (1 - self.risk_tolerance))
                else:
                    # Moderate penalty for employed past runway
                    runway_penalty = search_cost * (1 + 0.5 * (1 - self.risk_tolerance))
                    
                continuing_value -= runway_penalty
            
            # Adjust for search urgency and career considerations
            field_urgency_factor = 1.0
            if self.user_profile.get('field') == 'technology':
                # Technology field has higher cost of career gaps
                field_urgency_factor = 1.2
            
            urgency_adjustment = (t / self.max_time) * self.job_search_urgency * \
                              expected_improvement * 0.5 * field_urgency_factor
            continuing_value -= urgency_adjustment
            
            # MODIFICATION 11: Limit how negative the value function can get
            continuing_value = max(-100, continuing_value)
            
            # Store value
            self.value_function[i] = continuing_value
            
            # MODIFICATION 12: Debug every 20 iterations and first/last 5
            # if i % 20 == 0 or i > len(self.time_grid) - 5 or i < 5:
            #     print(f"\nIteration {i}, t = {t:.2f}:")
            #     print(f"  Expected improvement: {expected_improvement:.2f}")
            #     print(f"  Next value: {next_value:.2f}, discount factor: {np.exp(-self.discount_rate * dt):.4f}")
            #     print(f"  Discounted next value: {discounted_next_value:.2f}")
            #     print(f"  Total cost: {total_cost:.2f}")
            #     print(f"  Continuing value: {continuing_value:.2f}")
            
            # Calculate reservation utility
            self.reservation_utilities[i] = self._calculate_reservation_utility(t, continuing_value)
            
            # MODIFICATION 7: Add debug output for threshold calculation 
            # if i == 0:  # Only for first timestep
            #     print("\nTHRESHOLD CALCULATION FACTORS:")
            #     print(f"  Expected improvement: {expected_improvement:.2f}")
            #     print(f"  Discounted next value: {discounted_next_value:.2f}")
            #     print(f"  Total search cost: {total_cost:.2f}")
            #     print(f"  Fatigue multiplier: {fatigue_multiplier:.2f}")
            #     print(f"  Urgency adjustment: {urgency_adjustment:.2f}")
            #     if t > self.financial_runway:
            #         print(f"  Runway penalty: {runway_penalty:.2f}")
            #     print(f"  Continuing value: {continuing_value:.2f}")
    
    def _expected_improvement(self, t: float) -> float:
        """Calculate expected improvement with semantic understanding."""
        # Generate sample offers using enhanced Bayesian model
        n_samples = 50
        sample_offers = [self.belief_model.generate_job_offer(t) for _ in range(n_samples)]
        sample_utilities = [self.preference_processor.calculate_offer_utility(offer) for offer in sample_offers]
        
        # If we have a best offer, only count improvements
        if self.best_utility_so_far > -float('inf'):
            improvements = [max(0, u - self.best_utility_so_far) for u in sample_utilities]
        else:
            improvements = sample_utilities
        
        # Expected improvement with semantically-aware arrival probability
        base_arrival_prob = 1 - np.exp(-self.offer_arrival_rate * (self.max_time - t) / 100)
        
        # Adjust based on field and market condition
        market_condition = self.belief_model.get_market_condition(t)
        field = self.user_profile.get('field', 'general')
        
        # Default field-specific arrival rate adjustments
        default_field_factors = {
            "software_engineering": 1.2,  # More opportunities in tech
            "finance": 0.9,  # More competitive, fewer offers
            "general": 1.0  # Default
        }
        
        # Load field factors from MongoDB
        field_factors = mongodb_util.load_data("field_arrival_rate_factors", default_data=default_field_factors)
        
        # Get field factor
        field_factor = field_factors.get(field, field_factors.get("general", 1.0))
        
        # Final arrival probability
        adjusted_arrival_prob = base_arrival_prob * market_condition * field_factor
        
        expected_improvement = np.mean(improvements) * adjusted_arrival_prob
        
        # MODIFICATION 4: Boost expected improvement
        expected_improvement *= 1.5  # Boost by 50%
        
        return expected_improvement
    
    def _calculate_reservation_utility(self, t: float, continuing_value: float) -> float:
        """Calculate reservation utility with semantic understanding."""
        # MODIFICATION 13: Complete redesign of reservation utility calculation
        # Instead of using continuing_value (which has numerical issues),
        # calculate based on mean utility of sample offers
        
        # Generate sample offers to get baseline statistics
        n_samples = 30
        sample_offers = [self.belief_model.generate_job_offer(t) for _ in range(n_samples)]
        sample_utilities = [self.preference_processor.calculate_offer_utility(offer) for offer in sample_offers]
        
        # Sort utilities to find distribution
        sorted_utilities = sorted(sample_utilities)
        
        # Use 40th percentile as baseline reservation utility (somewhat selective)
        percentile_idx = int(0.4 * len(sorted_utilities))
        base_reservation = sorted_utilities[percentile_idx]
        
        # Get context-aware market condition
        market_condition = self.belief_model.get_market_condition(t)
        
        # Get adjustment factors - all normalized around 1.0
        risk_factor = self._get_risk_factor()
        time_factor = self._get_time_factor(t)
        runway_factor = self._get_runway_factor(t)
        market_factor = self._get_market_factor(market_condition)
        rejection_factor = self._get_rejection_factor()
        field_factor = self._get_field_factor()
        experience_factor = self._get_experience_factor()
        
        # Apply all adjustment factors to base reservation utility
        adjusted_utility = base_reservation * risk_factor * time_factor * runway_factor * \
                        market_factor * rejection_factor * field_factor * \
                        experience_factor
        
        # MODIFICATION 5: Boost reservation utility directly
        adjusted_utility += 3.0  # Add a flat boost to make more selective
        
        # Lower bound based on risk tolerance and employment status
        if self.employment_status == 'unemployed':
            # Lower floor when unemployed - can't be as selective
            min_acceptable = 0.6 * (2 - self.risk_tolerance) 
        else:
            min_acceptable = 2 * (1 - self.risk_tolerance)
        
        # Debug output for this calculation
        # if t < 0.1:  # Only for early time points
        #     print(f"\nRESERVATION UTILITY CALCULATION:")
        #     print(f"  Sample utilities: min={min(sample_utilities):.2f}, " +
        #           f"mean={np.mean(sample_utilities):.2f}, max={max(sample_utilities):.2f}")
        #     print(f"  Base reservation (40th percentile): {base_reservation:.2f}")
        #     print(f"  Risk factor: {risk_factor:.2f}")
        #     print(f"  Time factor: {time_factor:.2f}")
        #     print(f"  Runway factor: {runway_factor:.2f}")
        #     print(f"  Market factor: {market_factor:.2f}")
        #     print(f"  Rejection factor: {rejection_factor:.2f}")
        #     print(f"  Field factor: {field_factor:.2f}")
        #     print(f"  Experience factor: {experience_factor:.2f}")
        #     print(f"  After adjustments: {adjusted_utility:.2f}")
        #     print(f"  Min acceptable: {min_acceptable:.2f}")
        #     print(f"  Final reservation utility: {max(min_acceptable, adjusted_utility):.2f}")
            
        return max(min_acceptable, adjusted_utility)
    
    def _get_risk_factor(self) -> float:
        """Get risk adjustment factor based on risk tolerance."""
        if self.risk_tolerance > 0.7:  # High risk tolerance
            return 1.2  # Higher threshold - more selective
        elif self.risk_tolerance < 0.3:  # Low risk tolerance
            return 0.8  # Lower threshold - less selective
        else:
            return 1.0  # Neutral
    
    def _get_time_factor(self, t: float) -> float:
        """Get time adjustment factor with field-specific considerations."""
        # MODIFICATION 8: Make time effect more beneficial
        if t < 0.3 * self.max_time:
            # Stay more selective longer in early phase
            time_factor = 1.5  # Increased from 1.2
        elif t > 0.9 * self.max_time:
            # Only reduce selectivity at very end
            time_factor = 0.8
        else:
            time_factor = 1.2  # Stay more selective in middle phase
        
        # Field-specific adjustments
        field = self.user_profile.get('field', 'general')
        if field == 'seasonal':
            # For highly seasonal fields, timing is more important
            if t < 0.2 * self.max_time or t > 0.8 * self.max_time:
                time_factor *= 1.2
        
        return time_factor
    
    def _get_runway_factor(self, t: float) -> float:
        """Get financial runway adjustment factor with semantic understanding."""
        runway_remaining = self.financial_runway - t
        
        # Base factor
        if runway_remaining < 0:
            # Much less selective once runway is gone
            runway_factor = 0.5
        elif runway_remaining < 0.2:
            # Somewhat less selective when runway is low
            runway_factor = 0.8
        else:
            runway_factor = 1.0
        
        # Employment status affects runway pressure
        # MODIFICATION 6: Modify unemployment penalty
        if self.employment_status == 'unemployed':
            # Less penalty for being unemployed (was 0.9)
            runway_factor *= 0.95
        
        return runway_factor
    
    def _get_market_factor(self, market_condition: float) -> float:
        """Get market adjustment factor with semantic understanding."""
        if market_condition < 0.9:
            # Weaker market - be less selective
            return 0.9
        elif market_condition > 1.1:
            # Stronger market - be more selective
            return 1.1
        else:
            return 1.0
    
    def _get_rejection_factor(self) -> float:
        """Get adjustment factor based on rejection history."""
        # More rejections = less selective
        return max(0.8, 1.0 - 0.05 * self.rejections_made)
    
    def _get_field_factor(self) -> float:
        """Get field-specific adjustment factor."""
        field = self.user_profile.get('field', 'general')
        
        if field == 'software_engineering':
            # Tech has more options - can be more selective
            return 1.1
        elif field == 'academic':
            # Academic jobs are scarce - be less selective
            return 0.9
        else:
            return 1.0
    
    def _get_experience_factor(self) -> float:
        """Get experience-level adjustment factor."""
        experience = self.user_profile.get('experience_level', 'mid')
        
        if experience == 'entry':
            # Entry-level can't be as selective
            return 0.9
        elif experience == 'senior' or experience == 'executive':
            # Senior roles are more selective
            return 1.1
        else:
            return 1.0
    
    def get_reservation_utility(self, t: float) -> float:
        """Get the reservation utility at time t."""
        idx = np.argmin(np.abs(self.time_grid - t))
        return self.reservation_utilities[idx]
    
    def calculate_offer_utility(self, offer: Dict[str, Any]) -> float:
        """Calculate offer utility with semantic understanding."""
        return self.preference_processor.calculate_offer_utility(offer)
    
    def should_accept_offer(self, offer: Dict[str, Any], t: float) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine whether to accept a job offer with semantic understanding.
        
        Args:
            offer: The job offer data
            t: Current time point
            
        Returns:
            Tuple of (accept_decision, reason, decision_details)
        """
        # MODIFICATION 14: Completely redesign decision logic to properly use thresholds
        
        # First check hard constraints
        meets_constraints, constraint_reason = self.preference_processor.check_hard_constraints(offer)
        if not meets_constraints:
            return False, constraint_reason, {"hard_constraint_violation": True}
        
        # Calculate offer utility
        utility = self.calculate_offer_utility(offer)
        
        # Get current threshold
        threshold = self.get_reservation_utility(t)
        
        # Get market condition
        market_condition = self.belief_model.get_market_condition(t)
        
        # Calculate a small rejection cost (around 0.5% of the threshold)
        # This is a small cost for turning down an offer
        rejection_cost = 0.005 * threshold * (1 + 0.05 * self.rejections_made)
        
        # Decision details for transparency
        decision_details = {
            "offer_utility": utility,
            "threshold": threshold,
            "market_condition": market_condition,
            "time_factor": self._get_time_factor(t),
            "runway_factor": self._get_runway_factor(t),
            "rejection_cost": rejection_cost,
            "financial_runway_remaining": max(0, self.financial_runway - t),
            "employment_status": self.employment_status
        }
        
        # BASIC DECISION: If utility exceeds threshold, accept
        if utility >= threshold:
            return True, f"Accept: Utility ({utility:.2f}) exceeds threshold ({threshold:.2f})", decision_details
        
        # Calculate adjusted threshold with a reasonable rejection cost
        adjusted_threshold = threshold - rejection_cost
        
        # # Debug output for decision making
        # print("\nDECISION CALCULATION:")
        # print(f"  Offer utility: {utility:.2f}")
        # print(f"  Base threshold: {threshold:.2f}")
        # print(f"  Rejection cost: {rejection_cost:.2f}")
        # print(f"  Adjusted threshold: {adjusted_threshold:.2f}")
        
        # ADJUSTED DECISION: If utility exceeds adjusted threshold (considering rejection cost)
        if utility >= adjusted_threshold:
            return True, f"Accept considering rejection costs: Utility ({utility:.2f}) exceeds adjusted threshold ({adjusted_threshold:.2f})", decision_details
        
        # FINANCIAL PRESSURE: If past financial runway, lower standards
        if t > self.financial_runway:
            # Desperation factor depends on employment status
            desperation_factor = 0.6 if self.employment_status == 'unemployed' else 0.8
            financial_pressure_threshold = threshold * desperation_factor
            
            print(f"  Financial pressure threshold: {financial_pressure_threshold:.2f}")
            
            if utility >= financial_pressure_threshold:
                return True, f"Accept under financial pressure: Utility ({utility:.2f}) exceeds adjusted threshold ({financial_pressure_threshold:.2f})", decision_details
        
        # RANDOM ELEMENT: Occasionally accept suboptimal offers (irrational decision)
        emotional_factor = random.random()
        if utility > threshold * 0.8 and emotional_factor < 0.05:  # 5% chance if within 80% of threshold
            return True, f"Accept (unusual): Utility ({utility:.2f}) below threshold ({threshold:.2f}) but accepted due to intangible factors", decision_details
        
        # Default: REJECT
        return False, f"Reject: Utility ({utility:.2f}) below threshold ({threshold:.2f})", decision_details
    
    def observe_offer(self, offer: Dict[str, Any], t: float, was_accepted: bool = None):
        """Observe a new job offer and update models with semantic understanding."""
        # Update current time
        self.current_time = t
        
        # Store the offer
        self.observed_offers.append((offer, t))
        
        # Calculate offer utility
        utility = self.calculate_offer_utility(offer)
        
        # Update best offer if this is better
        if utility > self.best_utility_so_far:
            self.best_offer_so_far = offer
            self.best_utility_so_far = utility
        
        # Update enhanced Bayesian belief model
        self.belief_model.update_with_offer(offer, was_accepted)
        
        # Track rejection and update fatigue
        if was_accepted is False:
            self.rejections_made += 1
            # Fatigue increase depends on field
            field = self.user_profile.get('field', 'general')
            if field in ['high_stress', 'interview_intensive']:
                # More fatigue in demanding fields
                self.fatigue_factor += 0.15
            else:
                self.fatigue_factor += 0.1
        
        # Increase fatigue for every offer
        self.fatigue_factor += 0.02
        
        # Recalculate value function with updated info
        self._initialize_value_function()
    
    def get_search_insights(self) -> Dict[str, Any]:
        """Get insights about the job search with semantic understanding."""
        # Current time point
        t = self.current_time
        
        # Market insights with semantic understanding
        market_condition = self.belief_model.get_market_condition(t)
        
        # Salary expectations with field context
        salary_range = self.belief_model.get_expected_salary_range()
        field = self.user_profile.get('field', 'general')
        experience = self.user_profile.get('experience_level', 'mid')
        
        # Adjust salary range display based on location
        location = self.user_profile.get('location', 'other')
        location_factor = self.belief_model.context["location"]["cost_modifier"]
        adjusted_range = (salary_range[0] * location_factor, salary_range[1] * location_factor)
        
        # Top attributes with semantic meaning
        top_attributes = {}
        for attr_name in self.belief_model.attribute_models:
            var_type = VARIABLE_TYPES.get(attr_name)
            if not var_type:
                continue
                
            # Skip attributes not applicable to this field
            if attr_name in self.belief_model.context["field"] and \
               "applicable" in self.belief_model.context["field"][attr_name] and \
               not self.belief_model.context["field"][attr_name]["applicable"]:
                continue
                
            expected_value = self.belief_model.get_attribute_expectation(attr_name)
            top_attributes[attr_name] = expected_value
        
        # Sort and get top 3
        top_attributes = dict(sorted(top_attributes.items(), key=lambda x: x[1], reverse=True)[:3])
        
        # Expected time with field-specific considerations
        expected_time = self._estimate_time_to_acceptable_offer()
        
        insights = {
            "current_time": t,
            "market_condition": market_condition,
            "expected_salary_range": adjusted_range,
            "top_attributes": top_attributes,
            "field_context": field,
            "experience_level": experience,
            "offers_observed": len(self.observed_offers),
            "rejections_made": self.rejections_made,
            "current_threshold": self.get_reservation_utility(t),
            "initial_threshold": self.reservation_utilities[0],
            "estimated_time_remaining": expected_time,
            "financial_runway_remaining": max(0, self.financial_runway - t)
        }
        
        # Save insights to MongoDB for analytics
        insights_to_save = insights.copy()
        # Convert tuple to list for MongoDB (can't store tuples)
        if "expected_salary_range" in insights_to_save:
            insights_to_save["expected_salary_range"] = list(insights_to_save["expected_salary_range"])
        mongodb_util.save_document("search_insights", 
                                  {"insights": insights_to_save, 
                                   "timestamp": datetime.datetime.now(),
                                   "field": field,
                                   "experience_level": experience})
        
        return insights
    
    def save_user_feedback(self, offer: Dict[str, Any], accepted: bool, user_rating: float = None, feedback: str = None):
        """
        Save user feedback about an offer and decision to MongoDB for model improvement.
        
        Args:
            offer: The job offer data
            accepted: Whether the offer was accepted
            user_rating: Optional user rating of the decision (1-10)
            feedback: Optional text feedback
        """
        # Create feedback document
        feedback_doc = {
            "offer": offer,
            "accepted": accepted,
            "model_utility": self.calculate_offer_utility(offer),
            "timestamp": datetime.datetime.now(),
            "field": self.user_profile.get("field", "general"),
            "experience_level": self.user_profile.get("experience_level", "mid"),
            "employment_status": self.user_profile.get("employment_status", "employed")
        }
        
        # Add optional fields if provided
        if user_rating is not None:
            feedback_doc["user_rating"] = user_rating
        
        if feedback:
            feedback_doc["feedback_text"] = feedback
        
        # Save to MongoDB
        mongodb_util.save_document("user_feedback", feedback_doc)
    
    def _estimate_time_to_acceptable_offer(self) -> float:
        """Estimate time to acceptable offer with semantic understanding."""
        # Current time and threshold
        t = self.current_time
        threshold = self.get_reservation_utility(t)
        
        # Generate sample offers with field-specific models
        n_samples = 100
        sample_offers = [self.belief_model.generate_job_offer(t) for _ in range(n_samples)]
        sample_utilities = [self.preference_processor.calculate_offer_utility(offer) for offer in sample_offers]
        
        # Estimate probability of acceptable offer
        p_acceptable = sum(1 for u in sample_utilities if u >= threshold) / n_samples
        
        # Safety check
        if p_acceptable <= 0.01:  # Very low chance
            return self.max_time - t  # Pessimistic estimate
            
        # Default field-specific arrival rate factors
        default_field_arrival_factors = {
            "software_engineering": 1.2,  # More opportunities in tech
            "academic": 0.7,  # Fewer opportunities in academia
            "general": 1.0  # Default
        }
        
        # Load field arrival factors from MongoDB
        field_arrival_factors = mongodb_util.load_data("field_arrival_factors", default_data=default_field_arrival_factors)
        
        # Get field and its factor
        field = self.user_profile.get('field', 'general')
        field_arrival_factor = field_arrival_factors.get(field, field_arrival_factors.get("general", 1.0))
            
        # Expected time depends on field-specific arrival rate
        expected_offers_needed = 1 / p_acceptable
        field_adjusted_arrival_rate = self.offer_arrival_rate * field_arrival_factor
        expected_time = expected_offers_needed / field_adjusted_arrival_rate
        
        # Cap at remaining time
        return min(expected_time, self.max_time - t)

# ===========================================================================
# 6. EXAMPLE USER PROFILES FOR TESTING
# ===========================================================================

# Default profiles for different fields
DEFAULT_FIELD_PROFILES = {
    "software_engineering": {
        "field": "software_engineering",
        "experience_level": "mid",
        "location": "san_francisco",
        "employment_status": "employed",
        "deal_breakers": ["excessive_travel", "on_call_rotation"],
        "would_accept_lower_salary_for": ["remote_work", "work_life_balance"]
    },
    "marketing": {
        "field": "marketing",
        "experience_level": "senior",
        "location": "new_york",
        "employment_status": "unemployed",
        "deal_breakers": ["toxic_culture"],
        "would_accept_lower_salary_for": ["career_growth", "company_reputation"]
    }
}

# Default preferences for different fields
DEFAULT_FIELD_PREFERENCES = {
    "software_engineering": {
        "field": "software_engineering",
        "current_salary": 120000,
        "min_salary": 110000,
        "financial_runway": 12,
        "risk_tolerance": 9,
        "job_search_urgency": 5,
        "compensation_weight": 4,
        "career_growth_weight": 5,
        "work_life_balance_weight": 4,
        "company_reputation_weight": 3,
        "remote_work_weight": 4,
        "tech_stack_alignment_weight": 5,
        "team_collaboration_weight": 4,
        "role_responsibilities_weight": 3
    },
    "marketing": {
        "field": "marketing",
        "current_salary": 95000,
        "min_salary": 85000,
        "financial_runway": 12,
        "risk_tolerance": 9,
        "job_search_urgency": 8,
        "compensation_weight": 4,
        "career_growth_weight": 5,
        "work_life_balance_weight": 3,
        "company_reputation_weight": 5,
        "remote_work_weight": 2,
        "role_responsibilities_weight": 4
    }
}

# This initialization has been moved to create_user_profile to avoid multiple duplications
# It will only initialize if the profiles don't exist

def create_user_profile(field: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create a user profile for any field.
    
    This function follows a dynamic approach that allows adding new fields without modifying 
    the core code. It uses a single collection with a field indicator rather than creating 
    separate collections for each field type.
    
    Args:
        field: The professional field (e.g., "software_engineering", "marketing")
        
    Returns:
        Tuple of (user_profile, user_preferences)
    """
    # Get fallback field (default to software_engineering if field not found)
    fallback_field = "software_engineering"
    
    # Check if we already have profiles for this field in MongoDB
    filter_query = {"field": field}
    
    # Attempt to load existing profile
    collection = mongodb_util.db["user_profiles"]
    existing_profile = collection.find_one(filter_query) if mongodb_util.connected else None
    
    if existing_profile:
        # Profile exists - load from database
        user_profile = dict(existing_profile)
        if '_id' in user_profile:
            del user_profile['_id']
    else:
        # Profile doesn't exist - create new default and save
        default_user_profile = DEFAULT_FIELD_PROFILES.get(field, DEFAULT_FIELD_PROFILES.get(fallback_field)).copy()
        default_user_profile["field"] = field  # Ensure field is set correctly
        
        # Save to MongoDB (only if not already there)
        if mongodb_util.connected:
            collection.insert_one(default_user_profile)
            print(f"Created new user profile for field: {field}")
            
        user_profile = default_user_profile
    
    # Now do the same for preferences
    collection = mongodb_util.db["user_preferences"]
    existing_preferences = collection.find_one(filter_query) if mongodb_util.connected else None
    
    if existing_preferences:
        # Preferences exist - load from database
        user_preferences = dict(existing_preferences)
        if '_id' in user_preferences:
            del user_preferences['_id']
    else:
        # Preferences don't exist - create defaults and save
        default_user_preferences = DEFAULT_FIELD_PREFERENCES.get(field, DEFAULT_FIELD_PREFERENCES.get(fallback_field)).copy()
        default_user_preferences["field"] = field  # Ensure field is set correctly
        
        # Save to MongoDB (only if not already there)
        if mongodb_util.connected:
            collection.insert_one(default_user_preferences)
            print(f"Created new user preferences for field: {field}")
            
        user_preferences = default_user_preferences
    
    return user_profile, user_preferences

# For backward compatibility
def create_swe_profile() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create a software engineer profile for testing."""
    return create_user_profile("software_engineering")

def create_marketing_profile() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create a marketing professional profile for testing."""
    return create_user_profile("marketing")

def add_new_field(field_name: str, field_profile: Dict[str, Any], field_preferences: Dict[str, Any]) -> None:
    """
    Add a new professional field to the system with its default profile and preferences.
    
    This allows dynamically extending the system to support new fields without modifying code.
    
    Args:
        field_name: Name of the new field (e.g., "data_science", "product_management")
        field_profile: Default profile data for this field
        field_preferences: Default preferences data for this field
    """
    # Ensure field is set in both dictionaries
    field_profile["field"] = field_name
    field_preferences["field"] = field_name
    
    # Add to DEFAULT dictionaries in memory
    DEFAULT_FIELD_PROFILES[field_name] = field_profile
    DEFAULT_FIELD_PREFERENCES[field_name] = field_preferences
    
    # Save to MongoDB
    mongodb_util.save_document("user_profiles", field_profile, {"field": field_name})
    mongodb_util.save_document("user_preferences", field_preferences, {"field": field_name})
    
    print(f"Added new field: {field_name}")

# ===========================================================================
# 7. SIMULATION DEMO
# ===========================================================================

def visualize_ost_results(ost, offers_data=None):
    """
    Visualize the OST algorithm results.
    
    Args:
        ost: The SemanticOST instance
        offers_data: List of (time, utility, threshold, decision) tuples for each offer
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the reservation utilities over time
    plt.subplot(2, 1, 1)
    plt.plot(ost.time_grid, ost.reservation_utilities, 'b-', linewidth=2, label='Reservation Utility')
    
    # Add offer decision points if available
    if offers_data:
        accepted_t = []
        accepted_u = []
        rejected_t = []
        rejected_u = []
        thresholds = []
        threshold_times = []
        
        for t, utility, threshold, accepted in offers_data:
            if accepted:
                accepted_t.append(t)
                accepted_u.append(utility)
            else:
                rejected_t.append(t)
                rejected_u.append(utility)
            threshold_times.append(t)
            thresholds.append(threshold)
        
        plt.scatter(rejected_t, rejected_u, color='red', s=80, label='Rejected Offers', marker='x')
        if accepted_t:
            plt.scatter(accepted_t, accepted_u, color='green', s=100, label='Accepted Offer', marker='o')
        plt.scatter(threshold_times, thresholds, color='purple', s=60, label='Threshold at Decision', marker='_')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Utility')
    plt.title('Reservation Utility Threshold Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot market conditions and other factors over time if available
    plt.subplot(2, 1, 2)
    
    # Generate market conditions for visualization
    market_times = np.linspace(0, ost.max_time, 50)
    market_values = [ost.belief_model.get_market_condition(t) for t in market_times]
    plt.plot(market_times, market_values, 'g-', label='Market Condition')
    
    # Add other factors
    time_factors = [ost._get_time_factor(t) for t in market_times]
    runway_factors = [ost._get_runway_factor(t) for t in market_times]
    
    plt.plot(market_times, time_factors, 'r--', label='Time Factor')
    plt.plot(market_times, runway_factors, 'b--', label='Runway Factor')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Factor Value')
    plt.title('Contextual Factors Affecting Decisions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_decision_process(ost, verbose=True):
    """
    Visualize the decision process of the OST algorithm with a step-by-step breakdown.
    
    Args:
        ost: The SemanticOST instance
        verbose: Whether to print detailed explanation
    """
    # Create a set of synthetic offers at different time points
    time_points = np.linspace(0.1, 0.9, 6)
    offers = []
    
    if verbose:
        print("\n=== DECISION PROCESS VISUALIZATION ===")
        print("Generating synthetic offers to demonstrate the decision process...")
    
    # Generate a range of offers (some above threshold, some below)
    for i, t in enumerate(time_points):
        # Generate an offer from the belief model
        offer = ost.belief_model.generate_job_offer(t)
        
        # Calculate the utility
        utility = ost.preference_processor.calculate_offer_utility(offer)
        
        # Get the threshold at this time
        threshold = ost.get_reservation_utility(t)
        
        # Store the offer with its decision metrics
        offers.append({
            'time': t,
            'offer': offer,
            'utility': utility,
            'threshold': threshold,
            'accept': utility >= threshold,
            'company': offer['company'],
            'salary': offer['base_salary']
        })
        
        if verbose:
            if i == 0:
                print(f"\nOffer 1 (t={t:.2f}):")
                print(f"  Company: {offer['company']}")
                print(f"  Salary: ${offer['base_salary']:,.2f}")
                print(f"  Utility: {utility:.2f}")
                print(f"  Threshold: {threshold:.2f}")
                print(f"  Decision: {'ACCEPT' if utility >= threshold else 'REJECT'}")
                print("\nDecision Factors Analysis:")
                print(f"  Market condition: {ost.belief_model.get_market_condition(t):.2f}")
                print(f"  Time factor: {ost._get_time_factor(t):.2f}")
                print(f"  Risk factor: {ost._get_risk_factor():.2f}")
                print(f"  Runway factor: {ost._get_runway_factor(t):.2f}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: The offers and thresholds
    plt.subplot(2, 1, 1)
    
    # Plot the reservation utility curve
    plt.plot(ost.time_grid, ost.reservation_utilities, 'b-', linewidth=2, label='Threshold')
    
    # Plot the offers
    accepted_t = [o['time'] for o in offers if o['accept']]
    accepted_u = [o['utility'] for o in offers if o['accept']]
    rejected_t = [o['time'] for o in offers if not o['accept']]
    rejected_u = [o['utility'] for o in offers if not o['accept']]
    
    plt.scatter(rejected_t, rejected_u, color='red', s=100, label='Rejected Offers', marker='x')
    plt.scatter(accepted_t, accepted_u, color='green', s=120, label='Accepted Offers', marker='o')
    
    # Add annotations for offers
    for i, o in enumerate(offers):
        label = f"{i+1}: {o['company']}\n${o['salary']/1000:.0f}k"
        plt.annotate(label, (o['time'], o['utility']), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Utility')
    plt.title('Offer Evaluation Against Changing Threshold')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: The components that affect the threshold
    plt.subplot(2, 1, 2)
    
    times = np.linspace(0, ost.max_time, 50)
    
    # Plot each factor that affects the threshold
    market_conditions = [ost.belief_model.get_market_condition(t) for t in times]
    time_factors = [ost._get_time_factor(t) for t in times]
    runway_factors = [ost._get_runway_factor(t) for t in times]
    risk_factor = ost._get_risk_factor()
    risk_factors = [risk_factor] * len(times)
    
    plt.plot(times, market_conditions, 'g-', linewidth=2, label='Market Condition')
    plt.plot(times, time_factors, 'r-', linewidth=2, label='Time Pressure')
    plt.plot(times, runway_factors, 'b-', linewidth=2, label='Financial Runway')
    plt.plot(times, risk_factors, 'k--', linewidth=1, label='Risk Tolerance')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Factor Value')
    plt.title('Factors Affecting Decision Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_semantic_ost_simulation():
    """Run a simulation of the semantic OST algorithm."""
    print("=== SEMANTIC OPTIMAL STOPPING THEORY SIMULATION ===\n")
    
    # Choose a profile
    field_options = list(DEFAULT_FIELD_PROFILES.keys())
    profile_choice = random.choice(field_options)
    
    # Create user profile using the generic function
    user_profile, user_preferences = create_user_profile(profile_choice)
    
    # Determine display name for the field
    field_display_names = {
        "software_engineering": "Software Engineer",
        "marketing": "Marketing Professional"
    }
    display_name = field_display_names.get(profile_choice, profile_choice.capitalize())
    print(f"Using {display_name} profile")
    
    print("\nUser profile:")
    for key, value in sorted(user_profile.items()):
        print(f"  {key}: {value}")
    
    print("\nUser preferences:")
    for key, value in sorted(user_preferences.items()):
        print(f"  {key}: {value}")
    
    # Initialize Semantic OST
    ost = SemanticOST(user_profile, user_preferences, max_time=1.0, time_units="years")
    
    print("\nOST Model initialized with contextual parameters:")
    print(f"  Field: {user_profile['field']}")
    print(f"  Experience: {user_profile['experience_level']}")
    print(f"  Location: {user_profile['location']}")
    print(f"  Employment status: {user_profile['employment_status']}")
    print(f"  Financial runway: {ost.financial_runway:.2f} years")
    print(f"  Expected salary: ${ost.belief_model.context['expected_salary']:,.2f}")
    print(f"  Initial reservation utility: {ost.reservation_utilities[0]:.2f}")
    
    # Simulate offer evaluation
    print("\nPROCESSING OFFERS WITH SEMANTIC UNDERSTANDING...")
    
    # MODIFICATION 15: Generate more offers and ensure they're spread out over time
    num_offers = random.randint(8, 12)  # More offers to test with
    accepted = False
    
    # Track offers for visualization
    offers_data = []
    
    for i in range(num_offers):
        # Distribute offers throughout the search period, with exact timing based on offer number
        t = i / num_offers  # Ensures even distribution from 0 to almost 1.0
        t += random.random() * 0.05  # Add slight randomness but keep sequential
        t = min(t, 0.99)  # Cap at 0.99 to stay within max_time
        
        print(f"\n--- TIME: {t:.2f} YEARS ---")
        
        # Generate an offer with semantic understanding
        offer = ost.belief_model.generate_job_offer(t)
        
        # Print offer details with semantic context
        print(f"Received offer from {offer['company']} for {offer['position']}")
        print(f"Base Salary: ${offer['base_salary']:,.2f}")
        print("Key attributes:")
        
        # Print top attributes (excluding monetary)
        # Check if variable types are dictionaries or objects
        top_attrs = []
        for k, v in offer.items():
            if k in VARIABLE_TYPES and isinstance(v, (int, float)):
                var_type = VARIABLE_TYPES[k]
                # Check if it's a dictionary
                if isinstance(var_type, dict):
                    var_type_value = var_type.get('var_type', '')
                    if var_type_value not in ["monetary"]:
                        top_attrs.append((k, v))
                else:
                    # It's an object
                    if var_type.var_type not in ["monetary"]:
                        top_attrs.append((k, v))
                        
        # Sort and take top 4
        top_attrs = sorted(top_attrs, key=lambda x: x[1], reverse=True)[:4]
        
        for attr_name, value in top_attrs:
            var_type = VARIABLE_TYPES.get(attr_name)
            if var_type:
                # Check if it's a dictionary
                if isinstance(var_type, dict):
                    description = var_type.get('description', attr_name)
                else:
                    description = var_type.description
                # Check if this is a monetary attribute (base_salary, total_compensation)
                is_monetary = False
                if isinstance(var_type, dict):
                    is_monetary = var_type.get('var_type') == 'monetary'
                else:
                    is_monetary = var_type.var_type == 'monetary'
                
                # Format differently based on attribute type
                if attr_name in ['base_salary', 'total_compensation'] or is_monetary:
                    print(f"  {attr_name}: ${value:,.2f} ({description})")
                else:
                    print(f"  {attr_name}: {value}/10 ({description})")
        
        # Check hard constraints first
        meets_constraints, constraint_reason = ost.preference_processor.check_hard_constraints(offer)
        if not meets_constraints:
            print(f"\nDECISION: REJECT (Hard constraint violation)")
            print(f"Reason: {constraint_reason}")
            
            # Update model with rejection
            ost.observe_offer(offer, t, False)
            continue
        
        # Calculate utility with semantic understanding
        utility = ost.calculate_offer_utility(offer)
        print(f"\nCalculated utility: {utility:.2f}/10")
        
        # Make decision with semantic understanding
        should_accept, reason, details = ost.should_accept_offer(offer, t)
        
        # Track this offer for visualization
        offers_data.append((t, utility, details['threshold'], should_accept))
        
        # Print decision details
        print(f"Current threshold: {details['threshold']:.2f}")
        print(f"Current market condition: {details['market_condition']:.2f}")
        
        # Add field-specific context to the decision
        field_context = ""
        if user_profile['field'] == 'software_engineering':
            if offer.get('tech_stack_alignment', 0) > 7:
                field_context = " (Strong tech stack alignment is a positive factor)"
            elif 'remote_work' in offer and offer['remote_work'] > 7:
                field_context = " (Good remote work options are valued in tech)"
        elif user_profile['field'] == 'marketing':
            if offer.get('company_reputation', 0) > 7:
                field_context = " (Strong company reputation is highly valued in marketing)"
        
        print(f"DECISION: {'ACCEPT' if should_accept else 'REJECT'}{field_context}")
        print(f"Reason: {reason}")
        
        # End search if accepted
        if should_accept:
            print("\n=== OFFER ACCEPTED! SEARCH COMPLETE ===")
            print(f"Accepted offer from {offer['company']} at time {t:.2f} years")
            ost.observe_offer(offer, t, True)
            
            # Save user feedback to MongoDB
            # Simulate a user rating of the decision (8-10 for acceptance)
            user_rating = random.uniform(8, 10)
            feedback = "Good match for my skills and preferences"
            ost.save_user_feedback(offer, True, user_rating, feedback)
            
            accepted = True
            accepted_offer = offer
            break
        
        # Update Bayesian model with rejection
        ost.observe_offer(offer, t, False)
        
        # Save rejection feedback to MongoDB
        # Simulate a user rating of the decision (5-9 for rejection)
        user_rating = random.uniform(5, 9)
        feedback = "Didn't meet my expectations for salary and growth"
        ost.save_user_feedback(offer, False, user_rating, feedback)
        
        # Get field-specific insights
        insights = ost.get_search_insights()
        
        print("\nUpdated market insights with field context:")
        print(f"  Market condition: {insights['market_condition']:.2f}")
        print(f"  Expected salary range in {user_profile['location']}: ${insights['expected_salary_range'][0]:,.2f} - ${insights['expected_salary_range'][1]:,.2f}")
        print(f"  Current threshold: {ost.get_reservation_utility(t):.2f} (initially {ost.reservation_utilities[0]:.2f})")
        print(f"  Top attributes for {insights['field_context']} professionals:")
        for attr, value in insights['top_attributes'].items():
            var_type = VARIABLE_TYPES.get(attr)
            if var_type:
                print(f"    {attr}: {value:.1f}/10 ({var_type.description})")
        
        # Estimated time with field-specific context
        if user_profile['field'] == 'software_engineering':
            print(f"  Estimated time to acceptable offer: {insights['estimated_time_remaining']:.2f} years (Tech job market is typically more active)")
        else:
            print(f"  Estimated time to acceptable offer: {insights['estimated_time_remaining']:.2f} years")
    
    # Check if reached time horizon without accepting
    if not accepted:
        print("\n=== TIME HORIZON REACHED WITHOUT ACCEPTING ANY OFFER ===")
        print(f"Searched for full {ost.max_time} {ost.time_units} without finding acceptable offer")
        
        # Report best offer seen with semantic context
        if ost.best_offer_so_far:
            print("\nBest offer encountered:")
            best = ost.best_offer_so_far
            print(f"Company: {best['company']}")
            print(f"Position: {best['position']}")
            print(f"Salary: ${best['base_salary']:,.2f}")
            print(f"Utility: {ost.best_utility_so_far:.2f}/10")
            
            # Field-specific explanation
            if user_profile['field'] == 'software_engineering':
                if best.get('tech_stack_alignment', 0) < 5:
                    print("This offer had poor tech stack alignment, which is highly valued in software engineering")
                elif best.get('work_life_balance', 0) < 5 and user_preferences.get('work_life_balance_weight', 0) > 3:
                    print("This offer had poor work-life balance, which you indicated was important")
            elif user_profile['field'] == 'marketing':
                if best.get('company_reputation', 0) < 5 and user_preferences.get('company_reputation_weight', 0) > 3:
                    print("This offer had low company reputation, which is crucial in marketing roles")
    
    # Final insights with field-specific context
    final_insights = ost.get_search_insights()
    
    print("\n=== FINAL INSIGHTS WITH SEMANTIC UNDERSTANDING ===")
    print(f"Field: {user_profile['field']}")
    print(f"Experience level: {user_profile['experience_level']}")
    print(f"Location: {user_profile['location']}")
    print(f"Market condition: {final_insights['market_condition']:.2f}")
    
    # Field-specific salary expectations
    print(f"Salary expectations for {user_profile['experience_level']} {user_profile['field']} in {user_profile['location']}:")
    print(f"  ${final_insights['expected_salary_range'][0]:,.2f} - ${final_insights['expected_salary_range'][1]:,.2f}")
    
    # Most valued job attributes with semantic descriptions
    print("\nMost valued job attributes (from observed data):")
    for attr, value in final_insights['top_attributes'].items():
        var_type = VARIABLE_TYPES.get(attr)
        if var_type:
            print(f"  {attr}: {value:.1f}/10 ({var_type.description})")
    
    # Threshold evolution with semantic context
    print("\nThreshold evolution with employment context:")
    print(f"  Initial threshold: {ost.reservation_utilities[0]:.2f}")
    print(f"  Mid-search threshold: {ost.reservation_utilities[50]:.2f}")
    print(f"  Final threshold: {ost.reservation_utilities[-1]:.2f}")
    
    if user_profile['employment_status'] == 'unemployed':
        print("  Note: Thresholds decrease more rapidly for unemployed job seekers")
    else:
        print("  Note: Being employed allowed for more selective thresholds")
    
    # Field-specific fatigue factors
    if user_profile['field'] == 'software_engineering':
        print(f"\nTech interview fatigue factor: {ost.fatigue_factor:.2f}")
        print("  Note: Tech interviews often involve multiple rounds and technical assessments")
    else:
        print(f"\nInterview fatigue factor: {ost.fatigue_factor:.2f}")
    
    print(f"Total rejections made: {ost.rejections_made}")
    
    # Recommend creating a .env file if MongoDB connection failed
    if not hasattr(mongodb_util, 'connected') or not mongodb_util.connected:
        print("\nTIP: To use MongoDB for data persistence, create a .env file with:")
        print("MONGODB_URI=your_mongodb_connection_string")
    
    # Visualize the results
    print("\nGenerating visualization of the OST algorithm results...")
    visualize_ost_results(ost, offers_data)
    
    # Remove the separate call to visualize_decision_process to avoid duplicate visualization

# Run the simulation or evaluation based on command line args
if __name__ == "__main__":
    run_semantic_ost_simulation()