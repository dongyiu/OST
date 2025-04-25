# application_process.py

from typing import TypedDict, List, Dict, Any, Callable, Optional, Union, Type, Tuple
import json
import time
import random
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import re
from langgraph.graph import StateGraph
import google.generativeai as genai
# Import from main.py (the refactored code)
from main import (
    BaseWorkflowState, 
    BaseWorkflowBuilder,
    WorkflowConfig,
    WorkflowRegistry, 
    MetaReasoningAgent,
    AIModelInterface,
    StorageInterface,
    WorkflowExecutor,
    VariableSchema
)

###########################################
# APPLICATION PROCESSING DATA STRUCTURES
###########################################

class ApplicationState(BaseWorkflowState):
    """State structure for the application processing workflow"""
    application_data: Dict  # Raw application data
    enriched_data: Dict  # Data after enrichment
    user_preferences: Dict  # User preference variables
    job_quality_metrics: Dict  # Assessed job quality metrics
    reasoning_traces: List[Dict]  # Traces of reasoning steps
    knowledge_gaps: Dict  # Identified information gaps that need research

genai.configure(api_key="AIzaSyBGkvO4s4WlZ2p3bwZwxWxkKnueRm6npRU")
class JobQualityMetric:
    """Definition of a job quality metric"""
    def __init__(self, 
                 key: str, 
                 name: str,
                 description: str,
                 min_val: float, 
                 max_val: float, 
                 weight: float = 1.0,
                 extraction_prompt: str = None,
                 required_info: List[str] = None):
        self.key = key
        self.name = name
        self.description = description
        self.min = min_val
        self.max = max_val
        self.weight = weight  # Default importance weight
        self.extraction_prompt = extraction_prompt or f"Extract the {name} ({description}) from the job data."
        # New: List of information fields required to assess this metric
        self.required_info = required_info or []
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "min": self.min,
            "max": self.max,
            "weight": self.weight,
            "extraction_prompt": self.extraction_prompt,
            "required_info": self.required_info
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'JobQualityMetric':
        """Create from dictionary"""
        return cls(
            key=data["key"],
            name=data.get("name", data["key"]),
            description=data.get("description", ""),
            min_val=data["min"],
            max_val=data["max"],
            weight=data.get("weight", 1.0),
            extraction_prompt=data.get("extraction_prompt"),
            required_info=data.get("required_info", [])
        )


class ApplicationProcessingConfig:
    """Configuration for application processing workflow"""
    def __init__(self,
                 workflow_id: str = "application_processing",
                 storage_collection: str = "applications",
                 standard_metrics: List[Dict] = None,
                 confidence_threshold: float = 0.7,
                 max_reasoning_attempts: int = 3,
                 max_research_attempts: int = 2):
        self.workflow_id = workflow_id
        self.storage_collection = storage_collection
        self.standard_metrics = standard_metrics or []
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_attempts = max_reasoning_attempts
        self.max_research_attempts = max_research_attempts  # New: research attempt limit
        
        # Initialize standard metrics if none provided
        if not self.standard_metrics:
            self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """Initialize standard job quality metrics with required information fields"""
        self.standard_metrics = [
            JobQualityMetric(
                key="base_salary",
                name="Base Salary",
                description="Standardized annual value in USD",
                min_val=0,
                max_val=10,
                weight=1.0,
                extraction_prompt="Extract and evaluate the base annual salary in USD from the job data. Rate it on a scale of 1-10 based on how competitive it is.",
                required_info=["salary_data", "industry_avg_salary", "location_factor"]
            ).to_dict(),
            JobQualityMetric(
                key="total_compensation",
                name="Total Compensation",
                description="Including benefits, bonuses, stock options in USD",
                min_val=0,
                max_val=10,
                weight=1.0,
                extraction_prompt="Calculate and rate the total annual compensation including base salary, bonuses, benefits, and stock options on a scale of 1-10.",
                required_info=["salary_data", "benefits_data", "equity_data", "industry_comp_data"]
            ).to_dict(),
            JobQualityMetric(
                key="job_match_score",
                name="Job Match Score",
                description="How well job matches user's skills and experience",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well the job requirements match the user's skills and experience on a scale of 1-10.",
                required_info=["job_requirements", "user_skills", "user_experience"]
            ).to_dict(),
            JobQualityMetric(
                key="company_rating",
                name="Company Rating",
                description="Based on reputation, stability, and market position",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the company based on its reputation, financial stability, and market position on a scale of 1-10.",
                required_info=["company_data", "company_reviews", "company_financials", "industry_standing"]
            ).to_dict(),
            JobQualityMetric(
                key="work_life_balance",
                name="Work-Life Balance Score",
                description="Expected work-life balance based on company culture and role",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the expected work-life balance for this role and company on a scale of 1-10.",
                required_info=["company_culture", "work_hours", "remote_policy", "employee_reviews"]
            ).to_dict(),
            JobQualityMetric(
                key="career_growth",
                name="Career Growth Potential",
                description="Advancement opportunities and skill development",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the potential for career advancement and skill development in this role on a scale of 1-10.",
                required_info=["company_growth", "promotion_track", "learning_opportunities", "company_size"]
            ).to_dict(),
            JobQualityMetric(
                key="role_alignment",
                name="Role Alignment",
                description="Fit with user's career goals and interests",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well this role aligns with the user's stated career goals and interests on a scale of 1-10.",
                required_info=["job_responsibilities", "user_career_goals", "job_field", "job_level"]
            ).to_dict(),
            JobQualityMetric(
                key="team_manager_quality",
                name="Team/Manager Quality",
                description="Based on interview impressions and information",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the quality of the team and management based on interview impressions and available information on a scale of 1-10.",
                required_info=["team_structure", "manager_info", "interview_feedback", "management_style"]
            ).to_dict(),
            JobQualityMetric(
                key="culture_fit",
                name="Culture Fit",
                description="Compatibility with company culture and values",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well the company culture and values align with the user's preferences on a scale of 1-10.",
                required_info=["company_culture", "company_values", "user_preferences", "workplace_style"]
            ).to_dict(),
            JobQualityMetric(
                key="commute_remote_score",
                name="Commute/Remote Score",
                description="Location convenience and remote work options",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the convenience of the job location and remote work options on a scale of 1-10.",
                required_info=["job_location", "commute_info", "remote_policy", "user_location"]
            ).to_dict(),
        ]
    
    def get_metric(self, key: str) -> Optional[Dict]:
        """Get a specific metric by key"""
        for metric in self.standard_metrics:
            if metric["key"] == key:
                return metric
        return None
    
    def add_metric(self, metric: JobQualityMetric):
        """Add a custom job quality metric"""
        self.standard_metrics.append(metric.to_dict())
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "storage_collection": self.storage_collection,
            "standard_metrics": self.standard_metrics,
            "confidence_threshold": self.confidence_threshold,
            "max_reasoning_attempts": self.max_reasoning_attempts,
            "max_research_attempts": self.max_research_attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ApplicationProcessingConfig':
        """Create from dictionary"""
        return cls(
            workflow_id=data.get("workflow_id", "application_processing"),
            storage_collection=data.get("storage_collection", "applications"),
            standard_metrics=data.get("standard_metrics", []),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            max_reasoning_attempts=data.get("max_reasoning_attempts", 3),
            max_research_attempts=data.get("max_research_attempts", 2)
        )


###########################################
# APPLICATION PROCESSING AGENTS
###########################################

class EmailParserAgent:
    """
    Agent for parsing application information from emails.
    This is a placeholder implementation.
    """
    def __init__(self, ai_model: AIModelInterface):
        self.ai_model = ai_model
    
    def parse_email(self, email_content: str) -> Dict:
        """Parse job application data from email content"""
        # In a real implementation, this would use AI to extract structured data
        # For this example, we'll return a simple placeholder
        return {
            "parsed_date": datetime.now().isoformat(),
            "source": "email",
            "raw_content": email_content,
            "status": "parsed"
        }


class DynamicResearchAgent:
    """
    Enhanced agent for dynamically researching any type of information needed.
    Uses AI to simulate web searches, data analysis, and information synthesis.
    """
    def __init__(self, ai_model: AIModelInterface, meta_agent: Optional[MetaReasoningAgent] = None):
        self.ai_model = ai_model
        self.meta_agent = meta_agent
        
        # Track attempted research topics to avoid redundant searches
        self.researched_topics = set()
        
        # Define research strategies for different information types
        self.research_strategies = {
            "company_data": self._research_company_basic,
            "company_reviews": self._research_company_reviews,
            "company_culture": self._research_company_culture,
            "company_financials": self._research_company_financials,
            "company_growth": self._research_company_growth,
            "industry_avg_salary": self._research_industry_salary,
            "industry_standing": self._research_industry_standing,
            "industry_comp_data": self._research_industry_compensation,
            "remote_policy": self._research_remote_policy,
            "job_requirements": self._extract_job_requirements,
            "salary_data": self._extract_salary_data,
            "benefits_data": self._extract_benefits_data,
            "equity_data": self._extract_equity_data,
            "job_responsibilities": self._extract_job_responsibilities,
            "team_structure": self._research_team_structure,
            "location_factor": self._research_location_factor,
            "commute_info": self._research_commute_info,
            "promotion_track": self._research_promotion_track,
            "learning_opportunities": self._research_learning_opportunities,
            "management_style": self._research_management_style,
            "company_values": self._research_company_values,
            "work_hours": self._research_work_hours,
            "employee_reviews": self._research_employee_reviews
        }
        
        # Research prompts for different information types
        self.research_prompts = {
            "company_data": "Provide factual information about {company_name}. Include company size, industry, founding year, headquarters location, and notable products or services.",
            "company_reviews": "Based on typical employee reviews, what are the pros and cons of working at {company_name}?",
            "company_culture": "Describe the typical company culture at {company_name}, including work environment, values, and management approach.",
            "company_financials": "Provide information about {company_name}'s financial health, including revenue trends, profitability, and market position.",
            "company_growth": "How has {company_name} been growing recently? Include information about expansion, hiring trends, and business development.",
            "industry_avg_salary": "What is the average salary for {position} roles in the {industry} industry, particularly in the {location} area?",
            "industry_standing": "How does {company_name} compare to competitors in the {industry} industry in terms of market share, innovation, and reputation?",
            "industry_comp_data": "What is the typical total compensation package for {position} roles in {industry}, including benefits, bonuses, and equity?",
            "remote_policy": "What is {company_name}'s policy on remote work, flexible schedules, and work-from-home options?",
            "job_requirements": "Based on the job description, what are the key skills, qualifications, and experience requirements for this {position} role?",
            "salary_data": "Based on the information provided, what is the salary range for this {position} role at {company_name}?",
            "benefits_data": "What benefits are typically offered by {company_name} or mentioned in the job posting?",
            "equity_data": "What equity or stock option benefits are mentioned or typically offered for {position} roles at {company_name}?",
            "job_responsibilities": "What are the primary responsibilities and day-to-day tasks for this {position} role based on the job description?",
            "team_structure": "What is the typical team structure for {position} roles at {company_name}? Include reporting relationships and team size if available.",
            "location_factor": "How does the cost of living and job market in {location} compare to national averages?",
            "commute_info": "What is the commute situation for working at {company_name} in {location}?",
            "promotion_track": "What is the typical career progression path for someone in a {position} role at {company_name}?",
            "learning_opportunities": "What professional development and learning opportunities are typically available at {company_name}?",
            "management_style": "What is the management style and approach typically found at {company_name}?",
            "company_values": "What are the stated or known company values and mission of {company_name}?",
            "work_hours": "What are the typical work hours, overtime expectations, and work-life balance for {position} roles at {company_name}?",
            "employee_reviews": "What do employee reviews say about working at {company_name}, particularly regarding work-life balance and management?"
        }
    
    def research_all_gaps(self, application_data: Dict, knowledge_gaps: Dict) -> Dict:
        """
        Research all identified knowledge gaps and return compiled findings
        
        Args:
            application_data: Current application data
            knowledge_gaps: Dictionary of information gaps to fill
            
        Returns:
            Dictionary with researched information
        """
        research_results = {}
        
        company_name = application_data.get("company", "Unknown Company")
        position = application_data.get("position", "Unknown Position")
        location = application_data.get("location", "Unknown Location")
        industry = application_data.get("industry", "")
        
        # If industry is not in application data, try to determine it
        if not industry and company_name and company_name != "Unknown Company":
            industry_prompt = f"What industry is {company_name} primarily in? Provide just the industry name, no explanation."
            try:
                response = self.ai_model.generate_content(industry_prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                industry = response.text.strip()
                print(f"🔍 Identified industry: {industry}")
                research_results["industry"] = industry
            except Exception as e:
                print(f"❌ Error determining industry: {str(e)}")
                industry = "Unknown Industry"
        
        print(f"🔍 RESEARCHING INFORMATION GAPS")
        print(f"Company: {company_name}")
        print(f"Position: {position}")
        print(f"Location: {location}")
        print(f"Industry: {industry}")
        
        # Create context for research
        context = {
            "company_name": company_name,
            "position": position,
            "location": location,
            "industry": industry,
            "job_description": application_data.get("description", ""),
            "salary_info": application_data.get("salary", "")
        }
        
        # Prioritize gaps by importance
        priority_order = [
            "company_data",  # Basic company info first
            "salary_data",   # Salary is high priority
            "job_requirements",  # Job requirements are important
            "industry_avg_salary",  # Industry context matters
            "location_factor",  # Location affects salary assessment
            "remote_policy",  # Remote work is often a key factor
        ]
        
        # Create sorted list of gaps
        all_gaps = list(knowledge_gaps.keys())
        
        # Sort by priority first, then keep original order for the rest
        sorted_gaps = sorted(all_gaps, key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        # Research each gap
        for gap in sorted_gaps:
            if gap in research_results or gap not in self.research_strategies:
                continue
                
            # Check if we've already researched this topic
            if gap in self.researched_topics:
                print(f"⏩ Already researched {gap}, using cached findings")
                continue
                
            print(f"\n🔍 Researching: {gap}")
            
            # Get the research strategy for this gap
            research_func = self.research_strategies[gap]
            
            try:
                # Execute the research
                result = research_func(context)
                if result:
                    research_results[gap] = result
                    self.researched_topics.add(gap)
                    print(f"✅ Successfully gathered information on {gap}: {len(str(result))} characters of data")
                    if isinstance(result, dict) and 'data' in result:
                        preview = str(result['data'])[:100] + "..." if len(str(result['data'])) > 100 else str(result['data'])
                        print(f"   Preview: {preview}")
            except Exception as e:
                print(f"❌ Error researching {gap}: {str(e)}")
                
        print(f"\n✅ Research completed. Found information for {len(research_results)} topics")
        return research_results
    
    def _research_company_basic(self, context: Dict) -> Dict:
        """Research basic company information"""
        company_name = context["company_name"]
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_data"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company basic info: {str(e)}")
            return None
    
    def _research_company_reviews(self, context: Dict) -> Dict:
        """Research company reviews"""
        company_name = context["company_name"]
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_reviews"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company reviews: {str(e)}")
            return None
    
    def _research_company_culture(self, context: Dict) -> Dict:
        """Research company culture"""
        company_name = context["company_name"]
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_culture"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company culture: {str(e)}")
            return None
    
    def _research_company_financials(self, context: Dict) -> Dict:
        """Research company financial health"""
        company_name = context["company_name"]
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_financials"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company financials: {str(e)}")
            return None
    
    def _research_company_growth(self, context: Dict) -> Dict:
        """Research company growth trajectory"""
        company_name = context["company_name"]
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_growth"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company growth: {str(e)}")
            return None
    
    def _research_industry_salary(self, context: Dict) -> Dict:
        """Research industry average salary"""
        position = context["position"]
        industry = context.get("industry", "")
        location = context.get("location", "")
        
        if position == "Unknown Position":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["industry_avg_salary"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "position": position,
                "industry": industry,
                "location": location,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching industry salary: {str(e)}")
            return None
    
    def _research_industry_standing(self, context: Dict) -> Dict:
        """Research company's standing in the industry"""
        company_name = context["company_name"]
        industry = context.get("industry", "")
        
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["industry_standing"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "industry": industry,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching industry standing: {str(e)}")
            return None
    
    def _research_industry_compensation(self, context: Dict) -> Dict:
        """Research industry compensation packages"""
        position = context["position"]
        industry = context.get("industry", "")
        
        if position == "Unknown Position":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["industry_comp_data"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "position": position,
                "industry": industry,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching industry compensation: {str(e)}")
            return None
    
    def _research_remote_policy(self, context: Dict) -> Dict:
        """Research remote work policy"""
        company_name = context["company_name"]
        position = context["position"]
        job_description = context.get("job_description", "")
        
        # Check if job description already mentions remote work
        if job_description and any(term in job_description.lower() for term in ["remote", "work from home", "wfh", "hybrid"]):
            # Extract from job description instead of researching
            prompt = f"""
            Extract information about remote work policy from this job description:
            "{job_description[:1000]}..."
            
            Is this job remote, hybrid, or in-office? What flexibility is offered?
            Provide a concise assessment based only on the description.
            """
            
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                    
                return {
                    "source": "job_description_analysis",
                    "company_name": company_name,
                    "position": position,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"❌ Error extracting remote policy: {str(e)}")
        
        # If not found in description or no description, research company policy
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["remote_policy"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching remote policy: {str(e)}")
            return None
    
    def _extract_job_requirements(self, context: Dict) -> Dict:
        """Extract job requirements from description"""
        job_description = context.get("job_description", "")
        position = context["position"]
        
        if not job_description:
            return None
            
        prompt = f"""
        Extract the key job requirements from this job description:
        "{job_description[:1500]}..."
        
        Format as JSON with these fields:
        - required_skills: List of required technical skills/tools
        - preferred_skills: List of preferred or nice-to-have skills
        - experience_years: Number of years of experience required
        - education: Education requirements (degree level)
        - certifications: Any required or preferred certifications
        - soft_skills: Required soft skills or traits
        
        If a field isn't specified, use null. Be concise but complete.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            # Try to parse as JSON
            try:
                # Extract JSON using regex in case there's explanatory text
                import json
                import re
                
                text = response.text
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    requirements = json.loads(json_str)
                    
                    return {
                        "source": "job_description_analysis",
                        "position": position,
                        "data": requirements,
                        "raw_analysis": response.text.strip(),
                        "researched_date": datetime.now().isoformat()
                    }
                else:
                    # Return raw text if JSON extraction fails
                    return {
                        "source": "job_description_analysis",
                        "position": position,
                        "data": response.text.strip(),
                        "researched_date": datetime.now().isoformat()
                    }
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {
                    "source": "job_description_analysis",
                    "position": position,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error extracting job requirements: {str(e)}")
            return None
    
    def _extract_salary_data(self, context: Dict) -> Dict:
        """Extract structured salary data"""
        salary_info = context.get("salary_info", "")
        position = context["position"]
        
        if not salary_info:
            # If no explicit salary info, try to extract from job description
            job_description = context.get("job_description", "")
            if not job_description:
                return None
                
            prompt = f"""
            Extract any salary or compensation information from this job description:
            "{job_description[:1500]}..."
            
            If no specific numbers are mentioned, respond with "No salary information provided in the description."
            """
            
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                
                extracted_text = response.text.strip()
                if "No salary information" in extracted_text:
                    return None
                    
                salary_info = extracted_text
            except Exception as e:
                print(f"❌ Error extracting salary from description: {str(e)}")
                return None
        
        prompt = f"""
        Extract structured salary information from this text:
        "{salary_info}"
        
        Format as JSON with these fields:
        - base_salary_min: Minimum annual base salary in USD (numeric, no currency symbols)
        - base_salary_max: Maximum annual base salary in USD (numeric, no currency symbols)
        - has_range: Boolean indicating if a salary range is provided
        - has_bonus: Boolean indicating if bonus is mentioned
        - bonus_details: Details about bonus structure
        - has_equity: Boolean indicating if equity is mentioned
        - equity_details: Details about equity
        
        Convert any hourly rates to annual assuming 40hrs/week, 50 weeks/year.
        Convert any non-USD currencies to USD using typical exchange rates.
        If a value is unknown, use null.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            # Try to parse as JSON
            try:
                # Extract JSON using regex in case there's explanatory text
                import json
                import re
                
                text = response.text
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    salary_data = json.loads(json_str)
                    
                    return {
                        "source": "salary_analysis",
                        "position": position,
                        "data": salary_data,
                        "raw_text": salary_info,
                        "researched_date": datetime.now().isoformat()
                    }
                else:
                    # Return raw text if JSON extraction fails
                    return {
                        "source": "salary_analysis",
                        "position": position,
                        "data": response.text.strip(),
                        "raw_text": salary_info,
                        "researched_date": datetime.now().isoformat()
                    }
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {
                    "source": "salary_analysis",
                    "position": position,
                    "data": response.text.strip(),
                    "raw_text": salary_info,
                    "researched_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error extracting structured salary data: {str(e)}")
            return None
    
    def _extract_benefits_data(self, context: Dict) -> Dict:
        """Extract benefits information"""
        job_description = context.get("job_description", "")
        position = context["position"]
        company_name = context["company_name"]
        
        if not job_description:
            # If no job description, research typical benefits for the company
            if company_name == "Unknown Company":
                return None
                
            prompt = f"""
            What benefits are typically offered by {company_name}? Include health insurance, retirement, 
            vacation, parental leave, and any other notable benefits. If you're uncertain, indicate that.
            """
            
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                    
                return {
                    "source": "ai_research",
                    "company_name": company_name,
                    "position": position,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"❌ Error researching company benefits: {str(e)}")
                return None
        
        # Extract from job description
        prompt = f"""
        Extract all benefits mentioned in this job description:
        "{job_description[:1500]}..."
        
        Format as JSON with these fields:
        - health_insurance: Details about health benefits
        - retirement: Details about retirement benefits (401k, pension, etc.)
        - vacation: Paid time off or vacation policy
        - additional_benefits: List of any other benefits mentioned
        
        If a benefit isn't mentioned, use null for that field.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            # Try to parse as JSON
            try:
                # Extract JSON using regex in case there's explanatory text
                import json
                import re
                
                text = response.text
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    benefits_data = json.loads(json_str)
                    
                    return {
                        "source": "job_description_analysis",
                        "position": position,
                        "data": benefits_data,
                        "researched_date": datetime.now().isoformat()
                    }
                else:
                    # Return raw text if JSON extraction fails
                    return {
                        "source": "job_description_analysis",
                        "position": position,
                        "data": response.text.strip(),
                        "researched_date": datetime.now().isoformat()
                    }
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {
                    "source": "job_description_analysis",
                    "position": position,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error extracting benefits data: {str(e)}")
            return None
    
    def _extract_equity_data(self, context: Dict) -> Dict:
        """Extract equity information"""
        job_description = context.get("job_description", "")
        salary_info = context.get("salary_info", "")
        position = context["position"]
        company_name = context["company_name"]
        
        # Combine job description and salary info for analysis
        combined_text = f"{job_description}\n{salary_info}"
        
        if not combined_text.strip():
            # If no text to analyze, research typical equity for the company/position
            if company_name == "Unknown Company" or position == "Unknown Position":
                return None
                
            prompt = f"""
            What type of equity or stock options are typically offered for {position} roles at {company_name}? 
            If you're uncertain about this specific company, what is typical for this position in this industry?
            """
            
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                    
                return {
                    "source": "ai_research",
                    "company_name": company_name,
                    "position": position,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"❌ Error researching typical equity: {str(e)}")
                return None
        
        # Extract from combined text
        prompt = f"""
        Extract any information about equity, stock options, RSUs, or stock-based compensation from this text:
        "{combined_text[:1500]}..."
        
        Format as JSON with these fields:
        - has_equity: Boolean indicating if equity is mentioned
        - equity_type: Type of equity (options, RSUs, etc.)
        - equity_amount: Amount or percentage if mentioned
        - vesting_period: Vesting period if mentioned
        - additional_details: Any other relevant equity details
        
        If equity isn't mentioned or a field is unknown, use null.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            # Try to parse as JSON
            try:
                # Extract JSON using regex in case there's explanatory text
                import json
                import re
                
                text = response.text
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    equity_data = json.loads(json_str)
                    
                    return {
                        "source": "text_analysis",
                        "position": position,
                        "company_name": company_name,
                        "data": equity_data,
                        "researched_date": datetime.now().isoformat()
                    }
                else:
                    # Return raw text if JSON extraction fails
                    return {
                        "source": "text_analysis",
                        "position": position,
                        "company_name": company_name,
                        "data": response.text.strip(),
                        "researched_date": datetime.now().isoformat()
                    }
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {
                    "source": "text_analysis",
                    "position": position,
                    "company_name": company_name,
                    "data": response.text.strip(),
                    "researched_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error extracting equity data: {str(e)}")
            return None
    
    def _extract_job_responsibilities(self, context: Dict) -> Dict:
        """Extract job responsibilities"""
        job_description = context.get("job_description", "")
        position = context["position"]
        
        if not job_description:
            return None
            
        prompt = f"""
        Extract the primary job responsibilities and day-to-day tasks from this job description:
        "{job_description[:1500]}..."
        
        Format as a list of key responsibilities with brief descriptions. Focus only on the duties/tasks,
        not the requirements or qualifications.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "job_description_analysis",
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error extracting job responsibilities: {str(e)}")
            return None
    
    def _research_team_structure(self, context: Dict) -> Dict:
        """Research team structure"""
        company_name = context["company_name"]
        position = context["position"]
        job_description = context.get("job_description", "")
        
        # First try to extract from job description if available
        if job_description:
            prompt = f"""
            Extract information about team structure, reporting relationships, and team size from this job description:
            "{job_description[:1500]}..."
            
            If this information isn't in the description, indicate that.
            """
            
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                    
                if "isn't in the description" not in response.text and "not mentioned" not in response.text:
                    return {
                        "source": "job_description_analysis",
                        "position": position,
                        "company_name": company_name,
                        "data": response.text.strip(),
                        "researched_date": datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"❌ Error extracting team structure from description: {str(e)}")
        
        # If not found in description or no description, research typical structure
        if company_name == "Unknown Company" or position == "Unknown Position":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["team_structure"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching team structure: {str(e)}")
            return None
    
    def _research_location_factor(self, context: Dict) -> Dict:
        """Research location cost of living and job market"""
        location = context.get("location", "")
        position = context["position"]
        
        if location == "Unknown Location" or location == "Remote":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["location_factor"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "location": location,
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching location factor: {str(e)}")
            return None
    
    def _research_commute_info(self, context: Dict) -> Dict:
        """Research commute information"""
        location = context.get("location", "")
        company_name = context["company_name"]
        
        if location == "Unknown Location" or location == "Remote" or company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["commute_info"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "location": location,
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching commute info: {str(e)}")
            return None
    
    def _research_promotion_track(self, context: Dict) -> Dict:
        """Research typical promotion track"""
        company_name = context["company_name"]
        position = context["position"]
        
        if company_name == "Unknown Company" or position == "Unknown Position":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["promotion_track"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching promotion track: {str(e)}")
            return None
    
    def _research_learning_opportunities(self, context: Dict) -> Dict:
        """Research learning opportunities"""
        company_name = context["company_name"]
        
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["learning_opportunities"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching learning opportunities: {str(e)}")
            return None
    
    def _research_management_style(self, context: Dict) -> Dict:
        """Research management style"""
        company_name = context["company_name"]
        
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["management_style"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching management style: {str(e)}")
            return None
    
    def _research_company_values(self, context: Dict) -> Dict:
        """Research company values"""
        company_name = context["company_name"]
        
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["company_values"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching company values: {str(e)}")
            return None
    
    def _research_work_hours(self, context: Dict) -> Dict:
        """Research typical work hours"""
        company_name = context["company_name"]
        position = context["position"]
        
        if company_name == "Unknown Company" or position == "Unknown Position":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["work_hours"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "position": position,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching work hours: {str(e)}")
            return None
    
    def _research_employee_reviews(self, context: Dict) -> Dict:
        """Research employee reviews"""
        company_name = context["company_name"]
        
        if company_name == "Unknown Company":
            return None
            
        # Format the prompt using the template
        prompt = self.research_prompts["employee_reviews"].format(**context)
        
        try:
            response = self.ai_model.generate_content(prompt)
            if self.meta_agent:
                self.meta_agent.increment()
                
            return {
                "source": "ai_research",
                "company_name": company_name,
                "data": response.text.strip(),
                "researched_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error researching employee reviews: {str(e)}")
            return None


class DataEnrichmentAgent:
    """
    Agent for enriching application data with additional information.
    Now integrates with DynamicResearchAgent to gather any missing information.
    """
    def __init__(self, 
                 ai_model: AIModelInterface,
                 research_agent: DynamicResearchAgent = None,
                 meta_agent: MetaReasoningAgent = None):
        self.ai_model = ai_model
        self.research_agent = research_agent or DynamicResearchAgent(ai_model, meta_agent)
        self.meta_agent = meta_agent
    
    def enrich_application_data(self, application_data: Dict) -> Dict:
        """Enrich application data with additional information"""
        enriched_data = application_data.copy()
        
        # If meta_agent is available, reset consecutive counter
        if self.meta_agent:
            self.meta_agent.reset_consecutive()
        
        print("🔍 ENRICHING APPLICATION DATA")
        
        # Basic data extraction from job description
        if "description" in application_data and application_data["description"]:
            print("📄 Extracting basic information from job description")
            
            # Extract job requirements if not already present
            if "job_requirements" not in enriched_data:
                requirements_research = self.research_agent._extract_job_requirements({
                    "job_description": application_data["description"],
                    "position": application_data.get("position", "Unknown Position")
                })
                
                if requirements_research:
                    enriched_data["job_requirements"] = requirements_research
                    print("✅ Job requirements extracted successfully")
        
        # Extract and structure salary information if not already present
        if "salary" in application_data and application_data["salary"] and "structured_salary" not in enriched_data:
            print("💰 Extracting structured salary information")
            
            salary_research = self.research_agent._extract_salary_data({
                "salary_info": application_data["salary"],
                "position": application_data.get("position", "Unknown Position")
            })
            
            if salary_research:
                enriched_data["structured_salary"] = salary_research
                print("✅ Salary information extracted successfully")
        
        # Identify gaps for more targeted research
        basic_research_needs = [
            "company_data",  # Basic company information
            "job_requirements",  # Job requirements if not already extracted
            "salary_data"  # Salary data if not already extracted
        ]
        
        research_context = {
            "company_name": application_data.get("company", "Unknown Company"),
            "position": application_data.get("position", "Unknown Position"),
            "location": application_data.get("location", "Unknown Location"),
            "job_description": application_data.get("description", ""),
            "salary_info": application_data.get("salary", "")
        }
        
        # Perform basic research for key fields
        print("\n🔍 CONDUCTING BASIC RESEARCH")
        for need in basic_research_needs:
            # Skip if we already have this data
            if need == "job_requirements" and "job_requirements" in enriched_data:
                continue
            if need == "salary_data" and "structured_salary" in enriched_data:
                continue
                
            research_func = getattr(self.research_agent, f"_research_{need}", None) or getattr(self.research_agent, f"_extract_{need}", None)
            
            if research_func:
                print(f"Researching: {need}")
                try:
                    result = research_func(research_context)
                    if result:
                        # Store with standardized names
                        if need == "company_data":
                            enriched_data["company_info"] = result
                        elif need == "job_requirements" and "job_requirements" not in enriched_data:
                            enriched_data["job_requirements"] = result
                        elif need == "salary_data" and "structured_salary" not in enriched_data:
                            enriched_data["structured_salary"] = result
                            
                        print(f"✅ Successfully gathered {need}")
                except Exception as e:
                    print(f"❌ Error during {need} research: {str(e)}")
        
        # Add metadata
        enriched_data["enrichment_date"] = datetime.now().isoformat()
        enriched_data["enrichment_status"] = "completed"
        
        print("✅ Initial data enrichment completed")
        
        return enriched_data


class JobQualityAssessmentAgent:
    """
    Enhanced agent for assessing job quality metrics using direct reasoning and dynamic research.
    """
    def __init__(self, 
                 ai_model: AIModelInterface,
                 config: ApplicationProcessingConfig,
                 research_agent: DynamicResearchAgent,
                 meta_agent: Optional[MetaReasoningAgent] = None):
        self.ai_model = ai_model
        self.config = config
        self.research_agent = research_agent
        self.meta_agent = meta_agent
        self.verbose = True  # Enable detailed output
    
    def identify_knowledge_gaps(self, 
                               metrics_to_assess: List[str], 
                               application_data: Dict) -> Dict[str, bool]:
        """
        Identify missing information needed for accurate assessment
        
        Args:
            metrics_to_assess: List of metric keys to assess
            application_data: Current application and enriched data
            
        Returns:
            Dictionary of information gaps with boolean priority
        """
        knowledge_gaps = {}
        
        print("\n🔍 IDENTIFYING KNOWLEDGE GAPS")
        
        # Get all required information fields for the metrics we need to assess
        for metric_key in metrics_to_assess:
            metric_dict = self.config.get_metric(metric_key)
            if not metric_dict:
                continue
                
            required_info = metric_dict.get("required_info", [])
            for info_key in required_info:
                # Check if we have this information already
                if self._check_info_available(info_key, application_data):
                    if self.verbose:
                        print(f"✅ {info_key} is available")
                    continue
                
                # This is a gap we need to fill
                # Determine priority based on which metric needs it
                if metric_key in ["base_salary", "total_compensation", "job_match_score", "company_rating"]:
                    # High priority for core metrics
                    knowledge_gaps[info_key] = True
                    print(f"❗ Critical gap identified: {info_key} (needed for {metric_key})")
                else:
                    # Standard priority
                    knowledge_gaps[info_key] = knowledge_gaps.get(info_key, False)
                    if self.verbose:
                        print(f"🔍 Gap identified: {info_key} (needed for {metric_key})")
        
        # Sort gaps by priority (critical first)
        sorted_gaps = sorted(knowledge_gaps.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Identified {len(sorted_gaps)} knowledge gaps to fill")
        return dict(sorted_gaps)
    
    def _check_info_available(self, info_key: str, application_data: Dict) -> bool:
        """
        Check if required information is available in application data
        
        Args:
            info_key: Information field to check
            application_data: Current application data
            
        Returns:
            Boolean indicating if information is available
        """
        # Map info keys to paths in application data where they might be found
        info_paths = {
            "salary_data": ["structured_salary", "salary"],
            "industry_avg_salary": ["industry_data", "researched_data.industry_avg_salary", "enriched_data.industry_avg_salary"],
            "location_factor": ["location_data", "researched_data.location_factor"],
            "benefits_data": ["benefits", "structured_benefits", "researched_data.benefits_data"],
            "equity_data": ["equity", "structured_equity", "researched_data.equity_data"],
            "job_requirements": ["job_requirements", "extracted_job_info.requirements"],
            "user_skills": ["user_preferences.skills", "user_data.skills"],
            "user_experience": ["user_preferences.experience", "user_data.experience"],
            "company_data": ["company_info", "researched_data.company_data"],
            "company_reviews": ["company_reviews", "researched_data.company_reviews"],
            "company_financials": ["company_financials", "researched_data.company_financials"],
            "industry_standing": ["industry_standing", "researched_data.industry_standing"],
            "company_culture": ["company_culture", "researched_data.company_culture", "extracted_job_info.culture"],
            "work_hours": ["work_hours", "extracted_job_info.hours", "researched_data.work_hours"],
            "remote_policy": ["remote_policy", "extracted_job_info.remote", "researched_data.remote_policy"],
            "employee_reviews": ["employee_reviews", "researched_data.employee_reviews"],
            "company_growth": ["company_growth", "researched_data.company_growth"],
            "promotion_track": ["promotion_track", "researched_data.promotion_track", "career_path"],
            "learning_opportunities": ["learning_opportunities", "professional_development", "researched_data.learning_opportunities"],
            "company_size": ["company_size", "company_info.company_size", "researched_data.company_size"],
            "job_responsibilities": ["job_responsibilities", "extracted_job_info.responsibilities", "researched_data.job_responsibilities"],
            "user_career_goals": ["user_preferences.career_goals", "user_data.career_goals"],
            "job_field": ["job_field", "industry", "field"],
            "job_level": ["job_level", "seniority", "level"],
            "team_structure": ["team_structure", "researched_data.team_structure"],
            "manager_info": ["manager_info", "researched_data.manager_info"],
            "interview_feedback": ["interview_feedback", "interview_notes"],
            "management_style": ["management_style", "researched_data.management_style"],
            "company_values": ["company_values", "researched_data.company_values"],
            "user_preferences": ["user_preferences"],
            "workplace_style": ["workplace_style", "researched_data.workplace_style"],
            "job_location": ["location", "job_location"],
            "commute_info": ["commute_info", "researched_data.commute_info"],
            "user_location": ["user_location", "user_preferences.location", "user_data.location"],
            "industry_comp_data": ["industry_comp_data", "researched_data.industry_comp_data"]
        }
        
        # Look for information in all possible paths
        if info_key in info_paths:
            for path in info_paths[info_key]:
                # Handle nested paths
                if "." in path:
                    parts = path.split(".")
                    current = application_data
                    found = True
                    for part in parts:
                        if part not in current:
                            found = False
                            break
                        current = current[part]
                    
                    if found and current:  # Make sure it's not None or empty
                        return True
                # Simple paths
                elif path in application_data and application_data[path]:
                    return True
        
        # Special case: If the info_key itself is in application_data
        if info_key in application_data and application_data[info_key]:
            return True
            
        # Special case: If the info_key is in researched_data
        if "researched_data" in application_data and info_key in application_data["researched_data"]:
            return True
        
        return False
    
    def assess_job_quality(self, 
                          application_data: Dict, 
                          user_preferences: Dict,
                          knowledge_gaps: Optional[Dict] = None) -> Dict:
        """
        Assess job quality metrics using direct reasoning and dynamic research.
        
        Args:
            application_data: Enriched application data
            user_preferences: User preference variables
            knowledge_gaps: Optional pre-identified knowledge gaps
            
        Returns:
            Dictionary of job quality metrics with scores and reasoning
        """
        job_quality_metrics = {}
        reasoning_traces = []
        
        # If meta_agent is available, reset consecutive counter
        if self.meta_agent:
            self.meta_agent.reset_consecutive()
        
        # Display application and preference summary
        if self.verbose:
            self._display_assessment_header(application_data, user_preferences)
        
        # Create a working copy of the application data
        working_data = application_data.copy()
        
        # Initialize researched_data if not present
        if "researched_data" not in working_data:
            working_data["researched_data"] = {}
        
        # Identify knowledge gaps if not provided
        metrics_to_assess = [metric_dict["key"] for metric_dict in self.config.standard_metrics]
        
        if knowledge_gaps is None:
            knowledge_gaps = self.identify_knowledge_gaps(metrics_to_assess, working_data)
        
        # Only conduct research if we have gaps to fill
        if knowledge_gaps:
            print("\n🔍 CONDUCTING RESEARCH TO FILL KNOWLEDGE GAPS")
            
            # Research all identified gaps
            research_results = self.research_agent.research_all_gaps(working_data, knowledge_gaps)
            
            # Add research results to the working data
            if research_results:
                for key, value in research_results.items():
                    working_data["researched_data"][key] = value
                
                print(f"✅ Added {len(research_results)} new findings to the assessment data")
        
        # Assess each standard metric
        for metric_dict in self.config.standard_metrics:
            metric_key = metric_dict["key"]
            
            print(f"\n🔍 ASSESSING: {metric_dict['name'].upper()}")
            
            # Evaluate the metric with direct reasoning
            score, confidence, reasoning_steps = self._evaluate_metric(
                metric_key=metric_key,
                application_data=working_data,
                user_preferences=user_preferences
            )
            
            # Generate reasoning summary
            reasoning_summary = self._generate_reasoning_summary(reasoning_steps)
            
            # Store the result
            job_quality_metrics[metric_key] = {
                "name": metric_dict["name"],
                "score": score,
                "confidence": confidence,
                "reasoning_summary": reasoning_summary,
                "reasoning_steps": reasoning_steps
            }
            
            # Display the summary
            if self.verbose:
                print(f"\n📊 METRIC ASSESSMENT COMPLETE: {metric_dict['name']}")
                print(f"   Final Score: {score:.1f}/10")
                print(f"   Confidence: {confidence:.2f}")
                print(f"\n📝 REASONING SUMMARY:")
                print(f"   {reasoning_summary}")
            
            # Store detailed reasoning traces
            reasoning_traces.extend([
                {
                    "metric": metric_key,
                    "step": i+1,
                    "reasoning": step["reasoning"],
                    "score": step["score"],
                    "confidence": step["confidence"],
                    "timestamp": datetime.now().isoformat()
                }
                for i, step in enumerate(reasoning_steps)
            ])
            
            # If meta_agent is available, track this assessment
            if self.meta_agent:
                self.meta_agent.track_reasoning(
                    variable=metric_key,
                    response=score,
                    confidence=confidence
                )
        
        # Generate an overall job quality score
        try:
            if self.verbose:
                print(f"\n🏆 CALCULATING OVERALL JOB QUALITY SCORE")
                
            overall_score = self._calculate_overall_score(job_quality_metrics, user_preferences)
            job_quality_metrics["overall_score"] = overall_score
            
            if self.verbose:
                print(f"\n🌟 OVERALL ASSESSMENT COMPLETE")
                print(f"   Final Score: {overall_score['score']:.1f}/10")
                print(f"   Confidence: {overall_score['confidence']:.2f}")
                print(f"\n📝 REASONING SUMMARY:")
                print(f"   {overall_score['reasoning_summary']}")
                
        except Exception as e:
            print(f"❌ Error calculating overall score: {str(e)}")
            job_quality_metrics["overall_score"] = {
                "name": "Overall Job Quality Score",
                "score": 5.0,  # Neutral score
                "confidence": 0.5,
                "reasoning_summary": "Error calculating overall score."
            }
        
        return {
            "metrics": job_quality_metrics,
            "reasoning_traces": reasoning_traces,
            "assessment_date": datetime.now().isoformat(),
            "researched_data": working_data.get("researched_data", {})
        }
    
    def _evaluate_metric(self, metric_key: str, application_data: Dict, user_preferences: Dict) -> Tuple[float, float, List[Dict]]:
        """
        Evaluate a job quality metric using direct reasoning with research-enhanced data.
        
        Args:
            metric_key: The key of the metric to evaluate
            application_data: Enriched application data with research
            user_preferences: User preference variables
            
        Returns:
            Tuple of (score, confidence, reasoning_steps)
        """
        metric = self.config.get_metric(metric_key)
        reasoning_steps = []
        
        if not metric:
            print(f"❌ Metric '{metric_key}' not found")
            return 5.0, 0.5, []
        
        # Start reasoning process
        attempt = 0
        confidence = 0.0
        current_score = None
        
        while confidence < self.config.confidence_threshold and attempt < self.config.max_reasoning_attempts:
            attempt += 1
            
            if self.verbose:
                print(f"\n💭 REASONING STEP {attempt}/{self.config.max_reasoning_attempts}")
            
            # If we're doing a follow-up reasoning step, include previous steps
            previous_reasoning = ""
            if reasoning_steps:
                previous_reasoning = "Previous reasoning attempts:\n"
                for i, step in enumerate(reasoning_steps):
                    previous_reasoning += f"Attempt {i+1}: {step['reasoning']}\n"
                    previous_reasoning += f"Score: {step['score']}/10, Confidence: {step['confidence']}\n\n"
            
            # Identify which pieces of researched data are relevant for this metric
            relevant_research = self._get_relevant_research(metric_key, application_data)
            research_context = ""
            if relevant_research:
                research_context = "Additional researched information:\n"
                for key, value in relevant_research.items():
                    if isinstance(value, dict) and "data" in value:
                        research_context += f"{key}: {value['data']}\n\n"
                    else:
                        research_context += f"{key}: {value}\n\n"
            
            # Create a prompt for this reasoning step
            prompt = f"""
            Evaluate the {metric['name']} ({metric['description']}) for this job application.
            
            Application details:
            Company: {application_data.get('company', 'Unknown')}
            Position: {application_data.get('position', 'Unknown')}
            Description: {application_data.get('description', 'No description available')[:500]}...
            Salary information: {application_data.get('salary', 'Not specified')}
            Location: {application_data.get('location', 'Not specified')}
            Notes: {application_data.get('notes', '')}
            
            {research_context}
            
            User preferences:
            {json.dumps(user_preferences, indent=2)}
            
            {previous_reasoning}
            
            Provide a detailed evaluation of this metric. Consider multiple factors in your reasoning.
            Explicitly state what information you're using and how it impacts your assessment.
            
            Output format:
            Reasoning: [your detailed analysis]
            Score: [numerical score between {metric['min']} and {metric['max']}]
            Confidence: [confidence in your assessment, between 0.0 and 1.0]
            """
            
            # Make the API call
            if self.verbose:
                print(f"🤔 Reasoning about {metric['name']}...")
                
            try:
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                    
                # Parse the response
                text = response.text.strip()
                
                # Extract reasoning, score, and confidence
                reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=Score:|$)", text, re.DOTALL)
                score_match = re.search(r"Score:\s*(\d*\.?\d+)", text)
                confidence_match = re.search(r"Confidence:\s*(\d*\.?\d+)", text)
                
                reasoning = "No reasoning provided"
                score = (metric['min'] + metric['max']) / 2  # Default middle score
                confidence = 0.5  # Default medium confidence
                
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        # Ensure score is within bounds
                        score = max(metric['min'], min(metric['max'], score))
                    except ValueError:
                        pass
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        # Ensure confidence is between 0 and 1
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        pass
                
                # Boost confidence based on research
                if relevant_research:
                    # The more research we have, the higher the confidence
                    research_boost = min(0.2, 0.05 * len(relevant_research))
                    confidence = min(1.0, confidence + research_boost)
                    
                # Store this reasoning step
                reasoning_steps.append({
                    "reasoning": reasoning,
                    "score": score,
                    "confidence": confidence,
                    "research_used": bool(relevant_research)
                })
                
                # Update current score
                current_score = score
                
                # Display reasoning if verbose mode is on
                if self.verbose:
                    print(f"\n💡 REASONING OUTPUT:")
                    print(reasoning)
                    print(f"📊 Score: {score:.1f}/10 (Confidence: {confidence:.2f})")
                    if confidence >= self.config.confidence_threshold:
                        print(f"✅ Confidence threshold met!")
                    else:
                        print(f"⚠️ Confidence below threshold. Will attempt additional reasoning.")
                
                # Check if meta-agent suggests we should stop
                if self.meta_agent and not self.meta_agent.check_and_handle(metric_key, confidence):
                    if self.verbose:
                        print(f"🛑 Meta-agent suggests stopping further reasoning.")
                    break
            
            except Exception as e:
                print(f"❌ Error in reasoning step: {str(e)}")
                confidence = 0.1
                if not current_score:
                    current_score = (metric['min'] + metric['max']) / 2
                break
        
        # Calculate final score from all reasoning steps (weighted by confidence)
        if reasoning_steps:
            confidence_sum = sum(step["confidence"] for step in reasoning_steps)
            if confidence_sum > 0:
                weighted_score = sum(step["score"] * step["confidence"] for step in reasoning_steps) / confidence_sum
                avg_confidence = sum(step["confidence"] for step in reasoning_steps) / len(reasoning_steps)
                return weighted_score, avg_confidence, reasoning_steps
        
        # Fallback
        return current_score or 5.0, confidence, reasoning_steps
    
    def _get_relevant_research(self, metric_key: str, application_data: Dict) -> Dict:
        """
        Get research data relevant to this metric
        
        Args:
            metric_key: The metric being evaluated
            application_data: Enriched application data with research
            
        Returns:
            Dictionary of relevant research findings
        """
        if "researched_data" not in application_data:
            return {}
            
        researched_data = application_data["researched_data"]
        
        # Get required info for this metric
        metric = self.config.get_metric(metric_key)
        if not metric or "required_info" not in metric:
            return {}
            
        required_info = metric["required_info"]
        
        # Get all research relevant to the required info
        relevant_research = {}
        for key in required_info:
            if key in researched_data:
                relevant_research[key] = researched_data[key]
                
        return relevant_research
    
    def _generate_reasoning_summary(self, reasoning_steps: List[Dict]) -> str:
        """Generate a concise summary of the reasoning process"""
        if not reasoning_steps:
            return "No detailed reasoning available."
        
        try:
            # Extract the scores and confidences
            scores = [step["score"] for step in reasoning_steps]
            confidences = [step["confidence"] for step in reasoning_steps]
            
            # Find the step with the highest confidence
            max_conf_idx = confidences.index(max(confidences))
            best_reasoning = reasoning_steps[max_conf_idx]["reasoning"]
            
            # Use the last paragraph of the best reasoning as a summary
            paragraphs = best_reasoning.split('\n\n')
            if len(paragraphs) > 1:
                summary = paragraphs[-1]
            else:
                # If no paragraphs, use the last sentence
                sentences = best_reasoning.split('. ')
                if len(sentences) > 1:
                    summary = sentences[-1]
                else:
                    summary = best_reasoning
            
            # Add score progression if multiple steps
            if len(reasoning_steps) > 1:
                summary += f" Score progressed from {scores[0]:.1f} to {scores[-1]:.1f} across {len(reasoning_steps)} reasoning steps."
            
            # Mention if research was used
            if any(step.get("research_used", False) for step in reasoning_steps):
                summary += " Assessment includes data from focused research."
            
            return summary
        except Exception as e:
            print(f"❌ Error generating reasoning summary: {str(e)}")
            return "Error generating reasoning summary."
    
    def _calculate_overall_score(self, metrics: Dict, user_preferences: Dict) -> Dict:
        """
        Calculate an overall job quality score based on metrics and user preferences.
        
        Args:
            metrics: Dictionary of job quality metrics
            user_preferences: User preference variables
            
        Returns:
            Overall score dictionary with score, confidence, and reasoning
        """
        # Extract preference weights
        weights = {}
        
        # Map user preference weights to corresponding metrics
        if "preference_weights" in user_preferences:
            pref_weights = user_preferences["preference_weights"]
            # Map from preference categories to metrics
            weight_mapping = {
                "compensation": ["base_salary", "total_compensation"],
                "career_growth": ["career_growth"],
                "work_life_balance": ["work_life_balance"],
                "company_reputation": ["company_rating"],
                "location": ["commute_remote_score"],
                "role_responsibilities": ["role_alignment", "job_match_score"]
            }
            
            # Apply weights from preferences, defaulting to 1.0 if not specified
            for pref_key, metric_keys in weight_mapping.items():
                weight = pref_weights.get(pref_key, 3)  # Default to medium importance (3)
                for metric_key in metric_keys:
                    weights[metric_key] = weight
        
        # Fill in any missing weights with default of 1.0
        for metric_key in metrics:
            if metric_key != "overall_score" and metric_key not in weights:
                weights[metric_key] = 1.0
        
        # Calculate weighted average score
        total_weight = 0
        weighted_sum = 0
        confidences = []
        
        for metric_key, metric_data in metrics.items():
            if metric_key != "overall_score" and "score" in metric_data:
                weight = weights.get(metric_key, 1.0)
                weighted_sum += metric_data["score"] * weight
                total_weight += weight
                confidences.append(metric_data["confidence"])
        
        # Calculate overall score and confidence
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        else:
            overall_score = 5.0  # Neutral score
            overall_confidence = 0.5
        
        # Generate reasoning for overall score
        reasoning = f"Overall job quality score of {overall_score:.1f}/10 calculated from weighted metrics. "
        
        # Add information about high and low scoring areas
        sorted_metrics = [(k, v) for k, v in metrics.items() if k != "overall_score"]
        sorted_metrics.sort(key=lambda x: x[1]["score"], reverse=True)
        
        if sorted_metrics:
            best_metric = sorted_metrics[0]
            reasoning += f"Strongest area: {best_metric[1]['name']} ({best_metric[1]['score']:.1f}/10). "
            
            worst_metric = sorted_metrics[-1]
            reasoning += f"Needs improvement: {worst_metric[1]['name']} ({worst_metric[1]['score']:.1f}/10)."
        
        return {
            "name": "Overall Job Quality Score",
            "score": overall_score,
            "confidence": overall_confidence,
            "reasoning_summary": reasoning
        }
        
    def _display_assessment_header(self, application_data: Dict, user_preferences: Dict):
        """Display a summary of the application and user preferences"""
        print("\n🔍 JOB QUALITY ASSESSMENT START")
        
        # Application info
        print("📋 APPLICATION DETAILS:")
        print(f"   Company: {application_data.get('company', 'Unknown')}")
        print(f"   Position: {application_data.get('position', 'Unknown')}")
        print(f"   Location: {application_data.get('location', 'Not specified')}")
        print(f"   Status: {application_data.get('stage', 'Unknown')}")
        
        # User preferences
        print("\n👤 USER PREFERENCES:")
        print(f"   Minimum Salary: ${user_preferences.get('min_salary', 'Not specified')}")
        
        if 'preference_weights' in user_preferences:
            print(f"   Priority Areas:")
            weights = user_preferences['preference_weights']
            for area, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                stars = "★" * weight + "☆" * (5 - weight)
                print(f"   - {area.replace('_', ' ').title()}: {stars} ({weight}/5)")
        
        # Risk profile
        if 'risk_tolerance' in user_preferences:
            risk = user_preferences['risk_tolerance']
            risk_profile = "Very Conservative" if risk <= 2 else "Conservative" if risk <= 4 else "Moderate" if risk <= 6 else "Aggressive" if risk <= 8 else "Very Aggressive"
            print(f"   Risk Tolerance: {risk_profile} ({risk}/10)")
        
        print(f"   Current Salary: ${user_preferences.get('current_salary', 'Not specified')}")
        print(f"   Financial Runway: {user_preferences.get('financial_runway', 'Not specified')} months")

class DataCollectionAgent:
    """
    Agent for collecting missing data from the user.
    """
    def __init__(self, ai_model: AIModelInterface):
        self.ai_model = ai_model
    
    def identify_missing_data(self, application_data: Dict) -> List[Dict]:
        """Identify missing data fields and generate questions"""
        # Essential fields to check
        essential_fields = [
            ("company", "What is the company name?"),
            ("position", "What is the job title or position?"),
            ("description", "Can you provide the job description?"),
            ("salary", "What is the compensation information (salary, benefits, etc.)?"),
            ("location", "Where is the job located? Is it remote, hybrid, or in-office?")
        ]
        
        missing_fields = []
        for field, question in essential_fields:
            if field not in application_data or not application_data[field]:
                missing_fields.append({
                    "field": field,
                    "question": question
                })
        
        return missing_fields
    
    def collect_missing_data(self, application_data: Dict) -> Dict:
        """Collect missing data from the user"""
        updated_data = application_data.copy()
        
        # Identify missing fields
        missing_fields = self.identify_missing_data(updated_data)
        
        if not missing_fields:
            print("✅ All essential information is available")
            return updated_data
        
        print("\n📋 We need some additional information about this job application:")
        
        # Ask user for each missing field
        for item in missing_fields:
            field = item["field"]
            question = item["question"]
            
            print(f"\n❓ {question}")
            response = input("Your answer: ")
            
            updated_data[field] = response
        
        return updated_data

class AutomatedDataCollectionAgent(DataCollectionAgent):
    """
    Automated version of DataCollectionAgent that doesn't require user input.
    """
    def collect_missing_data(self, application_data: Dict) -> Dict:
        """Collect missing data automatically without user input"""
        updated_data = application_data.copy()
        
        # Identify missing fields
        missing_fields = self.identify_missing_data(updated_data)
        
        if not missing_fields:
            print("✅ All essential information is available")
            return updated_data
        
        print("\n📋 Automatically filling missing information about this job application:")
        
        # Default values for common fields
        default_values = {
            "company": "Unknown Company",
            "position": "Unspecified Position",
            "description": "No description provided. This is a placeholder job description " +
                          "for automated processing. The role involves general professional duties.",
            "salary": "Salary information not available",
            "location": "Remote/Flexible"
        }
        
        # Fill in missing fields with defaults
        for item in missing_fields:
            field = item["field"]
            question = item["question"]
            
            print(f"\n{question}")
            default_value = default_values.get(field, "Not specified")
            print(f"Auto-filled with: {default_value}")
            
            updated_data[field] = default_value
        
        return updated_data
    
###########################################
# APPLICATION PROCESSING WORKFLOW
###########################################

class ApplicationProcessingWorkflowBuilder(BaseWorkflowBuilder):
    """
    Builder for the application processing workflow.
    """
    def __init__(self, 
                registry: WorkflowRegistry,
                meta_agent: MetaReasoningAgent,
                storage: StorageInterface,
                config: ApplicationProcessingConfig,
                automated: bool = True):  # Add automated flag
        super().__init__(registry, meta_agent, storage)
        self.config = config
        self.automated = automated
        
        # Initialize agents
        self.email_parser = EmailParserAgent(meta_agent.ai_model)
        
        # Create research agent first to be shared
        self.research_agent = DynamicResearchAgent(
            ai_model=meta_agent.ai_model,
            meta_agent=meta_agent
        )
        
        self.data_enrichment = DataEnrichmentAgent(
            ai_model=meta_agent.ai_model,
            research_agent=self.research_agent,
            meta_agent=meta_agent
        )
        
        self.job_quality = JobQualityAssessmentAgent(
            ai_model=meta_agent.ai_model,
            config=config,
            research_agent=self.research_agent,
            meta_agent=meta_agent
        )
        
        # Use automated data collection agent if automated flag is set
        if automated:
            self.data_collection = AutomatedDataCollectionAgent(meta_agent.ai_model)
        else:
            self.data_collection = DataCollectionAgent(meta_agent.ai_model)
            
    def build_graph(self) -> StateGraph:
        """Build the application processing workflow graph"""
        
        # Define node functions
        def process_application_data(state: ApplicationState) -> dict:
            """Process application data node"""
            print("📄 Processing application data...")
            
            # Ensure required fields exist
            if "application_data" not in state:
                raise ValueError("Application data is missing from state")
            
            # Reset meta agent counter since we're starting a new workflow
            if self.meta_agent:
                self.meta_agent.reset_consecutive()
                
            return {}
        
        def enrich_application_data(state: ApplicationState) -> dict:
            """Enrich application data node"""
            print("🔍 Enriching application data...")
            
            # Collect missing data if needed
            application_data = state["application_data"]
            updated_data = self.data_collection.collect_missing_data(application_data)
            
            # Enrich data
            enriched_data = self.data_enrichment.enrich_application_data(updated_data)
            
            return {"enriched_data": enriched_data, "application_data": updated_data}
        
        def assess_job_quality(state: ApplicationState) -> dict:
            """Assess job quality node"""
            print("⭐ Assessing job quality...")
            
            # Get enriched data and user preferences
            enriched_data = state["enriched_data"]
            user_preferences = state["user_preferences"]
            
            # Identify knowledge gaps
            metrics_to_assess = [metric_dict["key"] for metric_dict in self.config.standard_metrics]
            knowledge_gaps = self.job_quality.identify_knowledge_gaps(metrics_to_assess, enriched_data)
            
            # Assess job quality
            assessment_result = self.job_quality.assess_job_quality(
                application_data=enriched_data,
                user_preferences=user_preferences,
                knowledge_gaps=knowledge_gaps
            )
            
            return {
                "job_quality_metrics": assessment_result["metrics"],
                "reasoning_traces": assessment_result["reasoning_traces"],
                "researched_data": assessment_result.get("researched_data", {})
            }
        
        def store_assessment_results(state: ApplicationState) -> dict:
            """
            Store assessment results in appropriate MongoDB collections for OST.
            Instead of storing all data in a single collection, this method distributes
            the data across multiple domain-specific collections that OST.py can use.
            """
            print("💾 Storing assessment results to OST collections...")
            
            # Import MongoDB utility from ost.py
            from ost import MongoDBUtility
            mongodb_util = MongoDBUtility()
            
            try:
                # Extract key data from state
                application_data = state["application_data"]
                enriched_data = state["enriched_data"]
                job_quality_metrics = state["job_quality_metrics"]
                user_preferences = state["user_preferences"]
                researched_data = state.get("researched_data", {})
                
                # Extract field information
                field = application_data.get("field", enriched_data.get("field"))
                if not field or field == "unknown":
                    # Default to a general category if field is not available
                    field = "general"
                
                # Extract position information
                position = application_data.get("position", enriched_data.get("position"))
                if not position or position == "unknown":
                    position = f"position_in_{field}"
                
                # Extract company information
                company = application_data.get("company", enriched_data.get("company"))
                if not company or company == "unknown":
                    company = f"company_in_{field}"
                
                # Extract location information
                location = application_data.get("location", enriched_data.get("location"))
                if not location or location == "unknown":
                    location = "remote"  # Default to remote rather than unknown
                
                # Extract experience level
                experience_level = enriched_data.get("experience_level")
                if not experience_level or experience_level == "unknown":
                    experience_level = "mid"  # Default to mid-level experience
                
                print(f"📊 Storing data for field: {field}, position: {position}")
                
                # 1. Update variable_types collection with job attributes
                variable_types_updates = {}
                for key, metric in job_quality_metrics.items():
                    # Determine if higher is better (usually true for job metrics)
                    higher_is_better = True
                    if key in ["commute_time", "workload", "stress_level"]:
                        higher_is_better = False
                        
                    # Get description
                    description = f"Job quality metric for {key}"
                    if isinstance(metric, dict) and "description" in metric:
                        description = metric["description"]
                        
                    variable_types_updates[key] = {
                        "name": key,
                        "var_type": "job_metric",
                        "higher_is_better": higher_is_better,
                        "context_dependent": True,
                        "can_be_hard_constraint": key in user_preferences.get("deal_breakers", []),
                        "normalization_method": "absolute_scale",
                        "description": description
                    }
                
                # Store variable types
                for key, var_type in variable_types_updates.items():
                    mongodb_util.save_document("variable_types", var_type, {"name": key})
                    
                # 2. Update field_contexts collection with industry data
                # Create proper structure with field as key
                field_contexts = {}
                
                field_data = {}
                
                # Add salary data if available
                if "salary_data" in researched_data:
                    salary_data = researched_data["salary_data"]
                    if isinstance(salary_data, dict) and "data" in salary_data:
                        field_data["salary_data"] = salary_data["data"]
                        
                # Add base salary info if available in metrics
                if "base_salary" in job_quality_metrics:
                    base_salary = job_quality_metrics["base_salary"]
                    if isinstance(base_salary, (int, float)):
                        # Store actual value if numeric, or extract from text
                        field_data["base_salary"] = {"median": base_salary, "std_dev": base_salary * 0.15}
                
                # Set the field data in the contexts dictionary
                field_contexts[field] = field_data
                
                # Always include a general field for fallback
                if field != "general":
                    field_contexts["general"] = {"base_salary": {"median": 60000, "std_dev": 10000}}
                
                # Insert as a new document
                mongodb_util.db["field_contexts"].insert_one(field_contexts)
                
                # 3. Update experience_contexts collection
                # Create proper structure with level as key
                experience_contexts = {}
                
                # Create data for this experience level
                level_data = {}
                
                # Use AI model to generate appropriate experience level factors
                prompt = f"""
                Generate experience level factors for a {experience_level} role in {field}. 
                
                Include the following data points:
                1. salary_modifier - A multiplier for base salary (e.g., 1.0 for mid-level)
                2. career_growth_importance - How important career growth is at this level (e.g., 1.2 for mid-level)
                3. company_reputation_importance - How important company reputation is at this level (e.g., 1.0 for mid-level)
                
                Format as JSON with these three keys and numeric values.
                """
                
                try:
                    # Generate experience level factors using the AI model
                    response = self.meta_agent.ai_model.generate_content(prompt)
                    experience_factors_json = response.text
                    
                    # Track API usage
                    if hasattr(self.meta_agent, 'increment'):
                        self.meta_agent.increment()
                except Exception as e:
                    print(f"❌ Error generating experience factors: {str(e)}")
                    experience_factors_json = "{}"
                
                # Parse the JSON response
                try:
                    import json
                    import re
                    
                    # Extract JSON if it's embedded in text
                    json_match = re.search(r'\{[^}]*\}', experience_factors_json)
                    if json_match:
                        experience_factors = json.loads(json_match.group(0))
                    else:
                        experience_factors = json.loads(experience_factors_json)
                    
                    # Update level_data with the generated factors
                    level_data.update(experience_factors)
                    
                    # Ensure all required keys are present
                    required_keys = ["salary_modifier", "career_growth_importance", "company_reputation_importance"]
                    for key in required_keys:
                        if key not in level_data:
                            # If a key is missing, add a placeholder that won't break calculations
                            if key == "salary_modifier":
                                level_data[key] = 1.0
                            else:
                                level_data[key] = 1.0
                            print(f"⚠️ Warning: AI didn't generate {key}, using default value")
                            
                except Exception as e:
                    print(f"❌ Error parsing AI generated experience factors: {str(e)}")
                    # Ensure we have minimum required fields if parsing fails
                    level_data["salary_modifier"] = 1.0
                    level_data["career_growth_importance"] = 1.0
                    level_data["company_reputation_importance"] = 1.0
                
                # Add salary modifiers if available
                if "salary" in enriched_data:
                    salary = enriched_data["salary"]
                    if isinstance(salary, (int, float)):
                        level_data["observed_salary"] = salary
                
                # Store as a key-value pair where key is the experience level
                experience_contexts[experience_level] = level_data
                
                # For other experience levels, use AI to generate appropriate factors
                standard_levels = ["entry", "junior", "mid", "senior", "lead"]
                for level in standard_levels:
                    if level != experience_level and level not in experience_contexts:
                        # Generate data for this level
                        level_prompt = f"""
                        Generate experience level factors for a {level} role in {field}. 
                        
                        Include the following data points:
                        1. salary_modifier - A multiplier for base salary (e.g., 1.0 for mid-level)
                        2. career_growth_importance - How important career growth is at this level (e.g., 1.2 for mid-level)
                        3. company_reputation_importance - How important company reputation is at this level (e.g., 1.0 for mid-level)
                        
                        Format as JSON with these three keys and numeric values.
                        """
                        
                        # Generate factors for this level using the AI model
                        try:
                            response = self.meta_agent.ai_model.generate_content(level_prompt)
                            level_factors_json = response.text
                            
                            # Track API usage
                            if hasattr(self.meta_agent, 'increment'):
                                self.meta_agent.increment()
                        except Exception as e:
                            print(f"❌ Error generating experience factors for {level}: {str(e)}")
                            level_factors_json = "{}"
                        
                        # Parse the JSON response
                        try:
                            # Extract JSON if it's embedded in text
                            json_match = re.search(r'\{[^}]*\}', level_factors_json)
                            if json_match:
                                level_factors = json.loads(json_match.group(0))
                            else:
                                level_factors = json.loads(level_factors_json)
                                
                            # Ensure all required keys are present
                            for key in required_keys:
                                if key not in level_factors:
                                    # If a key is missing, add a placeholder
                                    if key == "salary_modifier":
                                        level_factors[key] = 1.0
                                    else:
                                        level_factors[key] = 1.0
                                        
                            experience_contexts[level] = level_factors
                            
                        except Exception as e:
                            print(f"❌ Error parsing AI generated factors for {level}: {str(e)}")
                            # Provide minimum required fields if parsing fails
                            experience_contexts[level] = {
                                "salary_modifier": 1.0,
                                "career_growth_importance": 1.0,
                                "company_reputation_importance": 1.0
                            }
                
                # Print for debugging
                print(f"📊 Generated experience contexts: {experience_contexts}")
                
                # Extract user_id from metadata if available
                user_id = state.get("meta_data", {}).get("user_id")
                if user_id:
                    # Add user_id to experience_contexts
                    experience_contexts["user_id"] = user_id
                    print(f"👤 Adding user_id {user_id} to experience_contexts")
                
                # Insert as a new document
                mongodb_util.db["experience_contexts"].insert_one(experience_contexts)
                
                # 4. Update location_contexts collection
                # Create proper structure with location as key
                location_contexts = {}
                
                # Create data for this location
                location_data = {}
                
                # Add cost of living data if available
                if "location_factor" in researched_data:
                    loc_factor = researched_data["location_factor"]
                    if isinstance(loc_factor, dict) and "data" in loc_factor:
                        location_data["cost_data"] = loc_factor["data"]
                
                # Store as a key-value pair where key is the location
                location_contexts[location] = location_data
                
                # Extract user_id from metadata if available
                user_id = state.get("meta_data", {}).get("user_id")
                if user_id:
                    # Add user_id to location_contexts
                    location_contexts["user_id"] = user_id
                
                # Insert as a new document
                mongodb_util.db["location_contexts"].insert_one(location_contexts)
                
                # 5. Update variable_interactions collection
                # Look for relationships between variables in the job quality metrics
                variable_interactions = {}
                if "work_life_balance" in job_quality_metrics and "base_salary" in job_quality_metrics:
                    variable_interactions["work_life_balance"] = {"base_salary": 0.15}
                    
                if "remote_work" in job_quality_metrics and "commute_time" in job_quality_metrics:
                    variable_interactions["remote_work"] = {"commute_time": 0.8}
                    
                # Store variable interactions
                mongodb_util.save_document("variable_interactions", variable_interactions)
                
                # 6. Create common_tradeoffs document using available metrics
                tradeoffs = {}
                
                # Get available metrics to create tradeoffs
                available_metrics = list(job_quality_metrics.keys())
                
                # Create field-specific tradeoffs based on available metrics
                field_tradeoffs = []
                
                # Create tradeoffs based on available metrics
                if "base_salary" in available_metrics:
                    # Find metrics to trade off with salary
                    for metric in available_metrics:
                        if metric != "base_salary":
                            # Weight determined by agent analysis
                            weight = 0.0
                            if isinstance(job_quality_metrics[metric], dict) and "importance" in job_quality_metrics[metric]:
                                weight = float(job_quality_metrics[metric]["importance"]) / 10.0
                            else:
                                # Generate a tradeoff weight based on the metric value
                                if isinstance(job_quality_metrics[metric], (int, float)):
                                    weight = min(0.25, max(0.05, job_quality_metrics[metric] / 100.0))
                                else:
                                    weight = 0.1
                            
                            field_tradeoffs.append([metric, "base_salary", weight])
                
                # Add only metrics that were analyzed
                tradeoffs[field] = field_tradeoffs
                
                # Insert as a new document
                mongodb_util.db["common_tradeoffs"].insert_one(tradeoffs)
                
                # 7. Create seasonality_factors with proper structure
                current_month = datetime.now().strftime("%B").lower()
                
                # Create simple field-to-seasonality dictionary
                seasonality_factors = {}
                
                # Create a month-based seasonality factor for the field
                seasonality_by_month = {}
                seasonality_by_month[current_month] = 1.0
                
                # Add to the field-specific entry
                seasonality_factors[field] = seasonality_by_month
                
                # Insert as a new document
                mongodb_util.db["seasonality_factors"].insert_one(seasonality_factors)
                
                # 8. Create field_companies with proper structure
                # Only proceed if we have company information
                if company.lower() != "unknown":
                    # Create simple dictionary with field as key and companies list as value
                    field_companies = {}
                    field_companies[field] = [company]
                    
                    # Insert as a new document
                    mongodb_util.db["field_companies"].insert_one(field_companies)
                
                # 9. Create field_positions with proper structure
                # Only proceed if we have position information
                if position.lower() != "unknown":
                    # Create simple dictionary with field as key and positions list as value
                    field_positions = {}
                    field_positions[field] = [position]
                    
                    # Insert as a new document
                    mongodb_util.db["field_positions"].insert_one(field_positions)
                
                # 10. Update offer_quality_weights collection
                offer_quality_weights = {"weights": {}}
                
                # Extract weights from job quality metrics if available
                for key, metric in job_quality_metrics.items():
                    if isinstance(metric, dict) and "weight" in metric:
                        offer_quality_weights["weights"][key] = metric["weight"]
                    else:
                        offer_quality_weights["weights"][key] = 1.0  # Default weight
                        
                # Store offer quality weights
                mongodb_util.save_document("offer_quality_weights", offer_quality_weights)
                
                # 11. Create field_arrival_rate_factors based on research data
                field_arrival_rates = {}
                
                # Set the arrival rate for this field based on researched data
                # If job market data is available, use it to determine arrival rate
                arrival_rate = 1.0  # Start with neutral
                
                # Use researched data to adjust if available
                if "job_market" in researched_data:
                    job_market = researched_data["job_market"]
                    if isinstance(job_market, dict):
                        # More jobs = higher arrival rate
                        if "demand" in job_market and isinstance(job_market["demand"], (int, float)):
                            demand_factor = job_market["demand"] / 5.0  # Normalize from 0-5 scale
                            arrival_rate = max(0.5, min(1.5, demand_factor))
                        
                        # If we have qualitative assessment, use it
                        if "market_health" in job_market and isinstance(job_market["market_health"], str):
                            health = job_market["market_health"].lower()
                            if "hot" in health or "growing" in health:
                                arrival_rate = 1.2
                            elif "slow" in health or "declining" in health:
                                arrival_rate = 0.8
                
                # Set the rate for this field only - no hardcoded values
                field_arrival_rates[field] = arrival_rate
                
                # Insert as a new document
                mongodb_util.db["field_arrival_rate_factors"].insert_one(field_arrival_rates)
                
                # 12. Create field_arrival_factors from actual application data 
                # Create a simple, direct field-to-factor mapping without nested structures
                field_arrival_factors = {}
                
                # Simply add the field with its factor based on data analysis
                # Use application data to determine job availability
                factor = 1.0  # Neutral starting point
                
                # Adjust factor based on location if available
                if "location_factor" in researched_data:
                    location_data = researched_data["location_factor"]
                    if isinstance(location_data, dict) and "data" in location_data:
                        if "job_availability" in location_data["data"]:
                            factor *= float(location_data["data"]["job_availability"])
                
                # Adjust factor based on experience level
                if experience_level in ["junior", "entry"]:
                    factor *= 0.9  # Fewer entry-level positions
                elif experience_level in ["senior", "lead"]:
                    factor *= 1.1  # More senior positions
                
                # Add the field with its calculated factor
                field_arrival_factors[field] = factor
                
                # Insert as a new document
                mongodb_util.db["field_arrival_factors"].insert_one(field_arrival_factors)
                
                print("✅ Assessment data stored across all OST collections!")
                return {}
            except Exception as e:
                print(f"❌ Storage Error: {str(e)}")
                print("⚠️ Could not store assessment data.")
                return {}
        
        # Register nodes with the registry
        self.registry.register_node("process_application_data", process_application_data)
        self.registry.register_node("enrich_application_data", enrich_application_data)
        self.registry.register_node("assess_job_quality", assess_job_quality)
        self.registry.register_node("store_assessment_results", store_assessment_results)
        
        # Build the graph
        graph = StateGraph(ApplicationState)
        graph.add_node("process_application_data", process_application_data)
        graph.add_node("enrich_application_data", enrich_application_data)
        graph.add_node("assess_job_quality", assess_job_quality)
        graph.add_node("store_assessment_results", store_assessment_results)

        graph.add_edge("process_application_data", "enrich_application_data")
        graph.add_edge("enrich_application_data", "assess_job_quality")
        graph.add_edge("assess_job_quality", "store_assessment_results")
        graph.set_entry_point("process_application_data")

        return graph.compile()


###########################################
# APPLICATION SETUP AND USAGE EXAMPLE
###########################################

def setup_application_processing(automated=True):
    """
    Set up and configure the application processing workflow.
    
    Args:
        automated: If True, use the automated agents that don't require user input
    """
    # Create configuration
    config = ApplicationProcessingConfig(
        workflow_id="application_processing",
        storage_collection="job_applications",
        confidence_threshold=0.7,
        max_reasoning_attempts=3,
        max_research_attempts=2  # New: limit research attempts
    )

    # Initialize components
    registry = WorkflowRegistry()
    
    # Configure AI model (replace with your API key)
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Import the required classes
    from main import RateLimitedAPI, MetaReasoningAgent, AutomatedMetaReasoningAgent, MongoDBStorage
    
    ai_model = RateLimitedAPI(model, min_delay=0.5, max_delay=10.0)
    
    # Use the appropriate meta agent based on the automated flag
    if automated:
        meta_agent = AutomatedMetaReasoningAgent(
            ai_model=ai_model,
            max_consecutive_calls=10,
            max_reasoning_per_variable=5
        )
    else:
        meta_agent = MetaReasoningAgent(
            ai_model=ai_model,
            max_consecutive_calls=10,
            max_reasoning_per_variable=5
        )
    
    storage = MongoDBStorage(database_name="job_applications_db")
    
    # Build and register workflow
    builder = ApplicationProcessingWorkflowBuilder(
        registry, meta_agent, storage, config, automated=automated
    )
    builder.register_workflow("application_processing")
    
    return registry, config, meta_agent

def get_user_preferences_by_id(user_id):
    """
    Fetch user preferences from the database based on user ID.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        User preferences dictionary or None if not found
    """
    from ost import MongoDBUtility
    
    # Initialize MongoDB utility
    mongodb_util = MongoDBUtility()
    
    # Try to fetch user preferences by ID
    if not user_id:
        print("⚠️ No user ID provided, will use default preferences")
        return None
        
    try:
        # Try to load user preferences by user_id
        user_preferences = mongodb_util.load_data("user_preferences", {"user_id": user_id})
        
        if user_preferences:
            print(f"✅ Found user preferences for ID: {user_id}")
            return user_preferences
        else:
            print(f"⚠️ No user preferences found for ID: {user_id}, will use default preferences")
            return None
    except Exception as e:
        print(f"❌ Error fetching user preferences: {str(e)}")
        return None


def process_application(application_data, user_preferences=None, user_id=None, automated=True):
    """
    Process a job application and assess its quality.
    
    Args:
        application_data: Basic application information
        user_preferences: User preference variables (optional if user_id is provided)
        user_id: Optional user ID to fetch specific user preferences
        automated: If True, use the automated workflow that doesn't require user input
        
    Returns:
        The assessment results
    """
    print("🚀 Setting up application processing workflow...")
    registry, config, meta_agent = setup_application_processing(automated=automated)
    
    # If user_id provided but no user_preferences, try to fetch from database
    if user_id and not user_preferences:
        fetched_preferences = get_user_preferences_by_id(user_id)
        if fetched_preferences:
            user_preferences = fetched_preferences
            print("📊 Using user preferences from database")
    
    # If still no user_preferences, use defaults
    if not user_preferences:
        from ost import DEFAULT_FIELD_PREFERENCES
        field = application_data.get("field", "software_engineering")
        user_preferences = DEFAULT_FIELD_PREFERENCES.get(field, DEFAULT_FIELD_PREFERENCES["software_engineering"])
        print("📊 Using default preferences for field: " + field)
    
    # Create workflow executor
    from main import WorkflowExecutor
    executor = WorkflowExecutor(registry)
    
    # Initial state
    initial_state = {
        "workflow_id": config.workflow_id,
        "application_data": application_data,
        "user_preferences": user_preferences,
        "knowledge_gaps": {},  # Initialize empty knowledge gaps
        "meta_data": {"source": "api", "automated": automated, "user_id": user_id}
    }
    
    # Run the workflow
    print("⚙️ Processing application...")
    result = executor.run_workflow("application_processing", initial_state)
    
    if result:
        print("✅ Application processing completed successfully!")
        print(f"📄 Application ID: {application_data.get('id', 'N/A')}")
        if user_id:
            print(f"👤 User ID: {user_id}")
        
        # Print overall job quality score
        overall_score = result.get("job_quality_metrics", {}).get("overall_score", {})
        if overall_score:
            print(f"🌟 Overall Job Quality Score: {overall_score.get('score', 'N/A')}/10")
            print(f"📝 Reasoning: {overall_score.get('reasoning_summary', 'N/A')}")
        
        return result
    else:
        print("❌ Application processing failed")
        return None
    
# Example usage
if __name__ == "__main__":
    # Example application data
    application_data = {
        "id": "app-12345",
        "user_id": "user-7890",
        "company": "Sentra",
        "position": "Full Stack Engineer",
        "dateApplied": "2025-03-26",  # Assuming today's date
        "stage": "Applied",
        "lastUpdated": "2025-03-26",
        "description": """
        Sentra is building the AI backbone for post-acute care, revolutionizing how healthcare providers manage administrative workflows.
        
        The Role:
        We are looking for a Full Stack Developer to take ownership of end-to-end product development. As one of the first engineering hires, 
        you’ll work closely with the founding team, own technical decisions, and build highly scalable and secure systems that directly impact patient care and operational efficiency.

        Key Responsibilities:
        - Architect and implement features across the entire stack.
        - Design, build, and maintain AI-driven workflows that automate clinical and administrative tasks for post-acute providers.
        - Integrate AI models into production, staying ahead of advancements in LLMs to enhance platform intelligence.
        - Ensure high reliability, security, and scalability, integrating best practices in modular system design, testing, and observability.
        - Work closely with customers to iterate and refine the platform based on feedback.

        Qualifications:
        - 2+ years of experience in full-stack development with modern web frameworks and cloud architectures (Vue.js, Python, C#, SQL Server, Azure, or similar).
        - Degree in Computer Science, Engineering, Mathematics, or a related technical field (or equivalent experience).
        - Proven track record of building and maintaining scalable web applications in fast-paced environments.
        """,
        "salary": "Competitive compensation: Top-of-market salary + equity package",
        "location": "London, UK (Hybrid)",
        "notes": "Matches 5 out of 10 skills. Role aligns well with my experience in React, .NET, and web development.",
        "logs": [
            {"date": "2025-03-26", "action": "Applied"}
        ]
    }

    # Example user preferences
    user_preferences = {
        "min_salary": 100000,
        "preference_weights": {
            "compensation": 4,
            "career_growth": 5,
            "work_life_balance": 4,
            "company_reputation": 3,
            "location": 2,
            "role_responsibilities": 5
        },
        "risk_tolerance": 7,
        "financial_runway": 6,
        "current_employment_status": "Employed",
        "current_salary": 110000
    }
    
    # Create a unique user ID (in real usage, this would come from the user)
    import uuid
    user_id = str(uuid.uuid4())  # Generate a random UUID
    
    # First save user preferences to the database with the user_id
    from ost import MongoDBUtility
    mongodb_util = MongoDBUtility()
    user_preferences["user_id"] = user_id  # Add user_id to preferences
    mongodb_util.save_document("user_preferences", user_preferences, {"user_id": user_id})
    print(f"👤 Created and saved user preferences with ID: {user_id}")
    
    # Process the application - either providing user_preferences directly or via user_id
    # Uncomment one of these lines:
    assessment_result = process_application(application_data, user_preferences, user_id=user_id, automated=True)  # Provide both user_preferences and user_id
    # assessment_result = process_application(application_data, user_id=user_id, automated=True)  # Only provide user_id to fetch from DB
    
    # Examine results
    if assessment_result and "job_quality_metrics" in assessment_result:
        print("\n📊 JOB QUALITY ASSESSMENT RESULTS")
        
        metrics = assessment_result["job_quality_metrics"]
        overall = metrics.get("overall_score", {})
        
        print(f"🌟 Overall Job Quality Score: {overall.get('score', 'N/A'):.1f}/10")
        print(f"📝 Reasoning: {overall.get('reasoning_summary', 'N/A')}")
        
        print("\n📈 Individual Metrics:")
        for key, metric in metrics.items():
            if key != "overall_score":
                print(f"\n✓ {metric['name']}: {metric['score']:.1f}/10")
                print(f"  {metric['reasoning_summary']}")

        # Show research findings
        if "researched_data" in assessment_result:
            print("\n🔍 RESEARCH FINDINGS:")
            research = assessment_result["researched_data"]
            for key, value in research.items():
                if isinstance(value, dict) and "data" in value:
                    print(f"\n📌 {key}:")
                    if isinstance(value["data"], dict):
                        for k, v in value["data"].items():
                            print(f"  - {k}: {v}")
                    else:
                        print(f"  {value['data']}")