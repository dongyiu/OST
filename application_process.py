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


class JobQualityMetric:
    """Definition of a job quality metric"""
    def __init__(self, 
                 key: str, 
                 name: str,
                 description: str,
                 min_val: float, 
                 max_val: float, 
                 weight: float = 1.0,
                 extraction_prompt: str = None):
        self.key = key
        self.name = name
        self.description = description
        self.min = min_val
        self.max = max_val
        self.weight = weight  # Default importance weight
        self.extraction_prompt = extraction_prompt or f"Extract the {name} ({description}) from the job data."
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "min": self.min,
            "max": self.max,
            "weight": self.weight,
            "extraction_prompt": self.extraction_prompt
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
            extraction_prompt=data.get("extraction_prompt")
        )


class ApplicationProcessingConfig:
    """Configuration for application processing workflow"""
    def __init__(self,
                 workflow_id: str = "application_processing",
                 storage_collection: str = "applications",
                 standard_metrics: List[Dict] = None,
                 confidence_threshold: float = 0.7,
                 max_reasoning_attempts: int = 3):
        self.workflow_id = workflow_id
        self.storage_collection = storage_collection
        self.standard_metrics = standard_metrics or []
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_attempts = max_reasoning_attempts
        
        # Initialize standard metrics if none provided
        if not self.standard_metrics:
            self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """Initialize standard job quality metrics"""
        self.standard_metrics = [
            JobQualityMetric(
                key="base_salary",
                name="Base Salary",
                description="Standardized annual value in USD",
                min_val=0,
                max_val=10,
                weight=1.0,
                extraction_prompt="Extract and evaluate the base annual salary in USD from the job data. Rate it on a scale of 1-10 based on how competitive it is."
            ).to_dict(),
            JobQualityMetric(
                key="total_compensation",
                name="Total Compensation",
                description="Including benefits, bonuses, stock options in USD",
                min_val=0,
                max_val=10,
                weight=1.0,
                extraction_prompt="Calculate and rate the total annual compensation including base salary, bonuses, benefits, and stock options on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="job_match_score",
                name="Job Match Score",
                description="How well job matches user's skills and experience",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well the job requirements match the user's skills and experience on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="company_rating",
                name="Company Rating",
                description="Based on reputation, stability, and market position",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the company based on its reputation, financial stability, and market position on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="work_life_balance",
                name="Work-Life Balance Score",
                description="Expected work-life balance based on company culture and role",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the expected work-life balance for this role and company on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="career_growth",
                name="Career Growth Potential",
                description="Advancement opportunities and skill development",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the potential for career advancement and skill development in this role on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="role_alignment",
                name="Role Alignment",
                description="Fit with user's career goals and interests",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well this role aligns with the user's stated career goals and interests on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="team_manager_quality",
                name="Team/Manager Quality",
                description="Based on interview impressions and information",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the quality of the team and management based on interview impressions and available information on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="culture_fit",
                name="Culture Fit",
                description="Compatibility with company culture and values",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate how well the company culture and values align with the user's preferences on a scale of 1-10."
            ).to_dict(),
            JobQualityMetric(
                key="commute_remote_score",
                name="Commute/Remote Score",
                description="Location convenience and remote work options",
                min_val=1,
                max_val=10,
                weight=1.0,
                extraction_prompt="Rate the convenience of the job location and remote work options on a scale of 1-10."
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
            "max_reasoning_attempts": self.max_reasoning_attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ApplicationProcessingConfig':
        """Create from dictionary"""
        return cls(
            workflow_id=data.get("workflow_id", "application_processing"),
            storage_collection=data.get("storage_collection", "applications"),
            standard_metrics=data.get("standard_metrics", []),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            max_reasoning_attempts=data.get("max_reasoning_attempts", 3)
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


class WebScraperAgent:
    """
    Agent for scraping additional information from company websites.
    This is a placeholder implementation.
    """
    def __init__(self, ai_model: AIModelInterface):
        self.ai_model = ai_model
    
    def scrape_company_info(self, company_name: str, job_url: str = None) -> Dict:
        """Scrape additional information about the company"""
        # In a real implementation, this would use web scraping tools
        # For this example, we'll return a simple placeholder
        return {
            "company_name": company_name,
            "scraped_date": datetime.now().isoformat(),
            "company_size": "Unknown",
            "industry": "Unknown",
            "founded": "Unknown",
            "website": f"https://www.{company_name.lower().replace(' ', '')}.com",
            "status": "scraped"
        }


class DataEnrichmentAgent:
    """
    Agent for enriching application data with additional information.
    """
    def __init__(self, 
                 ai_model: AIModelInterface,
                 web_scraper: WebScraperAgent = None,
                 meta_agent: MetaReasoningAgent = None):
        self.ai_model = ai_model
        self.web_scraper = web_scraper or WebScraperAgent(ai_model)
        self.meta_agent = meta_agent
    
    def enrich_application_data(self, application_data: Dict) -> Dict:
        """Enrich application data with additional information"""
        enriched_data = application_data.copy()
        
        # If meta_agent is available, reset consecutive counter
        if self.meta_agent:
            self.meta_agent.reset_consecutive()
        
        print("🔍 ENRICHING APPLICATION DATA")
        
        # Add company information if available
        company_name = application_data.get("company")
        if company_name:
            print(f"🏢 Fetching information for company: {company_name}")
            company_info = self.web_scraper.scrape_company_info(company_name)
            enriched_data["company_info"] = company_info
        
        # Extract and structure salary information
        if "salary" in application_data and application_data["salary"]:
            print("💰 Extracting structured salary information")
            try:
                prompt = f"""
                Extract structured salary information from this text:
                "{application_data['salary']}"
                
                Format your response as JSON with these fields:
                - base_salary: The annual base salary in USD (convert if needed)
                - salary_min: Minimum salary range if provided
                - salary_max: Maximum salary range if provided
                - has_bonus: Boolean indicating if bonus is mentioned
                - bonus_details: Details about bonus structure
                - has_equity: Boolean indicating if equity is mentioned
                - equity_details: Details about equity
                - has_benefits: Boolean indicating if benefits are mentioned
                - benefits_details: Details about benefits
                
                Provide best estimates for numerical values.
                """
                
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                
                try:
                    # Try to parse as JSON
                    salary_info = json.loads(response.text)
                    enriched_data["structured_salary"] = salary_info
                    print("✅ Salary information extracted successfully")
                except json.JSONDecodeError:
                    # Fallback if not valid JSON
                    enriched_data["structured_salary"] = {
                        "base_salary": None,
                        "error": "Failed to parse salary information"
                    }
                    print("⚠️ Failed to parse salary information as JSON")
            except Exception as e:
                print(f"❌ Error extracting salary information: {str(e)}")
        
        # Extract job skills and requirements
        if "description" in application_data and application_data["description"]:
            print("🧠 Extracting job requirements and skills")
            try:
                prompt = f"""
                Extract key job requirements and skills from this job description:
                "{application_data['description']}"
                
                Format your response as JSON with these fields:
                - required_skills: List of explicitly required skills
                - preferred_skills: List of preferred/nice-to-have skills
                - experience_level: Required years of experience
                - education_requirements: Required education level
                - job_type: Full-time, part-time, contract, etc.
                - remote_status: Remote, hybrid, on-site, etc.
                
                Keep each list item concise.
                """
                
                response = self.ai_model.generate_content(prompt)
                if self.meta_agent:
                    self.meta_agent.increment()
                
                try:
                    # Try to parse as JSON
                    job_requirements = json.loads(response.text)
                    enriched_data["job_requirements"] = job_requirements
                    print("✅ Job requirements extracted successfully")
                except json.JSONDecodeError:
                    # Fallback if not valid JSON
                    enriched_data["job_requirements"] = {
                        "error": "Failed to parse job requirements"
                    }
                    print("⚠️ Failed to parse job requirements as JSON")
            except Exception as e:
                print(f"❌ Error extracting job requirements: {str(e)}")
        
        # Add metadata
        enriched_data["enrichment_date"] = datetime.now().isoformat()
        enriched_data["enrichment_status"] = "completed"
        
        print("✅ Data enrichment completed")
        
        return enriched_data

class JobQualityAssessmentAgent:
    """
    Agent for assessing job quality metrics using direct reasoning, similar to the user preference system.
    """
    def __init__(self, 
                 ai_model: AIModelInterface,
                 config: ApplicationProcessingConfig,
                 meta_agent: Optional[MetaReasoningAgent] = None):
        self.ai_model = ai_model
        self.config = config
        self.meta_agent = meta_agent
        self.verbose = True  # Enable detailed output
    
    def assess_job_quality(self, 
                          application_data: Dict, 
                          user_preferences: Dict) -> Dict:
        """
        Assess job quality metrics using direct reasoning.
        
        Args:
            application_data: Enriched application data
            user_preferences: User preference variables
            
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
        
        # Assess each standard metric
        for metric_dict in self.config.standard_metrics:
            metric_key = metric_dict["key"]
            
            print(f"\n🔍 ASSESSING: {metric_dict['name'].upper()}")
            
            # Evaluate the metric with direct reasoning
            score, confidence, reasoning_steps = self._evaluate_metric(
                metric_key=metric_key,
                application_data=application_data,
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
            "assessment_date": datetime.now().isoformat()
        }
    
    def _evaluate_metric(self, metric_key: str, application_data: Dict, user_preferences: Dict) -> Tuple[float, float, List[Dict]]:
        """
        Evaluate a job quality metric using direct reasoning.
        
        Args:
            metric_key: The key of the metric to evaluate
            application_data: Enriched application data
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
            
            # Create a prompt for this reasoning step
            prompt = f"""
            Evaluate the {metric['name']} ({metric['description']}) for this job application.
            
            Application details:
            Company: {application_data.get('company', 'Unknown')}
            Position: {application_data.get('position', 'Unknown')}
            Description: {application_data.get('description', 'No description available')}
            Salary information: {application_data.get('salary', 'Not specified')}
            Location: {application_data.get('location', 'Not specified')}
            Notes: {application_data.get('notes', '')}
            
            User preferences:
            {json.dumps(user_preferences, indent=2)}
            
            {previous_reasoning}
            
            Provide a detailed evaluation of this metric. Consider multiple factors in your reasoning.
            
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
                
                # Store this reasoning step
                reasoning_steps.append({
                    "reasoning": reasoning,
                    "score": score,
                    "confidence": confidence
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
        self.web_scraper = WebScraperAgent(meta_agent.ai_model)
        self.data_enrichment = DataEnrichmentAgent(
            ai_model=meta_agent.ai_model,
            web_scraper=self.web_scraper,
            meta_agent=meta_agent
        )
        self.job_quality = JobQualityAssessmentAgent(
            ai_model=meta_agent.ai_model,
            config=config,
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
            
            # Assess job quality
            assessment_result = self.job_quality.assess_job_quality(
                application_data=enriched_data,
                user_preferences=user_preferences
            )
            
            return {
                "job_quality_metrics": assessment_result["metrics"],
                "reasoning_traces": assessment_result["reasoning_traces"]
            }
        
        def store_assessment_results(state: ApplicationState) -> dict:
            """Store assessment results node"""
            print("💾 Storing assessment results...")
            
            try:
                document = {
                    "workflow_id": self.config.workflow_id,
                    "application_id": state["application_data"].get("id", str(time.time())),
                    "application_data": state["application_data"],
                    "enriched_data": state["enriched_data"],
                    "job_quality_metrics": state["job_quality_metrics"],
                    "user_preferences": state["user_preferences"],
                    "reasoning_traces": state["reasoning_traces"],
                    "assessment_date": datetime.now().isoformat(),
                    "api_stats": {
                        "total_calls": self.meta_agent.total_calls if self.meta_agent else 0
                    }
                }
                
                # Store data using the storage interface
                success = self.storage.store_data(
                    self.config.storage_collection, 
                    document
                )
                
                if success:
                    print("✅ Assessment data stored successfully!")
                else:
                    print("⚠️ Failed to store assessment data")
                    
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
        max_reasoning_attempts=3
    )

    # Initialize components
    registry = WorkflowRegistry()
    
    # Configure AI model (replace with your API key)
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Import the required classes
    from main import RateLimitedAPI, MetaReasoningAgent, AutomatedMetaReasoningAgent, MongoDBStorage
    
    ai_model = RateLimitedAPI(model, min_delay=2.0, max_delay=10.0)
    
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

def process_application(application_data, user_preferences, automated=True):
    """
    Process a job application and assess its quality.
    
    Args:
        application_data: Basic application information
        user_preferences: User preference variables
        automated: If True, use the automated workflow that doesn't require user input
        
    Returns:
        The assessment results
    """
    print("🚀 Setting up application processing workflow...")
    registry, config, meta_agent = setup_application_processing(automated=automated)
    
    # Create workflow executor
    from main import WorkflowExecutor
    executor = WorkflowExecutor(registry)
    
    # Initial state
    initial_state = {
        "workflow_id": config.workflow_id,
        "application_data": application_data,
        "user_preferences": user_preferences,
        "meta_data": {"source": "api", "automated": automated}
    }
    
    # Run the workflow
    print("⚙️ Processing application...")
    result = executor.run_workflow("application_processing", initial_state)
    
    if result:
        print("✅ Application processing completed successfully!")
        print(f"📄 Application ID: {application_data.get('id', 'N/A')}")
        
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
        "company": "TechCorp Inc.",
        "position": "Senior Software Engineer",
        "dateApplied": "2023-04-15",
        "stage": "Interview",
        "lastUpdated": "2023-04-20",
        "description": """
        We are looking for a Senior Software Engineer to join our team.
        
        Requirements:
        - 5+ years of experience in software development
        - Strong knowledge of Python, JavaScript, and cloud technologies
        - Experience with microservices architecture
        - Ability to mentor junior developers
        
        Benefits:
        - Competitive salary
        - Remote work options
        - Flexible hours
        - Health insurance
        - 401k matching
        """,
        "salary": "$120,000 - $150,000 per year, plus bonuses and stock options",
        "location": "San Francisco, CA (Hybrid - 2 days in office)",
        "notes": "Had a great initial call with the hiring manager. Team seems friendly.",
        "logs": [
            {"date": "2023-04-15", "action": "Applied"},
            {"date": "2023-04-18", "action": "Phone Screen"},
            {"date": "2023-04-20", "action": "Technical Interview Scheduled"}
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
    
    # Process the application with automated=True to avoid user input prompts
    assessment_result = process_application(application_data, user_preferences, automated=True)
    
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
