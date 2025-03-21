import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from dataclasses import dataclass
import json
import random
from typing import List, Dict, Tuple, Optional, Callable, Any
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Parse user data from the provided JSON format (keeping the original function)
def parse_user_data(data_string: str) -> dict:
    """Extract user preferences from the provided JSON data."""
    try:
        # First attempt - treat as complete JSON with just missing outer braces
        json_str = "{" + data_string + "}"
        user_data = json.loads(json_str)
        
        # Convert the list of question/response pairs into a dictionary for easier access
        preferences = {}
        for item in user_data['user_data']:
            preferences[item['question_type']] = item['response']
        
        return preferences
    except json.JSONDecodeError:
        try:
            # Second attempt - the string might already be a complete JSON object
            user_data = json.loads(data_string)
            
            # Convert the list of question/response pairs into a dictionary for easier access
            preferences = {}
            for item in user_data['user_data']:
                preferences[item['question_type']] = item['response']
            
            return preferences
        except json.JSONDecodeError:
            try:
                # Third attempt - try parsing as array directly
                json_str = '{"user_data": [' + data_string.strip().strip('"user_data": [').strip(']},"') + ']}'
                user_data = json.loads(json_str)
                
                # Convert the list of question/response pairs into a dictionary for easier access
                preferences = {}
                for item in user_data['user_data']:
                    preferences[item['question_type']] = item['response']
                
                return preferences
            except json.JSONDecodeError:
                # Last resort - manually extract key-value pairs 
                preferences = {}
                # Extract each question_type and response pair using regex or string operations
                import re
                question_types = re.findall(r'"question_type": "([^"]+)"', data_string)
                responses = re.findall(r'"response": ([^,\n]+)', data_string)
                
                # Match them up
                for i in range(min(len(question_types), len(responses))):
                    try:
                        # Try to convert numerical values to float
                        preferences[question_types[i]] = float(responses[i])
                    except ValueError:
                        # Keep as string if not a number
                        preferences[question_types[i]] = responses[i]
                
                return preferences

@dataclass
class JobOffer:
    """Represents a job offer with various attributes."""
    company: str
    salary: float
    career_growth: int  # 1-5 scale
    location_score: int  # 1-5 scale
    work_life_balance: int  # 1-5 scale
    company_reputation: int  # 1-5 scale
    role_responsibilities: int  # 1-5 scale
    ai_component: int  # 1-5 scale
    tech_stack_match: int  # 1-5 scale
    mentoring_opportunity: int  # 1-5 scale
    project_variety: int  # 1-5 scale
    team_collaboration: int  # 1-5 scale
    negotiation_flexibility: float = 0.0  # 0-1 scale, representing room for negotiation
    
    def calculate_utility(self, preferences: dict) -> float:
        """Calculate the utility/value of this job offer based on user preferences."""
        utility = 0.0
        
        # Salary component - normalized against minimum salary
        min_salary = preferences.get('min_salary', 20000)
        current_salary = preferences.get('current_salary', 25000)
        salary_improvement = max(0, (self.salary - current_salary) / current_salary)
        utility += salary_improvement * preferences.get('compensation_weight', 3)
        
        # Add weighted components for other job attributes
        utility += self.career_growth * preferences.get('career_growth_weight', 3) / 5
        utility += self.location_score * preferences.get('location_weight', 3) / 5
        utility += self.work_life_balance * preferences.get('work_life_balance_weight', 3) / 5
        utility += self.company_reputation * preferences.get('company_reputation_weight', 3) / 5
        utility += self.role_responsibilities * preferences.get('role_responsibilities_weight', 3) / 5
        utility += self.ai_component * preferences.get('ai_application_interest', 3) / 5
        utility += self.tech_stack_match * preferences.get('tech_stack_alignment_weight', 3) / 5
        utility += self.mentoring_opportunity * preferences.get('mentoring_opportunity_weight', 3) / 5
        utility += self.project_variety * preferences.get('project_variety_weight', 3) / 5
        utility += self.team_collaboration * preferences.get('team_collaboration_weight', 3) / 5
        
        # If salary is below minimum, heavily penalize
        if self.salary < min_salary:
            utility -= 10
            
        return utility
    
    def __str__(self) -> str:
        return f"{self.company} - £{self.salary:,.2f}/year (Career:{self.career_growth}/5, WLB:{self.work_life_balance}/5)"


class BayesianJobMarketModel:
    """
    A Bayesian model of the job market that updates beliefs based on observed offers.
    This represents a significant advancement over the original model by learning from
    the data as it's observed.
    """
    
    def __init__(self, preferences: dict):
        """
        Initialize the Bayesian job market model with prior beliefs.
        
        Args:
            preferences: User preferences for job attributes
        """
        self.preferences = preferences
        
        # Prior beliefs about job market - represented as distribution parameters
        # We'll use Gaussian distributions for continuous variables and Beta for bounded ones
        self.current_salary = preferences.get('current_salary', 25000)
        
        # Salary model - represented as a Gaussian mixture
        # Components: below market, at market, above market, significantly above market
        self.salary_weights = np.array([0.3, 0.4, 0.2, 0.1])  # Initial mixture weights
        self.salary_means = np.array([
            0.9 * self.current_salary,  # Below market
            1.08 * self.current_salary,  # At market
            1.2 * self.current_salary,  # Above market
            1.4 * self.current_salary,  # Significantly above market
        ])
        self.salary_stds = np.array([
            0.1 * self.current_salary,
            0.05 * self.current_salary,
            0.08 * self.current_salary,
            0.15 * self.current_salary,
        ])
        
        # Prior for job attributes (alpha, beta parameters for Beta distributions)
        # Higher alpha relative to beta means we expect higher values
        self.attribute_priors = {
            'career_growth': (2, 2),          # Uniform prior initially
            'location_score': (2, 2),
            'work_life_balance': (2, 2),
            'company_reputation': (2, 2),
            'role_responsibilities': (2, 2),
            'ai_component': (2, 2),
            'tech_stack_match': (2, 2),
            'mentoring_opportunity': (2, 2),
            'project_variety': (2, 2),
            'team_collaboration': (2, 2),
        }
        
        # Keep observed data for updating our model
        self.observed_salaries = []
        self.observed_attributes = {attr: [] for attr in self.attribute_priors}
        
        # Market condition modeling 
        self.market_condition = 1.0  # 1.0 means neutral market
        self.market_volatility = 0.05  # How much market conditions change over time
        self.seasonality_factors = [
            1.05, 1.08, 1.10, 1.05,  # Q1: Hiring season in many industries
            0.98, 0.95, 0.92, 0.90,  # Q2: Slowing down, budgets allocated
            0.88, 0.90, 0.93, 0.95,  # Q3: Summer slowdown, then picking up
            1.02, 1.08, 1.12, 1.10,  # Q4: Year-end hiring push, then holiday slowdown
        ]
        
        # Strategic employer behavior model
        self.negotiation_model = {
            'initial_lowball_factor': 0.9,  # Initial offers are typically 90% of max
            'candidate_desirability': 1.0,  # How desirable the candidate is (1.0 = neutral)
            'urgency_to_fill': 0.5,         # How urgent is the position to be filled (0-1)
            'counter_offer_improvement': 0.05,  # 5% improvement on counter offers
        }
        
        # Time-variant skill improvement model
        self.skill_growth_rate = 0.01  # 1% skill growth per unit time
        self.last_update_time = 0
        
        # Learning model for interview success
        self.interview_gp = self._initialize_gp_model()
        self.interview_outcomes = []  # [(features, outcome), ...]
        
    def _initialize_gp_model(self):
        """Initialize a Gaussian Process model for learning interview success factors."""
        # RBF kernel with noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    
    def update_with_offer(self, offer: JobOffer, time_point: float, was_successful: bool = None):
        """
        Update our model with a new observed job offer.
        
        Args:
            offer: The observed job offer
            time_point: The time point when this offer was observed (0-1 scale)
            was_successful: Whether this job application was successful (None if not yet known)
        """
        # Track the salary
        self.observed_salaries.append(offer.salary)
        
        # Track the attributes
        for attr in self.attribute_priors:
            if hasattr(offer, attr):
                # Convert 1-5 scale to 0-1 scale for Beta distribution
                value = getattr(offer, attr) / 5.0
                self.observed_attributes[attr].append(value)
        
        # Update our Bayesian model
        self._update_salary_model()
        self._update_attribute_models()
        
        # Update market condition model
        self._update_market_condition(time_point)
        
        # Update skill growth model
        self._update_skills(time_point)
        
        # Update interview success model if outcome is known
        if was_successful is not None:
            self._update_interview_model(offer, was_successful)
    
    def _update_salary_model(self):
        """Update the salary model based on observed salaries."""
        if len(self.observed_salaries) < 2:
            return  # Not enough data to update
        
        # Fit a Gaussian Mixture Model to the observed salaries
        from sklearn.mixture import GaussianMixture
        
        # Reshape for scikit-learn
        data = np.array(self.observed_salaries).reshape(-1, 1)
        
        # Determine number of components based on available data
        # We need at least as many samples as components
        n_samples = len(self.observed_salaries)
        n_components = min(4, n_samples)  # Use at most 4 components, but no more than we have samples
        
        # Fit GMM with appropriate number of components
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data)
        
        # Extract the updated parameters
        self.salary_weights = gmm.weights_
        self.salary_means = gmm.means_.flatten()
        self.salary_stds = np.sqrt(gmm.covariances_.flatten())
        
        # Sort the components by mean for consistency
        idx = np.argsort(self.salary_means)
        self.salary_means = self.salary_means[idx]
        self.salary_stds = self.salary_stds[idx]
        self.salary_weights = self.salary_weights[idx]
        
        # If we have fewer than 4 components, pad the arrays to maintain consistent shape
        if n_components < 4:
            # Pad with zeros, maintaining the same structure
            if n_components == 1:
                # Special case for 1 component
                self.salary_weights = np.array([0.1, 0.1, 0.1, 0.7])  # Most weight on the single observed component
                self.salary_means = np.array([
                    0.85 * self.salary_means[0],
                    0.95 * self.salary_means[0],
                    1.05 * self.salary_means[0],
                    self.salary_means[0]
                ])
                self.salary_stds = np.array([
                    self.salary_stds[0],
                    self.salary_stds[0],
                    self.salary_stds[0],
                    self.salary_stds[0]
                ])
            else:
                # For 2 or 3 components, extend proportionally
                current_means = self.salary_means.copy()
                current_stds = self.salary_stds.copy()
                current_weights = self.salary_weights.copy()
                
                # Initialize full-sized arrays
                self.salary_means = np.zeros(4)
                self.salary_stds = np.zeros(4)
                self.salary_weights = np.zeros(4)
                
                # Copy existing components
                self.salary_means[:n_components] = current_means
                self.salary_stds[:n_components] = current_stds
                self.salary_weights[:n_components] = current_weights * 0.7  # Scale down to make room for synthetic components
                
                # Create synthetic components at the edges
                min_mean = np.min(current_means)
                max_mean = np.max(current_means)
                range_mean = max_mean - min_mean
                
                # Fill remaining components
                for i in range(n_components, 4):
                    if i == n_components:
                        # Add a component below the minimum
                        self.salary_means[i] = min_mean - (range_mean * 0.1)
                        self.salary_stds[i] = np.mean(current_stds)
                        self.salary_weights[i] = 0.3 / (4 - n_components)
                    elif i == n_components + 1 and n_components < 3:
                        # Add a component above the maximum
                        self.salary_means[i] = max_mean + (range_mean * 0.1)
                        self.salary_stds[i] = np.mean(current_stds)
                        self.salary_weights[i] = 0.3 / (4 - n_components)
                    else:
                        # Add a component in between
                        self.salary_means[i] = min_mean + (range_mean * 0.5)
                        self.salary_stds[i] = np.mean(current_stds)
                        self.salary_weights[i] = 0.3 / (4 - n_components)
    
    def _update_attribute_models(self):
        """Update the attribute models based on observed attributes."""
        for attr, values in self.observed_attributes.items():
            if not values:
                continue
                
            # Get prior parameters
            alpha_prior, beta_prior = self.attribute_priors[attr]
            
            # Convert 1-5 values to successes/failures for Beta
            n_values = len(values)
            successes = sum(values)  # Sum of normalized 0-1 values
            
            # Update the Beta parameters
            alpha_posterior = alpha_prior + successes
            beta_posterior = beta_prior + (n_values - successes)
            
            # Save the updated parameters
            self.attribute_priors[attr] = (alpha_posterior, beta_posterior)
    
    def _update_market_condition(self, time_point: float):
        """
        Update the market condition based on time and seasonality.
        
        Args:
            time_point: Time on 0-1 scale (0 = start of search, 1 = max time)
        """
        # Get the seasonal factor for this time
        week_of_year = int(time_point * 52) % 52
        season_index = (week_of_year // 3) % 16  # 16 3-week periods
        seasonal_factor = self.seasonality_factors[season_index]
        
        # Add random walk component for market changes
        random_walk = np.random.normal(0, self.market_volatility)
        
        # Update market condition
        self.market_condition = max(0.5, min(1.5, 
            self.market_condition * seasonal_factor + random_walk))
    
    def _update_skills(self, time_point: float):
        """
        Update the candidate's skills based on time passed.
        
        Args:
            time_point: Current time point (0-1 scale)
        """
        # Calculate time passed since last update
        time_passed = time_point - self.last_update_time
        self.last_update_time = time_point
        
        # Increase the candidate desirability due to skill growth
        # More learning happens when actively searching
        self.negotiation_model['candidate_desirability'] *= (1 + self.skill_growth_rate * time_passed)
    
    def _update_interview_model(self, offer: JobOffer, success: bool):
        """
        Update our model of interview success probability.
        
        Args:
            offer: The job offer that resulted from the interview
            success: Whether the interview was successful
        """
        # Extract features from the offer that might predict interview success
        features = [
            offer.salary / self.current_salary,  # Normalized salary
            offer.career_growth / 5.0,           # Normalized career growth
            offer.company_reputation / 5.0,      # Normalized company reputation
            offer.tech_stack_match / 5.0,        # Normalized tech stack match
        ]
        
        # Add to training data
        self.interview_outcomes.append((features, 1.0 if success else 0.0))
        
        # Only retrain if we have enough data
        if len(self.interview_outcomes) >= 5:
            X = np.array([f for f, _ in self.interview_outcomes])
            y = np.array([o for _, o in self.interview_outcomes])
            
            # Train the Gaussian Process model
            self.interview_gp.fit(X, y)
    
    def predict_interview_success(self, offer: JobOffer) -> float:
        """
        Predict the probability of success with a given job offer interview.
        
        Args:
            offer: The job offer to predict success for
            
        Returns:
            Probability of interview success (0-1)
        """
        # If we don't have enough data, return a default value
        if len(self.interview_outcomes) < 5:
            return 0.5
        
        # Extract features
        features = np.array([[
            offer.salary / self.current_salary,
            offer.career_growth / 5.0,
            offer.company_reputation / 5.0,
            offer.tech_stack_match / 5.0,
        ]])
        
        # Predict using the GP model
        pred, std = self.interview_gp.predict(features, return_std=True)
        
        # Bound the prediction between 0 and 1
        return max(0, min(1, pred[0]))
    
    def generate_job_offer(self, time_point: float) -> JobOffer:
        """
        Generate a realistic job offer based on current market model.
        
        Args:
            time_point: Current time point (0-1 scale)
            
        Returns:
            A generated job offer
        """
        # Generate company name
        companies = [
            "TechInnovate", "DataDynamics", "CodeCraft", "ByteBuilders", "QuantumQueries",
            "CyberSolutions", "CloudCore", "AIVentures", "WebWizards", "DevDreams",
            "SystemSage", "NetNavigators", "InfoInnovators", "SoftwareSynergy", "TechTrend",
            "DigitalDomain", "ByteBrilliance", "CodeConnect", "AppArchitects", "SecuritySystems",
            "MobileMasters", "AnalyticsAdvance", "QuantumQuest", "IntelImpact", "VirtualVanguard"
        ]
        company = random.choice(companies)
        
        # Generate salary from our Bayesian model
        # First, decide which mixture component to use
        component = np.random.choice(len(self.salary_means), p=self.salary_weights)
        
        # Then generate from that normal distribution
        mean = self.salary_means[component] * self.market_condition
        std = self.salary_stds[component]
        salary = max(10000, np.random.normal(mean, std))
        
        # Generate job attributes from our Beta distributions
        attributes = {}
        for attr, (alpha, beta) in self.attribute_priors.items():
            # Generate from Beta distribution and convert to 1-5 scale
            value_0_1 = np.random.beta(alpha, beta)
            value_1_5 = max(1, min(5, round(1 + 4 * value_0_1)))
            attributes[attr] = value_1_5
            
        # Apply strategic employer behavior
        # If the candidate is more desirable, the salary offered might be higher
        salary *= self.negotiation_model['candidate_desirability']
        
        # If the employer is urgent to fill, they might offer more
        urgency_bonus = 1.0 + (0.1 * self.negotiation_model['urgency_to_fill'])
        salary *= urgency_bonus
        
        # Employers typically lowball initial offers
        salary *= self.negotiation_model['initial_lowball_factor']
        
        # Set negotiation flexibility
        negotiation_flexibility = max(0, min(1, 
            0.15 * (1 + self.negotiation_model['urgency_to_fill']) - 
            0.05 * self.negotiation_model['candidate_desirability']))
            
        # Create and return the job offer
        return JobOffer(
            company=company,
            salary=salary,
            career_growth=attributes.get('career_growth', 3),
            location_score=attributes.get('location_score', 3),
            work_life_balance=attributes.get('work_life_balance', 3),
            company_reputation=attributes.get('company_reputation', 3),
            role_responsibilities=attributes.get('role_responsibilities', 3),
            ai_component=attributes.get('ai_component', 3),
            tech_stack_match=attributes.get('tech_stack_match', 3),
            mentoring_opportunity=attributes.get('mentoring_opportunity', 3),
            project_variety=attributes.get('project_variety', 3),
            team_collaboration=attributes.get('team_collaboration', 3),
            negotiation_flexibility=negotiation_flexibility
        )


class ContinuousTimeOST:
    """
    Advanced Optimal Stopping Theory implementation using continuous time stochastic 
    processes and Bayesian updating.
    """
    
    def __init__(self, preferences: dict, max_time: float = 1.0, time_units: str = "years"):
        """
        Initialize the continuous time optimal stopping model.
        
        Args:
            preferences: User preferences dictionary
            max_time: Maximum time horizon for job search (in chosen units)
            time_units: String description of time units (for display)
        """
        self.preferences = preferences
        self.max_time = max_time
        self.time_units = time_units
        
        # Create job market model
        self.job_market = BayesianJobMarketModel(preferences)
        
        # Financial parameters
        self.current_salary = preferences.get('current_salary', 25000)
        self.financial_runway = preferences.get('financial_runway', 3) / 12  # Convert months to years
        self.discount_rate = 0.05  # Annual discount rate for future cash flows
        
        # Psychological parameters
        self.job_search_urgency = preferences.get('job_search_urgency', 5) / 10  # Normalize to 0-1
        self.risk_tolerance = preferences.get('risk_tolerance', 5) / 10  # Normalize to 0-1
        self.optimism_factor = 0.5 + (self.risk_tolerance * 0.5)  # Optimism correlates with risk tolerance
        
        # Search parameters
        self.search_cost_rate = self.current_salary * 0.05  # Cost of searching (money, effort, stress)
        self.offer_arrival_rate = 12  # Expected number of offers per year
        
        # State variables
        self.current_time = 0.0
        self.best_offer_so_far = None
        self.best_utility_so_far = -float('inf')
        self.observed_offers = []
        
        # Offer negotiation model
        self.negotiation_success_rate = 0.7  # Probability of successful negotiation
        self.negotiation_improvement = 0.05  # Average improvement from negotiation
        
        # Value function approximation
        self.time_grid = np.linspace(0, max_time, 100)
        self.value_function = np.zeros_like(self.time_grid)
        self.reservation_utilities = np.zeros_like(self.time_grid)
        
        # Initialize the value function and reservation utilities
        self._initialize_value_function()
    
    def _initialize_value_function(self):
        """Initialize the value function using backward induction."""
        # Initialize terminal value
        self.value_function[-1] = 0
        
        # Backward induction
        for i in range(len(self.time_grid) - 2, -1, -1):
            t = self.time_grid[i]
            dt = self.time_grid[i+1] - t
            
            # Expected utility calculation
            expected_improvement = self._expected_improvement_from_continuing(t)
            
            # Update value function, accounting for:
            # 1. Time value of money (discounting)
            # 2. Search costs
            # 3. Financial runway constraints
            # 4. Market condition changes
            discounted_next_value = self.value_function[i+1] * np.exp(-self.discount_rate * dt)
            search_cost = self.search_cost_rate * dt
            
            # Value of continuing the search
            continuing_value = expected_improvement + discounted_next_value - search_cost
            
            # Adjust for financial runway
            time_remaining_factor = max(0, (self.financial_runway - t) / self.financial_runway)
            if time_remaining_factor <= 0:
                # Financial pressure when runway is depleted
                runway_penalty = search_cost * (1 + 0.5 * (1 - self.risk_tolerance))
                continuing_value -= runway_penalty
            
            # Adjust for search urgency
            urgency_adjustment = (t / self.max_time) * self.job_search_urgency * expected_improvement * 0.5
            continuing_value -= urgency_adjustment
            
            # Store the continuing value
            self.value_function[i] = continuing_value
            
            # Calculate the reservation utility - threshold for accepting an offer
            self.reservation_utilities[i] = self._calculate_reservation_utility(t, continuing_value)
    
    def _expected_improvement_from_continuing(self, t: float) -> float:
        """Calculate the expected improvement in utility from continuing the search."""
        # Generate sample offers to estimate expected value
        n_samples = 100
        sample_offers = [self.job_market.generate_job_offer(t) for _ in range(n_samples)]
        sample_utilities = [offer.calculate_utility(self.preferences) for offer in sample_offers]
        
        # If we have a best offer so far, we only benefit from better offers
        if self.best_offer_so_far:
            best_utility = self.best_offer_so_far.calculate_utility(self.preferences)
            improvements = [max(0, u - best_utility) for u in sample_utilities]
        else:
            improvements = [max(0, u) for u in sample_utilities]
        
        # Expected improvement adjusted for arrival rate
        arrival_prob = 1 - np.exp(-self.offer_arrival_rate * (self.max_time - t) / 100)
        expected_improvement = np.mean(improvements) * arrival_prob
        
        return expected_improvement
    
    def _calculate_reservation_utility(self, t: float, continuing_value: float) -> float:
        """
        Calculate the reservation utility (optimal stopping threshold) at time t.
        
        Args:
            t: Current time point
            continuing_value: Value of continuing the search
            
        Returns:
            The reservation utility (threshold for accepting an offer)
        """
        # Basic reservation utility is the value of continuing
        reservation_utility = continuing_value
        
        # Adjust for time remaining
        time_factor = 1.0
        if t < 0.2 * self.max_time:
            # Be more selective in early phases
            time_factor = 1.2
        elif t > 0.8 * self.max_time:
            # Be less selective near the end
            time_factor = 0.8
        
        # Adjust for financial runway
        runway_remaining = self.financial_runway - t
        runway_factor = 1.0
        if runway_remaining < 0:
            # Much less selective once runway is gone
            runway_factor = 0.5
        elif runway_remaining < 0.2:
            # Somewhat less selective when runway is low
            runway_factor = 0.8
        
        # Adjust for market conditions (if rapidly worsening, be less selective)
        season_idx = int((t * 52) // 3) % 16
        current_factor = self.job_market.seasonality_factors[season_idx]
        next_idx = (season_idx + 1) % 16
        next_factor = self.job_market.seasonality_factors[next_idx]
        market_trend = next_factor / current_factor
        
        market_factor = 1.0
        if market_trend < 0.95:
            # Market worsening - be less selective
            market_factor = 0.9
        elif market_trend > 1.05:
            # Market improving - be more selective
            market_factor = 1.1
        
        # Final adjusted reservation utility
        adjusted_utility = reservation_utility * time_factor * runway_factor * market_factor
        
        # Lower bound - never accept terrible offers
        min_acceptable = -5 * (1 - self.risk_tolerance)
        
        return max(min_acceptable, adjusted_utility)
    
    def get_reservation_utility(self, t: float) -> float:
        """
        Get the reservation utility (acceptance threshold) at time t.
        
        Args:
            t: Current time point
            
        Returns:
            The reservation utility at time t
        """
        # Find the closest point in our grid
        idx = np.argmin(np.abs(self.time_grid - t))
        return self.reservation_utilities[idx]
    
    def should_accept_offer(self, offer: JobOffer, t: float, attempt_negotiation: bool = True) -> Tuple[bool, str]:
        """
        Determine whether to accept a job offer, potentially after negotiation.
        
        Args:
            offer: The job offer to evaluate
            t: Current time point
            attempt_negotiation: Whether to attempt negotiation
            
        Returns:
            Tuple of (decision, explanation)
        """
        # Calculate the utility of this offer
        utility = offer.calculate_utility(self.preferences)
        
        # Get the reservation utility at this time
        reservation_utility = self.get_reservation_utility(t)
        
        # Check if offer exceeds best so far
        improves_on_best = False
        if self.best_offer_so_far:
            best_utility = self.best_offer_so_far.calculate_utility(self.preferences)
            improves_on_best = utility > best_utility
        else:
            improves_on_best = utility > 0
        
        # If offer isn't good enough, try negotiation
        negotiated_utility = utility
        negotiation_message = ""
        
        if attempt_negotiation and offer.negotiation_flexibility > 0:
            potential_improvement = offer.negotiation_flexibility * self.negotiation_improvement
            if np.random.random() < self.negotiation_success_rate:
                # Successful negotiation
                improved_salary = offer.salary * (1 + potential_improvement)
                
                # Create a new offer with the improved salary
                negotiated_offer = JobOffer(
                    company=offer.company,
                    salary=improved_salary,
                    career_growth=offer.career_growth,
                    location_score=offer.location_score,
                    work_life_balance=offer.work_life_balance,
                    company_reputation=offer.company_reputation,
                    role_responsibilities=offer.role_responsibilities,
                    ai_component=offer.ai_component,
                    tech_stack_match=offer.tech_stack_match,
                    mentoring_opportunity=offer.mentoring_opportunity,
                    project_variety=offer.project_variety,
                    team_collaboration=offer.team_collaboration,
                    negotiation_flexibility=0  # Used up flexibility
                )
                
                # Calculate new utility
                negotiated_utility = negotiated_offer.calculate_utility(self.preferences)
                negotiation_message = f"Negotiation successful: Salary increased from £{offer.salary:,.2f} to £{improved_salary:,.2f}"
            else:
                negotiation_message = "Negotiation attempted but unsuccessful"
        
        # Decision logic
        if negotiated_utility >= reservation_utility:
            if improves_on_best:
                return True, f"Accept: Utility ({negotiated_utility:.2f}) exceeds threshold ({reservation_utility:.2f}) and improves on best offer. {negotiation_message}"
            else:
                return True, f"Accept: Utility ({negotiated_utility:.2f}) exceeds threshold ({reservation_utility:.2f}). {negotiation_message}"
        else:
            if t > self.financial_runway:
                # Financial pressure - lower standards
                desperation_factor = 0.8
                adjusted_threshold = reservation_utility * desperation_factor
                
                if negotiated_utility >= adjusted_threshold:
                    return True, f"Accept under financial pressure: Utility ({negotiated_utility:.2f}) exceeds adjusted threshold ({adjusted_threshold:.2f}). {negotiation_message}"
            
            return False, f"Reject: Utility ({negotiated_utility:.2f}) below threshold ({reservation_utility:.2f}). {negotiation_message}"
    
    def observe_offer(self, offer: JobOffer, t: float, interview_successful: bool = None):
        """
        Observe a new job offer and update our models.
        
        Args:
            offer: The observed job offer
            t: Time when the offer was observed
            interview_successful: Whether the interview was successful (if known)
        """
        # Update our current time
        self.current_time = t
        
        # Store the offer
        self.observed_offers.append((offer, t))
        
        # Update our job market model
        self.job_market.update_with_offer(offer, t, interview_successful)
        
        # Track best offer so far
        utility = offer.calculate_utility(self.preferences)
        if utility > self.best_utility_so_far:
            self.best_offer_so_far = offer
            self.best_utility_so_far = utility
        
        # Recalculate our value function with updated information
        self._initialize_value_function()
    
    def plot_strategy(self):
        """Plot the job search strategy and offers."""
        plt.figure(figsize=(12, 8))
        
        # Plot reservation utility curve
        plt.plot(self.time_grid, self.reservation_utilities, 'b-', label='Reservation Utility')
        
        # Plot value function
        plt.plot(self.time_grid, self.value_function, 'g--', label='Value of Continuing Search')
        
        # Plot financial runway
        plt.axvline(x=self.financial_runway, color='r', linestyle='--', 
                   label=f'Financial Runway ({self.financial_runway:.2f} {self.time_units})')
        
        # Plot observed offers
        if self.observed_offers:
            offer_times = [t for _, t in self.observed_offers]
            offer_utilities = [o.calculate_utility(self.preferences) for o, _ in self.observed_offers]
            plt.scatter(offer_times, offer_utilities, color='orange', s=50, label='Observed Offers')
            
            # Highlight best offer
            if self.best_offer_so_far:
                best_utility = self.best_utility_so_far
                idx = offer_utilities.index(best_utility)
                plt.scatter([offer_times[idx]], [best_utility], color='green', s=100, 
                           label=f'Best Offer: {self.best_offer_so_far.company}')
        
        # Add seasonality effect on secondary axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Convert seasonality to continuous function
        season_t = np.linspace(0, 1, 16)
        season_values = self.job_market.seasonality_factors
        season_interp = np.interp(np.linspace(0, 1, 100), season_t, season_values)
        
        # Repeat the seasonality for the full time scale
        repeated_seasons = np.tile(season_interp, int(np.ceil(self.max_time)))[:len(self.time_grid)]
        ax2.plot(self.time_grid, repeated_seasons, 'k:', alpha=0.5, label='Market Seasonality')
        ax2.set_ylabel('Market Condition Factor')
        ax2.set_ylim(0.8, 1.2)
        
        # Legends and labels
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('Continuous-Time Optimal Stopping Strategy for Job Search')
        plt.xlabel(f'Time ({self.time_units})')
        plt.ylabel('Utility')
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def get_strategy_summary(self) -> Dict:
        """Generate a summary of the search strategy."""
        # Divide the time horizon into phases
        early_phase = self.time_grid[self.time_grid < 0.33 * self.max_time]
        mid_phase = self.time_grid[(self.time_grid >= 0.33 * self.max_time) & 
                                  (self.time_grid < 0.67 * self.max_time)]
        late_phase = self.time_grid[self.time_grid >= 0.67 * self.max_time]
        
        # Get corresponding utilities
        early_utils = [self.get_reservation_utility(t) for t in early_phase]
        mid_utils = [self.get_reservation_utility(t) for t in mid_phase]
        late_utils = [self.get_reservation_utility(t) for t in late_phase]
        
        # Analyze
        early_avg = np.mean(early_utils) if early_utils else 0
        mid_avg = np.mean(mid_utils) if mid_utils else 0
        late_avg = np.mean(late_utils) if late_utils else 0
        
        utility_decline_rate = (early_avg - late_avg) / self.max_time if self.max_time > 0 else 0
        
        # Create runway milestones
        runway_reached_time = min(self.financial_runway, self.max_time)
        runway_threshold = self.get_reservation_utility(runway_reached_time)
        
        return {
            "initial_threshold": self.reservation_utilities[0],
            "final_threshold": self.reservation_utilities[-1],
            "early_phase_avg_threshold": early_avg,
            "mid_phase_avg_threshold": mid_avg,
            "late_phase_avg_threshold": late_avg,
            "threshold_decline_rate": utility_decline_rate,
            "financial_runway_time": self.financial_runway,
            "financial_runway_threshold": runway_threshold,
            "best_utility_so_far": self.best_utility_so_far if self.best_utility_so_far > -float('inf') else None,
            "market_condition": self.job_market.market_condition,
            "candidate_desirability": self.job_market.negotiation_model['candidate_desirability'],
        }


class AdvancedJobSearchOptimizer:
    """
    Advanced job search optimizer that integrates continuous-time OST with
    Bayesian updating and strategic interactions.
    """
    
    def __init__(self, preferences: dict, max_time: float = 1.0, time_units: str = "years"):
        """
        Initialize the advanced job search optimizer.
        
        Args:
            preferences: User preferences dictionary
            max_time: Maximum time for job search
            time_units: Units for time (for display)
        """
        self.preferences = preferences
        self.max_time = max_time
        self.time_units = time_units
        
        # Create the OST model
        self.ost = ContinuousTimeOST(preferences, max_time, time_units)
        
        # Tracking variables
        self.current_time = 0.0
        self.accepted_offer = None
        self.accepted_time = None
        self.search_history = []
        
        # Decision variables
        self.application_strategy = {
            'time_per_application': 0.01,  # Time units spent per application
            'max_concurrent': 5,  # Maximum number of concurrent applications
            'success_probability': 0.3,  # Base probability of successful application
        }
        
        # Active applications
        self.active_applications = []  # [{company, start_time, progress, ...}]
        
        # For access in methods
        self.time_grid = self.ost.time_grid
        self.value_function = self.ost.value_function
    
    def strategic_search_step(self, time_step: float = 0.05):
        """
        Execute a strategic job search step, including:
        1. Processing active applications
        2. Starting new applications
        3. Evaluating offers
        4. Updating search strategy
        
        Args:
            time_step: Amount of time to simulate in this step
            
        Returns:
            Dictionary with information about this step
        """
        start_time = self.current_time
        end_time = min(self.max_time, start_time + time_step)
        self.current_time = end_time
        
        # Dictionary to store information about this step
        step_info = {
            'start_time': start_time,
            'end_time': end_time,
            'time_step': time_step,
            'offers_received': [],
            'applications_started': [],
            'applications_rejected': [],
            'applications_completed': [],
            'negotiation_attempts': [],
            'decisions': [],
            'strategy_updates': [],
            'reservation_utility': self.ost.get_reservation_utility(start_time),
        }
        
        # 1. Process existing applications
        self._process_applications(time_step, step_info)
        
        # 2. Generate new applications if capacity allows
        if len(self.active_applications) < self.application_strategy['max_concurrent']:
            self._generate_applications(time_step, step_info)
        
        # 3. Update search strategy based on new information
        self._update_search_strategy(step_info)
        
        # Add to search history
        self.search_history.append(step_info)
        
        return step_info
    
    def _process_applications(self, time_step: float, step_info: Dict):
        """Process active applications, advancing them or generating offers."""
        still_active = []
        
        for app in self.active_applications:
            # Advance progress
            app['progress'] += time_step / self.application_strategy['time_per_application']
            
            # Check for completion or rejection
            if app['progress'] >= 1.0:
                # Application process complete - determine if successful
                success_prob = app['base_success_prob'] * self.ost.job_market.negotiation_model['candidate_desirability']
                
                # Apply market condition factor to success probability
                market_factor = self.ost.job_market.market_condition
                adjusted_success_prob = success_prob * market_factor
                
                # Generate a detailed reason for success/failure
                company_tier = "top" if app['offer_model'].company_reputation >= 4 else \
                              "good" if app['offer_model'].company_reputation >= 3 else \
                              "average" if app['offer_model'].company_reputation >= 2 else "below_average"
                
                tech_match = app['offer_model'].tech_stack_match
                
                # Random failure scenarios for rejections
                failure_scenarios = [
                    f"Not enough experience for the {company_tier}-tier company",
                    f"Another candidate was a better fit for {app['company']}'s needs",
                    f"Limited technical match (your tech stack match was {tech_match}/5)",
                    f"Cultural fit concerns after the final interview round",
                    f"Position was filled internally or put on hold",
                    f"Budgetary constraints led to a hiring freeze",
                    f"Organizational restructuring affected the hiring process"
                ]
                
                success_scenarios = [
                    f"Strong interview performance impressed the hiring team",
                    f"Your background aligned well with {app['company']}'s needs",
                    f"Your tech stack expertise was highly valued (match: {tech_match}/5)",
                    f"Good rapport with the hiring manager during interviews",
                    f"Company is actively expanding and needs to fill positions quickly"
                ]
                
                # Determine outcome first
                is_success = np.random.random() < adjusted_success_prob
                
                # Select reason based on outcome
                if is_success:
                    reason = random.choice(success_scenarios)
                else:
                    reason = random.choice(failure_scenarios)
                
                if is_success:
                    # Success - generate a job offer
                    offer = self.ost.job_market.generate_job_offer(self.current_time)
                    
                    # Use the company name from the application
                    offer.company = app['company']
                    
                    # Record the offer with detailed attributes
                    step_info['offers_received'].append({
                        'company': offer.company,
                        'salary': offer.salary,
                        'attributes': {
                            'career_growth': offer.career_growth,
                            'work_life_balance': offer.work_life_balance,
                            'company_reputation': offer.company_reputation,
                            'role_responsibilities': offer.role_responsibilities,
                            'ai_component': offer.ai_component,
                            'tech_stack_match': offer.tech_stack_match,
                            'mentoring_opportunity': offer.mentoring_opportunity,
                            'project_variety': offer.project_variety,
                            'team_collaboration': offer.team_collaboration
                        },
                        'negotiation_flexibility': offer.negotiation_flexibility,
                        'success_reason': reason
                    })
                    
                    # Update our OST model with this offer
                    self.ost.observe_offer(offer, self.current_time, True)
                    
                    # Decide whether to accept
                    should_accept, explanation = self.ost.should_accept_offer(offer, self.current_time)
                    
                    # Record decision with detailed explanation
                    decision_info = {
                        'offer': offer.company,
                        'salary': offer.salary,
                        'utility': offer.calculate_utility(self.preferences),
                        'threshold': self.ost.get_reservation_utility(self.current_time),
                        'decision': 'accept' if should_accept else 'reject',
                        'explanation': explanation,
                        'offer_details': {
                            'career_growth': offer.career_growth,
                            'work_life_balance': offer.work_life_balance,
                            'company_reputation': offer.company_reputation,
                            'role_responsibilities': offer.role_responsibilities,
                            'ai_component': offer.ai_component,
                            'tech_stack_match': offer.tech_stack_match
                        }
                    }
                    
                    # Add specific reasons for rejection if applicable
                    if not should_accept:
                        low_attributes = []
                        for attr, value in decision_info['offer_details'].items():
                            weight_key = f"{attr}_weight" if attr != "ai_component" else "ai_application_interest"
                            weight = self.preferences.get(weight_key, 3)
                            if weight >= 4 and value <= 2:  # High importance but low score
                                low_attributes.append(f"{attr.replace('_', ' ')} is only {value}/5 but importance is {weight}/5")
                        
                        if low_attributes:
                            decision_info['low_attributes'] = low_attributes
                    
                    step_info['decisions'].append(decision_info)
                    
                    # If accepting, end the search
                    if should_accept:
                        self.accepted_offer = offer
                        self.accepted_time = self.current_time
                else:
                    # Rejection
                    step_info['applications_rejected'].append({
                        'company': app['company'],
                        'time_invested': app['progress'] * self.application_strategy['time_per_application'],
                        'reason': reason,
                        'market_factor': market_factor
                    })
                    
                    # Update our OST model with this rejection
                    self.ost.job_market.update_with_offer(app['offer_model'], self.current_time, False)
                
                # Record completion with detailed information
                outcome = 'offer' if is_success else 'rejection'
                completion_info = {
                    'company': app['company'],
                    'outcome': outcome,
                    'time_invested': app['progress'] * self.application_strategy['time_per_application'],
                    'company_tier': company_tier,
                    'tech_match': tech_match,
                    'success_probability': f"{adjusted_success_prob:.1%}",
                    'reason': reason
                }
                    
                step_info['applications_completed'].append(completion_info)
            else:
                # Application still in progress
                still_active.append(app)
        
        # Update active applications
        self.active_applications = still_active
    
    def _generate_applications(self, time_step: float, step_info: Dict):
        """Generate new job applications based on current strategy."""
        # Calculate number of new applications to start
        capacity = self.application_strategy['max_concurrent'] - len(self.active_applications)
        
        # Limit by how many we can realistically start in this time step
        max_new = int(time_step / (self.application_strategy['time_per_application'] * 0.2))
        num_new = min(capacity, max_new)
        
        # Generate applications
        for _ in range(num_new):
            # Create a model of what the offer might look like
            offer_model = self.ost.job_market.generate_job_offer(self.current_time)
            
            # Adjust success probability based on company tier and candidate desirability
            base_success_prob = self.application_strategy['success_probability']
            
            # Success probability increases with worse company reputation
            # (easier to get offers from less prestigious companies)
            company_factor = 1.0 + (0.1 * (6 - offer_model.company_reputation))
            base_success_prob *= company_factor
            
            # Create application
            app = {
                'company': offer_model.company,
                'start_time': self.current_time,
                'progress': 0.0,
                'offer_model': offer_model,
                'base_success_prob': base_success_prob,
            }
            
            self.active_applications.append(app)
            
            # Record in step info
            step_info['applications_started'].append({
                'company': app['company'],
                'estimated_salary': offer_model.salary,
                'estimated_career_growth': offer_model.career_growth,
            })
    
    def _update_search_strategy(self, step_info: Dict):
        """Update search strategy based on evolving market model."""
        # Get current market condition
        market_condition = self.ost.job_market.market_condition
        
        # Adjust max concurrent applications based on market conditions
        if market_condition < 0.9:
            # Poor market - apply to more places
            new_max = min(8, self.application_strategy['max_concurrent'] + 1)
            if new_max != self.application_strategy['max_concurrent']:
                self.application_strategy['max_concurrent'] = new_max
                step_info['strategy_updates'].append({
                    'parameter': 'max_concurrent_applications',
                    'old_value': new_max - 1,
                    'new_value': new_max,
                    'reason': 'Poor market conditions - increasing application volume'
                })
        elif market_condition > 1.1 and self.application_strategy['max_concurrent'] > 3:
            # Good market - be more selective
            new_max = self.application_strategy['max_concurrent'] - 1
            self.application_strategy['max_concurrent'] = new_max
            step_info['strategy_updates'].append({
                'parameter': 'max_concurrent_applications',
                'old_value': new_max + 1,
                'new_value': new_max,
                'reason': 'Strong market conditions - focusing on quality over quantity'
            })
        
        # Adjust time per application as search progresses
        runway_remaining = max(0, self.ost.financial_runway - self.current_time)
        if runway_remaining < 0.2 and self.application_strategy['time_per_application'] > 0.005:
            # Running out of runway - spend less time per application
            old_value = self.application_strategy['time_per_application']
            new_value = max(0.005, old_value * 0.9)
            self.application_strategy['time_per_application'] = new_value
            
            step_info['strategy_updates'].append({
                'parameter': 'time_per_application',
                'old_value': old_value,
                'new_value': new_value,
                'reason': 'Financial pressure increasing - accelerating application process'
            })
    
    def run_full_search(self, verbose=True) -> Dict:
        """
        Run the full job search until an offer is accepted or time runs out.
        
        Args:
            verbose: Whether to print detailed information during the search
            
        Returns:
            Dictionary with search results
        """
        # Reset state
        self.current_time = 0.0
        self.accepted_offer = None
        self.accepted_time = None
        self.search_history = []
        self.active_applications = []
        
        if verbose:
            print("\n===== BEGINNING JOB SEARCH SIMULATION =====")
            print(f"Financial runway: {self.ost.financial_runway:.2f} {self.time_units}")
            print(f"Initial reservation utility: {self.ost.get_reservation_utility(0):.2f}")
            print(f"Job search urgency: {self.ost.job_search_urgency:.2f} (0-1 scale)")
            print(f"Risk tolerance: {self.ost.risk_tolerance:.2f} (0-1 scale)")
            print("Starting search process...\n")
        
        # Run search steps until we accept an offer or run out of time
        step_counter = 0
        while self.current_time < self.max_time and not self.accepted_offer:
            step_counter += 1
            step_info = self.strategic_search_step()
            
            if verbose:
                self._print_step_details(step_counter, step_info)
                
                # Add pause every few steps when printing to make it easier to follow
                if step_counter % 5 == 0:
                    print(f"\n--- SEARCH PROGRESS: {self.current_time:.2f}/{self.max_time} {self.time_units} elapsed ---\n")
        
        # Compile results
        results = {
            'accepted_offer': None,
            'search_duration': self.current_time,
            'num_steps': len(self.search_history),
            'applications_started': sum(len(step['applications_started']) for step in self.search_history),
            'offers_received': sum(len(step['offers_received']) for step in self.search_history),
            'rejections_received': sum(len(step['applications_rejected']) for step in self.search_history),
            'financial_runway': self.ost.financial_runway,
            'strategy_summary': self.ost.get_strategy_summary(),
        }
        
        if self.accepted_offer:
            results['accepted_offer'] = {
                'company': self.accepted_offer.company,
                'salary': self.accepted_offer.salary,
                'career_growth': self.accepted_offer.career_growth,
                'work_life_balance': self.accepted_offer.work_life_balance,
                'company_reputation': self.accepted_offer.company_reputation,
                'role_responsibilities': self.accepted_offer.role_responsibilities,
                'utility': self.accepted_offer.calculate_utility(self.preferences),
                'time_accepted': self.accepted_time,
            }
            
            # Check if accepted within financial runway
            results['accepted_within_runway'] = self.accepted_time <= self.ost.financial_runway
            
            if verbose:
                print("\n===== OFFER ACCEPTED =====")
                print(f"Company: {self.accepted_offer.company}")
                print(f"Salary: £{self.accepted_offer.salary:,.2f}")
                print(f"Career Growth: {self.accepted_offer.career_growth}/5")
                print(f"Work-Life Balance: {self.accepted_offer.work_life_balance}/5")
                print(f"Company Reputation: {self.accepted_offer.company_reputation}/5")
                print(f"Role Responsibilities: {self.accepted_offer.role_responsibilities}/5")
                print(f"AI Component: {self.accepted_offer.ai_component}/5")
                print(f"Tech Stack Match: {self.accepted_offer.tech_stack_match}/5")
                print(f"Mentoring Opportunity: {self.accepted_offer.mentoring_opportunity}/5")
                print(f"Project Variety: {self.accepted_offer.project_variety}/5")
                print(f"Team Collaboration: {self.accepted_offer.team_collaboration}/5")
                print(f"Utility: {self.accepted_offer.calculate_utility(self.preferences):.2f}")
                print(f"Accepted at: {self.accepted_time:.2f} {self.time_units}")
                if results['accepted_within_runway']:
                    print(f"Accepted within financial runway of {self.ost.financial_runway:.2f} {self.time_units}")
                else:
                    time_past_runway = self.accepted_time - self.ost.financial_runway
                    print(f"Accepted {time_past_runway:.2f} {self.time_units} AFTER financial runway was depleted")
        else:
            if verbose:
                print("\n===== SEARCH COMPLETED WITHOUT ACCEPTING ANY OFFER =====")
                print(f"Searched for the entire {self.max_time} {self.time_units} without finding an acceptable offer")
                print(f"Total applications submitted: {results['applications_started']}")
                print(f"Total offers received: {results['offers_received']}")
                print(f"Total rejections received: {results['rejections_received']}")
                
                # Report on best offer seen
                if self.ost.best_offer_so_far:
                    best = self.ost.best_offer_so_far
                    print("\nBest offer encountered during search:")
                    print(f"Company: {best.company}")
                    print(f"Salary: £{best.salary:,.2f}")
                    print(f"Utility: {self.ost.best_utility_so_far:.2f}")
                    print(f"This offer was below your reservation utility threshold when received")
        
        if verbose:
            self._print_search_summary(results)
        
        return results
        
    def _print_step_details(self, step_number, step_info):
        """Print detailed information about a search step."""
        time_point = step_info['end_time']
        
        print(f"STEP {step_number} - Time: {time_point:.2f} {self.time_units}")
        
        # Print financial status
        runway_remaining = self.ost.financial_runway - time_point
        if runway_remaining > 0:
            print(f"Financial runway: {runway_remaining:.2f} {self.time_units} remaining")
        else:
            print(f"Financial runway: DEPLETED (exceeded by {abs(runway_remaining):.2f} {self.time_units})")
        
        # Print current threshold
        current_threshold = step_info['reservation_utility']
        print(f"Current reservation utility threshold: {current_threshold:.2f}")
        
        # Print market condition
        print(f"Current market condition factor: {self.ost.job_market.market_condition:.2f}")
        
        # Print new applications
        if step_info['applications_started']:
            print("\nNew applications started:")
            for i, app in enumerate(step_info['applications_started']):
                print(f"  {i+1}. {app['company']} - Est. salary: £{app['estimated_salary']:,.2f}, " +
                      f"Est. career growth: {app['estimated_career_growth']}/5")
        else:
            print("\nNo new applications started this period")
        
        # Print application rejections
        if step_info['applications_rejected']:
            print("\nApplications rejected:")
            for i, rej in enumerate(step_info['applications_rejected']):
                print(f"  {i+1}. {rej['company']} - Time invested: {rej['time_invested']:.2f} {self.time_units}")
                print(f"     REASON: Failed during interview/assessment process")
        
        # Print offers received
        if step_info['offers_received']:
            print("\nOffers received:")
            for i, offer in enumerate(step_info['offers_received']):
                print(f"  {i+1}. {offer['company']} - Salary: £{offer['salary']:,.2f}")
                print(f"     Key attributes: Career Growth: {offer['attributes']['career_growth']}/5, " +
                      f"Work-Life Balance: {offer['attributes']['work_life_balance']}/5, " +
                      f"Company Reputation: {offer['attributes']['company_reputation']}/5")
                print(f"     Negotiation flexibility: {offer['negotiation_flexibility']:.2f}")
        
        # Print decision details
        if step_info['decisions']:
            print("\nDecisions made:")
            for i, decision in enumerate(step_info['decisions']):
                print(f"  {i+1}. {decision['offer']} - Salary: £{decision['salary']:,.2f}")
                print(f"     Utility: {decision['utility']:.2f} vs Threshold: {decision['threshold']:.2f}")
                print(f"     DECISION: {decision['decision'].upper()}")
                print(f"     REASON: {decision['explanation']}")
        
        # Print strategy updates
        if step_info['strategy_updates']:
            print("\nStrategy updates:")
            for i, update in enumerate(step_info['strategy_updates']):
                print(f"  {i+1}. {update['parameter']}: {update['old_value']} → {update['new_value']}")
                print(f"     REASON: {update['reason']}")
        
        # Print current active applications
        active_count = len(self.active_applications)
        if active_count > 0:
            print(f"\nCurrently tracking {active_count} active applications")
            for i, app in enumerate(self.active_applications):
                progress_pct = app['progress'] * 100
                print(f"  {i+1}. {app['company']} - Progress: {progress_pct:.1f}%")
        
        print("\n" + "-" * 80 + "\n")
    
    def _print_search_summary(self, results):
        """Print a summary of the entire search process."""
        print("\n===== SEARCH SUMMARY =====")
        print(f"Search duration: {results['search_duration']:.2f} {self.time_units}")
        print(f"Search steps: {results['num_steps']}")
        print(f"Applications submitted: {results['applications_started']}")
        print(f"Offers received: {results['offers_received']}")
        print(f"Rejections received: {results['rejections_received']}")
        
        # Performance analysis
        success_rate = results['offers_received'] / max(1, results['applications_started']) * 100
        print(f"Application success rate: {success_rate:.1f}%")
        
        # Financial analysis
        runway = self.ost.financial_runway
        if results['search_duration'] <= runway:
            print(f"Search completed with {runway - results['search_duration']:.2f} {self.time_units} of runway remaining")
        else:
            print(f"Search extended {results['search_duration'] - runway:.2f} {self.time_units} beyond financial runway")
        
        # Timeline statistics
        timeline_buckets = 4
        bucket_size = self.max_time / timeline_buckets
        print("\nActivity distribution over time:")
        
        for i in range(timeline_buckets):
            start_time = i * bucket_size
            end_time = (i+1) * bucket_size
            
            # Count activities in this time bucket
            applications = sum(1 for step in self.search_history 
                            if start_time <= step['end_time'] < end_time
                            for _ in step['applications_started'])
            
            offers = sum(1 for step in self.search_history 
                        if start_time <= step['end_time'] < end_time
                        for _ in step['offers_received'])
            
            rejections = sum(1 for step in self.search_history 
                          if start_time <= step['end_time'] < end_time
                          for _ in step['applications_rejected'])
            
            print(f"  Period {i+1} ({start_time:.2f}-{end_time:.2f} {self.time_units}):")
            print(f"    Applications: {applications}, Offers: {offers}, Rejections: {rejections}")
        
        # Threshold evolution
        strategy = results['strategy_summary']
        print("\nThreshold evolution:")
        print(f"  Initial threshold: {strategy['initial_threshold']:.2f}")
        print(f"  Early phase avg: {strategy['early_phase_avg_threshold']:.2f}")
        print(f"  Mid phase avg: {strategy['mid_phase_avg_threshold']:.2f}")
        print(f"  Late phase avg: {strategy['late_phase_avg_threshold']:.2f}")
        print(f"  Final threshold: {strategy['final_threshold']:.2f}")
        
        # Runway threshold
        print(f"  Threshold at financial runway: {strategy['financial_runway_threshold']:.2f}")
        
        # Market conditions
        print(f"\nFinal market condition factor: {strategy['market_condition']:.2f}")
        print(f"Final candidate desirability: {strategy['candidate_desirability']:.2f}")
        
        if results['accepted_offer']:
            accepted = results['accepted_offer']
            # Calculate how much utility was gained compared to:
            # 1. Initial threshold
            utility_gain_vs_initial = accepted['utility'] - strategy['initial_threshold']
            # 2. Theoretical continuing value at time of acceptance
            time_of_acceptance = accepted['time_accepted']
            idx = min(len(self.time_grid) - 1, 
                    int(time_of_acceptance / self.max_time * (len(self.time_grid) - 1)))
            continuing_value = self.value_function[idx]
            utility_gain_vs_continuing = accepted['utility'] - continuing_value
            
            print("\nOffer analysis:")
            print(f"  Accepted utility vs initial threshold: {utility_gain_vs_initial:+.2f}")
            print(f"  Accepted utility vs theoretical continuing value: {utility_gain_vs_continuing:+.2f}")
            
            if utility_gain_vs_continuing > 0:
                print("  CONCLUSION: Made an optimal decision to accept this offer")
            else:
                print("  CONCLUSION: Theoretically could have gained more by continuing search")
                print("              but practical constraints may justify the decision")
    
    def plot_search_trajectory(self):
        """Plot the complete search trajectory, including offers and decisions."""
        plt = self.ost.plot_strategy()
        
        # Add specific points where decisions were made
        decision_times = []
        decision_utilities = []
        decision_outcomes = []
        
        for step in self.search_history:
            for decision in step.get('decisions', []):
                decision_times.append(step['end_time'])
                decision_utilities.append(decision['utility'])
                decision_outcomes.append(decision['decision'])
        
        # Plot accept/reject decisions
        for i, outcome in enumerate(decision_outcomes):
            color = 'green' if outcome == 'accept' else 'red'
            marker = 'o' if outcome == 'accept' else 'x'
            plt.scatter([decision_times[i]], [decision_utilities[i]], 
                       color=color, marker=marker, s=100)
        
        # Highlight the accepted offer if any
        if self.accepted_offer:
            utility = self.accepted_offer.calculate_utility(self.preferences)
            plt.scatter([self.accepted_time], [utility], color='lime', 
                       marker='*', s=200, label='Accepted Offer')
        
        # Update title
        if self.accepted_offer:
            plt.suptitle(f'Job Search Trajectory - Offer Accepted after {self.accepted_time:.2f} {self.time_units}')
        else:
            plt.suptitle(f'Job Search Trajectory - No Offer Accepted within {self.max_time} {self.time_units}')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
        return plt


def explain_advanced_ost_algorithm():
    """Explain the advanced Optimal Stopping Theory algorithm."""
    explanation = """
===== ADVANCED OPTIMAL STOPPING THEORY EXPLANATION =====

This advanced job search optimizer represents a significant leap forward from basic OST implementations:

1. CONTINUOUS-TIME STOCHASTIC MODELING:
   - Instead of discrete weeks, we model time as a continuous variable
   - Job offers arrive as a stochastic process (Poisson arrival process)
   - Value functions and thresholds are calculated for every point in time
   - Financial constraints modeled as continuous functions rather than discrete steps

2. BAYESIAN LEARNING MODEL:
   - Market conditions are unknown at the start but learned through observations
   - Each observed job offer updates our beliefs about:
     * The distribution of salaries in the market
     * The distribution of job attributes (work-life balance, career growth, etc.)
     * Market seasonality and trends
   - Model becomes more accurate as more data is gathered during the search

3. STRATEGIC INTERACTIONS WITH EMPLOYERS:
   - Models employer negotiation strategies
   - Considers candidate desirability and employer urgency
   - Allows for strategic negotiation based on candidate leverage
   - Adjusts strategies based on predicted interview success probability

4. NON-STATIONARY PROCESSES:
   - Job market evolves over time (seasonal effects, market trends)
   - Candidate skills and desirability change over time
   - Financial pressure increases non-linearly as runway depletes
   - Search costs and discount rates affect value of future opportunities

5. ADAPTIVE DECISION MAKING:
   - Continuously updates optimal stopping thresholds based on new information
   - Adapts application strategy based on market conditions and financial status
   - Uses probability theory to balance exploration (searching) and exploitation (accepting)
   - Considers opportunity costs of time spent searching vs. accepting suboptimal offers

This model represents a theoretical optimal search strategy based on advanced stochastic control theory
and dynamic programming. The mathematics behind it incorporate:

- Martingale theory for optimal stopping problems
- Bayesian updating for sequential decision making
- Stochastic differential equations for continuous time processes
- Utility theory for multi-attribute decision making

The algorithm adjusts to each candidate's unique preferences and financial constraints while
accounting for the fundamental uncertainty in job searching.
"""
    return explanation

def run_advanced_simulation(preferences: dict, verbose_timeline=True):
    """
    Run an advanced job search simulation with visualizations.
    
    Args:
        preferences: Dictionary of user preferences
        verbose_timeline: Whether to print detailed step-by-step timeline
    """
    # Create optimizer with 1 year time horizon
    optimizer = AdvancedJobSearchOptimizer(preferences, max_time=1.0, time_units="years")
    
    # Run the full search with detailed output
    results = optimizer.run_full_search(verbose=verbose_timeline)
    
    # Create plots
    plt = optimizer.plot_search_trajectory()
    plt.savefig("advanced_job_search_trajectory.png")
    
    # Print simulation summary (if not already printed in verbose mode)
    if not verbose_timeline:
        print("\n===== ADVANCED JOB SEARCH SIMULATION RESULTS =====")
        if results['accepted_offer']:
            print(f"Found acceptable job offer after {results['search_duration']:.2f} years:")
            print(f"Company: {results['accepted_offer']['company']}")
            print(f"Salary: £{results['accepted_offer']['salary']:,.2f}")
            print(f"Career Growth Score: {results['accepted_offer']['career_growth']}/5")
            print(f"Work-Life Balance: {results['accepted_offer']['work_life_balance']}/5")
            print(f"Utility: {results['accepted_offer']['utility']:.2f}")
            
            if results['accepted_within_runway']:
                print(f"Accepted within financial runway ({optimizer.ost.financial_runway:.2f} years)")
            else:
                print(f"Accepted after financial runway depleted")
        else:
            print(f"No acceptable job offer found within {optimizer.max_time} years.")
        
        print("\n===== SEARCH STATISTICS =====")
        print(f"Applications submitted: {results['applications_started']}")
        print(f"Offers received: {results['offers_received']}")
        print(f"Rejections received: {results['rejections_received']}")
        
        # Print strategy summary
        strategy = results['strategy_summary']
        print("\n===== SEARCH STRATEGY ANALYSIS =====")
        print(f"Initial utility threshold: {strategy['initial_threshold']:.2f}")
        print(f"Final utility threshold: {strategy['final_threshold']:.2f}")
        print(f"Threshold decline rate: {strategy['threshold_decline_rate']:.2f} per year")
    
    # Print detailed application history
    print("\n===== DETAILED APPLICATION TIMELINE =====")
    print("* Applications submitted in chronological order:")
    
    # Collect all applications from the history
    all_applications = []
    for step in optimizer.search_history:
        for app in step['applications_started']:
            all_applications.append({
                'company': app['company'],
                'time': step['end_time'],
                'type': 'submission',
                'details': f"Est. salary: £{app['estimated_salary']:,.2f}, Career growth: {app['estimated_career_growth']}/5"
            })
        
        for rej in step['applications_rejected']:
            all_applications.append({
                'company': rej['company'],
                'time': step['end_time'],
                'type': 'rejection',
                'details': f"Time invested: {rej['time_invested']:.2f} years, Failed interview process"
            })
        
        for offer in step['offers_received']:
            all_applications.append({
                'company': offer['company'],
                'time': step['end_time'],
                'type': 'offer',
                'details': f"Salary: £{offer['salary']:,.2f}, Career growth: {offer['attributes']['career_growth']}/5"
            })
    
    # Sort by time
    all_applications.sort(key=lambda x: x['time'])
    
    # Print the timeline
    for i, app in enumerate(all_applications):
        time_str = f"{app['time']:.2f} years"
        if app['type'] == 'submission':
            print(f"{i+1}. Time {time_str}: Applied to {app['company']}")
            print(f"   Details: {app['details']}")
        elif app['type'] == 'rejection':
            print(f"{i+1}. Time {time_str}: REJECTED by {app['company']}")
            print(f"   Reason: {app['details']}")
        elif app['type'] == 'offer':
            print(f"{i+1}. Time {time_str}: OFFER received from {app['company']}")
            print(f"   Details: {app['details']}")
    
    # Print detailed offer evaluation history
    print("\n===== OFFER EVALUATION HISTORY =====")
    print("* Decision process for each offer:")
    
    offer_decisions = []
    for step in optimizer.search_history:
        for decision in step.get('decisions', []):
            offer_decisions.append({
                'company': decision['offer'],
                'time': step['end_time'],
                'utility': decision['utility'],
                'threshold': decision['threshold'],
                'decision': decision['decision'],
                'explanation': decision['explanation']
            })
    
    for i, decision in enumerate(offer_decisions):
        time_str = f"{decision['time']:.2f} years"
        decision_str = decision['decision'].upper()
        print(f"{i+1}. Time {time_str}: {decision['company']} - {decision_str}")
        print(f"   Utility: {decision['utility']:.2f} vs Threshold: {decision['threshold']:.2f}")
        print(f"   Reasoning: {decision['explanation']}")
    
    # Print explanation of advanced OST
    print("\n" + explain_advanced_ost_algorithm())
    
    print("\nAdvanced job search simulation visualization saved as 'advanced_job_search_trajectory.png'")
    
    return results

def main():
    # Sample user preferences (same as original)
    data_string = """ "user_data": [
    {
      "question_type": "min_salary",
      "question": "What is the minimum annual salary you would accept for a full-time position in the UK?",
      "response": 21450,
      "data_type": "number"
    },
    {
      "question_type": "compensation_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is compensation to you in your job search?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "career_growth_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is career growth and advancement opportunities to you in your next role?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "location_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is the location of the job (considering your current location in the UK)?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "work_life_balance_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is work-life balance to you?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "company_reputation_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is the reputation and prestige of the company to you?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "role_responsibilities_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is the variety and challenge of the role responsibilities?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "risk_tolerance",
      "question": "On a scale of 1 to 10 (1=low, 10=high), how willing are you to wait for a better job offer if your current search isn't yielding ideal results?",
      "response": 4,
      "data_type": "number"
    },
    {
      "question_type": "job_search_urgency",
      "question": "On a scale of 1 to 10 (1=not urgent, 10=very urgent), how urgent is it for you to secure a new job?",
      "response": 8,
      "data_type": "number"
    },
    {
      "question_type": "financial_runway",
      "question": "Approximately how many months of living expenses do you currently have saved?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "current_salary",
      "question": "What was your last annual salary (or your current salary if employed)?",
      "response": 25000,
      "data_type": "number"
    },
    {
      "question_type": "ai_application_interest",
      "question": "Considering your projects involving AI (ResNet, Generative AI chatbot), on a scale of 1 to 5 (1=unimportant, 5=extremely important), how interested are you in working on projects with a strong AI component?",
      "response": 4,
      "data_type": "number"
    },
    {
      "question_type": "tech_stack_alignment_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is it for you that a role utilizes your preferred technologies (e.g., Python, React, FastAPI)?",
      "response": 3,
      "data_type": "number"
    },
    {
      "question_type": "mentoring_opportunity_weight",
      "question": "On a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is the opportunity for mentorship and professional development within a company?",
      "response": 4,
      "data_type": "number"
    },
    {
      "question_type": "project_variety_weight",
      "question": "Given your diverse experience across web design, software engineering, and system integration, on a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is having a variety of projects and challenges in your role?",
      "response": 4,
      "data_type": "number"
    },
    {
      "question_type": "team_collaboration_weight",
      "question": "Based on your experience at Sanoh UK and Johnson Control, on a scale of 1 to 5 (1=unimportant, 5=extremely important), how important is it for you to work in a collaborative team environment?",
      "response": 4,
      "data_type": "number"
    }
  ],"""
    
    # Parse user preferences
    preferences = parse_user_data(data_string)
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIMAL STOPPING THEORY JOB SEARCH SIMULATION")
    print("=" * 80)
    print("\nThis simulation provides a detailed view of the job search process using advanced")
    print("Optimal Stopping Theory with Bayesian updating and continuous-time stochastic modeling.")
    print("\nBALANCING FACTORS:")
    print(f"- Urgency level: {preferences.get('job_search_urgency', 5)}/10 (higher means less selective)")
    print(f"- Risk tolerance: {preferences.get('risk_tolerance', 5)}/10 (higher means more selective)")
    print(f"- Financial runway: {preferences.get('financial_runway', 3)} months")
    print(f"- Current salary: £{preferences.get('current_salary', 25000):,}")
    print(f"- Minimum acceptable salary: £{preferences.get('min_salary', 20000):,}")
    
    print("\nKEY PRIORITIES (rated 4-5 out of 5):")
    priorities = []
    for key, value in preferences.items():
        if '_weight' in key and value >= 4:
            name = key.replace('_weight', '').replace('_', ' ').title()
            priorities.append(f"{name} ({value}/5)")
    if preferences.get('ai_application_interest', 0) >= 4:
        priorities.append(f"AI Component ({preferences.get('ai_application_interest')}/5)")
    
    if priorities:
        for priority in priorities:
            print(f"- {priority}")
    else:
        print("- No strong priorities identified (all factors rated 3/5 or lower)")
    
    print("\nStarting simulation now. Each step represents approximately 2-3 weeks of job searching...\n")
    print("=" * 80)
    
    # Run the advanced simulation
    run_advanced_simulation(preferences)

if __name__ == "__main__":
    main()