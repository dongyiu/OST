import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Parse user data from the provided JSON format
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


class JobMarket:
    """Simulates a job market with various companies and positions."""
    
    def __init__(self, preferences: dict, num_companies: int = 50):
        self.preferences = preferences
        self.companies = [
            "TechInnovate", "DataDynamics", "CodeCraft", "ByteBuilders", "QuantumQueries",
            "CyberSolutions", "CloudCore", "AIVentures", "WebWizards", "DevDreams",
            "SystemSage", "NetNavigators", "InfoInnovators", "SoftwareSynergy", "TechTrend",
            "DigitalDomain", "ByteBrilliance", "CodeConnect", "AppArchitects", "SecuritySystems",
            "MobileMasters", "AnalyticsAdvance", "QuantumQuest", "IntelImpact", "VirtualVanguard",
            "BlockchainBrains", "RoboticRealm", "MachineMind", "CloudConnect", "NetworkNexus",
            "FinTechForward", "HealthTechHub", "RetailRealm", "MediaMatrix", "EduTechExperts",
            "GamingGenius", "SpaceSystems", "AutoTech", "SmartSolutions", "GreenCode",
            "BioTechByte", "NanoNetworks", "EnergyEngineers", "TransportTech", "FutureFusion",
            "MetaMatrix", "LogisticsLoop", "AgriTechAdvance", "UrbanUpgrade", "AeroAnalytics"
        ]
        random.shuffle(self.companies)
        self.num_companies = min(num_companies, len(self.companies))
        
    def generate_job_offer(self) -> JobOffer:
        """Generate a random job offer based on the job market."""
        company = self.companies[random.randint(0, self.num_companies - 1)]
        
        # Generate salary based on user's current and minimum salary
        min_salary = self.preferences.get('min_salary', 20000)
        current_salary = self.preferences.get('current_salary', 25000)
        
        # Most offers will be close to current salary, with a wider distribution
        # - 30% chance of offers below current salary
        # - 40% chance of offers up to 15% higher
        # - 20% chance of offers 15-30% higher
        # - 10% chance of offers 30%+ higher
        salary_bucket = random.choices(
            ["below", "slightly_above", "moderately_above", "significantly_above"],
            weights=[0.3, 0.4, 0.2, 0.1]
        )[0]
        
        if salary_bucket == "below":
            # Below current salary (but still usually above minimum)
            salary_mean = current_salary * 0.9
            salary_std = current_salary * 0.1
        elif salary_bucket == "slightly_above":
            # Slightly above current
            salary_mean = current_salary * 1.08
            salary_std = current_salary * 0.05
        elif salary_bucket == "moderately_above":
            # Moderately above current
            salary_mean = current_salary * 1.2
            salary_std = current_salary * 0.08
        else:
            # Significantly above current
            salary_mean = current_salary * 1.4
            salary_std = current_salary * 0.15
            
        salary = max(min_salary * 0.8, np.random.normal(salary_mean, salary_std))
        
        # Make job attributes more realistic by creating different company profiles
        company_tier = random.choices(
            ["top", "good", "average", "below_average"],
            weights=[0.1, 0.3, 0.4, 0.2]
        )[0]
        
        if company_tier == "top":
            # Top companies tend to have better everything, but may have worse work-life balance
            career_growth = random.choices([3, 4, 5], weights=[0.2, 0.3, 0.5])[0]
            location_score = random.choices([3, 4, 5], weights=[0.2, 0.3, 0.5])[0]
            work_life_balance = random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.1, 0.1])[0]
            company_reputation = random.choices([4, 5], weights=[0.3, 0.7])[0]
            role_responsibilities = random.choices([3, 4, 5], weights=[0.1, 0.3, 0.6])[0]
            ai_component = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.1, 0.2, 0.3, 0.3])[0]
            tech_stack_match = random.choices([2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.4])[0]
            mentoring_opportunity = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
            project_variety = random.choices([2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.4])[0]
            team_collaboration = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
        elif company_tier == "good":
            # Good companies have solid scores across the board
            career_growth = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
            location_score = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
            work_life_balance = random.choices([2, 3, 4, 5], weights=[0.1, 0.2, 0.4, 0.3])[0]
            company_reputation = random.choices([3, 4, 5], weights=[0.3, 0.5, 0.2])[0]
            role_responsibilities = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
            ai_component = random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.2, 0.2, 0.3, 0.1])[0]
            tech_stack_match = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.3, 0.1])[0]
            mentoring_opportunity = random.choices([2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.2])[0]
            project_variety = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
            team_collaboration = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
        elif company_tier == "average":
            # Average companies, average scores
            career_growth = random.choices([1, 2, 3, 4], weights=[0.1, 0.3, 0.4, 0.2])[0]
            location_score = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]
            work_life_balance = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]
            company_reputation = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
            role_responsibilities = random.choices([1, 2, 3, 4], weights=[0.1, 0.3, 0.4, 0.2])[0]
            ai_component = random.choices([1, 2, 3, 4], weights=[0.3, 0.3, 0.3, 0.1])[0]
            tech_stack_match = random.choices([1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.2])[0]
            mentoring_opportunity = random.choices([1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.2])[0]
            project_variety = random.choices([1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.2])[0]
            team_collaboration = random.choices([1, 2, 3, 4], weights=[0.1, 0.3, 0.4, 0.2])[0]
        else:
            # Below average companies
            career_growth = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            location_score = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
            work_life_balance = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
            company_reputation = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            role_responsibilities = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            ai_component = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            tech_stack_match = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            mentoring_opportunity = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            project_variety = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            team_collaboration = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            
        return JobOffer(
            company=company,
            salary=salary,
            career_growth=career_growth,
            location_score=location_score,
            work_life_balance=work_life_balance,
            company_reputation=company_reputation,
            role_responsibilities=role_responsibilities,
            ai_component=ai_component,
            tech_stack_match=tech_stack_match,
            mentoring_opportunity=mentoring_opportunity,
            project_variety=project_variety,
            team_collaboration=team_collaboration
        )


class JobSearchOptimizer:
    """Uses dynamic programming to optimize job search strategy."""
    
    def __init__(self, preferences: dict, max_weeks: int = 24):
        self.preferences = preferences
        self.job_market = JobMarket(preferences)
        self.max_weeks = max_weeks
        
        # Calculate weekly living expenses based on current salary
        current_salary = preferences.get('current_salary', 25000)
        self.weekly_expenses = current_salary / 52 * 0.6  # Assuming 60% of income goes to expenses
        
        # Financial constraints
        self.financial_runway = preferences.get('financial_runway', 3) * 4  # Convert months to weeks
        self.job_search_urgency = preferences.get('job_search_urgency', 5) / 10  # Normalize to 0-1
        self.risk_tolerance = preferences.get('risk_tolerance', 5) / 10  # Normalize to 0-1
        
        # DP table: value[week] = best expected value from week to end
        self.value_table = [0] * (max_weeks + 1)
        # Decision table: decision[week] = min utility to accept at this week
        self.decision_thresholds = [0] * (max_weeks + 1)
        
        # Job offer history for simulation
        self.job_offers_history = []
        self.utilities_history = []
        
    def simulate_offers(self, num_weeks: int) -> List[Tuple[JobOffer, float]]:
        """Simulate job offers over a specific period."""
        offers_with_utilities = []
        
        for _ in range(num_weeks):
            # Each week has a chance to receive 0-3 offers
            num_offers = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
            
            for _ in range(num_offers):
                offer = self.job_market.generate_job_offer()
                utility = offer.calculate_utility(self.preferences)
                offers_with_utilities.append((offer, utility))
                
        return offers_with_utilities
    
    def calculate_optimal_strategy(self) -> None:
        """
        Use backward induction (dynamic programming) to calculate the optimal
        decision threshold for each week of the job search.
        """
        # Value at the last week is 0 (if no job by then)
        self.value_table[self.max_weeks] = 0
        
        # Generate a large sample of job offers to understand the distribution
        sample_size = 500
        sample_offers = [self.job_market.generate_job_offer() for _ in range(sample_size)]
        sample_utilities = [offer.calculate_utility(self.preferences) for offer in sample_offers]
        
        # Sort utilities to understand the distribution better
        sorted_utilities = sorted(sample_utilities)
        median_utility = sorted_utilities[len(sorted_utilities) // 2]
        p75_utility = sorted_utilities[int(len(sorted_utilities) * 0.75)]
        p90_utility = sorted_utilities[int(len(sorted_utilities) * 0.9)]
        
        # Set a base threshold using the distribution - initially aim for top 25% of offers
        base_threshold = p75_utility
        
        # Factor in risk tolerance - higher risk tolerance means higher initial standards
        risk_factor = 0.5 + self.risk_tolerance  # Range from 0.5 to 1.5
        initial_threshold = base_threshold * risk_factor
        
        # Weekly offer probability (realistic job search has many weeks with no offers)
        weekly_offer_prob = 0.7  # 70% chance of getting at least one offer per week
        avg_weekly_offers = 0.8  # Less than 1 offer per week on average
        
        # Backward induction from the second-last week
        for week in range(self.max_weeks - 1, -1, -1):
            remaining_runway = self.financial_runway - week
            
            # Early weeks should be more selective
            time_factor = 1.0
            if week < self.max_weeks * 0.2:  # First 20% of search period
                time_factor = 1.2  # Be more selective initially
            
            # Increasing urgency as weeks go by and runway shrinks
            urgency_factor = 1.0
            if remaining_runway <= 0:
                # High urgency once runway is depleted
                urgency_factor = 2.0 + (abs(remaining_runway) * 0.1)  # Increases with time beyond runway
            elif remaining_runway < 4:  # Less than a month left
                urgency_factor = 1.5  # Moderate urgency
            
            # Expected utility of waiting one more week
            continuation_value = self.value_table[week + 1]
            
            # Cost of continuing search beyond financial runway
            if remaining_runway <= 0:
                # Increasing penalty for each week beyond runway
                penalty = self.weekly_expenses * (1 + (abs(remaining_runway) * 0.2))
                continuation_value -= penalty * (1 - self.risk_tolerance)
            
            # Adjust for job search urgency
            urgency_adjustment = (self.job_search_urgency / 10) * urgency_factor
            
            # Calculate the threshold for this week
            if week == 0:  # First week should be very selective
                threshold = initial_threshold
            else:
                # Gradual decrease in threshold as weeks progress
                threshold = max(
                    continuation_value - urgency_adjustment,
                    initial_threshold * max(0.1, 1 - (week / self.max_weeks) * (1 + urgency_adjustment))
                )
            
            # Add realistic constraints:
            # 1. Threshold shouldn't be too negative (desperate) until very late
            if week < self.max_weeks - 3 and threshold < 0:
                threshold = 0
                
            # 2. After runway, threshold should drop significantly but not absurdly
            if remaining_runway <= 0 and threshold < -20:
                threshold = -20
            
            # Save the threshold
            self.decision_thresholds[week] = threshold
            
            # Calculate probability of getting an acceptable offer this week
            prob_getting_offer = weekly_offer_prob * avg_weekly_offers
            prob_acceptable = sum(1 for u in sample_utilities if u > threshold) / sample_size
            combined_prob = prob_getting_offer * prob_acceptable
            
            # Calculate expected value if we accept (weighted average of utilities above threshold)
            acceptable_utilities = [u for u in sample_utilities if u > threshold]
            if acceptable_utilities:
                expected_value_if_accepted = sum(acceptable_utilities) / len(acceptable_utilities)
            else:
                expected_value_if_accepted = 0
            
            # Calculate the expected value of being at this week
            self.value_table[week] = (
                combined_prob * expected_value_if_accepted + 
                (1 - combined_prob) * continuation_value
            )
    
    def run_simulation(self) -> Tuple[Optional[JobOffer], int, List[Dict]]:
        """
        Run a simulation of the job search process using the calculated optimal strategy.
        
        Returns:
            Tuple containing the accepted job offer (if any), the week it was accepted,
            and history of all offers with decisions.
        """
        self.calculate_optimal_strategy()
        
        # Track both actual accepted offer and potential best offer
        accepted_offer = None
        accepted_week = -1
        best_possible_offer = None
        best_possible_utility = -float('inf')
        history = []
        
        # Track active applications/interviews
        active_applications = []  # List of applications in progress
        
        for week in range(self.max_weeks):
            # Process active applications - some may become offers
            new_offers_from_apps = []
            remaining_applications = []
            
            for app in active_applications:
                app["stages_remaining"] -= 1
                
                # Random chance of application failing at this stage
                if random.random() < app["fail_probability"]:
                    # Application failed/rejected
                    continue
                    
                if app["stages_remaining"] <= 0:
                    # Application turned into a job offer
                    offer = self.job_market.generate_job_offer()
                    utility = offer.calculate_utility(self.preferences)
                    new_offers_from_apps.append({"offer": offer, "utility": utility, "accepted": False})
                    self.job_offers_history.append(offer)
                    self.utilities_history.append(utility)
                else:
                    # Application still in progress
                    remaining_applications.append(app)
            
            active_applications = remaining_applications
            
            # Generate new applications for this week
            # Weeks without new application opportunities simulate "ghosting" or dry spells
            has_new_applications = random.random() < 0.8  # 80% chance of finding new opportunities
            
            if has_new_applications:
                num_new_applications = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                
                for _ in range(num_new_applications):
                    # Create a new application with random number of stages (interviews, etc)
                    stages = random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
                    fail_prob = random.uniform(0.3, 0.6)  # 30-60% chance of failing at each stage
                    
                    active_applications.append({
                        "stages_remaining": stages,
                        "fail_probability": fail_prob
                    })
            
            # Check if we've run out of financial runway
            runway_status = ""
            if week > self.financial_runway and accepted_offer is None:
                runway_status = "FINANCIAL RUNWAY DEPLETED"
                
            # Sometimes, get direct offers (recruiters reaching out, etc.)
            direct_offers = []
            has_direct_offers = random.random() < 0.3  # 30% chance of direct offers
            
            if has_direct_offers:
                num_direct_offers = random.choices([1, 2], weights=[0.8, 0.2])[0]
                
                for _ in range(num_direct_offers):
                    offer = self.job_market.generate_job_offer()
                    utility = offer.calculate_utility(self.preferences)
                    direct_offers.append({"offer": offer, "utility": utility, "accepted": False})
                    self.job_offers_history.append(offer)
                    self.utilities_history.append(utility)
            
            # Combine all offers for this week
            offers_this_week = new_offers_from_apps + direct_offers
            
            # If we have offers, check against our threshold
            threshold = self.decision_thresholds[week]
            
            best_offer = None
            best_utility = -float('inf')
            offer_explanations = []
            
            for offer_data in offers_this_week:
                offer = offer_data["offer"]
                utility = offer_data["utility"]
                
                # Create explanation for this offer
                explanation = {
                    "company": offer.company,
                    "salary": offer.salary,
                    "utility": utility,
                    "threshold": threshold,
                    "verdict": "Below threshold" if utility <= threshold else "Above threshold"
                }
                
                # Add more explanation details
                if utility <= threshold:
                    gap = threshold - utility
                    explanation["reason"] = f"Rejected: Utility score {utility:.2f} is {gap:.2f} points below current threshold of {threshold:.2f}"
                    explanation["details"] = []
                    
                    # Identify weaknesses in the offer
                    if offer.salary < self.preferences.get('min_salary', 20000):
                        explanation["details"].append(f"Salary (£{offer.salary:,.2f}) is below minimum acceptable (£{self.preferences.get('min_salary', 20000):,.2f})")
                    
                    # Check low scores in high-priority areas
                    for attr, weight_key in [
                        ("career_growth", "career_growth_weight"),
                        ("work_life_balance", "work_life_balance_weight"),
                        ("mentoring_opportunity", "mentoring_opportunity_weight"),
                        ("project_variety", "project_variety_weight"),
                        ("team_collaboration", "team_collaboration_weight")
                    ]:
                        weight = self.preferences.get(weight_key, 3)
                        if weight >= 4 and getattr(offer, attr) <= 2:
                            explanation["details"].append(f"Low {attr.replace('_', ' ')} score ({getattr(offer, attr)}/5) in a high-priority area")
                else:
                    explanation["reason"] = f"Acceptable: Utility score {utility:.2f} exceeds threshold of {threshold:.2f}"
                    
                    # Compare to accepted offer if we have one
                    if accepted_offer:
                        accepted_utility = accepted_offer.calculate_utility(self.preferences)
                        if utility > accepted_utility:
                            explanation["comparison"] = f"Better than current accepted offer ({utility:.2f} vs {accepted_utility:.2f})"
                        else:
                            explanation["comparison"] = f"Not better than current accepted offer ({utility:.2f} vs {accepted_utility:.2f})"
                            explanation["verdict"] = "Below current best"
                
                offer_explanations.append(explanation)
                
                # Track the best offer we've ever seen (for post-analysis)
                if utility > best_possible_utility:
                    best_possible_offer = offer
                    best_possible_utility = utility
                
                if utility > best_utility:
                    best_offer = offer
                    best_utility = utility
            
            # Determine decision and explanation
            decision = "No offers"
            decision_explanation = "No job offers received this week"
            would_accept = False
            
            if offers_this_week:
                if best_offer and best_utility > threshold:
                    # Check if it's better than any offer we've already accepted
                    if accepted_offer is None:
                        decision = "Would accept best offer"
                        decision_explanation = f"This offer from {best_offer.company} exceeds our current threshold ({best_utility:.2f} > {threshold:.2f})"
                        would_accept = True
                    else:
                        accepted_utility = accepted_offer.calculate_utility(self.preferences)
                        if best_utility > accepted_utility:
                            decision = "Would accept best offer (better than previous)"
                            decision_explanation = f"This offer is better than our previously accepted offer ({best_utility:.2f} > {accepted_utility:.2f})"
                            would_accept = True
                        else:
                            decision = "Rejected all offers"
                            decision_explanation = f"Best offer doesn't improve on our previously accepted offer ({best_utility:.2f} vs {accepted_utility:.2f})"
                else:
                    decision = "Rejected all offers"
                    if best_offer:
                        decision_explanation = f"Best offer utility ({best_utility:.2f}) is below our current threshold ({threshold:.2f})"
                    else:
                        decision_explanation = "No valid offers to consider"
            
            # Special case for financial pressure
            financial_pressure = False
            if week >= self.financial_runway - 1 and not accepted_offer and best_offer:
                desperation_factor = 0.7 - min(0.4, max(0, week - self.financial_runway) * 0.05)
                adjusted_threshold = threshold * desperation_factor
                
                if best_utility > adjusted_threshold:
                    decision = "Would accept under financial pressure"
                    decision_explanation = f"Financial runway ending: accepting with reduced threshold ({adjusted_threshold:.2f} vs normal {threshold:.2f})"
                    would_accept = True
                    financial_pressure = True
            
            # Create a detailed week record
            week_record = {
                "week": week + 1,
                "threshold": threshold,
                "offers": offers_this_week,
                "active_applications": len(active_applications),
                "best_utility": best_utility if best_offer else None,
                "runway_status": runway_status,
                "decision": decision,
                "decision_explanation": decision_explanation,
                "offer_explanations": offer_explanations,
                "is_accepted_week": False  # Will be updated later for the final accepted week
            }
            
            # Actually accept the offer if appropriate
            if would_accept:
                # Only one offer gets truly accepted in the entire simulation
                if accepted_offer is None or best_utility > accepted_offer.calculate_utility(self.preferences):
                    accepted_offer = best_offer
                    accepted_week = week
                    for offer_data in offers_this_week:
                        if offer_data["offer"] == best_offer:
                            offer_data["accepted"] = True
                    
                    if financial_pressure:
                        week_record["decision"] = "Accepted under financial pressure" 
                    else:
                        week_record["decision"] = "Accepted offer"
            
            # Add this week to history
            history.append(week_record)
        
        # Mark the actual accepted week in history
        if accepted_week >= 0 and accepted_week < len(history):
            history[accepted_week]["is_accepted_week"] = True
        
        # Final analysis
        if accepted_offer:
            accept_utility = accepted_offer.calculate_utility(self.preferences)
            opportunity_cost = max(0, best_possible_utility - accept_utility)
            
            # Add opportunity cost analysis to last week record
            if history:
                history[-1]["opportunity_cost"] = opportunity_cost
                history[-1]["best_possible_utility"] = best_possible_utility
                
                if best_possible_offer and best_possible_offer != accepted_offer:
                    history[-1]["best_possible_offer"] = {
                        "company": best_possible_offer.company,
                        "salary": best_possible_offer.salary,
                        "utility": best_possible_utility
                    }
        
        return accepted_offer, accepted_week, history
    
    def plot_decision_thresholds(self):
        """Plot the decision thresholds over time."""
        # Fix: Make sure weeks and decision_thresholds have the same length
        weeks = list(range(1, len(self.decision_thresholds) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(weeks, self.decision_thresholds, marker='o', linestyle='-', color='blue')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=self.financial_runway, color='r', linestyle='--', alpha=0.3, 
                label=f'Financial Runway ({self.financial_runway} weeks)')
        
        # Plot the utilities of offers if we have them
        if self.utilities_history:
            offer_weeks = []
            for i in range(len(self.utilities_history)):
                # Assuming equally distributed offers over time
                week = 1 + int(i * len(self.decision_thresholds) / len(self.utilities_history))
                offer_weeks.append(week)
            
            plt.scatter(offer_weeks, self.utilities_history, color='green', alpha=0.6, label='Job Offers')
        
        plt.title('Job Search Strategy: Utility Threshold Over Time')
        plt.xlabel('Weeks')
        plt.ylabel('Utility Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt

# Function to print a simple explanation of the OST algorithm
def explain_ost_algorithm():
    explanation = """
===== OPTIMAL STOPPING THEORY EXPLANATION =====

The job search optimizer uses Optimal Stopping Theory (OST) to determine when to accept a job offer:

1. DYNAMIC PROGRAMMING APPROACH:
   - The algorithm works backwards from the end of the search period
   - For each week, it calculates a "threshold utility" - the minimum value an offer needs to be accepted
   - These thresholds decrease over time as your options narrow and financial pressure increases

2. UTILITY CALCULATION:
   - Each job offer gets a utility score based on your preferences
   - Salary, career growth, work-life balance, etc. are weighted according to your priorities
   - The higher the utility score, the better the overall job offer

3. DECISION RULE:
   - Accept the first offer that exceeds your current threshold
   - Early in the search: Be selective, only accept exceptional offers
   - Middle of search: Become gradually less selective
   - Near end of financial runway: Accept more marginal offers to avoid financial distress

4. KEY FACTORS AFFECTING DECISIONS:
   - Financial runway: How long can you afford to search?
   - Job search urgency: How quickly do you need a new job?
   - Risk tolerance: How willing are you to wait for better offers?
   - Preference weights: What job factors matter most to you?

The simulation runs through the entire search period, making realistic decisions at each point based on
the available information. This mirrors the real-world challenge of job searching where you must decide
whether to accept an offer or keep looking, without knowing what future offers might come.
"""
    print(explanation)

def main():
    # Sample data string (you would replace this with actual data)
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
    
    # Parse the user data
    preferences = parse_user_data(data_string)
    
    # Create optimizer and run simulation
    optimizer = JobSearchOptimizer(preferences, max_weeks=24)
    accepted_offer, accepted_week, history = optimizer.run_simulation()
    
    # Print results
    print("\n===== JOB SEARCH OPTIMIZATION RESULTS =====")
    if accepted_offer:
        print(f"Found acceptable job offer in week {accepted_week + 1}:")
        print(f"Company: {accepted_offer.company}")
        print(f"Salary: £{accepted_offer.salary:,.2f}")
        print(f"Career Growth Score: {accepted_offer.career_growth}/5")
        print(f"Work-Life Balance: {accepted_offer.work_life_balance}/5")
        
        # Show more attributes that were important
        top_prefs = []
        for pref, value in preferences.items():
            if '_weight' in pref and value >= 4:
                top_prefs.append(pref.replace('_weight', ''))
        
        for pref in top_prefs:
            attr_name = pref
            if hasattr(accepted_offer, attr_name):
                attr_value = getattr(accepted_offer, attr_name)
                print(f"{attr_name.replace('_', ' ').title()}: {attr_value}/5")
        
        print(f"Overall Utility: {accepted_offer.calculate_utility(preferences):.2f}")
        
        # Show when the offer was accepted
        if accepted_week + 1 <= optimizer.financial_runway:
            print(f"Accepted within financial runway (Week {accepted_week + 1} of {optimizer.financial_runway} weeks)")
        else:
            print(f"Accepted after financial runway depleted (Week {accepted_week + 1}, {accepted_week + 1 - optimizer.financial_runway} weeks past runway)")
    else:
        print("No acceptable job offer found within the simulation period.")
    
    # Check if there was a better offer that wasn't taken
    if history and "best_possible_offer" in history[-1]:
        best_offer = history[-1]["best_possible_offer"]
        print("\n===== MISSED OPPORTUNITY ANALYSIS =====")
        print(f"Best possible offer that appeared during search:")
        print(f"Company: {best_offer['company']}")
        print(f"Salary: £{best_offer['salary']:,.2f}")
        print(f"Utility: {best_offer['utility']:.2f}")
        
        if accepted_offer:
            improvement = ((best_offer['utility'] / accepted_offer.calculate_utility(preferences)) - 1) * 100
            print(f"This would have been {improvement:.1f}% better than the accepted offer")
    
    print("\n===== SEARCH PARAMETERS & STATISTICS =====")
    print(f"Financial Runway: {preferences.get('financial_runway', 3)} months ({optimizer.financial_runway} weeks)")
    print(f"Job Search Urgency: {preferences.get('job_search_urgency', 5)}/10")
    print(f"Risk Tolerance: {preferences.get('risk_tolerance', 5)}/10")
    
    # Print historical summary
    offers_received = sum(1 for week in history if 'offers' in week and len(week['offers']) > 0)
    total_offers = sum(len(week.get('offers', [])) for week in history)
    weeks_with_applications = sum(1 for week in history if 'active_applications' in week and week['active_applications'] > 0)
    
    print(f"\nWeeks simulated: {len(history)}")
    print(f"Weeks with active applications: {weeks_with_applications}")
    print(f"Weeks with offers: {offers_received}")
    print(f"Total offers received: {total_offers}")
    
    if accepted_offer:
        print(f"Offer acceptance rate: {1/total_offers:.1%} (1 out of {total_offers})")
    else:
        print(f"Offer acceptance rate: 0% (0 out of {total_offers})")
    
    # Generate detailed weekly breakdown
    print("\n===== WEEKLY SEARCH TIMELINE =====")
    for week_data in history:
        week_num = week_data["week"]
        apps = week_data.get("active_applications", 0)
        offers = len(week_data.get("offers", []))
        decision = week_data.get("decision", "")
        is_accepted_week = week_data.get("is_accepted_week", False)
        
        status = f"Week {week_num}: "
        if apps > 0:
            status += f"{apps} active applications, "
        status += f"{offers} offers received"
        
        # Only show ACCEPTED OFFER for the actual accepted week
        if is_accepted_week:
            status += f" - ACCEPTED OFFER"
        elif week_data.get("runway_status", ""):
            status += f" - {week_data['runway_status']}"
            
        print(status)
        
        # Print explanations when we have offers
        if offers > 0:
            explanation = week_data.get("decision_explanation", "")
            if explanation:
                print(f"   Decision: {explanation}")
            
            # Only print detailed offer explanations for the accepted week or weeks with interesting decisions
            if is_accepted_week or "Would accept" in decision:
                offer_explanations = week_data.get("offer_explanations", [])
                for i, exp in enumerate(offer_explanations):
                    if i > 0:  # Only show details for the first offer to avoid clutter
                        print(f"   Offer {i+1}: {exp['company']} - £{exp['salary']:,.2f} - {exp['verdict']}")
                    else:
                        print(f"   Offer {i+1}: {exp['company']} - £{exp['salary']:,.2f} - {exp['verdict']}")
                        print(f"      {exp['reason']}")
                        if "details" in exp and exp["details"]:
                            for detail in exp["details"][:2]:  # Limit to 2 details
                                print(f"      - {detail}")
                print()
    
    # Display the strategy plot
    plt = optimizer.plot_decision_thresholds()
    plt.savefig("job_search_strategy.png")
    print("\nStrategy visualization saved as 'job_search_strategy.png'")
    
    # Additional tips
    print("\n===== ADDITIONAL RECOMMENDATIONS =====")
    if preferences.get('financial_runway', 3) < 6:
        print("- Consider building a larger financial runway to give yourself more flexibility")
        print("  Having only 3 months of savings significantly increases pressure to accept early offers")
    
    if preferences.get('job_search_urgency', 5) > 7:
        print("- Your high job search urgency (8/10) makes you more likely to accept suboptimal offers")
        print("  If possible, reducing this urgency would allow for a more selective search")
    
    print("- Focus on opportunities that align with your highest weighted preferences:")
    top_prefs = []
    for pref, value in preferences.items():
        if '_weight' in pref and value >= 4:
            top_prefs.append((pref.replace('_weight', '').replace('_', ' '), value))
    
    for pref, value in sorted(top_prefs, key=lambda x: x[1], reverse=True):
        print(f"  * {pref.title()} (importance: {value}/5)")
    
    # Targeted strategy recommendations
    if accepted_week and accepted_week < 4:
        print("\n- Your simulation accepted an offer very early. Consider:")
        print("  * Being more selective in the first month (higher initial threshold)")
        print("  * Conducting more research on market rates for your skills")
        print("  * Improving application materials to generate more/better offers")
    elif accepted_week and accepted_week > optimizer.financial_runway:
        print("\n- Your simulation accepted an offer after financial runway was depleted. Consider:")
        print("  * Building a larger emergency fund")
        print("  * Being less selective as runway approaches its end")
        print("  * Exploring part-time or contract work during the search")
    
    print("\nJob Search Strategy Complete!")

if __name__ == "__main__":
    main()