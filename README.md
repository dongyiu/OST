# OST Job Decision Support System

A context-aware job application system utilising Optimal Stopping Theory and intelligent agents to provide data-driven decision support for job seekers.

## Overview

This system helps job applicants make optimal decisions about job opportunities by combining preference-based analysis with mathematical optimisation. Rather than relying solely on intuition, the system uses AI agents to gather, enrich, and analyse job application data, then applies Optimal Stopping Theory to determine when an offer should be accepted.

## Key Features

- **Intelligent User Profiling**: Processes user CVs and preferences to build personalised decision models
- **Dynamic Data Enrichment**: Automatically researches and fills in missing information about job opportunities
- **Job Quality Assessment**: Evaluates offers across multiple dimensions based on user preferences
- **Optimal Stopping Algorithm**: Calculates reservation utilities and thresholds to determine when to accept offers
- **Meta-reasoning**: Avoids redundant analysis and optimises API usage during processing
- **Conversational Interface**: Guides users through preference collection with adaptive questioning

## System Components

The system consists of three primary modules:

### User Onboarding (`main.py`)
- Extracts professional field from CV
- Conducts personalised preference interviews
- Creates individual user profiles and preference models

### Application Processing (`application_process.py`) 
- Parses application data from various sources
- Identifies and fills knowledge gaps through research
- Evaluates job quality metrics relative to user preferences
- Produces detailed quality assessments with reasoning traces

### Decision Optimisation (`ost.py`)
- Implements Semantic Optimal Stopping Theory
- Calculates time-dependent decision thresholds
- Provides accept/reject recommendations
- Learns from user feedback to improve future recommendations

## Getting Started

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Run the onboarding process:
```
python main.py
```

4. Process job applications:
```
python application_process.py
```

## How It Works

1. **Onboarding**: The system extracts information from your CV and asks targeted questions to understand your preferences.

2. **Application Processing**: When you receive a job application or offer, the system enriches it with additional research and evaluates it against your preferences.

3. **Decision Support**: The OST algorithm calculates whether to accept the current offer or wait for a potentially better one based on your time constraints and risk tolerance.

## Integrations

The modular design enables integration with job boards, application tracking systems, and other career tools. The system can process applications in various formats and provide decision support throughout your job search journey.

## Licence

MIT
