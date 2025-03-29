#!/usr/bin/env python3
# Example using a specific user ID with application_process.py

from application_process import process_application

# Your user ID from main.py
USER_ID = "b759618d-1b25-44ff-81ec-6f695dc6d471"  # Replace with your actual user ID

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

# Process the application using ONLY your user ID
# This will fetch your preferences from MongoDB automatically
print(f"Processing application using user ID: {USER_ID}")
assessment_result = process_application(
    application_data=application_data,
    user_id=USER_ID,  # Pass just the user ID
    automated=True    # Avoid interactive prompts
)

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