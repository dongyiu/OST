#!/usr/bin/env python
"""
OST Algorithm Visualization Tool

This script provides visualizations of the Optimal Stopping Theory (OST) algorithm
for job search decision making.

Usage:
    python visualize_ost.py [simulation|decision|all]
    
    - simulation: Run a full job search simulation with visualization (1 graph)
    - decision: Show a step-by-step breakdown of the decision process (1 graph)
    - all: Run only the simulation mode (default, 1 graph)
"""

import sys
from ost import run_semantic_ost_simulation, visualize_decision_process
from ost import SemanticOST, create_user_profile

def run_decision_only():
    """Run only the decision process visualization with a sample profile"""
    print("=== DECISION PROCESS DEMONSTRATION ===")
    
    # Create a software engineer profile
    user_profile, user_preferences = create_user_profile("software_engineering")
    
    # Initialize OST
    ost = SemanticOST(user_profile, user_preferences, max_time=1.0)
    
    # Run the decision process visualization
    visualize_decision_process(ost)

def main():
    """Main entry point for the visualization tool"""
    mode = "all"
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "simulation" or mode == "all":
        # Only run simulation when "all" is specified
        run_semantic_ost_simulation()
    
    if mode == "decision" and mode != "all":
        # Only run decision when explicitly requested
        run_decision_only()
    
    if mode not in ["simulation", "decision", "all"]:
        print(__doc__)

if __name__ == "__main__":
    main() 