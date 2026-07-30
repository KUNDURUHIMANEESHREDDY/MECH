from backend.science.benchmarking_orchestrator import BenchmarkingOrchestrator
import os

def main():
    """
    Main entry point to execute the full benchmarking campaign.
    This script would be invoked to run IOI, Induction, SAE, and ACDC 
    across GPT-2, Gemma, Llama, and Qwen.
    """
    print("Initializing MECH Benchmarking Campaign...")
    
    # Ensure portal structure exists
    os.makedirs("backend/science/portal/projects", exist_ok=True)
    os.makedirs("backend/reproductions/projects", exist_ok=True)
    
    orchestrator = BenchmarkingOrchestrator(database_path="backend/science/benchmark_database.json")
    
    # Running the campaign generates real entries in the benchmark database
    # and produces the statistical traces required for papers.
    orchestrator.run_full_campaign()
    
    print("\nCampaign finalized. Results stored in 'backend/science/benchmark_database.json'.")
    print("Next step: Use 'backend/science/publications/paper_generator.py' to draft scientific outputs.")

if __name__ == "__main__":
    main()
