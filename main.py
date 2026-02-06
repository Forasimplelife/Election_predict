import pandas as pd
import sys
import os
from src.preprocessing import DataPreprocessor
from src.analysis_models import BayesianPollAggregator, ElectionSimulator

# Setup encoding for Windows console output if needed
sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("=== Initialize Japan 2026 Election Prediction Project ===")

    # 1. Initialize Modules
    processor = DataPreprocessor()
    aggregator = BayesianPollAggregator()
    simulator = ElectionSimulator(num_simulations=5000)

    # 2. Data Loading & Cleaning
    # Load real training data (2024 results + 2022 Census)
    print("\n--- Phase 1: Data Preparation ---")
    data_path = os.path.join("data", "training_dataset.csv")
    train_df = processor.load_and_clean_training_data(data_path)

    if train_df is None or train_df.empty:
        print("[Error] Failed to load training data. Aborting.")
        return

    # 3. Model Training
    # Train a RandomForestRegressor on the 2024 data to learn the demographics -> LDP Vote mapping
    print("\n--- Phase 2: Model Training ---")
    trained_model = processor.train_baseline_model(train_df)

    # 4. Feature Engineering for Simulation
    # Calculate 'ldp_base' (predicted support) and 'policy_backlash' for each district
    print("\n--- Phase 3: District Feature Engineering ---")
    # For simulation, we currently act as if we are predicting on the same districts we trained on (Self-prediction for baseline)
    # In a real future scenario, we would load NEW demographics for 2026.
    # Here uses 2022 Census data which is still valid for 2026.
    district_features = processor.engineer_district_features(train_df, None)

    # Add Komeito dependency proxy if not present
    # Assume Komeito splits 10-15% of the conservative vote block, or specifically has ~20-25k solid votes.
    # Here we model it as a percentage of the total potential conservative coalition.
    if 'komeito_votes_pct' not in district_features.columns:
        # Estimated ~12% reliance
        district_features['komeito_votes_pct'] = 0.12

    # 5. Simulation
    total_districts = len(district_features)
    print(f"\n[Info] Ready to simulate for {total_districts} districts.")

    # Scenario A: Komeito Neutral (Voters stay home)
    print("\n--- Running Scenario A: Komeito voters abstain ---")
    sim_results_A = simulator.run(
        district_features, komeito_scenario='neutral')

    # Scenario B: Komeito Hostile (Voters switch to Opposition)
    print("\n--- Running Scenario B: Komeito voters switch to Opposition ---")
    sim_results_B = simulator.run(
        district_features, komeito_scenario='hostile')

    # 6. Results
    print("\n=== Prediction Summary (2026 Hypothetical) ===")

    # Calculate PR Seats (Proportional Representation)
    average_smd_support = district_features['ldp_base'].mean()
    estimated_pr_seats = simulator.estimate_pr_seats(average_smd_support)

    # Official House Majority (233 / 465)
    total_house_seats = 465
    official_majority = 233

    print(f"Total Districts Analyzed (SMD Data): {total_districts} / 289")
    print(
        f"Calculated PR Seats (LDP): {estimated_pr_seats} / 176 (Based on {average_smd_support:.1%} vote share)")
    print(f"Official Majority Line: {official_majority} / {total_house_seats}")
    print("-" * 30)

    mean_smd_A = sim_results_A.mean()
    mean_smd_B = sim_results_B.mean()

    # Extrapolate missing 18 districts (Assuming purely proportional)
    # 我们只分析了 271 个区，还有 18 个区没匹配上，按比例补齐
    coverage_ratio = total_districts / 289

    # Total = (SMD_Wins / 0.94) + PR_Seats
    total_seats_A = (mean_smd_A / coverage_ratio) + estimated_pr_seats
    total_seats_B = (mean_smd_B / coverage_ratio) + estimated_pr_seats

    print(
        f"[Scenario A: Neutral] Total Estimated Seats: {int(total_seats_A)} (SMD: {int(mean_smd_A/coverage_ratio)} + PR: {estimated_pr_seats})")
    print(
        f"[Scenario B: Hostile] Total Estimated Seats: {int(total_seats_B)} (SMD: {int(mean_smd_B/coverage_ratio)} + PR: {estimated_pr_seats})")
    print("-" * 30)

    if total_seats_B < official_majority:
        print(f">> [ALERT] REGIME CHANGE LIKELY (政权更替可能性大).")
        print(
            f"   LDP seats ({int(total_seats_B)}) < Majority ({official_majority}).")
    else:
        print(f">> [RESULT] LDP retains power (自民党保住政权).")


if __name__ == "__main__":
    main()
