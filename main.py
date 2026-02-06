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

    # 4. Feature Engineering for Simulation (2026 Context)
    print("\n--- Phase 3: District Feature Engineering (2026 Context) ---")

    # Key Adjustment: 2024 was LDP's "scandal low point". By 2026:
    # 关键调整：2024是自民党"丑闻低谷"。到2026年：
    # 1. Scandal fatigue (丑闻疲劳) - voters have short memory
    # 2. Takaichi effect (高市效应) - new PM brings novelty
    # 3. Economic recovery (if any) - 经济复苏（如果有的话）

    # Apply "scandal recovery" factor: +5% baseline boost
    # 应用"丑闻淡化"因子：基础支持率回升5%
    train_df['LDP_Share_2026_adjusted'] = train_df['LDP_Share'] * \
        1.08  # 8% recovery from 2024 low

    # Temporarily rename for preprocessing compatibility
    train_df['LDP_Share_original'] = train_df['LDP_Share']
    train_df['LDP_Share'] = train_df['LDP_Share_2026_adjusted']

    district_features = processor.engineer_district_features(train_df, None)

    # Restore original
    train_df['LDP_Share'] = train_df['LDP_Share_original']

    # Add Komeito dependency proxy if not present
    # Assume Komeito splits 10-15% of the conservative vote block, or specifically has ~20-25k solid votes.
    # Here we model it as a percentage of the total potential conservative coalition.
    if 'komeito_votes_pct' not in district_features.columns:
        # Estimated ~12% reliance
        district_features['komeito_votes_pct'] = 0.12

    # 5. Three-Scenario Simulation (Coalition Retention Analysis)
    # 5. 三场景模拟（新党保留率分析）
    total_districts = len(district_features)
    print(f"\n[Info] Ready to simulate for {total_districts} districts.")

    scenarios = {
        'Optimistic': {'retention': 0.90, 'description': 'Strong unity, voters accept merger'},
        'Baseline': {'retention': 0.70, 'description': 'Moderate leakage, realistic scenario'},
        'Pessimistic': {'retention': 0.50, 'description': 'Failed coalition, 2021 CDP-JCP repeat'}
    }

    results = {}

    for scenario_name, config in scenarios.items():
        print(
            f"\n--- Running Scenario: {scenario_name} (Retention: {config['retention']*100:.0f}%) ---")
        print(f"    Description: {config['description']}")
        sim_results = simulator.run(
            district_features,
            coalition_retention_rate=config['retention'],
            scenario_name=scenario_name
        )
        results[scenario_name] = sim_results

    # 6. Results Compilation
    # 6. 结果汇总
    print("\n" + "="*60)
    print("=== 2026 Election Prediction: Coalition Analysis ===")
    print("=== 2026年选举预测：新党整合分析 ===")
    print("="*60)

    # Calculate PR Seats (Proportional Representation)
    average_smd_support = district_features['ldp_base'].mean()
    estimated_pr_seats = simulator.estimate_pr_seats(average_smd_support)

    # Official House Majority (233 / 465)
    total_house_seats = 465
    official_majority = 233
    coverage_ratio = total_districts / 289

    print(f"\nBase Information:")
    print(f"  Districts Analyzed: {total_districts} / 289")
    print(f"  LDP PR Seats (Estimated): {estimated_pr_seats} / 176")
    print(f"  Majority Line: {official_majority} / {total_house_seats}")
    print("\n" + "-"*60)

    # Display results for each scenario
    print("\nScenario Results:")
    print(f"{'Scenario':<15} {'SMD Seats':<12} {'Total Seats':<12} {'Majority?':<10}")
    print("-"*60)

    for scenario_name in ['Optimistic', 'Baseline', 'Pessimistic']:
        mean_smd = results[scenario_name].mean()
        total_seats = (mean_smd / coverage_ratio) + estimated_pr_seats
        majority_status = "✓ YES" if total_seats >= official_majority else "✗ NO"

        print(f"{scenario_name:<15} {int(mean_smd/coverage_ratio):<12} {int(total_seats):<12} {majority_status:<10}")

    print("-"*60)

    # Key Insight
    baseline_seats = (results['Baseline'].mean() /
                      coverage_ratio) + estimated_pr_seats
    print("\n" + "="*60)
    print("KEY INSIGHT / 关键结论:")
    print("="*60)
    if baseline_seats >= official_majority:
        print(">> LDP likely retains majority even with coalition breakdown.")
        print(">> 即使联盟解体，自民党仍可能保持多数。")
        print(f">> Main reason: Opposition integration failure (野党整合失败).")
    else:
        print(">> Coalition breakdown threatens LDP majority.")
        print(">> 联盟解体威胁自民党多数地位。")
    print("="*60)


if __name__ == "__main__":
    main()
