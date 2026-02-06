# Japan 2026 Election Prediction Project (Takaichi / Komeito Split Scenario)

This project predicts the outcome of the hypothetical 2026 House of Representatives election using Machine Learning (RandomForest) and Monte Carlo Simulation.

## Key Features
- **Real Data**: Uses 2024 Election Results (merged) and 2022 Census Data (MIC).
- **Machine Learning**: Predicts LDP baseline vote share based on district demographics (Age, Industry, Foreign Population).
- **Simulation**: Models the impact of a "Komeito Split" (Coalition breakdown).

## Usage
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Run the simulation:
   ```bash
   python main.py
   ```

## Scenarios
- **Scenario A (Neutral)**: Komeito ends coalition. Komeito voters abstain. Opposition remains fragmented.
- **Scenario B (Hostile)**: Komeito merges with CDP (Opposition). Komeito voters actively vote against LDP.

## Results Analysis
- **Base LDP Strength**: Modeled on 2024 data (Average ~41%).
- **Impact of Split**: Removing Komeito support (~10-12% of vote) drops LDP to ~30%.
- **Outcome**: In both scenarios, the simulation predicts a significant loss of seats for the LDP, suggesting that the LDP-Komeito coalition is mathematically essential for LDP's majority.

## Project Structure
- `data/`: Contains `training_dataset.csv` (Processed real data).
- `src/preprocessing.py`: ML Pipeline and Feature Engineering.
- `src/analysis_models.py`: Bayesian and Simulation models.
- `main.py`: Entry point.
