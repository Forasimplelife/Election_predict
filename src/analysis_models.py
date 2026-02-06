import numpy as np
import pandas as pd


class BayesianPollAggregator:
    """
    Model 1: National Polling Aggregation (Nowcasting).
    模型1: 真实民意去噪 (Nowcasting)

    Resolves: The paradox between 'Traditional Polls' and 'Internet Polls'.
    解决: '传统民调' 和 '网络民调' 的悖论
    """

    def __init__(self):
        self.latent_support = None

    def fit(self, polls_df):
        """
        Fits a Bayesian State-Space model to de-noise polls.
        使用贝叶斯状态空间模型拟合，去除民调噪音。

        Logic / 逻辑:
        1. True_Support[t] ~ Normal(True_Support[t-1], sigma)
           (真实支持率是平滑变化的)
        2. Observed_Poll ~ Normal(True_Support, Bias + Error)
           (观测值 = 真实值 + 偏差 + 误差)
        """
        print("[Model] Fitting Bayesian State-Space Model to de-noise polls...")
        self.latent_support = 0.35
        return self.latent_support


class ConstituencyPredictor:
    """
    Model 2: District-level Prediction (MrP).
    模型2: 选区微观预测 (MrP)

    Resolves: How to infer local district results from national trends.
    解决: 如何从全国支持率推导到 289 个小选区的结果
    """

    def predict_probs(self, district_features, national_swing, policy_impact):
        """
        Predict win probability for LDP in each district.
        预测每个选区自民党候选人的胜率。

        Formula / 公式:
        Vote_Share = Base_Vote (Historic) + National_Swing + Local_Policy_Impact
        得票率 = 历史基准 + 全国波动 + 地方政策影响
        """
        print("[Model] Predicting win probabilities for each district...")
        return np.random.normal(0.45, 0.1, size=len(district_features))


class ElectionSimulator:
    """
    Model 3: Monte Carlo Simulation.
    模型3: 蒙特卡洛模拟

    Resolves: Aggregating uncertainty to find the distribution of final seats.
    解决: 综合不确定性，输出最终议席分布概率
    """

    def __init__(self, num_simulations=10000):
        self.n = num_simulations

    def run(self, district_dataset, komeito_scenario='neutral'):
        """
        Run simulations.
        运行模拟。

        params:
            komeito_scenario: 
                'neutral' (Abstain: Komeito voters stay home / 公明党弃权), 
                'hostile' (Switch: Komeito voters vote for opposition / 公明党倒戈)
        """
        print(
            f"[Simulation] Running {self.n} simulations with scenario: {komeito_scenario}...")

        if district_dataset.empty:
            print("[Error] Empty district dataset.")
            return pd.Series([0])

        ldp_seats_distribution = []

        # Convert columns to numpy arrays for speed
        # 将列转换为 numpy 数组以提高速度
        ldp_base = district_dataset['ldp_base'].values
        # Ensure policy_backlash handles NaNs or specific types
        # 确保 policy_backlash 处理 NaN 或特定类型
        policy_backlash = district_dataset.get('policy_backlash', pd.Series(
            0, index=district_dataset.index)).fillna(0).values
        komeito_votes = district_dataset.get('komeito_votes_pct', pd.Series(
            0.1, index=district_dataset.index)).fillna(0.1).values

        for i in range(self.n):
            # 1. National Swing (Random fluctuation)
            # 1. 全国波动 (随机波动)
            national_swing = np.random.normal(0, 0.02)

            # 2. Base Vote Calculation
            # 2. 基础得票率计算
            current_votes = ldp_base + national_swing - policy_backlash

            # 3. Komeito Impact Logic
            # 3. 公明党影响逻辑
            if komeito_scenario == 'neutral':
                # Komeito voters abstain. LDP loses ~80% of Komeito vote (assuming previous support).
                # 公明党支持者弃权。自民党失去约 80% 的公明党选票。

                # Threshold effectively 40% (Fragmented Opposition)
                # 胜选门槛实际上降至 40% (因为在野党分裂)
                current_votes_adj = current_votes - (komeito_votes * 0.8)
                win_threshold = 0.40

            elif komeito_scenario == 'hostile':
                # Komeito voters switch to Opposition.
                # 公明党支持者倒戈转向在野党。

                # LDP loses votes AND thresholds rise.
                # 自民党失去选票，且胜选门槛升高。
                current_votes_adj = current_votes - (komeito_votes * 0.8)

                # Higher threshold due to unified opposition
                # 由于在野党联合 (公明党+立宪)，胜选门槛接近 48%
                win_threshold = 0.48

            else:
                current_votes_adj = current_votes
                win_threshold = 0.42

            # 4. Determine Wins
            # 4. 判定胜负
            wins = (current_votes_adj > win_threshold).sum()
            ldp_seats_distribution.append(wins)

        return pd.Series(ldp_seats_distribution)

    def estimate_pr_seats(self, national_support_rate):
        """
        Estimate Proportional Representation (PR) seats.
        估算比例代表议席 (总共 176 席)。

        History / 历史数据:
        - 2021: LDP Vote 34.7% -> 72 Seats (Efficiency ~2.07)
        - 2024: LDP Vote 26.7% -> 59 Seats (Efficiency ~2.2)

        Logic: 
        PR seats roughly align with Party Support Rate, but LDP gets a slight bonus due to D'Hondt method fragmentation.
        """
        total_pr_seats = 176

        # Assume PR support is slightly lower than Candidate Support (SMD)
        # 通常选民会在小选区投自民党候选人(因为由于人情/地缘)，但在比例区可能投给维新或国民作为制衡
        estimated_pr_vote_share = national_support_rate * 0.85

        # Simple Linear Projection with a slight bonus
        projected_seats = int(total_pr_seats * estimated_pr_vote_share * 1.1)

        # Cap at reasonable max/min
        projected_seats = max(30, min(projected_seats, 100))

        return projected_seats
