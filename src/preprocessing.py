import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class DataPreprocessor:
    """
    数据清洗、特征选择与基础模型训练。
    """

    def __init__(self):
        self.model = None
        self.feature_columns = []

    def load_and_clean_training_data(self, filepath="data/training_dataset.csv"):
        """
        加载合并后的训练集，并筛选有效特征。
        """
        if not os.path.exists(filepath):
            print(f"[Error] Training data not found: {filepath}")
            return None

        print(f"[Process] Loading training data from {filepath}...")
        df = pd.read_csv(filepath)

        # 1. 剔除没有选举结果的目标行 (Label missing)
        initial_len = len(df)
        df = df.dropna(subset=['LDP_Share'])
        print(
            f" -> Dropped {initial_len - len(df)} rows with missing election results. Remaining: {len(df)}")

        # 2. 特征工程 (Feature Engineering)
        # 我们需要从那一堆 '就業者数' 和 '人口' 这种绝对数值中，计算出相对比例 (%)

        # 老龄化率 (65岁以上 / 总人口)
        # CSV列名: '人口_（再掲）65歳以上'
        if '人口_（再掲）65歳以上' in df.columns:
            df['pct_elderly'] = df['人口_（再掲）65歳以上'] / df['人口_総数']
        else:
            print("[Warning] Elderly population column not found. Dummying.")
            df['pct_elderly'] = 0.0

        # 农业从业占比 (就業者数_01_うち農業 / 就業者数_0_総数)
        # 第一产业往往是自民党铁票
        if '就業者数_01_うち農業' in df.columns and '就業者数_0_総数' in df.columns:
            df['pct_agriculture'] = df['就業者数_01_うち農業'] / df['就業者数_0_総数']
        else:
            df['pct_agriculture'] = 0.0

        # 建筑业从业占比 (就業者数_D_建設業) - 也是组织票大户
        if '就業者数_D_建設業' in df.columns:
            df['pct_construction'] = df['就業者数_D_建設業'] / df['就業者数_0_総数']
        else:
            df['pct_construction'] = 0.0

        # 3. 定义要在模型中使用的特征列表 (X)
        self.feature_columns = [
            'pct_foreigners', 'pct_elderly', 'pct_agriculture', 'pct_construction']
        df[self.feature_columns] = df[self.feature_columns].fillna(0.0)

        return df

    def train_baseline_model(self, df):
        """
        训练一个简单的随机森林回归模型，用来学习特征和 LDP 得票率的关系。
        """
        print("[Process] Training Baseline Regression Model (RandomForest)...")

        X = df[self.feature_columns]
        y = df['LDP_Share']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f" -> Model Performance: RMSE={rmse:.4f}")

        # Feature Importance
        importances = self.model.feature_importances_
        print(" -> Feature Importances:")
        for name, imp in zip(self.feature_columns, importances):
            print(f"    {name}: {imp:.4f}")

        return self.model

    def engineer_district_features(self, demographics_df, economics):
        """
        [Legacy/Compatibility] 
        应用模型预测基准得票率，并加入政策反噬。
        """
        print("[Process] Predicting LDP baseline with trained model...")

        # 复用特征计算逻辑 (这部分其实应该封装，但为保持简单我们在 load_and_clean 里做了)
        # 假设 demographics_df 已经有了 pct_elderly 等基本特征 (因为它来自于 training_dataset)
        # 如果是全新的 dataframe，需要重新计算一遍 pct_... (TODO)

        if self.model:
            X = demographics_df[self.feature_columns].fillna(0)
            # 使用模型预测此选区的"正常"支持率
            demographics_df['ldp_base'] = self.model.predict(X)
        else:
            print("[Warning] Model not trained. Using 0.45 default.")
            demographics_df['ldp_base'] = 0.45

        # 政策反噬 (Policy Backlash) - 这是模型学不到的未来变量
        if 'pct_foreigners' in demographics_df.columns:
            # 假设排外政策导致额外反弹
            demographics_df['policy_backlash'] = demographics_df['pct_foreigners'] * 0.5
        else:
            demographics_df['policy_backlash'] = 0.0

        return demographics_df

    def clean_polls(self, raw_polls):
        """
        清洗民调数据。
        """
        print("[Process] Cleaning and weighting polling data...")
        return raw_polls

        # 1. 基础票仓 (Based on History/Demographics)
        # 如果 csv 里有 LDP_Stronghold_Score 就用，没有就默认 0.5
        if 'LDP_Stronghold_Score' in df.columns:
            df['ldp_base'] = df['LDP_Stronghold_Score']
        else:
            df['ldp_base'] = 0.5

        # 2. 政策反噬风险 (Policy Backlash)
        # 假设：外国居民比例每增加 1%，自民党支持率下降 1% (系数0.01)
        # 假设：第一产业(农业)对"高市激进政策"其实可能反感(劳动力短缺)，系数设为 0.05
        # 注意：这里仅作演示，实际系数需要回归分析得出
        if 'Foreign_Residents_Pct' in df.columns:
            df['policy_backlash'] = df['Foreign_Residents_Pct'] * 0.01
        else:
            df['policy_backlash'] = 0.0

        return df

    def merge_komeito_impact(self, district_features):
        """
        量化这一特殊事件：公明党-立宪民主党合并。
        估算每个选区原本属于公明党的'组织票'数量，并标记为'Risk_Votes'。
        """
        print("[Process] Calculating Komeito split impact per district...")

        # 假设公明党组织票占全选区的 10% - 15%
        # 在实际中，这个比例因地区而异（城市高，农村低）
        # 这里我们给一个固定均值，加点波动
        district_features['komeito_votes_pct'] = np.random.uniform(
            0.10, 0.15, size=len(district_features))

        return district_features
