import pandas as pd
import numpy as np


class DataLoader:
    """
    负责各类数据的加载与初步格式化。
    实际项目中需要对接API或读取具体的CSV/Excel文件。
    """

    def load_historical_election_results(self, filepath="data/history_election.csv"):
        """
        加载过去20年的选区选举结果。
        列需求: [Year, District_ID, Party, Candidate, Votes, Vote_Share, Is_Incumbent]
        """
        print(f"[Info] Loading historical election results from {filepath}...")
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            print(
                f"[Warning] File not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()

    def load_polling_data(self, filepath="data/polls.csv"):
        """
        加载民调数据。
        关键在于区分调查方式以计算所谓的'Bias'。
        列需求: [Date, Pollster, Method(Phone/Net), Sample_Size, Support_LDP, Support_Opposition, ...]
        """
        print(f"[Info] Loading polling data from {filepath}...")
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            print(
                f"[Warning] File not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()

    def load_economic_indicators(self):
        """
        加载宏观经济指标，用于衡量反噬效应。
        包括: CPI, 实质工资指数, 访日外国人数(Proxy for 旅游业景气度), 建筑业人手不足率
        """
        print("[Info] Loading economic indicators (Mocked for now)...")
        # 返回一个简单的mock数据
        return pd.DataFrame({'CPI': [102.5], 'Real_Wage_Index': [98.2]})

    def load_constituency_demographics(self, filepath="data/demographics.csv"):
        """
        加载选区维度的微观数据，用于MrP模型(多层回归)。
        列需求: [District_ID, Avg_Age, Median_Income, Industry_Makeup(Construction/Tourism/Manufacturing), Pct_Foreign_Residents]
        """
        print(f"[Info] Loading constituency demographics from {filepath}...")
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            print(
                f"[Warning] File not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()

    def load_social_sentiment(self, filepath="data/social_sentiment.csv"):
        """
        加载社交媒体情感数据（X/Twitter）。
        列需求: [Date, Topic(Immigration/China/Taxes), Sentiment_Score, Volume, Bot_Ratio]
        """
        print("[Info] Loading social sentiment data...")
        pass
