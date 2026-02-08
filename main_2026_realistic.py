"""
2026年2月选举预测 - 现实版本
Realistic Prediction for Feb 2026 Election

基于最新政治形势：
1. 高市早苗效应 - 媒体预测自民党大胜
2. 新党整合失败 - 公明+立宪合并出现严重问题
3. 2025参议院自民党胜利（假设）- 巩固基本盘
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RealisticElectionPredictor2026:
    """
    基于媒体预测"自民党大胜"的现实模型
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = ['pct_foreigners', 'pct_elderly', 'pct_agriculture', 'pct_construction']
    
    def load_training_data(self):
        """加载训练数据"""
        filepath = "data/training_dataset.csv"
        print(f"[1/5] 加载数据: {filepath}")
        df = pd.read_csv(filepath)
        
        # 基础特征工程
        df['pct_elderly'] = df.get('人口_（再掲）65歳以上', 0) / df['人口_総数'].clip(lower=1)
        df['pct_agriculture'] = df.get('就業者数_01_うち農業', 0) / df['就業者数_0_総数'].clip(lower=1)
        df['pct_construction'] = df.get('就業者数_D_建設業', 0) / df['就業者数_0_総数'].clip(lower=1)
        df[self.feature_columns] = df[self.feature_columns].fillna(0.0)
        
        print(f"   -> 数据行数: {len(df)}")
        return df
    
    def train_baseline_model(self, df):
        """训练基准模型"""
        print("[2/5] 训练RandomForest模型...")
        
        # 只使用有LDP_Share数据的行
        df_clean = df.dropna(subset=['LDP_Share'])
        print(f"   -> 有效数据行数: {len(df_clean)}")
        
        X = df_clean[self.feature_columns]
        y = df_clean['LDP_Share']
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        baseline_support = y.mean()
        print(f"   -> 2024年LDP平均支持率: {baseline_support:.3f} (41%左右，丑闻低谷)")
        
        return self.model
    
    def create_2026_features(self, df):
        """
        创建2026选举特征
        
        【历史参考】：
        - 2024年（丑闻年）：191席，41%得票率
        - 2021年（正常）：261席，约48%得票率
        - 2017年（安倍巅峰）：284席，约49%得票率
        
        【合理目标】：
        - 乐观：270-280席（接近历史高点）
        - 基准：250-260席（恢复正常）
        - 保守：230-240席（适度恢复）
        
        关键调整（保守估计）：
        1. 丑闻淡化：+2-3%（不是+4%）
        2. 高市早苗效应：+1-2%（网络≠实际选票）
        3. 新党整合失败：+1-2%（不是+4%，因为公明票本就不多）
        4. 总提升：+5-7%（从41% → 46-48%）
        """
        print("[3/5] 构建2026选举特征...")
        print("   [历史参考] 2024年: 191席(41%), 2021年: 261席(48%), 2017年: 284席(49%)")
        
        # 预测基准支持率
        X = df[self.feature_columns].fillna(0)
        df['ldp_base_2024'] = self.model.predict(X)
        
        # === 保守校准的参数（基于历史现实） ===
        
        # 1. 丑闻淡化效应（2024→2026）
        # 历史：2021→2024下降7%，2024→2026恢复不会超过下降幅度
        scandal_recovery = 0.025  # +2.5% (保守)
        
        # 2. 高市早苗效应
        # 现实：网络人气≠选票，特别是网络用户投票率低
        df['young_ratio'] = (df['人口_（再掲）15～64歳'] / df['人口_総数']).fillna(0.6)
        df['urban_score'] = (df['人口_総数'] / df['人口_総数'].max()).fillna(0.5)
        # 降低系数：从0.08降到0.03
        df['takaichi_boost'] = df['young_ratio'] * df['urban_score'] * 0.03  # 平均+1%
        
        # 3. 参议院胜利效应
        # 假设2025年自民稳住基本盘（未大败）
        senate_momentum = 0.015  # +1.5% (保守)
        
        # 4. 新党整合失败的影响
        # 现实：公明党在小选区只帮助自民党，但公明票本身只有6-8%
        komeito_organizational_loss = -0.015  # 失去组织动员能力 -1.5%
        komeito_conservative_return = 0.020   # 保守派回流 +2%
        # 净效果：+0.5%
        
        # 综合计算2026支持率（保守估计）
        df['ldp_2026_adjusted'] = (
            df['ldp_base_2024'] 
            + scandal_recovery           # +2.5%
            + df['takaichi_boost']       # +1.0%
            + senate_momentum            # +1.5%
            + komeito_conservative_return # +2.0%
            + komeito_organizational_loss # -1.5%
        )
        # 总提升：约+5.5%
        
        # 农村地区微调（不能太激进）
        df['rural_bonus'] = np.where(
            df['pct_agriculture'] > df['pct_agriculture'].median(),
            0.01,  # 农村+1% (降低)
            0.0
        )
        df['ldp_2026_adjusted'] += df['rural_bonus']
        
        # 限制在合理范围（不能超过50%的选区太多）
        df['ldp_2026_adjusted'] = df['ldp_2026_adjusted'].clip(0.25, 0.58)
        
        print(f"   -> 2026年预测平均支持率: {df['ldp_2026_adjusted'].mean():.3f}")
        print(f"   -> 较2024年平均提升: +{(df['ldp_2026_adjusted'].mean() - df['ldp_base_2024'].mean())*100:.1f}%")
        
        return df
    
    def simulate_election(self, df, n_simulations=5000):
        """
        蒙特卡洛模拟
        """
        print(f"[4/5] 运行{n_simulations}次蒙特卡洛模拟...")
        
        ldp_seats_list = []
        
        ldp_support = df['ldp_2026_adjusted'].values
        
        for i in range(n_simulations):
            # 全国波动
            national_swing = np.random.normal(0, 0.02)
            
            # 地区随机误差
            # 但不能太乐观，历史上48%得票率只赢261席，不是300席
            win_threshold = np.random.uniform(0.40, 0.45, size=len(df))  # 提高阈值
            
            # 计算各选区得票率
            final_support = ldp_support + national_swing + local_noise
            final_support = np.clip(final_support, 0.2, 0.8)
            
            # 胜选阈值（考虑在野党分裂）
            # 关键：新党整合失败 → 共产党继续单独出战 → 反自民票分裂
            win_threshold = np.random.uniform(0.37, 0.42, size=len(df))
            
            # 计算赢得的小选区数
            ldp_wins = (final_support > win_threshold).sum()
            ldp_seats_list.append(ldp_wins)
        
        ldp_seats = np.array(ldp_seats_list)
        
        print(f"   -> 小选区席次: {ldp_seats.mean():.1f} ± {ldp_seats.std():.1f}")
        print(f"   -> 90%置信区间: [{np.percentile(ldp_seats, 5):.0f}, {np.percentile(ldp_seats, 95):.0f}]")
        
        return ldp_seats
    
    def estimate_pr_seats(self, smd_performance):
        """
        根据小选区表现估算比例代表席次
        【历史参考】：
        - 2024年：自民党SMD约140席 → PR约51席（比例0.36）
        - 2021年：自民党SMD约189席 → PR约72席（比例0.38）
        - 2017年：自民党SMD约218席 → PR约66席（比例0.30）
        
        【规律】：比例代表通常是小选区的30-40%
        """
        # 保守估计：用35%的比例
        pr_seats = smd_performance * 0.35
        
        # 但不能超过历史最高（2021年的72席）
        pr_seats = min(pr_seats, 75)
        9
        pr_seats = avg_vote_share * 176 * 1.1
        return int(pr_seats)
    
    def generate_report(self, smd_seats_distribution):
        """生成预测报告"""
        print("\n" + "="*70)
        print("2026年2月众议院选举预测报告")
        print("Prediction Report: February 2026 House Election")
        print("="*70)
        
        smd_mean = smd_seats_distribution.mean()
        smd_lower = np.percentile(smd_seats_distribution, 5)
        smd_upper = np.percentile(smd_seats_distribution, 95)
        
        # 覆盖率调整（我们分析了289选区）
        total_smd = 289
        
        pr_seats = self.estimate_pr_seats(smd_mean)
        total_seats = int(smd_mean) + pr_seats
        
        majority_line = 233
        
        print(f"\n【小选区预测】Single-Member Districts (SMD)")
        print(f"  预测席次: {smd_mean:.1f} / {total_smd}")
        print(f"  90%区间: [{smd_lower:.0f}, {smd_upper:.0f}]")
        
        print(f"\n【比例代表预测】Proportional Representation (PR)")
        print(f"  预测席次: {pr_seats} / 176")
        
        print(f"\n【总席次预测】Total Seats")
        print(f"  预测总席: {total_seats} / 465")
        print(f"  过半线: {majority_line}")
        
        if total_seats >= majority_line:
            print(f"  结论: ✓ 自民党单独过半 (领先 {total_seats - majority_line} 席)")
            verdict = "LDP Majority"
        else:
            print(f"  结论: ✗ 未达过半 (差距 {majority_line - total_seats} 席)")
            verdict = "Minority Government"
        
        print("\n【关键假设】Key Assumptions (保守校准):")
        print("  1. 丑闻淡化: +2.5% (距2024已2年，但记忆犹新)")
        print("  2. 高市效应: +1% (网络≠选票，投票率低)")
        print("  3. 参议院稳定: +1.5% (假设未大败)")
        print("  4. 新党失败: 净+0.5% (公明票本就不多)")
        print("  5. 总提升: +5.5% (41%→46.5%)")
        print("\n【历史对比】Historical Reference:")
        print("  2024年(丑闻): 191席, 41%得票")
        print("  2021年(正常): 261席, 48%得票")
        print("  2017年(巅峰): 284席, 49%得票")
        
        print("\n【关键假设】Key Assumptions:")
        print("  1. 高市早苗效应: +3-5% (网络人气+年轻选民)")
        print("  2. 丑闻淡化: +4% (距2024已2年)")
        print("  3. 参议院胜利: +2.5% (巩固基本盘)")
        print("  4. 新党整合失败: 40%公明票回流自民")
        print("  5. 在野党分裂: 共产党继续单独出战")
        
        print("\n【不确定性】Uncertainties:")
        print("  ⚠️ 缺少2025年参议院实际数据")
        print("  ⚠️ 高市政策风险未完全显现")
        print("  ⚠️ 新党实际整合度未知")
        
        print("="*70 + "\n")
        
        return {
            'smd_seats': int(smd_mean),
            'pr_seats': pr_seats,
            'total_seats': total_seats,
            'verdict': verdict
        }
    
    def visualize_results(self, smd_distribution, summary):
        """可视化结果"""
        print("[5/5] 生成可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 席次分布直方图
        ax1 = axes[0, 0]
        ax1.hist(smd_distribution, bins=30, color='#dc3545', alpha=0.7, edgecolor='black')
        ax1.axvline(smd_distribution.mean(), color='darkred', linestyle='--', linewidth=2, label='平均值')
        ax1.set_xlabel('小选区席次 (SMD Seats)', fontsize=11)
        ax1.set_ylabel('频次 (Frequency)', fontsize=11)
        ax1.set_title('小选区席次分布 (5000次模拟)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. 总席次预测
        ax2 = axes[0, 1]
        categories = ['小选区\nSMD', '比例代表\nPR', '总计\nTotal']
        values = [summary['smd_seats'], summary['pr_seats'], summary['total_seats']]
        colors = ['#dc3545', '#fd7e14', '#6c757d']
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(233, color='black', linestyle='--', linewidth=2, label='过半线 (233)')
        ax2.set_ylabel('席次 (Seats)', fontsize=11)
        ax2.set_title('2026预测：自民党席次构成', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{val}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 3. 情景对比
        ax3 = axes[1, 0]
        scenarios = ['旧模型\n(悲观)', '旧模型\n(基准)', '新模型\n(现实)']
        old_seats = [199, 172, summary['total_seats']]
        colors_scenario = ['#28a745', '#ffc107', '#dc3545']
        bars2 = ax3.bar(scenarios, old_seats, color=colors_scenario, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.axhline(233, color='black', linestyle='--', linewidth=2, label='过半线')
        ax3.set_ylabel('总席次 (Total Seats)', fontsize=11)
        ax3.set_title('新旧模型对比', fontsize=12, fontweight='bold')
        ax3.legend()稳定', '新党\n失败', '2026\n预测']
        cumulative = [41.0, 43.5, 44.5, 46.0, 46.5, 46.5]  # 保守校准后的数据
        ax4.plot(factors, cumulative, marker='o', linewidth=2.5, markersize=8, color='#dc3545')
        ax4.fill_between(range(len(factors)), cumulative, alpha=0.2, color='#dc3545')
        ax4.axhline(y=48, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='2021水平(48%)')
        ax4.set_ylabel('支持率 (%)', fontsize=11)
        ax4.set_title('自民党支持率演变路径（保守估计）', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        ax4.set_ylim(38, 52ter', va='bottom', fontweight='bold', fontsize=11)
        
        # 4. 关键要素分解
        ax4 = axes[1, 1]
        factors = ['2024\n基准', '丑闻\n淡化', '高市\n效应', '参院\n胜利', '公明\n回流', '2026\n预测']
        cumulative = [41.0, 45.0, 48.5, 51.0, 55.0, 54.0]  # 示意数据
        ax4.plot(factors, cumulative, marker='o', linewidth=2.5, markersize=8, color='#dc3545')
        ax4.fill_between(range(len(factors)), cumulative, alpha=0.2, color='#dc3545')
        ax4.set_ylabel('支持率 (%)', fontsize=11)
        ax4.set_title('自民党支持率演变路径', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.set_ylim(35, 60)
        
        plt.tight_layout()
        output_path = 'results/realistic_prediction_2026.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   -> 图表已保存: {output_path}")
        plt.close()


def main():
    print("\n" + "="*70)
    print("2026年2月日本众议院选举 - 现实预测模型")
    print("Japan 2026 Election - Realistic Prediction Model")
    print("="*70 + "\n")
    
    predictor = RealisticElectionPredictor2026()
    
    # 1. 加载数据
    df = predictor.load_training_data()
    
    # 2. 训练基准模型
    predictor.train_baseline_model(df)
    
    # 3. 创建2026特征
    df = predictor.create_2026_features(df)
    
    # 4. 模拟选举
    smd_distribution = predictor.simulate_election(df, n_simulations=5000)
    
    # 5. 生成报告
    summary = predictor.generate_report(smd_distribution)
    
    # 6. 可视化
    predictor.visualize_results(smd_distribution, summary)
    
    print("\n✓ 预测完成！")
    print("📊 查看结果: results/realistic_prediction_2026.png")
    
    return summary


if __name__ == "__main__":
    main()
