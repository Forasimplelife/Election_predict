"""
选举结果验证脚本 - 2026年2月8日使用
Election Result Validation Script

使用方法:
1. 选举结果公布后，填入实际数据
2. 运行脚本: python validate_prediction.py
3. 查看预测准确度分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ========================================
# 我的预测（2026年2月7日）
# ========================================
PREDICTION_DATE = "2026-02-07"
PREDICTED_SMD = 254
PREDICTED_PR = 170
PREDICTED_TOTAL = 424

print("="*70)
print("2026年众议院选举预测验证")
print("="*70)
print(f"\n预测日期: {PREDICTION_DATE}")
print(f"预测结果:")
print(f"  小选区: {PREDICTED_SMD} / 289")
print(f"  比例代表: {PREDICTED_PR} / 176")
print(f"  总席次: {PREDICTED_TOTAL} / 465")
print("\n" + "-"*70)

# ========================================
# 实际结果（选举后填入）
# ========================================
print("\n请输入实际选举结果:")
print("（如果还没有结果，按回车跳过）\n")

try:
    actual_smd_input = input(f"自民党小选区席次 [预测:{PREDICTED_SMD}]: ")
    actual_pr_input = input(f"自民党比例代表席次 [预测:{PREDICTED_PR}]: ")

    if actual_smd_input and actual_pr_input:
        actual_smd = int(actual_smd_input)
        actual_pr = int(actual_pr_input)
        actual_total = actual_smd + actual_pr

        print("\n" + "="*70)
        print("验证结果分析")
        print("="*70)

        # 计算误差
        error_smd = actual_smd - PREDICTED_SMD
        error_pr = actual_pr - PREDICTED_PR
        error_total = actual_total - PREDICTED_TOTAL

        error_pct_smd = (abs(error_smd) / PREDICTED_SMD) * 100
        error_pct_pr = (abs(error_pr) / PREDICTED_PR) * 100
        error_pct_total = (abs(error_total) / PREDICTED_TOTAL) * 100

        # 显示对比
        print(f"\n{'项目':<15} {'预测':>10} {'实际':>10} {'误差':>10} {'准确度':>10}")
        print("-"*70)
        print(f"{'小选区':<15} {PREDICTED_SMD:>10} {actual_smd:>10} {error_smd:>+10} {100-error_pct_smd:>9.1f}%")
        print(
            f"{'比例代表':<15} {PREDICTED_PR:>10} {actual_pr:>10} {error_pr:>+10} {100-error_pct_pr:>9.1f}%")
        print(f"{'总席次':<15} {PREDICTED_TOTAL:>10} {actual_total:>10} {error_total:>+10} {100-error_pct_total:>9.1f}%")

        # 评级
        print("\n" + "="*70)
        print("预测评级")
        print("="*70)

        if error_pct_total < 5:
            rating = "⭐⭐⭐⭐⭐ 优秀 (Excellent)"
        elif error_pct_total < 10:
            rating = "⭐⭐⭐⭐ 良好 (Good)"
        elif error_pct_total < 15:
            rating = "⭐⭐⭐ 中等 (Fair)"
        elif error_pct_total < 25:
            rating = "⭐⭐ 需改进 (Needs Improvement)"
        else:
            rating = "⭐ 偏差较大 (Large Deviation)"

        print(f"\n总体准确度: {100-error_pct_total:.1f}%")
        print(f"评级: {rating}")

        # 分析偏差原因
        print("\n" + "="*70)
        print("偏差分析")
        print("="*70)

        if error_total > 50:
            print("\n⚠️ 模型过于保守（低估自民党实力）")
            print("可能原因:")
            print("  - 高市效应比预期更强")
            print("  - 新党整合失败比预期更严重")
            print("  - 在野党分裂程度超出预期")
        elif error_total < -50:
            print("\n⚠️ 模型过于乐观（高估自民党实力）")
            print("可能原因:")
            print("  - 高市政策风险显现")
            print("  - 新党意外成功")
            print("  - 投票率变化不利于自民党")
        else:
            print("\n✅ 预测基本准确，方向和量级都合理")

        # 关键假设验证
        print("\n" + "="*70)
        print("关键假设验证")
        print("="*70)

        # 反推实际支持率
        actual_vote_share_est = (actual_smd / 289) * 0.85  # 简化估算
        predicted_vote_share = 0.552
        baseline_2024 = 0.413

        actual_boost = actual_vote_share_est - baseline_2024
        predicted_boost = predicted_vote_share - baseline_2024

        print(f"\n估算实际支持率: {actual_vote_share_est:.1%}")
        print(f"预测支持率: {predicted_vote_share:.1%}")
        print(f"\n实际较2024提升: +{actual_boost*100:.1f}%")
        print(f"预测较2024提升: +{predicted_boost*100:.1f}%")

        if abs(actual_boost - predicted_boost) < 0.05:
            print("\n✅ 支持率提升幅度预测准确")
            print("   → 高市效应+丑闻淡化+新党失败的综合建模合理")
        elif actual_boost > predicted_boost:
            print("\n⚠️ 实际提升超出预期")
            print("   → 可能低估了高市效应或新党失败程度")
        else:
            print("\n⚠️ 实际提升低于预期")
            print("   → 可能高估了高市效应或低估了政策风险")

        # 可视化对比
        print("\n[生成对比图表中...]")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 图1: 预测vs实际
        categories = ['小选区', '比例代表', '总席次']
        predicted = [PREDICTED_SMD, PREDICTED_PR, PREDICTED_TOTAL]
        actual = [actual_smd, actual_pr, actual_total]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, predicted, width,
                label='预测', color='#6c757d', alpha=0.8)
        ax1.bar(x + width/2, actual, width,
                label='实际', color='#dc3545', alpha=0.8)
        ax1.set_ylabel('席次')
        ax1.set_title('预测 vs 实际对比', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, (p, a) in enumerate(zip(predicted, actual)):
            ax1.text(i - width/2, p + 5, str(p), ha='center', va='bottom')
            ax1.text(i + width/2, a + 5, str(a), ha='center', va='bottom')

        # 图2: 误差分析
        errors = [error_smd, error_pr, error_total]
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.barh(categories, errors, color=colors, alpha=0.7)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('误差 (实际 - 预测)')
        ax2.set_title('预测误差分析', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (cat, err) in enumerate(zip(categories, errors)):
            ax2.text(err + (3 if err > 0 else -3), i, f'{err:+d}',
                     ha='left' if err > 0 else 'right', va='center', fontweight='bold')

        plt.tight_layout()
        output_path = 'results/validation_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_path}")
        plt.close()

        # 保存验证结果
        validation_data = {
            'prediction_date': PREDICTION_DATE,
            'election_date': datetime.now().strftime("%Y-%m-%d"),
            'predicted_smd': PREDICTED_SMD,
            'predicted_pr': PREDICTED_PR,
            'predicted_total': PREDICTED_TOTAL,
            'actual_smd': actual_smd,
            'actual_pr': actual_pr,
            'actual_total': actual_total,
            'error_smd': error_smd,
            'error_pr': error_pr,
            'error_total': error_total,
            'accuracy_pct': 100 - error_pct_total,
            'rating': rating
        }

        pd.DataFrame([validation_data]).to_csv(
            'results/validation_results.csv', index=False)
        print(f"✓ 验证数据已保存: results/validation_results.csv")

        print("\n" + "="*70)
        print("下一步建议")
        print("="*70)
        print("\n1. 下载详细选举数据（各选区得票）")
        print("2. 分析哪些选区预测偏差最大")
        print("3. 验证'高市效应'和'公明党回流'假设")
        print("4. 更新模型，为下次选举做准备")
        print("\n" + "="*70)

    else:
        print("\n暂无实际数据，等选举结果公布后再运行此脚本。")

except ValueError:
    print("\n输入格式错误，请输入数字。")
except KeyboardInterrupt:
    print("\n\n程序已取消。")

print("\n验证完成！\n")
