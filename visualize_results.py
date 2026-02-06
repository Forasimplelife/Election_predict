import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# Set font to support Japanese
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# Scenario data (manually input from simulation results)
# 场景数据（从模拟结果手动输入）
scenarios = {
    'Optimistic\n乐观': {'smd': 74, 'pr': 78, 'total': 152, 'retention': 90},
    'Baseline\n基准': {'smd': 94, 'pr': 78, 'total': 172, 'retention': 70},
    'Pessimistic\n悲观': {'smd': 121, 'pr': 78, 'total': 199, 'retention': 50},
}

# Constants
MAJORITY_LINE = 233
TOTAL_SEATS = 465


def plot_scenario_comparison():
    """
    Plot 1: Stacked bar chart comparing three scenarios
    图表1：堆叠柱状图对比三个场景
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios_list = list(scenarios.keys())
    smd_seats = [scenarios[s]['smd'] for s in scenarios_list]
    pr_seats = [scenarios[s]['pr'] for s in scenarios_list]

    x = np.arange(len(scenarios_list))
    width = 0.6

    # Stacked bars
    p1 = ax.bar(x, smd_seats, width, label='SMD Seats (小选区)', color='#d62728')
    p2 = ax.bar(x, pr_seats, width, bottom=smd_seats,
                label='PR Seats (比例代表)', color='#ff7f0e')

    # Majority line
    ax.axhline(y=MAJORITY_LINE, color='black', linestyle='--',
               linewidth=2, label=f'Majority Line (过半线): {MAJORITY_LINE}')

    # Labels
    ax.set_ylabel('Seats (议席数)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Scenario (场景)', fontsize=12, fontweight='bold')
    ax.set_title('2026 Election Prediction: LDP Seats by Scenario\n2026选举预测：自民党各场景议席数',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_list, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)

    # Add value labels on bars
    for i, (scenario, data) in enumerate(scenarios.items()):
        total = data['total']
        ax.text(i, total + 5, f"{total}", ha='center',
                va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylim(0, 280)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/scenario_comparison.png',
                dpi=300, bbox_inches='tight')
    print("[Saved] results/scenario_comparison.png")
    plt.close()


def plot_sensitivity_curve():
    """
    Plot 2: Sensitivity analysis - Retention Rate vs Total Seats
    图表2：敏感性分析 - 新党保留率 vs 总议席数
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate curve data (interpolated)
    # 生成曲线数据（插值）
    retention_rates = np.linspace(40, 100, 50)

    # Simple interpolation based on our three data points
    # 基于我们三个数据点的简单插值
    def estimate_seats(retention):
        if retention >= 70:
            # Between Baseline and Optimistic
            slope = (152 - 172) / (90 - 70)
            seats = 172 + slope * (retention - 70)
        else:
            # Between Pessimistic and Baseline
            slope = (172 - 199) / (70 - 50)
            seats = 199 + slope * (retention - 50)
        return seats

    total_seats_curve = [estimate_seats(r) for r in retention_rates]

    ax.plot(retention_rates, total_seats_curve, linewidth=3,
            color='#1f77b4', label='LDP Total Seats')

    # Mark our three scenarios
    for scenario, data in scenarios.items():
        ax.scatter(data['retention'], data['total'], s=200,
                   color='red', zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f"{scenario}\n{data['total']} seats",
                    xy=(data['retention'], data['total']),
                    xytext=(10, 15), textcoords='offset points',
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

    # Majority line
    ax.axhline(y=MAJORITY_LINE, color='black', linestyle='--',
               linewidth=2, label=f'Majority Line: {MAJORITY_LINE}')

    # Shaded region (Below majority)
    ax.fill_between(retention_rates, 0, MAJORITY_LINE, alpha=0.2,
                    color='red', label='Minority Gov\'t (少数政权)')

    ax.set_xlabel('Coalition Retention Rate (新党选民保留率) %',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('LDP Total Seats (自民党总议席)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Analysis: Coalition Success vs LDP Performance\n敏感性分析：新党成功率 vs 自民党议席',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(40, 100)
    ax.set_ylim(140, 210)

    plt.tight_layout()
    plt.savefig('results/sensitivity_curve.png', dpi=300, bbox_inches='tight')
    print("[Saved] results/sensitivity_curve.png")
    plt.close()


def plot_coalition_leakage_breakdown():
    """
    Plot 3: Pie chart showing where leaked coalition votes go
    图表3：饼图展示新党流失选票的去向
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios_data = [
        ('Optimistic\n(10% Leakage)', [5, 4, 1], [
         'Abstention\n弃权', 'Back to LDP\n回流自民', 'Other Parties\n其他小党']),
        ('Baseline\n(30% Leakage)', [15, 12, 3], [
         'Abstention\n弃权', 'Back to LDP\n回流自民', 'Other Parties\n其他小党']),
        ('Pessimistic\n(50% Leakage)', [25, 20, 5], [
         'Abstention\n弃权', 'Back to LDP\n回流自民', 'Other Parties\n其他小党'])
    ]

    colors = ['#ff9999', '#66b3ff', '#99ff99']

    for ax, (title, sizes, labels) in zip(axes, scenarios_data):
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('Where Do Leaked Coalition Votes Go?\n新党流失选票的去向分析',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('results/leakage_breakdown.png', dpi=300, bbox_inches='tight')
    print("[Saved] results/leakage_breakdown.png")
    plt.close()


def generate_summary_table():
    """
    Generate a summary table in Markdown format
    生成Markdown格式的汇总表
    """
    table_md = """
## Simulation Results Summary / 模拟结果汇总

| Scenario | Retention Rate | SMD Seats | PR Seats | Total Seats | Majority? |
|----------|----------------|-----------|----------|-------------|-----------|
"""

    for scenario, data in scenarios.items():
        scenario_clean = scenario.replace('\n', ' ')
        majority_status = "✓ YES" if data['total'] >= MAJORITY_LINE else "✗ NO"
        table_md += f"| {scenario_clean} | {data['retention']}% | {data['smd']} | {data['pr']} | **{data['total']}** | {majority_status} |\n"

    table_md += f"\n**Majority Line**: {MAJORITY_LINE} / {TOTAL_SEATS} seats\n"

    with open('results/summary_table.md', 'w', encoding='utf-8') as f:
        f.write(table_md)

    print("[Saved] results/summary_table.md")


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)

    print("="*60)
    print("Generating Visualization / 生成可视化图表")
    print("="*60)

    plot_scenario_comparison()
    plot_sensitivity_curve()
    plot_coalition_leakage_breakdown()
    generate_summary_table()

    print("\n✓ All visualizations generated successfully!")
    print("✓ 所有可视化图表生成成功！")
