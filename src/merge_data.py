import pandas as pd
import os
import re

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELECTION_RESULT_FILE = os.path.join(
    PROJECT_ROOT, "data", "election_results", "smd_data2024.csv")
DEMOGRAPHICS_FILE = os.path.join(PROJECT_ROOT, "data", "demographics_real.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "training_dataset.csv")


def extract_district_total_rows(df):
    """
    从 smd_data2024.csv 中提取包含 "合計" 或 "計" 的行，代表该选区的最终汇总。
    """
    # 筛选 municipality 列包含 '計' 的行
    # 注意：实际上有些文件可能是 "三重県第１区 合計" 这种格式
    # 观察用户提供的 sample: "三重県第１区 合計"

    # 1. 过滤出合计行
    df_totals = df[df['municipality'].astype(str).str.contains('計')].copy()

    # 2. 只需要主要政党的数据 (LDP, CDP, Ishin, Komeito)
    # Pivot 一下，把政党变成列
    # 目标结构: District | LDP_Votes | CDP_Votes | ... | Total_Votes

    # 首先清洗政党名称，避免 "自由民主党(1)" 这种异体
    party_map = {
        '自由民主党': 'LDP',
        '自民': 'LDP',
        '立憲民主党': 'CDP',
        '立憲': 'CDP',
        '日本維新の会': 'ISHIN',
        '維新': 'ISHIN',
        '公明党': 'KOMEITO',
        '日本共産党': 'JCP',
        '共産': 'JCP',
        '国民民主党': 'DPP',
        '国民': 'DPP',
        'れいわ新選組': 'REIWA',
        'れいわ': 'REIWA',
        '参政党': 'SANSEI',
        '社民党': 'SDP',
        'みんなでつくる党': 'MINNA',
        '無所属': 'INDEPENDENT'
    }

    def normalize_party(p_name):
        for k, v in party_map.items():
            if k in str(p_name):
                return v
        return 'OTHER'

    df_totals['party_code'] = df_totals['parties'].apply(normalize_party)

    # 3. Pivot table:
    # Index: district
    # Columns: party_code
    # Values: votes
    df_pivot = df_totals.pivot_table(
        index='district', columns='party_code', values='votes', aggfunc='sum', fill_value=0)

    # 计算总票数 (Turnout)
    df_pivot['Total_Votes'] = df_pivot.sum(axis=1)

    # 计算 LDP 得票率
    if 'LDP' in df_pivot.columns:
        df_pivot['LDP_Share'] = df_pivot['LDP'] / df_pivot['Total_Votes']
    else:
        df_pivot['LDP_Share'] = 0.0

    return df_pivot.reset_index()


def clean_district_name(name):
    """
    将 CSV 中的 "三重県第１区" 清洗为标准格式以便匹配。
    我们需要建立一种通用的匹配键。
    demographics_real.csv 里有 'district_name' (如 "北海道1区" from kucode translation, or similar)
    但是 demographics_real.csv 实际上只有 kucode (101, 102...) 和 pref_id。
    我们需要一个映射逻辑，把 kucode 101 -> 北海道1区
    或者把 election_data 的 "北海道第１区" -> kucode 101
    """
    # 统一全角数字为半角
    name = str(name).translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    # 移除空格 "三重県第1区 合計" -> "三重県第1区"
    name = name.replace(' 合計', '').replace('計', '').strip()
    return name


def generate_kucode_from_name(district_name):
    """
    (Heuristic) 尝试从中文选区名生成 kucode
    规则大致是: Pref_ID * 100 + District_No
    你需要一个 县名 -> Pref_ID 的映射表
    """
    pref_map = {
        '北海道': 1, '青森': 2, '岩手': 3, '宮城': 4, '秋田': 5, '山形': 6, '福島': 7,
        '茨城': 8, '栃木': 9, '群馬': 10, '埼玉': 11, '千葉': 12, '東京': 13, '神奈川': 14,
        '新潟': 15, '富山': 16, '石川': 17, '福井': 18, '山梨': 19, '長野': 20,
        '岐阜': 21, '静岡': 22, '愛知': 23, '三重': 24, '滋賀': 25, '京都': 26,
        '大阪': 27, '兵庫': 28, '奈良': 29, '和歌山': 30, '鳥取': 31, '島根': 32,
        '岡山': 33, '広島': 34, '山口': 35, '徳島': 36, '香川': 37, '愛媛': 38,
        '高知': 39, '福岡': 40, '佐賀': 41, '長崎': 42, '熊本': 43, '大分': 44,
        '宮崎': 45, '鹿児島': 46, '沖縄': 47
    }

    # Extract Prefecture Name
    # district_name like "北海道第1区" or "東京都第10区"
    match = re.search(r'(.+?)[都道府県]?第(\d+)区',
                      clean_district_name(district_name))
    if match:
        pref_str = match.group(1)
        dist_num = int(match.group(2))

        # 处理特殊县名后缀
        if pref_str.endswith('県') or pref_str.endswith('府') or pref_str.endswith('都'):
            pref_str = pref_str[:-1]

        pref_id = pref_map.get(pref_str)
        if pref_id:
            return pref_id * 100 + dist_num

    return None


def main():
    print("=== Merging Election Results with Demographics ===")

    # 1. Load Election Results
    print(f"Loading Election Data: {ELECTION_RESULT_FILE}")
    if not os.path.exists(ELECTION_RESULT_FILE):
        print(f"[Error] File not found: {ELECTION_RESULT_FILE}")
        return

    df_elect = pd.read_csv(ELECTION_RESULT_FILE)
    df_votes = extract_district_total_rows(df_elect)

    # 添加 kucode 以便 Join
    # 注意：这里的 'district' 列来自于 smd_data2024.csv，内容是 "三重県第１区"
    df_votes['kucode'] = df_votes['district'].apply(generate_kucode_from_name)

    # Check match rate
    matched = df_votes['kucode'].notna().sum()
    print(f"Parsed {matched} / {len(df_votes)} district codes.")

    # 2. Load Demographics (Real)
    print(f"Loading Demographics: {DEMOGRAPHICS_FILE}")
    if not os.path.exists(DEMOGRAPHICS_FILE):
        print(f"[Error] File not found: {DEMOGRAPHICS_FILE}")
        return

    df_demo = pd.read_csv(DEMOGRAPHICS_FILE)

    # 3. Merge
    # Outer join to see if we miss anything due to code mismatch
    df_final = pd.merge(df_demo, df_votes, on='kucode', how='left')

    print(f"Merged Data Shape: {df_final.shape}")

    # 4. Save
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"[Success] Created Training Dataset: {OUTPUT_FILE}")
    print("Sample Columns:", df_final.columns.tolist())

    # Check for missing LDP Shares (indicates join failure or missing election data)
    missing_votes = df_final['LDP_Share'].isna().sum()
    if missing_votes > 0:
        print(
            f"[Warning] {missing_votes} districts have no election result match (maybe kucode mismatch).")


if __name__ == "__main__":
    main()
