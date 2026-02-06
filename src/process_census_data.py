import pandas as pd
import os
import glob

# 获取当前脚本所在的绝对路径 (src目录)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 由此推导项目根目录 (即 src 的上一级)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 路径配置 (使用绝对路径，避免运行目录不同导致的报错)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "senkyoku2022_toukei")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "demographics_real.csv")


def load_japan_csv(filepath):
    """
    读取日本政府格式的CSV (Shift-JIS编码)
    """
    try:
        # cp932 是 Shift-JIS 的超集，容错率更高
        return pd.read_csv(filepath, encoding='cp932')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def main():
    print("=== Processing Census Data for 2026 Prediction ===")

    # 1. 读取基础人口数据 (含外国人)
    # 02_人口総数_外国人人口_世帯数.csv
    # 假设列: [ken, kuno, kucode, kuname, 总人口, 男, 女, 外国人人口, ...]
    base_file = os.path.join(INPUT_DIR, "02_人口総数_外国人人口_世帯数.csv")
    if not os.path.exists(base_file):
        print(f"[Error] Base file not found: {base_file}")
        return

    df_base = load_japan_csv(base_file)
    # 重命名列以便理解 (根据实际文件头，这里做动态映射或硬编码)
    # 通常第1列是ken, 第3列是kucode, 第5列是Total_Pop, 第8列是Foreigners
    # 打印一下columns确认
    print(f"Base columns: {df_base.columns.tolist()}")

    # 提取关键列 (根据刚才的read_file结果推断)
    # kucode 是唯一键
    df_merged = df_base.iloc[:, [2, 0, 3, 4, 7]].copy()
    df_merged.columns = ['kucode', 'pref_id',
                         'district_name', 'total_pop', 'foreigners']

    # 计算外国人占比
    df_merged['pct_foreigners'] = df_merged['foreigners'] / \
        df_merged['total_pop']

    # 2. 读取产业数据 (农业/建筑业)
    # 11_a_産業（大分類）別就業者数（15歳以上）_男女計.csv
    # 通常结构: A农业, B林业, C渔业, D矿业, E建筑业...
    industry_file = os.path.join(INPUT_DIR, "11_a_産業（大分類）別就業者数（15歳以上）_男女計.csv")
    if os.path.exists(industry_file):
        df_ind = load_japan_csv(industry_file)
        # 假设:
        # Col A-C = Primary Industry (Agriculture)
        # Col E = Construction (往往是单独的一列，或者是第二产业的一部分)
        # 这里我们需要看具体的列名，作为演示，我们假设第5列开始是产业数据
        # 更好的做法是打印列名人工确认一次，但这里我们先合并整个表

        # 简化处理：只保留 kucode 和 所有产业列
        df_ind_subset = df_ind.iloc[:, 2:]  # 保留 kucode 及之后
        df_ind_subset = df_ind_subset.rename(
            columns={df_ind.columns[2]: 'kucode'})

        # Merge
        df_merged = pd.merge(df_merged, df_ind_subset, on='kucode', how='left')
        print(" -> Merged Industry Data")

    # 3. 读取年龄数据 (老龄化)
    # 03_a_年齢別人口_男女計.csv
    age_file = os.path.join(INPUT_DIR, "03_a_年齢別人口_男女計.csv")
    if os.path.exists(age_file):
        df_age = load_japan_csv(age_file)
        # 同样保留 kucode Merge
        df_age_subset = df_age.iloc[:, 2:]
        df_age_subset = df_age_subset.rename(
            columns={df_age.columns[2]: 'kucode'})

        df_merged = pd.merge(df_merged, df_age_subset, on='kucode', how='left')
        print(" -> Merged Age Data")

    # 保存
    df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n[Success] Processed {len(df_merged)} districts.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Columns available:", df_merged.columns.tolist()[:10], "...")


if __name__ == "__main__":
    main()
