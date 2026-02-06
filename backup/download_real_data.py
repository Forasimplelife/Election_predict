import os
import pandas as pd
import requests
import io

# 定义目标保存目录
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Yusaku Horiuchi 教授的选举数据仓库Mapping
# 格式: Year: (Repo_Name, File_Path_in_Repo)
# 注意：2024年的repo名称可能需要根据实际情况微调，此处基于常见命名惯例 'jelec2024lh' 推断
# 如果下载失败，请手动确认GitHub上具体的raw url
REPOS = {
    2021: {
        "smd": "https://raw.githubusercontent.com/yhoriuchi/jelec2021lh/main/output/smd_data.csv",
        "pr": "https://raw.githubusercontent.com/yhoriuchi/jelec2021lh/main/output/pr_data.csv"
    },
    2017: {
        "smd": "https://raw.githubusercontent.com/yhoriuchi/jelec2017lh/master/output/smd_data.csv", # 2017通常在master分支
        "pr": "https://raw.githubusercontent.com/yhoriuchi/jelec2017lh/master/output/pr_data.csv"
    },
    2024: {
        # 假设2024的命名规则一致。如果不一致，可以在浏览器访问 github.com/yhoriuchi 查看最新repo名
        "smd": "https://raw.githubusercontent.com/yhoriuchi/jelec2024lh/main/output/smd_data.csv",
        "pr": "https://raw.githubusercontent.com/yhoriuchi/jelec2024lh/main/output/pr_data.csv"
    }
}

# e-stat相关的仓库 (如果存在)
# 由于estat通常涉及API或复杂Excel，这里仅作为占位符或特定CSV下载
# 如果 Yusaku 老师有专门的 estat repo，也需要确认 URL

def download_csv(url, save_name):
    print(f"Downloading {save_name} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # 检查请求是否成功
        
        # 使用pandas读取以确认它是有效的CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # 保存到本地
        save_path = os.path.join(DATA_DIR, save_name)
        df.to_csv(save_path, index=False)
        print(f" -> Saved to {save_path} (Rows: {len(df)})")
        return df
    except Exception as e:
        print(f" -> Failed to download or parse {url}")
        print(f"    Error: {e}")
        return None

def main():
    print("=== Starting Real Data Download from GitHub ===")
    
    # 1. 下载各年份的众议院选举数据
    for year, urls in REPOS.items():
        print(f"\n--- Processing Year {year} ---")
        # 下载小选区 (Single Member District) 数据
        download_csv(urls["smd"], f"real_election_{year}_smd.csv")
        # 下载比例代表 (Proportional Representation) 数据
        download_csv(urls["pr"], f"real_election_{year}_pr.csv")

    print("\n=== Download Complete ===")
    print("Check the 'data' folder for the files.")
    print("If 2024 or other files failed, please check the repository URL manually.")

if __name__ == "__main__":
    main()
