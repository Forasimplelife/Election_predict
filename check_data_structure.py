import pandas as pd
import os

# Define file paths
file_path_2021 = r'data\election_results\pr_data2021.csv'
file_path_2024 = r'data\election_results\pr_data2024.csv'


def inspect_csv(file_path):
    print(f"--- Inspecting {file_path} ---")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Basic Info
        print(f"Columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")

        # Check Municipality column for split districts (indicated by parenthesis)
        if 'municipality' in df.columns:
            split_districts = df[df['municipality'].str.contains(
                r'[（\(]', regex=True, na=False)]
            split_count = split_districts['municipality'].nunique()
            print(f"Unique municipalities: {df['municipality'].nunique()}")
            print(
                f"Municipalities with parenthesis (potential splits): {split_count}")

            if split_count > 0:
                print("Sample of split municipalities:")
                print(split_districts['municipality'].unique()[:10])

        # Check for 'prefecture' or 'district' info
        if 'prefecture' in df.columns:
            print(f"Prefectures found: {df['prefecture'].nunique()}")
        if 'district' in df.columns:
            print(f"District column sample: {df['district'].unique()[:5]}")

        print("\nHead of data:")
        print(df.head())

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n")


if __name__ == "__main__":
    inspect_csv(file_path_2021)
    inspect_csv(file_path_2024)
