#!C:\Users\VHACONPeterJ\labeler\.labeler\Scripts\python.exe
import pandas as pd
import glob
import os
import sys

def main():
    # 1. Settings
    parser = argparse.ArgumentParser(description="Merge all CSVs in the current folder into one.")
    parser.add_argument(
        "--outfile", 
        required=True, 
        help="The name of the resulting merged CSV (e.g., test.csv)"
    )
    args = parser.parse_args()

    output_filename = args.outfile #'CXR_2017-2025_labeled_MASTER.csv'
    current_dir = os.getcwd()  # Get current working directory

    # 2. Find all CSV files in the current directory
    # os.path.join(current_dir, "*.csv") ensures we look in the right place
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    
    # Ensure they merge in order (0001, 0002, 0003...)
    csv_files.sort() 
    
    # Exclude output file if already exists...
    csv_files = [f for f in csv_files if os.path.basename(f) != output_filename]

    # SAFETY CHECK: Exclude the output file if it already exists
    # so we don't try to read the file we are about to create/overwrite.
    csv_files = [f for f in csv_files if os.path.basename(f) != output_filename]

    if not csv_files:
        print(f"No CSV files found in {current_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to merge in current folder:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # 3. Read and Concatenate
    df_list = []
    for filename in csv_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    if not df_list:
        print("No valid data found.")
        sys.exit(1)

    print("\nMerging files...")
    combined_df = pd.concat(df_list, ignore_index=True)

    # 4. Save Output
    combined_df.to_csv(output_filename, index=False)

    print(f"Success! Merged {len(df_list)} files.")
    print(f"Total rows: {len(combined_df)}")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    main()