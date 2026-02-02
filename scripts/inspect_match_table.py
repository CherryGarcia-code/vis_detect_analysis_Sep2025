import pandas as pd
import os

def main():
    # Define the path relative to the script execution location (assuming root of repo)
    file_path = os.path.join('data', 'unit_match', 'output', 'BG_046', 'MatchTable.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print the columns
        print("Columns in MatchTable.csv:")
        for col in df.columns:
            print(f"- {col}")
            
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()
