import kagglehub
import shutil
import os
import pandas as pd

def download_churn_dataset():
    """
    Download customer churn dataset from Kaggle and prepare it
    """
    try:
        # Download the Telco Customer Churn dataset
        dataset_id = "yeanzc/telco-customer-churn-ibm-dataset"
        
        print("Downloading customer churn dataset from Kaggle...")
        path = kagglehub.dataset_download(dataset_id)
        
        print(f"Dataset downloaded to: {path}")
        
        # List all files in the downloaded dataset
        files = os.listdir(path)
        print(f"Files in dataset: {files}")
        
        # Find CSV or Excel files
        data_files = [f for f in files if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if data_files:
            # Use the first data file
            main_file = data_files[0]
            source_file = os.path.join(path, main_file)
            destination_csv = os.path.join(os.path.dirname(__file__), 'raw.csv')
            
            print(f"Processing file: {main_file}")
            
            # Read the file and convert to CSV if needed
            if main_file.endswith('.csv'):
                # Just copy the CSV file
                shutil.copy2(source_file, destination_csv)
                print(f"CSV file copied to: {destination_csv}")
            else:
                # Read Excel file and save as CSV
                print("Converting Excel file to CSV...")
                df = pd.read_excel(source_file)
                df.to_csv(destination_csv, index=False)
                print(f"Excel file converted and saved to: {destination_csv}")
            
            # Display basic info about the dataset
            df = pd.read_csv(destination_csv)
            print(f"\nDataset Info:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            # Check for potential target column
            churn_columns = [col for col in df.columns if 'churn' in col.lower()]
            if churn_columns:
                print(f"\nPotential target column(s): {churn_columns}")
                for col in churn_columns:
                    print(f"{col} values: {df[col].value_counts().to_dict()}")
            
            return destination_csv
        else:
            print("No data files (CSV/Excel) found in the dataset")
            return None
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have kagglehub installed: pip install kagglehub")
        print("2. Make sure you have pandas and openpyxl installed: pip install pandas openpyxl")
        print("3. Authenticate with Kaggle (set up API key)")
        print("4. Check if the dataset ID is correct")
        return None

if __name__ == "__main__":
    result = download_churn_dataset()
    if result:
        print(f"\n✅ Success! Dataset ready at: {result}")
    else:
        print("\n❌ Failed to download dataset")