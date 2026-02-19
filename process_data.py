import os
from src.prepare_data import create_files_per_language

DATA_FOLDER = 'data'

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    print(f"Starting data preparation in '{DATA_FOLDER}'...")
    create_files_per_language(DATA_FOLDER)
    print("Data preparation finished.")
