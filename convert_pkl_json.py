import pickle
import json
import pandas as pd
import datetime

# Custom JSON encoder that handles DataFrames
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            # Convert DataFrame to a list of records or a dictionary compatible format
            return obj.to_dict(orient='records') 
        if isinstance(obj, pd.Series):
             return obj.to_dict()
        return super().default(obj)

def convert_pkl_to_json(pkl_path, json_path):
    print(f"Converting {pkl_path} to {json_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # If the keys are timestamps, convert them to strings
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_key = k
                if isinstance(k, (datetime.date, datetime.datetime, pd.Timestamp)):
                    new_key = k.isoformat()
                elif hasattr(k, 'isoformat'): 
                     new_key = k.isoformat()
                new_data[str(new_key)] = v
            data = new_data

        with open(json_path, 'w') as f:
            json.dump(data, f, cls=CustomEncoder, indent=4)
        print(f"Successfully converted {pkl_path} to {json_path}")
        return True
    except Exception as e:
        print(f"Error converting {pkl_path}: {e}")
        return False

if __name__ == "__main__":
    files_to_convert = [
        ('data/federal_gpt_answ.pkl', 'data/federal_gpt_answ.json'),
        ('data/referendum_gpt_answ.pkl', 'data/referendum_gpt_answ.json')
    ]

    for pkl, json_file in files_to_convert:
        if convert_pkl_to_json(pkl, json_file):
            print(f"Conversion successful for {pkl}")
        else:
             print(f"Conversion failed for {pkl}")
