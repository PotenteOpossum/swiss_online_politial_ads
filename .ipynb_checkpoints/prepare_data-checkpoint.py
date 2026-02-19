import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates, RatesNotAvailableError
import numpy as np
import time
import ast, glob

import json
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from forex_python.converter import CurrencyRates, RatesNotAvailableError
import numpy as np
import time, glob

parties = ['SP', 'GRÜNE', 'GLP', 'Die Mitte', 'FDP', 'SVP']
party_colors = {
	'FDP': '#cc78bc', 'SP': '#0473b2', 'GRÜNE': '#de8f05',
	'SVP': '#ca9161', 'Die Mitte': '#d55e00', 'GLP': '#009e74'
}
cantons = [
	"Zürich",
	"Bern",
	"Aargau",
	"Canton of St. Gallen",
	"Luzern",
	"Basel-City",
	"Solothurn",
	"Basel-Landschaft",
	"Thurgau",
	"Graubünden",
	"Fribourg",
	"Zug",
	"Vaud",
	"Valais",
	"Schaffhausen",
	"Schwyz",
	"Canton of Obwalden",
	"Appenzell Ausserrhoden",
	"Canton of Geneva",
	"Canton of Glarus",
	"Appenzell Innerrhoden",
	"Neuchâtel",
	"Canton of Nidwalden",
	"Jura",
	"Uri",
	"Ticino"
]

def get_df(df):
    c = CurrencyRates()

    # --- Data Cleaning and Preparation ---
    df['impressions_lower'] = pd.to_numeric(df['impressions'].apply(lambda x: x.get('lower_bound', 0)), errors='coerce').fillna(0)
    df['spend_lower'] = pd.to_numeric(df['spend'].apply(lambda x: x.get('lower_bound', 0)), errors='coerce').fillna(0)

    df['impressions_upper'] = pd.to_numeric(df['impressions'].apply(lambda x: x.get('upper_bound', 0)), errors='coerce').fillna(0)
    df['spend_upper'] = pd.to_numeric(df['spend'].apply(lambda x: x.get('upper_bound', 0)), errors='coerce').fillna(0)

    df['spend_avg'] = df['spend_lower'] + (df['spend_upper'] - df['spend_lower'])/2
    df['impressions_avg'] = df['impressions_lower'] + (df['impressions_upper'] - df['impressions_lower'])/2

    df['currency'] = df['currency'].astype(str)
    df['date'] = pd.to_datetime(df['ad_delivery_start_time'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # --- Currency Conversion ---
    print("Converting currencies to CHF. This may take a moment...")
    
    def convert_to_chf(row):
        spend = row['spend_avg']
        currency_code = row['currency']
        
        if spend == 0:
            return 0
        if currency_code == 'CHF':
            return spend
        time.sleep(1)
        try:
            # Note: This uses the latest available exchange rates.
            return c.convert(currency_code, 'CHF', spend)
        except (RatesNotAvailableError, ValueError) as e:
            # Handle cases where the currency code is invalid or not found
            print(f"Warning: Could not convert {spend} {currency_code}. Error: {e}")
            return np.nan # Treat as 0 if conversion fails

    df['spend_chf'] = df.apply(convert_to_chf, axis=1)
    print("Currency conversion complete.")

    # Drop rows where currency conversion failed
    original_rows = len(df)
    df.dropna(subset=['spend_chf'], inplace=True)
    dropped_rows = original_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to currency conversion errors.")

    return df

def create_files_per_language():
	df = pd.DataFrame()
	for lang in ['italian', 'german', 'french']:
		file_pattern = os.path.join(f'{data_folder}/ads/{lang}', '*.json')
		file_list = glob.glob(file_pattern)
	
		data_list = []
		for file in file_list:
			with open(file, 'r') as f:
				content = json.load(f)
				if isinstance(content, list):
					data_list.extend(content)  # Flattens the lists into one big list
				else:
					data_list.append(content)

		temp = pd.DataFrame(data_list)
		temp = temp.drop_duplicates('id')
		temp = get_df(temp)
		temp['lang'] = lang
		df = pd.concat([df, temp])
		return df

def load_data_for_demographic(data_folder=''):
	df = pd.DataFrame()
	for lang in ['italian', 'german', 'french']:
		file_pattern = os.path.join(f'{data_folder}/ads/{lang}', '*.json')
		file_list = glob.glob(file_pattern)
	
		data_list = []
		for file in file_list:
			with open(file, 'r') as f:
				content = json.load(f)
				if isinstance(content, list):
					data_list.extend(content)  # Flattens the lists into one big list
				else:
					data_list.append(content)

		# temp = pd.DataFrame(data_list)
		temp = get_df(data_list)
		temp['lang'] = lang
		df = pd.concat([df, temp])
	
	def get_right_ids(df):
		df = df.dropna(subset=['delivery_by_region', 'spend_chf']).copy()
		
		def parse_json_string(data):
			if isinstance(data, str):
				try:
					return ast.literal_eval(data)
				except (ValueError, SyntaxError):
					return []
			return data if isinstance(data, list) else []
		
		df['regions'] = df['delivery_by_region'].apply(parse_json_string)
		df = df[df['regions'].apply(len) > 0]
		
		# Explode by regions to attribute spend and impressions to each region
		df_exploded = df.explode('regions')
		region_data = pd.json_normalize(df_exploded['regions'])
		df_exploded = pd.concat([df_exploded.reset_index(drop=True), region_data.reset_index(drop=True)], axis=1)
		
		# --- Calculations ---
		df_exploded['region_percentage'] = pd.to_numeric(df_exploded['percentage'], errors='coerce')
		df_exploded['regional_spend'] = df_exploded['spend_chf'] * df_exploded['region_percentage']
		df_exploded['regional_impressions'] = df_exploded['impressions_lower'] * df_exploded['region_percentage']
		
		df_exploded['swiss'] = df_exploded['region'].apply(lambda x: True if x in cantons else False)
		check_r = df_exploded[df_exploded['swiss']==True].groupby('id')[['region_percentage']].sum()
		
		ads_less_than_5 =  set(check_r[check_r['region_percentage']<0.5].index)
		
		df_exploded = df_exploded[~df_exploded['id'].isin(ads_less_than_5)].reset_index(drop=True)
		
		df_exploded = df_exploded[df_exploded['region'].isin(cantons)].reset_index(drop=True)
		
		return list(df_exploded['id'])
	
	ids = get_right_ids(df.copy())
	
	start_date_filter = pd.to_datetime('2021-01-01', utc=True)
	# end_date_filter = '2025-01-10'
	end_date_filter = pd.to_datetime('2025-10-01', utc=True)
	df['date'] = pd.to_datetime(df['date'], utc=True)
	df['ad_creation_time'] = pd.to_datetime(df['ad_creation_time'], utc=True)
	
	df = df[df['date'] >= pd.to_datetime(start_date_filter, utc=True)]
	df = df[df['date'] <= pd.to_datetime(end_date_filter, utc=True)]
	
	print(len(df))
	df = df[df['id'].isin(ids)].reset_index(drop=True)
	print(len(df))
	
	
	name = f'{data_folder}/federal_elections_authors_annotation.csv'
	party = pd.read_csv(f'ad_data/{name}.csv')
	party = party[~party['party_name'].isna()][['page_name', 'party_name']]
	party['party_name'] = party['party_name'].str.replace('GRÜNEN', 'GRÜNE')
	party['party_name'] = party['party_name'].str.replace('PS', 'SP')
	party['party_name'] = party['party_name'].str.replace('Alternative – die Grünen', 'GRÜNE')
	party['party_name'] = party['party_name'].str.replace('SPV', 'SVP')
	df2 = df.merge(party, on='page_name')
	
	parties = [
		'SVP',
		'FDP',
		'Die Mitte',
		'GLP',
		'GRÜNE',
		'SP']
	df2 = df2[df2['party_name'].isin(parties)]

	start_date_filter = pd.to_datetime('2021-01-01', utc=True)
	# end_date_filter = '2025-01-10'
	end_date_filter = pd.to_datetime('2025-10-01', utc=True)
	df2['date'] = pd.to_datetime(df2['date'], utc=True)
	df2['ad_creation_time'] = pd.to_datetime(df2['ad_creation_time'], utc=True)

	df2 = df2[df2['date'] >= pd.to_datetime(start_date_filter, utc=True)]
	df2 = df2[df2['date'] <= pd.to_datetime(end_date_filter, utc=True)]
	return df2