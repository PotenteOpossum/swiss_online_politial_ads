import json
import os
import pandas as pd

from forex_python.converter import CurrencyRates, RatesNotAvailableError
import numpy as np
import time
import ast, glob
from scipy import stats
import pickle, tqdm

parties = ['SP', 'GRÜNE', 'GLP', 'Die Mitte', 'FDP', 'SVP']
party_colors = {
	'FDP': '#cc78bc', 'SP': '#0473b2', 'GRÜNE': '#de8f05',
	'SVP': '#ca9161', 'Die Mitte': '#d55e00', 'GLP': '#009e74'
}
languages = ['DE', 'FR', 'IT']
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

# get referendum official topics
def get_referendum_topics(data_folder=''):
	with open(f'{data_folder}/referendums_topic.json') as f:
		ref_data = json.load(f)
	return ref_data

def get_GPT_federal_election_data(data_folder=''):
	with open(f'{data_folder}/federal_gpt_answ.json') as file:
		gpt_answ = json.load(file)
	return gpt_answ

# for each ad, get the referendum stance
def get_GPT_referendum_data(data_folder=''):
	with open(f'{data_folder}/referendum_gpt_answ.json') as file:
		gpt_answ = json.load(file)
	return gpt_answ

def get_referendum_topics_mapping(data_folder=''):
	with open(f'{data_folder}/topic_mapping.json') as f:
		ref_data = json.load(f)
	return ref_data

def get_GPT_referendum_data_correlation(df, data_folder='', ref_data=None, referendums=None):
	gpt_answ = get_GPT_referendum_data(data_folder)
	referendum_stances = {}
	for k in tqdm.tqdm(ref_data, desc="Processing GPT Answers"):
		# Access gpt_answ using timestamp key
		# Keys in JSON are strings, ensure k is formatted correctly or access directly if k is string
		key_str = pd.to_datetime(k).isoformat()
		if key_str not in gpt_answ:
			# Try simpler string match if isoformat doesn't match exactly
			key_str = str(k)
			
		if key_str in gpt_answ:
			temp_data = gpt_answ[key_str]
			# If it's a list (records), create DataFrame
			if isinstance(temp_data, list):
				temp = pd.DataFrame(temp_data)
			else:
				# It might be in the 'results' dictionary if structure matches original pkl exactly
				if 'results' in temp_data:
					temp = pd.DataFrame(temp_data['results']).T.reset_index()
				else:
					temp = pd.DataFrame(temp_data) # Fallback

			if 'index' in temp.columns:
				temp['index'] = pd.to_numeric(temp['index'])
				temp = temp.rename(columns={'index': 'id'})
			referendum_stances[k] = temp


	stance_titles = []
	titles = []

	for k in ref_data.keys():
		stance_titles += list(referendum_stances[k]['referendum_title'])
		titles += list(ref_data[k].keys())

	stance_titles = list(set(stance_titles))
	titles = list(set(titles))

	ref_ads = {}
	counter = 0
	counter_i = 0
	
	if referendums is None:
		referendums = []

	for r in referendums:
		lower = r - pd.Timedelta(days=30)
		
		df['ad_delivery_start_time'] = pd.to_datetime(df['ad_delivery_start_time'], utc=True)
		df['ad_delivery_stop_time'] = pd.to_datetime(df['ad_delivery_stop_time'], utc=True)

		temp = df[(df['ad_delivery_start_time'] <= r) & (df['ad_delivery_stop_time'] >= lower)].reset_index(drop=True)
		ldf = len(temp)
		counter += ldf
		counter_i += temp['impressions_avg'].sum()

		ref_ads[r] = temp

	referendum_table = []
	# column = 'spend_avg'
	column = 'impressions_avg'

	for k in ref_data.keys():
		k_date = pd.to_datetime(k, utc=True)
		if k_date not in ref_ads:
			print(f"Warning: Date {k} not found in referendum ads. Skipping.")
			continue
			
		ads = ref_ads[k_date].copy()
		print(f"Processing referendum date: {k}")

		temp_ads = referendum_stances[k].merge(df[['id', 'impressions_avg', 'spend_avg']], on='id')

		for t in ref_data[k].keys():
			temp_ads2 = temp_ads[(temp_ads['related']=='yes')&(temp_ads['referendum_title'].isin([t]))]
			tlen = len(temp_ads2)
			timpy = temp_ads2[temp_ads2['stance']=='yes'][column].sum()
			timpn = temp_ads2[temp_ads2['stance']=='no'][column].sum()
			timpne = temp_ads2[temp_ads2['stance']=='unclear'][column].sum()
			referendum_table.append(
				{
					'date': k,
					'title': t,
					'ADs': tlen,
					'Impressions yes': timpy,
					'Impressions no': timpn,
					'Impressions neutral': timpne
				}
			)

	try:
		ref_results_path = f'{data_folder}/refimp_with_results_and_participation.csv'
		results_df = pd.read_csv(ref_results_path)
		referendum_ads_votes = pd.DataFrame(referendum_table).merge(results_df, on=['title', 'date'])
		
		referendum_ads_votes['approved'] = referendum_ads_votes['approved'].map({
			'Yes': 'Accepted',
			'No': 'Rejected'
		})
		return referendum_ads_votes
	except Exception as e:
		print(f"Error loading results CSV or merging: {e}")
		# Return the table mostly constructed so far if merge fails
		return pd.DataFrame(referendum_table)


def get_referendums(data_folder=''):
	referendums = []
	for s in glob.glob(f'{data_folder}/swiss_referendum_data/votes_data/*'):
		referendums.append(s.replace(f'{data_folder}/swiss_referendum_data/votes_data/', ''))
	referendums = sorted(referendums)
	referendums.append('2023-10-22')
	referendums = sorted(pd.to_datetime(referendums, utc=True))
	return referendums

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
	
	
	rate_cache = {}
	
	def convert_to_chf(row):
		spend = row['spend_avg']
		currency_code = row['currency']
		
		if spend == 0:
			return 0
		if currency_code == 'CHF':
			return spend
			
		if currency_code in rate_cache:
			return spend * rate_cache[currency_code]
			
		try:
			# Note: This uses the latest available exchange rates.
			rate = c.get_rate(currency_code, 'CHF')
			rate_cache[currency_code] = rate
			return spend * rate
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

def create_files_per_language(data_folder):
	"""
	Reads JSON files per language, processes them, and saves to CSV in the data folder.
	"""
	for lang in ['italian', 'german', 'french']:
		print(f"Processing {lang}...")
		file_pattern = os.path.join(data_folder, 'ads', lang, '*.json')
		file_list = glob.glob(file_pattern)
		print(f"Found {len(file_list)} files for {lang}")
	
		data_list = []
		for file in file_list:
			with open(file, 'r') as f:
				content = json.load(f)
				if isinstance(content, list):
					data_list.extend(content)  # Flattens the lists into one big list
				else:
					data_list.append(content)

		if not data_list:
			print(f"No data found for {lang}, skipping.")
			continue

		temp = pd.DataFrame(data_list)
		temp = temp.drop_duplicates('id')
		temp = get_df(temp)
		temp['lang'] = lang
		
		output_file = os.path.join(data_folder, f'{lang}.csv')
		temp.to_csv(output_file, index=False)
		print(f"Saved {lang} data to {output_file}")

def load_data_for_demographic(data_folder=''):
	df = pd.DataFrame()
	for lang in ['italian', 'german', 'french']:
		temp = pd.read_csv(f'{data_folder}/{lang}.csv')
		temp['lang'] = lang
		df = pd.concat([df, temp])
		del temp
	
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
	
	
	name = f'{data_folder}/federal_elections_authors_annotation'
	party = pd.read_csv(f'{name}.csv')
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

def get_data_for_topic_modelling(df, referendum_ads_stance, data_folder=''):

	mapping = get_referendum_topics_mapping(data_folder)

	new = {}
	for k in mapping.keys():
		for v in mapping[k]:
			new[v] = k

	temp = pd.DataFrame.from_dict(referendum_ads_stance, orient='index')
	temp = temp.dropna()
	
	keys = pd.DataFrame(temp['keywords'].tolist(), index=temp.index)
	keys = keys.reset_index()
	keys['index'] = pd.to_numeric(keys['index'])
	
	temp = temp.reset_index()
	temp['index'] = pd.to_numeric(temp['index'])
	
	temp = pd.merge(temp, keys, on = 'index')
	temp = temp.rename(columns={
		0: 'keyword 1',
		1: 'keyword 2',
		2: 'keyword 3'
	})
	temp = temp.merge(df[['id', 'impressions_avg']], left_on='index', right_on='id').merge(df[['id', 'page_name', 'lang']], on='id').drop_duplicates('index')

	name = f'{data_folder}/federal_elections_authors_annotation'
	party = pd.read_csv(f'{name}.csv')

	party = party[~party['party_name'].isna()][['page_name', 'party_name']]
	party['party_name'] = party['party_name'].str.replace('GRÜNEN', 'GRÜNE')
	party['party_name'] = party['party_name'].str.replace('PS', 'SP')
	party['party_name'] = party['party_name'].str.replace('Alternative – die Grünen', 'GRÜNE')
	party['party_name'] = party['party_name'].str.replace('SPV', 'SVP')
	temp2 = temp.merge(party, on='page_name')
	
	del temp2['keywords']
	for i in range(1,4):
		temp2['keyword '+str(i)] = temp2['keyword '+str(i)].map(new)
	return temp2

def prepare_data_for_topic_modelling(df):
	from sklearn.preprocessing import MultiLabelBinarizer
	df_processed = df.copy()
	keyword_cols = ['keyword 1', 'keyword 2', 'keyword 3']
	df_processed['topics'] = df_processed[keyword_cols].apply(
		lambda row: [topic for topic in row if pd.notna(topic)], axis=1
	)
	
	mlb = MultiLabelBinarizer()
	topic_dummies = pd.DataFrame(
		mlb.fit_transform(df_processed['topics']),
		columns=mlb.classes_,
		index=df_processed.index
	)
	
	df_processed = pd.concat([df_processed, topic_dummies], axis=1)
	df_processed.drop(columns=keyword_cols + ['topics'], inplace=True)
	return df_processed, list(mlb.classes_)

def reshape_topic_keywords(df_topic, df_processed, parties):
	# 2. Reshape (melt) the keyword columns
	df_long = pd.melt(
		df_topic[df_topic['party_name'].isin(parties)],
		id_vars=['party_name', 'impressions_avg'],
		value_vars=['keyword 1', 'keyword 2', 'keyword 3'],
		value_name='keyword'
	)
	df_long = df_long[df_long['keyword']!='Governance & Politics'].reset_index(drop=True)
	# 3. Drop rows with missing keywords
	df_long.dropna(subset=['keyword'], inplace=True)

	# 4. Group by party and keyword, then sum the impressions
	keyword_totals = df_long.groupby(['party_name', 'keyword'])['impressions_avg'].sum().reset_index()

	# 5. Sort by impressions and select the top 5 for each party
	top_5_per_party = (
		keyword_totals.sort_values(by='impressions_avg', ascending=False)
		.groupby('party_name')
		.head(10)
		.set_index('party_name')
	)
	topics2 = pd.DataFrame(top_5_per_party).reset_index().sort_values('party_name')['keyword'].unique()

	feature_groups = {
		'Agriculture_Food_Security': 'Environment',
		'Business_Regulation': 'Economy',
		'Civil_Liberties_Rights': 'Society',
		'Climate_Environment': 'Environment',
		'Culture_Society': 'Society',
		'Democratic_Process': 'Governance',
		'Digital_Transformation': 'Technology',
		'Economy_Labor': 'Economy',
		'Education_System': 'Society',
		'Energy_Policy': 'Environment',
		'Family_Youth_Policy': 'Society',
		'Foreign_Relations': 'Governance',
		'Gender_LGBTQ_Rights': 'Society',
		'Governance_Politics': 'Governance',
		'Healthcare_System': 'Society',
		'Housing_Rent': 'Economy',
		'Immigration_Asylum': 'Society',
		'Infrastructure_Mobility': 'Environment',
		'Media_Information': 'Technology',
		'National_Security': 'Governance',
		'Pensions_Retirement': 'Economy',
		'Political_Spectrum': 'Governance',
		'Social_Justice_Equality': 'Society',
		'Taxation_Public_Finance': 'Economy',
		'Urban_Regional_Development': 'Environment',
		'Gender_LGBTQ+_Rights': 'Society',
	}
	df_processed = df_processed.rename(
		columns={
			k: k.replace(' & ', '_').replace(' ', '_') for k in df_processed.keys()
		}
	).copy()
	return df_processed, topics2, feature_groups

def prepare_data_for_prediction(df, df2, parties, topics):

	def get_demo(df):
		dft = df.copy()
		
		def parse_json_string(data):
			if isinstance(data, str):
				try:
					return ast.literal_eval(data)
				except (ValueError, SyntaxError):
					return []
			return data if isinstance(data, list) else []
		
		dft['demo'] = dft['demographic_distribution'].apply(parse_json_string)
		def process_demographics(row_data):
			categories = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+',
						'male', 'female', 'unknown']
			results = {cat: 0.0 for cat in categories}
			data_list = []
		
			try:
				if isinstance(row_data, str):
					cleaned_string = row_data.strip()[5:-1]
					data_list = ast.literal_eval(cleaned_string)
				elif isinstance(row_data, list):
					data_list = row_data
		
				for item in data_list:
					percentage = float(item['percentage'])
					age_group = item['age']
					gender = item['gender']
		
					if age_group in results:
						results[age_group] += percentage
					if gender in results:
						results[gender] += percentage
		
			except (ValueError, SyntaxError, KeyError, TypeError):
				return results
		
			return results

		return dft['demo'].apply(process_demographics).apply(pd.Series)

	df_party = df[df['party_name'].isin(parties)][['impressions_avg', 'party_name', 'id']+list(topics)].copy()

	temp = df_party.merge(df2[['id', 'demographic_distribution']], on='id').copy()
	temp = temp.drop_duplicates('id').reset_index(drop=True)

	df_party = pd.concat([df_party.reset_index(drop=True), get_demo(temp).reset_index(drop=True)], axis=1)
	df_party = df_party.rename(
		columns={
			t: t.replace(' ', '_').replace('&', '').replace('__', '_').replace('+','') for t in topics
		}
	)

	df_party['party_code'] = pd.Categorical(df_party['party_name']).codes
	
	party_mapping = dict(enumerate(pd.Categorical(df_party['party_name']).categories))
	
	return df_party, party_mapping