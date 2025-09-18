import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
import pickle
import time
import re
from word2number import w2n
import numpy as np
from openmeteo_requests.Client import OpenMeteoRequestsError

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
with open('locations.p', 'rb') as file:
    locations = pickle.load(file)
    

url = "https://archive-api.open-meteo.com/v1/archive"


collection = []
for station in locations:
	print(f'Collection weather information for {station["location"]}')
	params = {
		"latitude": station['latitude'],
		"longitude": station['longitude'],
		"start_date": "2022-07-14",
		"end_date": "2025-07-19",
		"hourly": ['wind_speed_10m', 'wind_direction_10m', 'temperature_2m'
				, 'relative_humidity_2m', 'precipitation', 'rain',
				'surface_pressure', 'pressure_msl', 'shortwave_radiation']
	}
	for attempt in range(3):
		try:
			responses = openmeteo.weather_api(url, params=params)
			break
		except OpenMeteoRequestsError as e:
			print(f'   > {e}')
			msg = str(e).lower()

			wait_time = None
			try:
				match = re.search(r'(\d+)', msg)
				if match:
					wait_time = int(match.group(1))
				else:
					word_match = re.search(r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)+', msg)
					if word_match:
						wait_time = w2n.word_to_num(word_match.group(0))
					else:
						wait_time = 60
				print(f"   > Waiting for {wait_time + 1} seconds before retrying...")
				time.sleep(wait_time + 1)
			except Exception:
				print("   > Could not parse wait time. Waiting 60 seconds.")
				time.sleep(60)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	hourly = response.Hourly()

	hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
	)}

	for k in range(len(params['hourly'])):
		data = hourly.Variables(k).ValuesAsNumpy()
		measure = params['hourly'][k]
		hourly_data['station'] = station['location']
		hourly_data[measure] = data
	collection.append(pd.DataFrame(hourly_data))




df = pd.concat(collection, ignore_index=True)
df.to_csv('ff.csv', index=False)