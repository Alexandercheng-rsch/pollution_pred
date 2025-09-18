from openaq import OpenAQ
from openaq.shared.exceptions import RateLimitError
from datetime import datetime, timedelta, timezone
import time
import sys
import re
import pandas as pd 
import pickle
import httpx

sens = {}
keys = []
records = []
req = ['no2', 'o3', 'pm25']
# 10, 32
location_coordinates = []
with OpenAQ(api_key="51c217c5dde05073698ed0c97a35aa86133048d272a49627f3b40949c4e05b30") as client:
    locations_response = client.locations.list(countries_id=79, limit=1000)
    location_id_to_check = locations_response.results
    for items in location_id_to_check:
        ids = [stuff.parameter.name for stuff in items.sensors]
        print(ids)

        if set(req).issubset(set(ids)): # Check to ensure that no2, o3, pm25 is collected by the station
            temp_dic = {}
            for k in items.sensors:
                if k.parameter.name in req:
                    temp_dic[k.parameter.name] = k.id

            sens[items.name] = temp_dic
            keys.append(items.name)
        
            temp_dic_2 = {
                'location': items.name,
                'longitude': items.coordinates.longitude,
                'latitude' : items.coordinates.latitude
            }

            location_coordinates.append(temp_dic_2)

    if len(keys)==0:
        print('Failed to grab locations')
        sys.exit()

    overall_end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=3)
    overall_start_date = overall_end_date - timedelta(days=1095)
    chunk_size_days = 50
    current_start = overall_start_date

    for idx, sensors in enumerate(keys):
        print(f'Currently fetching sensor {sensors}')
        current_start = overall_start_date

        while current_start < overall_end_date:
            current_end = current_start + timedelta(days=chunk_size_days)
            if current_end > overall_end_date:
                current_end = overall_end_date
            date_from_str = current_start.isoformat()
            date_to_str = current_end.isoformat()
            print(f"Fetching data from: {date_from_str} to {date_to_str}")


            for particles in req:
                for attempt in range(3):
                    try:
                        measure = client.measurements.list(sensors_id=sens[keys[idx]][particles], datetime_from=date_from_str, datetime_to=date_to_str, limit=1000)
                        measure = measure.results

                        for j in measure:
                            fuck = {
                                'station': sensors,
                                'time': j.period.datetime_from.utc,
                                'parameter': particles,
                                'value': j.value,
                                'units': j.parameter.units
                            }
                            records.append(fuck)
                        break
                    except RateLimitError as e:
                        print(f"   > {e}")
                        try:
                            wait_time = int(re.search(r'(\d+)', str(e)).group(1))
                            print(f"   > Waiting for {wait_time + 1} seconds before retrying...")
                            time.sleep(wait_time + 1)
                        except:
                            print("   > Could not parse wait time. Waiting 60 seconds.")
                            time.sleep(60)
                    except httpx.ReadTimeout as ee:
                        print(f'>    {ee}')
                        wait_time = 5 * (attempt + 1)
                        time.sleep(wait_time)

            current_start = current_end


with open("datas2.p", "wb") as f:
    pickle.dump(records, f)
print(len(location_coordinates))
with open("locations2.p", "wb") as f:
    pickle.dump(location_coordinates, f)






