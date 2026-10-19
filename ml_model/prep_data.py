#!/bin/python
# Getting Data for F1 Prediction Model
##trying to get the data from the Ergast API and the FastF1 API. With this data
##we are create the first machine learning model to predict the race winner

import pandas as pd
from fastf1.ergast import Ergast 
from pathlib import Path
import time

Path('ml_model/data').mkdir(parents=True, exist_ok=True)

years = list(range(2024, 2025))
years = [2025]  # For now, only get data for 2024. Add other years once the CSV files are available.

for year in years:
    print(f"Getting data for {year}...", flush=True)
    ergast = Ergast()
    time.sleep(2)  # Wait before making the API request
    race_schedule = ergast.get_race_schedule(year)
    race_results = []

    for rnd, race in race_schedule['raceName'].items():
        print(f"Round {rnd + 1}: {race} getting information...", flush=True)
        round_number = rnd + 1
        response = ergast.get_race_results(
            season=year,
            round=round_number
        )
        temp = response.content[0].copy()
        temp['round'] = round_number
        temp['race'] = race.removesuffix(' Grand Prix')
        for col in ['position', 'points', 'grid', 'laps', 'status', 
                    'driverNumber', 'driverCode', 'givenName', 'familyName', 
                    'constructorId', 'constructorName', 'totalRaceTimeMillis', 
                    'totalRaceTime', 'fastestLapRank', 'fastestLapNumber', 
                    'fastestLapTime', 'fastestLapAvgSpeedUnits', 
                    'fastestLapAvgSpeed']:
            if col not in temp.columns:
                temp[col] = None
        temp = temp[['position', 'points', 'grid', 'laps', 'status', 
                     'driverNumber', 'driverCode', 'givenName', 'familyName', 
                     'constructorId', 'constructorName', 'totalRaceTimeMillis', 
                     'totalRaceTime', 'fastestLapRank', 'fastestLapNumber', 
                     'fastestLapTime', 'fastestLapAvgSpeedUnits', 'fastestLapAvgSpeed', 
                     'round', 'race']]
        race_results.append(temp)
        print("Race data received", flush=True)

    # Append all races into a single dataframe
    race_results_df = pd.concat(race_results, ignore_index=True)
    #Add the year column to the dataframe
    race_results_df['year'] = year

    #Sort the dataset based on the round and driverCode columns
    sorted_results_df = race_results_df.sort_values(
        by=['round', 'driverCode']
    )

    sorted_results_df.to_csv(
        f'ml_model/data/race_results_{year}.csv',
        index=False
    )