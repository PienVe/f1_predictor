#!/bin/python
# Getting Data for F1 Prediction Model
##trying to get the data from the Ergast API and the FastF1 API. With this data
##we are create the first machine learning model to predict the race winner

import fastf1 # type: ignore
from pathlib import Path

Path('ml_model/data').mkdir(parents=True, exist_ok=True)

season_2025 = fastf1.get_event_schedule(2025, include_testing=False)

season_2025.to_csv('ml_model/data/season_2025.csv', index=False)

for index, race in season_2025.iterrows():
    #selecting the event name, round number, location and country from the dataframe
    event_name = race['EventName']
    event_round = race['RoundNumber']
    event_location = race['Location']
    event_country = race['Country']
    print(f"Getting data for {event_name} - Round {event_round} in {event_location}, {event_country}")
    #see if there is a sprint race for the event, if there is then get the data for the sprint race as well
    if race['EventFormat'] == 'sprint_qualifying':
        #get the sprintrace data for the event
        print(f"Getting data for {event_name} - Round {event_round} in {event_location}, {event_country} (Sprint Race)")
        sprint = fastf1.get_session(2025, event_round, 'S') #this one is specifically for the sprint race
        sprint.load()
        sprint_quali = fastf1.get_session(2025, event_round, 'SQ') #this one is specifically for the sprint qualifying
        sprint_quali.load()
        sprint_results = sprint.results
        sprint_quali_results = sprint_quali.results
    race = fastf1.get_session(2025, event_round, 'R')
    race.load()
    race_results = race.results
    race_quali = fastf1.get_session(2025, event_round, 'Q')
    race_quali.load()
    race_quali_results = race_quali.results


    