#!/bin/python
#creating file that plots driver standings

import pandas as pd
import plotly.express as px
from plotly.io import show

from fastf1.ergast import Ergast
import time

ergast = Ergast()
races = ergast.get_race_schedule(2022)  # Races in year 2022
results = []

# For each race in the season
for rnd, race in races['raceName'].items():
    print(f"Round {rnd + 1}: {race} getting information...", flush=True)
    round_number = rnd + 1

    # Wacht vóór iedere API-aanvraag
    time.sleep(2)
    response = ergast.get_race_results(
        season=2022, round=round_number
    )
    temp = response.content[0].copy()

    time.sleep(2)
    sprint = ergast.get_sprint_results(
        season=2022, round=round_number
    )

    if sprint.content:
        sprint_points = sprint.content[0][['driverCode', 'points']]

        temp = pd.merge(
            temp,
            sprint_points,
            on='driverCode',
            how='left',
            suffixes=('', '_sprint')
        )

        temp['points'] = (
            temp['points'] + temp['points_sprint'].fillna(0)
        )
        temp.drop(columns='points_sprint', inplace=True)

    temp['round'] = round_number
    temp['race'] = race.removesuffix(' Grand Prix')
    temp = temp[['round', 'race', 'driverCode', 'points']]
    results.append(temp)
    print("Racegegevens ontvangen", flush=True)


# Append all races into a single dataframe
results = pd.concat(results)
races = results['race'].drop_duplicates()


results = results.pivot(index='driverCode', columns='round', values='points')
# Here we have a 22-by-22 matrix (22 races and 22 drivers, incl. DEV and HUL)

# Rank the drivers by their total points
results['total_points'] = results.sum(axis=1)
results = results.sort_values(by='total_points', ascending=False)
results.drop(columns='total_points', inplace=True)

# Use race name, instead of round no., as column names
results.columns = races

fig = px.imshow(
    results,
    text_auto=True,
    aspect='auto',  # Automatically adjust the aspect ratio
    color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                            [0.25, 'rgb(107, 174, 214)'],
                            [0.5,  'rgb(33,  113, 181)'],
                            [0.75, 'rgb(8,   81,  156)'],
                            [1,    'rgb(8,   48,  107)']],
    labels={'x': 'Race',
            'y': 'Driver',
            'color': 'Points'}       # Change hover texts
)
fig.update_xaxes(title_text='')      # Remove axis titles
fig.update_yaxes(title_text='')
fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                 showline=False,
                 tickson='boundaries')              # Show horizontal grid only
fig.update_xaxes(showgrid=False, showline=False)    # And remove vertical grid
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
fig.update_layout(coloraxis_showscale=False)        # Remove legend
fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # Remove border margins
fig
show(fig)