#!/bin/python
# Getting Data for F1 Prediction Model
##for this project we are using the FastF1 library to get 
##the data from the Ergast API and the FastF1 API. With this data
##we are creating some plots for practice and getting a better understanding of the data.

import seaborn as sns # type: ignore
from matplotlib import pyplot as plt # type: ignore
import fastf1 # type: ignore
import fastf1.plotting # type: ignore

# Enable Matplotlib patches for plotting timedelta values and load
# FastF1's dark color scheme
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

race = fastf1.get_session(2026, 'Singapore', 'R')
race.load()  # Download and parse all data

#gets the laps for the Max Verstappen for the Singapore GP 2026
driver_laps = race.laps.pick_drivers("VER").pick_quicklaps().reset_index()

fig, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(data=driver_laps,
                x="LapNumber",
                y="LapTime",
                ax=ax,
                hue="Compound",
                palette=fastf1.plotting.get_compound_mapping(session=race),
                s=80,
                linewidth=0,
                legend='auto')

ax.set_xlabel("Lap Number")
ax.set_ylabel("Lap Time")

# The y-axis increases from bottom to top by default
# Since we are plotting time, it makes sense to invert the axis
ax.invert_yaxis()
plt.suptitle("Verstappen Laptimes in the 2026 Singapore Grand Prix")

# Turn on major grid lines
plt.grid(color='w', which='major', axis='both')
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()