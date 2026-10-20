# Add historical features to the race results:
# - Average finish position in the previous 5 races
# - Number of previous races used
# - Finish position at the same Grand Prix in the previous year

import pandas as pd
from pathlib import Path

data_folder = Path('ml_model/data')

# Voeg andere jaren toe zodra de CSV-bestanden beschikbaar zijn
years = [2022, 2023, 2024, 2025, 2026]

yearly_results = []

# Lees de race results voor ieder jaar
for year in years:
    file_path = data_folder / f'race_results_{year}.csv'
    results = pd.read_csv(file_path)

    results['year'] = year
    yearly_results.append(results)

# Combineer alle jaren
all_results = pd.concat(yearly_results, ignore_index=True)

# Zorg dat de finishpositie een getal is
all_results['position'] = pd.to_numeric(
    all_results['position'],
    errors='raise'
)

# Zet alle races in chronologische volgorde
all_results = all_results.sort_values(
    by=['year', 'round', 'driverCode']
).reset_index(drop=True)

# Controleer op dubbele resultaten
if all_results.duplicated(['year', 'round', 'driverCode']).any():
    raise ValueError(
        'Er staan dubbele resultaten voor een coureur in een race.'
    )

# Gemiddelde finishpositie in maximaal 5 eerdere races
all_results['avg_finish_last_5'] = (
    all_results.groupby('driverCode')['position']
    .transform(
        lambda positions: (
            positions.shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    )
)

# Aantal eerdere resultaten gebruikt voor het gemiddelde
all_results['previous_races_used'] = (
    all_results.groupby('driverCode')['position']
    .transform(
        lambda positions: (
            positions.shift(1)
            .rolling(window=5, min_periods=0)
            .count()
        )
    )
    .astype(int)
)

# NIEUW: finishpositie bij dezelfde Grand Prix in het vorige jaar
previous_year_results = all_results[
    ['year', 'race', 'driverCode', 'position']
].copy()

# Een resultaat uit 2022 wordt zo gekoppeld aan een rij uit 2023
previous_year_results['year'] += 1

previous_year_results = previous_year_results.rename(
    columns={
        'position': 'finish_same_race_previous_year'
    }
)

all_results = all_results.merge(
    previous_year_results,
    on=['year', 'race', 'driverCode'],
    how='left',
    validate='one_to_one'
)

# Sorteer opnieuw voor het opslaan
all_results = all_results.sort_values(
    by=['year', 'round', 'driverCode']
).reset_index(drop=True)

# Eén rij per team per race:
# gemiddelde finishpositie van alle deelnemende teamcoureurs
team_results = (
    all_results
    .groupby(
        ['year', 'round', 'constructorId'],
        as_index=False
    )
    .agg(team_avg_finish=('position', 'mean'))
    .sort_values(['year', 'round', 'constructorId'])
)

# Gemiddelde teamprestatie over maximaal vijf eerdere races.
# shift(1) sluit de huidige race volledig uit.
team_results['team_avg_finish_last_5'] = (
    team_results.groupby('constructorId')['team_avg_finish']
    .transform(
        lambda positions: (
            positions.shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    )
)

# Koppel hetzelfde historische teamgemiddelde aan beide coureurs
all_results = all_results.merge(
    team_results[
        ['year', 'round', 'constructorId', 'team_avg_finish_last_5']
    ],
    on=['year', 'round', 'constructorId'],
    how='left',
    validate='many_to_one'
)

# Sla de tabel met de nieuwe kenmerken apart op
all_results.to_csv(
    data_folder / 'race_results_with_features.csv',
    index=False
)

# Bekijk de resultaten van Verstappen
print("Race results for Verstappen with historical features:")

print(
    all_results.loc[
        all_results['driverCode'] == 'VER',
        [
            'year',
            'round',
            'race',
            'position',
            'avg_finish_last_5',
            'previous_races_used',
            'finish_same_race_previous_year'
        ]
    ].to_string(index=False)
)