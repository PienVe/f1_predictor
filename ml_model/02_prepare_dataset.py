# Add historical features to the race results:
# - Average finish position in the previous 5 races
# - Number of previous races used
# - Finish position at the same Grand Prix in the previous year

import pandas as pd
from pathlib import Path

data_folder = Path('data')

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


print(
    all_results['status']
    .fillna('ONBEKEND')
    .value_counts(dropna=False)
    .to_string()
)

# Maak de tekst consistent
status = (
    all_results['status']
    .astype('string')
    .str.strip()
)

# Vul deze lijsten aan op basis van je eigen statusoverzicht
technical_statuses = {
    'Engine',
    'Gearbox',
    'Transmission',
    'Clutch',
    'Hydraulics',
    'Electrical',
    'Suspension',
    'Brakes',
    'Differential',
    'Overheating',
    'Mechanical',
    'Power Unit',
    'Turbo',
    'Fuel pressure',
    'Water leak',
    'Water pressure',
    'Water pump',
    'Cooling system',
    'Fuel leak',
    'Front wing',
    'Power loss',
    'Vibrations',
    'Fuel pump',
    'Oil leak',
    'Undertray'

}

incident_statuses = {
    'Accident',
    'Collision',
    'Collision damage',
    'Spun off',
    'Disqualified',
    'Retired',
}

non_start_statuses = {
    'Did not start',
    'Did not qualify',
    'Did not prequalify',
    'Withdrew',
}

all_results['status_category'] = 'unknown'

# Ook "+1 Lap" en "+2 Laps" betekenen dat een coureur finishte,
# maar met een rondeachterstand.
finished = (
    status.eq('Finished')
    | status.str.fullmatch(r'\+\d+ Laps?', na=False)
    | status.str.fullmatch(r'Lapped', na=False)
)

all_results.loc[
    finished.fillna(False), 'status_category'
] = 'finished'

all_results.loc[
    status.isin(technical_statuses), 'status_category'
] = 'technical'

all_results.loc[
    status.isin(incident_statuses), 'status_category'
] = 'incident'

all_results.loc[
    status.isin(non_start_statuses), 'status_category'
] = 'not_started'

# Voorbeeldgewichten, geen vaststaande juiste waarden
weights = {
    'finished': 1.0,
    'technical': 0.2,
    'incident': 0.5,
    'not_started': 0.0,
    'unknown': 0.5,
}

all_results['result_weight'] = (
    all_results['status_category'].map(weights)
)

# Bekijk welke statussen je nog moet indelen
print("\nNog te beoordelen statussen:")
print(
    all_results.loc[
        all_results['status_category'] == 'unknown',
        'status'
    ]
    .value_counts(dropna=False)
    .to_string()
)

# Zorg dat de resultaten chronologisch staan
all_results = all_results.sort_values(
    ['year', 'round', 'driverCode']
).reset_index(drop=True)

# Een ontbrekende finishpositie mag niet bijdragen aan het gemiddelde
all_results['usable_weight'] = (
    all_results['result_weight']
    .where(all_results['position'].notna(), 0.0)
)

all_results['weighted_position'] = (
    all_results['position'].fillna(0)
    * all_results['usable_weight']
)

# Teller: som van finishpositie × gewicht in de vorige vijf races
weighted_sum = (
    all_results.groupby('driverCode')['weighted_position']
    .transform(
        lambda values: (
            values.shift(1)
            .rolling(window=5, min_periods=1)
            .sum()
        )
    )
)

# Noemer: som van de gewichten van diezelfde eerdere races
weight_sum = (
    all_results.groupby('driverCode')['usable_weight']
    .transform(
        lambda values: (
            values.shift(1)
            .rolling(window=5, min_periods=1)
            .sum()
        )
    )
)

# Bij geen bruikbare historie blijft het gemiddelde NaN
all_results['weighted_avg_finish_last_5'] = (
    weighted_sum / weight_sum.where(weight_sum > 0)
)

# Bewaar hoeveel totaalgewicht achter het gemiddelde zit
all_results['finish_weight_last_5'] = weight_sum

# Verwijder tijdelijke hulpkolommen
all_results = all_results.drop(
    columns=['usable_weight', 'weighted_position']
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