import pandas as pd

from model_utils import (
    DATA_DIR,
    FEATURES,
    TARGET,
    create_model,
    load_data,
)

# Deze historische dataset mag alleen eerdere races bevatten.
history = load_data()

# Deelnemers, startopstelling en historie voor de komende race
upcoming = pd.read_csv(DATA_DIR / "upcoming_race.csv")

model = create_model()
model.fit(history[FEATURES], history[TARGET])

upcoming["predicted_score"] = model.predict(upcoming[FEATURES])

upcoming = upcoming.sort_values(
    ["predicted_score", "driverCode"]
).reset_index(drop=True)

upcoming["predicted_position"] = upcoming.index + 1

print(
    upcoming[
        ["driverCode", "predicted_position"]
    ].to_string(index=False)
)

print(f"\nVoorspelde winnaar: {upcoming.iloc[0]['driverCode']}")