from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent / "data"

data = pd.read_csv(
    DATA_DIR / "race_results_with_features.csv"
)

data = data.sort_values(
    ["year", "round", "driverCode"]
).reset_index(drop=True)

# Alleen informatie die vóór de race beschikbaar is
numeric_features = [
    "grid",
    "avg_finish_last_5",
    "previous_races_used",
    "finish_same_race_previous_year",
    "team_avg_finish_last_5",
]

categorical_features = [
    "driverCode",
    "constructorId",
    "race",
]

features = numeric_features + categorical_features
target = "position"

for column in numeric_features + ["year", "round", target]:
    data[column] = pd.to_numeric(data[column], errors="raise")

# Een gridwaarde van 0 behandelen als ontbrekende startpositie.
# Bewaar apart of het om zo'n start gaat.
data["special_start"] = data["grid"].eq(0).astype(int)
data["grid"] = data["grid"].replace(0, float("nan"))
numeric_features.append("special_start")
features = numeric_features + categorical_features

if data[["year", "round", "driverCode", target]].isna().any().any():
    raise ValueError("Er ontbreken racegegevens of finishposities.")

if data.duplicated(["year", "round", "driverCode"]).any():
    raise ValueError("Dubbele coureurs binnen dezelfde race.")

train = data[data["year"].between(2022, 2023)].copy()
validation = data[data["year"] == 2024].copy()
test = data[data["year"] == 2025].copy()

if train.empty or validation.empty or test.empty:
    raise ValueError("Er ontbreken gegevens voor de gekozen seizoenen.")

# Ontbrekende historische waarden invullen op basis van trainingsdata
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print(f"Trainen op {len(train)} coureur-raceresultaten...", flush=True)
model.fit(train[features], train[target])


def evaluate_races(frame, label):
    results = frame[
        ["year", "round", "race", "driverCode", "grid", target]
    ].copy()

    # Lagere voorspelde waarde betekent een betere finishpositie
    results["predicted_score"] = model.predict(frame[features])

    # Sorteren per race levert unieke posities op.
    # driverCode geeft een vaste volgorde bij gelijke scores.
    results = results.sort_values(
        ["year", "round", "predicted_score", "driverCode"]
    )
    results["predicted_position"] = (
        results.groupby(["year", "round"]).cumcount() + 1
    )

    # Eenvoudige vergelijking: eindvolgorde = startvolgorde.
    # Ontbrekende gridwaarden komen achteraan.
    baseline = results.sort_values(
        ["year", "round", "grid", "driverCode"],
        na_position="last",
    ).copy()

    baseline["baseline_position"] = (
        baseline.groupby(["year", "round"]).cumcount() + 1
    )

    results = results.merge(
        baseline[
            ["year", "round", "driverCode", "baseline_position"]
        ],
        on=["year", "round", "driverCode"],
        validate="one_to_one",
    )

    model_mae = mean_absolute_error(
        results[target],
        results["predicted_position"],
    )
    baseline_mae = mean_absolute_error(
        results[target],
        results["baseline_position"],
    )

    print(f"\n{label}")
    print(f"Model: gemiddeld {model_mae:.2f} plaatsen afwijking")
    print(f"Startvolgorde: gemiddeld {baseline_mae:.2f} plaatsen afwijking")

    results = results.sort_values(
        ["year", "round", "predicted_position"]
    )

    results.to_csv(
        DATA_DIR / f"predictions_{label}.csv",
        index=False,
    )

    return results


validation_results = evaluate_races(validation, "2024")

# Laatste test met hetzelfde model, zonder op 2025 te trainen
test_results = evaluate_races(test, "2025")

# Toon de voorspelde volgorde van de laatste testrace
last_round = test_results["round"].max()

print("\nVoorspelde eindvolgorde van de laatste testrace:")
print(
    test_results.loc[
        test_results["round"] == last_round,
        ["driverCode", "predicted_position", "position"],
    ].to_string(index=False)
)

validation_results["error"] = (
    validation_results["predicted_position"]
    - validation_results["position"]
).abs()

print(
    validation_results.groupby("race")["error"]
    .mean()
    .sort_values(ascending=False)
)

print(
    validation_results.groupby("driverCode")["error"]
    .mean()
    .sort_values(ascending=False)
)

validation_results["signed_error"] = (
    validation_results["predicted_position"]
    - validation_results["position"]
)

driver_summary = (
    validation_results.groupby("driverCode")
    .agg(
        races=("error", "size"),
        mean_error=("error", "mean"),
        mean_signed_error=("signed_error", "mean"),
    )
    .sort_values("mean_error", ascending=False)
)

print(driver_summary.round(2))

print("Features:", features)
print("Predicted column:", target)