from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent / "data"

NUMERIC_FEATURES = [
    "grid",
    "avg_finish_last_5",
    "previous_races_used",
    "finish_same_race_previous_year",
    "team_avg_finish_last_5",
    "special_start",
]

CATEGORICAL_FEATURES = [
    "driverCode",
    "constructorId",
    "race",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "position"


def load_data():
    """Lees en controleer de dataset met historische kenmerken."""
    file_path = DATA_DIR / "race_results_with_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Bestand niet gevonden: {file_path}\n"
            "Voer eerst 02_prepare_dataset.py uit."
        )

    data = pd.read_csv(file_path)

    # special_start maken we hieronder zelf.
    required = [
        "year",
        "round",
        TARGET,
    ] + [
        column for column in FEATURES
        if column != "special_start"
    ]

    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"Ontbrekende kolommen: {missing}")

    for column in ["year", "round", TARGET] + [
        column for column in NUMERIC_FEATURES
        if column != "special_start"
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    identifiers = [
        "year",
        "round",
        "driverCode",
        "constructorId",
        "race",
        TARGET,
    ]

    if data[identifiers].isna().any().any():
        raise ValueError(
            "Er ontbreken racegegevens, teams, coureurs of uitslagen."
        )

    if data.duplicated(["year", "round", "driverCode"]).any():
        raise ValueError("Dubbele coureurs binnen dezelfde race.")

    # Bewaar de oorspronkelijke startpositie voor de uitvoer.
    data["original_grid"] = data["grid"]

    data["special_start"] = data["grid"].eq(0).astype(int)
    data["grid"] = data["grid"].replace(0, float("nan"))

    return data.sort_values(
        ["year", "round", "driverCode"]
    ).reset_index(drop=True)


def create_model():
    """Maak een nieuw, nog niet getraind model."""
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
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


def evaluate_races(model, frame, label):
    """Voorspel, rangschik per race en vergelijk met de uitslag."""
    if frame.empty:
        raise ValueError(f"Geen racegegevens voor {label}.")

    results = frame[
        [
            "year",
            "round",
            "race",
            "driverCode",
            "original_grid",
            TARGET,
        ]
    ].copy()

    results.rename(
        columns={"original_grid": "grid"},
        inplace=True,
    )

    # Het model krijgt uitsluitend FEATURES als invoer.
    results["predicted_score"] = model.predict(frame[FEATURES])

    # Lagere score = betere voorspelde finishpositie.
    results = results.sort_values(
        ["year", "round", "predicted_score", "driverCode"]
    )

    results["predicted_position"] = (
        results.groupby(["year", "round"]).cumcount() + 1
    )

    # Baseline: iedereen finisht in de startvolgorde.
    # Grid 0 en ontbrekende gridwaarden komen achteraan.
    baseline = results.copy()
    baseline["baseline_grid"] = baseline["grid"].replace(
        0,
        float("nan"),
    )

    baseline = baseline.sort_values(
        ["year", "round", "baseline_grid", "driverCode"],
        na_position="last",
    )

    baseline["baseline_position"] = (
        baseline.groupby(["year", "round"]).cumcount() + 1
    )

    results = results.merge(
        baseline[
            ["year", "round", "driverCode", "baseline_position"]
        ],
        on=["year", "round", "driverCode"],
        how="left",
        validate="one_to_one",
    )

    results["signed_error"] = (
        results["predicted_position"] - results[TARGET]
    )
    results["error"] = results["signed_error"].abs()

    model_mae = mean_absolute_error(
        results[TARGET],
        results["predicted_position"],
    )

    baseline_mae = mean_absolute_error(
        results[TARGET],
        results["baseline_position"],
    )

    print(f"\n{label}")
    print(f"Model: gemiddeld {model_mae:.2f} plaatsen afwijking")
    print(
        f"Startvolgorde: gemiddeld "
        f"{baseline_mae:.2f} plaatsen afwijking"
    )

    results = results.sort_values(
        ["year", "round", "predicted_position"]
    ).reset_index(drop=True)

    output_folder = DATA_DIR / "predictions"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"predictions_{label}.csv"
    results.to_csv(output_file, index=False)

    print(f"Opgeslagen: {output_file}")

    return results