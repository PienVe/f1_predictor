import pandas as pd

from model_utils import (
    DATA_DIR,
    FEATURES,
    TARGET,
    create_model,
    load_data,
)


def main():
    upcoming_file = DATA_DIR / "upcoming_race.csv"

    if not upcoming_file.exists():
        raise FileNotFoundError(
            f"Bestand niet gevonden: {upcoming_file}"
        )

    upcoming = pd.read_csv(upcoming_file)

    required = [
        "year",
        "round",
        "race",
        "driverCode",
        "constructorId",
        "grid",
    ]

    missing = set(required) - set(upcoming.columns)

    if missing:
        raise ValueError(f"Ontbrekende kolommen: {sorted(missing)}")

    if upcoming.empty or upcoming[required].isna().any().any():
        raise ValueError("Vul alle deelnemersgegevens in.")

    for column in ["year", "round", "grid"]:
        upcoming[column] = pd.to_numeric(
            upcoming[column],
            errors="raise",
        )

    if upcoming["driverCode"].duplicated().any():
        raise ValueError("Dubbele coureurs in upcoming_race.csv.")

    if len(upcoming[["year", "round", "race"]].drop_duplicates()) != 1:
        raise ValueError("Het bestand moet precies één race bevatten.")

    year = int(upcoming["year"].iloc[0])
    round_number = int(upcoming["round"].iloc[0])
    race_name = upcoming["race"].iloc[0]

    history = load_data()

    # Alleen races vóór de gekozen race.
    # Bij Singapore worden Singapore zelf en latere races uitgesloten.
    history = history.loc[
        (history["year"] < year)
        | (
            (history["year"] == year)
            & (history["round"] < round_number)
        )
    ].sort_values(
        ["year", "round", "driverCode"]
    ).copy()

    if history.empty:
        raise ValueError("Geen eerdere races beschikbaar.")

    # Gemiddelde finishpositie per team per historische race.
    team_history = (
        history.groupby(
            ["year", "round", "constructorId"],
            as_index=False,
        )
        .agg(team_finish=("position", "mean"))
        .sort_values(["year", "round"])
    )

    # Maak de historische kenmerken voor de komende race.
    for index, driver in upcoming.iterrows():
        driver_history = history.loc[
            history["driverCode"] == driver["driverCode"]
        ]

        # Inclusief de laatste afgeronde race.
        recent = driver_history.tail(5)

        upcoming.loc[index, "avg_finish_last_5"] = (
            recent["position"].mean()
        )
        upcoming.loc[index, "previous_races_used"] = len(recent)

        previous_year = driver_history.loc[
            (driver_history["year"] == year - 1)
            & (driver_history["race"] == race_name)
        ]

        upcoming.loc[index, "finish_same_race_previous_year"] = (
            previous_year["position"].iloc[0]
            if not previous_year.empty
            else float("nan")
        )

        recent_team = team_history.loc[
            team_history["constructorId"] == driver["constructorId"]
        ].tail(5)

        upcoming.loc[index, "team_avg_finish_last_5"] = (
            recent_team["team_finish"].mean()
        )

    # Bewaar de startpositie voor we pitlanestarts verwerken.
    upcoming["original_grid"] = upcoming["grid"]

    upcoming["special_start"] = (
        upcoming["grid"].eq(0).astype(int)
    )
    upcoming["grid"] = upcoming["grid"].replace(
        0,
        float("nan"),
    )

    missing_features = set(FEATURES) - set(upcoming.columns)

    if missing_features:
        raise ValueError(
            f"Ontbrekende modelkenmerken: {sorted(missing_features)}"
        )

    # Gebruik dezelfde modeldefinitie als in 03 en 04.
    model = create_model()

    print(
        f"Trainen op {len(history)} historische coureurresultaten...",
        flush=True,
    )
    model.fit(history[FEATURES], history[TARGET])

    upcoming["predicted_score"] = model.predict(
        upcoming[FEATURES]
    )

    upcoming = upcoming.sort_values(
        ["predicted_score", "driverCode"]
    ).reset_index(drop=True)

    upcoming["predicted_position"] = upcoming.index + 1

    print(f"\nVoorspelde eindvolgorde: {race_name} {year}")
    print(
        upcoming[
            [
                "driverCode",
                "original_grid",
                "predicted_position",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nVoorspelde winnaar: "
        f"{upcoming.iloc[0]['driverCode']}"
    )

    output_file = (
    DATA_DIR
    / "predictions"
    / f"prediction_{year}_{round_number:02d}.csv"
)
    upcoming.to_csv(output_file, index=False)

    print(f"Opgeslagen: {output_file}")


if __name__ == "__main__":
    main()