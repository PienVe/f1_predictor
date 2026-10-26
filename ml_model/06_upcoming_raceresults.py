import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

        upcoming.loc[index, "weighted_avg_finish_last_5"] = (
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

    #Namen na de verwerking, inclusief de losse categorieën
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": (
            model.named_steps["regressor"].feature_importances_
        ),
    })

    importances = importances.sort_values(
        "importance",
        ascending=False,
    )

    print("\nBelangrijkste kenmerken in het Random Forest:")
    print(importances.head(25).to_string(index=False))

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

    # Gebruik races waarop het model NIET is getraind
    importance = permutation_importance(
        model,
        history[FEATURES],
        history[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance.importances_mean,
        "variation": importance.importances_std,
    }).sort_values("importance", ascending=True)

    # Optioneel: duidelijkere namen in de figuur
    labels = {
        "grid": "Startpositie",
        "avg_finish_last_5": "Gem. finish vorige 5 races",
        "weighted_avg_finish_last_5": "Gewogen gem. finish vorige 5 races",
        "previous_races_used": "Aantal eerdere races gebruikt",
        "finish_same_race_previous_year": "Finish bij deze GP vorig jaar",
        "team_avg_finish_last_5": "Gem. teamfinish vorige 5 races",
        "special_start": "Bijzondere start / pitstraat",
        "driverCode": "Coureur",
        "constructorId": "Team",
        "race": "Grand Prix",
    }

    importance_df["label"] = (
        importance_df["feature"]
        .map(labels)
        .fillna(importance_df["feature"])
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        importance_df["label"],
        importance_df["importance"],
        color="#24699B",
    )

    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_title("Invloed van kenmerken op het F1-model", loc="left")
    ax.set_xlabel(
        "Toename in gemiddelde fout na door elkaar husselen\n"
        "(finishplaatsen; hoger = belangrijker)"
    )

    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output = DATA_DIR / "feature_importance.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")

    print(f"Figuur opgeslagen: {output}")

    plt.show()


if __name__ == "__main__":
    main()