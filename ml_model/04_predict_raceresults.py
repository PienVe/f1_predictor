from model_utils import (
    FEATURES,
    TARGET,
    create_model,
    evaluate_races,
    load_data,
)


def main():
    data = load_data()

    try:
        year = int(input("Welk seizoen? Bijvoorbeeld 2026: "))
    except ValueError:
        raise SystemExit("Vul een geldig jaartal in.")

    available_races = (
        data.loc[
            data["year"] == year,
            ["round", "race"],
        ]
        .drop_duplicates()
        .sort_values("round")
    )

    if available_races.empty:
        raise SystemExit(
            f"Geen racegegevens voor {year}. "
            "Verzamel die eerst en voer "
            "02_prepare_dataset.py opnieuw uit."
        )

    print(f"\nBeschikbare races in {year}:")
    print(available_races.to_string(index=False))

    try:
        round_number = int(input("\nWelke ronde? "))
    except ValueError:
        raise SystemExit("Vul een geldig rondenummer in.")

    # Alle beschikbare races vóór de gekozen race.
    before_target = (
        (data["year"] < year)
        | (
            (data["year"] == year)
            & (data["round"] < round_number)
        )
    )

    train = data.loc[before_target].copy()

    test = data.loc[
        (data["year"] == year)
        & (data["round"] == round_number)
    ].copy()

    if test.empty:
        raise SystemExit("Deze ronde staat niet in de dataset.")

    if train.empty:
        raise SystemExit(
            "Er zijn geen eerdere races om op te trainen."
        )

    race_name = test["race"].iloc[0]

    print(
        f"\nHistorische voorspelling: {race_name} {year}"
        f"\nTraining: {len(train)} coureurresultaten "
        "uit eerdere races.",
        flush=True,
    )

    model = create_model()
    model.fit(train[FEATURES], train[TARGET])

    predictions = evaluate_races(
        model,
        test,
        f"backtest_{year}_round_{round_number:02d}",
    )

    print("\nVoorspelde en werkelijke eindvolgorde:")
    print(
        predictions[
            [
                "driverCode",
                "grid",
                "predicted_position",
                "position",
                "error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()