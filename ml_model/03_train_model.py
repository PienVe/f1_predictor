from model_utils import (
    FEATURES,
    TARGET,
    create_model,
    evaluate_races,
    load_data,
)


def main():
    data = load_data()

    train = data[data["year"].between(2022, 2023)].copy()
    validation = data[data["year"] == 2024].copy()
    test = data[data["year"] == 2025].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Er ontbreken gegevens voor training "
            "(2022–2023), validatie (2024) of test (2025)."
        )

    model = create_model()

    print(
        f"Trainen op {len(train)} coureur-raceresultaten...",
        flush=True,
    )

    model.fit(train[FEATURES], train[TARGET])

    validation_results = evaluate_races(
        model,
        validation,
        "validation_2024",
    )

    # Gebruik deze uitkomst voor de uiteindelijke beoordeling.
    test_results = evaluate_races(
        model,
        test,
        "test_2025",
    )

    print("\nAfwijking per race in 2024:")
    print(
        validation_results.groupby("race")["error"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    print("\nAfwijking per coureur in 2024:")
    print(
        validation_results.groupby("driverCode")
        .agg(
            races=("error", "size"),
            mean_error=("error", "mean"),
            mean_signed_error=("signed_error", "mean"),
        )
        .sort_values("mean_error", ascending=False)
        .round(2)
    )

    last_round = test_results["round"].max()

    print("\nVoorspelling voor de laatste race van 2025:")
    print(
        test_results.loc[
            test_results["round"] == last_round,
            [
                "driverCode",
                "grid",
                "predicted_position",
                "position",
                "error",
            ],
        ].to_string(index=False)
    )

    print("\nGebruikte kenmerken:", FEATURES)


if __name__ == "__main__":
    main()