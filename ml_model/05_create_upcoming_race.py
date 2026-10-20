import time

from fastf1.ergast import Ergast

from model_utils import DATA_DIR


def main():
    try:
        year = int(input("Welk seizoen? Bijvoorbeeld 2026: "))
    except ValueError:
        raise SystemExit("Vul een geldig jaartal in.")

    ergast = Ergast()

    print("Racekalender ophalen...", flush=True)
    schedule = ergast.get_race_schedule(season=year)

    if schedule.empty:
        raise SystemExit(f"Geen racekalender gevonden voor {year}.")

    print(f"\nRaces in {year}:")
    print(
        schedule[["round", "raceName"]]
        .sort_values("round")
        .to_string(index=False)
    )

    try:
        round_number = int(input("\nWelke ronde wil je voorspellen? "))
    except ValueError:
        raise SystemExit("Vul een geldig rondenummer in.")

    selected = schedule.loc[
        schedule["round"] == round_number
    ]

    if selected.empty:
        raise SystemExit("Deze ronde staat niet in de kalender.")

    race_name = selected.iloc[0]["raceName"]

    print(
        f"\nStartposities ophalen voor {race_name} {year}...",
        flush=True,
    )

    time.sleep(2)

    response = ergast.get_race_results(
        season=year,
        round=round_number,
    )

    if not response.content:
        raise SystemExit(
            "Er zijn nog geen racegegevens beschikbaar.\n"
            "Deze importer werkt voor afgeronde races.\n"
            "Voor een toekomstige race moet je de startopstelling "
            "apart aanleveren. Een bestaand CSV-bestand is niet gewijzigd."
        )

    # Alleen deelnemers, teams en startposities bewaren.
    upcoming = response.content[0][
        ["driverCode", "constructorId", "grid"]
    ].copy()

    upcoming["year"] = year
    upcoming["round"] = round_number
    upcoming["race"] = race_name.removesuffix(" Grand Prix")

    upcoming = upcoming[
        [
            "year",
            "round",
            "race",
            "driverCode",
            "constructorId",
            "grid",
        ]
    ]

    if upcoming["driverCode"].duplicated().any():
        raise ValueError("Dubbele coureurs gevonden.")

    # Pitlanestarts onderaan tonen.
    upcoming = (
        upcoming.assign(
            sort_grid=upcoming["grid"].replace(0, float("inf"))
        )
        .sort_values(["sort_grid", "driverCode"])
        .drop(columns="sort_grid")
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "upcoming_race.csv"

    # Vervangt de invoer van een eventueel eerder gekozen race.
    upcoming.to_csv(output, index=False)

    print(upcoming.to_string(index=False))
    print(f"\nOpgeslagen: {output}")


if __name__ == "__main__":
    main()