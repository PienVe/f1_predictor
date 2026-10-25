import time
import unicodedata

import pandas as pd
import requests
from fastf1.ergast import Ergast

from model_utils import DATA_DIR


BASE_URL = "https://api.openf1.org/v1"


def get_openf1(endpoint, *, allow_not_found=False, **params):
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=params,
        timeout=30,
    )

    # Alleen bij een expliciet optionele aanvraag doorgaan bij 404
    if response.status_code == 404 and allow_not_found:
        print(
            f"OpenF1: {endpoint} niet beschikbaar; "
            "alternatieve gegevens proberen.",
            flush=True,
        )
        time.sleep(2)
        return pd.DataFrame()

    if response.status_code == 429:
        raise SystemExit(
            "OpenF1 meldt te veel aanvragen. Probeer het later opnieuw."
        )

    if response.status_code in (401, 403):
        raise SystemExit(
            "OpenF1 weigert toegang. Gegevens binnen het livevenster "
            "kunnen authenticatie vereisen."
        )

    response.raise_for_status()
    time.sleep(2)

    return pd.DataFrame(response.json())


def normalize_name(value):
    """Vergelijk teamnamen zonder hoofdletters, spaties of accenten."""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(
        character.lower()
        for character in text
        if character.isalnum()
    )


def load_team_mapping():
    """
    Gebruik constructorId's uit je eigen trainingsgegevens.
    Zo houden training en voorspelling dezelfde teamidentiteiten.
    """
    history_path = DATA_DIR / "race_results_with_features.csv"
    history = pd.read_csv(history_path)

    required = {"constructorId", "constructorName"}

    if not required.issubset(history.columns):
        raise ValueError(
            "Het historische bestand moet constructorId "
            "en constructorName bevatten."
        )

    teams = history[
        ["constructorId", "constructorName"]
    ].dropna().drop_duplicates()

    mapping = {}

    # Herken zowel de historische teamnaam als de ID zelf
    for row in teams.itertuples(index=False):
        mapping[normalize_name(row.constructorName)] = row.constructorId
        mapping[normalize_name(row.constructorId)] = row.constructorId

    known_ids = set(teams["constructorId"])

    # Naamvarianten tussen OpenF1 en Jolpica.
    # Gebruik een alias alleen als die ID in jouw historie bestaat.
    aliases = {
        "Red Bull Racing": "red_bull",
        "Racing Bulls": "rb",
        "RB": "rb",
        "RB F1 Team": "rb",
        "Alpine": "alpine",
        "Alpine F1 Team": "alpine",
        "Aston Martin": "aston_martin",
        "Haas F1 Team": "haas",
        "Kick Sauber": "sauber",
        "Sauber": "sauber",
        "Audi": "audi",
        "Cadillac": "cadillac",
    }

    for name, constructor_id in aliases.items():
        if constructor_id in known_ids:
            # Een directe koppeling uit jouw historie heeft voorrang
            mapping.setdefault(
                normalize_name(name),
                constructor_id,
            )

    return mapping


def main():
    try:
        year = int(input("Welk seizoen? Bijvoorbeeld 2026: "))
    except ValueError:
        raise SystemExit("Vul een geldig jaartal in.")

    # Behoud je bestaande kalender en rondenummers
    ergast = Ergast()

    print("Racekalender ophalen...", flush=True)
    schedule = ergast.get_race_schedule(season=year)

    if schedule.empty:
        raise SystemExit(f"Geen racekalender gevonden voor {year}.")

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

    selected_race = selected.iloc[0]
    race_name = selected_race["raceName"]

    # Zoek de OpenF1-racesessie op dezelfde kalenderdatum.
    # Zo verwarren we Austin niet met Miami of Las Vegas.
    race_date = pd.to_datetime(
        selected_race["raceDate"],
        utc=True,
    ).date()

    print(
        f"\nOpenF1-sessie zoeken voor {race_name}, {race_date}...",
        flush=True,
    )

    race_sessions = get_openf1(
        "sessions",
        year=year,
        session_name="Race",
    )

    if race_sessions.empty:
        raise SystemExit(
            f"OpenF1 heeft geen racesessies gevonden voor {year}."
        )

    session_dates = pd.to_datetime(
        race_sessions["date_start"],
        utc=True,
    ).dt.date

    matching_sessions = race_sessions.loc[
        session_dates == race_date
    ]

    if len(matching_sessions) != 1:
        raise SystemExit(
            "De kalenderdatum kon niet aan precies één OpenF1-race "
            "worden gekoppeld. Het CSV-bestand is niet gewijzigd."
        )

    race_session = matching_sessions.iloc[0]
    race_session_key = int(race_session["session_key"])
    meeting_key = int(race_session["meeting_key"])

    print(
        f"Gevonden: {race_session['location']} "
        f"(session_key={race_session_key})",
        flush=True,
    )

    # Eerst de gepubliceerde startopstelling proberen
    grid = get_openf1(
    "starting_grid",
    session_key=race_session_key,
    allow_not_found=True,
)

    if not grid.empty:
        source = "gepubliceerde startopstelling"
        driver_session_key = race_session_key

    else:
        print(
            "Nog geen startopstelling. Kwalificatieuitslag ophalen...",
            flush=True,
        )

        qualifying_sessions = get_openf1(
            "sessions",
            meeting_key=meeting_key,
            session_name="Qualifying",
        )

        if len(qualifying_sessions) != 1:
            raise SystemExit(
                "Geen unieke gewone kwalificatiesessie gevonden. "
                "Het CSV-bestand is niet gewijzigd."
            )

        qualifying_key = int(
            qualifying_sessions.iloc[0]["session_key"]
        )

        grid = get_openf1(
            "session_result",
            session_key=qualifying_key,
        )

        if grid.empty:
            raise SystemExit(
                "OpenF1 heeft nog geen startopstelling of "
                "kwalificatieuitslag voor deze race. "
                "Het CSV-bestand is niet gewijzigd."
            )

        source = "kwalificatiepositie, zonder latere gridstraffen"
        driver_session_key = qualifying_key

    # Coureurcodes en teamnamen ophalen
    drivers = get_openf1(
        "drivers",
        session_key=driver_session_key,
    )

    # De deelnemerslijst van een toekomstige racesessie kan nog ontbreken.
    # Gebruik dan de gewone kwalificatiesessie van hetzelfde weekend.
    if drivers.empty and driver_session_key == race_session_key:
        qualifying_sessions = get_openf1(
            "sessions",
            meeting_key=meeting_key,
            session_name="Qualifying",
        )

        if len(qualifying_sessions) == 1:
            drivers = get_openf1(
                "drivers",
                session_key=int(
                    qualifying_sessions.iloc[0]["session_key"]
                ),
            )

    if drivers.empty:
        raise SystemExit(
            "Geen deelnemersgegevens ontvangen. "
            "Het CSV-bestand is niet gewijzigd."
        )

    upcoming = grid[
        ["driver_number", "position"]
    ].merge(
        drivers[
            ["driver_number", "name_acronym", "team_name"]
        ],
        on="driver_number",
        how="left",
        validate="one_to_one",
    )

    upcoming = upcoming.rename(
        columns={
            "position": "grid",
            "name_acronym": "driverCode",
        }
    )

    # Vertaal de huidige teamnaam naar jouw historische constructorId
    team_mapping = load_team_mapping()

    upcoming["constructorId"] = (
        upcoming["team_name"]
        .map(normalize_name)
        .map(team_mapping)
    )

    missing_teams = upcoming.loc[
        upcoming["constructorId"].isna(),
        "team_name",
    ].unique()

    if len(missing_teams):
        raise SystemExit(
            "Deze OpenF1-teamnamen konden niet aan je historische "
            f"constructorId's worden gekoppeld: {list(missing_teams)}.\n"
            "Voeg hun juiste naamkoppeling toe in aliases. "
            "Het CSV-bestand is niet gewijzigd."
        )

    upcoming["grid"] = pd.to_numeric(
        upcoming["grid"],
        errors="raise",
    )

    if (
        upcoming[["driverCode", "grid"]].isna().any().any()
        or upcoming["driverCode"].fillna("").str.strip().eq("").any()
        or upcoming["driverCode"].duplicated().any()
        or upcoming["grid"].duplicated().any()
        or upcoming["grid"].le(0).any()
        or upcoming["grid"].mod(1).ne(0).any()
    ):
        raise ValueError(
            "De ontvangen opstelling bevat ontbrekende of "
            "ongeldige gegevens. Het CSV-bestand is niet gewijzigd."
        )

    upcoming["grid"] = upcoming["grid"].astype(int)
    upcoming["year"] = year
    upcoming["round"] = round_number
    upcoming["race"] = race_name.removesuffix(" Grand Prix")

    upcoming = (
        upcoming[
            [
                "year",
                "round",
                "race",
                "driverCode",
                "constructorId",
                "grid",
            ]
        ]
        .sort_values(["grid", "driverCode"])
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "upcoming_race.csv"
    print("\nOpgehaalde startopstelling:")
    print(
        upcoming[["driverCode", "grid"]]
        .sort_values("grid")
        .to_string(index=False)
    )

    answer = input(
        "\nWil je de startopstelling handmatig aanpassen? (j/n): "
    ).strip().lower()

    if answer in {"j", "ja"}:
        print(
            "\nVoer ALLE coureurcodes in de definitieve startvolgorde in."
            "\nBijvoorbeeld: VER, NOR, LEC, ..."
            "\nNeem coureurs die uit de pitstraat starten hier niet in op."
        )

        grid_input = input("Startvolgorde: ")
        grid_codes = [
            code.strip().upper()
            for code in grid_input.split(",")
            if code.strip()
        ]

        pit_input = input(
            "Pitstraatstarters, gescheiden door komma's "
            "(Enter als er geen zijn): "
        )

        pit_codes = [
            code.strip().upper()
            for code in pit_input.split(",")
            if code.strip()
        ]

        entered_codes = grid_codes + pit_codes
        expected_codes = set(upcoming["driverCode"])

        if len(entered_codes) != len(set(entered_codes)):
            raise ValueError(
                "Een coureur is meerdere keren ingevoerd. "
                "Het CSV-bestand is niet gewijzigd."
            )

        unknown = set(entered_codes) - expected_codes
        missing = expected_codes - set(entered_codes)

        if unknown or missing:
            raise ValueError(
                f"Onbekende codes: {sorted(unknown)}. "
                f"Ontbrekende codes: {sorted(missing)}. "
                "Het CSV-bestand is niet gewijzigd."
            )

        # Gewone startposities: 1, 2, 3, ...
        positions = {
            code: position
            for position, code in enumerate(grid_codes, start=1)
        }

        # Pitstraatstart: dezelfde conventie als grid=0 in je racegegevens
        positions.update({
            code: 0
            for code in pit_codes
        })

        upcoming["grid"] = (
            upcoming["driverCode"]
            .map(positions)
            .astype(int)
        )

    # Pitstraatstarters onderaan weergeven
    upcoming = (
        upcoming.assign(
            sort_grid=upcoming["grid"].replace(0, float("inf"))
        )
        .sort_values(["sort_grid", "driverCode"])
        .drop(columns="sort_grid")
        .reset_index(drop=True)
    )

    upcoming.to_csv(output, index=False)

    print(f"\nGebruikte bron: {source}")
    print(upcoming.to_string(index=False))
    print(f"\nOpgeslagen: {output}")


if __name__ == "__main__":
    main()