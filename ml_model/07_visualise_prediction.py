from pathlib import Path

import pandas as pd
import plotly.graph_objects as go #type: ignore


PREDICTIONS_DIR = (
    Path(__file__).resolve().parent / "data" / "predictions"
)


def main():
    files = sorted(PREDICTIONS_DIR.glob("prediction_*.csv"))

    if not files:
        raise SystemExit(
            f"Geen prediction_*.csv bestanden gevonden in:\n"
            f"{PREDICTIONS_DIR}"
        )

    print("Beschikbare voorspellingen:\n")

    for number, file in enumerate(files, start=1):
        print(f"{number}: {file.name}")

    try:
        choice = int(input("\nWelke voorspelling wil je bekijken? "))
    except ValueError:
        raise SystemExit("Vul een nummer in.")

    if not 1 <= choice <= len(files):
        raise SystemExit("Ongeldige keuze.")

    selected_file = files[choice - 1]
    data = pd.read_csv(selected_file)

    required = [
        "driverCode",
        "original_grid",
        "predicted_position",
        "race",
        "year",
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"Ontbrekende kolommen: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Het gekozen bestand is leeg.")

    for column in ["original_grid", "predicted_position"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    data = data.sort_values("predicted_position")

    race_name = data["race"].iloc[0]
    year = int(data["year"].iloc[0])

    fig = go.Figure()

    # Verbind startpositie en voorspelde finishpositie.
    for row in data.itertuples():
        if pd.notna(row.original_grid) and row.original_grid > 0:
            fig.add_trace(
                go.Scatter(
                    x=[
                        row.original_grid,
                        row.predicted_position,
                    ],
                    y=[row.driverCode, row.driverCode],
                    mode="lines",
                    line=dict(color="lightgrey", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    grid_starters = data[data["original_grid"] > 0]

    fig.add_trace(
        go.Scatter(
            x=grid_starters["original_grid"],
            y=grid_starters["driverCode"],
            mode="markers",
            name="Startpositie",
            marker=dict(
                color="grey",
                size=10,
                symbol="diamond",
            ),
            hovertemplate=(
                "%{y}<br>Startpositie: %{x}<extra></extra>"
            ),
        )
    )

    def start_label(position):
        if pd.isna(position):
            return "Onbekend"
        if position == 0:
            return "Pitstraat"
        return str(int(position))

    data["start_label"] = data["original_grid"].apply(start_label)

    fig.add_trace(
        go.Scatter(
            x=data["predicted_position"],
            y=data["driverCode"],
            customdata=data[["start_label"]],
            mode="markers",
            name="Voorspelde finish",
            marker=dict(color="#2563eb", size=13),
            hovertemplate=(
                "%{y}<br>"
                "Voorspelde finish: %{x}<br>"
                "Startpositie: %{customdata[0]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Voorspelde eindvolgorde — {race_name} {year}",
        template="plotly_white",
        xaxis=dict(
            title="Positie",
            dtick=1,
            range=[0.5, len(data) + 0.5],
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=data["driverCode"].tolist(),
            autorange="reversed",
        ),
        height=max(600, len(data) * 30),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=70, r=30, t=110, b=60),
    )

    output_file = selected_file.with_suffix(".html")

    fig.write_html(
        output_file,
        include_plotlyjs=True,
        auto_open=True,
    )

    print(f"\nGrafiek opgeslagen: {output_file}")


if __name__ == "__main__":
    main()