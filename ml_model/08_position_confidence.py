"""Estimate uncertainty around an existing prediction using earlier race errors."""

import argparse
import html
import json
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd


def adapt_residuals(residuals, target_size):
    """Interpolate errors by relative predicted rank and scale to the target field."""
    if target_size < 2 or not len(residuals):
        raise ValueError("At least two drivers and historical error patterns are required.")
    adapted = []
    for pattern in residuals:
        pattern = np.asarray(pattern, dtype=float)
        if pattern.ndim != 1 or len(pattern) < 2 or not np.isfinite(pattern).all():
            raise ValueError("Each historical pattern must contain finite errors for at least two drivers.")
        source_rank = np.linspace(0, 1, len(pattern))
        target_rank = np.linspace(0, 1, target_size)
        # A one-place error in a 20-driver field is 1/19 of its finishing range.
        relative_error = pattern / (len(pattern) - 1)
        adapted.append(np.interp(target_rank, source_rank, relative_error) * (target_size - 1))
    return np.asarray(adapted)


def simulate(scores, residuals, simulations=10000, seed=42):
    """Resample historical races, including races with different field sizes."""
    scores = np.asarray(scores, dtype=float)
    residuals = adapt_residuals(residuals, len(scores))
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores or historical errors found.")
    rng = np.random.default_rng(seed)
    simulated_scores = scores + residuals[rng.integers(len(residuals), size=simulations)]
    simulated_scores += rng.uniform(-1e-8, 1e-8, simulated_scores.shape)
    order = np.argsort(simulated_scores, axis=1)
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(1, len(scores) + 1)[None, :], axis=1)
    return positions


def summarize(simulations, predicted_positions):
    """Keep the user's original predicted positions, rather than replacing them."""
    predicted_positions = np.asarray(predicted_positions)
    n = simulations.shape[1]
    if sorted(predicted_positions.tolist()) != list(range(1, n + 1)):
        raise ValueError("Predicted positions must be unique, from 1 to the number of drivers.")
    probabilities = np.column_stack([
        (simulations == place).mean(axis=0) for place in range(1, n + 1)
    ])
    low, high = np.quantile(simulations, [0.1, 0.9], axis=0, method="inverted_cdf")
    exact = probabilities[np.arange(n), predicted_positions.astype(int) - 1]
    return probabilities, exact, low.astype(int), high.astype(int)


def select_prediction(folder, features=()):
    # Do not select raw upcoming grids or season-wide evaluation CSVs.
    files = []
    for path in sorted(folder.glob("prediction*.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        required = set(features) | {"year", "round", "race", "predicted_position", "predicted_score", "driverCode"}
        if not required.issubset(frame.columns) or len(frame) < 2:
            continue
        if len(frame[["year", "round", "race"]].drop_duplicates()) != 1:
            continue
        if frame["driverCode"].isna().any() or frame["driverCode"].duplicated().any():
            continue
        ranked = frame.sort_values(["predicted_score", "driverCode"])
        if ranked["predicted_position"].tolist() != list(range(1, len(frame) + 1)):
            continue
        files.append(path)
    if not files:
        raise ValueError(f"No saved predictions in {folder}. Run script 06 first.")
    for number, path in enumerate(files, 1):
        print(f"{number}: {path.name}")
    print("Only complete single-race predictions are shown; evaluation files have been skipped.")
    while True:
        try:
            answer = input("\nWhich prediction would you like to view? (q = quit) ").strip()
        except EOFError:
            raise SystemExit("No selection received. You can use --prediction FILE instead.") from None
        if answer.lower() == "q":
            raise SystemExit("Stopped.")
        try:
            choice = int(answer)
        except ValueError:
            print(f"Enter a number from 1 to {len(files)}.")
            continue
        if 1 <= choice <= len(files):
            return files[choice - 1]
        print(f"Invalid selection. Choose 1 to {len(files)}.")


def write_report(result, probabilities, validation, title, path):
    """A standalone, accessible report with one primary chart and optional detail."""
    order = np.argsort(result["predicted_position"].to_numpy(), kind="stable")
    result = result.iloc[order].reset_index(drop=True)
    probabilities = np.asarray(probabilities)[order]
    n = len(result)
    esc = lambda value: html.escape(str(value), quote=True)

    def share_label(value):
        if value == 0:
            return "Not observed in simulations"
        if value < 0.1:
            return "&lt;0.1% of simulations"
        return f"{value:.1f}% of simulations"

    width, left, right, top, step = 1120, 140, 950, 82, 39
    height = top + n * step + 45
    def x(position):
        return left + (position - 1) * (right - left) / max(n - 1, 1)
    chart = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-description">',
        '<title id="chart-title">Predicted finishing positions and simulated finishing ranges</title>',
        '<desc id="chart-description">One row per driver. A blue dot marks the original model prediction. '
        'A grey bar shows the central 80 percent of simulated finishing positions. '
        'The same information is available in the table below.</desc>',
        f'<text x="{left}" y="19" class="axis-label">FRONT OF FIELD</text>',
        f'<text x="{right}" y="19" text-anchor="end" class="axis-label">BACK OF FIELD</text>',
        '<text x="20" y="47" class="axis-label">DRIVER</text>',
        '<text x="990" y="47" class="axis-label">SIMULATED RANGE</text>',
    ]
    for place in range(1, n + 1):
        chart.append(f'<line x1="{x(place)}" x2="{x(place)}" y1="61" y2="{height-28}" stroke="#e2e8f0"/>')
        chart.append(f'<text x="{x(place)}" y="47" text-anchor="middle" class="tick">P{place}</text>')
    table_rows, heat_rows = [], []
    for i, row in enumerate(result.itertuples()):
        driver = esc(row.driverCode)
        pos, low, high = int(row.predicted_position), int(row.position_low_80), int(row.position_high_80)
        y = top + i * step
        outside = not low <= pos <= high
        if i % 2 == 0:
            chart.append(f'<rect x="0" y="{y-16}" width="{width}" height="32" fill="#f8fafc"/>')
        chart.append(f'<text x="20" y="{y+5}" class="driver">{driver}</text>')
        chart.append(f'<line x1="{x(low)}" x2="{x(high)}" y1="{y}" y2="{y}" stroke="#94a3b8" stroke-width="9" stroke-linecap="round"><title>{driver}: P{low}–P{high}, central 80% of simulated finishes</title></line>')
        chart.append(f'<circle cx="{x(pos)}" cy="{y}" r="7" fill="#2563eb" stroke="white" stroke-width="2"><title>{driver}: model prediction P{pos}</title></circle>')
        chart.append(f'<text x="990" y="{y+5}" class="range-label">P{low}–P{high}{" *" if outside else ""}</text>')
        exact = share_label(row.exact_position_pct)
        near = share_label(row.within_2_positions_pct)
        table_rows.append(
            f'<tr><th scope="row">{driver}</th><td><strong>P{pos}</strong></td>'
            f'<td>P{low}–P{high}{" *" if outside else ""}</td><td>{exact}</td><td>{near}</td></tr>'
        )
        cells = []
        for place, probability in enumerate(probabilities[i], 1):
            pct = probability * 100
            label = "—" if probability == 0 else ("&lt;0.1%" if pct < .1 else f"{pct:.1f}%")
            tooltip = (f"{row.driverCode}, P{place}: not observed in these simulations; this does not mean impossible."
                       if probability == 0 else f"{row.driverCode}, P{place}: {pct:.1f}% of simulations")
            cells.append(f'<td title="{esc(tooltip)}" style="background:rgba(37,99,235,{probability:.3f});color:{"white" if probability > .55 else "#172033"}">{label}</td>')
        heat_rows.append(f'<tr><th scope="row">{driver}</th>{"".join(cells)}</tr>')
    chart.append('</svg>')
    coverage = validation["interval_coverage"]
    check = (
        f"Across {validation['checked_races']} later historical races, {coverage:.1%} of actual finishes "
        "fell within the simulated 80% ranges. This is a limited diagnostic, not proof of reliable probabilities."
        if coverage is not None else
        "There are not enough later historical races to check the ranges yet."
    )
    bins = pd.DataFrame(validation["exact_probability_bins"]).rename(columns={
        "probability_bin": "Estimated share band", "count": "Driver predictions checked",
        "mean_estimated_pct": "Average estimated share (%)", "observed_exact_pct": "Actual exact-position hits (%)",
    })
    bin_table = bins.to_html(index=False, float_format=lambda value: f"{value:.1f}") if not bins.empty else '<p>No historical checks available yet.</p>'
    sample = validation["residual_races"]
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#172033;background:#f3f6fa}}
body{{max-width:1260px;margin:auto;padding:32px 20px 60px}}h1{{font-size:30px;margin-bottom:8px}}
h2{{font-size:22px;margin-top:0}}p{{line-height:1.6}}.subtle{{color:#526078}}.card{{background:white;border:1px solid #dce3ed;border-radius:14px;padding:24px;margin:20px 0}}
.notice{{border-left:4px solid #d99b19;background:#fff8e8;padding:14px 18px;border-radius:6px}}
.stats{{display:flex;gap:16px;flex-wrap:wrap;margin-top:20px}}.stat{{background:white;border:1px solid #dce3ed;border-radius:10px;padding:14px 20px;min-width:150px}}.stat strong{{display:block;font-size:23px}}
.legend{{display:flex;flex-wrap:wrap;gap:22px;align-items:center;margin:16px 0;font-size:14px}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;background:#2563eb;margin-right:8px}}.bar{{display:inline-block;width:30px;height:8px;background:#94a3b8;border-radius:5px;margin-right:8px}}
.scroll{{overflow-x:auto}}svg{{display:block;width:100%;min-width:960px}}svg text{{font-family:system-ui,sans-serif;fill:#334155}}.axis-label{{font-size:11px;font-weight:700;letter-spacing:.5px}}.tick{{font-size:11px}}.driver{{font-size:14px;font-weight:700}}.range-label{{font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e2e8f0}}thead th{{background:#f8fafc}}.heatmap td{{text-align:center;min-width:60px;padding:10px 6px;font-size:12px}}.heatmap th{{background:white;position:sticky;left:0}}summary{{cursor:pointer;font-size:19px;font-weight:650;padding:4px 0}}.small{{font-size:13px}}details p{{max-width:1000px}}
</style></head><body>
<h1>{esc(title)}</h1><p class="subtle">Where each driver is predicted to finish — and how much the simulated finishes vary.</p>
<div class="stats"><div class="stat"><strong>{n}</strong>drivers</div><div class="stat"><strong>{sample}</strong>historical races sampled</div><div class="stat"><strong>{validation['simulations']:,}</strong>simulated race finishes</div></div>
<p class="notice"><strong>Exploratory uncertainty estimate.</strong> These are simulated outcomes, not validated real-world probabilities. Repeating {sample} historical race patterns thousands of times does not create more historical evidence.</p>
<section class="card"><h2>1. Start here: predicted position and finishing range</h2>
<p>Read one row at a time. Positions further left are better. <strong>A wider grey bar means more variation in the simulations.</strong></p>
<div class="legend"><span><span class="dot"></span>Blue dot = original predicted position</span><span><span class="bar"></span>Grey bar = middle 80% of simulated finishes</span></div>
<div class="scroll">{''.join(chart)}</div>
<p class="small subtle">Example: a dot at P4 and a bar from P2 to P8 means “predicted P4, with simulated finishes commonly between P2 and P8”. It does not mean an 80% chance of P4. Hover over a dot or bar for its values.</p>
<p class="small subtle">* The original prediction is outside the simulated range. The historical error patterns suggest a shift from that prediction; this is not a missing driver or missing result.</p></section>
<section class="card"><h2>2. Numbers for each driver</h2><p>“Exact position” means the blue-dot position. “Within 2 places” includes two places ahead and two behind it.</p>
<p class="notice"><strong>Not observed in simulations ≠ impossible.</strong> It means none of the sampled historical error patterns produced that outcome. Small nonzero percentages may also be unstable.</p>
<div class="scroll"><table><thead><tr><th>Driver</th><th>Model prediction</th><th>Middle 80% of simulated finishes</th><th>Simulations matching the exact prediction</th><th>Simulations within 2 places of prediction</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<details class="card"><summary>3. Optional: all simulated finishing positions</summary><p>Each row is one driver; each column is a finishing position. Darker cells mean that outcome appeared more often. “—” means not observed, not impossible. Each row adds to 100% before rounding.</p>
<div class="scroll"><table class="heatmap"><thead><tr><th>Driver</th>{''.join(f'<th>P{i}</th>' for i in range(1,n+1))}</tr></thead><tbody>{''.join(heat_rows)}</tbody></table></div></details>
<details class="card"><summary>4. How much should I trust this?</summary><p>{esc(check)}</p><p>{esc(validation.get('sample_note',''))}</p>
<h3>Exact-position check on historical predictions</h3><p>Compare the estimated simulation share with how often those historical predictions actually matched the finish. The columns should be similar if the estimates are reliable, but small samples and dependence between drivers limit this check.</p><div class="scroll">{bin_table}</div>
<h3>Method and limitations</h3><p>The original model positions are preserved. Each simulation adds an error pattern from one earlier race to the model scores, then sorts all drivers into a unique finishing order. Errors are matched by relative predicted rank, not driver identity. Races with different field sizes are included: errors are divided by the source field size minus one, interpolated across relative ranks, then scaled to the target field size minus one. This is an approximation, not an observation of extra drivers. Historical models and interval checks use only earlier data.</p><p>The middle 80% range uses the 10th and 90th percentiles. With discrete finishing positions it may contain more than 80% of simulated outcomes. New regulations, weather, crashes and team changes can make earlier patterns unrepresentative. Exact-position percentages have not been separately calibrated.</p></details>
</body></html>'''
    path.write_text(page, encoding="utf-8")


def main():
    from model_utils import DATA_DIR, FEATURES, TARGET, create_model, load_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--races", type=int, default=30, help="Maximum number of historical race error vectors")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.races < 2 or args.simulations < 100:
        parser.error("Use at least 2 historical races and 100 simulations.")
    path = args.prediction or select_prediction(DATA_DIR / "predictions", FEATURES)
    upcoming = pd.read_csv(path)
    required = set(FEATURES) | {"year", "round", "race", "driverCode", "predicted_score", "predicted_position"}
    missing = required - set(upcoming.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Save the complete upcoming table in script 06.")
    for col in ["year", "round", "predicted_score", "predicted_position"]:
        upcoming[col] = pd.to_numeric(upcoming[col], errors="raise")
    if upcoming.empty or len(upcoming[["year", "round", "race"]].drop_duplicates()) != 1:
        raise ValueError("Choose a prediction for exactly one race.")
    if upcoming["driverCode"].isna().any() or upcoming["driverCode"].duplicated().any():
        raise ValueError("Missing or duplicate drivers.")
    upcoming = upcoming.sort_values(["predicted_score", "driverCode"]).reset_index(drop=True)
    n = len(upcoming)
    if n < 2 or upcoming["predicted_position"].tolist() != list(range(1, n + 1)):
        raise ValueError("The file must contain a complete finishing order based on predicted_score.")
    year, rnd = int(upcoming.iloc[0]["year"]), int(upcoming.iloc[0]["round"])
    history = load_data()
    history = history[(history["year"] < year) | ((history["year"] == year) & (history["round"] < rnd))]
    history = history.sort_values(["year", "round", "driverCode"])
    if history.empty:
        raise ValueError("No historical data before this race.")

    # Ensure we are analysing the actual existing prediction with the same model/data.
    print("Checking that the model and historical dataset match this prediction...", flush=True)
    current = create_model()
    current.fit(history[FEATURES], history[TARGET])
    if not np.allclose(current.predict(upcoming[FEATURES]), upcoming["predicted_score"], atol=1e-7, rtol=1e-7):
        raise ValueError("The model or history has changed since this prediction. Recreate it using script 06 with the same history.")

    groups = list(history.groupby(["year", "round"], sort=True))
    # Include complete fields of any size, after 20 races available for training.
    candidates = [i for i in range(20, len(groups)) if len(groups[i][1]) >= 2][-args.races:]
    if len(candidates) < 2:
        raise ValueError(f"Only {len(candidates)} usable historical races. At least 2 are required after the first 20 training races. Choose a later race or add complete earlier races.")
    sample_note = (
        f"Limited sample: only {len(candidates)} historical races, including different field sizes. "
        "Percentages are exploratory and may change substantially as new races are added. "
        "Additional simulations do not increase the historical sample size."
        if len(candidates) < 20 else
        f"Estimate based on {len(candidates)} historical races, including different field sizes; probabilities have not been separately calibrated."
    )
    print(sample_note, flush=True)
    residuals, checks = [], []
    covered = []
    checked_races = 0
    for count, idx in enumerate(candidates, 1):
        key, frame = groups[idx]
        print(f"Historical prediction {count}/{len(candidates)}: {key}", flush=True)
        earlier = pd.concat([g[1] for g in groups[:idx]], ignore_index=True)
        model = create_model()
        model.fit(earlier[FEATURES], earlier[TARGET])
        frame = frame.copy()
        frame["score"] = model.predict(frame[FEATURES])
        frame = frame.sort_values(["score", "driverCode"])
        scores, actual = frame["score"].to_numpy(), frame[TARGET].to_numpy()
        source_size = len(frame)
        if sorted(actual.tolist()) != list(range(1, source_size + 1)):
            raise ValueError(f"Race {key} does not have complete results with unique positions from 1 to {source_size}.")
        if len(residuals) >= 5:
            sims = simulate(scores, residuals, args.simulations, seed=42 + count)
            _, exact, low, high = summarize(sims, np.arange(1, source_size + 1))
            covered.extend(((actual >= low) & (actual <= high)).tolist())
            checks.extend(zip(exact.tolist(), (actual == np.arange(1, source_size + 1)).astype(int).tolist()))
            checked_races += 1
        residuals.append(actual - scores)

    sims = simulate(upcoming["predicted_score"], residuals, args.simulations)
    probabilities, exact, low, high = summarize(sims, upcoming["predicted_position"])
    result = upcoming[["driverCode", "predicted_position"]].copy()
    result["exact_position_pct"] = exact * 100
    result["position_low_80"] = low
    result["position_high_80"] = high
    declared = upcoming["predicted_position"].to_numpy()
    result["within_2_positions_pct"] = (np.abs(sims - declared) <= 2).mean(axis=0) * 100
    result["year"], result["round"], result["race"] = year, rnd, upcoming.iloc[0]["race"]
    bins = []
    if checks:
        check = pd.DataFrame(checks, columns=["probability", "hit"])
        for left, right in zip([0, .1, .2, .3, .5], [.1, .2, .3, .5, 1.000001]):
            subset = check[(check.probability >= left) & (check.probability < right)]
            if not subset.empty:
                bins.append({"probability_bin": f"{left:.0%}–{min(right, 1):.0%}",
                    "count": len(subset), "mean_estimated_pct": float(subset.probability.mean() * 100),
                    "observed_exact_pct": float(subset.hit.mean() * 100)})
    validation = {"residual_races": len(residuals), "checked_races": checked_races,
        "simulations": args.simulations, "interval_coverage": float(np.mean(covered)) if covered else None,
        "exact_probability_bins": bins, "method": "Whole-race residual bootstrap, interpolated by relative rank and scaled by field size minus one",
        "historical_field_sizes": sorted({len(groups[i][1]) for i in candidates}),
        "source_prediction": str(path), "calibrated_probabilities": False,
        "sample_note": sample_note, "validation_min_prior_residual_races": 5}
    folder = DATA_DIR / "predictions" / "confidence"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"confidence_{year}_{rnd:02d}"
    result.to_csv(folder / f"{stem}.csv", index=False)
    probability_table = pd.DataFrame(probabilities, columns=[f"P{i}" for i in range(1, n + 1)])
    probability_table.insert(0, "driverCode", result["driverCode"])
    probability_table.to_csv(folder / f"{stem}_probabilities.csv", index=False)
    (folder / f"{stem}_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    report = folder / f"{stem}.html"
    write_report(result, probabilities, validation, f"Prediction uncertainty — {upcoming.iloc[0]['race']} {year}", report)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nCharts saved: {report}")
    print(f"Historical race error vectors: {len(residuals)}; later validation races: {checked_races}.")
    if not args.no_open:
        webbrowser.open(report.resolve().as_uri())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Unable to estimate uncertainty: {exc}") from None

