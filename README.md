# 🏎️ F1 2026 Forecast — Powered by TimeCopilot

An automated Formula 1 prediction pipeline that uses [TimeCopilot](https://timecopilot.dev/) — a GenAI forecasting agent combining LLMs with time series foundation models — to predict race winners and World Championship standings throughout the 2026 F1 season.

Runs automatically via GitHub Actions for every session of all 24 races. Predictions are published to a [GitHub Pages site](https://sebhgtz.github.io/timecopilot-f1-2026-forecast/) after each session.

---

## What it predicts

- **Race winner** — updated after FP1, FP2, FP3, Qualifying (and Sprint sessions)
- **Drivers' World Championship** — predicted final standings after each race
- **Constructors' Championship** — predicted final standings after each race

---

## Backtest results (2025 season)

Validated on 4 races from the 2025 season, forecasting from post-qualifying data:

| Race | Predicted winner | Actual winner | Correct |
|---|---|---|---|
| 🇧🇪 Belgian GP (Rd 13) | NOR | PIA | ✗ (NOR P2) |
| 🇦🇿 Azerbaijan GP (Rd 17) | VER | VER | ✓ |
| 🇲🇨 Monaco GP (Rd 8) | NOR | NOR | ✓ |
| 🇦🇪 Abu Dhabi GP (Rd 24) | VER | VER | ✓ |

**3 / 4 race winners correct · 4 / 4 WDC champion calls correct**

---

## How it works

```
FastF1 (historical sessions)  ─┐
Jolpica-F1 API (standings)     ├─▶  Time series features  ─▶  TimeCopilot  ─▶  Predictions
OpenF1 API (strategy / live)  ─┘                               (LLM + TSFMs)
Open-Meteo (weather)
```

For each race weekend, the pipeline:

1. **Collects data** — historical lap times, circuit performance, championship standings, pit strategy, weather
2. **Builds time series** — per-driver championship points series (2015–present), circuit finish position series (2018–present), session pace series
3. **Forecasts** — TimeCopilot runs statistical models (Prophet, N-HiTS, N-BEATS, AutoETS) then uses GPT to select the best model and generate a narrative
4. **Publishes** — Twitter card (≤280 chars), LinkedIn post, Plotly charts, GitHub Pages site, email notification

---

## Pipeline architecture

```
f1_pipeline/
├── collectors/
│   ├── historical_collector.py    # FastF1: 2018–2025 sessions
│   ├── jolpica_collector.py       # Championship standings
│   ├── openf1_collector.py        # Pit stops, stints, race control
│   ├── race_weekend_collector.py  # Live session data aggregator
│   ├── weather_fetcher.py         # Open-Meteo forecasts
│   └── calendar_manager.py        # 2026 race calendar
├── features/
│   ├── championship_series.py     # Driver/constructor points time series
│   ├── circuit_series.py          # Per-driver circuit history
│   └── session_features.py        # FP/Qualifying pace features
├── forecasting/
│   ├── championship_forecaster.py # WDC/WCC prediction
│   ├── race_forecaster.py         # Race winner prediction
│   ├── race_weekend_updater.py    # Per-session pipeline update
│   └── orchestrator.py            # Coordinates all stages
└── reporting/
    └── report_generator.py        # Social cards, LinkedIn posts, charts
```

---

## Running locally

### Setup

```bash
pip install fastf1 pandas numpy requests plotly kaleido timecopilot openai
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

### Run the pipeline

```bash
# Auto-detect current race weekend and session
python run_f1_forecast.py --current

# Specific race, pre-weekend
python run_f1_forecast.py --race australia

# Specific race, after a session
python run_f1_forecast.py --race australia --session qualifying

# Check calendar against live FastF1 data
python run_f1_forecast.py --check-calendar
```

### Run a backtest

```bash
python backtest_monaco_2025.py
python backtest_azerbaijan_2025.py
python backtest_belgium_2025.py
python backtest_abudhabi_2025.py
```

---

## GitHub Actions automation

The pipeline runs automatically every day at 09:00 UTC. On non-race days it exits in seconds with no API calls. During a race weekend (Friday → Monday) it detects which sessions have completed and updates the forecast accordingly.

To trigger manually:

```bash
gh workflow run f1_predictions.yml -f race=bahrain -f session=qualifying
```

### Required secrets

| Secret | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used by TimeCopilot) |
| `GMAIL_USERNAME` | Gmail address for email notifications |
| `GMAIL_APP_PASSWORD` | 16-char Google App Password |
| `NOTIFY_EMAIL` | Recipient address for prediction emails |

---

## 2026 calendar

24 races · 6 sprint weekends (China, Miami, Canada, Great Britain, Netherlands, Singapore)

First race: **Australian GP — March 6–8, 2026**

---

## Data sources

- **[FastF1](https://github.com/theOehrly/Fast-F1)** — historical session data (2018–2025)
- **[Jolpica-F1 API](http://api.jolpi.ca/ergast/f1/)** — championship standings (Ergast successor)
- **[OpenF1 API](https://openf1.org/)** — live pit stops, stints, race control messages
- **[Open-Meteo](https://open-meteo.com/)** — circuit weather forecasts
- **[TimeCopilot](https://timecopilot.dev/)** — forecasting agent (LLMs × time series foundation models)

---

## Live predictions

**GitHub Pages:** [sebhgtz.github.io/timecopilot-f1-2026-forecast](https://sebhgtz.github.io/timecopilot-f1-2026-forecast/)

Reports are committed back to this repo after each session under `reports/`.

---

*Built by [@_sebhgtz](https://twitter.com/_sebhgtz) using [TimeCopilot](https://timecopilot.dev/)*
