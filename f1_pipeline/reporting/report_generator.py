"""
Report Generator
=================
Generates all output artifacts for public posting:

  1. Twitter/X social card (≤280 chars) — punchy race prediction + championship snippet
  2. LinkedIn post (markdown, 800–1200 words) — full narrative with TimeCopilot analysis
  3. Plotly charts (PNG 1200×675px):
       - Championship standings prediction + confidence bands
       - Race podium probability bar chart
       - Prediction evolution across the race weekend (FP1→Quali)

All outputs saved to reports/{race_slug}_{year}/

Usage:
    rg = ReportGenerator(race_slug="australia", year=2026)
    rg.generate_all(
        race_forecast=race_fc,
        driver_champ_forecast=champ_fc,
        constructor_champ_forecast=cons_fc,
        prediction_evolution=updater.prediction_evolution(),
        strategy_context=strategy_features,
        session_stage="qualifying",
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

REPORTS_DIR = Path("reports")


class ReportGenerator:
    """Generates social posts and charts for F1 predictions."""

    F1_FLAG = "🏎️"
    TROPHY = "🏆"
    CHECKERED = "🏁"

    def __init__(self, race_slug: str, year: int = 2026, race_name: str = ""):
        self.race_slug = race_slug
        self.year = year
        # race_name used as fallback when race_forecast is None (e.g. post_race)
        self.race_name = race_name or race_slug.replace("_", " ").title() + " Grand Prix"
        self.race_dir = REPORTS_DIR / f"{race_slug}_{year}"
        self.race_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.race_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)

    def generate_all(
        self,
        race_forecast,
        driver_champ_forecast,
        constructor_champ_forecast=None,
        prediction_evolution: Optional[pd.DataFrame] = None,
        strategy_context: Optional[dict] = None,
        session_stage: str = "pre_weekend",
    ) -> None:
        """Generate all report formats for a given session stage."""
        print(f"  📝 Generating reports for stage: {session_stage}")

        # 1. Social card (Twitter/X)
        twitter_text = self.generate_twitter_card(
            race_forecast, driver_champ_forecast, session_stage
        )
        self._write(f"social_card_{session_stage}.txt", twitter_text)
        print(f"     ✓ Twitter card ({len(twitter_text)} chars)")

        # 2. LinkedIn post
        linkedin_text = self.generate_linkedin_post(
            race_forecast, driver_champ_forecast, constructor_champ_forecast,
            prediction_evolution, strategy_context, session_stage
        )
        self._write(f"linkedin_post_{session_stage}.md", linkedin_text)
        print(f"     ✓ LinkedIn post ({len(linkedin_text)} chars)")

        # 3. Save top10 CSV (used by accuracy tracker)
        if race_forecast and not race_forecast.predicted_top10.empty:
            top10_path = self.race_dir / f"top10_{session_stage}.csv"
            race_forecast.predicted_top10.head(10).to_csv(top10_path, index=False)
            print(f"     ✓ Top-10 CSV saved")

        # 4. Charts
        self._generate_charts(
            race_forecast, driver_champ_forecast, prediction_evolution, session_stage
        )

    # ── Twitter / X social card ───────────────────────────────────────────────

    def generate_twitter_card(
        self,
        race_forecast,
        driver_champ_forecast,
        session_stage: str,
    ) -> str:
        """
        Generate ≤280 character Twitter/X post.
        Format: emoji + race prediction + championship snippet + hashtags
        """
        stage_labels = {
            "pre_weekend": "Pre-weekend",
            "fp1": "After FP1",
            "fp2": "After FP2",
            "fp3": "After FP3",
            "qualifying": "After Qualifying",
            "sprint_qualifying": "After Sprint Qualifying",
            "sprint": "After Sprint",
            "post_race": "Post-Race",
        }
        stage_label = stage_labels.get(session_stage, session_stage.upper())
        short_name = _shorten_race_name(self.race_name)

        # Championship-only card (post-race or when race forecast unavailable)
        if race_forecast is None:
            if driver_champ_forecast:
                champ = driver_champ_forecast.predicted_champion()
                champ_name = champ.get("name", "?")
                champ_pts = champ.get("predicted_points", 0)
                base = (
                    f"{self.F1_FLAG} {self.year} {short_name} [{stage_label}]: "
                    f"{self.TROPHY} Predicted Champion: {champ_name} "
                    f"({champ_pts:.0f} pts) #F1 #Formula1 #TimeCopilot"
                )
                if len(base) > 275:
                    base = base[:272] + "..."
                return base
            return f"{self.F1_FLAG} F1 championship update — {short_name} [{stage_label}]. #F1"

        winner = race_forecast.predicted_winner()
        race_name = getattr(race_forecast, "race_name", self.race_name)
        short_name = _shorten_race_name(race_name)

        # Podium
        podium = race_forecast.predicted_podium()
        if len(podium) >= 3:
            p1 = podium[0]["driver_code"]
            p2 = podium[1]["driver_code"]
            p3 = podium[2]["driver_code"]
            win_pct = podium[0]["probability"] * 100
            podium_str = f"P1 {p1} ({win_pct:.0f}%), P2 {p2}, P3 {p3}"
        else:
            podium_str = winner.get("driver_code", "?")

        # Championship snippet
        champ_snippet = ""
        if driver_champ_forecast:
            champ = driver_champ_forecast.predicted_champion()
            champ_name = champ.get("name", "?")
            champ_pts = champ.get("predicted_points", 0)
            champ_snippet = f" | {self.TROPHY} Champ: {champ_name} ({champ_pts:.0f}pts predicted)"

        base = f"{self.F1_FLAG} {short_name} Prediction [{stage_label}]: {podium_str}{champ_snippet} #F1 #Formula1 #TimeCopilot"

        # Trim to 280 chars if needed
        if len(base) > 275:
            base = base[:272] + "..."

        return base

    # ── LinkedIn post ─────────────────────────────────────────────────────────

    def generate_linkedin_post(
        self,
        race_forecast,
        driver_champ_forecast,
        constructor_champ_forecast,
        prediction_evolution: Optional[pd.DataFrame],
        strategy_context: Optional[dict],
        session_stage: str,
    ) -> str:
        """
        Generate a full LinkedIn post in markdown format.
        ~800–1200 words, includes TimeCopilot's LLM narrative.
        """
        race_name = getattr(race_forecast, "race_name", None) or self.race_name
        now = datetime.now().strftime("%B %d, %Y")
        stage_labels = {
            "pre_weekend": "Pre-Weekend Prediction",
            "fp1": "Updated Prediction — After FP1",
            "fp2": "Updated Prediction — After FP2 (Race Pace Data In)",
            "fp3": "Updated Prediction — After FP3",
            "qualifying": "Final Pre-Race Prediction — After Qualifying",
            "sprint_qualifying": "Updated Prediction — After Sprint Qualifying",
            "sprint": "Updated Prediction — After Sprint Race",
            "post_race": "Post-Race Championship Update",
        }
        stage_label = stage_labels.get(session_stage, session_stage.upper())

        lines = [
            f"## {self.F1_FLAG} {race_name} — {stage_label}",
            f"*Powered by [TimeCopilot](https://timecopilot.dev/) — AI time series forecasting | {now}*",
            "",
            "---",
            "",
        ]

        # Race prediction section
        if race_forecast and not race_forecast.predicted_top10.empty:
            lines += self._race_prediction_section(race_forecast, strategy_context)

        # Championship prediction section
        if driver_champ_forecast:
            lines += self._championship_section(
                driver_champ_forecast, constructor_champ_forecast
            )

        # Prediction evolution (if multiple stages)
        if prediction_evolution is not None and not prediction_evolution.empty and len(prediction_evolution) > 1:
            lines += self._prediction_evolution_section(prediction_evolution)

        # TimeCopilot LLM narrative
        if race_forecast and getattr(race_forecast, "narrative", ""):
            lines += self._narrative_section(race_forecast.narrative, "Race")

        if driver_champ_forecast and getattr(driver_champ_forecast, "narrative", ""):
            lines += self._narrative_section(driver_champ_forecast.narrative, "Championship")

        # Note when LLM was unavailable (rate limit / key error)
        rf_model = getattr(race_forecast, "model_used", "") if race_forecast else ""
        cf_model = getattr(driver_champ_forecast, "model_used", "") if driver_champ_forecast else ""
        if "failed" in (rf_model, cf_model):
            lines += [
                "> ⚠️ *LLM analysis unavailable for this update — statistical model used instead.*",
                "",
            ]

        # Footer
        lines += [
            "",
            "---",
            "",
            "**Methodology:** Historical F1 data (2015–2025) from FastF1 + OpenF1 API, "
            "structured as time series and forecasted using [TimeCopilot](https://timecopilot.dev/) — "
            "the #1 ranked time series forecasting agent on the GIFT-Eval benchmark (NeurIPS 2025). "
            "Championship prediction uses cumulative points series. Race prediction uses "
            "circuit-specific finishing position trends enriched with current weekend pace data.",
            "",
            "#F1 #Formula1 #TimeCopilot #DataScience #MachineLearning #TimeSeriesForecasting #Racing",
        ]

        return "\n".join(lines)

    def _race_prediction_section(self, race_forecast, strategy_context: Optional[dict]) -> list[str]:
        lines = [
            f"### {self.CHECKERED} Race Prediction",
            "",
        ]

        winner = race_forecast.predicted_winner()
        podium = race_forecast.predicted_podium()

        if podium:
            lines.append("**Predicted Podium:**")
            for p in podium:
                emoji = ["🥇", "🥈", "🥉"][p["position"] - 1]
                lines.append(
                    f"{emoji} P{p['position']}: **{p['driver_code']}** ({p['constructor']}) "
                    f"— {p['probability']*100:.0f}% probability"
                )
            lines.append("")

        if not race_forecast.predicted_top10.empty:
            lines += [
                "**Full Top 10 Prediction:**",
                "",
                "| Pos | Driver | Team | Win Probability |",
                "|-----|--------|------|-----------------|",
            ]
            for _, row in race_forecast.predicted_top10.head(10).iterrows():
                pos = int(row.get("predicted_rank", 0))
                driver = str(row.get("driver_code", "?"))
                team = _normalize_constructor(str(row.get("constructor", "?")))
                prob = float(row.get("win_probability", 0)) * 100
                lines.append(f"| {pos} | {driver} | {team} | {prob:.0f}% |")
            lines.append("")

        if race_forecast.key_insight:
            lines += [
                f"💡 **Key Insight:** {race_forecast.key_insight}",
                "",
            ]

        # Strategy outlook
        if strategy_context:
            modal_stops = strategy_context.get("modal_stops", 2)
            sc_prob = strategy_context.get("sc_probability", 0)
            if modal_stops or sc_prob:
                lines += [
                    f"**Strategy Outlook:** Historically, teams opt for a "
                    f"{modal_stops:.0f}-stop strategy at this circuit. "
                    f"Safety car probability: {sc_prob*100:.0f}%.",
                    "",
                ]

        return lines

    def _championship_section(self, driver_fc, constructor_fc) -> list[str]:
        lines = [
            f"### {self.TROPHY} Championship Prediction",
            "",
        ]

        champ = driver_fc.predicted_champion()
        lines += [
            f"**Predicted {self.year} Driver Champion:** {champ.get('name', '?')} "
            f"({champ.get('predicted_points', 0):.0f} pts predicted final)",
            "",
        ]

        top5 = driver_fc.top_n(5)
        if not top5.empty:
            lines += [
                f"**Predicted Final {self.year} Driver Standings (Top 5):**",
                "",
                "| Pos | Driver | Current Pts | Predicted Final |",
                "|-----|--------|-------------|-----------------|",
            ]
            for i, row in top5.iterrows():
                name = str(row.get("unique_id", "")).replace("driver_", "")
                curr = float(row.get("current_points", 0))
                pred = float(row.get("predicted_points", 0))
                pos = int(row.get("predicted_position", i + 1))
                lines.append(f"| {pos} | {name} | {curr:.0f} | **{pred:.0f}** |")
            lines.append("")

        if constructor_fc:
            cons_champ = constructor_fc.predicted_champion()
            lines += [
                f"**Predicted {self.year} Constructor Champion:** {_normalize_constructor(cons_champ.get('name', '?'))} "
                f"({cons_champ.get('predicted_points', 0):.0f} pts)",
                "",
            ]

        remaining = driver_fc.remaining_races
        if remaining <= 0:
            lines.append("*Season complete — final championship standings.*")
        elif remaining == 1:
            lines.append("*1 race remaining in the season.*")
        else:
            lines.append(f"*{remaining} races remaining in the season.*")
        lines.append("")
        return lines

    def _prediction_evolution_section(self, evolution: pd.DataFrame) -> list[str]:
        lines = [
            "### 🔄 Prediction Evolution This Weekend",
            "",
            "| Stage | Predicted Winner | Confidence |",
            "|-------|-----------------|------------|",
        ]
        for _, row in evolution.iterrows():
            stage = str(row.get("stage", ""))
            winner = str(row.get("predicted_winner", "?"))
            prob = float(row.get("win_probability", 0)) * 100
            lines.append(f"| {stage} | {winner} | {prob:.0f}% |")
        lines.append("")
        return lines

    def _narrative_section(self, narrative: str, label: str) -> list[str]:
        if not narrative or "error" in narrative.lower()[:20]:
            return []
        # Take first 600 chars of narrative
        truncated = narrative[:600].strip()
        if len(narrative) > 600:
            truncated += "..."
        return [
            f"### 🤖 TimeCopilot Analysis — {label}",
            "",
            f"> {truncated}",
            "",
        ]

    # ── Charts ────────────────────────────────────────────────────────────────

    def _generate_charts(
        self,
        race_forecast,
        driver_champ_forecast,
        prediction_evolution: Optional[pd.DataFrame],
        session_stage: str,
    ) -> None:
        """Generate all Plotly charts and save as PNG."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("     ⚠️  plotly not installed — skipping chart generation")
            return

        # Chart 1: Race podium probabilities
        if race_forecast and not race_forecast.predicted_top10.empty:
            self._chart_race_probabilities(race_forecast, session_stage)

        # Chart 2: Championship standings
        if driver_champ_forecast:
            self._chart_championship_standings(driver_champ_forecast, session_stage)

        # Chart 3: Prediction evolution (qualifying only, when we have full data)
        if prediction_evolution is not None and not prediction_evolution.empty and len(prediction_evolution) >= 2:
            self._chart_prediction_evolution(prediction_evolution, session_stage)

    def _chart_race_probabilities(self, race_forecast, session_stage: str) -> None:
        """Bar chart: top 10 drivers by win probability."""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            top10 = race_forecast.predicted_top10.head(10).copy()
            top10 = top10.sort_values("win_probability", ascending=True)

            drivers = top10["driver_code"].tolist()
            probs = (top10["win_probability"] * 100).tolist()
            teams = top10.get("constructor", pd.Series([""] * len(top10))).tolist()

            # F1 team colors — 2026 lineup
            team_colors = {
                "Red Bull Racing": "#3671C6", "Red Bull": "#3671C6",
                "Ferrari": "#E8002D",
                "Mercedes": "#27F4D2",
                "McLaren": "#FF8000",
                "Aston Martin": "#229971",
                "Alpine": "#FF87BC",
                "Williams": "#64C4FF",
                "Haas": "#B6BABD",
                "Racing Bulls": "#6692FF", "RB": "#6692FF",
                "Audi": "#52E252", "Kick Sauber": "#52E252", "Sauber": "#52E252",
                "Cadillac": "#C0C0C0",
            }
            colors = [team_colors.get(_normalize_constructor(t), "#888888") for t in teams]

            fig = go.Figure(go.Bar(
                x=probs,
                y=drivers,
                orientation="h",
                marker_color=colors,
                text=[f"{p:.0f}%" for p in probs],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text=f"{race_forecast.race_name}<br><sup>Race Winner Probability — {session_stage.replace('_', ' ').title()}</sup>",
                    font=dict(size=18),
                ),
                xaxis_title="Win Probability (%)",
                xaxis=dict(range=[0, max(probs) * 1.3]),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="white", size=13),
                height=500,
                width=1200,
                margin=dict(l=80, r=80, t=80, b=60),
            )
            _save_chart(fig, self.charts_dir / f"race_probabilities_{session_stage}.png")
            print(f"     ✓ Race probability chart saved")
        except Exception as exc:
            print(f"     ⚠️  Race chart error: {exc}")

    def _chart_championship_standings(self, champ_fc, session_stage: str) -> None:
        """Horizontal bar chart: predicted final championship standings."""
        try:
            import plotly.graph_objects as go

            top8 = champ_fc.top_n(8).copy()
            if top8.empty:
                return

            names = [str(r).replace("driver_", "") for r in top8["unique_id"]]
            current = top8["current_points"].tolist()
            predicted = top8["predicted_points"].tolist()
            names_rev = names[::-1]
            current_rev = current[::-1]
            predicted_rev = predicted[::-1]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Current Points",
                x=current_rev, y=names_rev,
                orientation="h",
                marker_color="#4a9eff",
                opacity=0.7,
            ))
            fig.add_trace(go.Bar(
                name="Predicted Final",
                x=predicted_rev, y=names_rev,
                orientation="h",
                marker_color="#ff6b35",
                opacity=0.9,
            ))

            fig.update_layout(
                barmode="overlay",
                title=dict(
                    text=f"{self.year} F1 Driver Championship — Predicted Final Standings<br>"
                         f"<sup>After {champ_fc.race_name} | {champ_fc.remaining_races} races remaining</sup>",
                    font=dict(size=18),
                ),
                xaxis_title="Championship Points",
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="white", size=13),
                height=500,
                width=1200,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=120, r=80, t=100, b=60),
            )
            _save_chart(fig, self.charts_dir / f"championship_standings_{session_stage}.png")
            print(f"     ✓ Championship standings chart saved")
        except Exception as exc:
            print(f"     ⚠️  Championship chart error: {exc}")

    def _chart_prediction_evolution(
        self, evolution: pd.DataFrame, session_stage: str
    ) -> None:
        """Line chart: how win probability of predicted winner evolved across stages."""
        try:
            import plotly.graph_objects as go

            if evolution.empty or "stage" not in evolution.columns:
                return

            stages = evolution["stage"].tolist()
            probs = (evolution["win_probability"] * 100).tolist()
            winners = evolution["predicted_winner"].tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stages,
                y=probs,
                mode="lines+markers+text",
                text=winners,
                textposition="top center",
                line=dict(color="#ff6b35", width=3),
                marker=dict(size=10, color="#ff6b35"),
            ))
            fig.update_layout(
                title=dict(
                    text="Prediction Evolution Across Race Weekend<br>"
                         "<sup>How the predicted winner changed as more data came in</sup>",
                    font=dict(size=18),
                ),
                xaxis_title="Weekend Stage",
                yaxis_title="Win Probability (%)",
                yaxis=dict(range=[0, 100]),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="white", size=13),
                height=450,
                width=1200,
                margin=dict(l=80, r=80, t=100, b=60),
            )
            _save_chart(fig, self.charts_dir / f"prediction_evolution_{session_stage}.png")
            print(f"     ✓ Prediction evolution chart saved")
        except Exception as exc:
            print(f"     ⚠️  Evolution chart error: {exc}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _write(self, filename: str, content: str) -> None:
        (self.race_dir / filename).write_text(content, encoding="utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shorten_race_name(name: str) -> str:
    """Shorten 'Australian Grand Prix' → 'Australian GP'."""
    return name.replace("Grand Prix", "GP").replace("grand prix", "GP")


# Maps historical/Jolpica constructor names → current 2026 display names
_CONSTRUCTOR_DISPLAY: dict[str, str] = {
    # Team renames for 2026
    "kick sauber": "Audi",
    "sauber": "Audi",
    "alfa romeo": "Audi",
    "alfa romeo racing": "Audi",
    "rb": "Racing Bulls",
    "alphatauri": "Racing Bulls",
    "alpha tauri": "Racing Bulls",
    "scuderia alphatauri": "Racing Bulls",
    "toro rosso": "Racing Bulls",
    "scuderia toro rosso": "Racing Bulls",
    # Capitalisation fixes for names Jolpica returns in mixed case
    "red bull": "Red Bull Racing",
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "ferrari": "Ferrari",
    "aston martin": "Aston Martin",
    "alpine f1 team": "Alpine",
    "alpine": "Alpine",
    "williams": "Williams",
    "haas f1 team": "Haas",
    "haas": "Haas",
    "cadillac": "Cadillac",
    "racing bulls": "Racing Bulls",
    "red bull racing": "Red Bull Racing",
}


def _normalize_constructor(name: str) -> str:
    """
    Return the correct 2026 display name for a constructor.
    Handles historical renames (Kick Sauber → Audi, RB → Racing Bulls, etc.)
    and capitalisation issues from Jolpica.
    Falls back to title-casing the original name if no match is found.
    """
    if not name:
        return name
    key = name.strip().lower()
    return _CONSTRUCTOR_DISPLAY.get(key, name)


def _save_chart(fig, path: Path) -> None:
    """Save a Plotly figure as PNG. Falls back to HTML if kaleido not installed."""
    try:
        fig.write_image(str(path), width=1200, height=500, scale=2)
    except Exception:
        # Fallback: save as HTML if kaleido (PNG export) is not installed
        html_path = path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"       (saved as HTML — install kaleido for PNG: pip install kaleido)")
