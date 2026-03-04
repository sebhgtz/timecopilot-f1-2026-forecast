"""
GitHub Pages Static Site Generator
=====================================
Generates a public HTML site from the reports/ folder.

Each race weekend gets its own page with:
  - Social card (Twitter text)
  - Embedded prediction charts (PNG)
  - Full LinkedIn post rendered as HTML

The index page lists all races with links to the latest prediction stage.

Usage:
    python scripts/generate_github_pages.py

Output: docs/index.html + docs/races/{race_slug}_{year}/index.html
Deployed automatically via GitHub Actions → GitHub Pages.
"""

from __future__ import annotations

import csv
import re
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Allow importing f1_pipeline from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS_DIR = Path("reports")
DOCS_DIR = Path("docs")
RACES_DIR = DOCS_DIR / "races"

# Stage display labels (most-recent-first priority order)
STAGE_PRIORITY = [
    "post_race", "qualifying", "sprint", "sprint_qualifying",
    "fp3", "fp2", "fp1", "pre_weekend",
]
STAGE_LABELS = {
    "pre_weekend": "Pre-Weekend",
    "fp1": "After FP1",
    "fp2": "After FP2",
    "fp3": "After FP3",
    "sprint_qualifying": "After Sprint Qualifying",
    "sprint": "After Sprint",
    "qualifying": "After Qualifying",
    "post_race": "Post-Race",
}

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f1a; color: #e0e0e0; margin: 0; padding: 0; }
header { background: #1a1a2e; border-bottom: 2px solid #e10600; padding: 16px 32px;
         display: flex; align-items: center; gap: 16px; }
header h1 { margin: 0; font-size: 1.4rem; color: #fff; }
header .subtitle { color: #aaa; font-size: 0.85rem; }
main { max-width: 1100px; margin: 32px auto; padding: 0 24px; }
h2 { color: #e10600; font-size: 1.1rem; margin-top: 2em; border-bottom: 1px solid #333; padding-bottom: 6px; }
h3 { color: #fff; }
.race-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin: 24px 0; }
.race-card { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 8px; padding: 16px;
             text-decoration: none; color: #e0e0e0; transition: border-color 0.2s; }
.race-card:hover { border-color: #e10600; }
.race-card .round { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
.race-card .name { font-size: 1rem; font-weight: 600; color: #fff; }
.race-card .date { font-size: 0.8rem; color: #aaa; margin-top: 4px; }
.race-card .stage-badge { display: inline-block; margin-top: 8px; background: #e10600; color: #fff;
                           font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; }
.race-card .sprint-badge { display: inline-block; margin-top: 4px; background: #ff8c00; color: #fff;
                            font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; margin-left: 4px; }
.social-card { background: #1a1a2e; border-left: 4px solid #e10600; padding: 16px 20px;
               border-radius: 0 8px 8px 0; font-family: monospace; font-size: 0.9rem;
               white-space: pre-wrap; word-break: break-word; margin: 16px 0; }
.stage-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
.stage-tab { padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; text-decoration: none;
             background: #2a2a4e; color: #aaa; border: 1px solid #3a3a6e; }
.stage-tab.active { background: #e10600; color: #fff; border-color: #e10600; }
.chart-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
.chart-row img { max-width: 100%; border-radius: 8px; border: 1px solid #2a2a4e; }
.linkedin-post { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 8px; padding: 20px 24px; margin: 16px 0; }
.linkedin-post h3 { color: #4a9eff; }
.linkedin-post table { border-collapse: collapse; width: 100%; }
.linkedin-post th, .linkedin-post td { border: 1px solid #3a3a5e; padding: 6px 12px; text-align: left; }
.linkedin-post th { background: #2a2a4e; color: #aaa; }
.linkedin-post blockquote { border-left: 3px solid #e10600; margin: 0; padding-left: 16px; color: #bbb; }
footer { text-align: center; padding: 32px; color: #555; font-size: 0.8rem; border-top: 1px solid #1a1a2e; margin-top: 40px; }
a { color: #4a9eff; }
"""


def _markdown_to_html(md: str) -> str:
    """
    Very lightweight markdown → HTML converter (no external deps).
    Handles: headers, bold, italic, tables, blockquotes, horizontal rules, links.
    """
    lines = md.split("\n")
    html_lines = []
    in_table = False

    for line in lines:
        # Horizontal rule
        if re.match(r"^---+$", line.strip()):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append("<hr>")
            continue

        # Headings
        h_match = re.match(r"^(#{1,4})\s+(.*)", line)
        if h_match:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            level = len(h_match.group(1))
            text = _inline(h_match.group(2))
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # Blockquote
        if line.startswith("> "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            continue

        # Table row
        if line.startswith("|") and "|" in line[1:]:
            cols = [c.strip() for c in line.strip("|").split("|")]
            # Check if it's a separator row (|---|---|)
            if all(re.match(r"^[-: ]+$", c) for c in cols if c):
                continue  # skip separator
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            row_html = "".join(f"<td>{_inline(c)}</td>" for c in cols)
            html_lines.append(f"<tr>{row_html}</tr>")
            continue

        if in_table:
            html_lines.append("</table>")
            in_table = False

        # Empty line → paragraph break
        if not line.strip():
            html_lines.append("<br>")
            continue

        html_lines.append(f"<p>{_inline(line)}</p>")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)


def _inline(text: str) -> str:
    """Convert inline markdown: bold, italic, code, links."""
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic *text* (don't match ** already consumed)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _page_html(title: str, body: str, show_home: bool = False) -> str:
    home_link = ' · <a href="/timecopilot-f1-2026-forecast/" style="color:#aaa">← Home</a>' if show_home else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>🏎️ F1 TimeCopilot Predictions</h1>
    <div class="subtitle">AI-powered Formula 1 forecasting · Powered by
      <a href="https://timecopilot.dev/" style="color:#e10600">TimeCopilot</a>
    </div>
  </div>
  <div style="margin-left:auto; font-size:0.8rem; color:#888;">
    <a href="https://github.com/sebhgtz/timecopilot-f1-2026-forecast" style="color:#aaa">GitHub ↗</a>{home_link}
  </div>
</header>
<main>
{body}
</main>
<footer>
  Generated by <a href="https://timecopilot.dev/">TimeCopilot</a> ·
  Data sources: FastF1, OpenF1, Jolpica-F1 ·
  Updated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</footer>
</body>
</html>
"""


def _load_accuracy_log() -> dict[tuple[str, int], dict]:
    """Load accuracy_log.csv keyed by (slug, year). Returns best (latest) row per race."""
    log_path = REPORTS_DIR / "accuracy_log.csv"
    results: dict[tuple[str, int], dict] = {}
    if not log_path.exists():
        return results
    with open(log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["race_slug"], int(row["year"]))
            results[key] = row  # later rows overwrite earlier ones (most recent stage wins)
    return results


def _result_banner_html(accuracy: dict) -> str:
    """Render a result banner from an accuracy_log row."""
    predicted = accuracy["predicted_winner"]
    actual = accuracy["actual_winner"]
    correct = accuracy["correct"].lower() == "true"
    stage = STAGE_LABELS.get(accuracy["session_stage"], accuracy["session_stage"].upper())
    icon = "✅" if correct else "❌"
    verdict = "Correct" if correct else "Wrong"
    color = "#1a3a1a" if correct else "#3a1a1a"
    border = "#2a6a2a" if correct else "#6a2a2a"
    pos = accuracy.get("actual_position_of_predicted", "")
    pos_note = f" (finished P{pos})" if pos and pos != "1" else ""
    return f"""<div style="background:{color}; border:1px solid {border}; border-radius:8px;
  padding:12px 18px; margin-bottom:16px; font-size:0.9rem;">
  {icon} <strong>Result ({stage}):</strong>
  Predicted <strong>{predicted}</strong> · Actual winner <strong>{actual}</strong>{pos_note} · {verdict}
</div>"""


def _get_round(slug: str, year: int) -> int:
    """Return the round number for a race slug+year using the F1 calendar."""
    try:
        from f1_pipeline.collectors.calendar_manager import F1Calendar
        cal = F1Calendar(year)
        for race in cal.races:
            if race.slug == slug:
                return race.round
    except Exception:
        pass
    # Fallback for known backtests
    return {
        ("abu_dhabi", 2025): 24,
        ("azerbaijan", 2025): 17,
        ("belgium", 2025): 13,
        ("monaco", 2025): 8,
        ("australia", 2026): 1,
    }.get((slug, year), 99)


def _discover_races() -> list[dict]:
    """
    Scan reports/ to find all race directories with their stages.
    Returns list of dicts (unsorted — build_index handles grouping/ordering).
    """
    races = []
    for race_dir in sorted(REPORTS_DIR.iterdir()):
        if not race_dir.is_dir():
            continue

        # Parse slug and year from dir name e.g. australia_2026
        parts = race_dir.name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue

        slug = parts[0]
        year = int(parts[1])

        # Find available stages
        stages_present = []
        for stage in STAGE_PRIORITY:
            if (race_dir / f"social_card_{stage}.txt").exists():
                stages_present.append(stage)

        if not stages_present:
            continue

        latest_stage = stages_present[0]  # highest priority = most recent
        social_card = (race_dir / f"social_card_{latest_stage}.txt").read_text(encoding="utf-8").strip()

        races.append({
            "slug": slug,
            "year": year,
            "round": _get_round(slug, year),
            "dir": race_dir,
            "name": slug.replace("_", " ").title() + " Grand Prix",
            "stages": stages_present,
            "latest_stage": latest_stage,
            "social_card": social_card,
        })

    return races


def build_race_page(race: dict, accuracy_log: dict) -> None:
    """Generate the race detail page."""
    out_dir = RACES_DIR / race["dir"].name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy chart images
    charts_src = race["dir"] / "charts"
    charts_dst = out_dir / "charts"
    if charts_src.exists():
        if charts_dst.exists():
            shutil.rmtree(charts_dst)
        shutil.copytree(charts_src, charts_dst)

    # Build stage tabs HTML
    tabs_html = '<div class="stage-tabs">'
    for stage in reversed(race["stages"]):  # chronological order in tabs
        label = STAGE_LABELS.get(stage, stage.upper())
        active = " active" if stage == race["latest_stage"] else ""
        tabs_html += f'<a href="#{stage}" class="stage-tab{active}">{label}</a>'
    tabs_html += "</div>"

    # Build content for each stage
    content_html = ""
    for stage in race["stages"]:
        label = STAGE_LABELS.get(stage, stage.upper())

        # Social card
        social_file = race["dir"] / f"social_card_{stage}.txt"
        social_text = social_file.read_text(encoding="utf-8").strip() if social_file.exists() else ""

        # LinkedIn post
        linkedin_file = race["dir"] / f"linkedin_post_{stage}.md"
        linkedin_html = ""
        if linkedin_file.exists():
            linkedin_md = linkedin_file.read_text(encoding="utf-8")
            linkedin_html = f'<div class="linkedin-post">{_markdown_to_html(linkedin_md)}</div>'

        # Charts for this stage
        chart_html = ""
        if charts_dst.exists():
            chart_files = sorted(charts_dst.glob(f"*_{stage}.png"))
            if chart_files:
                chart_html = '<div class="chart-row">'
                for cf in chart_files:
                    chart_html += f'<img src="charts/{cf.name}" alt="{cf.stem}" loading="lazy">'
                chart_html += "</div>"

        content_html += f"""
<section id="{stage}">
  <h2>{label}</h2>
  <div class="social-card">{social_text}</div>
  {chart_html}
  {linkedin_html}
</section>
"""

    result_key = (race["dir"].name.rsplit("_", 1)[0], race["year"])
    result_html = _result_banner_html(accuracy_log[result_key]) if result_key in accuracy_log else ""

    body = f"""
<h2 style="font-size:1.6rem; border:none; color:#fff;">
  🏁 {race['name']} <span style="color:#888;font-size:1rem;">{race['year']}</span>
</h2>
{result_html}{tabs_html}
{content_html}
"""

    html = _page_html(f"{race['name']} {race['year']} — F1 TimeCopilot", body, show_home=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"  ✓ {race['dir'].name}/index.html")


def _race_cards_html(races: list[dict]) -> str:
    """Render a grid of race cards for a given list of races."""
    html = '<div class="race-grid">'
    for race in races:
        stage = race["latest_stage"]
        label = STAGE_LABELS.get(stage, stage.upper())
        href = f"races/{race['dir'].name}/"
        snippet = race["social_card"][:120] + ("..." if len(race["social_card"]) > 120 else "")
        round_label = f"Rd {race['round']}" if race["round"] != 99 else str(race["year"])
        html += f"""
<a href="{href}" class="race-card">
  <div class="round">{round_label}</div>
  <div class="name">{race['name']}</div>
  <div class="date">{snippet}</div>
  <span class="stage-badge">{label}</span>
</a>"""
    html += "</div>"
    return html


def _accuracy_banner_html(accuracy_log: dict, year: int) -> str:
    """Render a per-year accuracy summary banner."""
    year_rows = {k: v for k, v in accuracy_log.items() if k[1] == year}
    # Only show races that have already happened (have an actual_winner)
    completed = {k: v for k, v in year_rows.items() if v.get("actual_winner")}
    if not completed:
        return '<p style="color:#666; font-size:0.85rem; margin:4px 0 12px;">No completed races yet.</p>'
    total = len(completed)
    correct = sum(1 for r in completed.values() if r["correct"].lower() == "true")
    pct = round(correct / total * 100)
    items = ""
    for row in sorted(completed.values(), key=lambda r: int(r["round"])):
        icon = "✅" if row["correct"].lower() == "true" else "❌"
        items += (
            f'<span style="margin-right:16px; white-space:nowrap;">'
            f'{icon} {row["race_name"].replace(" Grand Prix", " GP")}: '
            f'<strong>{row["predicted_winner"]}</strong>'
            f'</span>'
        )
    return f"""<div style="background:#1a1a2e; border:1px solid #2a2a4e; border-radius:8px;
  padding:14px 18px; margin:8px 0 16px;">
  <div style="font-size:0.95rem; font-weight:600; color:#fff; margin-bottom:6px;">
    🎯 {correct}/{total} race winners correctly predicted ({pct}%)
    <span style="font-size:0.75rem; color:#888; font-weight:400; margin-left:8px;">after qualifying</span>
  </div>
  <div style="font-size:0.85rem; color:#ccc; flex-wrap:wrap; display:flex;">
    {items}
  </div>
</div>"""


def build_index(races: list[dict], accuracy_log: dict) -> None:
    """Generate the home index page, grouped by year (newest first)."""
    n_races = len(races)
    n_reports = sum(len(r["stages"]) for r in races)

    # Group by year, each group sorted most recent round first
    years = sorted({r["year"] for r in races}, reverse=True)
    sections_html = ""
    for year in years:
        year_races = sorted(
            [r for r in races if r["year"] == year],
            key=lambda r: r["round"],
            reverse=True,
        )
        year_label = f"🏆 {year} Season"
        accuracy_html = _accuracy_banner_html(accuracy_log, year)
        sections_html += f"<h2>{year_label}</h2>\n{accuracy_html}{_race_cards_html(year_races)}\n"

    all_completed = {k: v for k, v in accuracy_log.items() if v.get("actual_winner")}
    all_total = len(all_completed)
    all_correct = sum(1 for r in all_completed.values() if r["correct"].lower() == "true")
    global_banner = (
        f'<p style="color:#aaa; font-size:0.9rem; margin:0 0 16px;">'
        f'🎯 <strong>{all_correct}/{all_total}</strong> race winners correctly predicted across all backtests</p>'
    ) if all_total else ""

    body = f"""
<h2 style="font-size:1.5rem; border:none; color:#fff; margin-top:0;">
  F1 TimeCopilot Predictions
</h2>
{global_banner}<p style="color:#aaa;">
  AI-powered Formula 1 predictions for every race weekend.
  Updated after each session (FP1 → Sprint → Qualifying → Race).
  Forecasts generated by <a href="https://timecopilot.dev/">TimeCopilot</a>
  using historical FastF1 + OpenF1 data.
</p>
<p style="color:#666; font-size:0.85rem;">
  {n_races} race(s) · {n_reports} prediction update(s) · Last updated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</p>
{sections_html}
<h2>About</h2>
<p>
  Predictions are generated using <a href="https://timecopilot.dev/">TimeCopilot</a>,
  the #1 ranked time series forecasting agent on the GIFT-Eval benchmark (NeurIPS 2025).
  Historical data covers 2015–2025 from FastF1 (lap times, session pace) and
  Jolpica-F1 (championship standings). Strategy data from OpenF1 API.
</p>
<p style="color:#888; font-size:0.85rem;">
  Source code: <a href="https://github.com/sebhgtz/timecopilot-f1-2026-forecast">GitHub</a>
</p>
"""

    html = _page_html("F1 2026 TimeCopilot Predictions", body)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"  ✓ docs/index.html ({n_races} races)")


def main() -> None:
    print("\n🌐 Generating GitHub Pages static site...")
    DOCS_DIR.mkdir(exist_ok=True)
    RACES_DIR.mkdir(exist_ok=True)

    # .nojekyll prevents GitHub Pages from ignoring underscored paths
    (DOCS_DIR / ".nojekyll").touch()

    races = _discover_races()
    if not races:
        print("  ⚠️  No reports found in reports/ — nothing to generate.")
        return

    accuracy_log = _load_accuracy_log()

    for race in races:
        build_race_page(race, accuracy_log)

    build_index(races, accuracy_log)
    print(f"\n✅ Site generated → {DOCS_DIR}/index.html ({len(races)} races)")
    print(f"   Deploy with: GitHub Pages → source: /docs")


if __name__ == "__main__":
    main()
