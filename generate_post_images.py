"""
Generate two polished images for posting:
  1. Backtest results summary (Belgium, Azerbaijan, Monaco, Abu Dhabi 2025)
  2. Australia 2026 pre-weekend prediction
"""

import plotly.graph_objects as go
import os

OUTPUT_DIR = "reports/post_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
BG         = "#FFFFFF"
PANEL_BG   = "#F8F8F8"
F1_RED     = "#E10600"
GREEN      = "#00A650"
MISS_RED   = "#CC1400"
TEXT_DARK  = "#111111"
TEXT_MID   = "#444444"
TEXT_LIGHT = "#888888"
GOLD       = "#8B6914"   # dark gold, readable on white
FONT       = "Inter, Arial, sans-serif"

FOOTNOTE = (
    "Formula 1 Forecaster built with TimeCopilot by @_sebhgtz  "
    "|  Data: FastF1 · Jolpica-F1 API · OpenF1 API · Open-Meteo"
)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — BACKTEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

backtests = [
    dict(race="Belgian GP 2025",     flag="🇧🇪", round="Rd 13", predicted="NOR", actual="PIA", correct=False, note="NOR finished P2", prob="11.4%"),
    dict(race="Azerbaijan GP 2025",  flag="🇦🇿", round="Rd 17", predicted="VER", actual="VER", correct=True,  note="P1  ✓",           prob="22.8%"),
    dict(race="Monaco GP 2025",      flag="🇲🇨", round="Rd 8",  predicted="NOR", actual="NOR", correct=True,  note="P1  ✓",           prob="13.8%"),
    dict(race="Abu Dhabi GP 2025",   flag="🇦🇪", round="Rd 24", predicted="VER", actual="VER", correct=True,  note="P1  ✓",           prob="18.0%"),
]

fig1 = go.Figure()

# Figure layout — no axes, pure annotation canvas
fig1.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    width=920,
    height=700,
    margin=dict(l=30, r=30, t=180, b=80),
    font=dict(family=FONT, color=TEXT_DARK),
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0, 1]),
)

# ── Header ────────────────────────────────────────────────────────────────────
# Title
fig1.add_annotation(
    text="<b>TimeCopilot F1 Backtest Results</b>",
    xref="paper", yref="paper",
    x=0.5, y=1.28,
    font=dict(size=26, color=TEXT_DARK, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

# Subtitle — above the red line
fig1.add_annotation(
    text=(
        "2025 Season Backtest Results  |  "
        "Prediction Stage: After Qualifying  |  "
        "Compared to real 2025 race results"
    ),
    xref="paper", yref="paper",
    x=0.5, y=1.15,
    font=dict(size=12, color=TEXT_MID, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

# Red separator line — below subtitle, above cards
fig1.add_shape(
    type="line",
    xref="paper", yref="paper",
    x0=0.0, x1=1.0, y0=1.065, y1=1.065,
    line=dict(color=F1_RED, width=2.5),
)

# ── Cards ─────────────────────────────────────────────────────────────────────
# 2×2 grid. In paper coords: y increases upward.
# Each card occupies card_h in y and card_w in x.
card_w = 0.46
card_h = 0.45

# (cx, cy) = bottom-left corner of each card
positions = [
    (0.02, 0.51),   # top-left
    (0.52, 0.51),   # top-right
    (0.02, 0.03),   # bottom-left
    (0.52, 0.03),   # bottom-right
]

for bt, (cx, cy) in zip(backtests, positions):
    col  = GREEN if bt["correct"] else MISS_RED
    lbl  = "✓  CORRECT" if bt["correct"] else "✗  MISSED"
    top  = cy + card_h   # y of top edge of card

    # ── Card background box ────────────────────────────────────────────────
    fig1.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=cx, x1=cx + card_w,
        y0=cy, y1=top,
        fillcolor=PANEL_BG,
        line=dict(color=col, width=2.5),
        layer="below",
    )

    # ── Round badge (top-left) ─────────────────────────────────────────────
    fig1.add_annotation(
        text=bt["round"],
        xref="paper", yref="paper",
        x=cx + 0.015, y=top - 0.025,
        font=dict(size=11, color=TEXT_LIGHT, family=FONT),
        xanchor="left", yanchor="top",
        showarrow=False,
    )

    # ── Correct/Missed badge (top-right) ──────────────────────────────────
    fig1.add_annotation(
        text=f"<b>{lbl}</b>",
        xref="paper", yref="paper",
        x=cx + card_w - 0.015, y=top - 0.025,
        font=dict(size=12, color=col, family=FONT),
        xanchor="right", yanchor="top",
        showarrow=False,
    )

    # ── Race name (flag + name) ────────────────────────────────────────────
    fig1.add_annotation(
        text=f"<b>{bt['flag']}  {bt['race']}</b>",
        xref="paper", yref="paper",
        x=cx + card_w / 2, y=top - 0.095,
        font=dict(size=16, color=TEXT_DARK, family=FONT),
        xanchor="center", yanchor="middle",
        showarrow=False,
    )

    # ── Divider below race name ────────────────────────────────────────────
    fig1.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=cx + 0.02, x1=cx + card_w - 0.02,
        y0=top - 0.165, y1=top - 0.165,
        line=dict(color="#DDDDDD", width=1),
    )

    # ── Column labels ─────────────────────────────────────────────────────
    for col_x, col_lbl in [(0.27, "PREDICTED"), (0.73, "ACTUAL")]:
        fig1.add_annotation(
            text=col_lbl,
            xref="paper", yref="paper",
            x=cx + card_w * col_x, y=top - 0.215,
            font=dict(size=10, color=TEXT_LIGHT, family=FONT),
            xanchor="center", yanchor="middle",
            showarrow=False,
        )

    # ── Predicted driver (large) ───────────────────────────────────────────
    fig1.add_annotation(
        text=f"<b>{bt['predicted']}</b>",
        xref="paper", yref="paper",
        x=cx + card_w * 0.27, y=top - 0.31,
        font=dict(size=28, color=TEXT_DARK, family=FONT),
        xanchor="center", yanchor="middle",
        showarrow=False,
    )

    # ── Arrow ──────────────────────────────────────────────────────────────
    fig1.add_annotation(
        text="→",
        xref="paper", yref="paper",
        x=cx + card_w * 0.50, y=top - 0.31,
        font=dict(size=20, color=TEXT_LIGHT, family=FONT),
        xanchor="center", yanchor="middle",
        showarrow=False,
    )

    # ── Actual driver (large, colored) ────────────────────────────────────
    fig1.add_annotation(
        text=f"<b>{bt['actual']}</b>",
        xref="paper", yref="paper",
        x=cx + card_w * 0.73, y=top - 0.31,
        font=dict(size=28, color=col, family=FONT),
        xanchor="center", yanchor="middle",
        showarrow=False,
    )

    # ── Divider above note ─────────────────────────────────────────────────
    fig1.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=cx + 0.02, x1=cx + card_w - 0.02,
        y0=cy + 0.105, y1=cy + 0.105,
        line=dict(color="#DDDDDD", width=1),
    )

    # ── Win prob + note ────────────────────────────────────────────────────
    fig1.add_annotation(
        text=f"Win prob: <b>{bt['prob']}</b>  ·  {bt['note']}",
        xref="paper", yref="paper",
        x=cx + card_w / 2, y=cy + 0.058,
        font=dict(size=12, color=TEXT_MID, family=FONT),
        xanchor="center", yanchor="middle",
        showarrow=False,
    )

# ── Summary ───────────────────────────────────────────────────────────────────
fig1.add_annotation(
    text="<b>3 / 4 race winners correct  ·  4 / 4 WDC champion calls correct</b>",
    xref="paper", yref="paper",
    x=0.5, y=-0.055,
    font=dict(size=14, color=F1_RED, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
    bgcolor="#FFF5F5",
    bordercolor=F1_RED,
    borderwidth=1,
    borderpad=6,
)

# ── Footnote ──────────────────────────────────────────────────────────────────
fig1.add_annotation(
    text=FOOTNOTE,
    xref="paper", yref="paper",
    x=0.5, y=-0.115,
    font=dict(size=9, color=TEXT_LIGHT, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

out1 = f"{OUTPUT_DIR}/backtest_results.png"
fig1.write_image(out1, scale=2)
print(f"Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — AUSTRALIA 2026 PRE-WEEKEND PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

champ_data = [
    ("VER", "Red Bull Racing",  111.0),
    ("PER", "Red Bull Racing",  109.4),
    ("PIA", "McLaren",           99.0),
    ("NOR", "McLaren",           89.0),
    ("RUS", "Mercedes",          73.0),
    ("SAI", "Williams",          40.8),
    ("ANT", "Mercedes",          38.0),
    ("LEC", "Ferrari",           31.3),
    ("STR", "Aston Martin",      17.8),
    ("HAM", "Ferrari",           15.6),
]

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "McLaren":         "#FF8000",
    "Mercedes":        "#00A19C",
    "Ferrari":         "#E8002D",
    "Williams":        "#005AFF",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Haas":            "#888888",
    "Racing Bulls":    "#6692FF",
    "Kick Sauber":     "#39B54A",
}

drivers    = [d[0] for d in champ_data]
teams      = [d[1] for d in champ_data]
points     = [d[2] for d in champ_data]
bar_colors = [TEAM_COLORS.get(t, "#888888") for t in teams]

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=drivers,
    y=points,
    marker_color=bar_colors,
    marker_line=dict(width=0),
    text=[f"<b>{p:.0f}</b>" for p in points],
    textposition="outside",
    textfont=dict(size=13, color=TEXT_DARK, family=FONT),
    hovertemplate="%{x}<br>%{y:.0f} pts<extra></extra>",
    width=0.65,
))

fig2.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor="#FAFAFA",
    width=900,
    height=600,
    margin=dict(l=65, r=40, t=155, b=90),
    font=dict(family=FONT, color=TEXT_DARK),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        tickfont=dict(size=14, color=TEXT_DARK),
        linecolor="#CCCCCC",
        showline=True,
        ticks="outside",
        tickcolor="#CCCCCC",
    ),
    yaxis=dict(
        title=dict(text="Predicted Season Points", font=dict(size=12, color=TEXT_LIGHT)),
        showgrid=True,
        gridcolor="#EEEEEE",
        tickfont=dict(size=12, color=TEXT_LIGHT),
        range=[0, max(points) * 1.22],
        zeroline=False,
    ),
    bargap=0.3,
)

# Title
fig2.add_annotation(
    text="<b>2026 Australian GP — Pre-Weekend Prediction</b>",
    xref="paper", yref="paper",
    x=0.5, y=1.28,
    font=dict(size=24, color=TEXT_DARK, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

# Red separator line
fig2.add_shape(
    type="line",
    xref="paper", yref="paper",
    x0=0, x1=1, y0=1.165, y1=1.165,
    line=dict(color=F1_RED, width=2.5),
)

# Subtitle (below red line)
fig2.add_annotation(
    text="Predicted WDC: <b>VER</b>  ·  Championship Points Forecast — End of 2026 Season",
    xref="paper", yref="paper",
    x=0.5, y=1.09,
    font=dict(size=13, color=TEXT_MID, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

# VER champion callout
fig2.add_annotation(
    x="VER", y=111.0,
    text="<b>Predicted<br>Champion</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor=GOLD,
    arrowsize=1,
    arrowwidth=2,
    ax=60, ay=-50,
    font=dict(size=11, color=GOLD, family=FONT),
    bgcolor="#FFFDE7",
    bordercolor=GOLD,
    borderwidth=1,
    borderpad=5,
)

# Footnote
fig2.add_annotation(
    text=FOOTNOTE,
    xref="paper", yref="paper",
    x=0.5, y=-0.155,
    font=dict(size=9, color=TEXT_LIGHT, family=FONT),
    xanchor="center", yanchor="middle",
    showarrow=False,
)

out2 = f"{OUTPUT_DIR}/australia_2026_prediction.png"
fig2.write_image(out2, scale=2)
print(f"Saved: {out2}")

print("\nDone. Both images saved to reports/post_images/")
