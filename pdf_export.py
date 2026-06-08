"""Exportação server-side de relatórios em PDF via WeasyPrint."""

from __future__ import annotations

import base64
import html as html_lib
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template


def build_fig_png(fig: go.Figure, scale: int = 2) -> str:
    """Converte figura Plotly em data URI PNG (base64) para embutir no HTML."""
    png_bytes: bytes | None = None
    try:
        png_bytes = fig.to_image(format="png", scale=scale)
    except Exception:
        try:
            from kaleido.scopes.plotly import PlotlyScope

            scope = PlotlyScope()
            png_bytes = scope.transform(fig, format="png", scale=scale)
        except Exception:
            return ""
    if not png_bytes:
        return ""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_pdf_bytes(html_content: str) -> bytes:
    """Renderiza HTML em bytes PDF via WeasyPrint."""
    try:
        from weasyprint import HTML

        return HTML(string=html_content).write_pdf()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "WeasyPrint indisponível neste ambiente (libs Pango/Cairo ausentes). "
            "No Streamlit Cloud, verifique packages.txt na raiz do repositório."
        ) from exc


def _df_to_html_table(df: pd.DataFrame | None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return '<p class="muted">Sem dados para exibir.</p>'
    view = df.head(max_rows) if max_rows else df
    return view.to_html(index=False, classes="data-table", border=0, escape=True)


def _fig_img(fig: go.Figure | None, alt: str = "Gráfico") -> str:
    if fig is None:
        return ""
    src = build_fig_png(fig)
    return f'<img class="chart-img" src="{src}" alt="{html_lib.escape(alt)}" />'


def _diario_executivo_css() -> str:
    return """
    @page {
      size: A4 portrait;
      margin: 12mm 10mm 14mm 10mm;
    }
    * { box-sizing: border-box; }
    body {
      font-family: DejaVu Sans, Inter, system-ui, sans-serif;
      font-size: 10pt;
      color: #1e293b;
      line-height: 1.45;
      margin: 0;
    }
    .hero {
      background: linear-gradient(135deg, #0b2447 0%, #19376d 100%);
      color: white;
      border-radius: 10px;
      padding: 16px 18px;
      margin-bottom: 14px;
    }
    .hero-badge {
      display: inline-block;
      font-size: 8pt;
      font-weight: 700;
      letter-spacing: 0.08em;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.15);
      margin-bottom: 8px;
    }
    .hero h1 {
      margin: 0 0 4px 0;
      font-size: 18pt;
      font-weight: 700;
    }
    .hero .subtitle {
      margin: 0;
      font-size: 9.5pt;
      opacity: 0.92;
    }
    .hero .generated {
      margin: 8px 0 0 0;
      font-size: 8pt;
      opacity: 0.75;
    }
    .section {
      margin: 16px 0;
      page-break-inside: avoid;
    }
    .page-break {
      page-break-before: always;
    }
    h2 {
      font-size: 13pt;
      color: #0b2447;
      margin: 0 0 8px 0;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 4px;
    }
    h3 {
      font-size: 10.5pt;
      color: #19376d;
      margin: 12px 0 6px 0;
    }
    .muted { color: #64748b; font-size: 9pt; }
    .kpi-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 12px 0;
    }
    .kpi-card {
      flex: 1 1 140px;
      min-width: 120px;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #0b2447;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      page-break-inside: avoid;
    }
    .kpi-label {
      font-size: 7.5pt;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .kpi-value {
      font-size: 13pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 4px;
    }
    .kpi-delta {
      font-size: 8pt;
      color: #475569;
      margin-top: 3px;
    }
    .highlight-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0;
    }
    .highlight-card {
      flex: 1 1 180px;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #0b2447;
      border-radius: 8px;
      padding: 8px 10px;
      background: #f8fafc;
      page-break-inside: avoid;
    }
    .highlight-title {
      font-size: 7pt;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
    }
    .highlight-value {
      font-size: 11pt;
      font-weight: 700;
      color: #0b2447;
      margin-top: 2px;
    }
    .highlight-hint {
      font-size: 8pt;
      color: #475569;
      margin-top: 2px;
    }
    .metrics-row {
      display: flex;
      gap: 12px;
      margin: 8px 0;
      flex-wrap: wrap;
    }
    .metric-pill {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px 12px;
      background: #f8fafc;
      min-width: 120px;
    }
    .metric-pill .label { font-size: 8pt; color: #64748b; }
    .metric-pill .value { font-size: 12pt; font-weight: 700; color: #0b2447; }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 8.5pt;
      margin: 6px 0 10px 0;
      page-break-inside: avoid;
    }
    table.data-table th {
      background: #f1f5f9;
      color: #334155;
      font-weight: 600;
      text-align: left;
      padding: 5px 6px;
      border: 1px solid #e2e8f0;
    }
    table.data-table td {
      padding: 4px 6px;
      border: 1px solid #e2e8f0;
      vertical-align: top;
      word-wrap: break-word;
    }
    .chart-img {
      width: 100%;
      max-width: 100%;
      height: auto;
      margin: 8px 0;
      page-break-inside: avoid;
    }
    .two-col {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    .two-col > * { flex: 1 1 45%; min-width: 200px; }
    footer {
      margin-top: 20px;
      padding-top: 8px;
      border-top: 1px solid #e2e8f0;
      font-size: 8pt;
      color: #94a3b8;
      text-align: center;
    }
    """


DIARIO_EXECUTIVO_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>{{ css }}</style>
</head>
<body>
  <header class="hero">
    <div class="hero-badge">{{ badge }}</div>
    <h1>{{ title }}</h1>
    <p class="subtitle">{{ subtitle }}</p>
    <p class="generated">Gerado em {{ generated_at }}</p>
  </header>

  <section class="section">
    <h2>KPIs do dia</h2>
    <div class="kpi-wrap">
      {% for item in kpi_items %}
      <div class="kpi-card">
        <div class="kpi-label">{{ item.label }}</div>
        <div class="kpi-value">{{ item.value }}</div>
        {% if item.delta %}<div class="kpi-delta">{{ item.delta }}</div>{% endif %}
      </div>
      {% endfor %}
    </div>
  </section>

  {% if highlights %}
  <section class="section">
    <h2>Destaques do dia</h2>
    <div class="highlight-wrap">
      {% for h in highlights %}
      <div class="highlight-card">
        <div class="highlight-title">{{ h.titulo }}</div>
        <div class="highlight-value">{{ h.valor }}</div>
        <div class="highlight-hint">{{ h.hint }}</div>
      </div>
      {% endfor %}
    </div>
  </section>
  {% endif %}

  <section class="section page-break">
    <h2>Projeções e ritmo de vendas</h2>
    {{ progress.route_table_html }}
    <h3>Metas por percurso — desempenho vs meta</h3>
    {{ progress.metas_fig_html }}
    {% for fig_html in progress.cadence_figs_html %}
    {{ fig_html }}
    {% endfor %}
    <div class="metrics-row">
      <div class="metric-pill"><div class="label">% prazo decorrido</div><div class="value">{{ progress.pct_elapsed }}</div></div>
      <div class="metric-pill"><div class="label">% meta de inscritos</div><div class="value">{{ progress.pct_meta }}</div></div>
    </div>
    {% if progress.projections %}
    <div class="metrics-row">
      <div class="metric-pill"><div class="label">Projeção total (MM 7d)</div><div class="value">{{ progress.projections.mm7 }}</div></div>
      <div class="metric-pill"><div class="label">Projeção total (MM 15d)</div><div class="value">{{ progress.projections.mm15 }}</div></div>
      <div class="metric-pill"><div class="label">Projeção total (MM 30d)</div><div class="value">{{ progress.projections.mm30 }}</div></div>
    </div>
    {% endif %}
    {{ progress.ma_fig_html }}
  </section>

  <section class="section page-break">
    <h2>Gauges de metas de atletas (Marketing)</h2>
    {{ gauges_fig_html }}
  </section>

  <section class="section page-break">
    <h2>Demografia</h2>
    <div class="metrics-row">
      <div class="metric-pill"><div class="label">Idade média</div><div class="value">{{ demography.idade_media }}</div></div>
      <div class="metric-pill"><div class="label">Idade mínima</div><div class="value">{{ demography.idade_min }}</div></div>
      <div class="metric-pill"><div class="label">Idade máxima</div><div class="value">{{ demography.idade_max }}</div></div>
    </div>
    {{ demography.age_hist_html }}
    {% if demography.gender_table_html %}
    <h3>Gênero</h3>
    <div class="two-col">
      <div>{{ demography.gender_table_html }}</div>
      <div>{{ demography.gender_pie_html }}</div>
    </div>
    {% endif %}
    {% if demography.age_table_html %}
    <h3>Faixas etárias</h3>
    <div class="two-col">
      <div>{{ demography.age_table_html }}</div>
      <div>{{ demography.age_bar_html }}</div>
    </div>
    {% endif %}
  </section>

  {% if geography %}
  <section class="section page-break">
    <h2>Geografia Brasil</h2>
    <h3>Top cidades ({{ geography.n_cidades }} cidades)</h3>
    {{ geography.city_table_html }}
    <h3>Top Estados ({{ geography.n_ufs }} estados)</h3>
    {{ geography.uf_table_html }}
    <h3>Regiões do Brasil</h3>
    <div class="two-col">
      <div>{{ geography.reg_table_html }}</div>
      <div>{{ geography.reg_pie_html }}</div>
    </div>
    <p class="muted">NA = cidades sem correspondência no IBGE para mapeamento de região.</p>
  </section>
  {% endif %}

  {% if international %}
  <section class="section">
    <h2>Internacional</h2>
    <h3>Top 5 países</h3>
    {{ international.top5_html }}
    <h3>Lista completa de países</h3>
    {{ international.full_html }}
  </section>
  {% endif %}

  {% if coupons %}
  <section class="section page-break">
    <h2>Cupom de descontos</h2>
    <div class="metrics-row">
      <div class="metric-pill"><div class="label">Cupons usados</div><div class="value">{{ coupons.metrics.total_coupons_used }}</div></div>
      <div class="metric-pill"><div class="label">Cupons únicos</div><div class="value">{{ coupons.metrics.unique_coupons }}</div></div>
      <div class="metric-pill"><div class="label">Categorias únicas</div><div class="value">{{ coupons.metrics.unique_categories }}</div></div>
      <div class="metric-pill"><div class="label">Uso BRL</div><div class="value">{{ coupons.metrics.brl_used }}</div></div>
      <div class="metric-pill"><div class="label">Uso USD</div><div class="value">{{ coupons.metrics.usd_used }}</div></div>
    </div>
    <h3>Uso por planilha</h3>{{ coupons.currency_html }}
    <h3>Uso por categoria</h3>{{ coupons.category_html }}
    <h3>Uso por percurso</h3>{{ coupons.route_html }}
    <h3>Cruzamento categoria x percurso (Top 20)</h3>{{ coupons.cruzamento_html }}
    {{ coupons.category_fig_html }}
  </section>
  {% endif %}

  {% if team %}
  <section class="section">
    <h2>Assessorias, atestado e empresa</h2>
    <div class="metrics-row">
      <div class="metric-pill"><div class="label">Inscritos ativos</div><div class="value">{{ team.total }}</div></div>
      <div class="metric-pill"><div class="label">Preencheram company</div><div class="value">{{ team.company_filled }} ({{ team.company_pct }})</div></div>
      <div class="metric-pill"><div class="label">Subiram atestado</div><div class="value">{{ team.medical_uploaded }} ({{ team.medical_pct }})</div></div>
    </div>
    {% if team.team_table_html %}
    <h3>Top 10 assessorias</h3>
    <div class="two-col">
      <div>{{ team.team_table_html }}</div>
      <div>{{ team.team_fig_html }}</div>
    </div>
    {% endif %}
  </section>
  {% endif %}

  <section class="section page-break">
    <h2>Comparativo: ontem vs hoje</h2>
    <p class="muted">Apenas volume de inscrições (sem receita).</p>
    {{ compare_fig_html }}
  </section>

  <footer>Paraty Brazil by UTMB — Marketing Diário Executivo</footer>
</body>
</html>"""
)


def build_diario_executivo_html(
    scoped: pd.DataFrame,
    percurso_targets: dict[str, int],
    data_base_label: str,
    data_base_ts: pd.Timestamp | None,
    start_date: date,
    end_date: date,
    ibge_df: pd.DataFrame,
) -> str:
    """Monta HTML completo do relatório Marketing Diário — Executivo."""
    import dashboard_2026 as d

    deltas = d.compute_daily_deltas(scoped, data_base_ts)
    ref_day: date = deltas["ref_day"]  # type: ignore[assignment]
    weekday_label = deltas["weekday_label"]
    today_df = d._filter_by_day(scoped, ref_day)
    inscritos_ontem = int(deltas["inscritos_ontem"])  # type: ignore[arg-type]
    inscritos_hoje = int(deltas["inscritos_hoje"])  # type: ignore[arg-type]

    kpi_items = d._build_marketing_diario_kpi_items(scoped, today_df, deltas, percurso_targets)
    highlights = d.compute_marketing_highlights(
        scoped, granularidade="diario", ref_ts=data_base_ts, targets=percurso_targets
    )[:5]

    progress_bundle = d.build_progress_projection_bundle(
        scoped,
        percurso_targets,
        start_date,
        end_date,
        data_base_ts,
        daily_sales_window=15,
    )
    progress_ctx: dict[str, Any] = {
        "route_table_html": _df_to_html_table(progress_bundle["route_table"]),
        "metas_fig_html": _fig_img(progress_bundle["metas_fig"], "Metas por percurso"),
        "cadence_figs_html": [
            _fig_img(fig, "Cadência de inscrições") for fig in progress_bundle["cadence_figs"]
        ],
        "pct_elapsed": progress_bundle["pct_elapsed"],
        "pct_meta": progress_bundle["pct_meta"],
        "ma_fig_html": _fig_img(progress_bundle["ma_fig"], "Média móvel e projeção"),
        "projections": progress_bundle["projections"] if progress_bundle["has_series"] else None,
    }

    gauges_fig = d.build_marketing_gauges_grid_figure(scoped, percurso_targets)

    demo = d.build_demography_bundle(scoped, expandido=True)
    demography_ctx = {
        "idade_media": demo["idade_media"],
        "idade_min": demo["idade_min"],
        "idade_max": demo["idade_max"],
        "age_hist_html": _fig_img(demo["age_hist_fig"], "Distribuição de idade"),
        "gender_table_html": _df_to_html_table(demo["gender_table"]) if demo["has_gender"] else "",
        "gender_pie_html": _fig_img(demo["gender_pie_fig"], "Gênero") if demo["has_gender"] else "",
        "age_table_html": _df_to_html_table(demo["age_table"]) if demo["has_age"] else "",
        "age_bar_html": _fig_img(demo["age_bar_fig"], "Faixas etárias") if demo["has_age"] else "",
    }

    geo_bundle = d.build_geography_bundle(scoped, ibge_df)
    geography_ctx = None
    if geo_bundle:
        geography_ctx = {
            "n_cidades": geo_bundle["n_cidades"],
            "n_ufs": geo_bundle["n_ufs"],
            "city_table_html": _df_to_html_table(geo_bundle["city_counts"], max_rows=30),
            "uf_table_html": _df_to_html_table(geo_bundle["uf_counts"]),
            "reg_table_html": _df_to_html_table(geo_bundle["reg_counts"]),
            "reg_pie_html": _fig_img(geo_bundle["reg_pie_fig"], "Regiões do Brasil"),
        }

    intl_bundle = d.build_international_bundle(scoped)
    international_ctx = None
    if intl_bundle:
        international_ctx = {
            "top5_html": _df_to_html_table(intl_bundle["top_5"]),
            "full_html": _df_to_html_table(intl_bundle["full"]),
        }

    coupon_bundle = d.build_coupon_bundle(scoped)
    coupons_ctx = None
    if coupon_bundle:
        coupons_ctx = {
            "metrics": coupon_bundle["metrics"],
            "currency_html": _df_to_html_table(coupon_bundle["currency_table"]),
            "category_html": _df_to_html_table(coupon_bundle["category_table"]),
            "route_html": _df_to_html_table(coupon_bundle["route_table"]),
            "cruzamento_html": _df_to_html_table(coupon_bundle["cruzamento_table"]),
            "category_fig_html": _fig_img(coupon_bundle["category_fig"], "Top categorias de cupom"),
        }

    team_bundle = d.build_team_medical_company_bundle(scoped)
    team_ctx = None
    if team_bundle:
        team_ctx = {
            "total": team_bundle["total"],
            "company_filled": team_bundle["company_filled"],
            "company_pct": team_bundle["company_pct"],
            "medical_uploaded": team_bundle["medical_uploaded"],
            "medical_pct": team_bundle["medical_pct"],
            "team_table_html": _df_to_html_table(team_bundle["team_table"]) if team_bundle["has_team"] else "",
            "team_fig_html": _fig_img(team_bundle["team_fig"], "Top assessorias") if team_bundle["has_team"] else "",
        }

    compare_fig = d._build_daily_inscritos_compare_figure(inscritos_ontem, inscritos_hoje)

    return DIARIO_EXECUTIVO_TEMPLATE.render(
        css=_diario_executivo_css(),
        badge="DIÁRIO · EXECUTIVO",
        title="Marketing Diário — Executivo",
        subtitle=(
            f"{weekday_label}, {ref_day.strftime('%d/%m/%Y')} | "
            f"Base atualizada até {data_base_label}"
        ),
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
        kpi_items=kpi_items,
        highlights=highlights,
        progress=progress_ctx,
        gauges_fig_html=_fig_img(gauges_fig, "Gauges de metas"),
        demography=demography_ctx,
        geography=geography_ctx,
        international=international_ctx,
        coupons=coupons_ctx,
        team=team_ctx,
        compare_fig_html=_fig_img(compare_fig, "Ontem vs hoje"),
    )
