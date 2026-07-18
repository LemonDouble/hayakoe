"""품질 리포트 HTML / SVG 렌더링 — torch 의존 없는 순수 렌더링 계층."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cli.i18n import t
from cli.ui.html_theme import BASE_CSS


@dataclass
class CheckpointInfo:
    path: Path
    epoch: int
    step: int

    @property
    def label(self) -> str:
        return f"e{self.epoch} s{self.step}"


def _parse_checkpoint(path: Path) -> CheckpointInfo:
    match = re.search(r"_e(\d+)_s(\d+)\.safetensors$", path.name)
    if match:
        return CheckpointInfo(path=path, epoch=int(match.group(1)), step=int(match.group(2)))
    return CheckpointInfo(path=path, epoch=0, step=0)


_CSS = BASE_CSS + """\
.c{max-width:1400px}
.mg{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.mc{background:var(--color-surface);border:1px solid var(--color-border);border-radius:12px;padding:16px 20px;transition:border-color 0.2s}
.mc:hover{border-color:var(--color-primary-dim-border)}
.mt{color:var(--color-text-secondary);font-size:13px;font-weight:600;margin-bottom:4px}
.mv{color:var(--color-text-primary);font-weight:700}
.ms{color:var(--color-text-muted);font-size:11px;line-height:1.4;margin-bottom:8px}
.mn{color:var(--color-text-muted);font-size:12px}
.tc{text-align:left!important;max-width:320px;color:var(--color-text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.err{color:var(--color-text-muted);font-size:12px}
audio{width:200px;height:32px}
"""

_JS = """\
document.querySelectorAll('audio').forEach(a=>{
  a.addEventListener('play',()=>{
    document.querySelectorAll('audio').forEach(o=>{if(o!==a)o.pause()});
  });
});
"""


def _svg_chart(
    data: list[tuple[int, float]],
    title: str,
    description: str = "",
    width: int = 500,
    height: int = 150,
    color: str = "#F0B90B",
) -> str:
    desc_html = f'<div class="ms">{description}</div>' if description else ""
    if not data or len(data) < 2:
        return (
            f'<div class="mc"><div class="mt">{title}</div>'
            f'{desc_html}<p class="mn">{t("report.html.no_data")}</p></div>'
        )

    steps = [d[0] for d in data]
    values = [d[1] for d in data]

    min_s, max_s = min(steps), max(steps)
    min_v, max_v = min(values), max(values)
    v_range = max_v - min_v or 1.0
    s_range = max_s - min_s or 1.0

    ml, mr, mt, mb = 10, 10, 8, 18
    pw = width - ml - mr
    ph = height - mt - mb

    # 데이터 포인트가 너무 많으면 다운샘플
    sampled = data
    if len(data) > 400:
        step_size = max(1, len(data) // 400)
        sampled = data[::step_size]
        steps = [d[0] for d in sampled]
        values = [d[1] for d in sampled]

    points = []
    for s, v in zip(steps, values):
        x = ml + (s - min_s) / s_range * pw
        y = mt + ph - (v - min_v) / v_range * ph
        points.append(f"{x:.1f},{y:.1f}")

    last_val = values[-1]

    # 가로 그리드라인
    grid = ""
    for i in range(1, 4):
        gy = mt + ph * i / 4
        grid += f'<line x1="{ml}" y1="{gy:.0f}" x2="{ml + pw}" y2="{gy:.0f}" stroke="#2E2723" stroke-width="1"/>'

    return f'''<div class="mc">
  <div class="mt">{title} <span class="mv">{last_val:.4f}</span></div>
  {desc_html}
  <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">
    {grid}
    <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>
    <text x="{ml}" y="{height - 1}" fill="#5C524A" font-size="10" font-family="Pretendard,sans-serif">{min_s}</text>
    <text x="{width - mr}" y="{height - 1}" fill="#5C524A" font-size="10" font-family="Pretendard,sans-serif" text-anchor="end">{max_s}</text>
  </svg>
</div>'''


def _build_html(
    speaker_name: str,
    checkpoints: list[CheckpointInfo],
    texts: list[str],
    audio_matrix: dict[str, dict[str, str]],
    metrics: dict[str, list[tuple[int, float]]],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 차트 — 디자인 시스템 토큰 + 구분용 보조 색
    chart_specs = [
        ("loss/g/total", t("report.html.chart.generator_loss"),
         t("report.html.chart.generator_loss_desc"), "#F0B90B"),
        ("loss/g/mel", t("report.html.chart.mel_loss"),
         t("report.html.chart.mel_loss_desc"), "#4ADE80"),
        ("loss/d/total", t("report.html.chart.discriminator_loss"),
         t("report.html.chart.discriminator_loss_desc"), "#CD6B5E"),
        ("loss/g/kl", t("report.html.chart.kl_loss"),
         t("report.html.chart.kl_loss_desc"), "#60A5FA"),
        ("loss/g/dur", t("report.html.chart.duration_loss"),
         t("report.html.chart.duration_loss_desc"), "#EC4899"),
        ("loss/g/fm", t("report.html.chart.fm_loss"),
         t("report.html.chart.fm_loss_desc"), "#A78BFA"),
    ]

    charts = ""
    for tag, title, desc, color in chart_specs:
        if tag in metrics and len(metrics[tag]) >= 2:
            charts += _svg_chart(metrics[tag], title, desc, color=color)

    metrics_section = ""
    if charts:
        metrics_section = f"<section><h2>{t('report.html.metrics_heading')}</h2><div class='mg'>{charts}</div></section>"

    # 테이블
    ths = f'<th class="tc">{t("report.html.col_text")}</th>'
    for ckpt in checkpoints:
        ths += f"<th>{ckpt.label}</th>"

    rows = ""
    for text in texts:
        short = text if len(text) <= 50 else text[:47] + "..."
        escaped = text.replace('"', "&quot;").replace("<", "&lt;")
        tds = f'<td class="tc" title="{escaped}">{short}</td>'
        for ckpt in checkpoints:
            uri = audio_matrix.get(ckpt.label, {}).get(text, "")
            if uri:
                tds += f'<td><audio controls preload="none"><source src="{uri}" type="audio/wav"></audio></td>'
            else:
                tds += f'<td class="err">{t("report.html.error_cell")}</td>'
        rows += f"<tr>{tds}</tr>\n"

    html_title = t("report.html.title", name=speaker_name)
    html_heading = t("report.html.heading")
    html_subtitle = t("report.html.subtitle", name=speaker_name, timestamp=timestamp, ckpt_count=len(checkpoints), text_count=len(texts))
    html_comparison = t("report.html.comparison_heading")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{html_title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="c">
  <header>
    <h1>{html_heading}</h1>
    <p class="sub">{html_subtitle}</p>
  </header>
  {metrics_section}
  <section>
    <h2>{html_comparison}</h2>
    <div class="tw">
      <table>
        <thead><tr>{ths}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
  <div class="ft">Generated by HayaKoe Dev Tools</div>
</div>
<script>{_JS}</script>
</body>
</html>"""
