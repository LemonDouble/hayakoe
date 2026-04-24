"""품질 리포트 생성기 — 추론 + 지표 수집 + HTML 빌드."""

import base64
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from cli.i18n import t
from cli.ui.console import console


# ── 체크포인트 파싱 ────────────────────────────────────────────


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


# ── TensorBoard 지표 ──────────────────────────────────────────


def _read_metrics(training_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    if not training_dir.exists():
        return {}

    metrics: dict[str, list[tuple[int, float]]] = {}

    for log_dir in [training_dir, training_dir / "eval"]:
        if not log_dir.exists():
            continue
        try:
            ea = EventAccumulator(str(log_dir))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                metrics[tag] = [(e.step, e.value) for e in events]
        except Exception:
            continue

    return metrics


# ── 오디오 인코딩 ─────────────────────────────────────────────


def _audio_to_data_uri(sr: int, audio: np.ndarray) -> str:
    buf = io.BytesIO()
    audio_f = audio.astype(np.float32)
    if audio.dtype == np.int16:
        audio_f = audio_f / 32768.0
    sf.write(buf, audio_f, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


# ── SVG 차트 ──────────────────────────────────────────────────


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


# ── HTML 빌드 ─────────────────────────────────────────────────

_CSS = """\
@import url('https://cdn.jsdelivr.net/npm/galmuri/dist/galmuri.css');
@import url('https://cdnjs.cloudflare.com/ajax/libs/pretendard/1.3.9/static/pretendard.min.css');
:root{
  --color-primary:#F0B90B;--color-secondary:#CD6B5E;
  --color-bg-dark:#12100E;--color-surface:#1C1816;--color-surface-hover:#231E1B;
  --color-border:#2E2723;--color-border-hover:#3D3530;
  --color-text-primary:#F5F0EB;--color-text-secondary:#A89E95;--color-text-muted:#5C524A;
  --color-primary-dim:rgba(240,185,11,0.12);--color-primary-dim-border:rgba(240,185,11,0.25);
  --font-heading:'Galmuri11',monospace;
  --font-body:'Pretendard',-apple-system,BlinkMacSystemFont,sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--color-bg-dark);color:var(--color-text-secondary);font-family:var(--font-body);line-height:1.6;font-size:15px}
.c{max-width:1400px;margin:0 auto;padding:32px 24px}
header{margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid var(--color-border)}
h1{font-family:var(--font-heading);font-size:28px;font-weight:700;line-height:1.3;color:var(--color-text-primary);margin-bottom:8px}
.sub{color:var(--color-text-muted);font-size:13px}
h2{font-family:var(--font-heading);font-size:20px;font-weight:700;line-height:1.4;color:var(--color-text-primary);margin-bottom:16px}
section{margin-bottom:40px}
.mg{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.mc{background:var(--color-surface);border:1px solid var(--color-border);border-radius:12px;padding:16px 20px;transition:border-color 0.2s}
.mc:hover{border-color:var(--color-primary-dim-border)}
.mt{color:var(--color-text-secondary);font-size:13px;font-weight:600;margin-bottom:4px}
.mv{color:var(--color-text-primary);font-weight:700}
.ms{color:var(--color-text-muted);font-size:11px;line-height:1.4;margin-bottom:8px}
.mn{color:var(--color-text-muted);font-size:12px}
.tw{overflow-x:auto;border:1px solid var(--color-border);border-radius:12px;background:var(--color-surface)}
table{width:100%;border-collapse:collapse}
thead{background:var(--color-bg-dark)}
th{padding:12px 16px;font-family:var(--font-body);font-size:13px;font-weight:600;color:var(--color-text-secondary);text-align:center;border-bottom:1px solid var(--color-border);white-space:nowrap;position:sticky;top:0;z-index:1;background:var(--color-bg-dark)}
td{padding:12px 16px;font-size:13px;color:var(--color-text-primary);border-bottom:1px solid var(--color-border);text-align:center;vertical-align:middle}
.tc{text-align:left!important;max-width:320px;color:var(--color-text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.err{color:var(--color-text-muted);font-size:12px}
tbody tr:hover td{background:var(--color-surface-hover)}
tbody tr:last-child td{border-bottom:none}
audio{width:200px;height:32px}
.ft{text-align:center;color:var(--color-text-muted);font-size:12px;margin-top:32px;padding-top:16px;border-top:1px solid var(--color-border)}
"""

_JS = """\
document.querySelectorAll('audio').forEach(a=>{
  a.addEventListener('play',()=>{
    document.querySelectorAll('audio').forEach(o=>{if(o!==a)o.pause()});
  });
});
"""


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


# ── 리포트 생성 ───────────────────────────────────────────────


def generate_report(
    dataset_path: Path,
    checkpoint_paths: list[Path],
    texts: list[str],
) -> Path:
    """체크포인트별 추론 → 비교 HTML 리포트 생성.

    Returns:
        생성된 HTML 파일 경로.
    """
    import torch
    from hayakoe.tts_model import TTSModel

    checkpoints = [_parse_checkpoint(p) for p in checkpoint_paths]

    # 설정 파일 — 데이터셋 루트 config에서 model_name만 읽고,
    # 추론에는 exports 디렉토리의 config.json을 사용 (num_styles 등이 보정됨)
    root_config_path = dataset_path / "config.json"
    with open(root_config_path) as f:
        root_config = json.load(f)
    model_name = root_config.get("model_name", dataset_path.name)
    speaker_name = dataset_path.name

    exports_dir = dataset_path / "exports" / model_name
    config_path = exports_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(t("report.generator.config_not_found", path=config_path))

    style_vec_path = exports_dir / "style_vectors.npy"
    if not style_vec_path.exists():
        raise FileNotFoundError(t("report.generator.style_not_found", path=style_vec_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 학습 지표
    training_dir = dataset_path / "training"
    console.print(t("report.generator.reading_metrics"))
    metrics = _read_metrics(training_dir)

    # 체크포인트별 추론
    total = len(checkpoints) * len(texts)
    audio_matrix: dict[str, dict[str, str]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("report.generator.generating_audio"), total=total)

        for ckpt in checkpoints:
            progress.update(task, description=t("report.generator.model_loading", label=ckpt.label))

            model = TTSModel(
                model_path=ckpt.path,
                config_path=config_path,
                style_vec_path=style_vec_path,
                device=device,
            )
            model.load()

            audio_matrix[ckpt.label] = {}

            for text in texts:
                progress.update(task, description=t("report.generator.generating", label=ckpt.label))
                try:
                    sr, audio = model.infer(text)
                    audio_matrix[ckpt.label][text] = _audio_to_data_uri(sr, audio)
                except Exception as e:
                    console.print(t("report.generator.error", label=ckpt.label, error=e))
                    audio_matrix[ckpt.label][text] = ""
                progress.advance(task)

            model.unload()

    # HTML 생성
    console.print(t("report.generator.building_html"))
    html = _build_html(speaker_name, checkpoints, texts, audio_matrix, metrics)

    reports_dir = dataset_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"report_{ts}.html"
    output_path.write_text(html, encoding="utf-8")

    return output_path
