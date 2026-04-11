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
    width: int = 500,
    height: int = 150,
    color: str = "#3b82f6",
) -> str:
    if not data or len(data) < 2:
        return (
            f'<div class="mc"><div class="mt">{title}</div>'
            '<p style="color:#475569;font-size:12px">데이터 없음</p></div>'
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
        grid += f'<line x1="{ml}" y1="{gy:.0f}" x2="{ml + pw}" y2="{gy:.0f}" stroke="#1e293b" stroke-width="1"/>'

    return f'''<div class="mc">
  <div class="mt">{title} <span style="color:#e2e8f0;font-weight:600">{last_val:.4f}</span></div>
  <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">
    {grid}
    <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>
    <text x="{ml}" y="{height - 1}" fill="#475569" font-size="9" font-family="monospace">{min_s}</text>
    <text x="{width - mr}" y="{height - 1}" fill="#475569" font-size="9" font-family="monospace" text-anchor="end">{max_s}</text>
  </svg>
</div>'''


# ── HTML 빌드 ─────────────────────────────────────────────────

_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.5}
.c{max-width:1400px;margin:0 auto;padding:2rem}
header{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #1e293b}
h1{font-size:1.5rem;font-weight:700;margin-bottom:.25rem}
.sub{color:#64748b;font-size:.875rem}
h2{font-size:1.125rem;font-weight:600;margin-bottom:1rem;color:#94a3b8}
section{margin-bottom:2.5rem}
.mg{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.75rem}
.mc{background:#1e293b;border-radius:.75rem;padding:.75rem 1rem;border:1px solid #334155}
.mt{color:#94a3b8;font-size:.75rem;margin-bottom:.25rem}
.tw{overflow-x:auto;border-radius:.75rem;border:1px solid #334155}
table{width:100%;border-collapse:collapse;background:#1e293b}
th{background:#0f172a;padding:.75rem;text-align:center;font-size:.75rem;color:#64748b;font-weight:600;white-space:nowrap;position:sticky;top:0;z-index:1}
td{padding:.75rem;border-top:1px solid #0f172a;text-align:center;vertical-align:middle}
.tc{text-align:left!important;max-width:320px;font-size:.8rem;color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
tr:hover td{background:#1e293b99}
audio{width:180px;height:32px}
.ft{text-align:center;color:#334155;font-size:.75rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b}
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

    # 차트
    chart_specs = [
        ("loss/g/total", "Generator Loss", "#3b82f6"),
        ("loss/g/mel", "Mel Loss", "#10b981"),
        ("loss/d/total", "Discriminator Loss", "#f59e0b"),
        ("loss/g/kl", "KL Loss", "#8b5cf6"),
        ("loss/g/dur", "Duration Loss", "#ec4899"),
        ("loss/g/fm", "Feature Matching Loss", "#06b6d4"),
    ]

    charts = ""
    for tag, title, color in chart_specs:
        if tag in metrics and len(metrics[tag]) >= 2:
            charts += _svg_chart(metrics[tag], title, color=color)

    metrics_section = ""
    if charts:
        metrics_section = f"<section><h2>학습 지표</h2><div class='mg'>{charts}</div></section>"

    # 테이블
    ths = '<th class="tc">텍스트</th>'
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
                tds += '<td style="color:#64748b;font-size:11px">오류</td>'
        rows += f"<tr>{tds}</tr>\n"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HayaKoe 품질 리포트 — {speaker_name}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="c">
  <header>
    <h1>HayaKoe 품질 리포트</h1>
    <p class="sub">{speaker_name} &mdash; {timestamp} &mdash; 체크포인트 {len(checkpoints)}개 &times; 텍스트 {len(texts)}개</p>
  </header>
  {metrics_section}
  <section>
    <h2>음성 비교</h2>
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
        raise FileNotFoundError(f"추론용 config.json을 찾을 수 없습니다: {config_path}")

    style_vec_path = exports_dir / "style_vectors.npy"
    if not style_vec_path.exists():
        raise FileNotFoundError(f"style_vectors.npy를 찾을 수 없습니다: {style_vec_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 학습 지표
    training_dir = dataset_path / "training"
    console.print("  [dim]학습 지표 읽는 중...[/dim]")
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
        task = progress.add_task("음성 생성 중", total=total)

        for ckpt in checkpoints:
            progress.update(task, description=f"[dim]{ckpt.label}[/dim] 모델 로딩...")

            model = TTSModel(
                model_path=ckpt.path,
                config_path=config_path,
                style_vec_path=style_vec_path,
                device=device,
            )
            model.load()

            audio_matrix[ckpt.label] = {}

            for text in texts:
                progress.update(task, description=f"[dim]{ckpt.label}[/dim] 생성 중...")
                try:
                    sr, audio = model.infer(text)
                    audio_matrix[ckpt.label][text] = _audio_to_data_uri(sr, audio)
                except Exception as e:
                    console.print(f"  [error]{ckpt.label} 오류: {e}[/error]")
                    audio_matrix[ckpt.label][text] = ""
                progress.advance(task)

            model.unload()

    # HTML 생성
    console.print("  [dim]HTML 생성 중...[/dim]")
    html = _build_html(speaker_name, checkpoints, texts, audio_matrix, metrics)

    reports_dir = dataset_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"report_{ts}.html"
    output_path.write_text(html, encoding="utf-8")

    return output_path
