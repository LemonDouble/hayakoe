"""벤치마크 HTML 리포트 생성."""

from __future__ import annotations

import platform
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.benchmark.runner import BenchmarkResult


_CSS = """\
@import url('https://cdn.jsdelivr.net/npm/galmuri/dist/galmuri.css');
@import url('https://cdnjs.cloudflare.com/ajax/libs/pretendard/1.3.9/static/pretendard.min.css');
:root{
  --color-primary:#F0B90B;--color-secondary:#CD6B5E;
  --color-bg-dark:#12100E;--color-surface:#1C1816;--color-surface-hover:#231E1B;
  --color-border:#2E2723;--color-border-hover:#3D3530;
  --color-text-primary:#F5F0EB;--color-text-secondary:#A89E95;--color-text-muted:#5C524A;
  --color-primary-dim:rgba(240,185,11,0.12);--color-primary-dim-border:rgba(240,185,11,0.25);
  --color-good:#4ADE80;--color-ok:#F0B90B;--color-slow:#CD6B5E;
  --font-heading:'Galmuri11',monospace;
  --font-body:'Pretendard',-apple-system,BlinkMacSystemFont,sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--color-bg-dark);color:var(--color-text-secondary);font-family:var(--font-body);line-height:1.6;font-size:15px}
.c{max-width:1000px;margin:0 auto;padding:32px 24px}
header{margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid var(--color-border)}
h1{font-family:var(--font-heading);font-size:28px;font-weight:700;line-height:1.3;color:var(--color-text-primary);margin-bottom:8px}
.sub{color:var(--color-text-muted);font-size:13px}
h2{font-family:var(--font-heading);font-size:20px;font-weight:700;line-height:1.4;color:var(--color-text-primary);margin-bottom:16px}
section{margin-bottom:40px}
.info{background:var(--color-surface);border-radius:12px;padding:16px 20px;border:1px solid var(--color-border);margin-bottom:24px}
.info dt{color:var(--color-text-muted);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.info dd{color:var(--color-text-primary);font-size:13px;margin-bottom:8px}
.tw{overflow-x:auto;border-radius:12px;border:1px solid var(--color-border);background:var(--color-surface)}
table{width:100%;border-collapse:collapse}
thead{background:var(--color-bg-dark)}
th{padding:12px 16px;font-family:var(--font-body);font-size:13px;font-weight:600;color:var(--color-text-secondary);text-align:center;border-bottom:1px solid var(--color-border);white-space:nowrap;position:sticky;top:0;z-index:1;background:var(--color-bg-dark)}
td{padding:12px 16px;font-size:13px;color:var(--color-text-primary);border-bottom:1px solid var(--color-border);text-align:center;vertical-align:middle}
tbody tr:hover td{background:var(--color-surface-hover)}
tbody tr:last-child td{border-bottom:none}
.tl{text-align:left!important}
.good{color:var(--color-good);font-weight:700}
.ok{color:var(--color-ok);font-weight:700}
.slow{color:var(--color-slow);font-weight:700}
.bar-bg{background:var(--color-border);border-radius:4px;height:20px;position:relative;min-width:60px}
.bar-fg{border-radius:4px;height:100%;position:absolute;left:0;top:0}
.bar-label{position:absolute;right:6px;top:1px;font-size:11px;font-weight:600;color:var(--color-text-primary)}
.note{background:var(--color-surface);border-radius:12px;padding:16px 20px;border:1px solid var(--color-border);margin-bottom:24px;font-size:13px;color:var(--color-text-secondary);line-height:1.7}
.note strong{color:var(--color-text-primary)}
.note .ex{color:var(--color-text-muted);font-size:12px}
.legend{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:12px}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.ft{text-align:center;color:var(--color-text-muted);font-size:12px;margin-top:32px;padding-top:16px;border-top:1px solid var(--color-border)}
"""


def _speed_class(speed: float) -> str:
    if speed >= 10:
        return "good"
    if speed >= 1:
        return "ok"
    return "slow"


def _get_system_info(devices: list[str]) -> dict[str, str]:
    """시스템 정보를 수집한다."""
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor() or platform.machine(),
        "Python": platform.python_version(),
    }

    try:
        import onnxruntime as ort
        info["ONNX Runtime"] = ort.__version__
    except ImportError:
        pass

    if "cuda" in devices:
        try:
            import torch
            info["PyTorch"] = torch.__version__
            if torch.cuda.is_available():
                info["GPU"] = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                info["VRAM"] = f"{mem:.1f} GB"
        except ImportError:
            pass

    return info


def build_benchmark_html(results: list[BenchmarkResult]) -> str:
    """벤치마크 결과를 HTML 리포트로 변환한다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    devices = list({r.device for r in results})

    # 시스템 정보
    sys_info = _get_system_info(devices)
    info_items = ""
    for k, v in sys_info.items():
        info_items += f"<dt>{k}</dt><dd>{v}</dd>"

    # 배속 바 차트의 최대값
    max_speed = max((r.speed for r in results), default=1)

    # 결과 테이블
    rows = ""
    for r in results:
        backend_label = f"{r.backend} ({r.device.upper()})"
        cls = _speed_class(r.speed)
        bar_pct = min(100, r.speed / max_speed * 100)

        if cls == "good":
            bar_color = "#4ADE80"
        elif cls == "ok":
            bar_color = "#F0B90B"
        else:
            bar_color = "#CD6B5E"

        bar_html = (
            f'<div class="bar-bg">'
            f'<div class="bar-fg" style="width:{bar_pct:.0f}%;background:{bar_color}"></div>'
            f'<div class="bar-label">{r.speed:.1f}x</div>'
            f'</div>'
        )

        rows += f"""<tr>
  <td class="tl">{backend_label}</td>
  <td>{r.text_label}</td>
  <td>{r.avg_time:.3f}s</td>
  <td>{r.audio_duration:.1f}s</td>
  <td>{bar_html}</td>
  <td class="{cls}">{r.speed:.1f}x</td>
</tr>
"""

    # 디바이스별 요약
    summary_rows = ""
    for device in sorted(devices):
        device_results = [r for r in results if r.device == device]
        backend = device_results[0].backend
        label = f"{backend} ({device.upper()})"

        avg_speed = sum(r.speed for r in device_results) / len(device_results)
        min_time = min(r.avg_time for r in device_results)
        max_time = max(r.avg_time for r in device_results)
        cls = _speed_class(avg_speed)

        summary_rows += f"""<tr>
  <td class="tl">{label}</td>
  <td>{min_time:.3f}s ~ {max_time:.3f}s</td>
  <td class="{cls}">{avg_speed:.1f}x</td>
</tr>
"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HayaKoe 벤치마크 리포트</title>
<style>{_CSS}</style>
</head>
<body>
<div class="c">
  <header>
    <h1>HayaKoe 벤치마크 리포트</h1>
    <p class="sub">{timestamp}</p>
  </header>

  <section>
    <h2>시스템 정보</h2>
    <dl class="info">{info_items}</dl>
  </section>

  <section>
    <h2>읽는 법</h2>
    <div class="note">
      <strong>배속</strong>이란 실시간 대비 음성 생성 속도입니다.<br>
      오디오 길이를 추론 시간으로 나눈 값으로, <strong>높을수록 빠릅니다.</strong><br>
      <span class="ex">예: 10.0x = 10초 분량의 음성을 1초 만에 생성</span><br><br>
      <strong>백엔드</strong>는 추론에 사용하는 엔진입니다.<br>
      · <strong>ONNX (CPU)</strong> — GPU 없이 동작합니다. 서버/로컬 배포에 적합합니다.<br>
      · <strong>PyTorch (CUDA)</strong> — NVIDIA GPU를 사용합니다. 고속 추론에 적합합니다.<br><br>
      <span class="legend"><span class="legend-dot" style="background:#4ADE80"></span> 10x 이상 (매우 빠름)</span>
      <span class="legend"><span class="legend-dot" style="background:#F0B90B"></span> 1x~10x (실시간 이상)</span>
      <span class="legend"><span class="legend-dot" style="background:#CD6B5E"></span> 1x 미만 (실시간 이하)</span>
    </div>
  </section>

  <section>
    <h2>요약</h2>
    <div class="tw">
      <table>
        <thead><tr>
          <th class="tl">백엔드</th>
          <th>추론 시간 범위</th>
          <th>평균 배속</th>
        </tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>상세 결과</h2>
    <p style="color:var(--color-text-muted);font-size:12px;margin-bottom:12px">
      짧은/중간/긴 텍스트를 각각 5회 추론한 평균값입니다. (워밍업 2회 제외)
    </p>
    <div class="tw">
      <table>
        <thead><tr>
          <th class="tl">백엔드</th>
          <th>텍스트</th>
          <th>추론 시간</th>
          <th>오디오 길이</th>
          <th style="min-width:120px">배속</th>
          <th></th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>

  <div class="ft">Generated by HayaKoe Dev Tools</div>
</div>
</body>
</html>"""
