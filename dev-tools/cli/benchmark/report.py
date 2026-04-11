"""벤치마크 HTML 리포트 생성."""

from __future__ import annotations

import platform
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.benchmark.runner import BenchmarkResult


_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.5}
.c{max-width:1000px;margin:0 auto;padding:2rem}
header{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #1e293b}
h1{font-size:1.5rem;font-weight:700;margin-bottom:.25rem}
.sub{color:#64748b;font-size:.875rem}
h2{font-size:1.125rem;font-weight:600;margin-bottom:1rem;color:#94a3b8}
section{margin-bottom:2.5rem}
.info{background:#1e293b;border-radius:.75rem;padding:1rem 1.25rem;border:1px solid #334155;margin-bottom:1.5rem}
.info dt{color:#64748b;font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.info dd{color:#e2e8f0;font-size:.875rem;margin-bottom:.5rem}
.tw{overflow-x:auto;border-radius:.75rem;border:1px solid #334155}
table{width:100%;border-collapse:collapse;background:#1e293b}
th{background:#0f172a;padding:.75rem;text-align:center;font-size:.75rem;color:#64748b;font-weight:600;white-space:nowrap}
td{padding:.75rem;border-top:1px solid #0f172a;text-align:center;vertical-align:middle;font-size:.875rem}
tr:hover td{background:#1e293b99}
.tl{text-align:left!important}
.good{color:#22c55e;font-weight:700}
.ok{color:#eab308;font-weight:700}
.slow{color:#ef4444;font-weight:700}
.bar-bg{background:#334155;border-radius:4px;height:20px;position:relative;min-width:60px}
.bar-fg{border-radius:4px;height:100%;position:absolute;left:0;top:0}
.bar-label{position:absolute;right:6px;top:1px;font-size:11px;font-weight:600;color:#e2e8f0}
.note{background:#1e293b;border-radius:.75rem;padding:1rem 1.25rem;border:1px solid #334155;margin-bottom:1.5rem;font-size:.85rem;color:#94a3b8;line-height:1.7}
.note strong{color:#e2e8f0}
.note .ex{color:#64748b;font-size:.8rem}
.legend{display:inline-flex;align-items:center;gap:.35rem;margin-right:1rem;font-size:.8rem}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.ft{text-align:center;color:#334155;font-size:.75rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b}
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
            bar_color = "#22c55e"
        elif cls == "ok":
            bar_color = "#eab308"
        else:
            bar_color = "#ef4444"

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
      <span class="legend"><span class="legend-dot" style="background:#22c55e"></span> 10x 이상 (매우 빠름)</span>
      <span class="legend"><span class="legend-dot" style="background:#eab308"></span> 1x~10x (실시간 이상)</span>
      <span class="legend"><span class="legend-dot" style="background:#ef4444"></span> 1x 미만 (실시간 이하)</span>
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
    <p style="color:#64748b;font-size:.8rem;margin-bottom:.75rem">
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
