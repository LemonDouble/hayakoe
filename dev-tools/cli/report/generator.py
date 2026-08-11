"""품질 리포트 생성기 — 추론 + 지표 수집.

HTML / SVG 렌더링은 :mod:`cli.report.html` 에 있다.
"""

import base64
import io
import json
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
from cli.report.html import _build_html, _parse_checkpoint
from cli.ui.console import console


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


def _audio_to_data_uri(sr: int, audio: np.ndarray) -> str:
    buf = io.BytesIO()
    audio_f = audio.astype(np.float32)
    if audio.dtype == np.int16:
        audio_f = audio_f / 32768.0
    sf.write(buf, audio_f, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def _synthesize(net_g, hps, style_vec, text, device) -> tuple[int, np.ndarray]:
    """단일 체크포인트 모델로 텍스트 합성 → (sr, int16 오디오).

    구 TTSModel.infer 의 기본 동작(줄바꿈 분할 + 0.5초 무음 연결 +
    피크 정규화 16-bit 변환)을 그대로 이식한 것.
    """
    import torch

    from hayakoe.models.infer import infer

    parts = [s for s in text.split("\n") if s != ""]
    audios = []
    with torch.no_grad():
        for i, part in enumerate(parts):
            audios.append(
                infer(  # (audio, phone_ids, durations) 중 오디오만 쓴다
                    text=part,
                    style_vec=style_vec,
                    sdp_ratio=0.2,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                    sid=0,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                )[0]
            )
            if i != len(parts) - 1:
                audios.append(np.zeros(int(44100 * 0.5)))
    audio = np.concatenate(audios)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return hps.data.sampling_rate, (audio * 32767).astype(np.int16)


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

    from hayakoe.models.hyper_parameters import HyperParameters
    from hayakoe.models.infer import get_net_g

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

    hps = HyperParameters.load_from_json(config_path)
    style_vectors = np.load(style_vec_path)
    # 구 TTSModel 기본 동작과 동일: Neutral 스타일, weight 1.0
    style2id = getattr(hps.data, "style2id", {"Neutral": 0})
    mean = style_vectors[0]
    vec = style_vectors[style2id.get("Neutral", 0)]
    style_vec = mean + (vec - mean) * 1.0

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

            net_g = get_net_g(str(ckpt.path), hps.version, device, hps)

            audio_matrix[ckpt.label] = {}

            for text in texts:
                progress.update(task, description=t("report.generator.generating", label=ckpt.label))
                try:
                    sr, audio = _synthesize(net_g, hps, style_vec, text, device)
                    audio_matrix[ckpt.label][text] = _audio_to_data_uri(sr, audio)
                except Exception as e:
                    console.print(t("report.generator.error", label=ckpt.label, error=e))
                    audio_matrix[ckpt.label][text] = ""
                progress.advance(task)

            # 체크포인트별 로드/해제로 VRAM 사용을 일정하게 유지
            del net_g
            if device == "cuda":
                torch.cuda.empty_cache()

    # HTML 생성
    console.print(t("report.generator.building_html"))
    html = _build_html(speaker_name, checkpoints, texts, audio_matrix, metrics)

    reports_dir = dataset_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"report_{ts}.html"
    output_path.write_text(html, encoding="utf-8")

    return output_path
