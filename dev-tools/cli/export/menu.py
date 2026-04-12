"""ONNX 내보내기 인터랙티브 메뉴.

학습된 Synthesizer 모델을 ONNX 형식으로 내보낸다.
CPU 추론 시 ONNX Runtime 백엔드로 사용된다.
(BERT는 공용 모델로 HuggingFace에 이미 Q8 ONNX가 존재하므로 내보내기 불필요)
"""

from pathlib import Path

from rich.panel import Panel

from cli.training.dataset import DatasetInfo, discover_datasets
from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm


def _find_checkpoints(ds: DatasetInfo) -> list[Path]:
    """exports 디렉토리에서 safetensors 체크포인트 목록을 반환."""
    import json

    config_path = ds.data_dir / "config.json"
    if not config_path.exists():
        return []
    with open(config_path) as f:
        model_name = json.load(f).get("model_name", "")
    if not model_name:
        return []

    exports_dir = ds.path / "exports" / model_name
    if not exports_dir.exists():
        return []
    return sorted(exports_dir.glob("*.safetensors"))


def _has_exports(ds: DatasetInfo) -> bool:
    return len(_find_checkpoints(ds)) > 0


def export_menu():
    """ONNX 내보내기 메인 메뉴."""
    console.print()
    console.print(Panel(
        "[accent]ONNX 내보내기[/accent] [dim]— CPU 추론용[/dim]\n\n"
        "[dim]학습된 Synthesizer를 ONNX로 변환합니다.\n"
        "CPU 환경에서 ONNX Runtime을 통해 추론할 때 사용됩니다.\n"
        "(BERT 모델은 HuggingFace에 이미 존재하므로 별도 내보내기 불필요)[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    datasets = discover_datasets()
    if not datasets:
        console.print("[warning]데이터셋을 찾을 수 없습니다.[/warning]")
        return

    eligible = [ds for ds in datasets if _has_exports(ds)]
    if not eligible:
        console.print(
            "[warning]추론 모델(.safetensors)이 있는 데이터셋이 없습니다.\n"
            "학습을 먼저 진행해주세요.[/warning]"
        )
        return

    # 화자 선택
    choices = [ds.name for ds in eligible] + ["뒤로"]
    selected_name = select_from_list("화자 선택", choices)
    if selected_name == "뒤로":
        return
    ds = next(d for d in eligible if d.name == selected_name)

    # 체크포인트 선택
    checkpoints = _find_checkpoints(ds)
    ckpt_choices = [ckpt.name for ckpt in checkpoints] + ["뒤로"]
    selected_ckpt_name = select_from_list("체크포인트 선택 (품질 리포트에서 확인한 최적 모델)", ckpt_choices)
    if selected_ckpt_name == "뒤로":
        return
    selected_ckpt = next(c for c in checkpoints if c.name == selected_ckpt_name)

    # 출력 경로
    output_dir = ds.path / "onnx"

    console.print()
    console.print("  [accent]내보내기 요약[/accent]")
    console.print(f"  화자:        [value]{ds.name}[/value]")
    console.print(f"  체크포인트:  [value]{selected_ckpt.name}[/value]")
    console.print(f"  출력 경로:   [dim]{output_dir}[/dim]")
    console.print(f"  용도:        [dim]CPU 추론 (ONNX Runtime)[/dim]")
    console.print()

    if not confirm("ONNX 내보내기를 시작하시겠습니까?"):
        return

    from cli.export.exporter import export_duration_predictor, export_synthesizer

    output_path = export_synthesizer(ds, selected_ckpt, output_dir)
    dp_path = export_duration_predictor(ds, selected_ckpt, output_dir)

    console.print(f"\n[success]ONNX 내보내기 완료![/success]")
    console.print(f"  [dim]{output_path}[/dim]")
    console.print(f"  [dim]{dp_path}[/dim]\n")
