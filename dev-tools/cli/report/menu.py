"""품질 리포트 생성 메뉴."""

import json
from pathlib import Path

from cli.training.dataset import DatasetInfo, discover_datasets
from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm


SAMPLE_TEXTS = {
    "짧음": [
        "おはようございます。今日もよろしくお願いします。",
        "えっ、本当ですか？それはすごいですね！",
        "静かな夜に、星が綺麗に見えます。",
    ],
    "중간": [
        "先週の土曜日、家族で動物園に行きました。子供たちはパンダを見てとても喜んでいました。天気も良くて、最高の一日になりました。",
        "音声合成の技術は年々進化しています。最近では人間の声と区別がつかないほど自然な音声を生成できるようになりました。今後の発展が楽しみです。",
    ],
    "김": [
        "春が来ると、日本中で桜が咲き始めます。人々は公園や川沿いに集まって、お花見を楽しみます。"
        "友人や家族と一緒にお弁当を広げ、美しい花びらが舞い散る様子を眺めるのは、日本の春の風物詩です。"
        "桜の季節は短く、わずか一週間ほどで散ってしまいますが、その儚さがまた人々の心を惹きつけるのかもしれません。",
    ],
}


def _get_model_name(dataset_path: Path) -> str:
    config_path = dataset_path / "config.json"
    if not config_path.exists():
        return ""
    with open(config_path) as f:
        return json.load(f).get("model_name", "")


def _find_checkpoints(dataset_path: Path) -> list[Path]:
    model_name = _get_model_name(dataset_path)
    if not model_name:
        return []
    exports_dir = dataset_path / "exports" / model_name
    if not exports_dir.exists():
        return []
    return sorted(exports_dir.glob("*.safetensors"))


def _has_exports(ds: DatasetInfo) -> bool:
    return len(_find_checkpoints(ds.path)) > 0


def _select_checkpoints(all_ckpts: list[Path], max_count: int = 8) -> list[Path]:
    """체크포인트가 너무 많으면 균등 샘플링."""
    if len(all_ckpts) <= max_count:
        return all_ckpts
    indices = [0]
    step = (len(all_ckpts) - 1) / (max_count - 1)
    for i in range(1, max_count - 1):
        indices.append(round(i * step))
    indices.append(len(all_ckpts) - 1)
    return [all_ckpts[i] for i in sorted(set(indices))]


def _input_custom_texts() -> list[str]:
    texts: list[str] = []
    console.print("  [dim]텍스트를 한 줄씩 입력하세요. 빈 줄을 입력하면 종료됩니다.[/dim]")
    while True:
        try:
            text = input(f"  [{len(texts) + 1}] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        texts.append(text)
    return texts


def _open_report(path: Path):
    """WSL2 환경에서 브라우저로 리포트 열기."""
    import subprocess

    try:
        win_path = subprocess.check_output(
            ["wslpath", "-w", str(path)], stderr=subprocess.DEVNULL
        ).decode().strip()
        subprocess.Popen(
            ["powershell.exe", "-Command", f"Start-Process '{win_path}'"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def report_menu():
    """품질 리포트 메인 메뉴."""
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

    # 데이터셋(화자) 선택
    choices = [ds.name for ds in eligible] + ["뒤로"]
    selected_name = select_from_list("화자 선택", choices)
    if selected_name == "뒤로":
        return
    ds = next(d for d in eligible if d.name == selected_name)

    # 체크포인트
    all_ckpts = _find_checkpoints(ds.path)
    checkpoints = _select_checkpoints(all_ckpts)

    console.print(f"\n  [dim]체크포인트 {len(checkpoints)}개 사용[/dim]")
    for ckpt in checkpoints:
        console.print(f"    [dim]· {ckpt.stem}[/dim]")
    console.print()

    # 텍스트 선택
    text_choice = select_from_list("텍스트 선택", [
        "샘플 - 짧음 (3개)",
        "샘플 - 중간 (2개)",
        "샘플 - 김 (1개)",
        "샘플 - 전체 (6개)",
        "직접 입력",
        "뒤로",
    ])

    if text_choice == "뒤로":
        return

    if text_choice == "직접 입력":
        texts = _input_custom_texts()
        if not texts:
            console.print("  [warning]텍스트가 입력되지 않았습니다.[/warning]")
            return
    elif "짧음" in text_choice:
        texts = SAMPLE_TEXTS["짧음"]
    elif "중간" in text_choice:
        texts = SAMPLE_TEXTS["중간"]
    elif "김" in text_choice:
        texts = SAMPLE_TEXTS["김"]
    else:
        texts = SAMPLE_TEXTS["짧음"] + SAMPLE_TEXTS["중간"] + SAMPLE_TEXTS["김"]

    # 요약
    audio_count = len(checkpoints) * len(texts)
    console.print()
    console.print("  [accent]리포트 생성 요약[/accent]")
    console.print(f"  화자:        [value]{ds.name}[/value]")
    console.print(f"  체크포인트:  [value]{len(checkpoints)}개[/value]")
    console.print(f"  텍스트:      [value]{len(texts)}개[/value]")
    console.print(f"  생성 오디오: [value]{audio_count}개[/value]")
    console.print()

    if not confirm("리포트를 생성하시겠습니까?"):
        return

    from cli.report.generator import generate_report

    output_path = generate_report(ds.path, checkpoints, texts)

    console.print(f"\n[success]리포트 생성 완료![/success]")
    console.print(f"  [dim]{output_path}[/dim]\n")

    if confirm("브라우저에서 열시겠습니까?", default=True):
        _open_report(output_path)
