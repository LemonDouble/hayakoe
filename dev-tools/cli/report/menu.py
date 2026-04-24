"""품질 리포트 생성 메뉴."""

import json
from pathlib import Path

from cli.i18n import t
from cli.training.dataset import DatasetInfo, discover_datasets
from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm


def _sample_keys():
    return {
        "short": t("report.menu.sample_label_short"),
        "medium": t("report.menu.sample_label_medium"),
        "long": t("report.menu.sample_label_long"),
    }


SAMPLE_TEXTS_RAW = {
    "short": [
        "おはようございます。今日もよろしくお願いします。",
        "えっ、本当ですか？それはすごいですね！",
        "静かな夜に、星が綺麗に見えます。",
    ],
    "medium": [
        "先週の土曜日、家族で動物園に行きました。子供たちはパンダを見てとても喜んでいました。天気も良くて、最高の一日になりました。",
        "音声合成の技術は年々進化しています。最近では人間の声と区別がつかないほど自然な音声を生成できるようになりました。今後の発展が楽しみです。",
    ],
    "long": [
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
    console.print(t("report.menu.custom_text_hint"))
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
        console.print(t("report.menu.no_datasets"))
        return

    eligible = [ds for ds in datasets if _has_exports(ds)]
    if not eligible:
        console.print(t("report.menu.no_exports"))
        return

    # 데이터셋(화자) 선택
    label_back = t("report.menu.back")
    choices = [ds.name for ds in eligible] + [label_back]
    selected_name = select_from_list(t("report.menu.speaker_select"), choices)
    if selected_name == label_back:
        return
    ds = next(d for d in eligible if d.name == selected_name)

    # 체크포인트
    all_ckpts = _find_checkpoints(ds.path)
    checkpoints = _select_checkpoints(all_ckpts)

    console.print(t("report.menu.checkpoint_count", count=len(checkpoints)))
    for ckpt in checkpoints:
        console.print(f"    [dim]· {ckpt.stem}[/dim]")
    console.print()

    # 텍스트 선택
    label_short = t("report.menu.text_short")
    label_medium = t("report.menu.text_medium")
    label_long = t("report.menu.text_long")
    label_all = t("report.menu.text_all")
    label_custom = t("report.menu.text_custom")

    text_choice = select_from_list(t("report.menu.text_select"), [
        label_short,
        label_medium,
        label_long,
        label_all,
        label_custom,
        label_back,
    ])

    if text_choice == label_back:
        return

    if text_choice == label_custom:
        texts = _input_custom_texts()
        if not texts:
            console.print(t("report.menu.no_text_input"))
            return
    elif text_choice == label_short:
        texts = SAMPLE_TEXTS_RAW["short"]
    elif text_choice == label_medium:
        texts = SAMPLE_TEXTS_RAW["medium"]
    elif text_choice == label_long:
        texts = SAMPLE_TEXTS_RAW["long"]
    else:
        texts = SAMPLE_TEXTS_RAW["short"] + SAMPLE_TEXTS_RAW["medium"] + SAMPLE_TEXTS_RAW["long"]

    # 요약
    audio_count = len(checkpoints) * len(texts)
    console.print()
    console.print(t("report.menu.summary_title"))
    console.print(t("report.menu.summary_speaker", name=ds.name))
    console.print(t("report.menu.summary_checkpoints", count=len(checkpoints)))
    console.print(t("report.menu.summary_texts", count=len(texts)))
    console.print(t("report.menu.summary_audio", count=audio_count))
    console.print()

    if not confirm(t("report.menu.confirm_generate")):
        return

    from cli.report.generator import generate_report

    output_path = generate_report(ds.path, checkpoints, texts)

    console.print(t("report.menu.complete"))
    console.print(f"  [dim]{output_path}[/dim]\n")

    if confirm(t("report.menu.open_browser"), default=True):
        _open_report(output_path)
