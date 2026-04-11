"""학습 파이프라인 인터랙티브 메뉴."""

import json
import math
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from cli.training.dataset import DatasetInfo, discover_datasets, scan_dataset, activate_dataset
from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm, edit_value


# ── 데이터셋 목록 ──────────────────────────────────────────────

def _render_dataset_table(datasets: list[DatasetInfo]):
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        title_style="bold",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("화자", style="bold bright_white")
    table.add_column("발화", justify="right", style="bright_white")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("상태", justify="center")

    for ds in datasets:
        if ds.all_preprocessed:
            status = f"[success]● 준비 완료[/success]"
        else:
            status = f"[warning]○ 전처리 필요[/warning]"

        table.add_row(
            ds.name,
            str(ds.utterance_count),
            str(ds.train_count),
            str(ds.val_count),
            status,
        )

    console.print(Panel(
        table,
        title="[accent]데이터셋[/accent]",
        border_style="cyan",
        padding=(1, 2),
    ))


# ── 전처리 상태 ───────────────────────────────────────────────

def _render_preprocessing_status(ds: DatasetInfo):
    def _icon(done: bool) -> str:
        return "[success]✓[/success]" if done else "[muted]○[/muted]"

    def _progress(done: int, total: int) -> str:
        if total == 0:
            return "[dim]—[/dim]"
        if done == total:
            return f"[success]{done}/{total}[/success]"
        return f"[warning]{done}/{total}[/warning]"

    lines = [
        f"  {_icon(True)}  오디오 파일        [value]{ds.utterance_count}[/value]개",
        f"  {_icon(ds.text_preprocessed)}  텍스트 전처리      {'[success]완료[/success]' if ds.text_preprocessed else '[muted]미완료[/muted]'}",
        f"  {_icon(ds.bert_done == ds.bert_total)}  BERT 임베딩        {_progress(ds.bert_done, ds.bert_total)}",
        f"  {_icon(ds.style_done == ds.style_total)}  스타일 벡터        {_progress(ds.style_done, ds.style_total)}",
        f"  {_icon(ds.default_style_done)}  기본 스타일        {'[success]완료[/success]' if ds.default_style_done else '[muted]미완료[/muted]'}",
    ]

    console.print(Panel(
        "\n".join(lines),
        title=f"[accent]{ds.name}[/accent] [dim]— 전처리 상태[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))


# ── 학습 브리핑 ───────────────────────────────────────────────

def _load_config(ds: DatasetInfo) -> dict:
    with open(ds.data_dir / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _save_config(ds: DatasetInfo, config: dict):
    with open(ds.data_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _training_briefing(ds: DatasetInfo) -> bool:
    """학습 시작 전 브리핑. True 반환 시 학습 진행."""
    config = _load_config(ds)
    train = config.get("train", {})

    epochs = train.get("epochs", 100)
    batch_size = train.get("batch_size", 2)
    eval_interval = train.get("eval_interval", 1000)
    log_interval = train.get("log_interval", 200)
    lr = train.get("learning_rate", 0.0001)
    bf16 = train.get("bf16_run", False)
    nproc = train.get("nproc_per_node", 1)

    steps_per_epoch = math.ceil(ds.train_count / batch_size)
    total_steps = steps_per_epoch * epochs
    num_checkpoints = max(0, total_steps // eval_interval)

    safetensors_size_mb = 240
    exports_disk_gb = (num_checkpoints * safetensors_size_mb) / 1024

    training_dir = ds.path / "training"
    exports_dir = ds.path / "exports"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("k", style="label", min_width=18)
    table.add_column("v", style="value", justify="right")

    table.add_row("[accent]하이퍼파라미터[/accent]", "")
    table.add_row("  데이터셋", f"{ds.train_count} train / {ds.val_count} val")
    table.add_row("  에포크", str(epochs))
    table.add_row("  배치 크기", str(batch_size))
    table.add_row("  학습률", str(lr))
    table.add_row("  bfloat16", "[success]ON[/success]" if bf16 else "[muted]OFF[/muted]")
    if nproc > 1:
        table.add_row("  GPU 수", f"[success]{nproc}[/success]")
    table.add_row("", "")
    table.add_row("[accent]스텝 & 체크포인트[/accent]", "")
    table.add_row("  에포크당 스텝", str(steps_per_epoch))
    table.add_row("  총 스텝", f"{total_steps:,}")
    table.add_row("  로그 간격", f"{log_interval} steps")
    table.add_row("  체크포인트 간격", f"{eval_interval} steps")
    table.add_row("  예상 저장 횟수", f"{num_checkpoints}회")
    table.add_row(
        "  예상 디스크 사용량",
        f"~{exports_disk_gb:.1f}GB ({num_checkpoints}개 × {safetensors_size_mb}MB)",
    )
    table.add_row("", "")
    table.add_row("[accent]저장 경로[/accent]", "")
    try:
        rel_training = training_dir.relative_to(Path.cwd())
        rel_exports = exports_dir.relative_to(Path.cwd())
    except ValueError:
        rel_training, rel_exports = training_dir, exports_dir
    table.add_row("  체크포인트", f"[dim]{rel_training}[/dim]")
    table.add_row("  추론 모델", f"[dim]{rel_exports}[/dim]")

    console.print()
    console.print(Panel(
        table,
        title=f"[accent]{ds.name}[/accent] [dim]— 학습 브리핑[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # eval_interval 수정 기회
    while True:
        choice = select_from_list("진행", [
            "학습 시작",
            f"체크포인트 간격 변경 (현재: {eval_interval} steps)",
            "취소",
        ])

        if choice == "학습 시작":
            return True
        elif choice == "취소":
            return False
        else:
            raw = edit_value("체크포인트 저장 간격 (steps)", eval_interval)
            try:
                new_interval = int(raw)
                if new_interval <= 0:
                    raise ValueError
            except ValueError:
                console.print("  [error]양의 정수를 입력해주세요.[/error]")
                continue

            train["eval_interval"] = new_interval
            config["train"] = train
            _save_config(ds, config)

            eval_interval = new_interval
            num_checkpoints = max(0, total_steps // eval_interval)
            exports_disk_gb = (num_checkpoints * safetensors_size_mb) / 1024
            console.print(f"  → 체크포인트 간격: [success]{eval_interval}[/success] steps · 예상 [success]{num_checkpoints}[/success]회 · ~{exports_disk_gb:.1f}GB")


# ── 데이터셋 서브메뉴 ─────────────────────────────────────────

def _refresh_dataset(ds: DatasetInfo) -> DatasetInfo:
    """데이터셋 상태를 다시 로드."""
    datasets = discover_datasets()
    for d in datasets:
        if d.path == ds.path:
            return d
    # 자동 탐색 목록에 없으면 직접 스캔
    return scan_dataset(ds.path) or ds


def _next_step_hint(ds: DatasetInfo) -> str:
    """현재 상태에 따른 다음 단계 안내."""
    if not ds.all_preprocessed:
        return "[dim]다음 단계: [accent]전처리 실행[/accent]으로 데이터를 준비하세요.[/dim]"
    return "[dim]다음 단계: [accent]학습 설정[/accent]을 확인한 뒤 [accent]학습 시작[/accent]하세요.[/dim]"


def _dataset_submenu(ds: DatasetInfo):
    """선택된 데이터셋에 대한 작업 메뉴."""
    while True:
        ds = _refresh_dataset(ds)
        console.print()
        _render_preprocessing_status(ds)
        console.print(_next_step_hint(ds))
        console.print()

        actions = [
            "전처리 실행 (남은 단계)",
            "전처리 전체 재실행",
            "학습 설정 편집",
        ]
        if ds.all_preprocessed:
            actions.append("학습 시작")
        actions.append("뒤로")

        choice = select_from_list(f"{ds.name} — 무엇을 할까요?", actions)

        if choice == "전처리 실행 (남은 단계)":
            from cli.training.preprocessing import run_all_preprocessing
            run_all_preprocessing(ds.path, ds.data_dir, force=False)

        elif choice == "전처리 전체 재실행":
            if confirm("모든 전처리를 처음부터 다시 실행합니다. 계속할까요?", default=False):
                from cli.training.preprocessing import run_all_preprocessing
                run_all_preprocessing(ds.path, ds.data_dir, force=True)

        elif choice == "학습 설정 편집":
            from cli.training.config_editor import edit_config
            edit_config(ds.data_dir)

        elif choice == "학습 시작":
            if not ds.all_preprocessed:
                console.print("  [error]전처리가 완료되지 않았습니다.[/error]")
                continue
            if _training_briefing(ds):
                from cli.training.runner import start_training_session
                start_training_session(ds.path, ds.data_dir)

        elif choice == "뒤로":
            break


# ── 메인 진입점 ───────────────────────────────────────────────

TRAINING_WORKFLOW_GUIDE = (
    "[dim]진행 순서:  "
    "[accent]① 데이터셋 선택[/accent] → "
    "[accent]② 전처리[/accent] → "
    "[accent]③ 학습 설정[/accent] → "
    "[accent]④ 학습 시작[/accent][/dim]"
)


def _prompt_dataset_path() -> DatasetInfo | None:
    """경로 직접 입력으로 데이터셋 로드."""
    raw = edit_value("데이터셋 경로 (esd.list가 있는 폴더)", "")
    if not raw.strip():
        return None

    ds = scan_dataset(Path(raw))
    if ds is None:
        console.print("[error]해당 경로에서 esd.list를 찾을 수 없습니다.[/error]")
        return None

    console.print(f"  [success]{ds.name}[/success] 데이터셋 로드됨 ({ds.utterance_count}개 발화)")
    return ds


def training_menu():
    """학습 파이프라인 메인 메뉴."""
    datasets = discover_datasets()

    if not datasets:
        console.print(Panel(
            "[warning]data/dataset/ 에서 자동 탐색된 데이터셋이 없습니다.[/warning]\n"
            "[dim]경로를 직접 입력하여 외부 데이터셋을 사용할 수 있습니다.[/dim]",
            border_style="yellow",
            padding=(1, 2),
        ))
        console.print()
        ds = _prompt_dataset_path()
        if ds is None:
            return
        activate_dataset(ds.path)
        _dataset_submenu(ds)
        return

    console.print()
    console.print(TRAINING_WORKFLOW_GUIDE)
    console.print()
    _render_dataset_table(datasets)
    console.print()

    # 데이터셋 선택
    choices = [ds.name for ds in datasets] + ["경로 직접 입력", "뒤로"]
    selected_name = select_from_list("데이터셋 선택", choices)

    if selected_name == "뒤로":
        return

    if selected_name == "경로 직접 입력":
        ds = _prompt_dataset_path()
        if ds is None:
            return
        activate_dataset(ds.path)
        _dataset_submenu(ds)
        return

    selected = next(ds for ds in datasets if ds.name == selected_name)
    activate_dataset(selected.path)
    _dataset_submenu(selected)
