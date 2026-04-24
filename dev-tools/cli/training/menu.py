"""학습 파이프라인 인터랙티브 메뉴."""

import json
import math
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from cli.i18n import t
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
    table.add_column(t("training.dataset_table.col_speaker"), style="bold bright_white")
    table.add_column(t("training.dataset_table.col_utterance"), justify="right", style="bright_white")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column(t("training.dataset_table.col_status"), justify="center")

    for ds in datasets:
        if ds.all_preprocessed:
            status = t("training.dataset_table.status_ready")
        else:
            status = t("training.dataset_table.status_need_preprocess")

        table.add_row(
            ds.name,
            str(ds.utterance_count),
            str(ds.train_count),
            str(ds.val_count),
            status,
        )

    console.print(Panel(
        table,
        title=t("training.dataset_table.title"),
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

    text_status = t("training.status.done") if ds.text_preprocessed else t("training.status.not_done")
    default_status = t("training.status.done") if ds.default_style_done else t("training.status.not_done")

    lines = [
        t("training.status.audio_files", icon=_icon(True), count=ds.utterance_count),
        t("training.status.text_preprocess", icon=_icon(ds.text_preprocessed), status=text_status),
        t("training.status.bert_embedding", icon=_icon(ds.bert_done == ds.bert_total), progress=_progress(ds.bert_done, ds.bert_total)),
        t("training.status.style_vector", icon=_icon(ds.style_done == ds.style_total), progress=_progress(ds.style_done, ds.style_total)),
        t("training.status.default_style", icon=_icon(ds.default_style_done), status=default_status),
    ]

    console.print(Panel(
        "\n".join(lines),
        title=t("training.status.title", name=ds.name),
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

    table.add_row(t("training.briefing.section_hyperparams"), "")
    table.add_row(t("training.briefing.dataset"), t("training.briefing.dataset_value", train=ds.train_count, val=ds.val_count))
    table.add_row(t("training.briefing.epochs"), str(epochs))
    table.add_row(t("training.briefing.batch_size"), str(batch_size))
    table.add_row(t("training.briefing.learning_rate"), str(lr))
    table.add_row(t("training.briefing.bfloat16"), "[success]ON[/success]" if bf16 else "[muted]OFF[/muted]")
    if nproc > 1:
        table.add_row(t("training.briefing.gpu_count"), f"[success]{nproc}[/success]")
    table.add_row("", "")
    table.add_row(t("training.briefing.section_steps"), "")
    table.add_row(t("training.briefing.steps_per_epoch"), str(steps_per_epoch))
    table.add_row(t("training.briefing.total_steps"), f"{total_steps:,}")
    table.add_row(t("training.briefing.log_interval"), f"{log_interval} steps")
    table.add_row(t("training.briefing.checkpoint_interval"), f"{eval_interval} steps")
    table.add_row(t("training.briefing.estimated_saves"), t("training.briefing.estimated_saves_value", count=num_checkpoints))
    table.add_row(
        t("training.briefing.estimated_disk"),
        t("training.briefing.estimated_disk_value", gb=exports_disk_gb, count=num_checkpoints, mb=safetensors_size_mb),
    )
    table.add_row("", "")
    table.add_row(t("training.briefing.section_paths"), "")
    try:
        rel_training = training_dir.relative_to(Path.cwd())
        rel_exports = exports_dir.relative_to(Path.cwd())
    except ValueError:
        rel_training, rel_exports = training_dir, exports_dir
    table.add_row(t("training.briefing.checkpoint_path"), f"[dim]{rel_training}[/dim]")
    table.add_row(t("training.briefing.inference_model_path"), f"[dim]{rel_exports}[/dim]")

    console.print()
    console.print(Panel(
        table,
        title=t("training.briefing.title", name=ds.name),
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # eval_interval 수정 기회
    label_start = t("training.briefing.start")
    label_cancel = t("training.briefing.cancel")

    while True:
        label_change = t("training.briefing.change_interval", interval=eval_interval)
        choice = select_from_list(t("training.briefing.prompt"), [
            label_start,
            label_change,
            label_cancel,
        ])

        if choice == label_start:
            return True
        elif choice == label_cancel:
            return False
        else:
            raw = edit_value(t("training.briefing.interval_prompt"), eval_interval)
            try:
                new_interval = int(raw)
                if new_interval <= 0:
                    raise ValueError
            except ValueError:
                console.print(t("training.briefing.invalid_number"))
                continue

            train["eval_interval"] = new_interval
            config["train"] = train
            _save_config(ds, config)

            eval_interval = new_interval
            num_checkpoints = max(0, total_steps // eval_interval)
            exports_disk_gb = (num_checkpoints * safetensors_size_mb) / 1024
            console.print(t("training.briefing.interval_changed", interval=eval_interval, count=num_checkpoints, gb=exports_disk_gb))


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
        return t("training.next_step.preprocess")
    return t("training.next_step.train")


def _dataset_submenu(ds: DatasetInfo):
    """선택된 데이터셋에 대한 작업 메뉴."""
    while True:
        ds = _refresh_dataset(ds)
        console.print()
        _render_preprocessing_status(ds)
        console.print(_next_step_hint(ds))
        console.print()

        label_preprocess = t("training.submenu.preprocess_remaining")
        label_reprocess = t("training.submenu.preprocess_all")
        label_edit = t("training.submenu.edit_config")
        label_train = t("training.submenu.start_training")
        label_back = t("training.submenu.back")

        actions = [label_preprocess, label_reprocess, label_edit]
        if ds.all_preprocessed:
            actions.append(label_train)
        actions.append(label_back)

        choice = select_from_list(t("training.submenu.prompt", name=ds.name), actions)

        if choice == label_preprocess:
            from cli.training.preprocessing import run_all_preprocessing
            run_all_preprocessing(ds.path, ds.data_dir, force=False)

        elif choice == label_reprocess:
            if confirm(t("training.submenu.confirm_reprocess"), default=False):
                from cli.training.preprocessing import run_all_preprocessing
                run_all_preprocessing(ds.path, ds.data_dir, force=True)

        elif choice == label_edit:
            from cli.training.config_editor import edit_config
            edit_config(ds.data_dir)

        elif choice == label_train:
            if not ds.all_preprocessed:
                console.print(t("training.submenu.preprocess_not_done"))
                continue
            if _training_briefing(ds):
                from cli.training.runner import start_training_session
                start_training_session(ds.path, ds.data_dir)

        elif choice == label_back:
            break


# ── 메인 진입점 ───────────────────────────────────────────────

def _workflow_guide():
    return t("training.workflow_guide")


def _prompt_dataset_path() -> DatasetInfo | None:
    """경로 직접 입력으로 데이터셋 로드."""
    raw = edit_value(t("training.menu.dataset_path_prompt"), "")
    if not raw.strip():
        return None

    ds = scan_dataset(Path(raw))
    if ds is None:
        console.print(t("training.menu.dataset_not_found"))
        return None

    console.print(t("training.menu.dataset_loaded", name=ds.name, count=ds.utterance_count))
    return ds


def training_menu():
    """학습 파이프라인 메인 메뉴."""
    datasets = discover_datasets()

    if not datasets:
        console.print(Panel(
            t("training.menu.no_datasets_panel"),
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
    console.print(_workflow_guide())
    console.print()
    _render_dataset_table(datasets)
    console.print()

    # 데이터셋 선택
    label_enter_path = t("training.menu.enter_path")
    label_back = t("training.menu.back")
    choices = [ds.name for ds in datasets] + [label_enter_path, label_back]
    selected_name = select_from_list(t("training.menu.dataset_select"), choices)

    if selected_name == label_back:
        return

    if selected_name == label_enter_path:
        ds = _prompt_dataset_path()
        if ds is None:
            return
        activate_dataset(ds.path)
        _dataset_submenu(ds)
        return

    selected = next(ds for ds in datasets if ds.name == selected_name)
    activate_dataset(selected.path)
    _dataset_submenu(selected)
