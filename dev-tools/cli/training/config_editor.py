"""config.json 인터랙티브 편집기."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from InquirerPy import inquirer
from InquirerPy.separator import Separator
from rich.panel import Panel
from rich.table import Table

from cli.i18n import t
from cli.ui.console import console
from cli.ui.prompts import edit_value


GROUP_BASIC = "basic"
GROUP_ADVANCED = "advanced"


_MISSING = object()


@dataclass
class ParamDef:
    key: str
    type: type
    i18n_key: str
    group: str
    default: Any = None

    @property
    def label(self) -> str:
        return t(f"training.config.param.{self.i18n_key}.label")

    @property
    def help(self) -> str:
        return t(f"training.config.param.{self.i18n_key}.help")


EDITABLE_PARAMS: list[ParamDef] = [
    # ── 기본 설정 ──────────────────────────────────────────────
    ParamDef(key="train.epochs", type=int, i18n_key="epochs", group=GROUP_BASIC),
    ParamDef(key="train.batch_size", type=int, i18n_key="batch_size", group=GROUP_BASIC),
    ParamDef(key="train.nproc_per_node", type=int, i18n_key="nproc_per_node", group=GROUP_BASIC, default=1),
    ParamDef(key="train.bf16_run", type=bool, i18n_key="bf16_run", group=GROUP_BASIC, default=False),
    ParamDef(key="train.eval_interval", type=int, i18n_key="eval_interval", group=GROUP_BASIC),
    ParamDef(key="train.log_interval", type=int, i18n_key="log_interval", group=GROUP_BASIC),
    # ── 고급 설정 ──────────────────────────────────────────────
    ParamDef(key="train.learning_rate", type=float, i18n_key="learning_rate", group=GROUP_ADVANCED),
    ParamDef(key="train.warmup_epochs", type=int, i18n_key="warmup_epochs", group=GROUP_ADVANCED, default=0),
    ParamDef(key="train.freeze_JP_bert", type=bool, i18n_key="freeze_JP_bert", group=GROUP_ADVANCED, default=False),
    ParamDef(key="train.freeze_style", type=bool, i18n_key="freeze_style", group=GROUP_ADVANCED, default=False),
    ParamDef(key="train.freeze_decoder", type=bool, i18n_key="freeze_decoder", group=GROUP_ADVANCED, default=False),
]


def _get_nested(data: dict, key: str) -> Any:
    """중첩 딕셔너리에서 키로 값을 가져옴. 키가 없으면 _MISSING 반환."""
    parts = key.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_nested(data: dict, key: str, value: Any):
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _format_value(param: ParamDef, value: Any) -> str:
    if param.type == bool:
        return "[success]ON[/success]" if value else "[muted]OFF[/muted]"
    return str(value)


def _count_train_lines(dataset_path: Path) -> int:
    train_list = dataset_path / "train.list"
    if not train_list.exists():
        return 0
    with open(train_list, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _render_config_table(config: dict, train_count: int):
    train = config.get("train", {})
    epochs = train.get("epochs", 100)
    batch_size = train.get("batch_size", 2)
    eval_interval = train.get("eval_interval", 1000)

    steps_per_epoch = math.ceil(train_count / batch_size) if train_count > 0 else 0
    total_steps = steps_per_epoch * epochs
    num_checkpoints = max(0, total_steps // eval_interval) if eval_interval > 0 else 0
    disk_gb = (num_checkpoints * 240) / 1024

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        padding=(0, 2),
        expand=True,
    )
    table.add_column(t("training.config.col_parameter"), style="label")
    table.add_column(t("training.config.col_description"), style="dim")
    table.add_column(t("training.config.col_value"), style="value", justify="right")

    group_labels = {
        GROUP_BASIC: t("training.config.group_basic"),
        GROUP_ADVANCED: t("training.config.group_advanced"),
    }

    current_group = None
    for param in EDITABLE_PARAMS:
        if param.group != current_group:
            if current_group is not None:
                table.add_row("", "", "")
            table.add_row(f"[accent]{group_labels.get(param.group, param.group)}[/accent]", "", "")
            current_group = param.group

        raw_value = _get_nested(config, param.key)
        value = param.default if raw_value is _MISSING else raw_value
        name = param.key.split(".")[-1]
        table.add_row(f"  {name}", param.label, _format_value(param, value))

    console.print(Panel(
        table,
        title=t("training.config.table_title"),
        border_style="cyan",
        padding=(1, 2),
    ))

    if train_count > 0:
        summary = t(
            "training.config.summary",
            total_steps=f"{total_steps:,}",
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            num_checkpoints=num_checkpoints,
            disk_gb=disk_gb,
        )
        console.print(summary)


def _show_help(param: ParamDef, current_value: Any):
    """파라미터 도움말 패널 표시."""
    val_display = _format_value(param, current_value)
    help_text = f"[dim]{param.help}[/dim]\n\n{t('training.config.help_current_value', value=val_display)}"

    name = param.key.split(".")[-1]
    console.print(Panel(
        help_text,
        title=f"[accent]{name}[/accent] [dim]— {param.label}[/dim]",
        border_style="bright_black",
        padding=(1, 2),
    ))


def edit_config(dataset_path: Path) -> dict:
    """인터랙티브 config 편집. 수정된 config dict 반환."""
    config_path = dataset_path / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    train_count = _count_train_lines(dataset_path)

    while True:
        console.print()
        _render_config_table(config, train_count)
        console.print()

        # 그룹 구분이 있는 선택 목록 구성
        group_labels = {
            GROUP_BASIC: t("training.config.group_basic"),
            GROUP_ADVANCED: t("training.config.group_advanced"),
        }
        label_done = t("training.config.done_save")

        choices: list = []
        current_group = None
        for p in EDITABLE_PARAMS:
            if p.group != current_group:
                choices.append(Separator(f"── {group_labels.get(p.group, p.group)} ──"))
                current_group = p.group
            name = p.key.split(".")[-1]
            choices.append(f"{name} — {p.label}")
        choices.append(Separator())
        choices.append(label_done)

        choice = inquirer.select(
            message=t("training.config.prompt_select"),
            choices=choices,
            pointer="❯",
        ).execute()

        if choice == label_done:
            break

        # 선택된 파라미터 찾기
        param_name = choice.split(" — ")[0]
        param = next(p for p in EDITABLE_PARAMS if p.key.split(".")[-1] == param_name)
        raw_value = _get_nested(config, param.key)
        current = param.default if raw_value is _MISSING else raw_value

        # 도움말 표시
        console.print()
        _show_help(param, current)

        if param.type == bool:
            new_val = not current
            label = _format_value(param, new_val)
            console.print(f"  {param_name}: → {label}")
        else:
            raw = edit_value(f"{param.label}", current)
            try:
                new_val = param.type(raw)
            except ValueError:
                console.print(t("training.config.invalid_value", value=raw))
                continue

        _set_nested(config, param.key, new_val)

    # 저장
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    console.print(t("training.config.save_complete"))

    return config
