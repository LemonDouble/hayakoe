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

from cli.ui.console import console
from cli.ui.prompts import edit_value


GROUP_BASIC = "기본 설정"
GROUP_ADVANCED = "고급 설정"


_MISSING = object()


@dataclass
class ParamDef:
    key: str
    type: type
    label: str
    group: str
    help: str
    default: Any = None


EDITABLE_PARAMS: list[ParamDef] = [
    # ── 기본 설정 ──────────────────────────────────────────────
    ParamDef(
        key="train.epochs",
        type=int,
        label="학습 에포크 수",
        group=GROUP_BASIC,
        help=(
            "전체 학습 데이터를 처음부터 끝까지 몇 번 반복 학습할지 설정합니다.\n"
            "\n"
            "▸ 높이면: 모델이 더 오래 학습하여 음질이 향상될 수 있습니다.\n"
            "  다만 너무 높으면 학습 데이터를 통째로 외워버려서,\n"
            "  새로운 문장을 읽을 때 오히려 부자연스러워집니다 (과적합).\n"
            "▸ 낮추면: 학습이 빨리 끝나지만, 모델이 충분히 배우지 못해\n"
            "  음질이 낮을 수 있습니다.\n"
            "\n"
            "권장: 데이터 100~500문장 → 300~500 에포크\n"
            "      데이터 1000문장 이상 → 100~200 에포크"
        ),
    ),
    ParamDef(
        key="train.batch_size",
        type=int,
        label="배치 크기 (GPU당)",
        group=GROUP_BASIC,
        help=(
            "한 번에 몇 개의 음성 샘플을 묶어서 학습할지 설정합니다.\n"
            "\n"
            "▸ 높이면: 학습 속도가 빨라지고 안정적이지만,\n"
            "  GPU 메모리(VRAM)를 더 많이 사용합니다.\n"
            "  메모리가 부족하면 \"CUDA out of memory\" 에러가 발생합니다.\n"
            "▸ 낮추면: 적은 메모리로도 학습 가능하지만, 속도가 느려집니다.\n"
            "\n"
            "GPU별 권장: RTX 3060 (12GB) → 2~4\n"
            "            RTX 3090/4090 (24GB) → 4~8"
        ),
    ),
    ParamDef(
        key="train.nproc_per_node",
        type=int,
        label="GPU 수",
        group=GROUP_BASIC,
        default=1,
        help=(
            "학습에 사용할 GPU 수를 설정합니다.\n"
            "여러 GPU가 있으면 데이터를 나누어 병렬로 학습하여\n"
            "학습 속도가 비례하여 빨라집니다 (DistributedDataParallel).\n"
            "\n"
            "▸ 높이면: 학습 속도가 GPU 수에 비례하여 빨라집니다.\n"
            "  단, 실제 장착된 GPU 수를 초과하면 에러가 발생합니다.\n"
            "▸ 1로 두면: 단일 GPU 학습 (기본값).\n"
            "\n"
            "주의: GPU가 1개뿐이면 이 값을 변경하지 마세요.\n"
            "      nvidia-smi 명령으로 GPU 수를 확인할 수 있습니다."
        ),
    ),
    ParamDef(
        key="train.bf16_run",
        type=bool,
        label="bfloat16 혼합 정밀도",
        group=GROUP_BASIC,
        default=False,
        help=(
            "계산 정밀도를 32비트에서 16비트로 줄여 학습 속도를 높이는 기법입니다.\n"
            "\n"
            "▸ 켜면: 학습 속도가 ~1.5배 빨라지고, GPU 메모리 사용량이 줄어듭니다.\n"
            "  단, Ampere 이상 GPU (RTX 30xx, 40xx, A100 등)가 필요합니다.\n"
            "▸ 끄면: 모든 GPU에서 동작하지만, 학습이 더 느립니다.\n"
            "\n"
            "주의: RTX 20xx 이하 GPU에서 켜면 에러가 발생하거나\n"
            "      품질이 떨어질 수 있습니다."
        ),
    ),
    ParamDef(
        key="train.eval_interval",
        type=int,
        label="체크포인트 저장 간격 (스텝)",
        group=GROUP_BASIC,
        help=(
            "몇 스텝마다 학습 중간 결과물(체크포인트)을 저장할지 설정합니다.\n"
            "체크포인트는 학습 중 모델의 스냅샷으로, 나중에 가장 좋은 시점의\n"
            "모델을 골라 사용할 수 있게 해줍니다.\n"
            "\n"
            "▸ 자주 저장하면: 세밀하게 비교할 수 있지만,\n"
            "  디스크 공간을 더 많이 사용합니다 (체크포인트당 ~240MB).\n"
            "▸ 드물게 저장하면: 디스크를 절약하지만,\n"
            "  좋은 시점의 모델을 놓칠 수 있습니다.\n"
            "\n"
            "권장: 총 스텝이 적으면 500, 많으면 2000~5000"
        ),
    ),
    ParamDef(
        key="train.log_interval",
        type=int,
        label="텐서보드 로깅 간격 (스텝)",
        group=GROUP_BASIC,
        help=(
            "몇 스텝마다 학습 상태(loss 등)를 텐서보드에 기록할지 설정합니다.\n"
            "텐서보드는 학습 진행 상황을 그래프로 볼 수 있는 모니터링 도구입니다.\n"
            "\n"
            "▸ 자주 기록하면: 학습 과정을 더 세밀하게 관찰할 수 있습니다.\n"
            "▸ 드물게 기록하면: 로그 파일이 작아지고\n"
            "  학습이 아주 약간 빨라집니다.\n"
            "\n"
            "대부분의 경우 기본값(200)이면 충분합니다."
        ),
    ),
    # ── 고급 설정 ──────────────────────────────────────────────
    ParamDef(
        key="train.learning_rate",
        type=float,
        label="학습률",
        group=GROUP_ADVANCED,
        help=(
            "모델이 한 스텝마다 얼마나 크게 수정될지를 결정합니다.\n"
            "자전거 핸들의 민감도와 비슷합니다.\n"
            "\n"
            "▸ 높이면: 빠르게 학습하지만, 값이 튀어서 학습이\n"
            "  불안정해지거나 완전히 발산할 수 있습니다.\n"
            "▸ 낮추면: 안정적이지만, 학습이 매우 느려지고\n"
            "  최적점에 도달하지 못할 수 있습니다.\n"
            "\n"
            "잘 모르겠으면 기본값(0.0001)을 그대로 사용하세요."
        ),
    ),
    ParamDef(
        key="train.warmup_epochs",
        type=int,
        label="워밍업 에포크",
        group=GROUP_ADVANCED,
        default=0,
        help=(
            "학습 초반에 학습률을 0부터 서서히 올려주는 구간입니다.\n"
            "자동차를 출발할 때 급가속 대신 천천히 속도를 올리는 것과 비슷합니다.\n"
            "\n"
            "▸ 설정하면: 학습 초반 불안정을 방지하여\n"
            "  더 안정적으로 수렴합니다.\n"
            "▸ 0으로 두면: 처음부터 바로 설정된 학습률로 시작합니다.\n"
            "\n"
            "권장: 0~3 에포크.\n"
            "      학습이 초반에 불안정하면 1~2로 설정해보세요."
        ),
    ),
    ParamDef(
        key="train.freeze_JP_bert",
        type=bool,
        label="JP BERT 인코더 동결",
        group=GROUP_ADVANCED,
        default=False,
        help=(
            "텍스트를 이해하는 부분(BERT)의 가중치를 고정하여\n"
            "학습하지 않습니다.\n"
            "\n"
            "▸ 켜면: BERT가 이미 학습한 일본어 지식을 보존합니다.\n"
            "  데이터가 적을 때(~200문장 이하) 과적합을 방지하는 데\n"
            "  효과적입니다. 학습 속도와 메모리 사용량도 개선됩니다.\n"
            "▸ 끄면: BERT도 함께 미세조정되어, 데이터가 충분할 때\n"
            "  더 자연스러운 발음과 억양을 학습할 수 있습니다.\n"
            "\n"
            "권장: 데이터 200문장 이하 → ON\n"
            "      데이터 500문장 이상 → OFF"
        ),
    ),
    ParamDef(
        key="train.freeze_style",
        type=bool,
        label="스타일 인코더 동결",
        group=GROUP_ADVANCED,
        default=False,
        help=(
            "화자의 말투/감정을 표현하는 스타일 인코더의 가중치를\n"
            "고정합니다.\n"
            "\n"
            "▸ 켜면: 기존 스타일 표현력을 보존합니다.\n"
            "  데이터가 적거나, 기존 모델의 감정 표현력을\n"
            "  유지하고 싶을 때 유용합니다.\n"
            "▸ 끄면: 새로운 화자의 스타일 특성을 자유롭게 학습합니다.\n"
            "  데이터가 충분하면 끄는 것이 일반적입니다.\n"
            "\n"
            "권장: 일반적으로 OFF.\n"
            "      학습 결과가 단조롭다면 ON으로 시도해보세요."
        ),
    ),
    ParamDef(
        key="train.freeze_decoder",
        type=bool,
        label="디코더 동결",
        group=GROUP_ADVANCED,
        default=False,
        help=(
            "최종 음성을 생성하는 디코더의 가중치를 고정합니다.\n"
            "\n"
            "▸ 켜면: 음성 생성 부분은 변경되지 않고,\n"
            "  텍스트 해석 부분만 학습합니다.\n"
            "  학습이 빨라지지만 음질 개선 폭이 제한됩니다.\n"
            "▸ 끄면: 디코더도 함께 학습하여\n"
            "  해당 화자에 맞는 음질을 낼 수 있습니다.\n"
            "\n"
            "권장: 일반적으로 OFF.\n"
            "      특수한 경우가 아니면 끌 필요 없습니다."
        ),
    ),
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
    table.add_column("파라미터", style="label")
    table.add_column("설명", style="dim")
    table.add_column("값", style="value", justify="right")

    current_group = None
    for param in EDITABLE_PARAMS:
        if param.group != current_group:
            if current_group is not None:
                table.add_row("", "", "")
            table.add_row(f"[accent]{param.group}[/accent]", "", "")
            current_group = param.group

        raw_value = _get_nested(config, param.key)
        value = param.default if raw_value is _MISSING else raw_value
        name = param.key.split(".")[-1]
        table.add_row(f"  {name}", param.label, _format_value(param, value))

    console.print(Panel(
        table,
        title="[accent]학습 설정[/accent]",
        border_style="cyan",
        padding=(1, 2),
    ))

    if train_count > 0:
        summary = (
            f"  [label]총 스텝[/label]  [value]{total_steps:,}[/value]"
            f"  [dim]({steps_per_epoch} steps/epoch × {epochs} epochs)[/dim]"
            f"    [label]예상 저장[/label]  [value]{num_checkpoints}[/value]회"
            f"    [label]디스크[/label]  [value]~{disk_gb:.1f}GB[/value]"
        )
        console.print(summary)


def _show_help(param: ParamDef, current_value: Any):
    """파라미터 도움말 패널 표시."""
    val_display = _format_value(param, current_value)
    help_text = f"[dim]{param.help}[/dim]\n\n현재 값: {val_display}"

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
        choices: list = []
        current_group = None
        for p in EDITABLE_PARAMS:
            if p.group != current_group:
                choices.append(Separator(f"── {p.group} ──"))
                current_group = p.group
            name = p.key.split(".")[-1]
            choices.append(f"{name} — {p.label}")
        choices.append(Separator())
        choices.append("완료 (저장)")

        choice = inquirer.select(
            message="수정할 파라미터",
            choices=choices,
            pointer="❯",
        ).execute()

        if choice == "완료 (저장)":
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
                console.print(f"  [error]잘못된 값: {raw}[/error]")
                continue

        _set_nested(config, param.key, new_val)

    # 저장
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    console.print("  [success]설정 저장 완료[/success]")

    return config
