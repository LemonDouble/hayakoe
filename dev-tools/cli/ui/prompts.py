"""InquirerPy 기반 인터랙티브 프롬프트 래퍼."""

from typing import Any

from InquirerPy import inquirer


def select_from_list(message: str, choices: list[str]) -> str:
    """화살표 키로 선택하는 리스트 프롬프트."""
    return inquirer.select(
        message=message,
        choices=choices,
        pointer="❯",
    ).execute()


def confirm(message: str, default: bool = True) -> bool:
    """예/아니오 확인 프롬프트."""
    return inquirer.confirm(message=message, default=default).execute()


def edit_value(message: str, current: Any) -> str:
    """현재 값을 기본값으로 보여주는 텍스트 입력 프롬프트."""
    return inquirer.text(
        message=message,
        default=str(current),
    ).execute()
