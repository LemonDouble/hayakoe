import subprocess
import sys
from typing import Any, Callable

from hayakoe.logging import logger
from hayakoe.utils.stdout_wrapper import SAFE_STDOUT


def run_script_with_log(
    cmd: list[str], ignore_warning: bool = False
) -> tuple[bool, str]:
    """
    지정된 커맨드를 실행하고 로그를 기록한다.

    Args:
        cmd: 실행할 커맨드 리스트
        ignore_warning: 경고를 무시할지 여부 플래그

    Returns:
        tuple[bool, str]: 실행 성공 여부의 불리언 값과, 에러 또는 경고 메시지(있는 경우)
    """

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        [sys.executable] + cmd,
        stdout=SAFE_STDOUT,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Error: {' '.join(cmd)}\n{result.stderr}")
        return False, result.stderr
    elif result.stderr and not ignore_warning:
        logger.warning(f"Warning: {' '.join(cmd)}\n{result.stderr}")
        return True, result.stderr
    logger.success(f"Success: {' '.join(cmd)}")

    return True, ""


def second_elem_of(
    original_function: Callable[..., tuple[Any, Any]],
) -> Callable[..., Any]:
    """
    주어진 함수를 래핑하여 반환값의 두 번째 요소만 반환하는 함수를 생성한다.

    Args:
        original_function (Callable[..., tuple[Any, Any]])): 래핑할 원본 함수

    Returns:
        Callable[..., Any]: 원본 함수의 반환값 중 두 번째 요소만 반환하는 함수
    """

    def inner_function(*args, **kwargs) -> Any:  # type: ignore
        return original_function(*args, **kwargs)[1]

    return inner_function
