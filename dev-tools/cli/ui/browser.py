"""브라우저로 파일 열기."""

import subprocess
from pathlib import Path


def open_in_browser(path: Path) -> None:
    """WSL2 환경에서 브라우저로 리포트 열기."""
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
