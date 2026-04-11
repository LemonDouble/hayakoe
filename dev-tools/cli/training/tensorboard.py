"""TensorBoard 프로세스 관리."""

import shutil
import subprocess
import sys
from pathlib import Path


def launch_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen:
    """TensorBoard를 백그라운드 프로세스로 실행. Popen 핸들 반환."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # venv 내 tensorboard CLI 엔트리포인트 사용
    venv_tb = Path(sys.executable).parent / "tensorboard"
    tb_cmd = str(venv_tb) if venv_tb.exists() else shutil.which("tensorboard") or "tensorboard"

    proc = subprocess.Popen(
        [tb_cmd, "--logdir", str(log_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def get_tensorboard_url(port: int = 6006) -> str:
    return f"http://localhost:{port}"
