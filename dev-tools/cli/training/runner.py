"""torchrun을 통한 학습 실행."""

import glob
import os
import signal
import subprocess
import sys
from pathlib import Path

from cli.i18n import t
from cli.training.tensorboard import launch_tensorboard, get_tensorboard_url
from cli.ui.console import console


TRAINING_SCRIPT = Path(__file__).resolve().parent / "core" / "train_ms_jp_extra.py"

PRETRAINED_FILES = ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]


def ensure_pretrained(training_dir: Path):
    """사전학습 모델이 없으면 HuggingFace에서 다운로드."""
    training_dir.mkdir(parents=True, exist_ok=True)

    # 기존 체크포인트가 있으면 재개 학습 → 스킵
    if glob.glob(str(training_dir / "G_*.pth")):
        return

    # 사전학습 모델이 이미 있으면 스킵
    if all((training_dir / f).exists() for f in PRETRAINED_FILES):
        return

    console.print(t("training.runner.downloading_pretrained"))

    from huggingface_hub import hf_hub_download
    from hayakoe.constants import HF_REPO

    for filename in PRETRAINED_FILES:
        console.print(f"  {filename}")
        hf_hub_download(
            HF_REPO,
            f"pretrained/{filename}",
            local_dir=str(training_dir),
        )
        # hf_hub_download은 pretrained/ 서브폴더에 저장하므로 training/ 루트로 이동
        downloaded = training_dir / "pretrained" / filename
        if downloaded.exists():
            downloaded.rename(training_dir / filename)

    # pretrained/ 빈 폴더 정리
    pretrained_subdir = training_dir / "pretrained"
    if pretrained_subdir.exists() and not any(pretrained_subdir.iterdir()):
        pretrained_subdir.rmdir()

    console.print(t("training.runner.pretrained_ready"))


def launch_training(
    dataset_path: Path,
    config_path: Path,
    nproc: int = 1,
    speedup: bool = False,
) -> subprocess.Popen:
    """torchrun으로 학습 프로세스 실행."""
    env = os.environ.copy()
    env["HAYAKOE_PROJECT_DIR"] = str(dataset_path)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(TRAINING_SCRIPT),
        "-c", str(config_path),
        "-m", str(dataset_path),
    ]
    if speedup:
        cmd.append("--speedup")

    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(TRAINING_SCRIPT.parent),
    )


def start_training_session(dataset_path: Path, data_dir: Path | None = None, speedup: bool = False):
    """TensorBoard + 학습을 실행하고, 완료 또는 Ctrl+C까지 대기."""
    import json

    data_dir = data_dir or dataset_path
    config_path = data_dir / "config.json"
    training_dir = dataset_path / "training"

    # config에서 GPU 수 읽기
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    nproc = config.get("train", {}).get("nproc_per_node", 1)

    # 사전학습 모델 확인/다운로드
    ensure_pretrained(training_dir)

    # TensorBoard 실행
    tb_proc = launch_tensorboard(training_dir)
    console.print(t("training.runner.tensorboard_url", url=get_tensorboard_url()))

    # 학습 실행
    gpu_label = t("training.runner.gpu_label", count=nproc)
    console.print(t("training.runner.training_start", label=gpu_label))
    train_proc = launch_training(dataset_path, config_path, nproc=nproc, speedup=speedup)

    try:
        exit_code = train_proc.wait()
        if exit_code == 0:
            console.print(t("training.runner.training_complete"))
        else:
            console.print(t("training.runner.training_exit_code", code=exit_code))
    except KeyboardInterrupt:
        console.print(t("training.runner.training_stopping"))
        train_proc.send_signal(signal.SIGINT)
        try:
            train_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            train_proc.kill()
    finally:
        tb_proc.terminate()
        try:
            tb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb_proc.kill()
        console.print(t("training.runner.tensorboard_stopped"))
