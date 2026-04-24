"""벤치마크 인터랙티브 메뉴."""

from cli.i18n import t
from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm


def benchmark_menu():
    """벤치마크 메인 메뉴."""
    console.print()
    console.print(t("benchmark.menu.intro"))
    console.print()

    label_cpu = t("benchmark.menu.mode_cpu")
    label_gpu = t("benchmark.menu.mode_gpu")
    label_both = t("benchmark.menu.mode_both")
    label_back = t("benchmark.menu.back")

    mode = select_from_list(t("benchmark.menu.mode_select"), [
        label_cpu,
        label_gpu,
        label_both,
        label_back,
    ])

    if mode == label_back:
        return

    if mode == label_both:
        devices = ["cpu", "cuda"]
    elif mode == label_cpu:
        devices = ["cpu"]
    else:
        devices = ["cuda"]

    # GPU 선택 시 CUDA 사용 가능 여부 확인
    if "cuda" in devices:
        try:
            import torch
            if not torch.cuda.is_available():
                console.print(t("benchmark.menu.cuda_unavailable"))
                console.print(t("benchmark.menu.cuda_check_hint"))
                return
            console.print(f"  [dim]GPU: {torch.cuda.get_device_name(0)}[/dim]")
        except ImportError:
            console.print(t("benchmark.menu.pytorch_required"))
            return

    device_labels = []
    for d in devices:
        if d == "cpu":
            device_labels.append("CPU (ONNX Runtime)")
        else:
            device_labels.append("GPU (torch.compile)")

    console.print()
    console.print(t("benchmark.menu.settings_title"))
    console.print(t("benchmark.menu.settings_mode", mode=" + ".join(device_labels)))
    console.print(t("benchmark.menu.settings_texts"))
    console.print(t("benchmark.menu.settings_runs"))
    console.print()

    if not confirm(t("benchmark.menu.confirm_start")):
        return

    from cli.benchmark.runner import run_benchmark

    output_path = run_benchmark(devices)

    console.print(t("benchmark.menu.complete"))
    console.print(f"  [dim]{output_path}[/dim]\n")

    if confirm(t("benchmark.menu.open_browser"), default=True):
        _open_report(output_path)


def _open_report(path):
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
