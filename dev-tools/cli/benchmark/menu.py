"""벤치마크 인터랙티브 메뉴."""

from cli.ui.console import console
from cli.ui.prompts import select_from_list, confirm


def benchmark_menu():
    """벤치마크 메인 메뉴."""
    console.print()
    console.print(
        "  [accent]벤치마크[/accent] [dim]— HayaKoe 추론 성능 측정[/dim]\n\n"
        "  [dim]텍스트를 음성으로 변환하는 데 걸리는 시간을 측정합니다.\n"
        "  결과는 '배속'으로 표시됩니다. 예를 들어 10.0x는\n"
        "  1초 분량의 음성을 0.1초 만에 생성할 수 있다는 뜻입니다.\n"
        "  1.0x 이상이면 실시간보다 빠르고, 높을수록 좋습니다.\n\n"
        "  · CPU — ONNX Runtime 사용 (GPU 없이 동작, 서버/로컬 배포용)\n"
        "  · GPU — PyTorch CUDA 사용 (NVIDIA GPU 필요, 고속 추론)[/dim]"
    )
    console.print()

    mode = select_from_list("벤치마크 모드", [
        "CPU (ONNX Runtime)",
        "GPU (PyTorch CUDA)",
        "CPU + GPU (전체)",
        "뒤로",
    ])

    if mode == "뒤로":
        return

    if "CPU + GPU" in mode:
        devices = ["cpu", "cuda"]
    elif "CPU" in mode:
        devices = ["cpu"]
    else:
        devices = ["cuda"]

    # GPU 선택 시 CUDA 사용 가능 여부 확인
    if "cuda" in devices:
        try:
            import torch
            if not torch.cuda.is_available():
                console.print("\n  [error]CUDA를 사용할 수 없습니다.[/error]")
                console.print("  [dim]CUDA 드라이버와 PyTorch CUDA 빌드를 확인하세요.[/dim]\n")
                return
            console.print(f"  [dim]GPU: {torch.cuda.get_device_name(0)}[/dim]")
        except ImportError:
            console.print("\n  [error]GPU 벤치마크에는 PyTorch(CUDA)가 필요합니다.[/error]\n")
            return

    device_labels = []
    for d in devices:
        if d == "cpu":
            device_labels.append("CPU (ONNX Runtime)")
        else:
            device_labels.append("GPU (PyTorch CUDA)")

    console.print()
    console.print("  [accent]벤치마크 설정[/accent]")
    console.print(f"  모드:     [value]{' + '.join(device_labels)}[/value]")
    console.print(f"  텍스트:   [value]3개 (짧음/중간/김)[/value]")
    console.print(f"  반복:     [value]워밍업 2회 + 측정 5회[/value]")
    console.print()

    if not confirm("벤치마크를 시작하시겠습니까?"):
        return

    from cli.benchmark.runner import run_benchmark

    output_path = run_benchmark(devices)

    console.print(f"\n[success]벤치마크 완료![/success]")
    console.print(f"  [dim]{output_path}[/dim]\n")

    if confirm("브라우저에서 열시겠습니까?", default=True):
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
