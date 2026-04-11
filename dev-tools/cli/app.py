"""Typer 앱 정의 및 서브커맨드 등록."""

import typer

from cli.ui.console import console, LOGO
from cli.ui.prompts import select_from_list


app = typer.Typer(
    name="hayakoe-dev",
    help="HayaKoe TTS 개발 도구",
    no_args_is_help=False,
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """인자 없이 실행하면 인터랙티브 모드 진입."""
    if ctx.invoked_subcommand is None:
        interactive_main_menu()


@app.command()
def train():
    """학습 파이프라인 메뉴 진입."""
    from cli.training.menu import training_menu

    training_menu()


@app.command()
def report():
    """품질 리포트 생성 메뉴 진입."""
    from cli.report.menu import report_menu

    report_menu()


@app.command()
def export():
    """ONNX 내보내기 메뉴 진입 (CPU 추론용)."""
    from cli.export.menu import export_menu

    export_menu()


@app.command()
def benchmark():
    """벤치마크 메뉴 진입."""
    from cli.benchmark.menu import benchmark_menu

    benchmark_menu()


INTRO = (
    "[dim]HayaKoe TTS 모델의 학습부터 배포 준비까지를 안내합니다.\n"
    "아래 순서대로 진행하세요.[/dim]\n\n"
    "[dim]  [accent]① 학습[/accent]          음성 데이터로 TTS 모델을 학습합니다.\n"
    "  [accent]② 품질 리포트[/accent]   학습된 체크포인트의 음성을 비교 시청합니다.\n"
    "                    만족할 때까지 ①↔② 를 반복합니다.\n"
    "  [accent]③ ONNX 내보내기[/accent]  최종 모델을 ONNX 형식으로 변환합니다.\n"
    "                    GPU 없는 환경(서버, 로컬 등)에서 추론할 때 필요합니다.\n"
    "                    GPU로만 추론한다면 이 단계는 건너뛰어도 됩니다.\n"
    "  [accent]④ 벤치마크[/accent]       CPU/GPU에서 추론 속도를 측정합니다.[/dim]"
)


def interactive_main_menu():
    """메인 인터랙티브 메뉴."""
    console.print(LOGO)
    console.print(INTRO)
    console.print()

    while True:
        choice = select_from_list("무엇을 할까요?", [
            "학습 파이프라인 — 데이터 전처리 + 모델 학습",
            "품질 리포트 — 체크포인트별 음성 비교 시청",
            "ONNX 내보내기 — GPU 없이 추론하려면 필요 (CPU 배포용)",
            "벤치마크 — CPU/GPU 추론 속도 측정",
            "종료",
        ])
        if "학습 파이프라인" in choice:
            from cli.training.menu import training_menu

            training_menu()
        elif "품질 리포트" in choice:
            from cli.report.menu import report_menu

            report_menu()
        elif "ONNX 내보내기" in choice:
            from cli.export.menu import export_menu

            export_menu()
        elif "벤치마크" in choice:
            from cli.benchmark.menu import benchmark_menu

            benchmark_menu()
        elif choice == "종료":
            console.print("  [dim]종료합니다.[/dim]")
            break
