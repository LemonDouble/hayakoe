"""Typer 앱 정의 및 서브커맨드 등록."""

from pathlib import Path
from typing import Optional

import typer

from cli.i18n import t, _SUPPORTED
from cli.i18n import _LANG_LABELS, set_lang, get_lang
from cli.ui.console import console, LOGO
from cli.ui.prompts import select_from_list


app = typer.Typer(
    name="hayakoe-dev",
    help=t("app.help"),
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
def benchmark():
    """벤치마크 메뉴 진입."""
    from cli.benchmark.menu import benchmark_menu

    benchmark_menu()


@app.command()
def publish():
    """화자 배포 메뉴 진입 (HF / S3 / 로컬 업로드)."""
    from cli.publish.menu import publish_menu

    publish_menu()


@app.command()
def export(
    checkpoint: Path = typer.Argument(..., help=".safetensors 체크포인트 경로"),
    output: Path = typer.Argument(..., help="ONNX 를 내보낼 디렉토리"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="config.json 경로 (기본: 체크포인트와 같은 폴더)"
    ),
    duration_predictor: bool = typer.Option(
        True, help="duration_predictor.onnx 도 함께 내보낼지"
    ),
):
    """체크포인트를 ONNX 로 내보낸다 (업로드 없이 로컬에만).

    publish 메뉴와 달리 기존 synthesizer.onnx 를 재사용하지 않고 항상 새로 만든다.
    """
    from cli.export.exporter import export_duration_predictor, export_synthesizer

    config_path = config or checkpoint.parent / "config.json"
    if not checkpoint.exists():
        console.print(f"[red]체크포인트를 찾을 수 없습니다: {checkpoint}[/red]")
        raise typer.Exit(1)
    if not config_path.exists():
        console.print(f"[red]config.json 을 찾을 수 없습니다: {config_path}[/red]")
        raise typer.Exit(1)

    console.print(f"synthesizer 내보내는 중 → {output}")
    export_synthesizer(config_path, checkpoint, output)
    if duration_predictor:
        console.print(f"duration_predictor 내보내는 중 → {output}")
        export_duration_predictor(config_path, checkpoint, output)
    console.print("[green]완료[/green]")


def _intro():
    return t("app.intro")


def interactive_main_menu():
    """메인 인터랙티브 메뉴."""
    console.print(LOGO)
    console.print(_intro())
    console.print()

    menu_training = t("app.menu.training")
    menu_report = t("app.menu.report")
    menu_benchmark = t("app.menu.benchmark")
    menu_publish = t("app.menu.publish")
    menu_lang = f"🌐 Language ({_LANG_LABELS[get_lang()]})"
    menu_exit = t("app.menu.exit")

    while True:
        choice = select_from_list(t("app.menu.prompt"), [
            menu_training,
            menu_report,
            menu_benchmark,
            menu_publish,
            menu_lang,
            menu_exit,
        ])
        if choice == menu_training:
            from cli.training.menu import training_menu

            training_menu()
        elif choice == menu_report:
            from cli.report.menu import report_menu

            report_menu()
        elif choice == menu_benchmark:
            from cli.benchmark.menu import benchmark_menu

            benchmark_menu()
        elif choice == menu_publish:
            from cli.publish.menu import publish_menu

            publish_menu()
        elif choice == menu_lang:
            _change_language()
        elif choice == menu_exit:
            console.print(t("app.exit_message"))
            break


def _change_language():
    current = get_lang()
    choices = [
        f"{_LANG_LABELS[code]}{' ✓' if code == current else ''}"
        for code in _SUPPORTED
    ]
    chosen = select_from_list("Select language / 언어 선택", choices)
    for code in _SUPPORTED:
        if chosen.startswith(_LANG_LABELS[code]):
            if code != current:
                set_lang(code)
                console.print(f"\n  ✓ {_LANG_LABELS[code]}")
                console.print("  Please restart the CLI. / 다시 실행해 주세요.\n")
                raise SystemExit(0)
            break
