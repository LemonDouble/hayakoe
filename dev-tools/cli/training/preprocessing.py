"""학습 전처리 4단계 오케스트레이션.

1. preprocess_text — G2P 변환, 4필드 → 7필드
2. bert_gen — BERT 임베딩 생성 (.bert.pt)
3. style_gen — 스타일 벡터 생성 (.npy)
4. default_style — 평균 스타일 벡터 생성
"""

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from cli.training.dataset import activate_dataset
from cli.ui.console import console


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.readlines() if line.strip()]


def run_preprocess_text(project_dir: Path, data_dir: Path):
    """Step 1: G2P 변환. 기존 train.list/val.list (4필드) → 7필드."""
    import json

    train_path = data_dir / "train.list"
    val_path = data_dir / "val.list"

    if not train_path.exists() or not val_path.exists():
        console.print(
            "  [error]train.list / val.list가 없습니다."
            " 먼저 데이터 분할을 진행해 주세요.[/error]"
        )
        raise SystemExit(1)

    activate_dataset(project_dir)
    from preprocess_text import process_line

    config_path = data_dir / "config.json"
    error_log_path = data_dir / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()

    spk_id_map: dict[str, int] = {}
    current_sid = 0
    error_count = 0

    for list_path in [train_path, val_path]:
        lines = _read_lines(list_path)
        processed: list[str] = []
        for line in lines:
            # 이미 7필드(전처리 완료)면 4필드로 잘라서 재처리
            fields = line.strip().split("|")
            if len(fields) == 7:
                line = "|".join(fields[:4]) + "\n"
            try:
                result = process_line(
                    line, list_path, correct_path=False,
                    use_jp_extra=True, yomi_error="skip",
                )
            except Exception as e:
                with error_log_path.open("a", encoding="utf-8") as err_f:
                    err_f.write(f"{line.strip()}\n{e}\n\n")
                error_count += 1
                continue
            utt, spk = result.strip().split("|")[:2]
            if not Path(utt).is_file():
                console.print(f"  [warning]오디오 없음: {utt}[/warning]")
                continue
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
            processed.append(result)
        list_path.write_text("".join(processed), encoding="utf-8")

    # Config 업데이트
    json_config = json.loads(config_path.read_text(encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    config_path.write_text(
        json.dumps(json_config, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if error_count > 0:
        console.print(f"  [warning]{error_count}개 라인 스킵됨. {error_log_path} 확인.[/warning]")


def run_bert_gen(project_dir: Path, data_dir: Path, force: bool = False):
    """Step 2: BERT 임베딩 생성."""
    activate_dataset(project_dir)

    import os
    from bert_gen import process_line
    from hayakoe.models.hyper_parameters import HyperParameters

    config_path = data_dir / "config.json"
    hps = HyperParameters.load_from_json(str(config_path))
    from config import get_config
    device = get_config().bert_gen_config.device

    lines = _read_lines(Path(hps.data.training_files)) + _read_lines(Path(hps.data.validation_files))

    if force:
        # 기존 .bert.pt 파일 삭제
        for line in lines:
            wav_path = line.strip().split("|")[0]
            bert_path = Path(wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt"))
            if bert_path.exists():
                bert_path.unlink()
        pending = lines
    else:
        # 미완료 라인만 필터
        pending = []
        for line in lines:
            wav_path = line.strip().split("|")[0]
            bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
            if not os.path.exists(bert_path):
                pending.append(line)

    if not pending:
        console.print("  [success]BERT 임베딩 이미 완료[/success]")
        return

    console.print(f"  BERT 임베딩 생성: {len(pending)}개 (전체 {len(lines)}개 중)")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), console=console,
    ) as progress:
        task = progress.add_task("BERT", total=len(pending))
        for line in pending:
            process_line(line, hps.data.add_blank, device)
            progress.advance(task)


def run_style_gen(project_dir: Path, data_dir: Path, force: bool = False):
    """Step 3: 스타일 벡터 생성."""
    activate_dataset(project_dir)

    import os
    from hayakoe.models.hyper_parameters import HyperParameters

    config_path = data_dir / "config.json"
    hps = HyperParameters.load_from_json(str(config_path))

    lines = _read_lines(Path(hps.data.training_files)) + _read_lines(Path(hps.data.validation_files))

    if force:
        # 기존 .npy 파일 삭제
        for line in lines:
            wav_path = line.strip().split("|")[0]
            npy_path = Path(f"{wav_path}.npy")
            if npy_path.exists():
                npy_path.unlink()
        pending = lines
    else:
        # 미완료 라인만 필터
        pending = []
        for line in lines:
            wav_path = line.strip().split("|")[0]
            npy_path = f"{wav_path}.npy"
            if not os.path.exists(npy_path):
                pending.append(line)

    if not pending:
        console.print("  [success]스타일 벡터 이미 완료[/success]")
        return

    console.print(f"  스타일 벡터 생성: {len(pending)}개 (전체 {len(lines)}개 중)")

    # pyannote 모델 지연 로드
    with console.status("[bold]스타일 모델 (pyannote) 로딩 중..."):
        from style_gen import process_line

    ok_lines: list[str] = []
    nan_lines: list[str] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), console=console,
    ) as progress:
        task = progress.add_task("Style", total=len(pending))
        for line in pending:
            result_line, error = process_line(line)
            if error is None:
                ok_lines.append(result_line)
            else:
                nan_lines.append(result_line)
            progress.advance(task)

    # NaN 파일 제거
    if nan_lines:
        nan_set = set(nan_lines)
        console.print(f"  [warning]NaN {len(nan_lines)}개 파일 제거[/warning]")
        for list_path in [Path(hps.data.training_files), Path(hps.data.validation_files)]:
            original = _read_lines(list_path)
            with open(list_path, "w", encoding="utf-8") as f:
                f.writelines(l for l in original if l not in nan_set)


def run_default_style(project_dir: Path, data_dir: Path):
    """Step 4: 평균 스타일 벡터 계산."""
    activate_dataset(project_dir)
    from default_style import save_styles_by_dirs
    from config import get_config

    cfg = get_config()
    # raw/ 디렉토리 (리샘플된 오디오) 또는 audio/ 디렉토리
    wav_dir = data_dir / "raw"
    if not wav_dir.exists():
        wav_dir = project_dir / "audio"

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    save_styles_by_dirs(
        wav_dir=wav_dir,
        output_dir=cfg.out_dir,
        config_path=cfg.preprocess_text_config.config_path,
        config_output_path=cfg.out_dir / "config.json",
    )


def run_all_preprocessing(project_dir: Path, data_dir: Path | None = None, force: bool = False):
    """전체 전처리 실행. 완료된 단계는 스킵."""
    from cli.training.dataset import _is_text_preprocessed, _count_feature_files, _get_model_name

    data_dir = data_dir or project_dir
    train_list = data_dir / "train.list"
    val_list = data_dir / "val.list"

    # train.list/val.list가 비어있으면 전처리 불가
    train_lines = _read_lines(train_list) if train_list.exists() else []
    val_lines = _read_lines(val_list) if val_list.exists() else []
    if not train_lines and not val_lines:
        console.print(
            "\n[error]train.list / val.list가 비어있습니다.[/error]\n"
            "  먼저 데이터 전처리 도구에서 train/val 분할을 실행해 주세요."
        )
        return

    # Step 1: 텍스트 전처리
    console.print("\n[step][1/4] 텍스트 전처리 (G2P)[/step]")
    if not force and _is_text_preprocessed(train_list):
        console.print("  [success]이미 완료[/success]")
    else:
        run_preprocess_text(project_dir, data_dir)
        console.print("  [success]완료[/success]")

    # Step 2: BERT 임베딩
    console.print("\n[step][2/4] BERT 임베딩 생성[/step]")
    run_bert_gen(project_dir, data_dir, force=force)

    # Step 3: 스타일 벡터
    console.print("\n[step][3/4] 스타일 벡터 생성[/step]")
    run_style_gen(project_dir, data_dir, force=force)

    # Step 4: 기본 스타일
    console.print("\n[step][4/4] 기본 스타일 벡터[/step]")
    model_name = _get_model_name(data_dir / "config.json")
    exports_dir = project_dir / "exports" / model_name if model_name else None
    if not force and exports_dir and (exports_dir / "style_vectors.npy").exists():
        console.print("  [success]이미 완료[/success]")
    else:
        run_default_style(project_dir, data_dir)
        console.print("  [success]완료[/success]")

    console.print("\n[success]전처리 완료![/success]")
