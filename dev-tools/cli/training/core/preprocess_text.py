"""텍스트 전처리: G2P 변환 + train/val 분할."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import Optional

from tqdm import tqdm

from config import get_config
from hayakoe.logging import logger
from hayakoe.nlp import clean_text
from hayakoe.utils.stdout_wrapper import SAFE_STDOUT


preprocess_text_config = get_config().preprocess_text_config


def process_line(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    use_jp_extra: bool,
    yomi_error: str,
) -> str:
    splitted_line = line.strip().split("|")
    if len(splitted_line) != 4:
        raise ValueError(f"Invalid line format: {line.strip()}")
    utt, spk, language, text = splitted_line
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        language=language,  # type: ignore
        use_jp_extra=use_jp_extra,
        raise_yomi_error=(yomi_error != "use"),
    )
    if correct_path:
        utt = str(transcription_path.parent / "audio" / utt)

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt, spk, language, norm_text,
        " ".join(phones),
        " ".join(str(i) for i in tones),
        " ".join(str(i) for i in word2ph),
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Optional[Path],
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_lang: int,
    max_val_total: int,
    use_jp_extra: bool,
    yomi_error: str,
    correct_path: bool,
):
    assert yomi_error in ["raise", "skip", "use"]
    if not cleaned_path:
        cleaned_path = transcription_path.with_name(transcription_path.name + ".cleaned")

    error_log_path = transcription_path.parent / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    total_lines = sum(1 for _ in transcription_path.open("r", encoding="utf-8"))

    # G2P 변환
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(trans_file, file=SAFE_STDOUT, total=total_lines, dynamic_ncols=True):
            try:
                out_file.write(process_line(line, transcription_path, correct_path, use_jp_extra, yomi_error))
            except Exception as e:
                logger.error(f"Error at line: {line.strip()}\n{e}")
                with error_log_path.open("a", encoding="utf-8") as err_f:
                    err_f.write(f"{line.strip()}\n{e}\n\n")
                error_count += 1

    # 화자 ID 매핑 + 오디오 검증
    spk_utt_map: dict[str, list[str]] = defaultdict(list)
    spk_id_map: dict[str, int] = {}
    current_sid = 0

    with cleaned_path.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk = line.strip().split("|")[:2]
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                continue
            spk_utt_map[spk].append(line)
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1

    # Train/Val 분할
    train_list: list[str] = []
    val_list: list[str] = []
    for spk, utts in spk_utt_map.items():
        if val_per_lang == 0 or len(utts) <= val_per_lang:
            train_list.extend(utts)
            continue
        val_indices = set(sample(range(len(utts)), val_per_lang))
        for i, utt in enumerate(utts):
            (val_list if i in val_indices else train_list).append(utt)

    if len(val_list) > max_val_total:
        train_list.extend(val_list[max_val_total:])
        val_list = val_list[:max_val_total]

    train_path.write_text("".join(train_list), encoding="utf-8")
    val_path.write_text("".join(val_list), encoding="utf-8")

    # Config 업데이트
    json_config = json.loads(config_path.read_text(encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    config_path.write_text(json.dumps(json_config, indent=2, ensure_ascii=False), encoding="utf-8")

    if error_count > 0 and yomi_error != "skip":
        raise Exception(f"{error_count}개 라인에서 에러 발생. {error_log_path} 확인.")
    elif error_count > 0:
        logger.warning(f"{error_count}개 라인 스킵됨. {error_log_path} 확인.")

    logger.info(f"텍스트 전처리 완료! train: {len(train_list)}, val: {len(val_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcription-path", default=preprocess_text_config.transcription_path)
    parser.add_argument("--cleaned-path", default=preprocess_text_config.cleaned_path)
    parser.add_argument("--train-path", default=preprocess_text_config.train_path)
    parser.add_argument("--val-path", default=preprocess_text_config.val_path)
    parser.add_argument("--config-path", default=preprocess_text_config.config_path)
    parser.add_argument("--val-per-lang", default=preprocess_text_config.val_per_lang)
    parser.add_argument("--max-val-total", default=preprocess_text_config.max_val_total)
    parser.add_argument("--use_jp_extra", action="store_true")
    parser.add_argument("--yomi_error", default="raise")
    parser.add_argument("--correct_path", action="store_true")
    args = parser.parse_args()

    preprocess(
        transcription_path=Path(args.transcription_path),
        cleaned_path=Path(args.cleaned_path) if args.cleaned_path else None,
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        config_path=Path(args.config_path),
        val_per_lang=int(args.val_per_lang),
        max_val_total=int(args.max_val_total),
        use_jp_extra=args.use_jp_extra,
        yomi_error=args.yomi_error,
        correct_path=args.correct_path,
    )
