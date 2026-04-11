"""BERT 피처 추출: 각 발화에 대해 BERT 임베딩 생성 후 캐시."""

import argparse
import os

import torch
from tqdm import tqdm

from config import get_config
from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models import commons
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.nlp import cleaned_text_to_sequence, extract_bert_feature
from hayakoe.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()


def process_line(line: str, add_blank: bool, device: str):
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    # 이미 생성된 파일은 스킵 (멱등성)
    if os.path.exists(bert_path):
        return

    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]

    phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages[language_str])

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert = extract_bert_feature(text, word2ph, Languages(language_str), device)
    assert bert.shape[-1] == len(phone)
    torch.save(bert, bert_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=config.bert_gen_config.config_path)
    args, _ = parser.parse_known_args()

    hps = HyperParameters.load_from_json(args.config)
    device = config.bert_gen_config.device

    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    for line in tqdm(lines, file=SAFE_STDOUT, dynamic_ncols=True):
        process_line(line, hps.data.add_blank, device)

    logger.info(f"BERT 피처 생성 완료! 총 {len(lines)}개 파일.")
