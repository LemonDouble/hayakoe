import json
from pathlib import Path
from typing import Union

import numpy as np

from hayakoe.constants import DEFAULT_STYLE
from hayakoe.logging import logger


def save_neutral_vector(
    wav_dir: Union[Path, str],
    output_dir: Union[Path, str],
    config_path: Union[Path, str],
    config_output_path: Union[Path, str],
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = np.load(file)
        embs.append(np.expand_dims(xvec, axis=0))

    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_dir / "style_vectors.npy", only_mean)
    logger.info(f"평균 스타일 벡터를 {output_dir}에 저장했습니다")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"스타일 설정을 {config_output_path}에 저장했습니다")


def save_styles_by_dirs(
    wav_dir: Union[Path, str],
    output_dir: Union[Path, str],
    config_path: Union[Path, str],
    config_output_path: Union[Path, str],
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_output_path = Path(config_output_path)

    subdirs = [d for d in wav_dir.iterdir() if d.is_dir()]
    subdirs.sort()
    if len(subdirs) in (0, 1):
        logger.info(
            f"스타일 벡터 생성을 위해 최소 2개의 하위 디렉터리가 필요하지만 {len(subdirs)}개만 발견했습니다."
        )
        logger.info("대신 중립(neutral) 스타일 벡터만 생성합니다.")
        save_neutral_vector(wav_dir, output_dir, config_path, config_output_path)
        return

    # Neutral용으로 전체 평균을 먼저 계산
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = np.load(file)
        embs.append(np.expand_dims(xvec, axis=0))
    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    style_vectors = [mean]

    names = [DEFAULT_STYLE]
    for style_dir in subdirs:
        npy_files = list(style_dir.rglob("*.npy"))
        if not npy_files:
            continue
        embs = []
        for file in npy_files:
            xvec = np.load(file)
            embs.append(np.expand_dims(xvec, axis=0))

        x = np.concatenate(embs, axis=0)  # (N, 256)
        mean = np.mean(x, axis=0)  # (256,)
        style_vectors.append(mean)
        names.append(style_dir.name)

    # (num_styles, 256) 형태로 스택
    style_vectors_npy = np.stack(style_vectors, axis=0)
    np.save(output_dir / "style_vectors.npy", style_vectors_npy)
    logger.info(f"스타일 벡터를 {output_dir / 'style_vectors.npy'}에 저장했습니다")

    # style2id 설정을 JSON으로 저장
    style2id = {name: i for i, name in enumerate(names)}
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(names)
    json_dict["data"]["style2id"] = style2id
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"스타일 설정을 {config_output_path}에 저장했습니다")
