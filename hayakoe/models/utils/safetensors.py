from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hayakoe.logging import logger


def load_safetensors(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    for_infer: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.nn.Module, Optional[int]]:
    """
    지정된 경로에서 safetensors 모델을 로드하고, 모델과 이터레이션을 반환한다.

    Args:
        checkpoint_path (Union[str, Path]): 모델의 체크포인트 파일 경로
        model (torch.nn.Module): 로드 대상 모델
        for_infer (bool): 추론용으로 로드할지 여부 플래그

    Returns:
        tuple[torch.nn.Module, Optional[int]]: 로드된 모델과 이터레이션 횟수 (존재하는 경우)
    """

    tensors: dict[str, Any] = {}
    iteration: Optional[int] = None
    with safe_open(str(checkpoint_path), framework="pt", device=device) as f:  # type: ignore
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
            tensors[key] = f.get_tensor(key)
    if hasattr(model, "module"):
        result = model.module.load_state_dict(tensors, strict=False)
    else:
        result = model.load_state_dict(tensors, strict=False)
    for key in result.missing_keys:
        if key.startswith("enc_q") and for_infer:
            continue
        logger.warning(f"Missing key: {key}")
    for key in result.unexpected_keys:
        if key == "iteration":
            continue
        logger.warning(f"Unexpected key: {key}")
    if iteration is None:
        logger.info(f"Loaded '{checkpoint_path}'")
    else:
        logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, iteration


def save_safetensors(
    model: torch.nn.Module,
    iteration: int,
    checkpoint_path: Union[str, Path],
    is_half: bool = False,
    for_infer: bool = False,
) -> None:
    """
    모델을 safetensors 형식으로 저장한다.

    Args:
        model (torch.nn.Module): 저장할 모델
        iteration (int): 이터레이션 횟수
        checkpoint_path (Union[str, Path]): 저장 경로
        is_half (bool): 모델을 반정밀도로 저장할지 여부 플래그
        for_infer (bool): 추론용으로 저장할지 여부 플래그
    """

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    keys = []
    for k in state_dict:
        if "enc_q" in k and for_infer:
            continue
        keys.append(k)

    new_dict = (
        {k: state_dict[k].half() for k in keys}
        if is_half
        else {k: state_dict[k] for k in keys}
    )
    new_dict["iteration"] = torch.LongTensor([iteration])
    logger.info(f"Saved safetensors to {checkpoint_path}")

    save_file(new_dict, checkpoint_path)
