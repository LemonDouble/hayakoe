"""
HayaKoe 모델의 하이퍼파라미터를 나타내는 Pydantic 모델.
기본값은 configs/config_jp_extra.json 내 정의와 대체로 동일하며,
로드한 config.json에 존재하지 않는 키가 있을 경우의 페일세이프로 적용된다.
"""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


class HyperParametersTrain(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 42
    epochs: int = 1000
    learning_rate: float = 0.0001
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    batch_size: int = 2
    bf16_run: bool = False
    fp16_run: bool = False
    lr_decay: float = 0.99996
    segment_size: int = 16384
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    c_commit: int = 100
    skip_optimizer: bool = False
    freeze_ZH_bert: bool = False
    freeze_JP_bert: bool = False
    freeze_EN_bert: bool = False
    freeze_emo: bool = False
    freeze_style: bool = False
    freeze_decoder: bool = False


class HyperParametersData(BaseModel):
    # use_jp_extra 필드가 존재하지 않는 구 모델과의 호환성을 위해 False를 기본값으로 설정
    use_jp_extra: bool = False
    training_files: str = "Data/Dummy/train.list"
    validation_files: str = "Data/Dummy/val.list"
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    add_blank: bool = True
    n_speakers: int = 1
    cleaned_text: bool = True
    spk2id: dict[str, int] = {
        "Dummy": 0,
    }
    num_styles: int = 1
    style2id: dict[str, int] = {
        "Neutral": 0,
    }


class HyperParametersModelSLM(BaseModel):
    model: str = "./slm/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


class HyperParametersModel(BaseModel):
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_mel_posterior_encoder: bool = False
    use_duration_discriminator: bool = False
    use_wavlm_discriminator: bool = True
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    ]
    upsample_rates: list[int] = [8, 8, 2, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = [16, 16, 8, 2, 2]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 512
    slm: HyperParametersModelSLM = HyperParametersModelSLM()


class HyperParameters(BaseModel):
    model_name: str = "Dummy"
    version: str = "2.0-JP-Extra"
    train: HyperParametersTrain = HyperParametersTrain()
    data: HyperParametersData = HyperParametersData()
    model: HyperParametersModel = HyperParametersModel()

    # 아래는 학습 시에만 동적으로 설정되는 파라미터 (일반적으로 config.json에는 존재하지 않음)
    model_dir: Optional[str] = None
    speedup: bool = False
    repo_id: Optional[str] = None

    # model_ 접두사를 Pydantic의 보호 대상에서 제외
    model_config = ConfigDict(protected_namespaces=())

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> "HyperParameters":
        """
        주어진 JSON 파일에서 하이퍼파라미터를 로드한다.

        Args:
            json_path (Union[str, Path]): JSON 파일의 경로

        Returns:
            HyperParameters: 하이퍼파라미터
        """

        with open(json_path, encoding="utf-8") as f:
            return HyperParameters.model_validate_json(f.read())
