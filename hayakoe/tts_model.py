from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from hayakoe.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.voice import adjust_voice


if TYPE_CHECKING:
    from hayakoe.models.models import SynthesizerTrn
    from hayakoe.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class NullModelParam(BaseModel):
    """
    Null 모델의 파라미터를 나타내는 Pydantic 모델.
    각 파라미터는 0.0에서 1.0 범위로 지정한다.
    """

    name: str  # 모델명
    path: Path  # 모델 파일 경로
    weight: float = Field(ge=0.0, le=1.0)  # 음색 가중치
    pitch: float = Field(ge=0.0, le=1.0)  # 음높이 가중치
    style: float = Field(ge=0.0, le=1.0)  # 말투 가중치
    tempo: float = Field(ge=0.0, le=1.0)  # 템포 가중치


class TTSModel:
    """
    HayaKoe 음성 합성 모델을 조작하는 클래스.
    모델/하이퍼파라미터/스타일 벡터의 경로와 디바이스를 지정하여 초기화하고, model.infer() 메서드를 호출하면 음성 합성을 수행한다.
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Union[Path, HyperParameters],
        style_vec_path: Union[Path, NDArray[Any]],
        device: str = "cpu",
        onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    ) -> None:  # fmt: skip
        """
        HayaKoe 음성 합성 모델을 초기화한다.
        이 시점에서는 모델이 로드되지 않은 상태이다 (명시적으로 로드하려면 model.load()를 호출한다).

        Args:
            model_path (Path): 모델 (.safetensors / .onnx) 경로
            config_path (Union[Path, HyperParameters]): 하이퍼파라미터 (config.json) 경로 (직접 HyperParameters를 지정할 수도 있음)
            style_vec_path (Union[Path, NDArray[Any]]): 스타일 벡터 (style_vectors.npy) 경로 (직접 NDArray를 지정할 수도 있음)
            device (str): PyTorch 추론 시 음성 합성에 사용할 디바이스 (cpu, cuda, mps 등)
            onnx_providers (list[str]): ONNX 추론에 사용할 ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider 등)
        """

        self.model_path: Path = model_path
        self.device: str = device
        self.onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = onnx_providers  # fmt: skip

        # ONNX 형식의 모델인지 여부
        if self.model_path.suffix == ".onnx":
            self.is_onnx_model = True
        else:
            self.is_onnx_model = False

        # 하이퍼파라미터의 Pydantic 모델이 직접 지정된 경우
        if isinstance(config_path, HyperParameters):
            self.config_path: Path = Path("")  # 호환성을 위해 빈 Path 설정
            self.hyper_parameters: HyperParameters = config_path
        # 하이퍼파라미터 경로가 지정된 경우
        else:
            self.config_path: Path = config_path
            self.hyper_parameters: HyperParameters = HyperParameters.load_from_json(
                self.config_path
            )

        # 스타일 벡터의 NDArray가 직접 지정된 경우
        if isinstance(style_vec_path, np.ndarray):
            self.style_vec_path: Path = Path("")  # 호환성을 위해 빈 Path 설정
            self.style_vectors: NDArray[Any] = style_vec_path
        # 스타일 벡터 경로가 지정된 경우
        else:
            self.style_vec_path: Path = style_vec_path
            self.style_vectors: NDArray[Any] = np.load(self.style_vec_path)

        self.spk2id: dict[str, int] = self.hyper_parameters.data.spk2id
        self.id2spk: dict[int, str] = {v: k for k, v in self.spk2id.items()}

        num_styles: int = self.hyper_parameters.data.num_styles
        if hasattr(self.hyper_parameters.data, "style2id"):
            self.style2id: dict[str, int] = self.hyper_parameters.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        if self.style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"The number of styles ({num_styles}) does not match the number of style vectors ({self.style_vectors.shape[0]})"
            )
        self.style_vector_inference: Optional[Any] = None

        # net_g / null_model_params는 PyTorch 추론 시에만 지연 초기화됨
        self.net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None
        self.null_model_params: Optional[dict[int, NullModelParam]] = None

        # onnx_session은 ONNX 추론 시에만 지연 초기화됨
        self.onnx_session: Optional[onnxruntime.InferenceSession] = None

    def load(self) -> None:
        """
        음성 합성 모델을 디바이스에 로드한다.
        """

        start_time = time.time()

        # PyTorch 추론 시
        if not self.is_onnx_model:
            from hayakoe.models.infer import get_net_g

            # PyTorch 모델 로드
            self.net_g = get_net_g(
                model_path=str(self.model_path),
                version=self.hyper_parameters.version,
                device=self.device,
                hps=self.hyper_parameters,
            )
            logger.info(
                f'Model loaded successfully from {self.model_path} to "{self.device}" device ({time.time() - start_time:.2f}s)'
            )

            # 이하는 Null 모델 로드용 파라미터가 지정된 경우에만 실행
            if self.null_model_params is None:
                return

            # 추론 대상 모델의 가중치와 Null 모델의 가중치를 병합
            for null_model_info in self.null_model_params.values():
                logger.info(f"Adding null model: {null_model_info.path}...")
                null_model_add = get_net_g(
                    model_path=str(null_model_info.path),
                    version=self.hyper_parameters.version,
                    device=self.device,
                    hps=self.hyper_parameters,
                )
                # 단순한 방식. 더 나은 방법이 있을 수 있음
                params = zip(
                    self.net_g.dec.parameters(), null_model_add.dec.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.weight))
                params = zip(
                    self.net_g.flow.parameters(), null_model_add.flow.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.pitch))

                params = zip(
                    self.net_g.enc_p.parameters(), null_model_add.enc_p.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.style))
                # 템포는 sdp와 dp 두 개가 있으므로 일단 둘 다 더함
                params = zip(
                    self.net_g.sdp.parameters(), null_model_add.sdp.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.tempo))
                params = zip(self.net_g.dp.parameters(), null_model_add.dp.parameters())
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.tempo))

            logger.info(
                f"Null models merged successfully ({time.time() - start_time:.2f}s)"
            )

        # ONNX 추론 시
        else:
            # 추론 시 가장 우선되는 ExecutionProvider 이름 취득
            assert len(self.onnx_providers) > 0
            first_provider_name = (
                self.onnx_providers[0]
                if type(self.onnx_providers[0]) is str
                else self.onnx_providers[0][0]
            )

            # 추론 세션 설정
            sess_options = onnxruntime.SessionOptions()
            ## ONNX 모델 생성 시 이미 onnxsim으로 최적화되어 있으므로, 로드 고속화를 위해 최적화를 비활성화함
            ## DmlExecutionProvider가 선두에 지정된 경우에만, DirectML 추론 고속화를 위해 모든 최적화를 활성화함
            if first_provider_name == "DmlExecutionProvider":
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # fmt: skip
            else:
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # fmt: skip
            ## 에러 이외의 로그를 출력하지 않음
            ## 원래는 log_severity_level = 3만으로 충분하지만, CUDA 관련 로그가 억제되지 않아 set_default_logger_severity()도 호출함
            sess_options.log_severity_level = 3
            onnxruntime.set_default_logger_severity(3)

            # ONNX 모델을 로드하고 추론 세션을 초기화
            self.onnx_session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.onnx_providers,
            )
            logger.info(
                f"Model loaded successfully from {self.model_path} to {self.onnx_session.get_providers()[0]} ({time.time() - start_time:.2f}s)"
            )

    def unload(self) -> None:
        """
        음성 합성 모델을 디바이스에서 언로드한다.
        PyTorch 모델의 경우 CUDA 메모리도 해제된다.
        """

        start_time = time.time()

        # PyTorch 추론 시
        if self.net_g is not None:
            import torch

            del self.net_g
            self.net_g = None

            # CUDA 캐시 클리어
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ONNX 추론 시
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None

        gc.collect()
        logger.info(f"Model unloaded successfully ({time.time() - start_time:.2f}s)")

    def get_style_vector(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        """
        스타일 벡터를 취득한다.

        Args:
            style_id (int): 스타일 ID (0부터 시작하는 인덱스)
            weight (float, optional): 스타일 벡터의 가중치. Defaults to 1.0.

        Returns:
            NDArray[Any]: 스타일 벡터
        """
        mean = self.style_vectors[0]
        style_vec = self.style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> NDArray[Any]:
        """
        음성으로부터 스타일 벡터를 추론한다.

        Args:
            audio_path (str): 음성 파일 경로
            weight (float, optional): 스타일 벡터의 가중치. Defaults to 1.0.
        Returns:
            NDArray[Any]: 스타일 벡터
        """

        if self.style_vector_inference is None:

            # pyannote.audio는 scikit-learn 등 대량의 무거운 라이브러리에 의존하므로,
            # TTSModel.infer()에 reference_audio_path를 지정하여 음성에서 스타일 벡터를 추론하는 경우에만 지연 import 함
            try:
                import pyannote.audio
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required to infer style vector from audio"
                )

            # 스타일 벡터를 취득하기 위한 추론 모델 초기화
            import torch

            self.style_vector_inference = pyannote.audio.Inference(
                model=pyannote.audio.Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                ),
                window="whole",
            )
            self.style_vector_inference.to(torch.device(self.device))

        # 음성으로부터 스타일 벡터를 추론
        xvec = self.style_vector_inference(audio_path)
        mean = self.style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    @staticmethod
    def convert_to_16_bit_wav(data: NDArray[Any]) -> NDArray[Any]:
        """
        음성 데이터를 16-bit int 형식으로 변환한다.
        gradio.processing_utils.convert_to_16_bit_wav()를 이식한 것.

        Args:
            data (NDArray[Any]): 음성 데이터

        Returns:
            NDArray[Any]: 16-bit int 형식의 음성 데이터
        """

        # 참고: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
        if data.dtype in [np.float64, np.float32, np.float16]:  # type: ignore
            data = data / np.abs(data).max()
            data = data * 32767
            data = data.astype(np.int16)
        elif data.dtype == np.int32:
            data = data / 65536
            data = data.astype(np.int16)
        elif data.dtype == np.int16:
            pass
        elif data.dtype == np.uint16:
            data = data - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.uint8:
            data = data * 257 - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.int8:
            data = data * 256
            data = data.astype(np.int16)
        else:
            raise ValueError(
                "Audio data cannot be converted automatically from "
                f"{data.dtype} to 16-bit int format."
            )

        return data

    def infer(
        self,
        text: str,
        language: Languages = Languages.JP,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
        null_model_params: Optional[dict[int, NullModelParam]] = None,
        force_reload_model: bool = False,
    ) -> tuple[int, NDArray[Any]]:
        """
        텍스트에서 음성을 합성한다.

        Args:
            text (str): 읽어줄 텍스트
            language (Languages, optional): 언어. Defaults to Languages.JP.
            speaker_id (int, optional): 화자 ID. Defaults to 0.
            reference_audio_path (Optional[str], optional): 음성 스타일 참조용 음성 파일 경로. Defaults to None.
            sdp_ratio (float, optional): DP와 SDP의 혼합비. 0이면 DP만, 1이면 SDP만 사용 (값이 클수록 템포에 완급이 생김). Defaults to DEFAULT_SDP_RATIO.
            noise (float, optional): DP에 부여되는 노이즈. Defaults to DEFAULT_NOISE.
            noise_w (float, optional): SDP에 부여되는 노이즈. Defaults to DEFAULT_NOISEW.
            length (float, optional): 생성 음성의 길이(말속도) 파라미터. 클수록 생성 음성이 길고 느리며, 작을수록 짧고 빨라짐. Defaults to DEFAULT_LENGTH.
            line_split (bool, optional): 텍스트를 줄바꿈 단위로 분할하여 생성할지 여부 (True일 경우 given_phone/given_tone은 무시됨). Defaults to DEFAULT_LINE_SPLIT.
            split_interval (float, optional): 줄바꿈 단위로 분할할 경우의 무음 (초). Defaults to DEFAULT_SPLIT_INTERVAL.
            assist_text (Optional[str], optional): 감정 표현 참조용 보조 텍스트. Defaults to None.
            assist_text_weight (float, optional): 감정 표현 보조 텍스트를 적용하는 강도. Defaults to DEFAULT_ASSIST_TEXT_WEIGHT.
            use_assist_text (bool, optional): 음성 합성 시 감정 표현 보조 텍스트를 사용할지 여부. Defaults to False.
            style (str, optional): 음성 스타일 (Neutral, Happy 등). Defaults to DEFAULT_STYLE.
            style_weight (float, optional): 음성 스타일을 적용하는 강도. Defaults to DEFAULT_STYLE_WEIGHT.
            given_phone (Optional[list[int]], optional): 읽기 텍스트의 발음을 나타내는 음소 시퀀스. 지정 시 given_tone도 별도로 지정 필요. Defaults to None.
            given_tone (Optional[list[int]], optional): 악센트 톤 리스트. Defaults to None.
            pitch_scale (float, optional): 피치 높이 (1.0에서 변경하면 약간 음질이 저하됨). Defaults to 1.0.
            intonation_scale (float, optional): 억양의 평균으로부터의 변화 폭 (1.0에서 변경하면 약간 음질이 저하됨). Defaults to 1.0.
            null_model_params (Optional[dict[int, NullModelParam]], optional): 추론 시 사용할 Null 모델 정보. ONNX 추론에서는 무시됨.
            force_reload_model (bool, optional): 모델을 강제로 재로드할지 여부. Defaults to False.
        Returns:
            tuple[int, NDArray[Any]]: 샘플링 레이트와 음성 데이터 (16bit PCM)
        """

        logger.info(f"Start generating audio data from text:\n{text}")
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        # 스타일 벡터 취득
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.get_style_vector(style_id, style_weight)
        else:
            style_vector = self.get_style_vector_from_audio(
                reference_audio_path, style_weight
            )

        # PyTorch 추론 시
        start_time = time.time()
        if not self.is_onnx_model:
            import torch

            from hayakoe.models.infer import infer

            if null_model_params is not None:
                self.null_model_params = null_model_params
            else:
                self.null_model_params = None

            # force_reload_model이 True일 때, 메모리에 보유 중인 모델을 파기함
            if force_reload_model is True:
                self.net_g = None

            # 모델이 로드되지 않은 경우 로드함
            if self.net_g is None:
                self.load()
            assert self.net_g is not None

            # 일반 텍스트에서 음성 생성
            if not line_split:
                with torch.no_grad():
                    audio = infer(
                        text=text,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise,
                        noise_scale_w=noise_w,
                        length_scale=length,
                        sid=speaker_id,
                        language=language,
                        hps=self.hyper_parameters,
                        net_g=self.net_g,
                        device=self.device,
                        assist_text=assist_text,
                        assist_text_weight=assist_text_weight,
                        style_vec=style_vector,
                        given_phone=given_phone,
                        given_tone=given_tone,
                    )

            # 줄바꿈 단위로 분할하여 음성 생성
            else:
                texts = [t for t in text.split("\n") if t != ""]
                audios = []
                with torch.no_grad():
                    for i, t in enumerate(texts):
                        audios.append(
                            infer(
                                text=t,
                                sdp_ratio=sdp_ratio,
                                noise_scale=noise,
                                noise_scale_w=noise_w,
                                length_scale=length,
                                sid=speaker_id,
                                language=language,
                                hps=self.hyper_parameters,
                                net_g=self.net_g,
                                device=self.device,
                                assist_text=assist_text,
                                assist_text_weight=assist_text_weight,
                                style_vec=style_vector,
                            )
                        )
                        if i != len(texts) - 1:
                            audios.append(np.zeros(int(44100 * split_interval)))
                    audio = np.concatenate(audios)

        # ONNX 추론 시
        else:
            from hayakoe.models.infer_onnx import infer_onnx

            # force_reload_model이 True일 때, 메모리에 보유 중인 모델을 파기함
            if force_reload_model is True:
                self.onnx_session = None

            # 모델이 로드되지 않은 경우 로드함
            if self.onnx_session is None:
                self.load()
            assert self.onnx_session is not None

            # 일반 텍스트에서 음성 생성
            if not line_split:
                audio = infer_onnx(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noise_w,
                    length_scale=length,
                    sid=speaker_id,
                    language=language,
                    hps=self.hyper_parameters,
                    onnx_session=self.onnx_session,
                    onnx_providers=self.onnx_providers,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_phone=given_phone,
                    given_tone=given_tone,
                )

            # 줄바꿈 단위로 분할하여 음성 생성
            else:
                texts = [t for t in text.split("\n") if t != ""]
                audios = []
                for i, t in enumerate(texts):
                    audios.append(
                        infer_onnx(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noise_w,
                            length_scale=length,
                            sid=speaker_id,
                            language=language,
                            hps=self.hyper_parameters,
                            onnx_session=self.onnx_session,
                            onnx_providers=self.onnx_providers,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)

        logger.info(
            f"Audio data generated successfully ({time.time() - start_time:.2f}s)"
        )

        if not (pitch_scale == 1.0 and intonation_scale == 1.0):
            _, audio = adjust_voice(
                fs=self.hyper_parameters.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        audio = self.convert_to_16_bit_wav(audio)
        return (self.hyper_parameters.data.sampling_rate, audio)


class TTSModelInfo(BaseModel):
    name: str
    files: list[str]
    styles: list[str]
    speakers: list[str]


class TTSModelHolder:
    """
    HayaKoe 음성 합성 모델을 관리하는 클래스.
    model_holder.models_info에서 지정된 디렉토리 내의 음성 합성 모델 목록을 취득할 수 있다.
    """

    def __init__(
        self,
        model_root_dir: Path,
        device: str,
        onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
        ignore_onnx: bool = False,
    ) -> None:
        """
        HayaKoe 음성 합성 모델을 관리하는 클래스를 초기화한다.
        음성 합성 모델은 아래와 같이 배치되어 있어야 한다 (.safetensors / .onnx 파일명은 자유).
        ```
        model_root_dir
        ├── model-name-1
        │   ├── config.json
        │   ├── model-name-1_e160_s14000.safetensors
        │   └── style_vectors.npy
        ├── model-name-2
        │   ├── config.json
        │   ├── model-name-2_e160_s14000.safetensors
        │   └── style_vectors.npy
        └── ...
        ```

        Args:
            model_root_dir (Path): 음성 합성 모델이 배치된 디렉토리 경로
            device (str): PyTorch 추론 시 음성 합성에 사용할 디바이스 (cpu, cuda, mps 등)
            onnx_providers (list[str]): ONNX 추론에 사용할 ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider 등)
            ignore_onnx (bool, optional): ONNX 모델을 제외할지 여부. Defaults to False.
        """

        self.root_dir: Path = model_root_dir
        self.device: str = device
        self.onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = onnx_providers  # fmt: skip
        self.ignore_onnx: bool = ignore_onnx
        self.model_files_dict: dict[str, list[Path]] = {}
        self.current_model: Optional[TTSModel] = None
        self.model_names: list[str] = []
        self.models_info: list[TTSModelInfo] = []
        self.refresh()

    def refresh(self) -> None:
        """
        음성 합성 모델 목록을 갱신한다.
        """

        self.model_files_dict = {}
        self.model_names = []
        self.current_model = None
        self.models_info = []

        model_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for model_dir in model_dirs:
            if model_dir.name.startswith("."):
                continue
            suffixes = [".pth", ".pt", ".safetensors"]
            if self.ignore_onnx is False:
                suffixes.append(".onnx")
            model_files = sorted(
                [
                    f
                    for f in model_dir.iterdir()
                    # 위 suffixes에 매칭되는 파일만 취득하고, .으로 시작하는 파일은 제외
                    if f.suffix in suffixes and not f.name.startswith(".")
                ],
                # 수정일시 기준 최신순 정렬
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if len(model_files) == 0:
                logger.warning(f"No model files found in {model_dir}, so skip it")
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(
                    f"Config file {config_path} not found, so skip {model_dir}"
                )
                continue
            self.model_files_dict[model_dir.name] = model_files
            self.model_names.append(model_dir.name)
            hyper_parameters = HyperParameters.load_from_json(config_path)
            style2id: dict[str, int] = hyper_parameters.data.style2id
            styles = list(style2id.keys())
            spk2id: dict[str, int] = hyper_parameters.data.spk2id
            speakers = list(spk2id.keys())
            self.models_info.append(
                TTSModelInfo(
                    name=model_dir.name,
                    files=[str(f) for f in model_files],
                    styles=styles,
                    speakers=speakers,
                )
            )

    def get_model(self, model_name: str, model_path_str: str) -> TTSModel:
        """
        지정된 음성 합성 모델의 인스턴스를 취득한다.
        이 시점에서는 모델이 로드되지 않은 상태이다 (명시적으로 로드하려면 model.load()를 호출한다).

        Args:
            model_name (str): 음성 합성 모델 이름
            model_path_str (str): 음성 합성 모델 파일 경로 (.safetensors)

        Returns:
            TTSModel: 음성 합성 모델 인스턴스
        """

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=self.root_dir / model_name / "config.json",
                style_vec_path=self.root_dir / model_name / "style_vectors.npy",
                device=self.device,
                onnx_providers=self.onnx_providers,
            )

        return self.current_model

    def get_model_for_gradio(self, model_name: str, model_path_str: str):
        import gradio as gr

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if (
            self.current_model is not None
            and self.current_model.model_path == model_path
        ):
            # 이미 로드됨
            speakers = list(self.current_model.spk2id.keys())
            styles = list(self.current_model.style2id.keys())
            return (
                gr.Dropdown(choices=styles, value=styles[0]),
                gr.Button(interactive=True, value="音声合成"),
                gr.Dropdown(choices=speakers, value=speakers[0]),
            )
        self.current_model = TTSModel(
            model_path=model_path,
            config_path=self.root_dir / model_name / "config.json",
            style_vec_path=self.root_dir / model_name / "style_vectors.npy",
            device=self.device,
            onnx_providers=self.onnx_providers,
        )
        speakers = list(self.current_model.spk2id.keys())
        styles = list(self.current_model.style2id.keys())
        return (
            gr.Dropdown(choices=styles, value=styles[0]),
            gr.Button(interactive=True, value="音声合成"),
            gr.Dropdown(choices=speakers, value=speakers[0]),
        )

    def update_model_files_for_gradio(self, model_name: str):
        import gradio as gr

        model_files = [str(f) for f in self.model_files_dict[model_name]]
        return gr.Dropdown(choices=model_files, value=model_files[0])

    def update_model_names_for_gradio(
        self,
    ):
        import gradio as gr

        self.refresh()
        initial_model_name = self.model_names[0]
        initial_model_files = [
            str(f) for f in self.model_files_dict[initial_model_name]
        ]
        return (
            gr.Dropdown(choices=self.model_names, value=initial_model_name),
            gr.Dropdown(choices=initial_model_files, value=initial_model_files[0]),
            gr.Button(interactive=False),  # tts_button용
        )
